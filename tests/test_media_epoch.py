"""Tests for the declarable media plane resolution epochs (HTTP plane).

These tests cover:

* :class:`falcon.media.Handlers` epoch bumps and resolver cache invalidation
  for every mapping mutator;
* the same :class:`falcon.App` serving sequential requests where one Request
  keeps its cached media, while a subsequent Request resolves against the
  newly published table;
* the top-level HTTP tables remaining independent from the nested multipart
  :attr:`MultipartParseOptions.media_handlers` table (both epochs and
  contents), including the per-part media cache and stream exhaustion;
* the request/response tables being separate mappings;
* ``MultipartFormHandler`` refusing to serialize.
"""

from __future__ import annotations

import json

import pytest

import falcon
from falcon import media
from falcon import testing
import falcon.asgi
from falcon.constants import MEDIA_JSON
from falcon.constants import MEDIA_MULTIPART
from falcon.constants import MEDIA_URLENCODED

CUSTOM_MEDIA_TYPE = 'application/x-falcon-media-epoch'
NESTED_MEDIA_TYPE = 'application/x-falcon-nested-part'
BOUNDARY = b'MediaEpochBoundary'


class TaggedHandler(media.BaseHandler):
    """HTTP handler that tags every deserialized document with a generation."""

    exhaust_stream = True

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def deserialize(self, stream, content_type, content_length):
        stream.read()
        return {'tag': self.tag}

    async def deserialize_async(self, stream, content_type, content_length):
        await stream.read()
        return {'tag': self.tag}

    def serialize(self, media_obj, content_type):
        return json.dumps(media_obj).encode()


class NestedPartHandler(media.BaseHandler):
    """Nested (multipart part) handler with an observable tag and exhaust."""

    exhaust_stream = True

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def deserialize(self, stream, content_type, content_length):
        raw = stream.read()
        return {'tag': self.tag, 'raw': raw}

    async def deserialize_async(self, stream, content_type, content_length):
        raw = await stream.read()
        return {'tag': self.tag, 'raw': raw}


class NestedTextHandler(media.BaseHandler):
    """Nested handler for parts that default to or declare text/plain."""

    exhaust_stream = True

    def deserialize(self, stream, content_type, content_length):
        return stream.read().decode()

    async def deserialize_async(self, stream, content_type, content_length):
        return (await stream.read()).decode()


def _multipart_body() -> bytes:
    return (
        b'--' + BOUNDARY + b'\r\n'
        b'Content-Disposition: form-data; name="json_part"\r\n'
        b'Content-Type: application/json\r\n\r\n'
        b'{"hello": "world"}\r\n'
        b'--' + BOUNDARY + b'\r\n'
        b'Content-Disposition: form-data; name="text_part"\r\n\r\n'
        b'plain body\r\n'
        b'--' + BOUNDARY + b'\r\n'
        b'Content-Disposition: form-data; name="custom_part"\r\n'
        b'Content-Type: ' + NESTED_MEDIA_TYPE.encode() + b'\r\n\r\n'
        b'nested-payload\r\n'
        b'--' + BOUNDARY + b'--\r\n'
    )


def test_handlers_epoch_starts_at_zero():
    handlers = media.Handlers()
    assert handlers.epoch == 0
    assert media.Handlers({CUSTOM_MEDIA_TYPE: TaggedHandler('x')}).epoch == 0
    assert handlers.copy().epoch == 0


def test_handlers_epoch_bumps_and_resolve_cache_invalidates():
    handlers = media.Handlers()
    v1 = TaggedHandler('v1')
    v2 = TaggedHandler('v2')

    handlers[CUSTOM_MEDIA_TYPE] = v1
    assert handlers.epoch == 1
    resolved, _, _ = handlers._resolve(CUSTOM_MEDIA_TYPE, MEDIA_JSON)
    assert resolved is v1

    handlers[CUSTOM_MEDIA_TYPE] = v2
    assert handlers.epoch == 2
    resolved, _, _ = handlers._resolve(CUSTOM_MEDIA_TYPE, MEDIA_JSON)
    assert resolved is v2

    del handlers[CUSTOM_MEDIA_TYPE]
    assert handlers.epoch == 3
    resolved, _, _ = handlers._resolve(
        CUSTOM_MEDIA_TYPE, MEDIA_JSON, raise_not_found=False
    )
    assert resolved is None

    with pytest.raises(falcon.HTTPUnsupportedMediaType):
        handlers._resolve(CUSTOM_MEDIA_TYPE, MEDIA_JSON)


def test_handlers_all_mutators_invalidate_resolver_cache():
    sentinel = TaggedHandler('sentinel')
    replacement = TaggedHandler('replacement')

    def resolve(handlers):
        handler, _, _ = handlers._resolve(
            CUSTOM_MEDIA_TYPE, MEDIA_JSON, raise_not_found=False
        )
        return handler

    # update()
    handlers = media.Handlers()
    epoch_before = handlers.epoch
    handlers.update({CUSTOM_MEDIA_TYPE: sentinel})
    assert handlers.epoch > epoch_before
    assert resolve(handlers) is sentinel
    handlers.update({CUSTOM_MEDIA_TYPE: replacement})
    assert resolve(handlers) is replacement

    # setdefault()
    handlers = media.Handlers()
    epoch_before = handlers.epoch
    handlers.setdefault(CUSTOM_MEDIA_TYPE, sentinel)
    assert handlers.epoch > epoch_before
    assert resolve(handlers) is sentinel
    handlers.setdefault(CUSTOM_MEDIA_TYPE, replacement)
    assert resolve(handlers) is sentinel

    # pop()
    handlers = media.Handlers({CUSTOM_MEDIA_TYPE: sentinel})
    assert resolve(handlers) is sentinel
    epoch_before = handlers.epoch
    handlers.pop(CUSTOM_MEDIA_TYPE)
    assert handlers.epoch > epoch_before
    assert resolve(handlers) is None

    # popitem()
    handlers = media.Handlers({CUSTOM_MEDIA_TYPE: sentinel})
    assert resolve(handlers) is sentinel
    handlers.popitem()
    assert resolve(handlers) is None

    # clear()
    handlers = media.Handlers({CUSTOM_MEDIA_TYPE: sentinel})
    assert resolve(handlers) is sentinel
    handlers.clear()
    assert resolve(handlers) is None
    assert CUSTOM_MEDIA_TYPE not in handlers


def test_app_http_epoch_starts_at_zero_and_advances():
    app = falcon.App()
    assert app.http_media_epoch == 0
    assert app.req_options._media_epoch == 0
    assert app.resp_options._media_epoch == 0

    assert app.advance_http_media_epoch() == 1
    assert app.http_media_epoch == 1
    assert app.req_options._media_epoch == 1
    assert app.resp_options._media_epoch == 1
    assert app.advance_http_media_epoch() == 2
    assert app.http_media_epoch == 2


def test_advance_http_epoch_is_resolve_cache_publication_barrier():
    app = falcon.App()
    handlers = app.req_options.media_handlers

    v1 = TaggedHandler('v1')
    v2 = TaggedHandler('v2')

    # NOTE: Direct manipulation of ``data`` bypasses the mapping mutators.
    handlers.data[CUSTOM_MEDIA_TYPE] = v1
    handler, _, _ = handlers._resolve(CUSTOM_MEDIA_TYPE, MEDIA_JSON)
    assert handler is v1

    handlers.data[CUSTOM_MEDIA_TYPE] = v2
    # NOTE: The stale cache entry is still served until the host declares a
    #   new HTTP media resolution epoch.
    handler, _, _ = handlers._resolve(CUSTOM_MEDIA_TYPE, MEDIA_JSON)
    assert handler is v1

    app.advance_http_media_epoch()
    handler, _, _ = handlers._resolve(CUSTOM_MEDIA_TYPE, MEDIA_JSON)
    assert handler is v2


def test_req_and_resp_tables_are_independent():
    app = falcon.App()
    req_handlers = app.req_options.media_handlers
    resp_handlers = app.resp_options.media_handlers
    assert req_handlers is not resp_handlers

    req_epoch_before = req_handlers.epoch
    resp_epoch_before = resp_handlers.epoch
    req_handlers[CUSTOM_MEDIA_TYPE] = TaggedHandler('req')
    assert req_handlers.epoch > req_epoch_before
    assert resp_handlers.epoch == resp_epoch_before
    assert CUSTOM_MEDIA_TYPE not in resp_handlers

    resp_handlers[CUSTOM_MEDIA_TYPE] = TaggedHandler('resp')
    assert req_handlers[CUSTOM_MEDIA_TYPE].tag == 'req'
    assert resp_handlers[CUSTOM_MEDIA_TYPE].tag == 'resp'


class _CachingResource:
    """Rewrites the HTTP table between two get_media() calls on one request."""

    def __init__(self, app, asgi):
        self.app = app
        self.asgi = asgi
        self.captured = []

    def _record(self, first_tag, second_tag):
        self.captured.append((first_tag, second_tag))
        return {'first': first_tag, 'second': second_tag}

    def on_post(self, req, resp):
        first = req.get_media()
        # NOTE: The host rewrites the top-level HTTP table mid-request.
        self.app.req_options.media_handlers[CUSTOM_MEDIA_TYPE] = TaggedHandler('v2')
        second = req.get_media()
        doc = self._record(first['tag'], second['tag'])
        resp.text = json.dumps(doc)

    async def on_post_async(self, req, resp):
        first = await req.get_media()
        self.app.req_options.media_handlers[CUSTOM_MEDIA_TYPE] = TaggedHandler('v2')
        second = await req.get_media()
        doc = self._record(first['tag'], second['tag'])
        resp.text = json.dumps(doc)


def test_request_media_cache_forks_from_new_request(asgi, util):
    app = util.create_app(asgi)
    app.req_options.media_handlers[CUSTOM_MEDIA_TYPE] = TaggedHandler('v1')

    resource = _CachingResource(app, asgi)
    app.add_route('/', resource, suffix=('async' if asgi else None))

    # NOTE: First request on this App instance caches v1; rewriting the table
    #   afterwards must not make the same Request recompute its media.
    resp = testing.simulate_post(
        app, '/', body=b'first', content_type=CUSTOM_MEDIA_TYPE
    )
    assert resp.status_code == 200
    assert resp.json == {'first': 'v1', 'second': 'v1'}
    assert resource.captured == [('v1', 'v1')]

    # NOTE: The host declares the next HTTP resolution epoch; a brand new
    #   Request on the same App must resolve against the new table.
    assert app.advance_http_media_epoch() == 1
    resp = testing.simulate_post(
        app, '/', body=b'second', content_type=CUSTOM_MEDIA_TYPE
    )
    assert resp.status_code == 200
    assert resp.json == {'first': 'v2', 'second': 'v2'}
    assert resource.captured[-1] == ('v2', 'v2')


class _TableSwapResource:
    """Replaces the whole top-level table before its first get_media()."""

    def __init__(self, app, tag_after_swap):
        self.app = app
        self.tag_after_swap = tag_after_swap

    def _swap_table(self):
        self.app.req_options.media_handlers = media.Handlers(
            {CUSTOM_MEDIA_TYPE: TaggedHandler(self.tag_after_swap)}
        )

    def on_post(self, req, resp):
        self._swap_table()
        resp.text = req.get_media()['tag']

    async def on_post_async(self, req, resp):
        self._swap_table()
        resp.text = (await req.get_media())['tag']


def test_epoch_pin_behavior_changes_only_after_declaration(asgi, util):
    app = util.create_app(asgi)
    app.req_options.media_handlers = media.Handlers(
        {CUSTOM_MEDIA_TYPE: TaggedHandler('pinned-v1')}
    )

    # NOTE: With no epoch declared yet, the legacy behavior is preserved: the
    #   request resolves through the live table on self.options, so swapping
    #   the table before get_media() changes what this request resolves to.
    resource_legacy = _TableSwapResource(app, 'live')
    app.add_route('/legacy', resource_legacy, suffix=('async' if asgi else None))
    resp = testing.simulate_post(
        app, '/legacy', body=b'legacy', content_type=CUSTOM_MEDIA_TYPE
    )
    assert resp.status_code == 200
    assert resp.text == 'live'

    # NOTE: Declare the HTTP epoch; install a fresh published table and open a
    #   new route on the same App instance.
    app.req_options.media_handlers = media.Handlers(
        {CUSTOM_MEDIA_TYPE: TaggedHandler('pinned-v1')}
    )
    app.advance_http_media_epoch()

    resource_pinned = _TableSwapResource(app, 'live-2')
    app.add_route('/pinned', resource_pinned, suffix=('async' if asgi else None))
    resp = testing.simulate_post(
        app, '/pinned', body=b'pinned', content_type=CUSTOM_MEDIA_TYPE
    )
    assert resp.status_code == 200
    assert resp.text == 'pinned-v1'


class _MultipartResource:
    def __init__(self, multipart_handler, nested_table):
        self._multipart_handler = multipart_handler
        self._nested = nested_table
        self.parts = []

    def _collect(self, req):
        form = req.get_media()
        # NOTE: Request-level caching: resolving the form again must return
        #   the very same cached object on this Request instance.
        assert req.get_media() is form

        for part in form:
            first = part.get_media()
            # NOTE: Mutating the nested table mid-part must not recompute the
            #   cached part media, nor change the pinned nested table ref.
            self._nested[CUSTOM_MEDIA_TYPE + '/other'] = NestedPartHandler(
                'nested-other'
            )
            second = part.get_media()

            # NOTE: Nested handlers requiring exhaust_stream must drain the
            #   part stream after get_media().
            eof = part.stream.read(1)
            self.parts.append(
                {
                    'name': part.name,
                    'content_type': part.content_type,
                    'first': first,
                    'second': second,
                    'eof': eof,
                }
            )
        return None

    def on_post(self, req, resp):
        self._collect(req)
        resp.status = falcon.HTTP_204

    async def on_post_async(self, req, resp):
        form = await req.get_media()
        assert (await req.get_media()) is form

        async for part in form:
            first = await part.get_media()
            self._nested[CUSTOM_MEDIA_TYPE + '/other'] = NestedPartHandler(
                'nested-other'
            )
            second = await part.get_media()

            # NOTE: Nested handlers requiring exhaust_stream must drain the
            #   part stream after get_media().
            eof = await part.stream.read(1)
            self.parts.append(
                {
                    'name': part.name,
                    'content_type': part.content_type,
                    'first': first,
                    'second': second,
                    'eof': eof,
                }
            )

        resp.status = falcon.HTTP_204


def test_multipart_nested_table_is_independent_plane(asgi, util):
    app = util.create_app(asgi)

    multipart_handler = media.MultipartFormHandler()
    nested = multipart_handler.parse_options.media_handlers
    top = app.req_options.media_handlers

    nested[NESTED_MEDIA_TYPE] = NestedPartHandler('nested-v1')
    # NOTE: Parts without an explicit Content-Type resolve against the nested
    #   default media type text/plain; register a handler for it so that the
    #   part's get_media() is observable.
    nested['text/plain'] = NestedTextHandler()
    top[MEDIA_MULTIPART] = multipart_handler

    top_epoch_before = top.epoch
    nested_epoch_before = nested.epoch

    # NOTE: Mutating the nested table must not bump the top-level epoch or
    #   leak nested media types into the top-level HTTP table.
    nested[MEDIA_URLENCODED] = media.URLEncodedFormHandler()
    assert nested.epoch > nested_epoch_before
    assert top.epoch == top_epoch_before
    assert NESTED_MEDIA_TYPE not in top
    assert MEDIA_URLENCODED in nested
    assert top[MEDIA_URLENCODED] is not nested[MEDIA_URLENCODED]

    # NOTE: And vice versa: mutating the top-level table must not touch the
    #   nested table's epoch or contents.
    top_epoch_before = top.epoch
    top[CUSTOM_MEDIA_TYPE] = TaggedHandler('top-only')
    assert top.epoch > top_epoch_before
    assert nested.epoch == nested_epoch_before + 1
    assert CUSTOM_MEDIA_TYPE not in nested

    # NOTE: Advancing the HTTP epoch must never rewrite/sync the nested table.
    assert app.advance_http_media_epoch() == 1
    assert nested.epoch == nested_epoch_before + 1
    assert NESTED_MEDIA_TYPE in nested
    assert nested[NESTED_MEDIA_TYPE].tag == 'nested-v1'

    resource = _MultipartResource(multipart_handler, nested)
    app.add_route('/form', resource, suffix=('async' if asgi else None))

    resp = testing.simulate_post(
        app,
        '/form',
        body=_multipart_body(),
        content_type='multipart/form-data; boundary=' + BOUNDARY.decode(),
    )
    assert resp.status_code == 204

    by_name = {part['name']: part for part in resource.parts}

    json_part = by_name['json_part']
    assert json_part['content_type'] == MEDIA_JSON
    assert json_part['first'] == {'hello': 'world'}
    assert json_part['second'] is json_part['first']

    text_part = by_name['text_part']
    assert text_part['content_type'] == 'text/plain'
    assert text_part['first'] == 'plain body'
    assert text_part['second'] is text_part['first']

    custom_part = by_name['custom_part']
    assert custom_part['content_type'] == NESTED_MEDIA_TYPE
    assert custom_part['first'] == {'tag': 'nested-v1', 'raw': b'nested-payload'}
    assert custom_part['second'] is custom_part['first']
    assert custom_part['eof'] == b''

    # NOTE: Nested table mutations made while iterating must not have leaked
    #   back into the top-level HTTP table.
    assert CUSTOM_MEDIA_TYPE + '/other' not in top
    assert CUSTOM_MEDIA_TYPE + '/other' in nested

    # NOTE: A second Request on the same App observes an independently
    #   reconfigured nested table (nested epoch bumped again), while the
    #   top-level table is untouched by the nested change.
    top_epoch_before = top.epoch
    nested[NESTED_MEDIA_TYPE] = NestedPartHandler('nested-v2')
    resource.parts.clear()
    resp = testing.simulate_post(
        app,
        '/form',
        body=_multipart_body(),
        content_type='multipart/form-data; boundary=' + BOUNDARY.decode(),
    )
    assert resp.status_code == 204
    assert top.epoch == top_epoch_before
    by_name = {part['name']: part for part in resource.parts}
    assert by_name['custom_part']['first'] == {
        'tag': 'nested-v2',
        'raw': b'nested-payload',
    }


def test_multipart_form_handler_refuses_serialization(asgi, util):
    with pytest.raises(NotImplementedError):
        media.MultipartFormHandler().serialize({'field': 'value'}, MEDIA_MULTIPART)

    # NOTE: The WSGI and ASGI apps share the same multipart handler class;
    #   ensure it is impossible to use it to serialize Response.media either.
    app = util.create_app(asgi)
    top_multipart = app.req_options.media_handlers[MEDIA_MULTIPART]
    assert isinstance(top_multipart, media.MultipartFormHandler)
    with pytest.raises(NotImplementedError):
        top_multipart.serialize({'field': 'value'}, MEDIA_MULTIPART)
