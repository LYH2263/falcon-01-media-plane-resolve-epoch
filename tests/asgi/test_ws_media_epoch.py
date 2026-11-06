"""Tests for the WebSocket media resolution epoch.

These tests all run against the SAME ASGI App instance to verify that:

* the default TEXT/BINARY handler fork (incl. the MissingDependencyHandler
  branch) is intact;
* replacing one payload type's handler never replaces the other's;
* an already accepted connection keeps the serialize/deserialize callables
  it bound at construction time, while connections created after the host
  declares a new WebSocket media epoch bind the newly published handlers;
* the WebSocket epoch is tracked independently from the HTTP epoch;
* the ws_options.media_handlers mapping stays a plain mapping rather than
  becoming an HTTP-style Handlers table.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import pytest

import falcon
from falcon import media
from falcon import testing
import falcon.asgi
from falcon.asgi.ws import WebSocketOptions

try:
    import msgpack
except ImportError:
    msgpack = None

TEXT = falcon.WebSocketPayloadType.TEXT
BINARY = falcon.WebSocketPayloadType.BINARY
_WsHandler = media.TextBaseHandlerWS | media.BinaryBaseHandlerWS


class TaggedTextHandler(media.TextBaseHandlerWS):
    """TEXT handler whose serialize/deserialize output carries a tag."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def serialize(self, media_obj: object) -> str:
        return json.dumps({'tag': self.tag, 'obj': media_obj}, sort_keys=True)

    def deserialize(self, payload: str) -> dict[str, Any]:
        return {'tag': self.tag, 'doc': json.loads(payload)}


class TaggedBinaryHandler(media.BinaryBaseHandlerWS):
    """BINARY handler whose serialize/deserialize output carries a tag."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._prefix = ('BIN:' + tag + ':').encode()

    def serialize(self, media_obj: object) -> bytes:
        return self._prefix + json.dumps(media_obj, sort_keys=True).encode()

    def deserialize(self, payload: bytes) -> dict[str, Any]:
        assert payload.startswith(self._prefix)
        return {
            'tag': self.tag,
            'doc': json.loads(payload[len(self._prefix) :].decode()),
        }


def test_default_ws_handlers_and_missing_dependency_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = WebSocketOptions()

    assert options.media_epoch == 0
    assert isinstance(options.media_handlers[TEXT], media.JSONHandlerWS)

    binary = options.media_handlers[BINARY]
    if msgpack:
        assert isinstance(binary, media.MessagePackHandlerWS)
    else:
        assert isinstance(binary, media.MissingDependencyHandler)

    # NOTE: Simulate msgpack being unavailable to deterministically exercise
    #   the other side of the fork.
    monkeypatch.setitem(sys.modules, 'msgpack', None)
    options = WebSocketOptions()
    binary = options.media_handlers[BINARY]
    assert isinstance(binary, media.MissingDependencyHandler)
    with pytest.raises(RuntimeError, match='msgpack'):
        binary.serialize({'answer': 42})
    with pytest.raises(RuntimeError, match='msgpack'):
        binary.deserialize(b'\x00')


def test_ws_options_media_handlers_stays_plain_mapping() -> None:
    app = falcon.asgi.App()
    mapping = app.ws_options.media_handlers
    assert isinstance(mapping, dict)
    assert not isinstance(mapping, media.Handlers)
    assert set(mapping.keys()) == {TEXT, BINARY}


async def test_missing_dependency_handler_raises_on_connection() -> None:
    app = falcon.asgi.App()
    app.ws_options.media_handlers[BINARY] = media.MissingDependencyHandler(
        'test WebSocket media handler for BINARY payloads', 'msgpack'
    )

    captured = []

    class Resource:
        async def on_websocket(self, req, ws):
            await ws.accept()

            try:
                await ws.send_media({'answer': 42}, payload_type=BINARY)
            except RuntimeError as ex:
                captured.append(('send', str(ex)))

            try:
                await ws.receive_media()
            except RuntimeError as ex:
                captured.append(('receive', str(ex)))

            await ws.close()

    app.add_route('/', Resource())

    conductor = testing.ASGIConductor(app)
    async with conductor as c:
        async with c.simulate_ws() as ws:
            await ws.send_data(b'\x00\x01\x02')

    assert [kind for kind, _ in captured] == ['send', 'receive']
    assert all('msgpack' in message for _, message in captured)


class _Owner:
    def __init__(self) -> None:
        self.epochs: list[tuple[str, int]] = []
        self.docs: list[tuple[str, str, object]] = []
        self.pinned: list[
            tuple[str, dict[falcon.WebSocketPayloadType, _WsHandler]]
        ] = []
        self.first_round_done = asyncio.Event()
        self.release_first = asyncio.Event()


class _MediaResource:
    def __init__(self, owner: _Owner, name: str, hold: bool) -> None:
        self._owner = owner
        self._name = name
        self._hold = hold

    async def _round(self, ws: falcon.asgi.WebSocket, round_no: int) -> None:
        text_doc = await ws.receive_media()
        self._owner.docs.append((self._name, 'text', text_doc))
        await ws.send_media({'round': round_no})

        bin_doc = await ws.receive_media()
        self._owner.docs.append((self._name, 'binary', bin_doc))
        await ws.send_media({'round': round_no}, payload_type=BINARY)

    async def on_websocket(self, req: object, ws: falcon.asgi.WebSocket) -> None:
        await ws.accept()

        self._owner.epochs.append((self._name, ws.media_epoch))
        # NOTE: Snapshot the connection-bound payload mapping so that the test
        #   can prove it diverges from ws_options.media_handlers later on.
        self._owner.pinned.append((self._name, dict(ws._media_handlers)))

        await self._round(ws, 1)

        if self._name == 'one':
            self._owner.first_round_done.set()
            await self._owner.release_first.wait()
            await self._round(ws, 2)

        await ws.close()


async def _client_round(
    ws: testing.ASGIWebSocketSimulator, tag: str, round_no: int
) -> None:
    payload = json.dumps({'round': round_no})
    await ws.send_text(payload)

    reply = json.loads(await ws.receive_text())
    assert reply == {'tag': tag, 'obj': {'round': round_no}}

    await ws.send_data(('BIN:' + tag + ':').encode() + payload.encode())
    binary_reply = await ws.receive_data()
    assert binary_reply == ('BIN:' + tag + ':').encode() + payload.encode()


async def test_ws_epoch_accepted_vs_new_connection_binding_fork() -> None:
    app = falcon.asgi.App()
    assert app.ws_media_epoch == 0
    assert app.http_media_epoch == 0

    # NOTE: Pre-publish v1 handlers on both payload types while the WS plane
    #   is still at epoch 0.
    text_v1 = TaggedTextHandler('v1')
    binary_v1 = TaggedBinaryHandler('v1')
    app.ws_options.media_handlers[TEXT] = text_v1
    app.ws_options.media_handlers[BINARY] = binary_v1

    owner = _Owner()
    app.add_route('/one', _MediaResource(owner, 'one', hold=True))
    app.add_route('/two', _MediaResource(owner, 'two', hold=False))
    app.add_route('/three', _MediaResource(owner, 'three', hold=False))

    conductor = testing.ASGIConductor(app)
    async with conductor as c:
        # NOTE: First connection is accepted (and binds) under epoch 0.
        async with c.simulate_ws('/one') as ws_one:
            await _client_round(ws_one, 'v1', 1)

            # NOTE: Give the server a moment to record its round, then publish
            #   a new WS generation with distinct TEXT/BINARY handlers.
            await owner.first_round_done.wait()

            mapping = app.ws_options.media_handlers
            text_v2 = TaggedTextHandler('v2')
            binary_v2 = TaggedBinaryHandler('v2')

            # NOTE: TEXT and BINARY stay independent: assigning TEXT must not
            #   replace the BINARY handler (and vice versa).
            mapping[TEXT] = text_v2
            assert mapping[BINARY] is binary_v1
            mapping[BINARY] = binary_v2
            assert mapping[TEXT] is text_v2

            # NOTE: WS and HTTP epochs are tracked independently.
            assert app.advance_ws_media_epoch() == 1
            assert app.ws_media_epoch == 1
            assert app.http_media_epoch == 0

            # NOTE: A brand new connection on the SAME app binds epoch 1.
            async with c.simulate_ws('/two') as ws_two:
                await _client_round(ws_two, 'v2', 1)

            # NOTE: Advancing the HTTP plane must not advance or rebind the WS
            #   plane; another new connection stays on WS epoch 1 and keeps v2.
            assert app.advance_http_media_epoch() == 1
            assert app.http_media_epoch == 1
            assert app.ws_media_epoch == 1
            async with c.simulate_ws('/three') as ws_three:
                await _client_round(ws_three, 'v2', 1)

            # NOTE: Finally resume the already accepted first connection and
            #   verify its serialize/deserialize callables are still the v1
            #   ones it bound at construction, rather than the v2 mappings.
            owner.release_first.set()
            await _client_round(ws_one, 'v1', 2)

    # NOTE: Server-side observations: epochs and per-connection deserialized
    #   documents line up with the bindings.
    assert [epoch for _, epoch in owner.epochs] == [0, 1, 1]
    assert [name for name, _ in owner.epochs] == ['one', 'two', 'three']

    one_pinned = owner.pinned[0][1]
    assert one_pinned[TEXT] is text_v1
    assert one_pinned[BINARY] is binary_v1

    two_pinned = owner.pinned[1][1]
    assert two_pinned[TEXT] is text_v2
    assert two_pinned[BINARY] is binary_v2

    three_pinned = owner.pinned[2][1]
    assert three_pinned[TEXT] is text_v2
    assert three_pinned[BINARY] is binary_v2

    one_docs = [(kind, doc) for name, kind, doc in owner.docs if name == 'one']
    assert one_docs == [
        ('text', {'tag': 'v1', 'doc': {'round': 1}}),
        ('binary', {'tag': 'v1', 'doc': {'round': 1}}),
        ('text', {'tag': 'v1', 'doc': {'round': 2}}),
        ('binary', {'tag': 'v1', 'doc': {'round': 2}}),
    ]
    two_docs = [(kind, doc) for name, kind, doc in owner.docs if name == 'two']
    assert two_docs == [
        ('text', {'tag': 'v2', 'doc': {'round': 1}}),
        ('binary', {'tag': 'v2', 'doc': {'round': 1}}),
    ]
    three_docs = [(kind, doc) for name, kind, doc in owner.docs if name == 'three']
    assert three_docs == [
        ('text', {'tag': 'v2', 'doc': {'round': 1}}),
        ('binary', {'tag': 'v2', 'doc': {'round': 1}}),
    ]


async def test_ws_epoch_handlers_are_bound_per_connection_from_mapping() -> None:
    # NOTE: A tighter, mapping-focused check: publish two generations in a row
    #   and ensure each accepted connection snapshots its own mapping copy, so
    #   mutating the shared ws_options mapping in place after acceptance can
    #   not retroactively affect the open connection.
    app = falcon.asgi.App()
    owner = _Owner()
    app.add_route('/one', _MediaResource(owner, 'one', hold=True))
    app.add_route('/two', _MediaResource(owner, 'two', hold=False))

    text_v1 = TaggedTextHandler('v1')
    binary_v1 = TaggedBinaryHandler('v1')
    app.ws_options.media_handlers[TEXT] = text_v1
    app.ws_options.media_handlers[BINARY] = binary_v1
    app.advance_ws_media_epoch()

    conductor = testing.ASGIConductor(app)
    async with conductor as c:
        async with c.simulate_ws('/one') as ws_one:
            await _client_round(ws_one, 'v1', 1)
            await owner.first_round_done.wait()

            # NOTE: Mutate the SAME mapping object in place and publish a new
            #   epoch; the accepted connection's snapshot must be untouched.
            app.ws_options.media_handlers[TEXT] = TaggedTextHandler('v2')
            app.ws_options.media_handlers[BINARY] = TaggedBinaryHandler('v2')
            app.advance_ws_media_epoch()

            async with c.simulate_ws('/two') as ws_two:
                await _client_round(ws_two, 'v2', 1)

            owner.release_first.set()
            await _client_round(ws_one, 'v1', 2)

    assert [epoch for _, epoch in owner.epochs] == [1, 2]
    assert owner.pinned[0][1][TEXT] is text_v1
    assert owner.pinned[0][1][BINARY] is binary_v1
    one_docs = [(kind, doc) for name, kind, doc in owner.docs if name == 'one']
    assert one_docs == [
        ('text', {'tag': 'v1', 'doc': {'round': 1}}),
        ('binary', {'tag': 'v1', 'doc': {'round': 1}}),
        ('text', {'tag': 'v1', 'doc': {'round': 2}}),
        ('binary', {'tag': 'v1', 'doc': {'round': 2}}),
    ]
    two_docs = [(kind, doc) for name, kind, doc in owner.docs if name == 'two']
    assert two_docs == [
        ('text', {'tag': 'v2', 'doc': {'round': 1}}),
        ('binary', {'tag': 'v2', 'doc': {'round': 1}}),
    ]
