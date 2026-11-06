import asyncio
from contextlib import asynccontextmanager

from frameworks import mcp_client
from frameworks.mcp_client import MCPClientWrapper


def test_http_client_headers_are_passed_through_mcp_http_client(monkeypatch):
    async def exercise():
        token_headers = {"Authorization": "Bearer test-token"}
        state = {}

        class FakeHttpClient:
            async def __aenter__(self):
                state["http_client_entered"] = True
                return self

            async def __aexit__(self, *_):
                state["http_client_closed"] = True

        def create_http_client(*, headers):
            state["configured_headers"] = headers
            return FakeHttpClient()

        @asynccontextmanager
        async def fake_streamable_http_client(url, *, http_client=None):
            state["url"] = url
            state["transport_http_client"] = http_client
            yield object(), object()

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def initialize(self):
                state["initialized"] = True

        monkeypatch.setattr(
            mcp_client, "create_mcp_http_client", create_http_client
        )
        monkeypatch.setattr(
            mcp_client, "streamable_http_client", fake_streamable_http_client
        )
        monkeypatch.setattr(mcp_client, "ClientSession", lambda *_: FakeSession())

        client = MCPClientWrapper(
            transport="http",
            url="https://example.test/mcp",
            headers=token_headers,
        )
        await client.connect()

        assert state["configured_headers"] == token_headers
        assert state["transport_http_client"] is not None
        assert state["initialized"] is True

        await client.close()
        assert state["http_client_closed"] is True

    asyncio.run(exercise())


def test_http_client_without_headers_uses_sdk_managed_client(monkeypatch):
    async def exercise():
        state = {}

        @asynccontextmanager
        async def fake_streamable_http_client(url, *, http_client=None):
            state["url"] = url
            state["http_client"] = http_client
            yield object(), object()

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def initialize(self):
                return None

        monkeypatch.setattr(
            mcp_client, "streamable_http_client", fake_streamable_http_client
        )
        monkeypatch.setattr(mcp_client, "ClientSession", lambda *_: FakeSession())

        client = MCPClientWrapper(transport="http", url="https://example.test/mcp")
        await client.connect()

        assert state["url"] == "https://example.test/mcp"
        assert state["http_client"] is None

        await client.close()

    asyncio.run(exercise())
