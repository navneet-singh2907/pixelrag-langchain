import httpx
import pytest
import respx

from pixelrag_langchain.client import (
    PixelRAGAPIError,
    PixelRAGClient,
    PixelRAGConnectionError,
    PixelRAGTimeoutError,
)
from pixelrag_langchain.config import PIXELRAG_HOSTED_BASE_URL, PixelRAGConfig


def test_hosted_config_points_at_documented_endpoint():
    cfg = PixelRAGConfig.hosted()
    assert cfg.base_url == PIXELRAG_HOSTED_BASE_URL == "https://api.pixelrag.ai"


def test_hosted_config_default_differs_from_plain_default():
    """Constructing PixelRAGConfig() directly must NOT hit a third party host."""
    assert PixelRAGConfig().base_url != PixelRAGConfig.hosted().base_url
    assert PixelRAGConfig().base_url.startswith("http://localhost")


def test_hosted_config_accepts_overrides():
    cfg = PixelRAGConfig.hosted(n_docs=10, timeout_s=5.0)
    assert cfg.n_docs == 10
    assert cfg.timeout_s == 5.0
    assert cfg.base_url == PIXELRAG_HOSTED_BASE_URL


@pytest.fixture
def client() -> PixelRAGClient:
    return PixelRAGClient(PixelRAGConfig(base_url="http://testserver:30001"))


@respx.mock
def test_search_happy_path(client: PixelRAGClient):
    respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    [
                        {"score": 0.91, "tile_id": "t1", "source": "France", "text": "Paris"},
                        {"score": 0.80, "tile_id": "t2", "source": "France_2"},
                    ]
                ]
            },
        )
    )
    tiles = client.search("capital of France", n_docs=2)
    assert len(tiles) == 2
    assert tiles[0].tile_id == "t1"
    assert tiles[0].caption == "Paris"
    assert tiles[0].score == 0.91


@respx.mock
def test_search_flat_results_shape(client: PixelRAGClient):
    """Server might return a flat list instead of a list-of-lists."""
    respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(200, json={"results": [{"score": 0.5, "id": "x"}]})
    )
    tiles = client.search("query")
    assert len(tiles) == 1
    assert tiles[0].tile_id == "x"


@respx.mock
def test_search_unrecognized_shape_returns_empty(client: PixelRAGClient):
    respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(200, json={"unexpected_key": []})
    )
    assert client.search("query") == []


def test_search_empty_query_short_circuits(client: PixelRAGClient):
    assert client.search("") == []
    assert client.search("   ") == []


@respx.mock
def test_search_connection_error_wrapped(client: PixelRAGClient):
    respx.post("http://testserver:30001/search").mock(
        side_effect=httpx.ConnectError("boom")
    )
    with pytest.raises(PixelRAGConnectionError):
        client.search("query")


@respx.mock
def test_search_timeout_wrapped(client: PixelRAGClient):
    respx.post("http://testserver:30001/search").mock(
        side_effect=httpx.TimeoutException("boom")
    )
    with pytest.raises(PixelRAGTimeoutError):
        client.search("query")


@respx.mock
def test_search_http_error_status_wrapped(client: PixelRAGClient):
    respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(500, text="index not loaded")
    )
    with pytest.raises(PixelRAGAPIError) as exc_info:
        client.search("query")
    assert exc_info.value.status_code == 500


@respx.mock
def test_search_malformed_json_wrapped(client: PixelRAGClient):
    respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(200, text="not json")
    )
    with pytest.raises(PixelRAGAPIError):
        client.search("query")


@respx.mock
def test_health_never_raises_on_down_server(client: PixelRAGClient):
    respx.get("http://testserver:30001/health").mock(side_effect=httpx.ConnectError("down"))
    assert client.health() is False
