import json

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


@respx.mock
def test_search_real_live_schema_july_2026(client: PixelRAGClient):
    """Regression test using the exact response shape observed from the live
    api.pixelrag.ai endpoint (captured July 2026): per-query objects wrap
    records in a "hits" list, "url" holds the article TITLE, "path" is a
    relative tile path, ids are integers."""
    live_response = {
        "results": [
            {
                "hits": [
                    {
                        "score": 0.5885330438613892,
                        "vector_id": 18869670,
                        "article_id": 5663927,
                        "tile_index": 0,
                        "chunk_index": 7,
                        "y_offset": 7168,
                        "tile_height": 1024,
                        "path": "shard_683/shard_00005/5663927.png.tiles/chunk_0000_07.png",
                        "url": "Outline_of_France",
                        "article_pages": "0:0-7,1:0-7,2:0-7",
                        "image_base64": None,
                    },
                    {
                        "score": 0.5356552600860596,
                        "vector_id": 19814915,
                        "article_id": 6007128,
                        "tile_index": 0,
                        "chunk_index": 0,
                        "y_offset": 0,
                        "tile_height": 1024,
                        "path": "shard_725/shard_00001/6007128.png.tiles/chunk_0000_00.png",
                        "url": "Portal:France/Geography",
                        "article_pages": "0:0-4",
                        "image_base64": None,
                    },
                ]
            }
        ]
    }
    respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(200, json=live_response)
    )
    tiles = client.search("What is the capital of France?", n_docs=2)

    assert len(tiles) == 2  # hits unwrapped, not treated as one record
    first = tiles[0]
    assert first.score == pytest.approx(0.5885330438613892)
    assert first.tile_id == "18869670"          # from vector_id
    assert first.article_id == "5663927"
    assert first.source == "Outline_of_France"  # "url" is a title -> source
    assert first.image_url is None              # title must NOT leak into image_url
    assert first.tile_path == "shard_683/shard_00005/5663927.png.tiles/chunk_0000_07.png"
    assert first.image_base64 is None
    assert tiles[1].source == "Portal:France/Geography"


def test_url_field_that_is_actually_a_url_maps_to_image_url():
    """If a server DOES return a real http(s) URL in "url", keep old behavior."""
    from pixelrag_langchain.client import PixelRAGTile

    tile = PixelRAGTile.from_api_record(
        {"score": 0.9, "id": "t1", "url": "https://tiles.example.com/1.png"}
    )
    assert tile.image_url == "https://tiles.example.com/1.png"
    assert tile.source is None or not tile.source.startswith("https://")


def _coordinate_hit(**overrides):
    hit = {
        "score": 0.58,
        "vector_id": 18869670,
        "article_id": 5663927,
        "tile_index": 0,
        "chunk_index": 7,
        "path": "shard_683/5663927.png.tiles/chunk_0000_07.png",
        "url": "Outline_of_France",
        "image_base64": None,
    }
    hit.update(overrides)
    return hit


def test_tile_coordinates_and_url_are_derived_from_hit(client: PixelRAGClient):
    from pixelrag_langchain.client import PixelRAGTile

    tile = PixelRAGTile.from_api_record(_coordinate_hit())

    assert tile.tile_index == 0
    assert tile.chunk_index == 7
    assert tile.has_tile_coordinates is True
    assert client.tile_url(tile) == "http://testserver:30001/tile/5663927/0/7"


def test_tile_coordinates_accept_numeric_strings(client: PixelRAGClient):
    from pixelrag_langchain.client import PixelRAGTile

    tile = PixelRAGTile.from_api_record(
        _coordinate_hit(tile_index="2", chunk_index="9")
    )

    assert client.tile_url(tile) == "http://testserver:30001/tile/5663927/2/9"


def test_tile_url_escapes_server_provided_article_id(client: PixelRAGClient):
    from pixelrag_langchain.client import PixelRAGTile

    tile = PixelRAGTile.from_api_record(
        _coordinate_hit(article_id="collection/item name")
    )

    assert client.tile_url(tile) == (
        "http://testserver:30001/tile/collection%2Fitem%20name/0/7"
    )


def test_tile_url_none_and_fetch_raises_without_coordinates(client: PixelRAGClient):
    from pixelrag_langchain.client import PixelRAGTile

    tile = PixelRAGTile.from_api_record({"score": 0.5})
    assert client.tile_url(tile) is None
    with pytest.raises(ValueError, match="missing"):
        client.fetch_tile(tile)


@respx.mock
def test_fetch_tile_returns_image_bytes(client: PixelRAGClient):
    from pixelrag_langchain.client import PixelRAGTile

    png = b"\x89PNG\r\n\x1a\nimage"
    respx.get("http://testserver:30001/tile/5663927/0/7").mock(
        return_value=httpx.Response(200, content=png, headers={"content-type": "image/png"})
    )

    assert client.fetch_tile(PixelRAGTile.from_api_record(_coordinate_hit())) == png


@respx.mock
def test_fetch_tile_wraps_http_and_transport_errors(client: PixelRAGClient):
    from pixelrag_langchain.client import PixelRAGTile

    tile = PixelRAGTile.from_api_record(_coordinate_hit())
    route = respx.get("http://testserver:30001/tile/5663927/0/7")
    route.mock(return_value=httpx.Response(404, text="missing tile"))
    with pytest.raises(PixelRAGAPIError):
        client.fetch_tile(tile)

    route.mock(side_effect=httpx.ReadError("connection reset"))
    with pytest.raises(PixelRAGConnectionError):
        client.fetch_tile(tile)


@respx.mock
def test_include_images_is_explicit_opt_in(client: PixelRAGClient):
    route = respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"hits": [_coordinate_hit(image_base64="aGVsbG8=")]}]},
        )
    )

    tiles = client.search("query", include_images=True)

    payload = json.loads(route.calls.last.request.content)
    assert payload["include_images"] is True
    assert tiles[0].image_base64 == "aGVsbG8="


@respx.mock
def test_include_images_is_omitted_by_default(client: PixelRAGClient):
    route = respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    client.search("query")

    assert b"include_images" not in route.calls.last.request.content


def test_search_rejects_non_positive_n_docs(client: PixelRAGClient):
    with pytest.raises(ValueError, match="at least 1"):
        client.search("query", n_docs=0)


@respx.mock
def test_search_non_object_json_degrades_to_empty(client: PixelRAGClient):
    respx.post("http://testserver:30001/search").mock(
        return_value=httpx.Response(200, json=[])
    )
    assert client.search("query") == []


@respx.mock
def test_status_returns_server_diagnostics(client: PixelRAGClient):
    respx.get("http://testserver:30001/status").mock(
        return_value=httpx.Response(200, json={"vectors": 123, "dimension": 2048})
    )

    assert client.status() == {"vectors": 123, "dimension": 2048}


@respx.mock
def test_status_rejects_non_object_json(client: PixelRAGClient):
    respx.get("http://testserver:30001/status").mock(
        return_value=httpx.Response(200, json=[])
    )
    with pytest.raises(PixelRAGAPIError):
        client.status()
