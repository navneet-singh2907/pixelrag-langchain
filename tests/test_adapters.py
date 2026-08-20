import json

import httpx
import respx

from pixelrag_langchain import (
    PixelRAGConfig,
    PixelRAGRetriever,
    PixelRAGSearchTool,
    PixelRAGTile,
)


BASE_URL = "http://testserver:30001"


def _search_response():
    return {
        "results": [
            {
                "hits": [
                    {
                        "score": 0.91,
                        "vector_id": "v1",
                        "article_id": 42,
                        "tile_index": 1,
                        "chunk_index": 3,
                        "url": "Example_article",
                        "image_base64": None,
                    }
                ]
            }
        ]
    }


@respx.mock
def test_retriever_uses_config_n_docs_and_exposes_fetchable_image_url():
    route = respx.post(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, json=_search_response())
    )
    retriever = PixelRAGRetriever(
        config=PixelRAGConfig(base_url=BASE_URL, n_docs=3)
    )

    docs = retriever.invoke("find the diagram")

    payload = json.loads(route.calls.last.request.content)
    assert payload["n_docs"] == 3
    assert docs[0].metadata["tile_url"] == f"{BASE_URL}/tile/42/1/3"
    assert docs[0].metadata["image_url"] == f"{BASE_URL}/tile/42/1/3"
    assert docs[0].metadata["tile_index"] == 1
    assert docs[0].metadata["chunk_index"] == 3
    retriever.close()


@respx.mock
def test_retriever_can_request_inline_images():
    response = _search_response()
    response["results"][0]["hits"][0]["image_base64"] = "aGVsbG8="
    route = respx.post(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, json=response)
    )
    retriever = PixelRAGRetriever(
        config=PixelRAGConfig(base_url=BASE_URL), include_images=True
    )

    docs = retriever.invoke("find the diagram")

    payload = json.loads(route.calls.last.request.content)
    assert payload["include_images"] is True
    assert docs[0].metadata["image_base64"] == "aGVsbG8="
    retriever.close()


@respx.mock
def test_tool_exposes_coordinate_url_and_bytes():
    respx.post(f"{BASE_URL}/search").mock(
        return_value=httpx.Response(200, json=_search_response())
    )
    respx.get(f"{BASE_URL}/tile/42/1/3").mock(
        return_value=httpx.Response(200, content=b"png")
    )
    tool = PixelRAGSearchTool(config=PixelRAGConfig(base_url=BASE_URL))

    tile = tool.search_tiles("find the diagram", n_docs=1)[0]

    assert tool.tile_image_url(tile) == f"{BASE_URL}/tile/42/1/3"
    assert tool.fetch_tile_bytes(tile) == b"png"
    assert f"image={BASE_URL}/tile/42/1/3" in tool.invoke(
        {"query": "find the diagram", "n_docs": 1}
    )
    tool.close()


def test_langgraph_example_hands_coordinate_url_to_vlm():
    from examples.langgraph_node import build_reader_message

    tile = PixelRAGTile.from_api_record(
        {
            "score": 0.9,
            "article_id": 42,
            "tile_index": 1,
            "chunk_index": 3,
        }
    )

    message = build_reader_message("read this", [tile])

    assert message.content[1]["image_url"]["url"] == (
        "http://localhost:30001/tile/42/1/3"
    )
