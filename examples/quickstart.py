"""Minimal end-to-end example.

Set PIXELRAG_MODE=hosted to hit upstream's public demo endpoint (zero setup,
Wikipedia only, api.pixelrag.ai — see the upstream README). Default is
`local`, which expects your own `pixelrag serve` instance:

    pip install 'pixelrag[serve]'
    pixelrag serve --index-dir ./index --port 30001

See https://github.com/StarTrail-org/PixelRAG for building/downloading an index.
"""

import os

from pixelrag_langchain import PixelRAGConfig, PixelRAGRetriever

MODE = os.environ.get("PIXELRAG_MODE", "local")  # "local" or "hosted"


def main() -> None:
    config = PixelRAGConfig.hosted(n_docs=3) if MODE == "hosted" else PixelRAGConfig(
        base_url="http://localhost:30001", n_docs=3
    )
    retriever = PixelRAGRetriever(config=config)

    if not retriever._get_client().health():
        print(f"No response from {config.base_url}.")
        if MODE == "local":
            print("Start `pixelrag serve` first, or run with PIXELRAG_MODE=hosted to skip setup.")
        return

    query = "What is the capital of France?"
    docs = retriever.invoke(query)

    print(f"{len(docs)} results for: {query!r}  (mode={MODE})\n")
    for doc in docs:
        md = doc.metadata
        print(f"  score={md['score']:.3f}  source={md['source']}  tile_id={md['tile_id']}")
        if doc.page_content:
            print(f"    caption: {doc.page_content}")
        if md.get("image_url"):
            print(f"    image:   {md['image_url']}")
        elif md.get("tile_path"):
            print(f"    tile:    {md['tile_path']}")


if __name__ == "__main__":
    main()
