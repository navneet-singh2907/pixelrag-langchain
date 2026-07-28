"""LangChain-compatible retriever backed by a pixelrag-serve instance.

Note on Documents: LangChain's `Document` is a text-container. Screenshot
tiles are images, so we don't lie about that — `page_content` holds whatever
caption/OCR text the server gave us (often none), and the actual image
reference (URL or base64) lives in `metadata`. If your downstream chain needs
the pixels, read `metadata["image_url"]` or `metadata["image_base64"]` and
pass them to a vision-capable model yourself.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pixelrag_langchain.client import PixelRAGClient, PixelRAGError
from pixelrag_langchain.config import PixelRAGConfig


class PixelRAGRetriever(BaseRetriever):
    """Retrieve screenshot tiles from a pixelrag-serve index as Documents.

    Example:
        >>> retriever = PixelRAGRetriever(config=PixelRAGConfig(base_url="http://localhost:30001"))
        >>> docs = retriever.invoke("What is the capital of France?")
    """

    config: PixelRAGConfig = PixelRAGConfig()
    n_docs: int = 5
    raise_on_error: bool = False
    """If False (default), connection/timeout errors return [] instead of
    raising, so a flaky visual index doesn't crash the whole chain. Set True
    if you'd rather fail loudly and handle it yourself."""

    _client: PixelRAGClient | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> PixelRAGClient:
        if self._client is None:
            self._client = PixelRAGClient(self.config)
        return self._client

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        client = self._get_client()
        try:
            tiles = client.search(query, n_docs=self.n_docs)
        except PixelRAGError as exc:
            run_manager.on_retriever_error(exc)
            if self.raise_on_error:
                raise
            return []

        documents: list[Document] = []
        for tile in tiles:
            metadata: dict[str, Any] = {
                "score": tile.score,
                "tile_id": tile.tile_id,
                "source": tile.source,
                "image_url": tile.image_url,
                "image_base64": tile.image_base64,
                "modality": "image",
            }
            documents.append(
                Document(page_content=tile.caption or "", metadata=metadata)
            )
        return documents
