"""LangChain Tool wrapping PixelRAG visual search for use in agents.

Two entry points, because agents have two different needs:

  - `PixelRAGSearchTool`  — a standard `BaseTool`. `_run()` returns a plain
    text summary (source, score, any caption) so it drops straight into a
    normal text-only tool-calling agent loop.

  - `PixelRAGSearchTool.search_tiles()` — returns the raw `PixelRAGTile`
    objects, image data included. Use this in a LangGraph node (or anywhere
    you control message construction) when you want to hand the retrieved
    screenshots to a vision-capable model as actual image content, which is
    the point of PixelRAG — reading pixels, not just a text summary of them.
"""

from __future__ import annotations

from typing import Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from pixelrag_langchain.client import PixelRAGClient, PixelRAGError, PixelRAGTile
from pixelrag_langchain.config import PixelRAGConfig


class PixelRAGSearchInput(BaseModel):
    query: str = Field(description="Natural-language search query.")
    n_docs: int = Field(default=5, ge=1, le=50, description="Number of tiles to retrieve.")


class PixelRAGSearchTool(BaseTool):
    """Search a PixelRAG visual index and return the top matching tiles.

    Requires a running pixelrag-serve instance (self-hosted by default —
    see the upstream README: https://github.com/StarTrail-org/PixelRAG).
    Point `config.base_url` at your own server; nothing is assumed for you.
    """

    name: str = "pixelrag_visual_search"
    description: str = (
        "Search an index of document screenshots (web pages, PDFs, images) "
        "and return the most visually/semantically relevant tiles, with "
        "similarity scores and source references. Prefer this over a plain "
        "text search when the answer likely lives in a table, chart, "
        "infographic, or other layout that plain text extraction would "
        "mangle."
    )
    args_schema: Type[BaseModel] = PixelRAGSearchInput

    config: PixelRAGConfig = PixelRAGConfig()
    _client: PixelRAGClient | None = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> PixelRAGClient:
        if self._client is None:
            self._client = PixelRAGClient(self.config)
        return self._client

    def search_tiles(self, query: str, n_docs: int = 5) -> list[PixelRAGTile]:
        """Structured access, image data intact. Use this from LangGraph nodes."""
        return self._get_client().search(query, n_docs=n_docs)

    def _run(
        self,
        query: str,
        n_docs: int = 5,
        *,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            tiles = self.search_tiles(query, n_docs=n_docs)
        except PixelRAGError as exc:
            return f"pixelrag_visual_search failed: {exc}"

        if not tiles:
            return "No results found."

        lines = [f"Top {len(tiles)} visual matches for: {query!r}"]
        for i, tile in enumerate(tiles, start=1):
            bits = [f"{i}. score={tile.score:.3f}"]
            if tile.source:
                bits.append(f"source={tile.source}")
            if tile.tile_id:
                bits.append(f"tile_id={tile.tile_id}")
            if tile.caption:
                bits.append(f'caption="{tile.caption[:200]}"')
            elif tile.image_url or tile.image_base64:
                bits.append("(image tile — no text caption; fetch the image to read it)")
            lines.append("  " + " | ".join(bits))

        return "\n".join(lines)
