"""Thin HTTP client for a pixelrag-serve `/search` endpoint.

Request shape follows the upstream README's documented example:

    POST /search
    {"queries": [{"text": "What is the capital of France?"}], "n_docs": 5}

The exact response schema isn't published as a formal spec anywhere we could
find (the project is young and pre-1.0), so parsing here is deliberately
defensive: it accepts a few plausible field-name variants per result and
falls back to keeping the raw dict around rather than raising, so a schema
tweak upstream degrades gracefully instead of crashing your agent mid-run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from pixelrag_langchain.config import PixelRAGConfig

logger = logging.getLogger(__name__)


class PixelRAGError(Exception):
    """Base class for all errors raised by this client."""


class PixelRAGConnectionError(PixelRAGError):
    """Could not reach the pixelrag-serve instance at all."""


class PixelRAGTimeoutError(PixelRAGError):
    """Request exceeded the configured timeout."""


class PixelRAGAPIError(PixelRAGError):
    """Server reachable but returned a non-2xx response."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"pixelrag-serve returned HTTP {status_code}: {body[:500]}")


@dataclass
class PixelRAGTile:
    """One retrieved screenshot tile, defensively parsed from the API response."""

    score: float
    tile_id: str | None = None
    source: str | None = None          # e.g. originating page/article/doc id
    image_url: str | None = None       # if server returns a fetchable URL
    image_base64: str | None = None    # if server returns inline image bytes
    caption: str | None = None         # any accompanying text/OCR the server provides
    raw: dict[str, Any] = field(default_factory=dict)  # untouched original record

    @classmethod
    def from_api_record(cls, record: dict[str, Any]) -> "PixelRAGTile":
        score = record.get("score", record.get("similarity", 0.0))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        return cls(
            score=score,
            tile_id=record.get("tile_id") or record.get("id"),
            source=record.get("source")
            or record.get("article_id")
            or record.get("doc_id"),
            image_url=record.get("image_url") or record.get("url"),
            image_base64=record.get("image_base64") or record.get("image"),
            caption=record.get("caption") or record.get("text"),
            raw=record,
        )


class PixelRAGClient:
    """Synchronous client for a running pixelrag-serve instance.

    Example:
        >>> client = PixelRAGClient(PixelRAGConfig(base_url="http://localhost:30001"))
        >>> tiles = client.search("capital of France", n_docs=3)
    """

    def __init__(self, config: PixelRAGConfig | None = None):
        self.config = config or PixelRAGConfig()
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self._http = httpx.Client(timeout=self.config.timeout_s, headers=headers)

    def search(self, query: str, n_docs: int | None = None) -> list[PixelRAGTile]:
        """Run one text query against the index and return ranked tiles.

        Raises:
            PixelRAGConnectionError: server unreachable (wrong URL, not running).
            PixelRAGTimeoutError: request exceeded `config.timeout_s`.
            PixelRAGAPIError: server responded with an error status.
        """
        if not query or not query.strip():
            return []

        payload = {
            "queries": [{"text": query}],
            "n_docs": n_docs or self.config.n_docs,
        }

        try:
            response = self._http.post(self.config.search_url, json=payload)
        except httpx.TimeoutException as exc:
            raise PixelRAGTimeoutError(
                f"pixelrag-serve at {self.config.base_url} timed out after "
                f"{self.config.timeout_s}s. Is it running and warmed up?"
            ) from exc
        except httpx.ConnectError as exc:
            raise PixelRAGConnectionError(
                f"Could not connect to pixelrag-serve at {self.config.base_url}. "
                "Is `pixelrag-serve --index-dir ./index --port 30001` running? "
                "See https://github.com/StarTrail-org/PixelRAG for setup."
            ) from exc

        if response.status_code >= 400:
            raise PixelRAGAPIError(response.status_code, response.text)

        try:
            data = response.json()
        except ValueError as exc:
            raise PixelRAGAPIError(response.status_code, response.text) from exc

        return self._parse_results(data)

    def health(self) -> bool:
        """Best-effort check that the server is up. Never raises."""
        try:
            r = self._http.get(self.config.health_url)
            return r.status_code < 400
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PixelRAGClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _parse_results(data: dict[str, Any]) -> list[PixelRAGTile]:
        """Handle a couple of plausible top-level shapes without raising.

        Known/expected shapes we defend against:
          {"results": [[{...}, {...}]]}          # per-query list of lists
          {"results": [{...}, {...}]}             # flat list (single query)
          {"data": {"results": [...]}}             # nested wrapper
        Anything else logs a warning and returns an empty list rather than
        raising, so a single malformed response doesn't take down an agent.
        """
        results = data.get("results")
        if results is None and isinstance(data.get("data"), dict):
            results = data["data"].get("results")

        if results is None:
            logger.warning(
                "Unrecognized pixelrag-serve response shape (no 'results' key); "
                "keys were: %s", list(data.keys()),
            )
            return []

        # Flatten one level if the server grouped by query.
        if results and isinstance(results[0], list):
            results = results[0]

        tiles: list[PixelRAGTile] = []
        for record in results:
            if isinstance(record, dict):
                tiles.append(PixelRAGTile.from_api_record(record))
        return tiles
