"""Thin HTTP client for a pixelrag-serve `/search` endpoint.

Request shape follows the upstream README's documented example:

    POST /search
    {"queries": [{"text": "What is the capital of France?"}], "n_docs": 5}

The authoritative schema is published by each server at `/openapi.json`.
PixelRAG is still pre-1.0, so parsing remains deliberately defensive: additive
fields are retained in `raw`, and malformed result containers degrade to an
empty result rather than crashing an agent mid-run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

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
    """One retrieved screenshot tile, parsed from the API response.

    Field mapping verified against a live api.pixelrag.ai response (July
    2026), where each hit looks like:

        {"score": 0.588, "vector_id": 18869670, "article_id": 5663927,
         "tile_index": 0, "chunk_index": 7, "y_offset": 7168,
         "tile_height": 1024,
         "path": "shard_683/.../5663927.png.tiles/chunk_0000_07.png",
         "url": "Outline_of_France",        # NB: article TITLE, not a URL
         "article_pages": "0:0-7,...", "image_base64": null}

    Quirks preserved here on purpose:
      - The server's "url" field holds the article title -> mapped to
        `source`, NOT to `image_url`.
      - "path" is a relative tile path within the server's tile store, not a
        directly fetchable URL -> exposed as `tile_path`.
      - Image bytes come back only when the server includes "image_base64".
    Anything unmapped stays available in `raw`.
    """

    score: float
    tile_id: str | None = None
    source: str | None = None          # article title (server's "url") or doc/article id
    article_id: str | None = None
    tile_index: int | None = None
    chunk_index: int | None = None
    tile_path: str | None = None       # server-relative and informational only
    image_url: str | None = None       # only if server returns a real fetchable URL
    image_base64: str | None = None    # inline image bytes, when provided
    caption: str | None = None         # accompanying text/OCR, when provided
    raw: dict[str, Any] = field(default_factory=dict)  # untouched original record

    @property
    def has_tile_coordinates(self) -> bool:
        """Whether this hit can be addressed through the `/tile` endpoint."""
        return (
            self.article_id is not None
            and self.tile_index is not None
            and self.chunk_index is not None
        )

    @classmethod
    def from_api_record(cls, record: dict[str, Any]) -> "PixelRAGTile":
        score = record.get("score", record.get("similarity", 0.0))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        article_id = record.get("article_id") or record.get("doc_id")
        tile_id = record.get("tile_id") or record.get("vector_id") or record.get("id")

        def as_int(value: Any) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        # "url" is the article title in the live schema. Only treat it as a
        # fetchable image URL if it actually looks like one.
        url_field = record.get("url")
        looks_like_real_url = isinstance(url_field, str) and url_field.startswith(
            ("http://", "https://")
        )

        return cls(
            score=score,
            tile_id=str(tile_id) if tile_id is not None else None,
            source=record.get("source")
            or (url_field if not looks_like_real_url else None)
            or (str(article_id) if article_id is not None else None),
            article_id=str(article_id) if article_id is not None else None,
            tile_index=as_int(record.get("tile_index")),
            chunk_index=as_int(record.get("chunk_index")),
            tile_path=record.get("path"),
            image_url=record.get("image_url")
            or (url_field if looks_like_real_url else None),
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

    def search(
        self,
        query: str,
        n_docs: int | None = None,
        *,
        include_images: bool = False,
    ) -> list[PixelRAGTile]:
        """Run one text query against the index and return ranked tiles.

        Set `include_images=True` to request inline base64 images. The default
        keeps search responses small; use `tile_url()` or `fetch_tile()` to
        retrieve only the images needed downstream.

        Raises:
            PixelRAGConnectionError: server unreachable (wrong URL, not running).
            PixelRAGTimeoutError: request exceeded `config.timeout_s`.
            PixelRAGAPIError: server responded with an error status.
        """
        if not query or not query.strip():
            return []

        requested_n_docs = self.config.n_docs if n_docs is None else n_docs
        if requested_n_docs < 1:
            raise ValueError("n_docs must be at least 1")

        payload: dict[str, Any] = {
            "queries": [{"text": query}],
            "n_docs": requested_n_docs,
        }
        if include_images:
            payload["include_images"] = True

        response = self._request("POST", self.config.search_url, json=payload)

        try:
            data = response.json()
        except ValueError as exc:
            raise PixelRAGAPIError(response.status_code, response.text) from exc

        return self._parse_results(data)

    def tile_url(self, tile: PixelRAGTile) -> str | None:
        """Return the server URL for a tile, or `None` without coordinates."""
        if not tile.has_tile_coordinates:
            return None
        return (
            f"{self.config.base_url.rstrip('/')}/tile/"
            f"{quote(tile.article_id, safe='')}/{tile.tile_index}/{tile.chunk_index}"
        )

    def fetch_tile(self, tile: PixelRAGTile) -> bytes:
        """Download image bytes for one retrieved tile."""
        url = self.tile_url(tile)
        if url is None:
            raise ValueError(
                "Tile is missing article_id, tile_index, or chunk_index; "
                "cannot construct its image URL."
            )
        return self._request("GET", url).content

    def status(self) -> dict[str, Any]:
        """Return server/index diagnostics from the public `/status` endpoint."""
        response = self._request("GET", self.config.status_url)
        try:
            data = response.json()
        except ValueError as exc:
            raise PixelRAGAPIError(response.status_code, response.text) from exc
        if not isinstance(data, dict):
            raise PixelRAGAPIError(
                response.status_code, "Expected a JSON object from /status"
            )
        return data

    def health(self) -> bool:
        """Best-effort check that the server is up. Never raises."""
        try:
            self._request("GET", self.config.health_url)
            return True
        except PixelRAGError:
            return False

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send one request and normalize all transport/API failures."""
        try:
            response = self._http.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise PixelRAGTimeoutError(
                f"Request to {url} timed out after {self.config.timeout_s}s."
            ) from exc
        except httpx.RequestError as exc:
            raise PixelRAGConnectionError(
                f"Could not reach pixelrag-serve at {self.config.base_url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise PixelRAGAPIError(response.status_code, response.text)
        return response

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PixelRAGClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _parse_results(data: Any) -> list[PixelRAGTile]:
        """Handle a couple of plausible top-level shapes without raising.

        Known/expected shapes we defend against:
          {"results": [[{...}, {...}]]}          # per-query list of lists
          {"results": [{...}, {...}]}             # flat list (single query)
          {"data": {"results": [...]}}             # nested wrapper
        Anything else logs a warning and returns an empty list rather than
        raising, so a single malformed response doesn't take down an agent.
        """
        if not isinstance(data, dict):
            logger.warning(
                "Unrecognized pixelrag-serve response type: %s", type(data).__name__
            )
            return []

        results = data.get("results")
        if results is None and isinstance(data.get("data"), dict):
            results = data["data"].get("results")

        if results is None:
            logger.warning(
                "Unrecognized pixelrag-serve response shape (no 'results' key); "
                "keys were: %s", list(data.keys()),
            )
            return []

        if not isinstance(results, list):
            logger.warning("pixelrag-serve 'results' field was not a list")
            return []

        # Real observed schema (api.pixelrag.ai, July 2026): results is a list
        # of per-query objects, each wrapping its records in a "hits" list:
        #   {"results": [{"hits": [{...}, {...}]}]}
        if results and isinstance(results[0], dict) and "hits" in results[0]:
            results = results[0].get("hits") or []
        # Older/alternate shape: list-of-lists grouped by query.
        elif results and isinstance(results[0], list):
            results = results[0]

        if not isinstance(results, list):
            logger.warning("pixelrag-serve 'hits' field was not a list")
            return []

        tiles: list[PixelRAGTile] = []
        for record in results:
            if isinstance(record, dict):
                tiles.append(PixelRAGTile.from_api_record(record))
        return tiles
