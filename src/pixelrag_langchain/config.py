"""Connection configuration for a pixelrag-serve-compatible search API.

The constructor default points at a local `pixelrag serve` instance
(`pip install 'pixelrag[serve]'`) rather than any third-party host — a
library shouldn't silently route your queries somewhere you didn't choose.

Upstream PixelRAG (github.com/StarTrail-org/PixelRAG) does also run an
officially documented public endpoint for a no-setup demo against their
8.28M-page Wikipedia index, no API key required, with its own status page
(status.pixelrag.ai). Use `PixelRAGConfig.hosted()` to point at it — that's
one explicit line, not something this package assumes for you.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BASE_URL = "http://localhost:30001"
# Documented in upstream's README as a first-class, intentional feature (not
# an internal/staging URL) — see the "Live, hosted endpoint" callout at
# https://github.com/StarTrail-org/PixelRAG#readme. It only serves their
# pre-built Wikipedia index, not your own documents.
PIXELRAG_HOSTED_BASE_URL = "https://api.pixelrag.ai"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_N_DOCS = 5


@dataclass(frozen=True)
class PixelRAGConfig:
    """Settings for talking to a pixelrag-serve search API.

    Attributes:
        base_url: Root URL of a running pixelrag-serve instance. Defaults to
            a local instance on the default port from the upstream README.
            This is intentionally NOT defaulted to any third-party hosted
            endpoint — point it there explicitly if you have one.
        timeout_s: Per-request timeout in seconds.
        n_docs: Default number of results to request per query.
        api_key: Optional bearer token, for deployments that sit behind auth
            (the local pixelrag-serve quickstart does not require one).
    """

    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = DEFAULT_TIMEOUT_S
    n_docs: int = DEFAULT_N_DOCS
    api_key: str | None = None

    @classmethod
    def hosted(cls, **overrides: object) -> "PixelRAGConfig":
        """Point at upstream's official public demo endpoint (api.pixelrag.ai).

        No API key needed. Searches their pre-built 8.28M-page Wikipedia
        index only — not your own documents. Good for trying this package
        out in under a minute; switch to a self-hosted index for real use.
        """
        return cls(base_url=PIXELRAG_HOSTED_BASE_URL, **overrides)  # type: ignore[arg-type]

    @property
    def search_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/search"

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/health"
