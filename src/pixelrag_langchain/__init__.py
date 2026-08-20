"""LangChain integration for PixelRAG (screenshot-native visual search).

Unofficial community package. Not affiliated with or endorsed by
StarTrail-org, the maintainers of the upstream PixelRAG project
(https://github.com/StarTrail-org/PixelRAG).
"""

from pixelrag_langchain.client import (
    PixelRAGAPIError,
    PixelRAGClient,
    PixelRAGConnectionError,
    PixelRAGError,
    PixelRAGTile,
    PixelRAGTimeoutError,
)
from pixelrag_langchain.config import PixelRAGConfig
from pixelrag_langchain.retriever import PixelRAGRetriever
from pixelrag_langchain.tool import PixelRAGSearchTool

__all__ = [
    "PixelRAGConfig",
    "PixelRAGClient",
    "PixelRAGTile",
    "PixelRAGError",
    "PixelRAGConnectionError",
    "PixelRAGTimeoutError",
    "PixelRAGAPIError",
    "PixelRAGRetriever",
    "PixelRAGSearchTool",
]

__version__ = "0.2.0"
