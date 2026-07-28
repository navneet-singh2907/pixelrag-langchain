[![CI](https://github.com/navneet-singh2907/pixelrag-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/navneet-singh2907/pixelrag-langchain/actions/workflows/ci.yml)

# pixelrag-langchain

A LangChain `Tool` + `Retriever` for [PixelRAG](https://github.com/StarTrail-org/PixelRAG) — visual, screenshot-native search for AI agents.

**This is an unofficial, community-built integration.** It is not affiliated with, endorsed by, or maintained by StarTrail-org or the PixelRAG paper authors. All credit for the underlying research and the `pixelrag-serve` engine belongs to them — see [Credits](#credits).

## Why visual search

Most agent pipelines fetch a page, strip the HTML down to text, chunk it, and embed the chunks. That throws away tables, charts, and layout — information that's often exactly what answers the question. [PixelRAG](https://arxiv.org/abs/2606.28344) (Wang, Li, Wang, Teiletche, Jin, Zaharia, Gonzalez, Min — UC Berkeley, Princeton, EPFL, Databricks) skips the text step entirely: it renders documents to screenshot tiles, retrieves over the images directly, and hands the retrieved tiles to a vision-language model to read.

This package doesn't reimplement any of that. It just wraps the existing `pixelrag-serve` search API so it drops into a LangChain (or LangGraph) agent as a normal tool/retriever.

## Architecture

```mermaid
flowchart LR
    A["Page / PDF"] --> B["pixelrag-render\n(screenshot tiles)"]
    B --> C["pixelrag-embed\n(Qwen3-VL-Embedding)"]
    C --> D["FAISS index\n(pixelrag-serve)"]
    D -- "this package talks to D" --> E["PixelRAGRetriever /\nPixelRAGSearchTool"]
    E --> F["Your LangChain / LangGraph agent"]
    F --> G["VLM reader\n(your model of choice)"]
```

Everything left of the dotted line (`render` → `embed` → `index` → `serve`) is upstream PixelRAG, run by you. This package is the box on the right: a client + LangChain adapters.

## Install

```bash
pip install pixelrag-langchain
```

This installs only the LangChain adapter (`httpx`, `langchain-core`, `pydantic`) — no GPU, no `torch`, no CUDA. It talks to a `pixelrag-serve` instance over HTTP; it doesn't run embedding or indexing itself.

## Quickstart — zero setup

Upstream PixelRAG runs an official public demo endpoint — no API key, no index to download, searches their pre-built 8.28M-page Wikipedia index. Nothing to install beyond this package:

```python
from pixelrag_langchain import PixelRAGRetriever, PixelRAGConfig

retriever = PixelRAGRetriever(config=PixelRAGConfig.hosted())
docs = retriever.invoke("What is the capital of France?")

for doc in docs:
    print(doc.metadata["score"], doc.metadata["source"], doc.metadata.get("image_url"))
```

`PixelRAGConfig.hosted()` points at `https://api.pixelrag.ai`, upstream's own documented endpoint (see the "Live, hosted endpoint" section of [their README](https://github.com/StarTrail-org/PixelRAG#readme)). It's their infrastructure and their Wikipedia index, not ours or yours — great for a first try, not for anything you'd want uptime guarantees on or that touches your own documents.

## Quickstart — your own documents

For real use — your own PDFs, internal docs, scraped pages — run `pixelrag serve` yourself:

```bash
pip install 'pixelrag[serve]'

# Download a pre-built Wikipedia index, or build your own from `pip install 'pixelrag[index]'`
huggingface-cli download StarTrail-org/pixelrag-faiss-indexes \
  --repo-type dataset --include "search_index_normed_v2/*" --local-dir ./index

pixelrag serve --index-dir ./index/search_index_normed_v2 --port 30001
```

```python
from pixelrag_langchain import PixelRAGRetriever, PixelRAGConfig

retriever = PixelRAGRetriever(config=PixelRAGConfig(base_url="http://localhost:30001"))
docs = retriever.invoke("What is the capital of France?")
```

## As an agent tool

```python
from langchain.agents import create_agent
from pixelrag_langchain import PixelRAGSearchTool, PixelRAGConfig

tool = PixelRAGSearchTool(config=PixelRAGConfig.hosted())  # or base_url="http://localhost:30001"
agent = create_agent(model="your-model", tools=[tool])
```

Nothing here defaults to a single host silently — `PixelRAGConfig(base_url=...)` accepts any pixelrag-serve-compatible URL, including your own deployment; `.hosted()` is one explicit opt-in line, not a hidden default.

## What this package is *not*

- Not a reimplementation of PixelRAG's render/embed/index/serve pipeline — install the [upstream project](https://github.com/StarTrail-org/PixelRAG) for that.
- Not a hosted service itself. `PixelRAGConfig.hosted()` is a convenience pointer to *upstream's* public endpoint, not infrastructure we run.
- Not affiliated with StarTrail-org, Berkeley Sky Computing Lab, BAIR, or the Berkeley NLP Group, who built the actual PixelRAG engine and research this wraps.

## Benchmarks (from the upstream paper, not measured by this package)

Reported in [PIXELRAG: Web Screenshots Beat Text for Retrieval-Augmented Generation](https://arxiv.org/abs/2606.28344) (arXiv:2606.28344):

| Metric | Text-based RAG | PixelRAG | Context |
|---|---|---|---|
| Prompt tokens, agentic benchmark | 37.5M | 3.6M | MoNaCo multi-step agent benchmark |
| Accuracy gain | baseline | up to +18.1% | vs. text-based RAG baselines, across NQ/SimpleQA/MMSearch/LiveVQA |
| Token cost via image compression | — | up to 3x reduction | lower-resolution tiles, accuracy preserved |

These are the paper's own reported numbers, not something we independently reproduced — treat them as a starting point for your own evaluation on your workload, not a guarantee.

## Development

```bash
git clone https://github.com/YOUR_USERNAME/pixelrag-langchain
cd pixelrag-langchain
pip install -e ".[dev]"
pytest
```

## Credits

- **PixelRAG** engine, paper, and research: Yichuan Wang, Zhifei Li, Zirui Wang, Paul Teiletche, Lesheng Jin, Matei Zaharia, Joseph E. Gonzalez, Sewon Min. [Paper](https://arxiv.org/abs/2606.28344) · [Code](https://github.com/StarTrail-org/PixelRAG) (Apache-2.0).
- This package: an independent, unofficial LangChain adapter around that work.

## License

Apache-2.0 — see [LICENSE](LICENSE). This is a new work built to talk to PixelRAG over HTTP; it does not vendor or redistribute any upstream PixelRAG code, so no NOTICE carryover is required, but attribution is given above regardless because it's their research this is built on.
