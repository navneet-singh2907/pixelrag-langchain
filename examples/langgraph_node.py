"""Example: use PixelRAGSearchTool inside a LangGraph node as the retrieval
step, then hand the actual screenshot tiles (not just a text summary) to a
vision-capable model as the "reader" — this is the point of PixelRAG: the
model reads pixels directly instead of a text description of them.

Not a full runnable app — the multimodal content-block format in the
`HumanMessage` below follows the common OpenAI-style convention that most
LangChain chat model integrations accept, but check your specific provider's
docs if it doesn't work; this varies slightly across providers/versions.

Requires: pip install langgraph (not a dependency of this package itself,
since the core client/tool/retriever have no reason to need it).
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

from pixelrag_langchain import PixelRAGConfig, PixelRAGSearchTool, PixelRAGTile

search_tool = PixelRAGSearchTool(config=PixelRAGConfig(base_url="http://localhost:30001"))


class AgentState(TypedDict):
    query: str
    tiles: list[PixelRAGTile]
    messages: list[BaseMessage]


def retrieve_node(state: AgentState) -> AgentState:
    tiles = search_tool.search_tiles(state["query"], n_docs=3)
    return {**state, "tiles": tiles}


def build_reader_message(query: str, tiles: list[PixelRAGTile]) -> HumanMessage:
    content: list[dict] = [{"type": "text", "text": f"Answer using these screenshots: {query}"}]
    for tile in tiles:
        if tile.image_url:
            content.append({"type": "image_url", "image_url": {"url": tile.image_url}})
        elif tile.image_base64:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tile.image_base64}"}}
            )
    return HumanMessage(content=content)


def reader_node(state: AgentState) -> AgentState:
    message = build_reader_message(state["query"], state["tiles"])
    return {**state, "messages": state.get("messages", []) + [message]}


# Wiring into an actual StateGraph:
#
#   from langgraph.graph import StateGraph, END
#   graph = StateGraph(AgentState)
#   graph.add_node("retrieve", retrieve_node)
#   graph.add_node("build_reader_message", reader_node)
#   graph.set_entry_point("retrieve")
#   graph.add_edge("retrieve", "build_reader_message")
#   graph.add_edge("build_reader_message", END)
#   app = graph.compile()
#   result = app.invoke({"query": "What is the capital of France?"})
#   # result["messages"][-1] is ready to send to your VLM
