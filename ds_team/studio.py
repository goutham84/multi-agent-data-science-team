"""Entry point for LangGraph Studio (`langgraph dev`). The dev server provides its own checkpointer."""
from ds_team.graph import build_graph

graph = build_graph()
