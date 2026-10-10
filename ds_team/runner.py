"""Run the team one turn at a time and stream what happens.

Shared by the terminal chat, the REST API and the Streamlit app. `stream_turn` yields
plain, JSON-serialisable events:

    {"type": "token", "text": ...}                      Team Leader reply, token by token
    {"type": "delegation", "agent": ..., "task": ...}   Team Leader handed a task to an agent
    {"type": "tool", "agent": ..., "tool": ...}         an agent called one of its tools
    {"type": "agent_done", "agent": ...}                an agent finished its task
    {"type": "approval", "approvals": [...]}            paused until the user approves (last event)
    {"type": "done", "reply": ...}                      the turn finished (last event)
"""
from collections.abc import Iterator

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from ds_team import config


def run_config(thread_id: str) -> dict:
    """Graph config: the thread id gives the team its memory. The run name, tags and
    metadata label the traces in LangSmith, grouped by conversation in the Threads view."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100,
        "run_name": "data_science_team",
        "tags": ["multi-agent-data-science-team"],
        "metadata": {"thread_id": thread_id, "provider": config.provider(), "model": config.model_name()},
    }


def pending_approvals(graph: CompiledStateGraph, thread_id: str) -> list[dict]:
    """Actions waiting for the user's approval in this thread."""
    return [{"id": i.id, **i.value} for i in graph.get_state(run_config(thread_id)).interrupts]


def stream_turn(graph: CompiledStateGraph, thread_id: str, graph_input) -> Iterator[dict]:
    """Run the graph until it finishes or pauses for approval, yielding events as they happen."""
    run = graph.stream(graph_input, run_config(thread_id), stream_mode=["messages", "updates"], subgraphs=True)
    for namespace, mode, data in run:
        # namespace is () for the top-level graph and ("<agent>:<task id>", ...) inside an agent
        agent = namespace[0].split(":")[0] if namespace else None

        if mode == "messages":
            message, metadata = data
            if agent is None and metadata.get("langgraph_node") == "team_leader" and isinstance(message, AIMessage) and message.text:
                yield {"type": "token", "text": message.text}
            continue

        for node, update in data.items():
            if node == "__interrupt__" or not isinstance(update, dict):
                continue
            for message in update.get("messages", []):
                for call in getattr(message, "tool_calls", None) or []:
                    if agent is None and call["name"].startswith("delegate_to_"):
                        yield {
                            "type": "delegation",
                            "agent": call["name"].removeprefix("delegate_to_"),
                            "task": call["args"].get("task", ""),
                        }
                    elif agent is not None:
                        yield {"type": "tool", "agent": agent, "tool": call["name"]}
            if agent is None and node not in ("team_leader", "remember"):
                yield {"type": "agent_done", "agent": node}

    approvals = pending_approvals(graph, thread_id)
    if approvals:
        yield {"type": "approval", "approvals": approvals}
    else:
        messages = graph.get_state(run_config(thread_id)).values["messages"]
        yield {"type": "done", "reply": messages[-1].text}


def run_turn(graph: CompiledStateGraph, thread_id: str, graph_input) -> dict:
    """Run a turn without streaming and return its last event (`done` or `approval`)."""
    event = {}
    for event in stream_turn(graph, thread_id, graph_input):
        pass
    return event
