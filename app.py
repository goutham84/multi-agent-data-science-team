"""Run the Data Science Team in the terminal or as a REST API.

    python app.py                    # chat in the terminal (live token streaming)
    python app.py --thread <id>      # continue an earlier terminal conversation
    python app.py --serve            # REST API on http://127.0.0.1:7777 (docs at /docs)

For the web UI, run `streamlit run streamlit_app.py`.
"""
import argparse
import json
import uuid
from collections.abc import Iterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from ds_team import config
from ds_team.graph import build_graph, sqlite_checkpointer
from ds_team.runner import pending_approvals, run_turn, stream_turn

# the team, with conversation memory stored in memory.db
graph = build_graph(checkpointer=sqlite_checkpointer())


def _user_message(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def _result(thread_id: str, event: dict) -> dict:
    """Turn the last event of a turn into the API response."""
    if event["type"] == "approval":
        return {"thread_id": thread_id, "status": "awaiting_approval", "approvals": event["approvals"]}
    return {"thread_id": thread_id, "status": "done", "reply": event["reply"]}


# ============================ REST API ==============================

app = FastAPI(
    title="Data Science Team API",
    description="Multi-agent data science team built with LangChain and LangGraph.",
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str
    answers: dict[str, str]  # approval id -> "yes", "no" or feedback for the agent


def _chat_input(request: ChatRequest) -> tuple[str, dict]:
    thread_id = request.thread_id or str(uuid.uuid4())
    if pending_approvals(graph, thread_id):
        raise HTTPException(409, "This thread is waiting for approval. Answer it with POST /approve first.")
    return thread_id, _user_message(request.message)


def _approval_input(request: ApprovalRequest) -> Command:
    pending = pending_approvals(graph, request.thread_id)
    if not pending:
        raise HTTPException(409, "Nothing is waiting for approval in this thread.")
    missing = [a["id"] for a in pending if a["id"] not in request.answers]
    if missing:
        raise HTTPException(400, f"Missing answers for approvals: {missing}")
    return Command(resume=request.answers)


def _sse(thread_id: str, graph_input) -> StreamingResponse:
    """Stream the turn's events as Server-Sent Events."""
    def events() -> Iterator[str]:
        yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id})}\n\n"
        for event in stream_turn(graph, thread_id, graph_input):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": config.MODEL,
        "code_execution": config.ALLOW_CODE_EXECUTION,
        "langsmith_tracing": config.tracing_enabled(),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Send a message. Returns the reply, or the actions waiting for approval."""
    thread_id, graph_input = _chat_input(request)
    return _result(thread_id, run_turn(graph, thread_id, graph_input))


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Like /chat, but streams tokens and agent activity as Server-Sent Events."""
    thread_id, graph_input = _chat_input(request)
    return _sse(thread_id, graph_input)


@app.post("/approve")
def approve(request: ApprovalRequest) -> dict:
    """Approve or reject the pending actions of a thread."""
    return _result(request.thread_id, run_turn(graph, request.thread_id, _approval_input(request)))


@app.post("/approve/stream")
def approve_stream(request: ApprovalRequest) -> StreamingResponse:
    """Like /approve, but streams the rest of the turn as Server-Sent Events."""
    return _sse(request.thread_id, _approval_input(request))


@app.get("/threads/{thread_id}")
def thread_history(thread_id: str) -> dict:
    """Conversation history and session notes of a thread."""
    values = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    return {
        "thread_id": thread_id,
        "notes": values.get("notes", {}),
        "messages": [{"role": m.type, "content": m.text} for m in values.get("messages", [])],
    }


# ============================ Terminal chat =========================

def _show_approval(approval: dict) -> None:
    print(f"\n[approval needed] {approval['action']}")
    if approval["action"] == "write_file":
        print(f"File: {approval['path']}\n{'-' * 60}\n{approval['content']}\n{'-' * 60}")
    elif approval["action"] == "install_package":
        print(f"Package: {approval['package']}")


def _print_turn(thread_id: str, graph_input) -> dict:
    """Stream one turn to the terminal and return its last event."""
    print("\nTeam Leader: ", end="", flush=True)
    for event in stream_turn(graph, thread_id, graph_input):
        match event["type"]:
            case "token":
                print(event["text"], end="", flush=True)
            case "delegation":
                print(f"\n  -> {event['agent']}: {event['task'][:100]}", flush=True)
            case "tool":
                print(f"     [{event['agent']}] {event['tool']}", flush=True)
            case "agent_done":
                print(f"  <- {event['agent']} done\nTeam Leader: ", end="", flush=True)
    print()
    return event


def chat_in_terminal(thread_id: str | None = None) -> None:
    thread_id = thread_id or str(uuid.uuid4())
    tracing = f"LangSmith project '{config.LANGSMITH_PROJECT}'" if config.tracing_enabled() else "off"
    print(f"Data Science Team | model {config.MODEL} | tracing {tracing}")
    print(f"Thread {thread_id}. Type 'exit' to quit.")

    while True:
        message = input("\nYou: ").strip()
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue

        event = _print_turn(thread_id, _user_message(message))
        while event["type"] == "approval":
            answers = {}
            for approval in event["approvals"]:
                _show_approval(approval)
                answers[approval["id"]] = input("Approve? (yes / no / feedback): ").strip() or "no"
            event = _print_turn(thread_id, Command(resume=answers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the Data Science Team.")
    parser.add_argument("--serve", action="store_true", help="start the REST API instead of the terminal chat")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--thread", help="thread id of an earlier conversation to continue")
    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    else:
        chat_in_terminal(args.thread)
