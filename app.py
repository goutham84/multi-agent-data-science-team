"""Run the Data Science Team in the terminal or as a REST API.

    python app.py                    # chat in the terminal
    python app.py --thread <id>      # continue an earlier terminal conversation
    python app.py --serve            # REST API on http://127.0.0.1:7777 (docs at /docs)
"""
import argparse
import uuid
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from ds_team.graph import build_graph, sqlite_checkpointer

# the team, with conversation memory stored in memory.db
graph = build_graph(checkpointer=sqlite_checkpointer())


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}


def run_turn(thread_id: str, graph_input, on_step: Callable[[str], None] | None = None) -> dict:
    """Run the graph until it finishes or pauses for the user's approval."""
    config = _config(thread_id)
    for update in graph.stream(graph_input, config, stream_mode="updates"):
        if on_step:
            for node in update:
                if node != "__interrupt__":
                    on_step(node)

    state = graph.get_state(config)
    if state.interrupts:
        return {
            "thread_id": thread_id,
            "status": "awaiting_approval",
            "approvals": [{"id": i.id, **i.value} for i in state.interrupts],
        }
    return {"thread_id": thread_id, "status": "done", "reply": state.values["messages"][-1].text}


def _user_message(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Send a message. Returns the reply, or the actions waiting for approval."""
    thread_id = request.thread_id or str(uuid.uuid4())
    if graph.get_state(_config(thread_id)).interrupts:
        raise HTTPException(409, "This thread is waiting for approval. Answer it with POST /approve first.")
    return run_turn(thread_id, _user_message(request.message))


@app.post("/approve")
def approve(request: ApprovalRequest) -> dict:
    """Approve or reject the pending actions of a thread."""
    pending = graph.get_state(_config(request.thread_id)).interrupts
    if not pending:
        raise HTTPException(409, "Nothing is waiting for approval in this thread.")
    missing = [i.id for i in pending if i.id not in request.answers]
    if missing:
        raise HTTPException(400, f"Missing answers for approvals: {missing}")
    return run_turn(request.thread_id, Command(resume=request.answers))


@app.get("/threads/{thread_id}")
def thread_history(thread_id: str) -> dict:
    """Conversation history and session notes of a thread."""
    values = graph.get_state(_config(thread_id)).values
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


def chat_in_terminal(thread_id: str | None = None) -> None:
    thread_id = thread_id or str(uuid.uuid4())
    print(f"Data Science Team (thread {thread_id}). Type 'exit' to quit.")
    on_step = lambda node: print(f"  -> {node} done")

    while True:
        message = input("\nYou: ").strip()
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue

        result = run_turn(thread_id, _user_message(message), on_step)
        while result["status"] == "awaiting_approval":
            answers = {}
            for approval in result["approvals"]:
                _show_approval(approval)
                answers[approval["id"]] = input("Approve? (yes / no / feedback): ").strip() or "no"
            result = run_turn(thread_id, Command(resume=answers), on_step)

        print(f"\nTeam Leader: {result['reply']}")


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
