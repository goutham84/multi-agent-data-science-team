"""Groq compatibility: the real ChatGroq class with a fake HTTP client (no API key needed).

The fake records every request that would be sent to Groq and answers in Groq's
OpenAI-compatible format, so these tests check the message/tool payloads, streaming,
and the retry of Groq's `tool_use_failed` error.
"""
import json
import time

import groq
import httpx
import pytest
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ds_team import config
from ds_team.graph import build_graph
from ds_team.runner import stream_turn


class FakeGroqCompletions:
    """Stands in for `groq.Groq().chat.completions`."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, messages, stream=False, **params):
        self.requests.append({"messages": messages, **params})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply.get("content", "")
        tool_calls = [
            {"id": f"call_{len(self.requests)}_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(reply.get("tool_calls", []))
        ]
        finish = "tool_calls" if tool_calls else "stop"
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        if not stream:
            message = {"role": "assistant", "content": content or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {"id": "chatcmpl", "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}

        def chunk(delta, finish_reason=None):
            return {"id": "chatcmpl", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

        words = content.split(" ") if content else []
        chunks = [chunk({"role": "assistant", "content": w if i == 0 else f" {w}"}) for i, w in enumerate(words)]
        if tool_calls:
            chunks.append(chunk({"tool_calls": [{"index": i, **call} for i, call in enumerate(tool_calls)]}))
        chunks.append({**chunk({}, finish), "x_groq": {"usage": usage}})
        return iter(chunks)


def tool_use_failed() -> groq.BadRequestError:
    """The error Groq returns when the model writes a malformed tool call."""
    body = {"error": {"message": "Failed to call a function. Please adjust your prompt.",
                      "type": "invalid_request_error", "code": "tool_use_failed"}}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.BadRequestError(f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body)


@pytest.fixture
def groq_env(monkeypatch):
    monkeypatch.setenv("MODEL", "groq:openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(time, "sleep", lambda seconds: None)  # skip retry backoff


def groq_model(*replies) -> tuple[ChatGroq, FakeGroqCompletions]:
    model = config.get_model()
    fake = FakeGroqCompletions(replies)
    model.client = fake
    return model, fake


def assert_valid_groq_conversation(messages: list[dict]) -> None:
    """Every assistant tool call must be answered by a tool message with the same id."""
    for i, message in enumerate(messages):
        assert message["role"] in {"system", "user", "assistant", "tool"}
        for call in message.get("tool_calls") or []:
            answers = [m for m in messages[i + 1:] if m["role"] == "tool"]
            assert call["id"] in {m["tool_call_id"] for m in answers}


def test_get_model_builds_chatgroq(groq_env):
    model = config.get_model()
    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"
    assert model.temperature == config.GROQ_TEMPERATURE
    assert model.max_retries == config.MAX_RETRIES
    assert config.api_key_name() == "GROQ_API_KEY"


def test_other_groq_model(groq_env, monkeypatch):
    monkeypatch.setenv("MODEL", "groq:qwen/qwen3.8-27b")
    assert config.get_model().model_name == "qwen/qwen3.8-27b"


def test_full_turn_with_streaming_and_approval(groq_env, project):
    model, fake = groq_model(
        {"content": "I'll ask the coding agent.", "tool_calls": [("delegate_to_coding", {"task": "Write hello.py"})]},
        {"tool_calls": [("write_file", {"path": "hello.py", "content": "print('hi')"})]},
        {"content": "Saved hello.py."},
        {"content": "Done, hello.py is saved."},
    )
    graph = build_graph(model, InMemorySaver())

    events = list(stream_turn(graph, "t", {"messages": [("user", "Write a hello world script")]}))
    assert events[-1]["type"] == "approval"
    events = list(stream_turn(graph, "t", Command(resume={events[-1]["approvals"][0]["id"]: "yes"})))

    assert events[-1] == {"type": "done", "reply": "Done, hello.py is saved."}
    assert (project / "hello.py").read_text() == "print('hi')"

    leader_tools = {t["function"]["name"] for t in fake.requests[0]["tools"]}
    assert {"delegate_to_coding", "delegate_to_data_loader", "remember"} <= leader_tools
    coding_tools = {t["function"]["name"] for t in fake.requests[1]["tools"]}
    assert {"write_file", "run_python_file"} <= coding_tools
    for request in fake.requests:
        assert request["model"] == "openai/gpt-oss-120b"
        assert_valid_groq_conversation(request["messages"])


def test_team_leader_retries_tool_use_failed(groq_env):
    model, fake = groq_model(tool_use_failed(), {"content": "Hello! How can I help?"})
    graph = build_graph(model, InMemorySaver())

    result = graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "t"}})

    assert result["messages"][-1].text == "Hello! How can I help?"
    assert len(fake.requests) == 2


def test_agent_retries_tool_use_failed(groq_env, project):
    model, fake = groq_model(
        {"tool_calls": [("delegate_to_data_understanding", {"task": "Shape of data/sample.csv"})]},
        tool_use_failed(),  # the agent's first attempt fails
        {"tool_calls": [("dataframe_operation", {"path": "data/sample.csv", "operation": "shape"})]},
        {"content": "20 rows x 8 columns."},
        {"content": "Your dataset has 20 rows."},
    )
    graph = build_graph(model, InMemorySaver())

    result = graph.invoke({"messages": [("user", "How big is my data?")]}, {"configurable": {"thread_id": "t"}})

    assert result["messages"][-1].text == "Your dataset has 20 rows."
    tool_results = [m["content"] for m in fake.requests[-2]["messages"] if m["role"] == "tool"]
    assert "20 rows x 8 columns" in tool_results  # the pandas tool really ran after the retry


def test_other_bad_requests_are_not_retried(groq_env):
    request = httpx.Request("POST", "https://api.groq.com")
    error = groq.BadRequestError("Error code: 400 - context_length_exceeded",
                                 response=httpx.Response(400, request=request), body=None)
    model, fake = groq_model(error)
    graph = build_graph(model, InMemorySaver())

    with pytest.raises(groq.BadRequestError):
        graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "t"}})
    assert len(fake.requests) == 1
