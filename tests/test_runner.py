from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ds_team.graph import _patch_dangling_tool_calls, build_graph
from ds_team.runner import run_config, stream_turn
from tests.conftest import ScriptedModel, tool_call


def test_stream_turn_events(project):
    model = ScriptedModel(responses=[
        AIMessage("Let me check.", tool_calls=[tool_call("delegate_to_coding", task="Write hello.py")]),
        AIMessage("", tool_calls=[tool_call("write_file", path="hello.py", content="print(1)")]),
        AIMessage("Saved."),
        AIMessage("All done."),
    ])
    graph = build_graph(model, InMemorySaver())

    events = list(stream_turn(graph, "t1", {"messages": [("user", "write a script")]}))

    assert "".join(e["text"] for e in events if e["type"] == "token") == "Let me check."
    assert {"type": "delegation", "agent": "coding", "task": "Write hello.py"} in events
    assert {"type": "tool", "agent": "coding", "tool": "write_file"} in events
    assert events[-1]["type"] == "approval"
    approval_id = events[-1]["approvals"][0]["id"]

    events = list(stream_turn(graph, "t1", Command(resume={approval_id: "yes"})))

    assert {"type": "agent_done", "agent": "coding"} in events
    assert [e["text"] for e in events if e["type"] == "token"] == ["All", " done."]  # streamed word by word
    assert events[-1] == {"type": "done", "reply": "All done."}


def test_run_config_labels_langsmith_traces():
    config = run_config("abc")
    assert config["configurable"]["thread_id"] == "abc"
    assert config["metadata"]["thread_id"] == "abc"  # groups traces in LangSmith's Threads view
    assert config["run_name"] == "data_science_team"


def test_dangling_tool_calls_get_a_placeholder_result():
    call = tool_call("delegate_to_shell", task="x")
    messages = [HumanMessage("hi"), AIMessage("", tool_calls=[call]), HumanMessage("hello again")]

    patched = _patch_dangling_tool_calls(messages)

    assert isinstance(patched[2], ToolMessage) and patched[2].tool_call_id == call["id"]
    assert patched[3].content == "hello again"
