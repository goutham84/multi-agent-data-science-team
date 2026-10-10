"""End-to-end tests of the LangGraph team with a scripted model (no API calls)."""
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ds_team.graph import _recent_turns, build_graph
from tests.conftest import ScriptedModel, tool_call

CONFIG = {"configurable": {"thread_id": "test"}}


def _tool_results_seen_by_model(model: ScriptedModel) -> list[str]:
    return [m.content for messages in model.calls for m in messages if isinstance(m, ToolMessage)]


def test_team_leader_delegates_and_remembers(project):
    model = ScriptedModel(responses=[
        # Team Leader delegates
        AIMessage("", tool_calls=[tool_call("delegate_to_data_understanding", task="Shape of data/sample.csv?")]),
        # Data Understanding agent uses its pandas tool, then answers
        AIMessage("", tool_calls=[tool_call("dataframe_operation", path="data/sample.csv", operation="shape")]),
        AIMessage("The dataset has 20 rows and 8 columns."),
        # Team Leader saves a note, then replies
        AIMessage("", tool_calls=[tool_call("remember", key="dataset", value="data/sample.csv")]),
        AIMessage("Your dataset has 20 rows and 8 columns."),
    ])
    graph = build_graph(model, checkpointer=InMemorySaver())

    result = graph.invoke({"messages": [("user", "How big is my dataset?")]}, CONFIG)

    assert result["messages"][-1].text == "Your dataset has 20 rows and 8 columns."
    assert result["notes"] == {"dataset": "data/sample.csv"}
    seen = _tool_results_seen_by_model(model)
    assert "20 rows x 8 columns" in seen  # the agent's real tool output
    assert "The dataset has 20 rows and 8 columns." in seen  # the agent's answer reached the leader
    assert model.responses == []


def test_parallel_delegation(project):
    model = ScriptedModel(responses=[
        AIMessage("", tool_calls=[
            tool_call("delegate_to_data_loader", task="List the CSV files"),
            tool_call("delegate_to_file_manager", task="List the files"),
        ]),
        AIMessage("loader done"),
        AIMessage("files done"),
        AIMessage("Both finished."),
    ])
    graph = build_graph(model, checkpointer=InMemorySaver())

    result = graph.invoke({"messages": [("user", "What files do I have?")]}, CONFIG)

    answers = sorted(m.content for m in result["messages"] if isinstance(m, ToolMessage))
    assert answers == ["files done", "loader done"]
    assert result["messages"][-1].text == "Both finished."


@pytest.mark.parametrize("answer, saved", [("yes", True), ("no, use pathlib instead", False)])
def test_writing_a_file_waits_for_user_approval(project, answer, saved):
    model = ScriptedModel(responses=[
        AIMessage("", tool_calls=[tool_call("delegate_to_coding", task="Write hello.py")]),
        AIMessage("", tool_calls=[tool_call("write_file", path="hello.py", content="print('hi')")]),
        AIMessage("Coding agent finished."),
        AIMessage("Team Leader finished."),
    ])
    graph = build_graph(model, checkpointer=InMemorySaver())

    graph.invoke({"messages": [("user", "Write a hello world script")]}, CONFIG)

    # the graph is paused inside the coding agent's write_file tool
    (pending,) = graph.get_state(CONFIG).interrupts
    assert pending.value == {"action": "write_file", "path": "hello.py", "content": "print('hi')"}
    assert not (project / "hello.py").exists()

    result = graph.invoke(Command(resume={pending.id: answer}), CONFIG)

    assert (project / "hello.py").exists() == saved
    expected = "Saved hello.py" if saved else f"Their feedback: {answer}"
    assert any(expected in r for r in _tool_results_seen_by_model(model))
    assert result["messages"][-1].text == "Team Leader finished."
    assert model.responses == []  # resuming did not repeat earlier model calls


def test_recent_turns_starts_at_a_user_message():
    messages = []
    for i in range(7):
        messages += [("human", f"q{i}"), ("ai", f"a{i}")]
    from langchain_core.messages import convert_to_messages

    kept = _recent_turns(convert_to_messages(messages), turns=5)
    assert [m.content for m in kept][:2] == ["q2", "a2"]
    assert len(kept) == 10
