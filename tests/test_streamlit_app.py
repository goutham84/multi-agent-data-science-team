"""Run the Streamlit app headless with Streamlit's AppTest and a scripted model."""
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from ds_team import config
from tests.conftest import ScriptedModel, tool_call

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def app_env(project, monkeypatch):
    """Point the app at the temp project, an in-memory database and a fake API key."""
    import streamlit as st
    from langgraph.checkpoint.memory import InMemorySaver

    import ds_team.graph

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setattr(ds_team.graph, "sqlite_checkpointer", lambda *a, **k: InMemorySaver())
    st.cache_resource.clear()

    def use_model(*responses):
        model = ScriptedModel(responses=list(responses))
        monkeypatch.setattr(config, "get_model", lambda api_key=None: model)
        return model

    yield use_model
    st.cache_resource.clear()


def test_chat_turn_shows_reply_and_activity(app_env):
    app_env(
        AIMessage("", tool_calls=[tool_call("delegate_to_data_understanding", task="Shape of data/sample.csv")]),
        AIMessage("", tool_calls=[tool_call("dataframe_operation", path="data/sample.csv", operation="shape")]),
        AIMessage("20 rows and 8 columns."),
        AIMessage("Your dataset has **20 rows** and 8 columns."),
    )
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    at.chat_input[0].set_value("How big is my dataset?").run()

    assert not at.exception
    markdown = [m.value for m in at.markdown]
    assert "Your dataset has **20 rows** and 8 columns." in markdown
    assert any("data_understanding" in m and "dataframe_operation" in m for m in markdown)  # activity log


def test_approval_form_saves_file_after_approve(app_env, project):
    app_env(
        AIMessage("", tool_calls=[tool_call("delegate_to_coding", task="Write hello.py")]),
        AIMessage("", tool_calls=[tool_call("write_file", path="hello.py", content="print('hi')")]),
        AIMessage("Saved hello.py."),
        AIMessage("Done, hello.py is saved."),
    )
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("Write a hello world script").run()

    assert at.code[0].value == "print('hi')"  # the code is shown for review
    assert at.chat_input[0].disabled
    assert not (project / "hello.py").exists()

    submit = next(b for b in at.button if b.label == "Submit")
    submit.click().run()  # default decision: Approve

    assert not at.exception
    assert (project / "hello.py").read_text() == "print('hi')"
    assert "Done, hello.py is saved." in [m.value for m in at.markdown]


def test_safe_mode_hides_execution_agents(app_env, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_CODE_EXECUTION", False)
    model = app_env(AIMessage("Hello!"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("hi").run()

    system_prompt = model.calls[0][0].content
    assert "delegate_to_shell" not in system_prompt and "- shell:" not in system_prompt
    assert "Code execution is disabled" in system_prompt
    assert any("off (safe mode)" in m.value for m in at.markdown)
