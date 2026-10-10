"""Run the Streamlit app headless with Streamlit's AppTest and a scripted model."""
import contextlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from ds_team import config
from tests.conftest import ScriptedModel, tool_call

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
ENV_VARS = (
    "LLM_PROVIDER", "LLM_MODEL", "LLM_REASONING_EFFORT", "GROQ_API_KEY", "OPENAI_API_KEY",
    "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "ALLOW_CODE_EXECUTION", "APP_PASSWORD",
)


@pytest.fixture
def app(project, monkeypatch):
    """A fresh app with no keys configured, an in-memory database and a scripted model.

    Returns `start(*responses)`, which runs the app and returns (AppTest, recorder).
    `recorder.models` lists the get_model() arguments, `recorder.tracing` the
    tracing_context() arguments, and `recorder.model` is the scripted model."""
    import langsmith
    import streamlit as st
    from langgraph.checkpoint.memory import InMemorySaver

    import ds_team.graph

    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(ds_team.graph, "sqlite_checkpointer", lambda *a, **k: InMemorySaver())
    st.cache_resource.clear()

    class Recorder:
        models, tracing, model = [], [], None

    def fake_get_model(**kwargs):
        Recorder.models.append(kwargs)
        return Recorder.model

    def fake_tracing_context(**kwargs):
        Recorder.tracing.append(kwargs)
        return contextlib.nullcontext()

    monkeypatch.setattr(config, "get_model", fake_get_model)
    monkeypatch.setattr(langsmith, "tracing_context", fake_tracing_context)

    def start(*responses):
        Recorder.model = ScriptedModel(responses=list(responses))
        return AppTest.from_file(APP, default_timeout=60).run(), Recorder

    yield start
    st.cache_resource.clear()


def _markdown(at) -> list[str]:
    return [m.value for m in at.markdown]


def _chat(at, text: str):
    at.chat_input[0].set_value(text).run()
    assert not at.exception
    return at


def test_asks_for_a_key_and_offers_groq_and_openai(app):
    at, _ = app()

    assert at.selectbox(key="llm_provider").options == ["Groq", "OpenAI"]
    assert at.selectbox(key="llm_model_groq").value == "openai/gpt-oss-120b"
    assert at.text_input(key="api_key_groq").label == "Your Groq API key"
    assert any("Groq API key" in info.value for info in at.info)
    assert not at.chat_input  # no chat until a key is given

    at.selectbox(key="llm_provider").set_value("openai").run()

    assert at.selectbox(key="llm_model_openai").value == "gpt-5-mini"
    assert at.text_input(key="api_key_openai").label == "Your OpenAI API key"
    assert any("OpenAI API key" in info.value for info in at.info)


def test_chat_with_the_visitors_groq_key(app):
    at, rec = app(
        AIMessage("", tool_calls=[tool_call("delegate_to_data_understanding", task="Shape of data/sample.csv")]),
        AIMessage("", tool_calls=[tool_call("dataframe_operation", path="data/sample.csv", operation="shape")]),
        AIMessage("20 rows and 8 columns."),
        AIMessage("Your dataset has **20 rows** and 8 columns."),
    )
    at.text_input(key="api_key_groq").input("gsk-visitor").run()

    _chat(at, "How big is my dataset?")

    assert "Your dataset has **20 rows** and 8 columns." in _markdown(at)
    assert any("data_understanding" in m and "dataframe_operation" in m for m in _markdown(at))
    assert rec.models[-1] == {
        "api_key": "gsk-visitor", "provider_name": "groq",
        "model": "openai/gpt-oss-120b", "reasoning_effort": "low",
    }


def test_openai_model_without_reasoning_effort(app):
    at, rec = app(AIMessage("Hello from OpenAI!"))
    at.selectbox(key="llm_provider").set_value("openai").run()
    at.selectbox(key="llm_model_openai").set_value("gpt-4.1-mini").run()
    assert not [s for s in at.selectbox if s.key == "llm_effort_openai"]  # gpt-4.1 has no reasoning effort
    at.text_input(key="api_key_openai").input("sk-visitor").run()

    _chat(at, "hi")

    assert "Hello from OpenAI!" in _markdown(at)
    assert rec.models[-1] == {
        "api_key": "sk-visitor", "provider_name": "openai", "model": "gpt-4.1-mini", "reasoning_effort": None,
    }


def test_custom_model_id_and_reasoning_effort(app):
    at, rec = app(AIMessage("ok"))
    at.selectbox(key="llm_model_groq").set_value("Custom model id...").run()
    at.text_input(key="llm_custom_model_groq").input("openai/gpt-oss-20b").run()
    at.selectbox(key="llm_effort_groq").set_value("high").run()
    at.text_input(key="api_key_groq").input("gsk-visitor").run()

    _chat(at, "hi")

    assert rec.models[-1]["model"] == "openai/gpt-oss-20b"
    assert rec.models[-1]["reasoning_effort"] == "high"


def test_uses_the_apps_key_from_secrets(app, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-owner")
    at, rec = app(AIMessage("ok"))

    assert at.text_input(key="api_key_groq").label == "Your Groq API key (optional)"
    _chat(at, "hi")
    assert rec.models[-1]["api_key"] is None  # falls back to the app's key


def test_safe_mode_is_the_default(app, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-owner")
    at, rec = app(AIMessage("Hello!"))
    _chat(at, "hi")

    system_prompt = rec.model.calls[0][0].content
    assert "Code execution is disabled" in system_prompt
    assert "delegate_to_shell" not in system_prompt and "- shell:" not in system_prompt
    assert "**Code execution:** off (safe mode)" in _markdown(at)


def test_approval_form_when_code_execution_is_enabled(app, monkeypatch, project):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-owner")
    monkeypatch.setenv("ALLOW_CODE_EXECUTION", "true")
    at, _ = app(
        AIMessage("", tool_calls=[tool_call("delegate_to_coding", task="Write hello.py")]),
        AIMessage("", tool_calls=[tool_call("write_file", path="hello.py", content="print('hi')")]),
        AIMessage("Saved hello.py."),
        AIMessage("Done, hello.py is saved."),
    )
    _chat(at, "Write a hello world script")

    assert at.code[0].value == "print('hi')"  # the code is shown for review
    assert at.chat_input[0].disabled
    assert not (project / "hello.py").exists()

    next(b for b in at.button if b.label == "Submit").click().run()  # default decision: Approve

    assert not at.exception
    assert (project / "hello.py").read_text() == "print('hi')"
    assert "Done, hello.py is saved." in _markdown(at)


def test_tracing_is_off_by_default_without_a_langsmith_key(app, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-owner")
    at, rec = app(AIMessage("ok"))
    assert at.toggle(key="tracing_enabled").value is False

    _chat(at, "hi")
    assert rec.tracing[-1] == {"enabled": False}


def test_tracing_needs_a_key(app, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-owner")
    at, rec = app(AIMessage("ok"))
    at.toggle(key="tracing_enabled").set_value(True).run()

    assert any("LangSmith API key" in w.value for w in at.warning)
    _chat(at, "hi")
    assert rec.tracing[-1] == {"enabled": False}


def test_tracing_with_the_visitors_langsmith_key(app, monkeypatch):
    from langsmith import Client

    monkeypatch.setenv("GROQ_API_KEY", "gsk-owner")
    at, rec = app(AIMessage("ok"))
    at.toggle(key="tracing_enabled").set_value(True).run()
    at.text_input(key="langsmith_api_key").input("lsv2-visitor").run()
    at.text_input(key="langsmith_project").input("my-project").run()

    _chat(at, "hi")

    settings = rec.tracing[-1]
    assert settings["enabled"] is True and settings["project_name"] == "my-project"
    assert isinstance(settings["client"], Client)


def test_tracing_with_the_apps_langsmith_key(app, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-owner")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-owner")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    at, rec = app(AIMessage("ok"))
    assert at.toggle(key="tracing_enabled").value is True  # on by default when the app has a key

    _chat(at, "hi")

    assert rec.tracing[-1] == {"enabled": True, "client": None, "project_name": "Multi_agent_datascience_team"}
