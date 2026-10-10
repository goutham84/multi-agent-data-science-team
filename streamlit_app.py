"""Streamlit web UI for the Data Science Team.

    streamlit run streamlit_app.py

Visitors choose the LLM provider (Groq or OpenAI), the model and LangSmith tracing in the
sidebar and paste their own API keys. Keys set in .env locally, or in the app's Secrets on
Streamlit Community Cloud, are used as defaults so visitors can skip pasting a key.
"""
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Data Science Team", page_icon="🤖", layout="wide")


def _load_secrets_into_env() -> None:
    """Copy Streamlit secrets into the environment before ds_team reads its settings."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:
        pass  # no secrets.toml: settings come from .env


_load_secrets_into_env()

from langgraph.types import Command  # noqa: E402
from langsmith import Client, tracing_context  # noqa: E402

from ds_team import config  # noqa: E402
from ds_team.graph import build_graph, sqlite_checkpointer  # noqa: E402
from ds_team.runner import run_config, stream_turn  # noqa: E402

UPLOAD_DIR = config.DATA_DIR / "uploads"
GITHUB_URL = "https://github.com/goutham84/multi-agent-data-science-team"
LANGSMITH_URL = "https://smith.langchain.com"
CUSTOM_MODEL = "Custom model id..."
EXAMPLE_PROMPTS = [
    "What datasets do I have? Preview car_details.csv.",
    "Explore data/car_details.csv: shape, column types, missing values and duplicates.",
    "Plot a histogram of selling_price and a bar chart of fuel types.",
    "Which features matter most for selling_price? Look at correlations.",
]

# A public app must not run visitors' code on the server, so unlike the CLI the
# web app keeps code execution off unless ALLOW_CODE_EXECUTION is set explicitly.
ALLOW_CODE_EXECUTION = config.env_flag("ALLOW_CODE_EXECUTION", False)


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    reasoning_effort: str | None
    api_key: str | None  # None = the app's key from secrets / .env

    @property
    def ready(self) -> bool:
        return bool(self.model and (self.api_key or os.getenv(config.api_key_name(self.provider))))


@dataclass(frozen=True)
class TracingSettings:
    enabled: bool
    api_key: str | None  # None = the app's LANGSMITH_API_KEY
    project: str


# ============================ Setup =================================

@st.cache_resource
def get_checkpointer():
    return sqlite_checkpointer()


@st.cache_resource(max_entries=20)
def get_graph(settings: ModelSettings):
    """One compiled graph per model choice and key, all sharing the same SQLite memory."""
    model = config.get_model(
        api_key=settings.api_key,
        provider_name=settings.provider,
        model=settings.model,
        reasoning_effort=settings.reasoning_effort,
    )
    return build_graph(model=model, checkpointer=get_checkpointer(), allow_code_execution=ALLOW_CODE_EXECUTION)


@st.cache_resource(max_entries=20)
def get_langsmith_client(api_key: str) -> Client:
    return Client(api_key=api_key, api_url=os.getenv("LANGSMITH_ENDPOINT") or None)


def _tracing(settings: TracingSettings):
    """LangSmith settings for this session only (other visitors are unaffected)."""
    if not settings.enabled:
        return tracing_context(enabled=False)
    client = get_langsmith_client(settings.api_key) if settings.api_key else None
    return tracing_context(enabled=True, client=client, project_name=settings.project)


def _init_session() -> None:
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("history", [])            # [{"role", "content", "activity", "images"}]
    st.session_state.setdefault("pending_approvals", [])  # actions waiting for the user's answer


def _check_password() -> bool:
    """Optional access gate: set APP_PASSWORD to keep strangers from spending your API credits."""
    password = os.getenv("APP_PASSWORD")
    if not password or st.session_state.get("authenticated"):
        return True
    st.title("🤖 Data Science Team")
    entered = st.text_input("Password", type="password")
    if entered == password:
        st.session_state.authenticated = True
        st.rerun()
    elif entered:
        st.error("Wrong password")
    return False


def _plot_mtimes() -> dict[Path, float]:
    return {p: p.stat().st_mtime for p in config.PLOTS_DIR.glob("*.png")} if config.PLOTS_DIR.exists() else {}


# ============================ Sidebar ===============================

def _default_provider() -> str:
    try:
        return config.provider()
    except ValueError:
        return "groq"


def _model_settings() -> ModelSettings:
    """Provider, model, reasoning effort and API key, chosen by the visitor."""
    st.subheader("🧠 Model")
    providers = list(config.PROVIDERS)
    provider = st.selectbox(
        "Provider", providers, index=providers.index(_default_provider()),
        format_func=lambda p: config.PROVIDERS[p]["label"], key="llm_provider",
    )
    info = config.PROVIDERS[provider]

    default_model = config.model_name(provider)
    options = list(dict.fromkeys([default_model, *info["models"]])) + [CUSTOM_MODEL]
    choice = st.selectbox("Model", options, index=options.index(default_model), key=f"llm_model_{provider}")
    model = choice
    if choice == CUSTOM_MODEL:
        model = st.text_input("Model id", key=f"llm_custom_model_{provider}", placeholder=info["default_model"]).strip()

    effort = None
    if model and config.supports_reasoning_effort(provider, model):
        env_effort = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
        default_effort = env_effort if env_effort in config.REASONING_EFFORTS else "low"
        effort = st.selectbox(
            "Reasoning effort", config.REASONING_EFFORTS,
            index=config.REASONING_EFFORTS.index(default_effort), key=f"llm_effort_{provider}",
            help="`low` is much faster and uses far fewer tokens (helps with Groq's rate limits).",
        )

    has_app_key = bool(os.getenv(info["api_key"]))
    label = f"Your {info['label']} API key" + (" (optional)" if has_app_key else "")
    api_key = st.text_input(
        label, type="password", key=f"api_key_{provider}",
        help="Kept in memory for this session only, never saved."
        + (" Leave empty to use the app's key." if has_app_key else ""),
    ).strip()
    if not has_app_key and not api_key:
        st.caption(f"Get a key at [{info['key_url'].split('//')[1]}]({info['key_url']})")
    return ModelSettings(provider, model, effort, api_key or None)


def _tracing_settings() -> TracingSettings:
    """LangSmith tracing on/off, key and project, chosen by the visitor."""
    st.subheader("🔍 LangSmith tracing")
    has_app_key = bool(os.getenv("LANGSMITH_API_KEY"))
    enabled = st.toggle("Trace runs in LangSmith", value=config.tracing_enabled(), key="tracing_enabled")
    project = os.getenv("LANGSMITH_PROJECT") or "Multi_agent_datascience_team"
    if not enabled:
        return TracingSettings(False, None, project)

    api_key = st.text_input(
        "Your LangSmith API key" + (" (optional)" if has_app_key else ""), type="password",
        key="langsmith_api_key", help="Kept in memory for this session only, never saved."
        + (" Leave empty to send traces to the app's LangSmith project." if has_app_key else ""),
    ).strip()
    project = st.text_input("Project", value=project, key="langsmith_project").strip() or project
    if not (api_key or has_app_key):
        st.warning("Add a LangSmith API key to trace runs.")
        return TracingSettings(False, None, project)
    st.caption(f"Traces appear in [LangSmith]({LANGSMITH_URL}) under project `{project}`.")
    return TracingSettings(True, api_key or None, project)


def _sidebar() -> tuple[ModelSettings, TracingSettings]:
    with st.sidebar:
        st.header("🤖 Data Science Team")
        st.caption("LangChain + LangGraph multi-agent system")
        model_settings = _model_settings()
        st.divider()
        tracing_settings = _tracing_settings()

        st.divider()
        st.subheader("📂 Data")
        uploaded = st.file_uploader("Upload a CSV dataset", type="csv")
        if uploaded is not None:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(uploaded.name).name)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            (UPLOAD_DIR / name).write_bytes(uploaded.getvalue())
            st.success(f"Saved as `data/uploads/{name}`. Ask the team about it!")
        with st.expander("Datasets"):
            for path in sorted(config.DATA_DIR.rglob("*.csv")):
                st.markdown(f"- `{path.relative_to(config.BASE_DIR).as_posix()}`")

        st.divider()
        st.markdown(f"**Code execution:** {'on (with approval)' if ALLOW_CODE_EXECUTION else 'off (safe mode)'}")
        if st.button("➕ New conversation", width="stretch"):
            for key in ("thread_id", "history", "pending_approvals"):
                st.session_state.pop(key, None)
            st.rerun()
        st.caption(f"Thread `{st.session_state.thread_id}`")
        st.markdown(f"[Source code on GitHub]({GITHUB_URL})")
    return model_settings, tracing_settings


def _show_notes(graph) -> None:
    notes = graph.get_state(run_config(st.session_state.thread_id)).values.get("notes")
    if notes:
        with st.sidebar.expander("📝 Session notes", expanded=True):
            for key, value in notes.items():
                st.markdown(f"- **{key}:** {value}")


# ============================ Chat ==================================

def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("activity"):
            with st.expander(f"Team activity ({len(message['activity'])} steps)"):
                st.markdown("\n".join(f"- {line}" for line in message["activity"]))
        st.markdown(message["content"])
        for image in message.get("images", []):
            if Path(image).exists():
                st.image(image, caption=Path(image).name)


def _run_turn(graph, graph_input, tracing: TracingSettings) -> None:
    """Stream one turn into the chat and save it to the history."""
    plots_before = _plot_mtimes()
    text, activity, approvals = "", [], []

    with st.chat_message("assistant"):
        status = st.status("Team Leader is thinking...")
        placeholder = st.empty()
        try:
            with _tracing(tracing):
                for event in stream_turn(graph, st.session_state.thread_id, graph_input):
                    match event["type"]:
                        case "token":
                            text += event["text"]
                            placeholder.markdown(text + "▌")
                        case "delegation":
                            if text.strip():
                                activity.append(f"💬 {text.strip()}")
                            text = ""
                            placeholder.empty()
                            line = f"🧭 **{event['agent']}**: {event['task']}"
                            activity.append(line)
                            status.markdown(line)
                            status.update(label=f"{event['agent']} is working...")
                        case "tool":
                            line = f"🔧 {event['agent']} used `{event['tool']}`"
                            activity.append(line)
                            status.markdown(line)
                        case "agent_done":
                            activity.append(f"✅ {event['agent']} finished")
                            status.update(label="Team Leader is reviewing the results...")
                        case "approval":
                            approvals = event["approvals"]
                        case "done":
                            text = event["reply"]
        except Exception as e:  # e.g. invalid API key, unknown model, rate limits
            status.update(label="Something went wrong", state="error")
            st.session_state.history.append(
                {"role": "assistant", "content": f"⚠️ **{type(e).__name__}:** {e}", "activity": activity}
            )
            return

        if approvals:
            status.update(label="Waiting for your approval", state="complete")
            text = text or "An agent needs your approval to continue (see below)."
        else:
            status.update(label=f"Done ({len(activity)} steps)", state="complete")
        placeholder.markdown(text)

    images = [str(p) for p, mtime in _plot_mtimes().items() if plots_before.get(p) != mtime]
    st.session_state.history.append({"role": "assistant", "content": text, "activity": activity, "images": images})
    st.session_state.pending_approvals = approvals


def _approval_form(graph, tracing: TracingSettings) -> None:
    """Let the user approve or reject what the agents want to do."""
    with st.form("approvals"):
        st.warning("The team wants to make changes. Review and approve or reject.")
        answers = {}
        for approval in st.session_state.pending_approvals:
            if approval["action"] == "write_file":
                st.markdown(f"**Write file** `{approval['path']}`")
                language = "python" if approval["path"].endswith(".py") else None
                st.code(approval["content"], language=language)
            elif approval["action"] == "install_package":
                st.markdown(f"**Install package** `{approval['package']}`")
            decision = st.radio("Decision", ["Approve", "Reject"], key=f"decision_{approval['id']}", horizontal=True)
            feedback = st.text_input("Feedback if rejecting (optional)", key=f"feedback_{approval['id']}")
            answers[approval["id"]] = "yes" if decision == "Approve" else (feedback or "no")
        submitted = st.form_submit_button("Submit")

    if submitted:
        summary = ", ".join("approved" if a == "yes" else f"rejected ({a})" for a in answers.values())
        st.session_state.history.append({"role": "user", "content": f"*Decision: {summary}*"})
        st.session_state.pending_approvals = []
        _run_turn(graph, Command(resume=answers), tracing)
        st.rerun()


def main() -> None:
    _init_session()
    if not _check_password():
        return
    model_settings, tracing_settings = _sidebar()

    st.title("🤖 Multi-Agent Data Science Team")
    st.caption(
        "A Team Leader agent delegates to specialist agents: data loader, file manager, "
        "data understanding, visualization and coding."
    )

    if not model_settings.ready:
        label = config.PROVIDERS[model_settings.provider]["label"]
        st.info(f"Choose a provider and model in the sidebar and add your **{label} API key** to start.")
        return

    graph = get_graph(model_settings)
    _show_notes(graph)

    for message in st.session_state.history:
        _render_message(message)

    prompt = None
    if not st.session_state.history:
        st.markdown("**Try one of these:**")
        columns = st.columns(len(EXAMPLE_PROMPTS))
        for column, example in zip(columns, EXAMPLE_PROMPTS):
            if column.button(example, width="stretch"):
                prompt = example

    if st.session_state.pending_approvals:
        _approval_form(graph, tracing_settings)
        st.chat_input("Answer the approval request above first", disabled=True)
        return

    prompt = st.chat_input("Ask the team...") or prompt
    if prompt:
        message = {"role": "user", "content": prompt}
        st.session_state.history.append(message)
        _render_message(message)
        _run_turn(graph, {"messages": [{"role": "user", "content": prompt}]}, tracing_settings)
        st.rerun()


main()
