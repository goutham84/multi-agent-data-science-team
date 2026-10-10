"""Streamlit web UI for the Data Science Team.

    streamlit run streamlit_app.py

Settings come from .env locally, or from the app's Secrets on Streamlit Community Cloud.
"""
import os
import re
import uuid
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

from ds_team import config  # noqa: E402
from ds_team.graph import build_graph, sqlite_checkpointer  # noqa: E402
from ds_team.runner import run_config, stream_turn  # noqa: E402

UPLOAD_DIR = config.DATA_DIR / "uploads"
GITHUB_URL = "https://github.com/goutham84/multi-agent-data-science-team"
EXAMPLE_PROMPTS = [
    "What datasets do I have? Preview car_details.csv.",
    "Explore data/car_details.csv: shape, column types, missing values and duplicates.",
    "Plot a histogram of selling_price and a bar chart of fuel types.",
    "Which features matter most for selling_price? Look at correlations.",
]


# ============================ Setup =================================

@st.cache_resource
def get_checkpointer():
    return sqlite_checkpointer()


@st.cache_resource(max_entries=20)
def get_graph(api_key: str | None):
    """One compiled graph per API key, all sharing the same SQLite memory."""
    return build_graph(model=config.get_model(api_key), checkpointer=get_checkpointer())


def _init_session() -> None:
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("history", [])     # [{"role", "content", "activity", "images"}]
    st.session_state.setdefault("pending_approvals", [])   # actions waiting for the user's answer
    st.session_state.setdefault("api_key", "")


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

def _sidebar() -> str | None:
    """Render the sidebar and return the API key to use (None = from the environment)."""
    with st.sidebar:
        st.header("🤖 Data Science Team")
        st.caption("LangChain + LangGraph multi-agent system")

        key_name = config.api_key_name()
        api_key = None
        if not os.getenv(key_name):
            st.session_state.api_key = st.text_input(
                key_name, value=st.session_state.api_key, type="password",
                help="Used only for this session and never stored.",
            )
            api_key = st.session_state.api_key or None

        st.markdown(f"**Model:** `{config.MODEL}`")
        if config.tracing_enabled():
            st.markdown(f"**LangSmith tracing:** on (project `{config.LANGSMITH_PROJECT}`)")
        else:
            st.markdown("**LangSmith tracing:** off")
        st.markdown(f"**Code execution:** {'on (with approval)' if config.ALLOW_CODE_EXECUTION else 'off (safe mode)'}")

        st.divider()
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
        if st.button("➕ New conversation", width="stretch"):
            for key in ("thread_id", "history", "pending_approvals"):
                st.session_state.pop(key, None)
            st.rerun()
        st.caption(f"Thread `{st.session_state.thread_id}`")
        st.markdown(f"[Source code on GitHub]({GITHUB_URL})")
    return api_key


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


def _run_turn(graph, graph_input) -> None:
    """Stream one turn into the chat and save it to the history."""
    plots_before = _plot_mtimes()
    text, activity, approvals = "", [], []

    with st.chat_message("assistant"):
        status = st.status("Team Leader is thinking...")
        placeholder = st.empty()
        try:
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
        except Exception as e:  # e.g. invalid API key, rate limits
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


def _approval_form(graph) -> None:
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
        _run_turn(graph, Command(resume=answers))
        st.rerun()


def main() -> None:
    _init_session()
    if not _check_password():
        return
    api_key = _sidebar()

    st.title("🤖 Multi-Agent Data Science Team")
    st.caption(
        "A Team Leader agent delegates to six specialists: data loader, file manager, "
        "data understanding, visualization, coding and shell."
    )

    if not (api_key or os.getenv(config.api_key_name())):
        st.info(f"Add your `{config.api_key_name()}` in the sidebar to start.")
        return

    graph = get_graph(api_key)
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
        _approval_form(graph)
        st.chat_input("Answer the approval request above first", disabled=True)
        return

    prompt = st.chat_input("Ask the team...") or prompt
    if prompt:
        message = {"role": "user", "content": prompt}
        st.session_state.history.append(message)
        _render_message(message)
        _run_turn(graph, {"messages": [{"role": "user", "content": prompt}]})
        st.rerun()


main()
