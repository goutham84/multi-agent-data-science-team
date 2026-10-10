"""Project paths and LLM configuration shared by every agent."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# load the api keys
load_dotenv()

# path for the base/project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

# SQLite file that stores conversation checkpoints (the team's memory)
DB_PATH = BASE_DIR / "memory.db"

# default dataset the team works on
DEFAULT_DATASET = DATA_DIR / "car_details.csv"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


# "provider:model", e.g. "openai:gpt-5-mini" or "groq:llama-3.3-70b-versatile"
MODEL = os.getenv("MODEL", "openai:gpt-5-mini")

# Let agents write files, run Python and install packages. Turn this off for public
# deployments: anyone who can open the app could otherwise approve code on your server.
ALLOW_CODE_EXECUTION = _env_flag("ALLOW_CODE_EXECUTION", True)

# LangSmith: with LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, LangChain and
# LangGraph trace every run automatically (no code changes needed).
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")


def tracing_enabled() -> bool:
    return _env_flag("LANGSMITH_TRACING", False) and bool(os.getenv("LANGSMITH_API_KEY"))


# tracing switched on without a key would log an error on every run, so switch it off
if _env_flag("LANGSMITH_TRACING", False) and not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"


def api_key_name() -> str:
    """Environment variable holding the key for the configured provider, e.g. OPENAI_API_KEY."""
    provider = os.getenv("MODEL", MODEL).split(":", 1)[0]
    return f"{provider.upper()}_API_KEY"


def get_model(api_key: str | None = None) -> BaseChatModel:
    """Create the chat model configured in .env (MODEL=provider:model).
    `api_key` overrides the key from the environment (used by the Streamlit app)."""
    kwargs = {"api_key": api_key} if api_key else {}
    return init_chat_model(os.getenv("MODEL", MODEL), **kwargs)
