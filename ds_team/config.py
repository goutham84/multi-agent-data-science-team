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


# ============================ LLM ===================================
# Choose the LLM in .env:
#   LLM_PROVIDER = groq | openai
#   LLM_MODEL    = model id on that provider (optional, defaults below)
# Groq model ids: https://console.groq.com/docs/models
PROVIDERS = {
    "groq": {"api_key": "GROQ_API_KEY", "default_model": "openai/gpt-oss-120b"},
    "openai": {"api_key": "OPENAI_API_KEY", "default_model": "gpt-5-mini"},
}

# Lower temperature makes open models write well-formed tool calls more reliably.
# Nothing is sent for OpenAI: gpt-5 models only accept their default temperature.
DEFAULT_TEMPERATURE = {"groq": 0.3}

# Retries with backoff for rate limits (429). Groq's per-minute token limits are low and a
# multi-agent turn makes many model calls, so retry generously; the SDK honours retry-after.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))

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


def provider() -> str:
    """The configured LLM provider: "groq" or "openai"."""
    name = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {name!r}")
    return name


def model_name() -> str:
    """The model id on the configured provider, e.g. "openai/gpt-oss-120b" on Groq."""
    return os.getenv("LLM_MODEL", "").strip() or PROVIDERS[provider()]["default_model"]


def model_label() -> str:
    """Short description for logs and the UI, e.g. "groq / openai/gpt-oss-120b"."""
    return f"{provider()} / {model_name()}"


def api_key_name() -> str:
    """Environment variable holding the key for the configured provider, e.g. GROQ_API_KEY."""
    return PROVIDERS[provider()]["api_key"]


def get_model(api_key: str | None = None) -> BaseChatModel:
    """Create the chat model configured in .env (LLM_PROVIDER, LLM_MODEL and optional
    LLM_TEMPERATURE / LLM_REASONING_EFFORT). `api_key` overrides the key from the
    environment (used by the Streamlit app)."""
    kwargs = {"model_provider": provider(), "max_retries": MAX_RETRIES}
    if api_key:
        kwargs["api_key"] = api_key

    temperature = os.getenv("LLM_TEMPERATURE", "").strip() or DEFAULT_TEMPERATURE.get(provider())
    if temperature is not None:
        kwargs["temperature"] = float(temperature)

    # low / medium / high, for reasoning models (gpt-oss on Groq, gpt-5 on OpenAI).
    # "low" uses far fewer tokens, which helps with Groq's per-minute token limits.
    if effort := os.getenv("LLM_REASONING_EFFORT", "").strip().lower():
        kwargs["reasoning_effort"] = effort

    return init_chat_model(model_name(), **kwargs)


def is_retryable_model_error(error: Exception) -> bool:
    """Errors worth retrying the model call for:
    - Groq `tool_use_failed`: the model produced a malformed tool call (a retry usually fixes it)
    - rate limits, timeouts and server errors still failing after the SDK's own retries
    """
    if type(error).__name__ == "BadRequestError":
        return "tool_use_failed" in str(error)
    return type(error).__name__ in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}
