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


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


# ============================ LLM ===================================
# Choose the LLM in .env (or per session in the Streamlit sidebar):
#   LLM_PROVIDER = groq | openai
#   LLM_MODEL    = model id on that provider (optional, defaults below)
# Groq model ids: https://console.groq.com/docs/models
PROVIDERS = {
    "groq": {
        "label": "Groq",
        "api_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "default_model": "openai/gpt-oss-120b",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    },
    "openai": {
        "label": "OpenAI",
        "api_key": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "default_model": "gpt-5-mini",
        "models": ["gpt-5-mini", "gpt-5", "gpt-5-nano", "gpt-4.1-mini", "gpt-4.1"],
    },
}
REASONING_EFFORTS = ["low", "medium", "high"]

# Lower temperature makes open models write well-formed tool calls more reliably.
# Nothing is sent for OpenAI: gpt-5 models only accept their default temperature.
DEFAULT_TEMPERATURE = {"groq": 0.3}

# Retries with backoff for rate limits (429). Groq's per-minute token limits are low and a
# multi-agent turn makes many model calls, so retry generously; the SDK honours retry-after.
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))

# Let agents write files, run Python and install packages. Turn this off for public
# deployments: anyone who can open the app could otherwise approve code on your server.
# (The Streamlit app is stricter: it keeps this off unless the variable is set explicitly.)
ALLOW_CODE_EXECUTION = env_flag("ALLOW_CODE_EXECUTION", True)

# LangSmith: with LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, LangChain and
# LangGraph trace every run automatically (no code changes needed).
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")


def tracing_enabled() -> bool:
    return env_flag("LANGSMITH_TRACING", False) and bool(os.getenv("LANGSMITH_API_KEY"))


# tracing switched on without a key would log an error on every run, so switch it off
if env_flag("LANGSMITH_TRACING", False) and not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"


def provider(name: str | None = None) -> str:
    """The LLM provider: `name` if given, else LLM_PROVIDER from the environment (default groq)."""
    name = (name or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {name!r}")
    return name


def model_name(provider_name: str | None = None) -> str:
    """The model id to use on a provider. LLM_MODEL applies only to the provider set in the
    environment; any other provider uses its default model."""
    name = provider(provider_name)
    if name == provider():
        return os.getenv("LLM_MODEL", "").strip() or PROVIDERS[name]["default_model"]
    return PROVIDERS[name]["default_model"]


def model_label() -> str:
    """Short description for logs and the UI, e.g. "groq / openai/gpt-oss-120b"."""
    return f"{provider()} / {model_name()}"


def api_key_name(provider_name: str | None = None) -> str:
    """Environment variable holding the provider's key, e.g. GROQ_API_KEY."""
    return PROVIDERS[provider(provider_name)]["api_key"]


def supports_reasoning_effort(provider_name: str, model: str) -> bool:
    """Whether the model accepts `reasoning_effort` (others reject the parameter)."""
    model = model.lower()
    if provider(provider_name) == "openai":
        return model.startswith(("gpt-5", "o1", "o3", "o4"))
    return model.startswith(("openai/gpt-oss", "qwen/"))


def get_model(
    api_key: str | None = None,
    provider_name: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    """Create the chat model. Arguments override the environment (LLM_PROVIDER, LLM_MODEL,
    LLM_REASONING_EFFORT, the provider's API key); the Streamlit sidebar uses them."""
    name = provider(provider_name)
    model = model or model_name(name)
    kwargs = {"model_provider": name, "max_retries": MAX_RETRIES}
    if api_key:
        kwargs["api_key"] = api_key

    temperature = os.getenv("LLM_TEMPERATURE", "").strip() or DEFAULT_TEMPERATURE.get(name)
    if temperature is not None:
        kwargs["temperature"] = float(temperature)

    # low / medium / high, for reasoning models (gpt-oss and Qwen on Groq, gpt-5 on OpenAI).
    # "low" uses far fewer tokens, which helps with Groq's per-minute token limits.
    effort = (reasoning_effort or os.getenv("LLM_REASONING_EFFORT", "")).strip().lower()
    if effort and supports_reasoning_effort(name, model):
        kwargs["reasoning_effort"] = effort

    return init_chat_model(model, **kwargs)


def is_retryable_model_error(error: Exception) -> bool:
    """Errors worth retrying the model call for:
    - Groq `tool_use_failed`: the model produced a malformed tool call (a retry usually fixes it)
    - rate limits, timeouts and server errors still failing after the SDK's own retries
    """
    if type(error).__name__ == "BadRequestError":
        return "tool_use_failed" in str(error)
    return type(error).__name__ in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}
