"""LLM selection from the environment: LLM_PROVIDER, LLM_MODEL and the optional tuning variables."""
import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ds_team import config

LLM_VARS = ("LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_REASONING_EFFORT")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in LLM_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_defaults_to_groq_gpt_oss_120b():
    model = config.get_model()
    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"
    assert model.temperature == 0.3
    assert model.max_retries == config.MAX_RETRIES
    assert config.api_key_name() == "GROQ_API_KEY"
    assert config.model_label() == "groq / openai/gpt-oss-120b"


def test_other_groq_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen/qwen3.8-27b")
    assert config.get_model().model_name == "qwen/qwen3.8-27b"


def test_openai_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")  # case-insensitive
    model = config.get_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5-mini"
    assert model.temperature is None  # gpt-5 models reject a custom temperature
    assert config.api_key_name() == "OPENAI_API_KEY"


def test_openai_with_custom_model_and_temperature(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    model = config.get_model()
    assert model.model_name == "gpt-4.1-mini"
    assert model.temperature == 0.0


@pytest.mark.parametrize("provider, model_class", [("groq", ChatGroq), ("openai", ChatOpenAI)])
def test_reasoning_effort(monkeypatch, provider, model_class):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "Low")
    model = config.get_model()
    assert isinstance(model, model_class)
    assert model.reasoning_effort == "low"


def test_api_key_argument_overrides_environment(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    model = config.get_model(api_key="gsk-from-sidebar")
    assert model.groq_api_key.get_secret_value() == "gsk-from-sidebar"


def test_unknown_provider_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="LLM_PROVIDER must be one of"):
        config.get_model()
