"""Shared fixtures: a temporary project folder and a scripted chat model (no API key needed)."""
import shutil
import uuid

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ds_team import config

SAMPLE_CSV = config.DATA_DIR / "sample_cars.csv"


class ScriptedModel(BaseChatModel):
    """Returns pre-written responses in order and records the messages it was called with."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project folder with data/sample.csv, used as the tools' BASE_DIR."""
    (tmp_path / "data").mkdir()
    shutil.copy(SAMPLE_CSV, tmp_path / "data" / "sample.csv")
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "PLOTS_DIR", tmp_path / "plots")
    return tmp_path
