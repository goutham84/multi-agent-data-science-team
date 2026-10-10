"""Shared fixtures: a temporary project folder and a scripted chat model (no API key needed)."""
import json
import os
import shutil
import uuid

# never send test runs to LangSmith, even if .env turns tracing on
# (set before ds_team.config runs load_dotenv, which doesn't override existing variables)
os.environ["LANGSMITH_TRACING"] = "false"

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ds_team import config
import ds_team.graph  # noqa: F401  (import before tests patch the project paths)

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

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Stream the scripted reply word by word, like a real model."""
        self.calls.append(messages)
        message = self.responses.pop(0)
        words = message.content.split(" ") if message.content else []
        chunks = [AIMessageChunk(content=w if i == 0 else f" {w}") for i, w in enumerate(words)]
        if message.tool_calls:
            chunks.append(AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i, "type": "tool_call_chunk"}
                for i, c in enumerate(message.tool_calls)
            ]))
        for chunk in chunks or [AIMessageChunk(content="")]:
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=ChatGenerationChunk(message=chunk))
            yield ChatGenerationChunk(message=chunk)


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
