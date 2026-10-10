"""LangGraph supervisor graph: the Team Leader delegates work to the specialist agents.

    START -> team_leader --delegate_to_<agent>--> <agent> --result--> team_leader -> ... -> END
                         --remember-----------> remember --------> team_leader

The Team Leader is a tool-calling LLM. Each `delegate_to_<agent>` tool call is routed
with `Send` to that agent's node (independent tasks run in parallel), and the agent's
answer comes back to the Team Leader as the tool result. `remember` calls store facts
in `notes`, the session memory. A checkpointer persists the whole state per thread.
"""
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Send
from pydantic import BaseModel, Field

from ds_team import config
from ds_team.agents import AgentSpec, build_agents, enabled_specs
from ds_team.config import DB_PATH, get_model

# how many past user turns the Team Leader sees (older messages stay in the checkpoint)
HISTORY_TURNS = 5

TEAM_LEADER_PROMPT = """You are an expert Data Scientist and the leader of a data science team.

Your team members (delegate with the matching `delegate_to_<name>` tool):
{team}

How to work:
- Guide the user through the machine learning pipeline: data loading, data understanding, data cleaning, visualisation, feature engineering, model training and model evaluation.
- Delegate each step to the right member with a clear, self-contained task. Members do not see this conversation, so include file paths and every detail they need.
- Delegate to several members at once when their tasks are independent.
- Writing files and installing packages ask the user for approval automatically; members report if the user rejected something.
- After each step summarise the results and suggest the next step, like a professional data scientist.
- Use `remember` to save important facts such as data paths, model paths and src files, and anything the user asks you to remember.
- If a member reports an error, work out an approach to debug and solve it.
- Answer simple questions yourself without delegating.
{execution_note}
Session notes:
{notes}"""


# ============================ State =================================

def _merge_notes(current: dict[str, str], update: dict[str, str]) -> dict[str, str]:
    return {**current, **update}


class TeamState(MessagesState):
    notes: Annotated[dict[str, str], _merge_notes]


class AgentTask(TypedDict):
    """Payload sent to a specialist agent node."""
    task: str
    tool_call_id: str
    notes: dict[str, str]


# ============================ Team Leader tools =====================

class Delegation(BaseModel):
    task: str = Field(
        description="Clear, self-contained instructions: what to do, which files to use, "
                    "and any code or details the user already gave or approved."
    )


def _delegation_tool(spec: AgentSpec) -> BaseTool:
    @tool(f"delegate_to_{spec.name}", args_schema=Delegation, description=f"Delegate a task to the {spec.name} agent. {spec.description}")
    def delegate(task: str) -> str:
        # never executed: route_team_leader sends this call to the agent's node
        raise NotImplementedError

    return delegate


@tool
def remember(key: str, value: str) -> str:
    """Save an important fact for this session (data paths, model paths, user preferences)."""
    # never executed: route_team_leader sends this call to the `remember` node
    raise NotImplementedError


def _recent_turns(messages: list[AnyMessage], turns: int = HISTORY_TURNS) -> list[AnyMessage]:
    """Keep the last `turns` user turns. Cutting at a user message never splits a tool call from its result."""
    user_turns = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    start = user_turns[-turns] if len(user_turns) >= turns else 0
    return messages[start:]


def _patch_dangling_tool_calls(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Add a placeholder result for tool calls that never got one (e.g. a turn that crashed),
    because chat APIs reject a tool call without a matching tool result."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    patched = []
    for message in messages:
        patched.append(message)
        for call in getattr(message, "tool_calls", None) or []:
            if call["id"] not in answered:
                patched.append(ToolMessage("Cancelled: this task was interrupted.", tool_call_id=call["id"], status="error"))
    return patched


def _format_notes(notes: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in notes.items())


# ============================ Graph =================================

def build_graph(
    model: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
    allow_code_execution: bool | None = None,
) -> CompiledStateGraph:
    """Build the Data Science Team graph. Without a checkpointer the graph has no memory."""
    model = model or get_model()
    if allow_code_execution is None:
        allow_code_execution = config.ALLOW_CODE_EXECUTION
    specs = enabled_specs(allow_code_execution)
    agents = build_agents(model, specs)
    delegations = {f"delegate_to_{spec.name}": spec.name for spec in specs}
    leader_model = model.bind_tools([_delegation_tool(spec) for spec in specs] + [remember])
    team = "\n".join(f"- {spec.name}: {spec.description}" for spec in specs)
    execution_note = "" if allow_code_execution else (
        "- Code execution is disabled in this deployment: nothing can be saved, run or installed. "
        "The coding member can still write code in its reply for the user to run locally.\n"
    )

    def team_leader(state: TeamState) -> dict:
        notes = _format_notes(state.get("notes") or {}) or "(none yet)"
        prompt = TEAM_LEADER_PROMPT.format(team=team, notes=notes, execution_note=execution_note)
        history = _patch_dangling_tool_calls(_recent_turns(state["messages"]))
        response = leader_model.invoke([SystemMessage(prompt), *history])
        return {"messages": [response]}

    def route_team_leader(state: TeamState) -> list[Send] | str:
        tool_calls = getattr(state["messages"][-1], "tool_calls", None)
        if not tool_calls:
            return END
        notes = state.get("notes") or {}
        sends = []
        for call in tool_calls:
            agent = delegations.get(call["name"])
            if agent:
                task = AgentTask(task=call["args"].get("task", ""), tool_call_id=call["id"], notes=notes)
                sends.append(Send(agent, task))
            else:
                sends.append(Send("remember", call))
        return sends

    def make_agent_node(name: str, agent: CompiledStateGraph):
        def run_agent(payload: AgentTask) -> dict:
            task = payload["task"]
            if payload["notes"]:
                task += f"\n\nSession notes from the Team Leader:\n{_format_notes(payload['notes'])}"
            result = agent.invoke({"messages": [HumanMessage(task)]})
            answer = result["messages"][-1].text or "(the agent finished without a reply)"
            return {"messages": [ToolMessage(answer, tool_call_id=payload["tool_call_id"], name=f"delegate_to_{name}")]}

        return run_agent

    def remember_node(call: ToolCall) -> dict:
        if call["name"] != "remember":
            return {"messages": [ToolMessage(f"Error: unknown tool '{call['name']}'", tool_call_id=call["id"], status="error")]}
        key, value = call["args"].get("key", "note"), call["args"].get("value", "")
        return {
            "notes": {key: value},
            "messages": [ToolMessage(f"Saved note '{key}'", tool_call_id=call["id"], name="remember")],
        }

    graph = StateGraph(TeamState)
    graph.add_node("team_leader", team_leader)
    graph.add_node("remember", remember_node)
    for name, agent in agents.items():
        graph.add_node(name, make_agent_node(name, agent))

    graph.add_edge(START, "team_leader")
    graph.add_conditional_edges("team_leader", route_team_leader, [*agents, "remember", END])
    for node in [*agents, "remember"]:
        graph.add_edge(node, "team_leader")

    return graph.compile(checkpointer=checkpointer, name="data_science_team")


def sqlite_checkpointer(path: Path = DB_PATH) -> SqliteSaver:
    """Persist conversations in a local SQLite file so the team remembers past sessions."""
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
