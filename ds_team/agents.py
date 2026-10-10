"""The six specialist agents.

Each agent is a LangChain `create_agent` (a ReAct loop built on LangGraph) with
its own tools and instructions. The Team Leader in graph.py delegates to them.
"""
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from ds_team import tools
from ds_team.config import BASE_DIR, DEFAULT_DATASET, is_retryable_model_error

DATASET = DEFAULT_DATASET.relative_to(BASE_DIR).as_posix()


@dataclass(frozen=True)
class AgentSpec:
    name: str                # graph node name, e.g. "data_loader"
    description: str         # tells the Team Leader when to delegate to this agent
    tools: list[BaseTool]
    instructions: list[str]


# instructions shared by every specialist
COMMON_INSTRUCTIONS = [
    "You receive one task from the Team Leader. Complete it with your tools.",
    "All paths are relative to the project root.",
    "Reply with a concise summary of what you did and found, including file paths.",
]

AGENT_SPECS = [
    # AGENT 1: Data loader agent
    AgentSpec(
        name="data_loader",
        description="Lists the CSV datasets and previews their rows (max 30 rows).",
        tools=[tools.list_csv_files, tools.read_csv_file, tools.list_files],
        instructions=[
            "You are an expert in loading CSV files from the project folder.",
            "You can list the CSV files and preview them with `read_csv_file`.",
            "Make sure to never read more than 20 to 30 rows of a CSV file.",
            f"The car details data is located at: {DATASET}",
        ],
    ),
    # AGENT 2: File manager agent
    AgentSpec(
        name="file_manager",
        description="Lists, reads and writes project files (writing needs user approval).",
        tools=[tools.list_files, tools.read_file, tools.write_file],
        instructions=[
            "You are an expert File Management Agent.",
            "Your task is to list, read and write project files when asked.",
            "Make sure to never read CSV files, you can only list them.",
            "Writing a file asks the user for approval. If they reject it, report their feedback.",
        ],
    ),
    # AGENT 3: Data understanding agent
    AgentSpec(
        name="data_understanding",
        description="Explores a dataset with pandas: shape, dtypes, describe, value_counts, missing values, duplicates, correlations.",
        tools=[tools.list_csv_files, tools.dataframe_operation],
        instructions=[
            "You are an expert in exploring data with pandas.",
            "Use head, tail, info, shape and describe for numerical columns, and value_counts for categorical columns.",
            "Always list the numerical and categorical columns of the dataset.",
            "Check data quality: missing values, duplicates and suspicious values.",
        ],
    ),
    # AGENT 4: Visualization agent
    AgentSpec(
        name="visualization",
        description="Creates bar, histogram, scatter, line, pie and box charts, saved to plots/.",
        tools=[tools.list_csv_files, tools.dataframe_operation, tools.create_chart],
        instructions=[
            "You are an expert in creating plots with matplotlib.",
            "Use bar charts for categorical columns and histograms for numerical columns.",
            "Use scatter plots to study the relationship between two numerical columns.",
            "Check the column names and types with `dataframe_operation` before plotting.",
        ],
    ),
    # AGENT 5: Coding agent
    AgentSpec(
        name="coding",
        description="Writes and runs Python code for ML: data cleaning, feature engineering, model training and evaluation.",
        tools=[
            tools.list_files, tools.read_file, tools.write_file,
            tools.run_python_file, tools.install_package, tools.web_search,
        ],
        instructions=[
            "You are an expert coding agent proficient in writing Python code for Machine Learning and Data Science.",
            "You may use numpy, pandas, scikit-learn, scipy, matplotlib and seaborn.",
            "Save code with `write_file`: the user reviews and approves it before it is saved. Then run it with `run_python_file`.",
            "Put reusable code in src/, read data from data/ and save models to models/.",
            "The existing pipeline is src/data_cleaning.py -> src/preprocessing.py -> src/train.py "
            "(trains models and reports MAE/RMSE/R² to reports/model_metrics.md). Reuse it before writing new code.",
            "If the user rejects your code, revise it using their feedback.",
            "If a package is missing, install it with `install_package`.",
            "Use `web_search` when you need the latest documentation of a library.",
        ],
    ),
    # AGENT 6: Shell agent
    AgentSpec(
        name="shell",
        description="Runs project scripts with `uv run <file.py>` and read-only commands (uv pip list, git status).",
        tools=[tools.run_shell_command, tools.list_files],
        instructions=[
            "You can run a small allow-list of shell commands; use this capability with caution.",
            "Use `uv run <python_file.py>` to run project scripts.",
            "Never try to delete files or change the project structure.",
        ],
    ),
]


# tools that change the project or run code (removed when code execution is disabled)
EXECUTION_TOOLS = {"write_file", "run_python_file", "install_package", "run_shell_command"}
EXECUTION_DISABLED_NOTE = (
    "Code execution is disabled in this deployment: you cannot save files or run code, "
    "so show any code in your reply instead."
)


def enabled_specs(allow_code_execution: bool = True) -> list[AgentSpec]:
    """The agents available in this deployment, with execution tools removed if needed."""
    if allow_code_execution:
        return AGENT_SPECS
    specs = []
    for spec in AGENT_SPECS:
        safe_tools = [t for t in spec.tools if t.name not in EXECUTION_TOOLS]
        if not any(t.name in EXECUTION_TOOLS for t in spec.tools):
            specs.append(spec)
        elif spec.name != "shell":  # the shell agent has nothing left to do
            specs.append(AgentSpec(spec.name, spec.description, safe_tools, spec.instructions + [EXECUTION_DISABLED_NOTE]))
    return specs


def build_agents(model: BaseChatModel, specs: list[AgentSpec] = AGENT_SPECS) -> dict[str, CompiledStateGraph]:
    """Create one LangChain agent per spec, keyed by agent name."""
    return {
        spec.name: create_agent(
            model,
            tools=spec.tools,
            system_prompt="\n".join(f"- {line}" for line in spec.instructions + COMMON_INSTRUCTIONS),
            middleware=[_retry_middleware()],
            name=spec.name,
        )
        for spec in specs
    }


def _retry_middleware() -> ModelRetryMiddleware:
    """Retry malformed tool calls (Groq `tool_use_failed`) and rate limits. If it still fails,
    the agent reports the error to the Team Leader instead of crashing the whole turn."""
    return ModelRetryMiddleware(
        max_retries=3,
        retry_on=is_retryable_model_error,
        on_failure=lambda error: f"I could not finish this task because the model kept failing: {error}",
        initial_delay=2.0,
    )
