"""LangChain tools used by the agents.

Every tool works inside the project folder only and returns text, so errors go back
to the model as a message instead of crashing the graph. Tools that change the
project (writing files, installing packages) pause the graph with a LangGraph
`interrupt` and wait for the user's approval.
"""
import functools
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

import matplotlib

matplotlib.use("Agg")  # render charts straight to files, no GUI window
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.tools import tool
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from ds_team import config

MAX_OUTPUT_CHARS = 8000
MAX_CSV_ROWS = 30
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
APPROVE_WORDS = {"y", "yes", "approve", "approved", "ok"}

# Commands the Shell Agent may run, besides `uv run <file.py>`
READ_ONLY_COMMANDS = [
    ["uv", "pip", "list"],
    ["uv", "tree"],
    ["python", "--version"],
    ["git", "status"],
    ["git", "log", "--oneline"],
]
PACKAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-\[\],<>=!~]*$")


# ============================ Helpers ===============================

def _safe(fn):
    """Return errors to the model as text instead of raising (interrupts still pass through)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except GraphBubbleUp:
            raise  # human-in-the-loop interrupts must reach LangGraph
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
    return wrapper


def _resolve(path: str) -> Path:
    """Resolve a path relative to the project folder and refuse anything outside it."""
    resolved = (config.BASE_DIR / path).resolve()
    if not resolved.is_relative_to(config.BASE_DIR):
        raise PermissionError(f"'{path}' is outside the project folder")
    return resolved


def _rel(path: Path) -> str:
    return path.relative_to(config.BASE_DIR).as_posix() or "."


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated {len(text) - MAX_OUTPUT_CHARS} characters]"


def _ask_user(action: str, **details) -> tuple[bool, str]:
    """Pause the graph until the user answers. Returns (approved, raw answer)."""
    answer = str(interrupt({"action": action, **details})).strip()
    return answer.lower() in APPROVE_WORDS, answer


def _run(command: list[str], timeout: int = 300) -> str:
    """Run a command in the project folder and return its exit code and output."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        command, cwd=config.BASE_DIR, env=env, timeout=timeout,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    output = (proc.stdout + proc.stderr).strip() or "(no output)"
    return _truncate(f"exit code: {proc.returncode}\n{output}")


# ============================ File tools ============================

@tool
@_safe
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a project directory. `pattern` is a glob, e.g. '*.py' or '**/*.csv' (recursive)."""
    root = _resolve(directory)
    files = sorted(
        p for p in root.glob(pattern)
        if p.is_file()
        and p.resolve().is_relative_to(config.BASE_DIR)
        and not IGNORED_DIRS.intersection(p.relative_to(config.BASE_DIR).parts)
    )
    if not files:
        return f"No files matching '{pattern}' in {directory}"
    return _truncate("\n".join(_rel(p) for p in files))


@tool
@_safe
def read_file(path: str) -> str:
    """Read a text file from the project (long files are truncated)."""
    return _truncate(_resolve(path).read_text(encoding="utf-8", errors="replace"))


@tool
@_safe
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file in the project. The user reviews and must approve the content first."""
    file = _resolve(path)
    approved, answer = _ask_user("write_file", path=_rel(file), content=content)
    if not approved:
        return f"The user did not approve writing {_rel(file)}. Their feedback: {answer}"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    return f"Saved {_rel(file)} ({len(content)} characters)"


# ============================ CSV tools =============================

@tool
@_safe
def list_csv_files() -> str:
    """List all CSV datasets in the data/ folder with their sizes."""
    files = sorted(config.DATA_DIR.rglob("*.csv"))
    return "\n".join(f"{_rel(p)} ({p.stat().st_size // 1024} KB)" for p in files) or "No CSV files in data/"


@tool
@_safe
def read_csv_file(path: str, rows: int = 20) -> str:
    """Preview the first rows of a CSV file (at most 30) and report its total size."""
    file = _resolve(path)
    rows = max(1, min(rows, MAX_CSV_ROWS))
    preview = pd.read_csv(file, nrows=rows)
    with open(file, encoding="utf-8", errors="replace") as f:
        total_rows = sum(1 for _ in f) - 1
    return f"{_rel(file)}: {total_rows} rows x {preview.shape[1]} columns\n\n{preview.to_string(index=False)}"


# ============================ Pandas tools ==========================

DataFrameOperation = Literal[
    "head", "tail", "info", "describe", "value_counts", "shape", "columns",
    "dtypes", "missing_values", "duplicates", "nunique", "correlation",
]


@tool
@_safe
def dataframe_operation(path: str, operation: DataFrameOperation, column: Optional[str] = None, n: int = 5) -> str:
    """Run a pandas operation on a CSV file and return the result as text.

    - head / tail: first or last `n` rows
    - shape / columns / dtypes / info: structure (columns lists numerical vs categorical)
    - describe: summary statistics (of one `column` if given)
    - value_counts: frequency of each value in `column`
    - missing_values / duplicates / nunique: data quality checks
    - correlation: correlation matrix of the numerical columns
    """
    df = pd.read_csv(_resolve(path))
    if column is not None and column not in df.columns:
        raise KeyError(f"column '{column}' not found. Columns: {list(df.columns)}")
    data = df[column] if column else df
    n = max(1, min(n, MAX_CSV_ROWS))

    match operation:
        case "head":
            result = data.head(n)
        case "tail":
            result = data.tail(n)
        case "info":
            buffer = io.StringIO()
            df.info(buf=buffer)
            result = buffer.getvalue()
        case "describe":
            result = data.describe() if column else df.describe(include="all")
        case "value_counts":
            if column is None:
                raise ValueError("value_counts needs a `column`")
            result = df[column].value_counts(dropna=False).head(50)
        case "shape":
            result = f"{df.shape[0]} rows x {df.shape[1]} columns"
        case "columns":
            numerical = df.select_dtypes("number").columns.tolist()
            categorical = [c for c in df.columns if c not in numerical]
            result = f"numerical: {numerical}\ncategorical: {categorical}"
        case "dtypes":
            result = df.dtypes
        case "missing_values":
            result = df.isna().sum()
        case "duplicates":
            result = f"{int(df.duplicated().sum())} exact duplicate rows"
        case "nunique":
            result = df.nunique()
        case "correlation":
            result = df.select_dtypes("number").corr().round(3)
        case _:
            raise ValueError(f"unknown operation '{operation}'")

    return _truncate(result if isinstance(result, str) else result.to_string())


# ============================ Visualization tools ===================

ChartType = Literal["bar", "histogram", "scatter", "line", "pie", "box"]


@tool
@_safe
def create_chart(
    path: str,
    chart_type: ChartType,
    x: str,
    y: Optional[str] = None,
    title: Optional[str] = None,
    bins: int = 30,
) -> str:
    """Create a chart from a CSV file and save it as a PNG in plots/.

    - bar: counts of categorical column `x` (or the mean of `y` for each `x`)
    - histogram: distribution of numerical column `x`
    - scatter / line: relationship between numerical columns `x` and `y`
    - pie: share of each category in `x` (top 10)
    - box: distribution of `x`, or of `y` for each category in `x`
    """
    df = pd.read_csv(_resolve(path))
    if chart_type in ("scatter", "line") and y is None:
        raise ValueError(f"{chart_type} charts need both `x` and `y`")

    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        match chart_type:
            case "bar":
                data = df.groupby(x)[y].mean().sort_values(ascending=False) if y else df[x].value_counts()
                data.head(30).plot.bar(ax=ax)
                ax.set_ylabel(f"mean {y}" if y else "count")
            case "histogram":
                df[x].dropna().plot.hist(bins=bins, ax=ax)
                ax.set_xlabel(x)
            case "scatter":
                df.plot.scatter(x=x, y=y, ax=ax, alpha=0.5)
            case "line":
                df.sort_values(x).plot.line(x=x, y=y, ax=ax)
            case "pie":
                df[x].value_counts().head(10).plot.pie(ax=ax, autopct="%1.1f%%")
                ax.set_ylabel("")
            case "box":
                df.boxplot(column=y, by=x, ax=ax) if y else df.boxplot(column=x, ax=ax)

        ax.set_title(title or f"{chart_type.title()} of {x}" + (f" vs {y}" if y else ""))
        fig.tight_layout()
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        file_name = re.sub(r"[^A-Za-z0-9_]+", "_", "_".join(filter(None, [chart_type, x, y])))
        out = config.PLOTS_DIR / f"{file_name}.png"
        fig.savefig(out, dpi=150)
    finally:
        plt.close(fig)
    return f"Chart saved to {_rel(out)}"


# ============================ Python & shell tools ==================

@tool
@_safe
def run_python_file(path: str) -> str:
    """Run a Python file from the project and return its output."""
    file = _resolve(path)
    if file.suffix != ".py":
        raise ValueError("only .py files can be run")
    if not file.exists():
        raise FileNotFoundError(_rel(file))
    return _run([sys.executable, str(file)])


@tool
@_safe
def install_package(package: str) -> str:
    """Install a Python package into the project (`uv add`). The user must approve it first."""
    if not PACKAGE_NAME.match(package):
        raise ValueError(f"invalid package name: {package!r}")
    approved, answer = _ask_user("install_package", package=package)
    if not approved:
        return f"The user did not approve installing {package}. Their feedback: {answer}"
    if shutil.which("uv"):
        return _run(["uv", "add", package])
    return _run([sys.executable, "-m", "pip", "install", package])


@tool
@_safe
def run_shell_command(command: str) -> str:
    """Run an allow-listed command in the project folder.

    Allowed: `uv run <file.py>`, `uv pip list`, `uv tree`, `python --version`,
    `git status`, `git log --oneline`.
    """
    args = shlex.split(command, posix=os.name != "nt")
    if args[:2] == ["uv", "run"]:
        if len(args) != 3 or not args[2].endswith(".py"):
            raise PermissionError("only `uv run <file.py>` is allowed")
        _resolve(args[2])
    elif args not in READ_ONLY_COMMANDS:
        raise PermissionError(f"command not allowed: {command}")
    return _run(args)


# ============================ Web search ============================

@tool
@_safe
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo, e.g. for the latest library documentation."""
    from ddgs import DDGS

    results = DDGS().text(query, max_results=max_results)
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results) or "No results"
