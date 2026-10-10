#!/usr/bin/env python3
"""
Runner script to generate car data markdown report and return structured output.
This script defines the functions needed (copied from the assistant's modified script)
and then executes generate_car_data_report for the requested CSV path. It captures
exceptions and returns a dict named `output`.
"""
from typing import Dict, Any, List, Tuple, Optional
import os
import pandas as pd
import numpy as np
from pandas.api import types as ptypes
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# TODO(improvement): add a small CLI (argparse) so the CSV path and report name can be
# passed in, e.g. `python run_car_report.py --csv data/car_details.csv`.

# --- Functions (copied and slightly adapted) ---

def load_csv(csv_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        return df
    except pd.errors.EmptyDataError as e:
        raise RuntimeError(f"The file appears to be empty: {csv_path}") from e
    except pd.errors.ParserError as e:
        raise RuntimeError(f"Pandas failed to parse the CSV: {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred when loading the CSV: {e}") from e


def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_columns: List[str] = []
    numerical_columns: List[str] = []

    for col in df.columns:
        ser = df[col]
        # is_categorical_dtype is deprecated in pandas 2.x, so check the dtype directly
        if ptypes.is_bool_dtype(ser) or isinstance(ser.dtype, pd.CategoricalDtype) or ptypes.is_object_dtype(ser):
            categorical_columns.append(col)
        elif ptypes.is_numeric_dtype(ser):
            numerical_columns.append(col)
        else:
            pass

    return categorical_columns, numerical_columns


def get_missing_values_info(df: pd.DataFrame) -> Tuple[Dict[str, int], bool]:
    missing_series = df.isnull().sum()
    missing_per_column = {col: int(cnt) for col, cnt in missing_series.to_dict().items()}
    has_missing = any(cnt > 0 for cnt in missing_per_column.values())
    return missing_per_column, has_missing


def get_duplicates_info(df: pd.DataFrame) -> Tuple[int, bool]:
    duplicate_count = int(df.duplicated(keep='first').sum())
    has_duplicates = duplicate_count > 0
    return duplicate_count, has_duplicates


def numeric_summary_to_serializable(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return {}

    desc = numeric_df.describe().to_dict()
    inverted: Dict[str, Dict[str, Any]] = {}
    for stat, colvals in desc.items():
        for col, val in colvals.items():
            inverted.setdefault(col, {})[stat] = val

    def _convert_value(v: Any) -> Any:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            if np.isnan(v):
                return None
            return float(v)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if pd.isna(v):
            return None
        return v

    serializable: Dict[str, Dict[str, Any]] = {}
    for col, stats in inverted.items():
        serializable[col] = {stat: _convert_value(value) for stat, value in stats.items()}

    return serializable


def make_report(df: pd.DataFrame, csv_path: str) -> Dict[str, Any]:
    rows, cols = df.shape
    categorical_columns, numerical_columns = classify_columns(df)
    missing_per_column, has_missing = get_missing_values_info(df)
    duplicate_count, has_duplicates = get_duplicates_info(df)
    numeric_summary = numeric_summary_to_serializable(df)

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    report: Dict[str, Any] = {
        "source_csv": os.path.abspath(csv_path),
        "shape": {"rows": int(rows), "columns": int(cols)},
        "columns": list(df.columns),
        "dtypes": dtypes,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "missing_values_per_column": missing_per_column,
        "has_missing": has_missing,
        "duplicate_count": duplicate_count,
        "has_duplicates": has_duplicates,
        "numeric_summary": numeric_summary,
    }

    return report


def _format_value_for_md(v: Any) -> str:
    if v is None:
        return "NA"
    if v is pd.NA:
        return "NA"
    if isinstance(v, float) and np.isnan(v):
        return "NA"
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, (np.floating,)):
        return str(float(v))
    if isinstance(v, (np.bool_, bool)):
        return "True" if bool(v) else "False"
    if isinstance(v, (list, tuple, np.ndarray)):
        return ", ".join(str(x) for x in v)
    return str(v)


def print_report(report: Dict[str, Any]) -> None:
    print("===== Car Data Report =====")
    print(f"Source CSV: {report.get('source_csv')}")
    shape = report.get('shape', {})
    print(f"Shape: {shape.get('rows', '?')} rows x {shape.get('columns', '?')} columns")
    print()

    print("Columns and dtypes:")
    dtypes = report.get('dtypes', {})
    for col in report.get('columns', []):
        print(f"  - {col} : {dtypes.get(col)}")
    print()

    print("Categorical columns:")
    for col in report.get('categorical_columns', []):
        print(f"  - {col}")
    print()

    print("Numerical columns:")
    for col in report.get('numerical_columns', []):
        print(f"  - {col}")
    print()

    print("Missing values per column:")
    missing = report.get('missing_values_per_column', {})
    for col, cnt in missing.items():
        print(f"  - {col}: {cnt}")
    print(f"Any missing values? {'Yes' if report.get('has_missing') else 'No'}")
    print()

    print(f"Duplicate rows count: {report.get('duplicate_count', 0)}")
    print(f"Any duplicates? {'Yes' if report.get('has_duplicates') else 'No'}")
    print()

    print("Numeric summary (per numeric column):")
    numeric_summary = report.get('numeric_summary', {})
    if not numeric_summary:
        print("  No numeric columns present.")
    else:
        for col, stats in numeric_summary.items():
            print(f"  - {col}:")
            for stat_name, stat_val in stats.items():
                print(f"      {stat_name}: {_format_value_for_md(stat_val)}")
    print("===========================")


def save_markdown_report(report: Dict[str, Any], directory: str = "reports", filename: str = "car_data_report.md") -> str:
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    abs_path = os.path.abspath(filepath)

    lines: List[str] = []
    lines.append("# Car Data Report")
    lines.append("")
    lines.append(f"**Source CSV:** `{report.get('source_csv')}`")
    shape = report.get('shape', {})
    lines.append(f"**Shape:** {shape.get('rows', '?')} rows × {shape.get('columns', '?')} columns")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Columns and dtypes")
    lines.append("")
    dtypes = report.get('dtypes', {})
    lines.append("| Column | dtype |")
    lines.append("|---|---:|")
    for col in report.get('columns', []):
        dtype_str = dtypes.get(col, "unknown")
        lines.append(f"| {col} | {dtype_str} |")
    lines.append("")

    lines.append("## Categorical columns")
    lines.append("")
    if report.get('categorical_columns'):
        for col in report.get('categorical_columns'):
            lines.append(f"- {col}")
    else:
        lines.append("None")
    lines.append("")

    lines.append("## Numerical columns")
    lines.append("")
    if report.get('numerical_columns'):
        for col in report.get('numerical_columns'):
            lines.append(f"- {col}")
    else:
        lines.append("None")
    lines.append("")

    lines.append("## Missing values per column")
    lines.append("")
    lines.append("| Column | Missing count |")
    lines.append("|---|---:|")
    missing = report.get('missing_values_per_column', {})
    for col in report.get('columns', []):
        cnt = missing.get(col, 0)
        lines.append(f"| {col} | {cnt} |")
    lines.append("")
    lines.append(f"**Any missing values?** {'Yes' if report.get('has_missing') else 'No'}")
    lines.append("")

    lines.append("## Duplicate rows")
    lines.append("")
    lines.append(f"- Duplicate count: {report.get('duplicate_count', 0)}")
    lines.append(f"- Any duplicates? {'Yes' if report.get('has_duplicates') else 'No'}")
    lines.append("")

    lines.append("## Numeric summary (per numeric column)")
    lines.append("")
    numeric_summary = report.get('numeric_summary', {})
    if not numeric_summary:
        lines.append("No numeric columns present.")
    else:
        for col, stats in numeric_summary.items():
            lines.append(f"### {col}")
            lines.append("")
            lines.append("| Statistic | Value |")
            lines.append("|---|---:|")
            stat_order = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            for stat in stat_order:
                if stat in stats:
                    val = _format_value_for_md(stats[stat])
                    lines.append(f"| {stat} | {val} |")
            for stat, value in stats.items():
                if stat not in stat_order:
                    val = _format_value_for_md(value)
                    lines.append(f"| {stat} | {val} |")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Report generated by runner script. 'NA' indicates missing data.")
    lines.append("")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return abs_path


def generate_car_data_report(csv_path: str, reports_dir: str = "reports", markdown_filename: str = "car_data_report.md", encoding: Optional[str] = None) -> Dict[str, Any]:
    df = load_csv(csv_path, encoding=encoding)
    report = make_report(df, csv_path)
    print_report(report)
    markdown_path = save_markdown_report(report, directory=reports_dir, filename=markdown_filename)
    report["_saved_markdown_report"] = markdown_path
    return report

# --- Execute and capture output ---

output = {}
CSV_PATH = BASE_DIR / "data" / "car_details.csv"
try:
    report = generate_car_data_report(str(CSV_PATH), reports_dir=str(BASE_DIR / "reports"))
    md_path = report.get('_saved_markdown_report')
    # Read snippet
    snippet = ''
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            snippet = content[:4000]
    except Exception as e:
        snippet = ''

    output = {
        'status': 'success',
        'error_message': None,
        'report_path': md_path,
        'report_snippet': snippet,
        'report_summary': {
            'shape': report.get('shape'),
            'dtypes': report.get('dtypes'),
            'categorical_columns': report.get('categorical_columns'),
            'numerical_columns': report.get('numerical_columns'),
            'missing_values_per_column': report.get('missing_values_per_column'),
            'has_missing': report.get('has_missing'),
            'duplicate_count': report.get('duplicate_count'),
            'has_duplicates': report.get('has_duplicates'),
            'numeric_summary': report.get('numeric_summary')
        }
    }
except Exception as e:
    tb = traceback.format_exc()
    output = {
        'status': 'error',
        'error_message': tb,
        'report_path': None,
        'report_snippet': None,
        'report_summary': None
    }

# The report itself is printed above; surface errors when run directly
if __name__ == "__main__" and output['status'] == 'error':
    print(output['error_message'])