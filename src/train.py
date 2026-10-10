"""Train regression models on the processed car features and report MAE / RMSE / R².

    python src/train.py                    # data/features/train.csv + test.csv
    python src/train.py --no-log-target    # train on the raw selling_price instead of its log

Saves the best model (lowest test RMSE) to models/model.joblib and the metrics to
reports/model_metrics.md and reports/model_metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent


def get_models(log_target: bool = False, random_state: int = 42) -> dict[str, RegressorMixin]:
    """Candidate models: a linear baseline and two tree ensembles."""
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_state),
    }
    if log_target:
        models = {
            name: TransformedTargetRegressor(model, func=np.log1p, inverse_func=np.expm1)
            for name, model in models.items()
        }
    return models


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(
    train_path: str | Path,
    test_path: str | Path,
    models_dir: str | Path,
    reports_dir: str | Path,
    target: str = "selling_price",
    log_target: bool = False,
    models: dict[str, RegressorMixin] | None = None,
) -> dict:
    """Fit every model on train, score it on test, save the best one and a metrics report."""
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    for name, df in (("train", train), ("test", test)):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in the {name} set.")
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    models = models or get_models(log_target)
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics[name] = evaluate(y_test, model.predict(X_test))

    best_name = min(metrics, key=lambda name: metrics[name]["rmse"])

    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.joblib"
    joblib.dump(models[best_name], model_path)

    summary = {
        "target": target,
        "log_target": log_target,
        "train_rows": len(train),
        "test_rows": len(test),
        "metrics": metrics,
        "best_model": best_name,
        "model_path": _display_path(model_path),
    }
    (reports_dir / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports_dir / "model_metrics.md").write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _display_path(path: Path) -> str:
    """Path relative to the project when possible, so reports don't contain local folders."""
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path)


def _markdown_report(summary: dict) -> str:
    lines = [
        "# Model metrics",
        "",
        f"Target: `{summary['target']}`" + (" (trained on log1p)" if summary["log_target"] else ""),
        f"Train rows: {summary['train_rows']} | Test rows: {summary['test_rows']}",
        "",
        "| Model | MAE | RMSE | R² |",
        "|---|---:|---:|---:|",
    ]
    for name, m in sorted(summary["metrics"].items(), key=lambda item: item[1]["rmse"]):
        marker = " **(best)**" if name == summary["best_model"] else ""
        lines.append(f"| {name}{marker} | {m['mae']:,.0f} | {m['rmse']:,.0f} | {m['r2']:.3f} |")
    lines += ["", f"Best model saved to `{Path(summary['model_path']).name}` (lowest test RMSE)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate selling-price models.")
    parser.add_argument("--train", default=BASE_DIR / "data" / "features" / "train.csv")
    parser.add_argument("--test", default=BASE_DIR / "data" / "features" / "test.csv")
    parser.add_argument("--target", default="selling_price")
    # selling_price is right-skewed, so training on log1p(price) gives better models
    parser.add_argument("--log-target", action=argparse.BooleanOptionalAction, default=True, help="train on log1p(target)")
    args = parser.parse_args()

    summary = train_and_evaluate(
        args.train, args.test, BASE_DIR / "models", BASE_DIR / "reports",
        target=args.target, log_target=args.log_target,
    )
    sys.stdout.reconfigure(encoding="utf-8")  # the report contains "²"
    print(_markdown_report(summary))
