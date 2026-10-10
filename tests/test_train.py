import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.train import evaluate, get_models, train_and_evaluate


@pytest.fixture
def feature_files(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    price = np.exp(12 + X @ [0.5, -0.3, 0.2])  # positive, right-skewed like car prices
    df = pd.DataFrame(X, columns=["km_driven", "age", "fuel_Diesel"]).assign(selling_price=price)
    df[:150].to_csv(tmp_path / "train.csv", index=False)
    df[150:].to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


def test_evaluate_metrics():
    assert evaluate([1, 2, 3], [1, 2, 3]) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_train_and_evaluate_saves_best_model_and_report(feature_files):
    summary = train_and_evaluate(
        feature_files / "train.csv", feature_files / "test.csv",
        feature_files / "models", feature_files / "reports",
        log_target=True,
        models={"linear": get_models(log_target=True)["ridge"], "mean_only": _MeanModel()},
    )

    assert summary["best_model"] == "linear"
    assert summary["metrics"]["linear"]["r2"] > 0.9
    assert (feature_files / "models" / "model.joblib").exists()
    assert json.loads((feature_files / "reports" / "model_metrics.json").read_text())["best_model"] == "linear"
    assert "| linear **(best)** |" in (feature_files / "reports" / "model_metrics.md").read_text(encoding="utf-8")


def test_missing_target_column(feature_files):
    with pytest.raises(ValueError, match="Target column 'price' not found"):
        train_and_evaluate(
            feature_files / "train.csv", feature_files / "test.csv",
            feature_files / "models", feature_files / "reports",
            target="price", models={"linear": LinearRegression()},
        )


class _MeanModel(LinearRegression):
    """Baseline that always predicts the training mean."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)
