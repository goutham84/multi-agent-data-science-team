# Model metrics

Target: `selling_price` (trained on log1p)
Train rows: 2811 | Test rows: 703

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| ridge **(best)** | 120,353 | 189,503 | 0.748 |
| hist_gradient_boosting | 120,230 | 199,168 | 0.722 |
| random_forest | 123,793 | 207,495 | 0.698 |

Best model saved to `model.joblib` (lowest test RMSE).
