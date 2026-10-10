import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# TODO(improvement): there is no model training/evaluation script yet. Add src/train.py that
# loads data/features/train.csv, fits a regressor (e.g. RandomForest/XGBoost), reports
# MAE/RMSE/R2 on test.csv and saves the model to models/. This completes the ML pipeline story.
# TODO(improvement): selling_price is right-skewed; try training on np.log1p(selling_price).


def preprocess_and_save(
    input_path: str,
    output_dir: str,
    models_dir: str,
    drop_columns: list = None,
    target_col: str = 'selling_price',
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Load cleaned data, drop specified columns, split into train/test,
    apply ColumnTransformer (OHE for categoricals, StandardScaler for numericals),
    transform data, save processed train/test (features+target) to output_dir,
    and save the fitted ColumnTransformer to models_dir/preprocessor.joblib.

    Returns a summary dict with paths and basic shapes.
    """
    drop_columns = drop_columns or ['name', 'year', 'model']
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(input_path)

    # Drop requested columns if they exist
    drop_list = [c for c in drop_columns if c in df.columns]
    if drop_list:
        df = df.drop(columns=drop_list)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Identify numerical and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build ColumnTransformer
    transformers = []
    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols))
    if categorical_cols:
        # Updated parameter: use sparse_output for modern scikit-learn
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

    if not transformers:
        # Nothing to transform: save combined train/test as-is
        X_train_proc = X_train.copy()
        X_test_proc = X_test.copy()
        ct = None
    else:
        ct = ColumnTransformer(transformers, remainder='drop')
        ct.fit(X_train)

        X_train_trans = ct.transform(X_train)
        X_test_trans = ct.transform(X_test)

        # Build feature names in correct order: numeric then OHE features
        feature_names = []
        if numeric_cols:
            feature_names.extend(numeric_cols)
        if categorical_cols:
            ohe = None
            if 'cat' in ct.named_transformers_:
                ohe = ct.named_transformers_['cat']
            if ohe is not None:
                try:
                    ohe_names = list(ohe.get_feature_names_out(categorical_cols))
                except Exception:
                    # fallback to simple names
                    ohe_names = []
                    for col in categorical_cols:
                        ohe_names.append(col)
                feature_names.extend(ohe_names)

        # Create DataFrames from transformed arrays
        X_train_proc = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
        X_test_proc = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # Reset indices and concatenate features + target for train and test
    combined_train = pd.concat([X_train_proc.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    combined_test = pd.concat([X_test_proc.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # Paths
    combined_train_path = output_dir / 'train.csv'
    combined_test_path = output_dir / 'test.csv'
    preprocessor_path = models_dir / 'preprocessor.joblib'

    # Save CSVs
    combined_train.to_csv(combined_train_path, index=False)
    combined_test.to_csv(combined_test_path, index=False)

    # Save ColumnTransformer if present
    if ct is not None:
        joblib.dump(ct, preprocessor_path)
        preprocessor_saved = True
    else:
        preprocessor_saved = False

    # Summary
    summary = {
        'input_path': str(input_path),
        'output_dir': str(output_dir),
        'models_dir': str(models_dir),
        'train_path': str(combined_train_path),
        'test_path': str(combined_test_path),
        'preprocessor_path': str(preprocessor_path) if preprocessor_saved else None,
        'X_train_shape': X_train_proc.shape,
        'X_test_shape': X_test_proc.shape,
        'y_train_shape': (int(y_train.shape[0]),),
        'y_test_shape': (int(y_test.shape[0]),),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols
    }

    return summary


if __name__ == '__main__':
    # Project root (one level above src/)
    base = Path(__file__).resolve().parent.parent
    cleaned_csv = base / 'data' / 'data_cleaned.csv'
    out_dir = base / 'data' / 'features'
    models_dir = base / 'models'

    summary = preprocess_and_save(str(cleaned_csv), str(out_dir), str(models_dir))
    print('Preprocessing completed. Summary:')
    for k, v in summary.items():
        print(f'{k}: {v}')