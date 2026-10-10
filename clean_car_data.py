#!/usr/bin/env python3
"""
Data cleaning runner for car_details.csv
Performs the requested cleaning steps and writes cleaned CSV.
Returns an `output` dict with status and details.
"""
from typing import Dict, Any
import os
import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# TODO(improvement): this script overlaps with src/data_cleaning.py. Keep one cleaning
# pipeline (src/data_cleaning.py) and turn this logic into a function there, instead of
# running everything at module import time.

output: Dict[str, Any] = {}

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "car_details.csv"
CLEANED_PATH = BASE_DIR / "data" / "car_data_cleaned.csv"

try:
    # 1. Load CSV with pandas, handling errors
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at path: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 2. Record initial shape
    rows_before, cols_before = df.shape

    # 3. Standardize string columns: strip and replace common placeholders with np.nan
    placeholders = set(["", " ", "NA", "N/A", "na", "None", "none", "?"])
    # Process object (string) columns
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        # Strip whitespace (if not null)
        df[col] = df[col].astype('object')
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        # Replace placeholders and empty strings with np.nan (case-insensitive where applicable)
        def replace_placeholder(x):
            if x is None:
                return np.nan
            if isinstance(x, str):
                if x == "":
                    return np.nan
                # compare case-insensitively for textual placeholders
                if x.lower() in {p.lower() for p in placeholders if isinstance(p, str)}:
                    return np.nan
            return x
        df[col] = df[col].apply(replace_placeholder)

    # 4. Convert numeric-like columns to numeric types (coerce errors to NaN)
    numeric_cols = ['year', 'selling_price', 'km_driven']
    conversion_issues = {}
    for col in numeric_cols:
        if col in df.columns:
            before_nonnull = df[col].notna().sum()
            # Remove commas if present and strip
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                # Convert empty or 'nan' strings to NaN
                df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'none': np.nan})
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_nonnull = df[col].notna().sum()
            conversion_issues[col] = int(before_nonnull - after_nonnull)
            # Convert to pandas nullable integer dtype if desired
            # Keep as float for now until validation; will convert to Int64 if no NaNs in column
        else:
            conversion_issues[col] = None

    # 5. Validate and fix 'year'
    current_year = datetime.now().year
    if 'year' in df.columns:
        # Set invalid years (<1900 or > current_year) to NaN
        df.loc[~df['year'].between(1900, current_year), 'year'] = np.nan

    # 6. Enforce selling_price > 0 and km_driven >= 0; set invalid to NaN
    if 'selling_price' in df.columns:
        df.loc[~(df['selling_price'] > 0), 'selling_price'] = np.nan
    if 'km_driven' in df.columns:
        df.loc[~(df['km_driven'] >= 0), 'km_driven'] = np.nan

    # 7. Drop exact duplicate rows (keep first) and count duplicates removed
    duplicates_before = df.duplicated(keep='first').sum()
    df = df.drop_duplicates(keep='first')
    duplicates_removed = int(duplicates_before)

    # 8. Optionally drop rows with critical missing values in ['year','selling_price','km_driven']
    critical_cols = [c for c in numeric_cols if c in df.columns]
    rows_before_drop = df.shape[0]
    df = df.dropna(subset=critical_cols)
    rows_after_drop = df.shape[0]
    dropped_rows_due_to_missing_critical_fields = int(rows_before_drop - rows_after_drop)

    # 9. Create new column 'age' = current_year - year (integer); if year is NaN, age is NaN
    if 'year' in df.columns:
        # year may be float; create age as Int64 nullable
        df['age'] = df['year'].apply(lambda y: int(current_year - y) if pd.notna(y) else pd.NA)
        # set dtype to Int64
        df['age'] = df['age'].astype('Int64')

    # 10. Convert low-cardinality object columns to category dtype
    low_card_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    for col in low_card_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # After cleaning, try to set numeric columns to nullable Int64 where possible
    for col in ['year', 'selling_price', 'km_driven']:
        if col in df.columns:
            # If column has no fractional values and dtype is numeric, convert to Int64
            # We'll check if all non-null values are integer-valued
            non_null = df[col].dropna()
            if not non_null.empty and (np.all(np.equal(np.mod(non_null, 1), 0))):
                # safe to convert
                df[col] = df[col].astype('Int64')

    # 11. Save cleaned DataFrame to path without the index
    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)

    # 12. Prepare output
    rows_after, cols_after = df.shape
    preview_csv = df.head(10).to_csv(index=False)
    dtypes_after = {col: str(dtype) for col, dtype in df.dtypes.items()}

    output = {
        'status': 'success',
        'error_message': None,
        'cleaned_path': os.path.abspath(CLEANED_PATH),
        'rows_before': int(rows_before),
        'rows_after': int(rows_after),
        'duplicates_removed': int(duplicates_removed),
        'dropped_rows_due_to_missing_critical_fields': int(dropped_rows_due_to_missing_critical_fields),
        'conversion_issues': conversion_issues,
        'preview': preview_csv,
        'dtypes_after': dtypes_after
    }

except Exception as e:
    tb = traceback.format_exc()
    output = {
        'status': 'error',
        'error_message': tb,
        'cleaned_path': None,
        'rows_before': None,
        'rows_after': None,
        'duplicates_removed': None,
        'dropped_rows_due_to_missing_critical_fields': None,
        'preview': None,
        'dtypes_after': None
    }

# Print a short summary when run directly (`output` stays available to callers)
if __name__ == "__main__":
    for key, value in output.items():
        if key != 'preview':
            print(f"{key}: {value}")