import pandas as pd
import numpy as np
from pathlib import Path

def clean_car_data(
    input_path: str,
    output_path: str,
    log_path: str,
    low_q: float = 0.005,
    high_q: float = 0.995,
    drop_exact_duplicates: bool = True
) -> dict:
    """
    Clean car dataset.

    Steps performed (per user instructions):
    1) Remove exact duplicate rows (if drop_exact_duplicates=True)
    2) Remove extreme outliers on selling_price and km_driven using quantiles [low_q, high_q]
    3) Extract brand (first token) and model (remaining tokens) from 'name'
    4) Compute age = max_year - year (use dataset max year)
    5) Do NOT transform the target 'selling_price'
    6) Do NOT encode categorical features
    7) Save cleaned CSV to output_path and write a cleaning log to log_path

    Returns a summary dict with keys:
      initial_shape, initial_duplicates, rows_dropped_dupes,
      sp_low, sp_high, km_low, km_high, rows_dropped_outliers,
      final_shape, output_path, log_path, preview (first 10 rows as list of dicts)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    initial_duplicates = int(df.duplicated().sum())

    # Drop exact duplicates
    if drop_exact_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    after_dupe_shape = df.shape
    rows_dropped_dupes = initial_shape[0] - after_dupe_shape[0]

    # Ensure numeric columns are numeric (remove common formatting like commas)
    for col in ['selling_price', 'km_driven', 'year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

    # Determine outlier thresholds
    sp_low, sp_high = df['selling_price'].quantile([low_q, high_q])
    km_low, km_high = df['km_driven'].quantile([low_q, high_q])

    # Filter rows: keep rows where both selling_price and km_driven are within thresholds
    mask_sp = df['selling_price'].between(sp_low, sp_high, inclusive='both')
    mask_km = df['km_driven'].between(km_low, km_high, inclusive='both')
    mask = mask_sp & mask_km
    rows_before_outlier_filter = df.shape[0]
    df = df[mask].reset_index(drop=True)
    after_outlier_shape = df.shape
    rows_dropped_outliers = rows_before_outlier_filter - after_outlier_shape[0]

    # Extract brand and model from 'name'
    if 'name' in df.columns:
        df['name'] = df['name'].fillna('').astype(str)
        df['brand'] = df['name'].str.split().str[0]
        df['model'] = df['name'].str.split().str[1:].str.join(' ')

    # Compute age using dataset max year as baseline
    if 'year' in df.columns:
        max_year = int(df['year'].max()) if df['year'].notna().any() else pd.Timestamp.now().year
        df['age'] = (max_year - df['year']).fillna(0).astype(int)
    else:
        max_year = None

    # Save cleaned CSV
    df.to_csv(output_path, index=False)

    # Save cleaning log
    log_lines = []
    log_lines.append('# Cleaning log for ' + input_path.name)
    log_lines.append(f'Original shape: {initial_shape[0]} rows × {initial_shape[1]} cols')
    log_lines.append(f'Exact duplicate rows dropped: {rows_dropped_dupes}')
    log_lines.append(f'Rows dropped due to outlier removal: {rows_dropped_outliers}')
    log_lines.append(f'Selling price thresholds ({low_q*100}th, {high_q*100}th): {sp_low}, {sp_high}')
    log_lines.append(f'km_driven thresholds ({low_q*100}th, {high_q*100}th): {km_low}, {km_high}')
    log_lines.append(f'Baseline year used for age computation: {max_year}')
    log_lines.append(f'Final shape after cleaning: {after_outlier_shape[0]} rows × {after_outlier_shape[1]} cols')
    log_lines.append(f'Cleaned CSV saved to: {str(output_path)}')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    result = {
        'initial_shape': initial_shape,
        'initial_duplicates': initial_duplicates,
        'rows_dropped_dupes': rows_dropped_dupes,
        'sp_low': float(sp_low),
        'sp_high': float(sp_high),
        'km_low': float(km_low),
        'km_high': float(km_high),
        'rows_dropped_outliers': rows_dropped_outliers,
        'final_shape': after_outlier_shape,
        'output_path': str(output_path),
        'log_path': str(log_path),
        'preview': df.head(10).to_dict(orient='records')
    }

    return result

if __name__ == '__main__':
    # Default paths (match workspace structure)
    # Project root (one level above src/)
    base = Path(__file__).resolve().parent.parent

    input_csv = base / 'data' / 'car_details.csv'
    cleaned_csv = base / 'data' / 'data_cleaned.csv'
    log_md = base / 'reports' / 'cleaning_log_data_cleaned.md'

    summary = clean_car_data(str(input_csv), str(cleaned_csv), str(log_md))
    print('Cleaning completed. Summary:')
    for k, v in summary.items():
        if k != 'preview':
            print(f'{k}: {v}')
    print('\nPreview (first rows):')
    for row in summary['preview']:
        print(row)