import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def ensure_numeric(series):
    """Convert a pandas Series to numeric by removing commas and coercing errors to NaN."""
    return pd.to_numeric(series.astype(str).str.replace(',', '' , regex=False).str.strip(), errors='coerce')


def plot_and_save_hist(data, column, out_path, bins=60, title=None, xlabel=None, ylabel='Count'):
    plt.figure(figsize=(8,6))
    sns.histplot(data[column].dropna(), bins=bins, kde=False)
    plt.title(title or f'Histogram of {column}')
    plt.xlabel(xlabel or column)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # Project root (one level above src/)
    base = Path(__file__).resolve().parent.parent
    plots_dir = base / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    original_csv = base / 'data' / 'car_details.csv'
    cleaned_csv = base / 'data' / 'data_cleaned.csv'

    # Load datasets
    df_orig = pd.read_csv(original_csv)
    df_clean = pd.read_csv(cleaned_csv)

    # Ensure selling_price numeric
    df_orig['selling_price'] = ensure_numeric(df_orig.get('selling_price', pd.Series(dtype=float)))
    df_clean['selling_price'] = ensure_numeric(df_clean.get('selling_price', pd.Series(dtype=float)))

    # Determine combined bin range (no clipping) for fair comparison
    combined = pd.concat([df_orig['selling_price'], df_clean['selling_price']], ignore_index=True).dropna()
    if combined.empty:
        raise ValueError('No numeric selling_price values found in either dataset.')

    overall_min = float(combined.min())
    overall_max = float(combined.max())
    bins = 60

    # Create histograms saving to files (no clipping)
    plot_and_save_hist(
        df_orig.assign(selling_price_orig=lambda d: d['selling_price']),
        'selling_price_orig',
        plots_dir / 'original_selling_price_hist.png',
        bins=bins,
        title='Original data — selling_price (no clipping)'
    )

    plot_and_save_hist(
        df_clean.assign(selling_price_clean=lambda d: d['selling_price']),
        'selling_price_clean',
        plots_dir / 'cleaned_selling_price_hist.png',
        bins=bins,
        title='Cleaned data — selling_price (no clipping)'
    )

    # Overlay comparison plot (use same bins and range)
    plt.figure(figsize=(9,6))
    sns.histplot(df_orig['selling_price'].dropna(), bins=bins, color='blue', alpha=0.5, label='Original')
    sns.histplot(df_clean['selling_price'].dropna(), bins=bins, color='orange', alpha=0.5, label='Cleaned')
    plt.title('Comparison — selling_price (no clipping)')
    plt.xlabel('selling_price')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'selling_price_comparison_hist.png', dpi=150)
    plt.close()

    # Print basic stats
    def summarize(s):
        s = s.dropna()
        return {
            'count': int(s.count()),
            'mean': float(s.mean()) if s.size>0 else None,
            'median': float(s.median()) if s.size>0 else None,
            'min': float(s.min()) if s.size>0 else None,
            'max': float(s.max()) if s.size>0 else None,
            '0.5%': float(s.quantile(0.005)) if s.size>0 else None,
            '99.5%': float(s.quantile(0.995)) if s.size>0 else None
        }

    summary_orig = summarize(df_orig['selling_price'])
    summary_clean = summarize(df_clean['selling_price'])

    print('Original selling_price summary:', summary_orig)
    print('Cleaned selling_price summary :', summary_clean)
    print('Plots saved to:', plots_dir)
    print('Files saved:')
    print('-', plots_dir / 'original_selling_price_hist.png')
    print('-', plots_dir / 'cleaned_selling_price_hist.png')
    print('-', plots_dir / 'selling_price_comparison_hist.png')