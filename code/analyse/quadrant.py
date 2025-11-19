"""
SDG Quadrant Analysis
Generate two versions of quadrant charts:
1. Standardized (linear)
2. Log-standardized
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Try to import adjustText for better label positioning
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Warning: adjustText not installed. Labels may overlap.")
    print("Install with: pip install adjustText")

def load_and_process(input_csv: str) -> pd.DataFrame:
    """
    Load data, aggregate Pos/Neg nodes, and apply both standardization methods

    For each SDG subtarget (e.g., 8.2):
    - P = total flow of 8.2_Pos node
    - N = total flow of 8.2_Neg node

    Args:
        input_csv: path to sdg_summary_statistics.csv

    Returns:
        DataFrame with both linear and log-standardized values
    """
    print(f"Loading data: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"  Original data: {len(df)} rows")

    # Calculate total flow for each node
    df['total_flow'] = df['total_inflow'] + df['total_outflow']

    # Extract base SDG number (remove _Pos/_Neg suffix)
    df['sdg_base'] = df['sdg_target'].str.replace('_Pos', '').str.replace('_Neg', '')

    # Separate Pos and Neg data
    pos_df = df[df['sdg_target'].str.contains('_Pos', na=False)][['sdg_base', 'total_flow']]
    pos_df = pos_df.rename(columns={'total_flow': 'positive_total_flow'})

    neg_df = df[df['sdg_target'].str.contains('_Neg', na=False)][['sdg_base', 'total_flow']]
    neg_df = neg_df.rename(columns={'total_flow': 'negative_total_flow'})

    print(f"  _Pos nodes: {len(pos_df)}")
    print(f"  _Neg nodes: {len(neg_df)}")

    # Merge (outer join to ensure all SDGs are included)
    result_df = pos_df.merge(neg_df, on='sdg_base', how='outer').fillna(0)
    result_df = result_df.rename(columns={'sdg_base': 'sdg_target'})

    print(f"  Aggregated: {len(result_df)} SDG subtargets")

    # Log transformation
    result_df['positive_log'] = np.log1p(result_df['positive_total_flow'])
    result_df['negative_log'] = np.log1p(result_df['negative_total_flow'])

    # Standardization - Linear
    scaler_linear = StandardScaler()
    result_df[['positive_standardized', 'negative_standardized']] = scaler_linear.fit_transform(
        result_df[['positive_total_flow', 'negative_total_flow']]
    )

    # Standardization - Log
    scaler_log = StandardScaler()
    result_df[['positive_log_standardized', 'negative_log_standardized']] = scaler_log.fit_transform(
        result_df[['positive_log', 'negative_log']]
    )

    print(f"  Standardization complete")

    return result_df

def plot_quadrant(df: pd.DataFrame, x_col: str, y_col: str,
                  output_png: str, title_suffix: str):
    """
    Plot quadrant scatter chart with Nature journal style

    Args:
        df: DataFrame with standardized values
        x_col: column name for X-axis
        y_col: column name for Y-axis
        output_png: output PNG file path
        title_suffix: suffix for the title
    """
    print(f"\nPlotting quadrant chart: {title_suffix}...")

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # Nature journal style settings - All font sizes defined here
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Microsoft YaHei']
    plt.rcParams['font.size'] = 12  # Base font size
    plt.rcParams['axes.labelsize'] = 16  # X and Y axis labels
    plt.rcParams['xtick.labelsize'] = 12  # X tick labels
    plt.rcParams['ytick.labelsize'] = 12  # Y tick labels
    plt.rcParams['legend.fontsize'] = 12  # Legend
    plt.rcParams['axes.unicode_minus'] = False

    # Font sizes for other elements
    SCATTER_LABEL_SIZE = 6  # SDG target labels on scatter points (increased for better visibility)
    QUADRANT_LABEL_SIZE = 6  # Quadrant annotation labels
    SPINE_WIDTH = 1.5  # Border line width

    # Create figure with gridspec for marginal density plots
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(6, 6, figure=fig, hspace=0.05, wspace=0.05)

    # Main scatter plot (larger)
    ax = fig.add_subplot(gs[1:, :-1])

    # Density plots (narrower margins)
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax)

    # Calculate point sizes with minimum and maximum
    min_size = 50
    max_size = 500
    total_activity = df['positive_total_flow'] + df['negative_total_flow']

    if total_activity.max() > total_activity.min():
        normalized_activity = (total_activity - total_activity.min()) / (total_activity.max() - total_activity.min())
        sizes = min_size + normalized_activity * (max_size - min_size)
    else:
        sizes = np.full(len(df), min_size)

    # Handle zero values
    sizes = np.where(total_activity == 0, min_size, sizes)

    # No jitter - use original coordinates
    x_plot = df[x_col]
    y_plot = df[y_col]

    # Determine boundaries for color assignment
    if 'standardized' in x_col:
        x_boundary = 0
        y_boundary = 0
    else:
        x_boundary = df[x_col].median()
        y_boundary = df[y_col].median()

    # Color points by quadrant (Nature color palette)
    # Note: Y-axis is inverted, so visually upper=low N, lower=high N
    colors = []
    quadrant_labels = []
    for _, row in df.iterrows():
        if row[x_col] >= x_boundary and row[y_col] < y_boundary:
            # Q1: Upper-right (P high, N low) - BEST
            colors.append('#009E73')  # Green
            quadrant_labels.append('Q1: Pos. Dominant')
        elif row[x_col] < x_boundary and row[y_col] < y_boundary:
            # Q2: Upper-left (P low, N low)
            colors.append('#999999')  # Gray
            quadrant_labels.append('Q2: Dual Low')
        elif row[x_col] < x_boundary and row[y_col] >= y_boundary:
            # Q3: Lower-left (P low, N high) - WORST
            colors.append('#D55E00')  # Red-Orange
            quadrant_labels.append('Q3: Neg. Dominant')
        else:
            # Q4: Lower-right (P high, N high)
            colors.append('#E69F00')  # Orange
            quadrant_labels.append('Q4: Dual High')

    # Prepare data for seaborn
    plot_df = pd.DataFrame({
        'x': x_plot,
        'y': y_plot,
        'size': sizes,
        'color': colors,
        'quadrant': quadrant_labels,
        'sdg_target': df['sdg_target'].values
    })

    # Invert Y-axis (larger N values go downward)
    ax.invert_yaxis()

    # Add coordinate axes - use median for non-standardized, 0 for standardized
    if 'standardized' in x_col:
        x_center = 0
        y_center = 0
    else:
        x_center = df[x_col].median()
        y_center = df[y_col].median()

    ax.axvline(x_center, color='black', linestyle='-', alpha=0.9, linewidth=SPINE_WIDTH, zorder=1)
    ax.axhline(y_center, color='black', linestyle='-', alpha=0.9, linewidth=SPINE_WIDTH, zorder=1)

    # Plot scatter points with seaborn (after axes lines so points are on top)
    sns.scatterplot(
        data=plot_df,
        x='x',
        y='y',
        size='size',
        sizes=(min_size, max_size),
        hue='quadrant',
        palette={'Q1: Pos. Dominant': '#009E73',
                 'Q2: Dual Low': '#999999',
                 'Q3: Neg. Dominant': '#D55E00',
                 'Q4: Dual High': '#E69F00'},
        alpha=0.7,
        edgecolor='white',
        linewidth=1,
        legend=False,
        zorder=3,
        ax=ax
    )

    # Add labels with adjustText if available
    if HAS_ADJUSTTEXT:
        texts = []
        texts_dense = []  # For y < -0.8 points
        for _, row in plot_df.iterrows():
            txt = ax.text(
                row['x'],
                row['y'],
                row['sdg_target'],
                fontsize=SCATTER_LABEL_SIZE,
                alpha=0.9,
                ha='center',
                va='center'
            )
            if row['y'] < -0.8:
                texts_dense.append(txt)
            else:
                texts.append(txt)

        # Adjust text positions to avoid overlap - different parameters for dense region
        # For y < -0.8 (dense region): larger parameters
        if texts_dense:
            adjust_text(
                texts_dense,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                expand_points=(3.5, 3.5),   # 更大的点周围排斥区域
                expand_text=(3.0, 3.0),     # 更大的文本框排斥区域
                force_points=1.2,           # 更强的点排斥力
                force_text=1.8,             # 更强的文本排斥力
                lim=2000,                   # 更多迭代次数
                ax=ax
            )

        # For y >= -0.8 (sparse region): normal parameters
        if texts:
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                expand_points=(2.5, 2.5),
                expand_text=(2.0, 2.0),
                force_points=0.8,
                force_text=1.2,
                lim=1500,
                ax=ax
            )
    else:
        # Fallback: simple annotation
        for _, row in plot_df.iterrows():
            ax.annotate(
                row['sdg_target'],
                (row['x'], row['y']),
                fontsize=SCATTER_LABEL_SIZE,
                alpha=0.7,
                ha='center',
                va='center'
            )

    # Draw histogram plots on top and right margins

    # Top histogram (X distribution)
    ax_top.hist(df[x_col], bins=30, alpha=0.5, color='#555555', edgecolor='black', linewidth=0.5)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.axis('off')

    # Right histogram (Y distribution)
    ax_right.hist(df[y_col], bins=30, orientation='horizontal', alpha=0.5, color='#555555', edgecolor='black', linewidth=0.5)
    ax_right.set_ylim(ax.get_ylim())
    ax_right.axis('off')

    # Axis labels (not bold, Nature style)
    if 'standardized' in x_col:
        # Standardized version
        ax.set_xlabel('Positive Flow P (log-standardized, z-score)')
        ax.set_ylabel('Negative Flow N (log-standardized, z-score, ↓ increasing)')
    elif 'log' in x_col:
        # Log-transformed only (not standardized)
        ax.set_xlabel('Positive Flow P (log-transformed)')
        ax.set_ylabel('Negative Flow N (log-transformed, ↓ increasing)')
    else:
        # Linear (for compatibility)
        ax.set_xlabel('Positive Flow P')
        ax.set_ylabel('Negative Flow N (↓ increasing)')

    # Improve spines (axes borders) - Nature style
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_edgecolor('black')

    # Tick parameters - Nature style
    ax.tick_params(axis='both', which='major', width=0.8, length=4)
    ax.tick_params(axis='both', which='minor', width=0.5, length=2)

    # Grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.2, color='gray', zorder=0)

    # Positions for quadrant labels
    x_max = df[x_col].max()
    x_min = df[x_col].min()
    y_max = df[y_col].max()
    y_min = df[y_col].min()

    q1_x = x_max * 0.65
    q1_y = y_min * 0.65
    q2_x = x_min * 0.65
    q2_y = y_min * 0.65
    q3_x = x_min * 0.65
    q3_y = y_max * 0.65
    q4_x = x_max * 0.65
    q4_y = y_max * 0.65

    # Quadrant annotations (Nature style)
    bbox_props = dict(
        boxstyle='round,pad=0.4',
        facecolor='white',
        edgecolor='gray',
        alpha=0.85,
        linewidth=0.5
    )

    # Determine label text based on whether data is standardized
    if 'standardized' in x_col:
        boundary_text = 'mean'
    else:
        boundary_text = 'median'

    # Quadrant 1 (top-right): P>boundary, N>boundary
    # ax.text(q1_x, q1_y, f'Q1: Dual High\nP>{boundary_text}, N>{boundary_text}',
    #         ha='center', va='center', fontsize=QUADRANT_LABEL_SIZE, weight='bold',
    #         bbox=bbox_props, color='#E69F00')

    # # Quadrant 2 (top-left): P<boundary, N>boundary
    # ax.text(q2_x, q2_y, f'Q2: Neg. Dominant\nP<{boundary_text}, N>{boundary_text}',
    #         ha='center', va='center', fontsize=QUADRANT_LABEL_SIZE, weight='bold',
    #         bbox=bbox_props, color='#D55E00')

    # # Quadrant 3 (bottom-left): P<boundary, N<boundary
    # ax.text(q3_x, q3_y, f'Q3: Dual Low\nP<{boundary_text}, N<{boundary_text}',
    #         ha='center', va='center', fontsize=QUADRANT_LABEL_SIZE, weight='bold',
    #         bbox=bbox_props, color='#999999')

    # # Quadrant 4 (bottom-right): P>boundary, N<boundary
    # ax.text(q4_x, q4_y, f'Q4: Pos. Dominant\nP>{boundary_text}, N<{boundary_text}',
    #         ha='center', va='center', fontsize=QUADRANT_LABEL_SIZE, weight='bold',
    #         bbox=bbox_props, color='#009E73')

    # Save with high resolution (Nature requires 300-600 DPI)
    plt.tight_layout()
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_png}")
    plt.close()

def print_quadrant_stats(df: pd.DataFrame, x_col: str, y_col: str, version_name: str):
    """Print quadrant distribution statistics"""
    # Determine boundary: use 0 for standardized, median for non-standardized
    if 'standardized' in x_col:
        x_boundary = 0
        y_boundary = 0
        boundary_label = "z=0"
    else:
        x_boundary = df[x_col].median()
        y_boundary = df[y_col].median()
        boundary_label = "median"

    q1 = df[(df[x_col] >= x_boundary) & (df[y_col] < y_boundary)]
    q2 = df[(df[x_col] < x_boundary) & (df[y_col] < y_boundary)]
    q3 = df[(df[x_col] < x_boundary) & (df[y_col] >= y_boundary)]
    q4 = df[(df[x_col] >= x_boundary) & (df[y_col] >= y_boundary)]

    print(f"\n{version_name} - Quadrant distribution ({boundary_label} as boundary):")
    print(f"  Q1 (P>={x_boundary:.2f}, N<{y_boundary:.2f} - Pos. Dominant): {len(q1)} items")
    if len(q1) > 0:
        print(f"    Examples: {', '.join(q1.head(5)['sdg_target'].tolist())}")
    print(f"  Q2 (P<{x_boundary:.2f}, N<{y_boundary:.2f} - Dual Low): {len(q2)} items")
    if len(q2) > 0:
        print(f"    Examples: {', '.join(q2.head(5)['sdg_target'].tolist())}")
    print(f"  Q3 (P<{x_boundary:.2f}, N>={y_boundary:.2f} - Neg. Dominant): {len(q3)} items")
    if len(q3) > 0:
        print(f"    Examples: {', '.join(q3.head(5)['sdg_target'].tolist())}")
    print(f"  Q4 (P>={x_boundary:.2f}, N>={y_boundary:.2f} - Dual High): {len(q4)} items")
    if len(q4) > 0:
        print(f"    Examples: {', '.join(q4.head(5)['sdg_target'].tolist())}")

def main():
    """Main function"""
    # Set file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv = os.path.join(base_dir, 'output', 'sdg_summary_statistics.csv')
    output_csv = os.path.join(base_dir, 'output', 'sdg_quadrant_data.csv')
    output_png_log = os.path.join(base_dir, 'output', 'sdg_quadrant_analysis_log.png')
    output_png_log_standardized = os.path.join(base_dir, 'output', 'sdg_quadrant_analysis_log_standardized.png')

    print("=" * 70)
    print("SDG Quadrant Analysis - Log Versions")
    print("=" * 70)

    # 1. Load and process data
    df = load_and_process(input_csv)

    # 2. Export CSV
    print(f"\nExporting data...")
    output_cols = ['sdg_target', 'positive_total_flow', 'negative_total_flow',
                   'positive_log', 'negative_log',
                   'positive_standardized', 'negative_standardized',
                   'positive_log_standardized', 'negative_log_standardized']
    df[output_cols].to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"  Data exported: {output_csv}")
    print(f"  Columns: {output_cols}")
    print(f"  Rows: {len(df)}")

    # 3. Plot quadrant charts
    print("\n" + "=" * 70)
    print("Generating Charts")
    print("=" * 70)

    # Version 1: Log-transformed (not standardized)
    plot_quadrant(df, 'positive_log', 'negative_log',
                  output_png_log, 'Log-transformed')

    # Version 2: Log-transformed and standardized
    plot_quadrant(df, 'positive_log_standardized', 'negative_log_standardized',
                  output_png_log_standardized, 'Log-Standardized')

    # 4. Print statistics
    print("\n" + "=" * 70)
    print("Statistics Summary")
    print("=" * 70)
    print(f"Original P range: [{df['positive_total_flow'].min():.0f}, {df['positive_total_flow'].max():.0f}]")
    print(f"Original N range: [{df['negative_total_flow'].min():.0f}, {df['negative_total_flow'].max():.0f}]")
    print(f"\nLog P range: [{df['positive_log'].min():.2f}, {df['positive_log'].max():.2f}]")
    print(f"Log N range: [{df['negative_log'].min():.2f}, {df['negative_log'].max():.2f}]")
    print(f"\nStandardized values: mean=0, std=1")

    # 5. Quadrant distribution statistics
    # For log-transformed, use median as boundary
    print_quadrant_stats(df, 'positive_log', 'negative_log',
                        'Version 1: Log-transformed')
    print_quadrant_stats(df, 'positive_log_standardized', 'negative_log_standardized',
                        'Version 2: Log-Standardized')

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()
