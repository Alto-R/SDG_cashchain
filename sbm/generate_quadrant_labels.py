"""
Generate quadrant_labels.csv from membership data.

This script assigns each SDG target to a quadrant based on the aggregated
positive and negative flow volumes.

Quadrant definitions:
- Q1: Positive dominant (P >= boundary, N < boundary)
- Q2: Dual low (P < boundary, N < boundary)
- Q3: Negative dominant (P < boundary, N >= boundary)
- Q4: Dual high (P >= boundary, N >= boundary)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_quadrant_labels():
    """Generate quadrant labels for each SDG target"""

    # Load membership data
    base_dir = Path(__file__).resolve().parent
    membership_path = base_dir / "output" / "sbm_graphtool_membership.csv"
    output_path = base_dir / "output" / "quadrant_labels.csv"

    print("Loading membership data...")
    df = pd.read_csv(membership_path)

    # Extract base SDG target (remove _Pos/_Neg suffix)
    df['sdg_base'] = df['node'].str.rsplit('_', n=1).str[0]
    df['node_polarity'] = df['node'].str.rsplit('_', n=1).str[1]

    # Aggregate positive and negative flows per SDG target
    pos_flow = df[df['node_polarity'] == 'Pos'].groupby('sdg_base')['total_degree'].sum()
    neg_flow = df[df['node_polarity'] == 'Neg'].groupby('sdg_base')['total_degree'].sum()

    # Combine into a single dataframe
    quadrant_df = pd.DataFrame({
        'sdg_target': pos_flow.index,
        'positive_flow': pos_flow.values,
        'negative_flow': neg_flow.reindex(pos_flow.index, fill_value=0).values
    })

    # Log-transform and standardize (following quadrant.py approach)
    quadrant_df['positive_log'] = np.log1p(quadrant_df['positive_flow'])
    quadrant_df['negative_log'] = np.log1p(quadrant_df['negative_flow'])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    standardized = scaler.fit_transform(quadrant_df[['positive_log', 'negative_log']])
    quadrant_df['positive_standardized'] = standardized[:, 0]
    quadrant_df['negative_standardized'] = standardized[:, 1]

    # Assign quadrants using standardized values (boundary = 0)
    def assign_quadrant(row):
        p = row['positive_standardized']
        n = row['negative_standardized']

        if p >= 0 and n < 0:
            return 'Q1'  # Positive dominant
        elif p < 0 and n < 0:
            return 'Q2'  # Dual low
        elif p < 0 and n >= 0:
            return 'Q3'  # Negative dominant
        else:  # p >= 0 and n >= 0
            return 'Q4'  # Dual high

    quadrant_df['quadrant'] = quadrant_df.apply(assign_quadrant, axis=1)

    # Save to CSV
    output_df = quadrant_df[['sdg_target', 'quadrant']]
    output_df.to_csv(output_path, index=False)

    print(f"\nQuadrant distribution:")
    print(quadrant_df['quadrant'].value_counts().sort_index())
    print(f"\nSaved: {output_path}")
    print(f"Total SDG targets: {len(output_df)}")

    return output_df

if __name__ == '__main__':
    generate_quadrant_labels()
