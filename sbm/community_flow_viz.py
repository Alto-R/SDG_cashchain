"""
Visualize community flow destinations.
Shows where cashflows from each community go (within-community vs to other communities).
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

def load_data():
    """Load membership and edge data."""
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"

    membership = pd.read_csv(output_dir / "sbm_graphtool_membership.csv")
    edges = pd.read_csv(base_dir / "sdg_cashflow_network.csv")

    return membership, edges


def compute_community_flow_by_type(membership, edges, top_n_communities=10):
    """
    Compute where cashflows from each community go, broken down by flow type.
    Returns DataFrame with (src_comm, flow_type, tgt_comm, cashflow) for each combination.
    """
    # Rename communities by size (consistent with result2.py)
    if "total_degree" not in membership.columns:
        membership["total_degree"] = (
            membership["pos_pos_degree"]
            + membership["neg_neg_degree"]
            + membership["pos_neg_degree"]
            + membership["neg_pos_degree"]
        )

    comm_flow = membership.groupby("community")["total_degree"].sum().sort_values(ascending=False)
    comm_rename_map = {old_id: new_id for new_id, old_id in enumerate(comm_flow.index, start=1)}
    membership["community"] = membership["community"].map(comm_rename_map)

    # Also rename the comm_flow index to match the new community IDs
    comm_flow.index = comm_flow.index.map(comm_rename_map)

    # Create node to community mapping
    node_to_comm = membership.set_index("node")["community"].to_dict()

    # Add community info to edges
    edges = edges.copy()
    edges["src_comm"] = edges["source_sdg"].map(node_to_comm)
    edges["tgt_comm"] = edges["target_sdg"].map(node_to_comm)
    edges = edges.dropna(subset=["src_comm", "tgt_comm"])

    # Compute flow by (src_comm, flow_type, tgt_comm)
    flow_by_type = (
        edges.groupby(["src_comm", "flow_type", "tgt_comm"])["cashflow"]
        .sum()
        .reset_index()
    )

    # Filter to top N communities
    top_communities = list(comm_flow.index[:top_n_communities])
    flow_by_type = flow_by_type[
        flow_by_type["src_comm"].isin(top_communities) &
        flow_by_type["tgt_comm"].isin(top_communities)
    ]

    return flow_by_type, top_communities


def get_comm_colors(community_order):
    """Get community colors matching result2.py color scheme."""
    custom_hex = [
        "#012A61",
        "#015696",
        "#4596CD",
        "#89CAEA",
        "#BBE6FA",
        "#F1EEED",
        "#FCCDC9",
        "#EE9D9F",
        "#DE6A69",
        "#982B2D",
    ][::-1]
    palette = [mcolors.to_rgb(c) for c in custom_hex]
    colors = {cid: mcolors.to_hex(palette[(cid - 1) % len(palette)]) for cid in community_order}
    return colors


def plot_community_flow_by_type(flow_by_type, community_order, output_path):
    """
    Plot 2x2 grid of heatmaps showing inter-community cashflow by flow type.
    Each subplot shows one flow type (Pos→Pos, Neg→Neg, Pos→Neg, Neg→Pos).
    Rows: Source community
    Columns: Target community
    Color: Absolute cashflow amount (log-scaled)
    """
    # Flow type mapping and colors
    flow_types = ["Pos_to_Pos", "Neg_to_Neg", "Pos_to_Neg", "Neg_to_Pos"]
    flow_labels = {
        "Pos_to_Pos": "Pos→Pos",
        "Neg_to_Neg": "Neg→Neg",
        "Pos_to_Neg": "Pos→Neg",
        "Neg_to_Pos": "Neg→Pos",
    }
    flow_colors = {
        "Pos_to_Pos": "#012A61",  # Dark blue
        "Neg_to_Neg": "#982B2D",  # Dark red
        "Pos_to_Neg": "#982B2D",  # Light red
        "Neg_to_Pos": "#012A61",  # Light blue
    }

    sns.set_context("paper", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=200)

    for ax, flow_type in zip(axes.flat, flow_types):
        # Filter data for this flow type
        flow_data = flow_by_type[flow_by_type["flow_type"] == flow_type]

        # Create matrix
        pivot = flow_data.pivot_table(
            index="src_comm",
            columns="tgt_comm",
            values="cashflow",
            fill_value=0.0
        ).reindex(index=community_order, columns=community_order, fill_value=0.0)

        # Use log scale for better visualization
        log_flow = np.log10(pivot + 1)

        # Create heatmap with flow-type-specific color
        sns.heatmap(
            log_flow,
            annot=False,
            cmap=sns.light_palette(flow_colors[flow_type], as_cmap=True),
            cbar_kws={"label": "log10(1 + cashflow)"},
            linewidths=0.3,
            linecolor="white",
            square=True,
            ax=ax,
        )

        # Formatting
        ax.set_title(flow_labels[flow_type], fontweight="bold", fontsize=14, pad=10)
        ax.set_xlabel("Target Community", fontweight="bold", fontsize=11)
        ax.set_ylabel("Source Community", fontweight="bold", fontsize=11)
        ax.set_xticklabels(community_order, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(community_order, rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved community flow by type heatmap: {output_path}")


def print_flow_summary_by_type(flow_by_type, community_order):
    """Print summary of flow patterns by type."""
    print("\nFlow destination summary by type:")
    print("=" * 80)

    flow_types = ["Pos_to_Pos", "Neg_to_Neg", "Pos_to_Neg", "Neg_to_Pos"]
    flow_labels = {
        "Pos_to_Pos": "Pos→Pos",
        "Neg_to_Neg": "Neg→Neg",
        "Pos_to_Neg": "Pos→Neg",
        "Neg_to_Pos": "Neg→Pos",
    }

    for src_comm in community_order:
        print(f"\nCommunity {int(src_comm)}:")

        src_flows = flow_by_type[flow_by_type["src_comm"] == src_comm]

        for flow_type in flow_types:
            type_flows = src_flows[src_flows["flow_type"] == flow_type]
            total = type_flows["cashflow"].sum()

            if total > 0:
                print(f"  {flow_labels[flow_type]}:")
                print(f"    Total: {total:12,.0f}")

                # Show top 3 destinations for this flow type
                top_dest = type_flows.nlargest(3, "cashflow")
                if len(top_dest) > 0:
                    print(f"    Top destinations:")
                    for _, row in top_dest.iterrows():
                        tgt = int(row["tgt_comm"])
                        amt = row["cashflow"]
                        pct = (amt / total * 100) if total > 0 else 0
                        print(f"      → Comm {tgt}: {amt:12,.0f} ({pct:5.1f}%)")


def main():
    membership, edges = load_data()
    flow_by_type, community_order = compute_community_flow_by_type(membership, edges, top_n_communities=10)

    # Print summary
    print_flow_summary_by_type(flow_by_type, community_order)

    # Save visualization
    output_dir = Path(__file__).resolve().parent / "visualization"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "community_flow_by_type.png"

    plot_community_flow_by_type(flow_by_type, community_order, output_path)

    # Save statistics to CSV
    csv_path = output_dir / "community_flow_by_type.csv"
    flow_by_type.to_csv(csv_path, index=False)
    print(f"\nSaved statistics: {csv_path}")


if __name__ == "__main__":
    main()