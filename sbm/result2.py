"""
Result 2 visuals:
- Fig.2a: 4-layer community block matrices (10 communities, ordered: core → positive → risk → base)
- Fig.2b: Quadrant (Result 1) × Community roles (Result 2) heat map

Input files (already produced by sbm.py):
- sbm/output/sbm_graphtool_membership.csv
- sbm/output/sbm_graphtool_communities.csv
- sbm/sdg_cashflow_network.csv
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# Four edge types (consistent with sbm.py)
FLOW_TYPES = ["Pos_to_Pos", "Neg_to_Neg", "Pos_to_Neg", "Neg_to_Pos"]
FLOW_LABELS = {
    "Pos_to_Pos": "Pos→Pos",
    "Neg_to_Neg": "Neg→Neg",
    "Pos_to_Neg": "Pos→Neg",
    "Neg_to_Pos": "Neg→Pos",
}
FLOW_COLORS = {
    "Pos_to_Pos": "#2ca25f",  # Teal
    "Neg_to_Neg": "#d95f02",  # Orange
    "Pos_to_Neg": "#7570b3",  # Purple
    "Neg_to_Pos": "#e63946",  # Red
}

def get_comm_colors(community_order: List[int]) -> Dict[int, tuple]:
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
    return {cid: palette[(cid - 1) % len(palette)] for cid in community_order}

# Community ordering for Fig.2a (core → positive → risk → base)
def load_data() -> Dict[str, pd.DataFrame]:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"

    membership = pd.read_csv(output_dir / "sbm_graphtool_membership.csv")
    communities = pd.read_csv(output_dir / "sbm_graphtool_communities.csv")
    edges = pd.read_csv(base_dir / "sdg_cashflow_network.csv")

    return {
        "membership": membership,
        "communities": communities,
        "edges": edges,
    }


def build_block_matrices(
    edges: pd.DataFrame,
    community_map: Dict[str, int],
    community_order: List[int],
) -> Dict[str, pd.DataFrame]:
    matrices: Dict[str, pd.DataFrame] = {}

    for flow in FLOW_TYPES:
        df = edges[edges["flow_type"] == flow].copy()
        df["src_comm"] = df["source_sdg"].map(community_map)
        df["tgt_comm"] = df["target_sdg"].map(community_map)
        df = df.dropna(subset=["src_comm", "tgt_comm"])

        pivot = (
            df.pivot_table(
                index="src_comm",
                columns="tgt_comm",
                values="cashflow",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(index=community_order, columns=community_order, fill_value=0.0)
        )

        matrices[flow] = pivot

    return matrices


def plot_block_matrices(
    matrices: Dict[str, pd.DataFrame],
    community_order: List[int],
    output_path: Path,
) -> None:
    sns.set_context("paper", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)

    for ax, flow in zip(axes.flat, FLOW_TYPES):
        data = matrices[flow]
        # Log-scale to compress extreme values
        log_data = np.log10(data + 1.0)

        sns.heatmap(
            log_data,
            ax=ax,
            cmap=sns.light_palette(FLOW_COLORS[flow], n_colors=200),
            cbar_kws={"label": "log10(1 + cashflow)"},
            square=True,
            linewidths=0.25,
            linecolor="#f0f0f0",
        )

        ax.set_title(FLOW_LABELS[flow], fontweight="bold", fontsize=12)
        ax.set_xticklabels(community_order, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(community_order, rotation=0, fontsize=9)
        ax.set_xlabel("Target community")
        ax.set_ylabel("Source community")

    # fig.suptitle(
    #     "Fig.2a | 4-layer community interaction (communities renumbered by size: 1..N)",
    #     fontsize=14,
    #     fontweight="bold",
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_quadrant_to_community(
    membership: pd.DataFrame,
    quadrant_name_map: Dict[str, str],
) -> pd.DataFrame:
    """Aggregate total_degree by (quadrant, community) with communities already renamed."""

    membership = membership.copy()
    membership["sdg_base"] = membership["node"].str.rsplit("_", n=1).str[0]

    base_dir = Path(__file__).resolve().parent
    quadrant_labels_df = pd.read_csv(base_dir / "output" / "quadrant_labels.csv")

    merged = membership.merge(
        quadrant_labels_df,
        left_on="sdg_base",
        right_on="sdg_target",
        how="left",
    )
    merged["quadrant_label"] = merged["quadrant"].map(quadrant_name_map)
    merged = merged.dropna(subset=["quadrant_label"])

    grouped = (
        merged.groupby(["quadrant_label", "community"])["total_degree"]
        .sum()
        .reset_index()
        .rename(columns={"community": "community_id"})
    )
    return grouped


def summarize_quadrant_counts(
    membership: pd.DataFrame,
    quadrant_name_map: Dict[str, str],
) -> pd.DataFrame:
    """Aggregate node counts by (quadrant, community) with communities already renamed."""

    membership = membership.copy()
    membership["sdg_base"] = membership["node"].str.rsplit("_", n=1).str[0]

    base_dir = Path(__file__).resolve().parent
    quadrant_labels_df = pd.read_csv(base_dir / "output" / "quadrant_labels.csv")

    merged = membership.merge(
        quadrant_labels_df,
        left_on="sdg_base",
        right_on="sdg_target",
        how="left",
    )
    merged["quadrant_label"] = merged["quadrant"].map(quadrant_name_map)
    merged = merged.dropna(subset=["quadrant_label"])

    grouped = (
        merged.groupby(["quadrant_label", "community"])
        .size()
        .reset_index(name="node_count")
        .rename(columns={"community": "community_id"})
    )
    return grouped


def plot_quadrant_heatmap(
    flow_df: pd.DataFrame,
    quadrant_order: List[str],
    community_order: List[int],
    output_path: Path,
) -> None:
    """
    4xN heatmap:
    rows = quadrants (Result 1)
    cols = community role categories (Result 2)
    values = share (%) of quadrant total_degree in that category
    """
    pivot = (
        flow_df.pivot_table(
            index="quadrant_label",
            columns="community_id",
            values="total_degree",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=quadrant_order, columns=community_order)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(9, 4.5), dpi=200)
    ax = sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "% of quadrant total degree"},
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_xlabel("Community role category")
    ax.set_ylabel("Quadrant (Result 1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_quadrant_count_heatmap(
    count_df: pd.DataFrame,
    quadrant_order: List[str],
    community_order: List[int],
    output_path: Path,
) -> None:
    """
    4xN heatmap of node proportions:
    rows = quadrants (Result 1)
    cols = communities (renamed)
    values = share (%) of nodes in each quadrant belonging to a community
    """
    pivot = (
        count_df.pivot_table(
            index="quadrant_label",
            columns="community_id",
            values="node_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=quadrant_order, columns=community_order)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(9, 4.5), dpi=200)
    ax = sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        cbar_kws={"label": "% of nodes in quadrant"},
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_xlabel("Community role category")
    ax.set_ylabel("Quadrant (Result 1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_network_by_community(
    membership: pd.DataFrame,
    edges: pd.DataFrame,
    community_order: List[int],
    output_path: Path,
    max_edges: int | None = None,
) -> None:
    """
    Visualize the SDG network with nodes colored by (renamed) community.
    Layout: Kamada-Kawai (weight-aware) to spread nodes more evenly; draw all edges by default.
    """
    comm_colors = get_comm_colors(community_order)

    # Use all edges unless capped
    edges_sorted = edges.sort_values("cashflow", ascending=False)
    if max_edges is not None:
        edges_sorted = edges_sorted.head(max_edges)

    G = nx.DiGraph()
    node_comm = membership.set_index("node")["community"].to_dict()
    node_deg = membership.set_index("node")["total_degree"].to_dict()
    node_type = membership.set_index("node")["node_type"].to_dict()
    for node, comm in node_comm.items():
        G.add_node(
            node,
            community=comm,
            degree=node_deg.get(node, 0.0),
            node_type=node_type.get(node, ""),
        )

    for _, row in edges_sorted.iterrows():
        u, v = row["source_sdg"], row["target_sdg"]
        if u in node_comm and v in node_comm:
            G.add_edge(u, v, weight=row["cashflow"])

    if G.number_of_nodes() == 0:
        return

    pos = nx.spectral_layout(G, weight="weight", scale=20.0)

    plt.figure(figsize=(12, 10), dpi=200)
    edges_no_self = [(u, v) for u, v in G.edges() if u != v]  # keep self-loops in data but do not draw
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_no_self,
        alpha=0.16,
        arrows=False,
        width=1.4,
        edge_color="#888888",
    )

    nodes_pos = [n for n in G.nodes() if G.nodes[n].get("node_type") == "Positive"]
    nodes_neg = [n for n in G.nodes() if G.nodes[n].get("node_type") == "Negative"]

    def _draw_nodes(nodelist, marker):
        if not nodelist:
            return
        sizes = [10 + 4 * np.log1p(G.nodes[n].get("degree", 1.0)) for n in nodelist]
        colors = [comm_colors.get(G.nodes[n]["community"], (0.6, 0.6, 0.6)) for n in nodelist]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=sizes,
            node_color=colors,
            node_shape=marker,
            linewidths=0.2,
            edgecolors="white",
        )

    _draw_nodes(nodes_pos, "o")
    _draw_nodes(nodes_neg, "^")

    # Legend for node type + top communities
    type_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", markersize=8, label="Positive SDG"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#888888", markersize=8, label="Negative SDG"),
    ]
    comm_counts = membership["community"].value_counts().loc[community_order]
    top_comm = comm_counts.head(10)
    comm_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=comm_colors[cid],
            markersize=8,
            label=f"Comm {cid}",
        )
        for cid in top_comm.index
    ]
    plt.legend(
        handles=type_handles + comm_handles,
        title="Node type & communities",
        frameon=True,
        loc="upper right",
        fontsize=8,
    )
    plt.axis("off")
    # plt.title("Fig.2c | SDG network colored by community (renamed IDs)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    data = load_data()
    membership = data["membership"]
    communities = data["communities"]
    edges = data["edges"]

    output_dir = Path(__file__).resolve().parent / "visualization"
    output_dir.mkdir(exist_ok=True)

    # Rename communities by size (1..N) to align with visual.py
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
    communities["community"] = communities["community"].map(comm_rename_map)
    comm_flow.index = comm_flow.index.map(comm_rename_map)
    community_order = comm_flow.index.tolist()

    # Mapping from SDG node to renamed community
    node_to_comm = membership.set_index("node")["community"].to_dict()

    matrices = build_block_matrices(edges, node_to_comm, community_order)
    plot_block_matrices(matrices, community_order, output_dir / "fig2a_block_matrices.png")

    # Fig.2b: Quadrant (Result 1) -> Community (Result 2) heatmap (all communities)
    quadrant_name_map = {
        "Q1": "Positive dominant",
        "Q2": "Dual low",
        "Q3": "Negative dominant",
        "Q4": "Dual high",
    }
    quadrant_order = ["Positive dominant", "Dual high", "Negative dominant", "Dual low"]

    flow_df = summarize_quadrant_to_community(membership, quadrant_name_map)
    plot_quadrant_heatmap(flow_df, quadrant_order, community_order, output_dir / "fig2b1_quadrant_flow.png")

    count_df = summarize_quadrant_counts(membership, quadrant_name_map)
    plot_quadrant_count_heatmap(count_df, quadrant_order, community_order, output_dir / "fig2b2_quadrant_nodes.png")

    # Fig.2c: Network layout colored by community
    plot_network_by_community(membership, edges, community_order, output_dir / "fig2c_network_by_community.png")

    print("Saved:")
    print(f"  Fig.2a block matrices -> {output_dir / 'fig2a_block_matrices.png'}")
    print(f"  Fig.2b1 quadrant→community (flow share) -> {output_dir / 'fig2b1_quadrant_flow.png'}")
    print(f"  Fig.2b2 quadrant→community (node share) -> {output_dir / 'fig2b2_quadrant_nodes.png'}")
    print(f"  Fig.2c network -> {output_dir / 'fig2c_network_by_community.png'}")


if __name__ == "__main__":
    main()
