"""
Signed modularity community detection for the SDG cashflow network.

The SDG network has split positive/negative nodes (e.g. ``8.1_Pos`` /
``8.1_Neg``). Edges connecting the same polarity (Pos→Pos or Neg→Neg)
are treated as positive ties while cross-polarity flows (Pos→Neg, Neg→Pos)
are treated as negative ties.  This module implements a Louvain-style local
moving heuristic that maximises the signed modularity objective described by
Gómez et al. (2009):

    Q = Σ_C [ (Σ_in^+ / 2m^+) - (Σ_tot^+ / 2m^+)^2 ]
        - Σ_C [ (Σ_in^- / 2m^-) - (Σ_tot^- / 2m^-)^2 ]

where Σ_in^+/Σ_in^- are the total positive/negative weights inside a community,
Σ_tot^+/Σ_tot^- are the total positive/negative strengths of the community,
and 2m^+, 2m^- are twice the total positive/negative edge weights.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# Flow type to polarity mapping
POSITIVE_FLOWS = {"Pos_to_Pos", "Neg_to_Neg"}
NEGATIVE_FLOWS = {"Pos_to_Neg", "Neg_to_Pos"}


def _normalize_pair(u: str, v: str) -> Tuple[str, str]:
    """Return an undirected key for the pair (order independent)."""
    if u <= v:
        return u, v
    return v, u


@dataclass(frozen=True)
class SignedEdge:
    """Container for aggregated undirected signed weights."""

    source: str
    target: str
    positive: float = 0.0
    negative: float = 0.0


class SignedGraph:
    """Undirected signed graph with helper methods for Louvain."""

    def __init__(self, nodes: Iterable[str], edges: Iterable[SignedEdge]) -> None:
        self.nodes: List[str] = sorted(set(nodes))
        self.edges: List[SignedEdge] = list(edges)
        self.pos_adj: Dict[str, Dict[str, float]] = {n: {} for n in self.nodes}
        self.neg_adj: Dict[str, Dict[str, float]] = {n: {} for n in self.nodes}
        self.pos_degree: Dict[str, float] = Counter()
        self.neg_degree: Dict[str, float] = Counter()
        self.total_pos_weight = 0.0
        self.total_neg_weight = 0.0

        for edge in self.edges:
            u, v, pos_w, neg_w = edge.source, edge.target, edge.positive, edge.negative
            if pos_w > 0:
                self.total_pos_weight += pos_w
                self._add_weight(self.pos_adj, u, v, pos_w)
                if u == v:
                    self.pos_degree[u] += 2.0 * pos_w
                else:
                    self.pos_degree[u] += pos_w
                    self.pos_degree[v] += pos_w

            if neg_w > 0:
                self.total_neg_weight += neg_w
                self._add_weight(self.neg_adj, u, v, neg_w)
                if u == v:
                    self.neg_degree[u] += 2.0 * neg_w
                else:
                    self.neg_degree[u] += neg_w
                    self.neg_degree[v] += neg_w

    @staticmethod
    def _add_weight(adj: Dict[str, Dict[str, float]], u: str, v: str, weight: float) -> None:
        adj[u][v] = adj[u].get(v, 0.0) + weight
        if u != v:
            adj[v][u] = adj[v].get(u, 0.0) + weight

    def neighbors(self, node: str) -> Set[str]:
        pos_neigh = set(self.pos_adj.get(node, {}))
        neg_neigh = set(self.neg_adj.get(node, {}))
        return pos_neigh | neg_neigh

    def num_nodes(self) -> int:
        return len(self.nodes)


def load_signed_graph(
    csv_path: str,
    min_cashflow: float = 0.0,
) -> SignedGraph:
    """Load sdg_cashflow_network.csv and split edges into signed weights."""

    df = pd.read_csv(csv_path)
    df = df[df["cashflow"] > 0]
    if min_cashflow > 0:
        df = df[df["cashflow"] >= min_cashflow]

    nodes: Set[str] = set(df["source_sdg"]).union(df["target_sdg"])
    edge_map: Dict[Tuple[str, str], Dict[str, float]] = {}

    for row in df.itertuples(index=False):
        flow_type = getattr(row, "flow_type")
        src = getattr(row, "source_sdg")
        tgt = getattr(row, "target_sdg")
        weight = float(getattr(row, "cashflow", 0.0))
        if weight <= 0:
            continue
        key = _normalize_pair(src, tgt)
        bucket = edge_map.setdefault(key, {"pos": 0.0, "neg": 0.0})
        if flow_type in POSITIVE_FLOWS:
            bucket["pos"] += weight
        elif flow_type in NEGATIVE_FLOWS:
            bucket["neg"] += weight
        else:
            raise ValueError(f"Unrecognised flow_type '{flow_type}'")

    edges = [
        SignedEdge(source=u, target=v, positive=weights["pos"], negative=weights["neg"])
        for (u, v), weights in edge_map.items()
        if weights["pos"] > 0 or weights["neg"] > 0
    ]

    return SignedGraph(nodes, edges)


def compute_signed_modularity(graph: SignedGraph, partition: Mapping[str, int]) -> float:
    """Compute signed modularity for a given node→community mapping."""

    communities: Dict[int, List[str]] = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)

    q_pos = 0.0
    q_neg = 0.0
    two_m_pos = 2.0 * graph.total_pos_weight
    two_m_neg = 2.0 * graph.total_neg_weight

    if two_m_pos > 0:
        pos_intra = Counter()
        for edge in graph.edges:
            if edge.positive <= 0:
                continue
            if partition[edge.source] == partition[edge.target]:
                pos_intra[partition[edge.source]] += edge.positive
        for comm, nodes in communities.items():
            sum_in = pos_intra.get(comm, 0.0)
            sum_tot = sum(graph.pos_degree.get(node, 0.0) for node in nodes)
            q_pos += (sum_in / two_m_pos) - (sum_tot / two_m_pos) ** 2

    if two_m_neg > 0:
        neg_intra = Counter()
        for edge in graph.edges:
            if edge.negative <= 0:
                continue
            if partition[edge.source] == partition[edge.target]:
                neg_intra[partition[edge.source]] += edge.negative
        for comm, nodes in communities.items():
            sum_in = neg_intra.get(comm, 0.0)
            sum_tot = sum(graph.neg_degree.get(node, 0.0) for node in nodes)
            q_neg += (sum_in / two_m_neg) - (sum_tot / two_m_neg) ** 2

    return q_pos - q_neg


def renumber_partition(partition: Mapping[str, int]) -> Dict[str, int]:
    """Remap community IDs to consecutive integers."""
    mapping: Dict[int, int] = {}
    new_partition: Dict[str, int] = {}
    for node, comm in partition.items():
        if comm not in mapping:
            mapping[comm] = len(mapping)
        new_partition[node] = mapping[comm]
    return new_partition


def local_movement(
    graph: SignedGraph,
    base_partition: Mapping[str, int],
    min_improvement: float,
    random_state: int,
) -> Tuple[Dict[str, int], float]:
    """One Louvain local-moving phase."""

    partition: Dict[str, int] = dict(base_partition)
    current_q = compute_signed_modularity(graph, partition)
    rng = random.Random(random_state)

    improved = True
    iteration = 0
    while improved:
        improved = False
        iteration += 1
        nodes = list(graph.nodes)
        rng.shuffle(nodes)
        for node in nodes:
            current_comm = partition[node]
            best_comm = current_comm
            best_q = current_q
            neighbor_comms = {partition[nbr] for nbr in graph.neighbors(node)}
            new_comm_label = ("new", node, iteration)
            neighbor_comms.add(new_comm_label)

            for comm in neighbor_comms:
                if comm == current_comm:
                    continue
                partition[node] = comm
                trial_q = compute_signed_modularity(graph, partition)
                delta = trial_q - best_q
                if delta > min_improvement:
                    best_q = trial_q
                    best_comm = comm
                partition[node] = current_comm

            if best_comm != current_comm:
                partition[node] = best_comm
                current_q = best_q
                improved = True

    return renumber_partition(partition), current_q


def aggregate_graph(graph: SignedGraph, partition: Mapping[str, int]) -> SignedGraph:
    """Collapse communities to build the next-level graph."""

    new_nodes = sorted(set(partition.values()))
    edge_accumulator: Dict[Tuple[int, int], List[float]] = {}
    for edge in graph.edges:
        src_comm = partition[edge.source]
        tgt_comm = partition[edge.target]
        key = (src_comm, tgt_comm) if src_comm <= tgt_comm else (tgt_comm, src_comm)
        bucket = edge_accumulator.setdefault(key, [0.0, 0.0])
        bucket[0] += edge.positive
        bucket[1] += edge.negative

    new_edges = [
        SignedEdge(source=str(u), target=str(v), positive=weights[0], negative=weights[1])
        for (u, v), weights in edge_accumulator.items()
        if weights[0] > 0 or weights[1] > 0
    ]
    new_node_labels = [str(node) for node in new_nodes]
    return SignedGraph(new_node_labels, new_edges)


def flatten_hierarchy(partitions: Sequence[Mapping[str, int]]) -> Dict[str, int]:
    """Map original nodes to the final communities."""
    if not partitions:
        return {}
    result: Dict[str, int] = dict(partitions[0])
    for level in range(1, len(partitions)):
        higher = partitions[level]
        for node, comm in result.items():
            result[node] = higher[str(comm)]
    return renumber_partition(result)


def signed_louvain(
    graph: SignedGraph,
    max_passes: int = 10,
    min_improvement: float = 1e-7,
    random_state: int = 42,
) -> Tuple[Dict[str, int], float, pd.DataFrame]:
    """Iteratively apply Louvain local moves until convergence.

    Returns:
        final_partition: node to community mapping
        final_q: final modularity value
        convergence_df: DataFrame tracking convergence metrics
    """
    partitions: List[Mapping[str, int]] = []
    level_graph = graph

    # Track convergence
    convergence_stats = []

    for level in range(max_passes):
        base_partition = {node: idx for idx, node in enumerate(level_graph.nodes)}
        moved_partition, q_value = local_movement(
            level_graph,
            base_partition,
            min_improvement=min_improvement,
            random_state=random_state + level,
        )
        partitions.append(moved_partition)
        next_graph = aggregate_graph(level_graph, moved_partition)

        # Record stats for this level
        num_communities = len(set(moved_partition.values()))
        converged = next_graph.num_nodes() == level_graph.num_nodes()
        convergence_stats.append({
            'level': level + 1,
            'nodes': level_graph.num_nodes(),
            'communities': num_communities,
            'modularity': q_value,
            'converged': converged
        })

        if converged:
            break
        level_graph = next_graph

    final_partition = flatten_hierarchy(partitions)
    final_q = compute_signed_modularity(graph, final_partition)
    convergence_df = pd.DataFrame(convergence_stats)

    return final_partition, final_q, convergence_df


def summarize_partition(
    graph: SignedGraph,
    partition: Mapping[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build node-level and community-level summary tables."""
    rows: List[Dict[str, object]] = []
    for node, comm in partition.items():
        node_type = "Positive" if node.endswith("_Pos") else "Negative"
        rows.append(
            {
                "node": node,
                "community": comm,
                "node_type": node_type,
                "positive_degree": graph.pos_degree.get(node, 0.0),
                "negative_degree": graph.neg_degree.get(node, 0.0),
            }
        )
    columns = ["node", "community", "node_type", "positive_degree", "negative_degree"]
    membership_df = pd.DataFrame(rows, columns=columns)
    membership_df = membership_df.dropna(how="all").fillna(
        {"positive_degree": 0.0, "negative_degree": 0.0}
    )
    membership_df = membership_df.sort_values(
        by=["community", "positive_degree"], ascending=[True, False], ignore_index=True
    )

    if membership_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "community",
                "num_nodes",
                "num_positive",
                "num_negative",
                "positive_degree",
                "negative_degree",
                "top_nodes",
            ]
        )
        return membership_df, summary_df

    summary = membership_df.groupby("community").agg(
        num_nodes=("node", "count"),
        num_positive=("node_type", lambda x: (x == "Positive").sum()),
        num_negative=("node_type", lambda x: (x == "Negative").sum()),
        positive_degree=("positive_degree", "sum"),
        negative_degree=("negative_degree", "sum"),
    )
    summary = summary.reset_index()

    def _top_nodes(group: pd.DataFrame, top_n: int = 5) -> str:
        subset = group.head(top_n)
        return ", ".join(subset["node"].tolist())

    top_nodes = (
        membership_df.groupby("community")
        .apply(_top_nodes)
        .rename("top_nodes")
        .reset_index()
    )
    summary_df = summary.merge(top_nodes, on="community", how="left")
    summary_df = summary_df.sort_values(by="positive_degree", ascending=False)
    return membership_df, summary_df


def plot_convergence(convergence_df: pd.DataFrame, output_path: str = '../../output/convergence.png') -> None:
    """Plot convergence metrics.

    Args:
        convergence_df: DataFrame with convergence statistics
        output_path: Path to save the plot
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Modularity across levels
    ax = axes[0]
    ax.plot(convergence_df['level'], convergence_df['modularity'],
            'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.set_xlabel('Louvain Level', fontsize=11)
    ax.set_ylabel('Signed Modularity Q', fontsize=11)
    ax.set_title('Modularity Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Number of communities across levels
    ax = axes[1]
    ax.plot(convergence_df['level'], convergence_df['communities'],
            's-', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Louvain Level', fontsize=11)
    ax.set_ylabel('Number of Communities', fontsize=11)
    ax.set_title('Community Count Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Signed modularity Louvain analysis for the SDG cashflow network.",
    )
    parser.add_argument(
        "--network-file",
        default="../../output/sdg_cashflow_network.csv",
        help="Path to sdg_cashflow_network.csv (default: output/sdg_cashflow_network.csv)",
    )
    parser.add_argument(
        "--min-cashflow",
        type=float,
        default=0.0,
        help="Ignore edges with cashflow below this threshold.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=50,
        help="Maximum number of Louvain aggregation passes.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=1e-7,
        help="Minimum modularity improvement required for node moves.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for node ordering.",
    )
    parser.add_argument(
        "--output-membership",
        default='../../output/sdg_signed_membership.csv ',
        help="Optional path to save node-level membership CSV.",
    )
    parser.add_argument(
        "--output-summary",
        default='../../output/sdg_signed_communities.csv',
        help="Optional path to save aggregated community summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_signed_graph(args.network_file, min_cashflow=args.min_cashflow)

    print("=== Signed Louvain Community Detection ===")
    print(f"Nodes: {graph.num_nodes():,}")
    print(f"Positive edges (weight sum): {graph.total_pos_weight:,.2f}")
    print(f"Negative edges (weight sum): {graph.total_neg_weight:,.2f}")

    partition, modularity, convergence_df = signed_louvain(
        graph,
        max_passes=args.max_passes,
        min_improvement=args.min_improvement,
        random_state=args.seed,
    )

    print(f"\nSigned modularity Q: {modularity:.6f}")
    print(f"Converged in {len(convergence_df)} levels")

    # Print convergence details
    print("\nConvergence details:")
    print(convergence_df.to_string(index=False))

    membership_df, summary_df = summarize_partition(graph, partition)
    if summary_df.empty:
        print("\nNo communities detected (check the network input).")
    else:
        print("\nTop communities by positive degree:")
        print(summary_df.head(10).to_string(index=False))

    if args.output_membership:
        membership_df.to_csv(args.output_membership, index=False)
        print(f"\nSaved membership table to {args.output_membership}")
    if args.output_summary:
        summary_df.to_csv(args.output_summary, index=False)
        print(f"Saved summary table to {args.output_summary}")

    # Save convergence data and plot
    convergence_csv = args.output_summary.replace('communities.csv', 'convergence.csv')
    convergence_df.to_csv(convergence_csv, index=False)
    print(f"Saved convergence table to {convergence_csv}")

    convergence_plot = args.output_summary.replace('communities.csv', 'convergence.png')
    plot_convergence(convergence_df, convergence_plot)


if __name__ == "__main__":
    main()
