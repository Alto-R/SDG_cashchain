"""
4-Layer Nested SBM using graph-tool for SDG cashflow network.

This implementation uses graph-tool's state-of-the-art Minimum Description Length
(MDL) based inference for community detection in directed signed networks.

Key features:
1. 4-layer edge treatment: Pos→Pos, Neg→Neg, Pos→Neg, Neg→Pos (each flow type as separate layer)
2. Real-valued edge weights: Cashflow amounts as covariates
3. Degree correction: Accounts for heterogeneous node degrees (essential for hub nodes)
4. Nested hierarchy: Discovers multi-level community structure

Reference:
Peixoto, T. P. (2014). "Hierarchical block structures and high-resolution model
selection in large networks." Physical Review X, 4(011047).
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import graph_tool.all as gt

try:
    import graph_tool.all as gt
    GRAPH_TOOL_AVAILABLE = True
except ImportError:
    GRAPH_TOOL_AVAILABLE = False
    print("\n" + "="*70)
    print("ERROR: graph-tool not installed")
    print("="*70)
    print("\nInstallation instructions:")
    print("  conda install -c conda-forge graph-tool")
    print("  OR visit: https://graph-tool.skewed.de/")
    print("="*70 + "\n")


# Flow type to layer mapping (4-layer model)
FLOW_LAYER = {
    'Pos_to_Pos': 0,  # Positive reinforcement
    'Neg_to_Neg': 1,  # Negative reinforcement
    'Pos_to_Neg': 2,  # Trade-off (Pos reduces Neg)
    'Neg_to_Pos': 3,  # Trade-off (Neg reduces Pos)
}

# Layer names for visualization and output
LAYER_NAMES = {
    0: 'Pos→Pos',
    1: 'Neg→Neg',
    2: 'Pos→Neg',
    3: 'Neg→Pos',
}


def get_layer_id(flow_type: str) -> int:
    """
    Map flow_type to edge layer.

    Pos_to_Pos → Layer 0 (Positive reinforcement)
    Neg_to_Neg → Layer 1 (Negative reinforcement)
    Pos_to_Neg → Layer 2 (Trade-off: Pos reduces Neg)
    Neg_to_Pos → Layer 3 (Trade-off: Neg reduces Pos)
    """
    if flow_type in FLOW_LAYER:
        return FLOW_LAYER[flow_type]
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")


def load_and_build_graph(
    csv_path: str,
    min_cashflow: float = 0.0
) -> Tuple[gt.Graph, Dict[int, str], Dict[str, int]]:
    """Load CSV and build graph-tool graph with edge layers and weights."""

    print("="*70)
    print("Loading data and building graph")
    print("="*70)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Raw data: {len(df)} edges")

    df = df[df['cashflow'] > 0]
    if min_cashflow > 0:
        df = df[df['cashflow'] >= min_cashflow]
        print(f"After filtering (cashflow >= {min_cashflow}): {len(df)} edges")

    # Create node mapping
    all_nodes = sorted(set(df['source_sdg']).union(set(df['target_sdg'])))
    node_map = {name: i for i, name in enumerate(all_nodes)}
    id_map = {i: name for name, i in node_map.items()}

    print(f"Nodes: {len(all_nodes)}")

    # Build graph
    g = gt.Graph(directed=True)
    g.add_vertex(len(all_nodes))

    # Edge properties
    ep_layer = g.new_edge_property("int")      # Layer index
    ep_weight = g.new_edge_property("double")  # Cashflow amount

    # Add edges
    layer_counts = {}  # Track counts per layer

    for _, row in df.iterrows():
        u = node_map[row['source_sdg']]
        v = node_map[row['target_sdg']]
        e = g.add_edge(u, v)

        layer = get_layer_id(row['flow_type'])
        ep_layer[e] = layer
        ep_weight[e] = row['cashflow']

        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    # Register properties
    g.ep["layer"] = ep_layer
    g.ep["weight"] = ep_weight

    # Add node names for reference
    vp_name = g.new_vertex_property("string")
    for i, name in id_map.items():
        vp_name[g.vertex(i)] = name
    g.vp["name"] = vp_name

    print(f"\nEdge breakdown:")
    for layer_id in range(4):
        count = layer_counts.get(layer_id, 0)
        name = LAYER_NAMES[layer_id]
        print(f"  Layer {layer_id} ({name}): {count:5d} ({count/g.num_edges()*100:.1f}%)")
    print(f"  Total edges:                  {g.num_edges():5d}")

    return g, id_map, node_map


def fit_nested_sbm(
    g: gt.Graph,
    deg_corr: bool = True,
    use_weights: bool = True,
    use_layered: bool = True,
    mcmc_steps: int = 100,
    verbose: bool = True,
    random_seed: int = 42
) -> gt.NestedBlockState:
    """
    Fit nested SBM using MDL minimization.
    """

    # Set random seed for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    gt.seed_rng(random_seed)

    if verbose:
        print("\n" + "="*70)
        print("Fitting Nested SBM (Degree-Corrected)")
        print("="*70)
        print(f"Random seed:       {random_seed}")
        print(f"Degree correction: {deg_corr}")
        print(f"Use edge weights:  {use_weights}")
        print(f"Use layered mode:  {use_layered}")
        print(f"MCMC steps:        {mcmc_steps}")

    # 1. 基础参数
    state_args = {
        "deg_corr": deg_corr,
    }

    # 2. 权重处理 (Real-valued covariates)
    if use_weights:
        state_args["recs"] = [g.ep["weight"]]
        state_args["rec_types"] = ["real-exponential"]
        if verbose:
            print("Using cashflow amounts as real-exponential covariates")

    # 3. 分层处理 (Layered SBM for Signs) - 关键修正区域
    if use_layered:
        # --- 修正开始 ---
        # 必须显式指定 base_type 为 LayeredBlockState
        # 否则 graph-tool 默认使用 BlockState，它不接受 'ec' 和 'layers'
        state_args["base_type"] = gt.LayeredBlockState
        
        # 传入层级所需的参数
        state_args["ec"] = g.ep["layer"]
        state_args["layers"] = True
        # --- 修正结束 ---
        
        if verbose:
            print("Using 4-layer SBM: Pos→Pos (0), Neg→Neg (1), Pos→Neg (2), Neg→Pos (3)")

    # 4. 最小化描述长度 (Inference)
    if verbose:
        print("\nMinimizing description length (this may take a few minutes)...")

    try:
        state = gt.minimize_nested_blockmodel_dl(
            g,
            state_args=state_args,
        )
    except Exception as e:
        # 错误处理逻辑
        if use_layered:
            print(f"\nWarning: Layered SBM failed ({e})")
            print("Falling back to standard degree-corrected SBM without layers...")
            # 清除分层参数重新尝试
            state_args.pop("ec", None)
            state_args.pop("layers", None)
            state_args.pop("base_type", None) # 也要移除 base_type
            state = gt.minimize_nested_blockmodel_dl(
                g,
                state_args=state_args,
            )
        else:
            raise

    # 5. MCMC 优化 (Optimization)
    if mcmc_steps > 0:
        if verbose:
            print(f"\nRunning {mcmc_steps} MCMC equilibration steps...")

        # 建议先用 beta=1.0 热身，再用 beta=np.inf 冷却
        # 这里为了稳健性，直接使用默认的 beta=np.inf (贪婪优化)
        for i in range(mcmc_steps):
            ret = state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

            if verbose and (i + 1) % 100 == 0: # 每100步打印一次
                dl = state.entropy()
                print(f"  Step {i+1:4d}/{mcmc_steps}: DL = {dl:10.2f}, ΔDL = {ret[0]:8.2f}")

    # 打印结果摘要
    if verbose:
        print("\n" + "="*70)
        print("Hierarchical structure")
        print("="*70)
        L = len(state.get_levels())
        print(f"Number of levels: {L}")
        
        # 验证是否真的使用了分层模型
        base_level = state.get_levels()[0]
        is_layered = isinstance(base_level, gt.LayeredBlockState)
        print(f"Base model type: {type(base_level).__name__} (Layered: {is_layered})")

        bs = state.get_bs()
        for level in range(L):
            n_blocks = len(set(bs[level]))
            print(f"  Level {level}: {n_blocks:3d} communities")

        dl = state.entropy()
        print(f"\nMinimum Description Length: {dl:.2f}")

    return state


def extract_results(
    state: gt.NestedBlockState,
    g: gt.Graph,
    id_map: Dict[int, str],
    node_map: Dict[str, int],
    level: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Extract community assignments and statistics.

    Returns:
        membership_df: Node-level assignments (with renumbered communities)
        summary_df: Community-level statistics
        comm_id_map: Mapping from original community ID to renumbered ID (1-based, sorted by total_degree)
    """

    # Get partition at specified level
    bs = state.get_bs()
    block_assignment = bs[level]

    # Build node-level table with original community IDs
    rows = []
    for node_id, comm_id in enumerate(block_assignment):
        node_name = id_map[node_id]
        node_type = "Positive" if node_name.endswith("_Pos") else "Negative"

        v = g.vertex(node_id)

        # Compute weighted degrees by layer (4-layer model)
        layer_degrees = {i: 0.0 for i in range(4)}
        for e in v.out_edges():
            layer_degrees[g.ep["layer"][e]] += g.ep["weight"][e]
        for e in v.in_edges():
            layer_degrees[g.ep["layer"][e]] += g.ep["weight"][e]

        total_deg = sum(layer_degrees.values())

        rows.append({
            "node": node_name,
            "node_id": node_id,  # Keep original node ID
            "community_original": int(comm_id),
            "node_type": node_type,
            "pos_pos_degree": layer_degrees[0],
            "neg_neg_degree": layer_degrees[1],
            "pos_neg_degree": layer_degrees[2],
            "neg_pos_degree": layer_degrees[3],
            "total_degree": total_deg,
        })

    temp_df = pd.DataFrame(rows)

    # Compute community total degrees and create renumbering map
    comm_degrees = temp_df.groupby("community_original")["total_degree"].sum().reset_index()
    comm_degrees = comm_degrees.sort_values(by="total_degree", ascending=False, ignore_index=True)

    # Create mapping: original_comm_id -> new_comm_id (1-based, sorted by degree)
    comm_id_map = {
        int(old_id): new_id + 1  # Start from 1
        for new_id, old_id in enumerate(comm_degrees["community_original"])
    }

    # Apply renumbering
    temp_df["community"] = temp_df["community_original"].map(comm_id_map)

    # Sort: by new community (ascending), then by original node_id (ascending)
    membership_df = temp_df.sort_values(
        by=["community", "node_id"],
        ascending=[True, True],
        ignore_index=True
    )

    # Drop temporary columns for output
    membership_df_output = membership_df.drop(columns=["community_original", "node_id"])

    # Build community-level summary
    summary = membership_df_output.groupby("community").agg(
        num_nodes=("node", "count"),
        num_positive=("node_type", lambda x: (x == "Positive").sum()),
        num_negative=("node_type", lambda x: (x == "Negative").sum()),
        pos_pos_degree=("pos_pos_degree", "sum"),
        neg_neg_degree=("neg_neg_degree", "sum"),
        pos_neg_degree=("pos_neg_degree", "sum"),
        neg_pos_degree=("neg_pos_degree", "sum"),
        total_degree=("total_degree", "sum"),
    ).reset_index()

    # Add top nodes per community
    def _top_nodes(group, n=5):
        return ", ".join(group.head(n)["node"].tolist())

    top_nodes = (
        membership_df_output.groupby("community")
        .apply(_top_nodes, include_groups=False)
        .rename("top_nodes")
        .reset_index()
    )

    summary_df = summary.merge(top_nodes, on="community", how="left")
    # Already sorted by community ID (which is now ordered by total_degree descending)

    # === Flow-to-community profile (4 layers × num_comms) ===
    num_comms = membership_df_output["community"].nunique()

    # Node ordering and lookup helpers
    node_list = membership_df_output["node"].tolist()
    node_index = {name: i for i, name in enumerate(node_list)}
    node2comm = dict(zip(membership_df_output["node"], membership_df_output["community"]))

    # (N_nodes, 4, num_comms) tensor accumulating outgoing flow by target community
    flow_profile = np.zeros((len(node_list), 4, num_comms), dtype=float)

    for e in g.edges():
        u = int(e.source())
        v = int(e.target())

        u_name = g.vp["name"][g.vertex(u)]
        v_name = g.vp["name"][g.vertex(v)]

        if u_name not in node_index or v_name not in node2comm:
            continue

        i = node_index[u_name]
        target_comm = node2comm[v_name]  # 1..num_comms
        layer = g.ep["layer"][e]         # 0..3
        w = g.ep["weight"][e]

        flow_profile[i, layer, target_comm - 1] += w

    # Flatten to 4×num_comms and row-normalize to focus on destination pattern
    FP = flow_profile.reshape(len(node_list), -1)
    row_sum = FP.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    FP_norm = FP / row_sum

    col_names = [
        f"flow_layer{layer}_toC{c}"
        for layer in range(4)
        for c in range(1, num_comms + 1)
    ]

    flow_df = pd.DataFrame(FP_norm, columns=col_names)
    flow_df.insert(0, "node", node_list)

    membership_df_output = membership_df_output.merge(flow_df, on="node", how="left")

    return membership_df_output, summary_df, comm_id_map


def save_all_levels(
    state: gt.NestedBlockState,
    id_map: Dict[int, str],
    output_prefix: str
) -> None:
    """Save all hierarchical levels to separate CSV files."""

    print("\n" + "="*70)
    print("Saving hierarchical partitions")
    print("="*70)

    bs = state.get_bs()

    for level in range(len(bs)):
        block_assignment = bs[level]

        # For level > 0, nodes are actually communities from level-1
        if level == 0:
            # Level 0: actual nodes
            data = [
                {"node": id_map[node_id], "community": int(comm_id)}
                for node_id, comm_id in enumerate(block_assignment)
            ]
        else:
            # Level > 0: meta-communities
            data = [
                {"lower_community": node_id, "upper_community": int(comm_id)}
                for node_id, comm_id in enumerate(block_assignment)
            ]

        df = pd.DataFrame(data)
        output_path = f"{output_prefix}_level{level}.csv"
        df.to_csv(output_path, index=False)

        print(f"  Level {level}: {len(set(block_assignment)):3d} communities → {output_path}")


def plot_layered_matrix(
    state: gt.NestedBlockState,
    g: gt.Graph,
    comm_id_map: Dict[int, int],
    id_map: Dict[int, str],
    output_path: str,
    title: str = "SDG Cashflow Network: 4-Layer Community Structure"
) -> None:
    """
    Plot 4-layer sorted adjacency matrix in 2x2 grid.

    Four panels showing:
    - Top-left: Pos→Pos (Positive reinforcement)
    - Top-right: Neg→Neg (Negative reinforcement)
    - Bottom-left: Pos→Neg (Trade-off: Pos reduces Neg)
    - Bottom-right: Neg→Pos (Trade-off: Neg reduces Pos)

    Nodes are sorted by community (renumbered by total degree), with node names labeled.
    """
    import matplotlib.pyplot as plt

    print(f"  Generating 4-layer matrix visualization...")

    # Get original community assignment and create renumbered version
    b_original = state.get_bs()[0]  # Original community assignment
    b_renumbered = [comm_id_map[b_original[i]] for i in range(g.num_vertices())]

    # Sort nodes by: renumbered community (ascending), then original node ID (ascending)
    node_order = sorted(range(g.num_vertices()), key=lambda i: (b_renumbered[i], i))
    order_map = {original: new for new, original in enumerate(node_order)}

    N = g.num_vertices()

    # Compute community boundaries and labels for grid lines
    sorted_b = [b_renumbered[i] for i in node_order]
    boundaries = []
    comm_labels = []
    comm_positions = []

    current_comm = sorted_b[0]
    start_pos = 0

    for i in range(1, N):
        if sorted_b[i] != sorted_b[i-1]:
            # Record boundary
            boundaries.append(i)
            # Record community label and position (center of the community block)
            comm_labels.append(f"C{current_comm}")
            comm_positions.append((start_pos + i) / 2)
            # Move to next community
            current_comm = sorted_b[i]
            start_pos = i

    # Don't forget the last community
    comm_labels.append(f"C{current_comm}")
    comm_positions.append((start_pos + N) / 2)

    # Get node names in sorted order
    node_names = [id_map[i] for i in node_order]

    # Initialize 4 matrices
    matrices = {i: np.full((N, N), np.nan) for i in range(4)}

    # Fill matrices
    for e in g.edges():
        u, v = int(e.source()), int(e.target())
        x, y = order_map[u], order_map[v]
        layer = g.ep["layer"][e]
        weight = g.ep["weight"][e]
        plot_val = np.log1p(weight)
        matrices[layer][x, y] = plot_val

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 18), dpi=150)

    # Color maps for each layer
    cmaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds]
    titles = [
        "Layer 0: Pos→Pos\nPositive Reinforcement",
        "Layer 1: Neg→Neg\nNegative Reinforcement",
        "Layer 2: Pos→Neg\nTrade-off (Pos reduces Neg)",
        "Layer 3: Neg→Pos\nTrade-off (Neg reduces Pos)",
    ]

    for idx, (ax, cmap, layer_title) in enumerate(zip(axes.flat, cmaps, titles)):
        cmap_copy = cmap.copy()
        cmap_copy.set_bad("white")

        im = ax.imshow(matrices[idx], cmap=cmap_copy, interpolation='nearest', aspect='auto')
        ax.set_title(layer_title, fontsize=11, fontweight='bold')

        # Set axis labels
        ax.set_xlabel("Target SDG", fontsize=10, fontweight='bold')
        if idx % 2 == 0:
            ax.set_ylabel("Source SDG", fontsize=10, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, label="log(1 + Cashflow)", fraction=0.046, pad=0.04)

        # Draw community boundaries (thick lines)
        ax.spines[:].set_color('black')
        ax.spines[:].set_linewidth(1.5)
        for pos in boundaries:
            ax.axhline(pos - 0.5, color='red', linewidth=1.5, alpha=0.7)
            ax.axvline(pos - 0.5, color='red', linewidth=1.5, alpha=0.7)

        # Add community labels at the top and right
        ax2 = ax.twiny()  # Create secondary x-axis for community labels
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(comm_positions)
        ax2.set_xticklabels(comm_labels, fontsize=9, fontweight='bold', color='red')
        ax2.tick_params(axis='x', length=0)

        ax3 = ax.twinx()  # Create secondary y-axis for community labels
        ax3.set_ylim(ax.get_ylim())
        ax3.set_yticks(comm_positions)
        ax3.set_yticklabels(comm_labels, fontsize=9, fontweight='bold', color='red')
        ax3.tick_params(axis='y', length=0)

        # Set node names as tick labels (show every Nth node to avoid crowding)
        step = max(1, N // 20)  # Show ~20 labels
        tick_positions = list(range(0, N, step))
        tick_labels = [node_names[i] for i in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6, ha='center')
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6, va='center')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved: {output_path}")
    print(f"    Communities labeled: {', '.join(comm_labels)}")


def plot_results(
    state: gt.NestedBlockState,
    g: gt.Graph,
    comm_id_map: Dict[int, int],
    id_map: Dict[int, str],
    output_dir: str
) -> None:
    """Generate visualization plots."""

    print("\n" + "="*70)
    print("Generating visualizations")
    print("="*70)

    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. 4-layer matrix plot (PRIMARY VISUALIZATION - Nature quality)
    matrix_path = f"{output_dir}/sbm_4layer_matrix.png"
    try:
        plot_layered_matrix(state, g, comm_id_map, id_map, matrix_path)
    except Exception as e:
        print(f"  Warning: Could not generate layered matrix plot: {e}")
        import traceback
        traceback.print_exc()

    # 2. Standard block matrix (optional, for comparison)
    standard_matrix_path = f"{output_dir}/sbm_standard_matrix.png"
    try:
        state.draw(
            output=standard_matrix_path,
            output_size=(1000, 1000),
        )
        print(f"  Standard block matrix: {standard_matrix_path}")
    except Exception as e:
        print(f"  Warning: Could not generate standard matrix plot: {e}")

    # 3. Network layout (only for smaller networks)
    if g.num_vertices() <= 200:
        network_path = f"{output_dir}/sbm_network.png"
        try:
            pos = gt.sfdp_layout(g)

            # Color by community
            vertex_color = state.get_levels()[0].get_blocks()

            # Edge color by layer (4 layers)
            edge_color_map = g.new_edge_property("vector<double>")
            layer_colors = {
                0: [0.2, 0.4, 0.8, 0.5],  # Blue for Pos→Pos
                1: [0.2, 0.6, 0.2, 0.5],  # Green for Neg→Neg
                2: [0.9, 0.6, 0.2, 0.5],  # Orange for Pos→Neg
                3: [0.8, 0.2, 0.2, 0.5],  # Red for Neg→Pos
            }
            for e in g.edges():
                layer = g.ep["layer"][e]
                edge_color_map[e] = layer_colors[layer]

            gt.graph_draw(
                g,
                pos=pos,
                vertex_fill_color=vertex_color,
                vertex_size=20,
                edge_color=edge_color_map,
                edge_pen_width=1.5,
                output=network_path,
                output_size=(1500, 1500),
            )
            print(f"  Network layout: {network_path}")
        except Exception as e:
            print(f"  Warning: Could not generate network plot: {e}")
    else:
        print(f"  Skipping network layout (too many nodes: {g.num_vertices()})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="4-Layer Nested SBM for SDG cashflow network using graph-tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--network-file",
        default="sdg_cashflow_network.csv",
        help="Path to sdg_cashflow_network.csv",
    )
    parser.add_argument(
        "--min-cashflow",
        type=float,
        default=0.0,
        help="Minimum cashflow threshold (filter weak edges)",
    )
    parser.add_argument(
        "--no-deg-corr",
        action="store_true",
        help="Disable degree correction (NOT recommended for networks with hubs)",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Ignore edge weights (use topology only)",
    )
    parser.add_argument(
        "--mcmc-steps",
        type=int,
        default=200,
        help="MCMC equilibration steps after initial optimization",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-membership",
        default="output/sbm_graphtool_membership.csv",
        help="Path to save node-level membership CSV",
    )
    parser.add_argument(
        "--output-summary",
        default="output/sbm_graphtool_communities.csv",
        help="Path to save community summary CSV",
    )
    parser.add_argument(
        "--no-save-all-levels",
        action="store_true",
        help="Disable saving all hierarchical levels to separate files",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable visualization plots generation",
    )
    return parser.parse_args()


def main() -> None:
    if not GRAPH_TOOL_AVAILABLE:
        return

    args = parse_args()

    print("\n" + "="*70)
    print("4-LAYER NESTED SBM ANALYSIS (graph-tool)")
    print("="*70)

    # Load and build graph
    g, id_map, node_map = load_and_build_graph(
        args.network_file,
        min_cashflow=args.min_cashflow,
    )

    # Fit SBM
    state = fit_nested_sbm(
        g,
        deg_corr=not args.no_deg_corr,
        use_weights=not args.no_weights,
        mcmc_steps=args.mcmc_steps,
        random_seed=args.random_seed,
    )

    # Extract results (finest level)
    membership_df, summary_df, comm_id_map = extract_results(state, g, id_map, node_map, level=0)

    print("\n" + "="*70)
    print("Community Summary (Level 0 - finest granularity)")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Save results
    print("\n" + "="*70)
    print("Saving results")
    print("="*70)

    membership_df.to_csv(args.output_membership, index=False)
    print(f"  Membership: {args.output_membership}")

    summary_df.to_csv(args.output_summary, index=False)
    print(f"  Summary:    {args.output_summary}")

    # Save all levels
    if not args.no_save_all_levels:
        output_prefix = args.output_membership.replace(".csv", "")
        save_all_levels(state, id_map, output_prefix)

    # Plot
    if not args.no_plot:
        output_dir = str(Path(args.output_summary).parent)
        plot_results(state, g, comm_id_map, id_map, output_dir)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
