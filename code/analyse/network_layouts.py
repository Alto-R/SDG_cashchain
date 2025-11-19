"""
使用不同布局算法可视化SDG现金流网络
1. Kamada-Kawai 布局 - 基于图论距离的能量最小化布局
2. 圆形分块布局 - 将各社区分配到圆形排列的独立块中
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os


def load_data(membership_csv: str, network_csv: str):
    """加载数据"""
    print("加载数据...")
    membership_df = pd.read_csv(membership_csv, encoding='utf-8')
    network_df = pd.read_csv(network_csv, encoding='utf-8')

    print(f"  社区成员数据: {len(membership_df)} 个节点")
    print(f"  现金流网络数据: {len(network_df)} 条边")
    print(f"  社区数: {membership_df['community'].nunique()}")

    return membership_df, network_df


def create_community_color_map(communities):
    """创建社区颜色映射"""
    n_communities = len(communities)

    if n_communities <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.tab20

    community_colors_map = {}
    for i, comm in enumerate(sorted(communities)):
        rgba = cmap(i / max(n_communities - 1, 1))
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255)
        )
        community_colors_map[comm] = hex_color

    return community_colors_map


def build_graph(membership_df: pd.DataFrame, network_df: pd.DataFrame):
    """构建网络图"""
    print("构建网络图...")
    G = nx.Graph()

    # 添加节点
    node_to_community = {}
    node_to_degree = {}

    for _, row in membership_df.iterrows():
        node = row['node']
        G.add_node(node,
                   community=row['community'],
                   node_type=row['node_type'])
        node_to_community[node] = row['community']
        node_to_degree[node] = row['positive_degree'] + row['negative_degree']

    # 添加真实的现金流边
    edge_count = 0
    for _, row in network_df.iterrows():
        if row['cashflow'] > 0:
            src = row['source_sdg']
            tgt = row['target_sdg']
            weight = float(row['cashflow'])

            if src in G.nodes() and tgt in G.nodes():
                if G.has_edge(src, tgt):
                    G[src][tgt]['weight'] += weight
                else:
                    G.add_edge(src, tgt, weight=weight)
                edge_count += 1

    print(f"  节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")

    return G, node_to_community, node_to_degree


def visualize_kamada_kawai(G, node_to_community, node_to_degree, communities,
                           community_colors_map, output_png):
    """使用 Kamada-Kawai 布局可视化"""
    print("\n" + "="*80)
    print("方法1: Kamada-Kawai 布局")
    print("="*80)

    # 计算布局
    print("计算 Kamada-Kawai 布局...")
    pos = nx.kamada_kawai_layout(G, weight='weight')

    # 添加jitter避免重叠
    np.random.seed(42)
    for node in pos:
        jitter_x = np.random.normal(0, 0.01)
        jitter_y = np.random.normal(0, 0.01)
        pos[node] = (pos[node][0] + jitter_x, pos[node][1] + jitter_y)

    # 绘图
    print("绘制图表...")
    fig, ax = plt.subplots(figsize=(18, 16))

    # 计算节点大小
    min_size = 80
    max_size = 800
    sizes = {}

    for node in G.nodes():
        total_degree = node_to_degree.get(node, 0)
        if total_degree > 0:
            size = min_size + (np.log10(total_degree + 1) / 10) * (max_size - min_size)
        else:
            size = min_size
        sizes[node] = size

    # 绘制边
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.1 + (w / max_weight) * 2.0 for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.1,
        edge_color='gray',
        ax=ax
    )

    # 按社区分组绘制节点
    for comm in sorted(communities, reverse=True):
        comm_nodes = [node for node in G.nodes()
                      if node_to_community[node] == comm]

        node_x = [pos[node][0] for node in comm_nodes]
        node_y = [pos[node][1] for node in comm_nodes]
        node_sizes = [sizes[node] for node in comm_nodes]

        ax.scatter(
            node_x,
            node_y,
            s=node_sizes,
            c=[community_colors_map[comm]],
            alpha=1.0,
            edgecolor='white',
            linewidth=1.5,
            zorder=3,
            label=f'Community {int(comm)}'
        )

    # 添加标签（只标注大节点）
    labels = {}
    degree_threshold = sorted(node_to_degree.values(), reverse=True)[min(20, len(node_to_degree)-1)]
    for node in G.nodes():
        if node_to_degree.get(node, 0) >= degree_threshold:
            labels[node] = node.replace('_Pos', '+').replace('_Neg', '-')

    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    # 样式设置
    ax.set_title('SDG Network - Kamada-Kawai Layout\n(Energy-minimization based on graph-theoretic distance)',
                 fontsize=16, fontweight='bold', pad=20)

    # 图例
    legend = ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        fancybox=False,
        fontsize=10
    )
    legend.get_frame().set_linewidth(0.5)

    ax.axis('off')

    # 保存
    plt.tight_layout()
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  已保存: {output_png}")
    plt.close()


def visualize_circular_block(G, node_to_community, node_to_degree, communities,
                             community_colors_map, output_png):
    """使用圆形分块布局可视化"""
    print("\n" + "="*80)
    print("方法2: 圆形分块布局（Circular Block Layout）")
    print("="*80)

    # 计算圆形分块布局
    print("计算圆形分块布局...")
    pos = {}

    # 统计每个社区的节点数
    community_nodes = {}
    for comm in communities:
        community_nodes[comm] = [node for node in G.nodes()
                                if node_to_community[node] == comm]

    n_communities = len(communities)

    # 为每个社区分配块半径（根据节点数）
    block_radius = {}
    for comm in communities:
        n_nodes = len(community_nodes[comm])
        if n_nodes == 1:
            block_radius[comm] = 0.5
        elif n_nodes <= 10:
            block_radius[comm] = 2.0
        elif n_nodes <= 30:
            block_radius[comm] = 4.0
        else:
            # 大社区需要更大空间
            block_radius[comm] = 6.0 + np.sqrt(n_nodes) * 0.3

    # 大圆半径（所有社区中心点形成的圆）
    max_block_radius = max(block_radius.values())
    main_circle_radius = max_block_radius * 3.5  # 确保社区块之间有足够间距

    # 为每个社区分配中心点位置（均匀分布在大圆上）
    community_centers = {}
    for i, comm in enumerate(sorted(communities)):
        angle = 2 * np.pi * i / n_communities
        cx = main_circle_radius * np.cos(angle)
        cy = main_circle_radius * np.sin(angle)
        community_centers[comm] = (cx, cy)

    # 为每个社区内的节点分配位置
    np.random.seed(42)
    for comm in communities:
        nodes = community_nodes[comm]
        n_nodes = len(nodes)
        cx, cy = community_centers[comm]
        radius = block_radius[comm]

        if n_nodes == 1:
            # 单节点直接放在中心
            pos[nodes[0]] = (cx, cy)
        else:
            # 多节点排成小圆圈，按度数排序
            sorted_nodes = sorted(nodes,
                                 key=lambda n: node_to_degree.get(n, 0),
                                 reverse=True)

            # 使用螺旋布局以容纳更多节点
            if n_nodes <= 20:
                # 单圈圆形布局
                for i, node in enumerate(sorted_nodes):
                    angle = 2 * np.pi * i / n_nodes
                    x = cx + radius * np.cos(angle)
                    y = cy + radius * np.sin(angle)
                    # 添加jitter避免重叠
                    jitter_x = np.random.normal(0, radius * 0.03)
                    jitter_y = np.random.normal(0, radius * 0.03)
                    pos[node] = (x + jitter_x, y + jitter_y)
            else:
                # 多圈螺旋布局
                nodes_per_ring = 15
                n_rings = int(np.ceil(n_nodes / nodes_per_ring))

                for i, node in enumerate(sorted_nodes):
                    ring_idx = i // nodes_per_ring
                    pos_in_ring = i % nodes_per_ring

                    # 每圈的半径递增
                    r = radius * (0.3 + 0.7 * ring_idx / max(n_rings - 1, 1))
                    angle = 2 * np.pi * pos_in_ring / min(nodes_per_ring, n_nodes - ring_idx * nodes_per_ring)

                    x = cx + r * np.cos(angle)
                    y = cy + r * np.sin(angle)

                    # 添加jitter避免重叠
                    jitter_x = np.random.normal(0, radius * 0.05)
                    jitter_y = np.random.normal(0, radius * 0.05)
                    pos[node] = (x + jitter_x, y + jitter_y)

    # 绘图
    print("绘制图表...")
    fig, ax = plt.subplots(figsize=(20, 20))

    # 计算节点大小
    min_size = 80
    max_size = 800
    sizes = {}

    for node in G.nodes():
        total_degree = node_to_degree.get(node, 0)
        if total_degree > 0:
            size = min_size + (np.log10(total_degree + 1) / 10) * (max_size - min_size)
        else:
            size = min_size
        sizes[node] = size

    # 绘制社区块的边界圆（可选，用于调试）
    # for comm in communities:
    #     cx, cy = community_centers[comm]
    #     circle = plt.Circle((cx, cy), block_radius[comm],
    #                         fill=False, edgecolor='lightgray',
    #                         linestyle='--', linewidth=1, alpha=0.3)
    #     ax.add_patch(circle)

    # 绘制边
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.1 + (w / max_weight) * 2.0 for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.05,
        edge_color='gray',
        ax=ax
    )

    # 按社区分组绘制节点
    for comm in sorted(communities, reverse=True):
        comm_nodes = [node for node in G.nodes()
                      if node_to_community[node] == comm]

        node_x = [pos[node][0] for node in comm_nodes]
        node_y = [pos[node][1] for node in comm_nodes]
        node_sizes = [sizes[node] for node in comm_nodes]

        ax.scatter(
            node_x,
            node_y,
            s=node_sizes,
            c=[community_colors_map[comm]],
            alpha=1.0,
            edgecolor='white',
            linewidth=1.5,
            zorder=3,
            label=f'Community {int(comm)}'
        )

    # 添加标签（只标注大节点）
    labels = {}
    degree_threshold = sorted(node_to_degree.values(), reverse=True)[min(25, len(node_to_degree)-1)]
    for node in G.nodes():
        if node_to_degree.get(node, 0) >= degree_threshold:
            labels[node] = node.replace('_Pos', '+').replace('_Neg', '-')

    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # 样式设置
    ax.set_title('SDG Network - Circular Block Layout\n(Communities arranged in circular blocks)',
                 fontsize=16, fontweight='bold', pad=20)

    # 图例
    legend = ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        fancybox=False,
        fontsize=10
    )
    legend.get_frame().set_linewidth(0.5)

    ax.axis('off')
    ax.set_aspect('equal')

    # 保存
    plt.tight_layout()
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  已保存: {output_png}")
    plt.close()


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    membership_csv = os.path.join(base_dir, 'output', 'sdg_signed_membership.csv')
    network_csv = os.path.join(base_dir, 'output', 'sdg_cashflow_network.csv')
    output_kamada = os.path.join(base_dir, 'output', 'network_kamada_kawai.png')
    output_circular = os.path.join(base_dir, 'output', 'network_circular_block.png')

    print("="*80)
    print("SDG网络可视化 - 多种布局方法")
    print("="*80)

    # 加载数据
    membership_df, network_df = load_data(membership_csv, network_csv)

    # 构建图
    G, node_to_community, node_to_degree = build_graph(membership_df, network_df)

    # 创建颜色映射
    communities = sorted(membership_df['community'].unique())
    community_colors_map = create_community_color_map(communities)

    # 方法1: Kamada-Kawai 布局
    visualize_kamada_kawai(G, node_to_community, node_to_degree,
                          communities, community_colors_map, output_kamada)

    # 方法2: 圆形分块布局
    visualize_circular_block(G, node_to_community, node_to_degree,
                            communities, community_colors_map, output_circular)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n生成的图表:")
    print(f"  1. Kamada-Kawai 布局: {output_kamada}")
    print(f"  2. 圆形分块布局: {output_circular}")


if __name__ == '__main__':
    main()