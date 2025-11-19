# -*- coding: utf-8 -*-
"""
SDG社区成员可视化
可视化sdg_signed_membership.csv中的社区检测结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import os


def load_membership_data(csv_file='../../output/sdg_signed_membership.csv'):
    """加载社区成员数据

    Args:
        csv_file: CSV文件路径

    Returns:
        DataFrame: 社区成员数据
    """
    print(f"正在加载社区成员数据: {csv_file}")
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"  - 加载了 {len(df)} 个节点")
    print(f"  - 社区数量: {df['community'].nunique()}")
    print(f"  - 节点类型: {df['node_type'].unique().tolist()}")
    return df


def visualize_community_distribution(df, output_dir='./community'):
    """可视化社区分布 - 堆叠条形图

    Args:
        df: 社区成员数据
        output_dir: 输出目录
    """
    print("\n生成社区分布图...")

    os.makedirs(output_dir, exist_ok=True)

    # 按社区和节点类型统计
    community_stats = df.groupby(['community', 'node_type']).size().unstack(fill_value=0)

    # 确保Positive和Negative列都存在
    if 'Positive' not in community_stats.columns:
        community_stats['Positive'] = 0
    if 'Negative' not in community_stats.columns:
        community_stats['Negative'] = 0

    # 按总节点数排序
    community_stats['total'] = community_stats['Positive'] + community_stats['Negative']
    community_stats = community_stats.sort_values('total', ascending=False)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(community_stats))
    width = 0.6

    # 绘制堆叠条形图
    bars_pos = ax.bar(x, community_stats['Positive'], width,
                      label='Positive', color='#6a88c2', edgecolor='white')
    bars_neg = ax.bar(x, community_stats['Negative'], width,
                      bottom=community_stats['Positive'],
                      label='Negative', color='#eb6468', edgecolor='white')

    # 设置标签
    ax.set_xlabel('Community', fontsize=12)
    ax.set_ylabel('Number of Nodes', fontsize=12)
    ax.set_title('SDG Community Membership Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{int(c)}' for c in community_stats.index])
    ax.legend()

    # 添加数值标签
    for i, (pos, neg) in enumerate(zip(community_stats['Positive'], community_stats['Negative'])):
        total = pos + neg
        ax.text(i, total + 0.5, str(int(total)), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'community_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def visualize_degree_scatter(df, output_dir='./community'):
    """可视化正负度数散点图 - 按社区着色

    Args:
        df: 社区成员数据
        output_dir: 输出目录
    """
    print("\n生成度数散点图...")

    os.makedirs(output_dir, exist_ok=True)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 获取唯一社区并分配颜色
    communities = sorted(df['community'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
    color_map = dict(zip(communities, colors))

    # 绘制散点图
    for community in communities:
        mask = df['community'] == community
        subset = df[mask]

        # 区分正负节点的标记
        pos_mask = subset['node_type'] == 'Positive'
        neg_mask = subset['node_type'] == 'Negative'

        # 正节点用圆形
        if pos_mask.any():
            ax.scatter(subset.loc[pos_mask, 'positive_degree'] / 1e6,
                      subset.loc[pos_mask, 'negative_degree'] / 1e6,
                      c=[color_map[community]], marker='o', s=50, alpha=0.7,
                      label=f'C{int(community)} Pos' if community in [0, 1, 2] else '')

        # 负节点用三角形
        if neg_mask.any():
            ax.scatter(subset.loc[neg_mask, 'positive_degree'] / 1e6,
                      subset.loc[neg_mask, 'negative_degree'] / 1e6,
                      c=[color_map[community]], marker='^', s=50, alpha=0.7,
                      label=f'C{int(community)} Neg' if community in [0, 1, 2] else '')

    # 添加对角线
    max_val = max(df['positive_degree'].max(), df['negative_degree'].max()) / 1e6
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal degree')

    ax.set_xlabel('Positive Degree (Millions)', fontsize=12)
    ax.set_ylabel('Negative Degree (Millions)', fontsize=12)
    ax.set_title('SDG Node Degree Distribution by Community', fontsize=14, fontweight='bold')

    # 创建图例
    legend_elements = []
    for i, community in enumerate(communities[:5]):  # 只显示前5个社区
        legend_elements.append(mpatches.Patch(color=color_map[community],
                                              label=f'Community {int(community)}'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='gray',
                                       label='Positive', markersize=8, linestyle='None'))
    legend_elements.append(plt.Line2D([0], [0], marker='^', color='gray',
                                       label='Negative', markersize=8, linestyle='None'))

    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'degree_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def visualize_community_heatmap(df, output_dir='./community'):
    """可视化社区-SDG大目标热力图

    Args:
        df: 社区成员数据
        output_dir: 输出目录
    """
    print("\n生成社区-SDG热力图...")

    os.makedirs(output_dir, exist_ok=True)

    # 提取SDG大目标编号
    df['sdg_goal'] = df['node'].apply(lambda x: int(str(x).split('.')[0].split('_')[0]))

    # 创建交叉表
    cross_tab = pd.crosstab(df['sdg_goal'], df['community'])

    # 按社区总数排序列
    col_order = cross_tab.sum().sort_values(ascending=False).index
    cross_tab = cross_tab[col_order]

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制热力图
    im = ax.imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')

    # 设置刻度
    ax.set_xticks(np.arange(len(cross_tab.columns)))
    ax.set_yticks(np.arange(len(cross_tab.index)))
    ax.set_xticklabels([f'C{int(c)}' for c in cross_tab.columns])
    ax.set_yticklabels([f'SDG {g}' for g in cross_tab.index])

    # 添加数值标签
    for i in range(len(cross_tab.index)):
        for j in range(len(cross_tab.columns)):
            value = cross_tab.iloc[i, j]
            if value > 0:
                text_color = 'white' if value > cross_tab.values.max() / 2 else 'black'
                ax.text(j, i, int(value), ha='center', va='center',
                       color=text_color, fontsize=8)

    ax.set_xlabel('Community', fontsize=12)
    ax.set_ylabel('SDG Goal', fontsize=12)
    ax.set_title('SDG Goals Distribution Across Communities', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Nodes', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'community_sdg_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def visualize_community_composition(df, output_dir='./community'):
    """可视化主要社区的节点组成

    Args:
        df: 社区成员数据
        output_dir: 输出目录
    """
    print("\n生成社区组成图...")

    os.makedirs(output_dir, exist_ok=True)

    # 获取所有社区
    community_sizes = df['community'].value_counts()
    all_communities = sorted(community_sizes.index.tolist())

    # 计算子图布局
    n_communities = len(all_communities)
    n_cols = 3
    n_rows = (n_communities + n_cols - 1) // n_cols

    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_communities > 1 else [axes]

    for idx, community in enumerate(all_communities):
        ax = axes[idx]
        subset = df[df['community'] == community]

        # 按节点类型和SDG目标统计
        subset['sdg_goal'] = subset['node'].apply(
            lambda x: int(str(x).split('.')[0].split('_')[0]))

        # 统计每个SDG目标的正负节点
        goal_stats = subset.groupby(['sdg_goal', 'node_type']).size().unstack(fill_value=0)

        if 'Positive' not in goal_stats.columns:
            goal_stats['Positive'] = 0
        if 'Negative' not in goal_stats.columns:
            goal_stats['Negative'] = 0

        # 绘制水平堆叠条形图
        y = np.arange(len(goal_stats))
        height = 0.6

        ax.barh(y, goal_stats['Positive'], height, label='Positive', color='#6a88c2')
        ax.barh(y, goal_stats['Negative'], height, left=goal_stats['Positive'],
                label='Negative', color='#eb6468')

        ax.set_yticks(y)
        ax.set_yticklabels([f'SDG {g}' for g in goal_stats.index], fontsize=8)
        ax.set_xlabel('Count', fontsize=9)
        ax.set_title(f'Community {int(community)} (n={len(subset)})', fontsize=11, fontweight='bold')

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # 隐藏多余的子图
    for idx in range(n_communities, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Community Composition by SDG Goals', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'community_composition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def visualize_subtarget_network(df, output_dir='./community'):
    """可视化子目标级别的社区网络图

    Args:
        df: 社区成员数据
        output_dir: 输出目录
    """
    print("\n生成子目标网络图...")

    os.makedirs(output_dir, exist_ok=True)

    # 创建网络图
    G = nx.Graph()

    # 获取社区颜色映射
    communities = sorted(df['community'].unique())
    community_colors = plt.cm.tab10(np.linspace(0, 1, max(len(communities), 10)))
    community_color_map = dict(zip(communities, community_colors))

    # 添加节点
    for _, row in df.iterrows():
        node_name = row['node']
        G.add_node(node_name,
                   community=row['community'],
                   node_type=row['node_type'],
                   pos_degree=row['positive_degree'],
                   neg_degree=row['negative_degree'])

    # 计算节点大小（基于总度数）
    node_sizes = []
    node_colors = []

    for node in G.nodes():
        data = G.nodes[node]
        total_degree = data['pos_degree'] + data['neg_degree']
        size = np.log10(total_degree + 1) * 50
        node_sizes.append(size)
        node_colors.append(community_color_map[data['community']])

    # 使用spring布局，同社区节点靠近
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 14))

    # 分别绘制Positive和Negative节点
    pos_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'Positive']
    neg_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'Negative']

    pos_sizes = [node_sizes[list(G.nodes()).index(n)] for n in pos_nodes]
    neg_sizes = [node_sizes[list(G.nodes()).index(n)] for n in neg_nodes]

    pos_colors = [node_colors[list(G.nodes()).index(n)] for n in pos_nodes]
    neg_colors = [node_colors[list(G.nodes()).index(n)] for n in neg_nodes]

    # 绘制Positive节点（圆形）
    nx.draw_networkx_nodes(G, pos, nodelist=pos_nodes, node_size=pos_sizes,
                           node_color=pos_colors, node_shape='o', alpha=0.8, ax=ax)

    # 绘制Negative节点（三角形）
    nx.draw_networkx_nodes(G, pos, nodelist=neg_nodes, node_size=neg_sizes,
                           node_color=neg_colors, node_shape='^', alpha=0.8, ax=ax)

    # 添加节点标签（只显示大节点的标签）
    labels = {}
    for node in G.nodes():
        data = G.nodes[node]
        total_degree = data['pos_degree'] + data['neg_degree']
        if total_degree > 50000000:  # 只显示度数大于5000万的节点标签
            # 简化标签：去掉_Pos/_Neg后缀
            label = node.replace('_Pos', '+').replace('_Neg', '-')
            labels[node] = label

    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # 创建图例
    legend_elements = []
    for community in communities[:5]:
        legend_elements.append(mpatches.Patch(
            color=community_color_map[community],
            label=f'Community {int(community)}'))

    legend_elements.append(plt.Line2D([0], [0], marker='o', color='gray',
                                       label='Positive', markersize=10, linestyle='None'))
    legend_elements.append(plt.Line2D([0], [0], marker='^', color='gray',
                                       label='Negative', markersize=10, linestyle='None'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.set_title('SDG Subtarget Community Network', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'subtarget_network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def visualize_major_goal_network(df, output_dir='./community'):
    """可视化大目标级别的社区网络图

    Args:
        df: 社区成员数据
        output_dir: 输出目录
    """
    print("\n生成大目标网络图...")

    os.makedirs(output_dir, exist_ok=True)

    # 提取大目标信息
    df_copy = df.copy()
    df_copy['sdg_goal'] = df_copy['node'].apply(
        lambda x: int(str(x).split('.')[0].split('_')[0]))

    # 按大目标和节点类型聚合
    goal_agg = df_copy.groupby(['sdg_goal', 'node_type']).agg({
        'positive_degree': 'sum',
        'negative_degree': 'sum',
        'community': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # 主要社区
        'node': 'count'  # 子目标数量
    }).reset_index()
    goal_agg.rename(columns={'node': 'subtarget_count'}, inplace=True)

    # 创建网络图
    G = nx.Graph()

    # 获取社区颜色映射
    communities = sorted(df['community'].unique())
    community_colors = plt.cm.tab10(np.linspace(0, 1, max(len(communities), 10)))
    community_color_map = dict(zip(communities, community_colors))

    # 添加节点
    for _, row in goal_agg.iterrows():
        node_name = f"SDG{row['sdg_goal']}_{row['node_type'][:3]}"
        G.add_node(node_name,
                   sdg_goal=row['sdg_goal'],
                   node_type=row['node_type'],
                   community=row['community'],
                   pos_degree=row['positive_degree'],
                   neg_degree=row['negative_degree'],
                   subtarget_count=row['subtarget_count'])

    # 计算节点大小
    node_sizes = []
    node_colors = []

    for node in G.nodes():
        data = G.nodes[node]
        total_degree = data['pos_degree'] + data['neg_degree']
        size = np.log10(total_degree + 1) * 100
        node_sizes.append(size)
        node_colors.append(community_color_map[data['community']])

    # 创建自定义布局：同一SDG的Pos和Neg靠近
    pos = {}
    n_goals = 17
    for i, goal in enumerate(range(1, 18)):
        angle = 2 * np.pi * i / n_goals - np.pi / 2
        radius = 1.0

        # Positive节点
        pos_node = f"SDG{goal}_Pos"
        if pos_node in G.nodes():
            pos[pos_node] = (radius * np.cos(angle) - 0.05, radius * np.sin(angle) + 0.05)

        # Negative节点
        neg_node = f"SDG{goal}_Neg"
        if neg_node in G.nodes():
            pos[neg_node] = (radius * np.cos(angle) + 0.05, radius * np.sin(angle) - 0.05)

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 14))

    # 分别绘制Positive和Negative节点
    pos_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'Positive']
    neg_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'Negative']

    pos_sizes = [node_sizes[list(G.nodes()).index(n)] for n in pos_nodes]
    neg_sizes = [node_sizes[list(G.nodes()).index(n)] for n in neg_nodes]

    pos_colors = [node_colors[list(G.nodes()).index(n)] for n in pos_nodes]
    neg_colors = [node_colors[list(G.nodes()).index(n)] for n in neg_nodes]

    # 绘制Positive节点（圆形）
    nx.draw_networkx_nodes(G, pos, nodelist=pos_nodes, node_size=pos_sizes,
                           node_color=pos_colors, node_shape='o', alpha=0.8, ax=ax,
                           edgecolors='white', linewidths=2)

    # 绘制Negative节点（正方形）
    nx.draw_networkx_nodes(G, pos, nodelist=neg_nodes, node_size=neg_sizes,
                           node_color=neg_colors, node_shape='s', alpha=0.8, ax=ax,
                           edgecolors='white', linewidths=2)

    # 添加节点标签
    labels = {node: f"{G.nodes[node]['sdg_goal']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

    # 创建图例
    legend_elements = []
    for community in communities[:5]:
        legend_elements.append(mpatches.Patch(
            color=community_color_map[community],
            label=f'Community {int(community)}'))

    legend_elements.append(plt.Line2D([0], [0], marker='o', color='gray',
                                       label='Positive', markersize=10, linestyle='None'))
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='gray',
                                       label='Negative', markersize=10, linestyle='None'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.set_title('SDG Major Goal Community Network', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'major_goal_network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def visualize_community_network_by_group(df, output_dir='./community'):
    """按社区分组可视化网络图

    Args:
        df: 社区成员数据
        output_dir: 输出目录
    """
    print("\n生成分社区网络图...")

    os.makedirs(output_dir, exist_ok=True)

    # 获取所有社区
    community_sizes = df['community'].value_counts()
    all_communities = sorted(community_sizes.index.tolist())

    # 计算子图布局
    n_communities = len(all_communities)
    n_cols = 3
    n_rows = (n_communities + n_cols - 1) // n_cols

    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_communities > 1 else [axes]

    for idx, community in enumerate(all_communities):
        ax = axes[idx]
        subset = df[df['community'] == community].copy()

        # 创建该社区的网络图
        G = nx.Graph()

        # 添加节点
        for _, row in subset.iterrows():
            node_name = row['node']
            G.add_node(node_name,
                       node_type=row['node_type'],
                       pos_degree=row['positive_degree'],
                       neg_degree=row['negative_degree'])

        # 计算节点大小和颜色
        node_sizes = []
        node_colors = []

        for node in G.nodes():
            data = G.nodes[node]
            total_degree = data['pos_degree'] + data['neg_degree']
            size = np.log10(total_degree + 1) * 80
            node_sizes.append(size)

            # 按节点类型着色
            if data['node_type'] == 'Positive':
                node_colors.append('#6a88c2')
            else:
                node_colors.append('#eb6468')

        # 使用spring布局
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

        # 分别绘制Positive和Negative节点
        pos_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'Positive']
        neg_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'Negative']

        pos_sizes_sub = [node_sizes[list(G.nodes()).index(n)] for n in pos_nodes]
        neg_sizes_sub = [node_sizes[list(G.nodes()).index(n)] for n in neg_nodes]

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, nodelist=pos_nodes, node_size=pos_sizes_sub,
                               node_color='#6a88c2', node_shape='o', alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=neg_nodes, node_size=neg_sizes_sub,
                               node_color='#eb6468', node_shape='^', alpha=0.8, ax=ax)

        # 添加标签（只显示大节点）
        labels = {}
        for node in G.nodes():
            data = G.nodes[node]
            total_degree = data['pos_degree'] + data['neg_degree']
            if total_degree > subset['positive_degree'].quantile(0.7) + subset['negative_degree'].quantile(0.7):
                label = node.replace('_Pos', '+').replace('_Neg', '-')
                labels[node] = label

        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

        ax.set_title(f'Community {int(community)} (n={len(subset)})',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(n_communities, len(axes)):
        axes[idx].set_visible(False)

    # 添加总标题和图例
    fig.suptitle('SDG Community Networks', fontsize=14, fontweight='bold', y=1.02)

    # 添加共享图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='#6a88c2',
                   label='Positive', markersize=10, linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='#eb6468',
                   label='Negative', markersize=10, linestyle='None')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'community_networks.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def visualize_membership(csv_file='../../output/sdg_signed_membership.csv',
                         output_dir='./community'):
    """主可视化函数 - 生成所有社区成员可视化图

    Args:
        csv_file: 输入CSV文件路径
        output_dir: 输出目录

    Returns:
        DataFrame: 加载的数据
    """
    print("=" * 80)
    print("SDG社区成员可视化")
    print("=" * 80)

    # 加载数据
    df = load_membership_data(csv_file)

    # 打印数据摘要
    print(f"\n数据摘要:")
    print(f"  - 总节点数: {len(df)}")
    print(f"  - Positive节点: {len(df[df['node_type'] == 'Positive'])}")
    print(f"  - Negative节点: {len(df[df['node_type'] == 'Negative'])}")
    print(f"  - 社区分布: {df['community'].value_counts().to_dict()}")

    # 生成所有可视化
    visualize_community_distribution(df, output_dir)
    visualize_degree_scatter(df, output_dir)
    visualize_community_heatmap(df, output_dir)
    visualize_community_composition(df, output_dir)

    # 生成网络图可视化
    visualize_subtarget_network(df, output_dir)
    visualize_major_goal_network(df, output_dir)
    visualize_community_network_by_group(df, output_dir)

    print("\n" + "=" * 80)
    print(f"完成！所有图表已保存到: {output_dir}")
    print("=" * 80)

    return df


def main():
    """主函数"""
    # 使用默认路径
    csv_file = '../../output/sdg_signed_membership.csv'
    output_dir = './community'

    # 运行可视化
    visualize_membership(csv_file, output_dir)


if __name__ == '__main__':
    main()
