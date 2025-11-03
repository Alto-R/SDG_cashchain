# -*- coding: utf-8 -*-
"""
SDG现金流网络分析
基于行业分类和SDG映射，构建SDG子目标之间的现金流网络
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_sdg_mappings(mappings_file):
    """加载行业-SDG映射表"""
    print("正在加载SDG映射表...")
    df = pd.read_csv(mappings_file, encoding='utf-8')
    print(f"  - 加载了 {len(df)} 条映射记录")
    print(f"  - 涉及 {df['industry_name'].nunique()} 个行业")
    print(f"  - 涉及 {df['sdg_target_id'].nunique()} 个SDG子目标")
    return df


def load_and_aggregate_cashflow(cashflow_file, chunksize=100000):
    """加载交易数据并按行业聚合现金流"""
    print("\n正在加载并聚合交易数据...")

    # 需要的列
    cols = ['発注社小分類', '受注社小分類', '年間取引高']

    # 使用分块读取处理大文件
    aggregated_data = defaultdict(lambda: {'cashflow': 0, 'count': 0})
    total_rows = 0
    valid_rows = 0

    for chunk in pd.read_csv(cashflow_file, usecols=cols, chunksize=chunksize,
                             encoding='utf-8', low_memory=False):
        total_rows += len(chunk)

        # 过滤掉年間取引高为空的记录
        chunk = chunk.dropna(subset=['年間取引高'])
        valid_rows += len(chunk)

        # 确保两个行业列都不为空
        chunk = chunk.dropna(subset=['発注社小分類', '受注社小分類'])

        # 确保年間取引高是数值类型
        chunk['年間取引高'] = pd.to_numeric(chunk['年間取引高'], errors='coerce')
        chunk = chunk.dropna(subset=['年間取引高'])

        # 聚合
        grouped = chunk.groupby(['発注社小分類', '受注社小分類'])
        for (source_ind, target_ind), group in grouped:
            key = (source_ind, target_ind)
            aggregated_data[key]['cashflow'] += group['年間取引高'].sum()
            aggregated_data[key]['count'] += len(group)

        if total_rows % 500000 == 0:
            print(f"  - 已处理 {total_rows:,} 行...")

    print(f"\n数据加载完成:")
    print(f"  - 总行数: {total_rows:,}")
    print(f"  - 有效行数（年間取引高非空）: {valid_rows:,}")
    print(f"  - 唯一行业对数: {len(aggregated_data):,}")

    # 转换为DataFrame
    result = pd.DataFrame([
        {
            'source_industry': k[0],
            'target_industry': k[1],
            'total_cashflow': v['cashflow'],
            'transaction_count': v['count']
        }
        for k, v in aggregated_data.items()
    ])

    return result


def map_industry_to_sdg(industry_cashflow_df, sdg_mappings_df):
    """将行业间现金流映射到SDG子目标间现金流"""
    print("\n正在将行业映射到SDG子目标...")

    # 创建行业到SDG的映射字典（分正负面）
    industry_to_sdg_positive = sdg_mappings_df[sdg_mappings_df['impact_type'] == 'Positive'][
        ['industry_name', 'sdg_target_id']
    ].drop_duplicates()

    industry_to_sdg_negative = sdg_mappings_df[sdg_mappings_df['impact_type'] == 'Negative'][
        ['industry_name', 'sdg_target_id']
    ].drop_duplicates()

    print(f"  - 正面影响映射: {len(industry_to_sdg_positive)} 条")
    print(f"  - 负面影响映射: {len(industry_to_sdg_negative)} 条")

    # 处理正面影响
    positive_flows = []
    for _, row in industry_cashflow_df.iterrows():
        source_sdgs = industry_to_sdg_positive[
            industry_to_sdg_positive['industry_name'] == row['source_industry']
        ]['sdg_target_id'].values

        target_sdgs = industry_to_sdg_positive[
            industry_to_sdg_positive['industry_name'] == row['target_industry']
        ]['sdg_target_id'].values

        # 每个SDG映射都计算全额（笛卡尔积）
        for source_sdg in source_sdgs:
            for target_sdg in target_sdgs:
                positive_flows.append({
                    'source_sdg': source_sdg,
                    'target_sdg': target_sdg,
                    'cashflow': row['total_cashflow'],
                    'transaction_count': row['transaction_count'],
                    'impact_type': 'Positive'
                })

    # 处理负面影响
    negative_flows = []
    for _, row in industry_cashflow_df.iterrows():
        source_sdgs = industry_to_sdg_negative[
            industry_to_sdg_negative['industry_name'] == row['source_industry']
        ]['sdg_target_id'].values

        target_sdgs = industry_to_sdg_negative[
            industry_to_sdg_negative['industry_name'] == row['target_industry']
        ]['sdg_target_id'].values

        for source_sdg in source_sdgs:
            for target_sdg in target_sdgs:
                negative_flows.append({
                    'source_sdg': source_sdg,
                    'target_sdg': target_sdg,
                    'cashflow': row['total_cashflow'],
                    'transaction_count': row['transaction_count'],
                    'impact_type': 'Negative'
                })

    print(f"\n映射结果:")
    print(f"  - 正面影响SDG流: {len(positive_flows):,} 条")
    print(f"  - 负面影响SDG流: {len(negative_flows):,} 条")

    # 合并并聚合
    all_flows = pd.DataFrame(positive_flows + negative_flows)

    if len(all_flows) == 0:
        print("警告: 没有找到匹配的SDG映射!")
        return pd.DataFrame()

    # 按source_sdg, target_sdg, impact_type聚合
    sdg_flows = all_flows.groupby(['source_sdg', 'target_sdg', 'impact_type']).agg({
        'cashflow': 'sum',
        'transaction_count': 'sum'
    }).reset_index()

    print(f"  - 聚合后唯一SDG流: {len(sdg_flows):,} 条")

    return sdg_flows


def build_sdg_network(sdg_flows_df):
    """构建SDG网络图（MultiDiGraph）"""
    print("\n正在构建SDG网络图...")

    G = nx.MultiDiGraph()

    for _, row in sdg_flows_df.iterrows():
        G.add_edge(
            row['source_sdg'],
            row['target_sdg'],
            weight=row['cashflow'],
            transaction_count=row['transaction_count'],
            impact_type=row['impact_type']
        )

    print(f"  - 节点数: {G.number_of_nodes()}")
    print(f"  - 边数: {G.number_of_edges()}")

    return G


def visualize_sdg_network(G, output_file='sdg_network_visualization.png'):
    """可视化SDG网络图"""
    print(f"\n正在生成网络可视化图...")

    # 创建更大的图
    fig, ax = plt.subplots(figsize=(30, 30))

    print("  - 正在计算节点布局...")
    # pos = nx.circular_layout(G)
    pos = nx.spring_layout(G, k=5, iterations=100, seed=42, scale=2)

    # 计算节点大小（基于总流入+流出）
    node_sizes = {}
    for node in G.nodes():
        in_flow = sum([d['weight'] for u, v, d in G.in_edges(node, data=True)])
        out_flow = sum([d['weight'] for u, v, d in G.out_edges(node, data=True)])
        node_sizes[node] = in_flow + out_flow

    # 归一化节点大小
    max_size = max(node_sizes.values()) if node_sizes else 1
    node_sizes = {k: (v / max_size) * 2000 + 500 for k, v in node_sizes.items()}

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[node_sizes[node] for node in G.nodes()],
        node_color='lightblue',
        alpha=0.7,
        ax=ax
    )

    # 绘制节点标签（Arial字体）
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        font_family='Arial',
        ax=ax
    )

    # 分别绘制正面和负面影响的边
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['impact_type'] == 'Positive']
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['impact_type'] == 'Negative']

    # 获取边的权重用于设置粗细
    positive_weights = [G[u][v][0]['weight'] for u, v in positive_edges] if positive_edges else []
    negative_weights = [G[u][v][0]['weight'] for u, v in negative_edges] if negative_edges else []

    # 归一化权重
    if positive_weights:
        max_pos_weight = max(positive_weights)
        positive_widths = [(w / max_pos_weight) * 5 + 0.5 for w in positive_weights]
    else:
        positive_widths = []

    if negative_weights:
        max_neg_weight = max(negative_weights)
        negative_widths = [(w / max_neg_weight) * 5 + 0.5 for w in negative_weights]
    else:
        negative_widths = []

    # 绘制正面影响边（蓝色）
    if positive_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=positive_edges,
            width=positive_widths,
            edge_color='#3E7fB7',
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )

    # 绘制负面影响边（红色）
    if negative_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=negative_edges,
            width=negative_widths,
            edge_color='#EB3136',
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3E7fB7', lw=3, label='Positive Impact', alpha=0.7),
        Line2D([0], [0], color='#EB3136', lw=3, label='Negative Impact', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16, prop={'family': 'Arial'})

    plt.title('SDG Target Cashflow Network\n(Node size = total flow, Edge width = cashflow amount)',
              fontsize=20, fontweight='bold', fontfamily='Arial')
    plt.axis('off')
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  - 网络图已保存至: {output_file}")

    plt.close()


def visualize_sdg_network_by_impact(G, impact_type='Positive', output_file='sdg_network_positive.png'):
    """根据影响类型可视化SDG网络图

    Parameters:
    -----------
    G : networkx.MultiDiGraph
        SDG网络图
    impact_type : str
        影响类型，'Positive' 或 'Negative'
    output_file : str
        输出文件名
    """
    print(f"\n正在生成{impact_type}影响网络可视化图...")

    # 筛选特定影响类型的边，构建子图
    filtered_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['impact_type'] == impact_type]

    if not filtered_edges:
        print(f"  - 警告: 没有找到{impact_type}影响的边！")
        return

    # 创建子图
    G_sub = nx.MultiDiGraph()
    for u, v, d in filtered_edges:
        G_sub.add_edge(u, v, **d)

    print(f"  - {impact_type}影响子图节点数: {G_sub.number_of_nodes()}")
    print(f"  - {impact_type}影响子图边数: {G_sub.number_of_edges()}")

    # 创建更大的图
    fig, ax = plt.subplots(figsize=(30, 30))

    print("  - 正在计算节点布局...")
    # pos = nx.circular_layout(G_sub)
    pos = nx.spring_layout(G_sub, k=5, iterations=100, seed=42, scale=2)

    # 计算节点大小（基于该影响类型的流入+流出）
    node_sizes = {}
    for node in G_sub.nodes():
        in_flow = sum([d['weight'] for u, v, d in G_sub.in_edges(node, data=True)])
        out_flow = sum([d['weight'] for u, v, d in G_sub.out_edges(node, data=True)])
        node_sizes[node] = in_flow + out_flow

    # 归一化节点大小
    max_size = max(node_sizes.values()) if node_sizes else 1
    node_sizes = {k: (v / max_size) * 2000 + 500 for k, v in node_sizes.items()}

    # 根据影响类型选择颜色
    if impact_type == 'Positive':
        node_color = '#B3D9FF'  # 浅蓝色
        edge_color = '#3E7fB7'  # 蓝色
        title_suffix = 'Positive Impact'
    else:
        node_color = '#FFB3B3'  # 浅红色
        edge_color = '#EB3136'  # 红色
        title_suffix = 'Negative Impact'

    # 绘制节点
    nx.draw_networkx_nodes(
        G_sub, pos,
        node_size=[node_sizes[node] for node in G_sub.nodes()],
        node_color=node_color,
        alpha=0.7,
        ax=ax
    )

    # 绘制节点标签
    nx.draw_networkx_labels(
        G_sub, pos,
        font_size=12,
        font_weight='bold',
        font_family='Arial',
        ax=ax
    )

    # 获取边的权重并归一化
    edges = list(G_sub.edges())
    weights = [G_sub[u][v][0]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    widths = [(w / max_weight) * 5 + 0.5 for w in weights]

    # 绘制边
    nx.draw_networkx_edges(
        G_sub, pos,
        edgelist=edges,
        width=widths,
        edge_color=edge_color,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )

    plt.title(f'SDG Target Cashflow Network - {title_suffix}\n(Node size = total flow, Edge width = cashflow amount)',
              fontsize=20, fontweight='bold', fontfamily='Arial')
    plt.axis('off')
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  - {impact_type}影响网络图已保存至: {output_file}")

    plt.close()


def aggregate_to_sdg_goals(sdg_flows_df):
    """将SDG子目标聚合到SDG大目标"""
    print("\n正在聚合到SDG大目标...")

    # 提取SDG大目标编号（例如从"1.1"提取"1"）
    def extract_goal(sdg_target):
        return str(sdg_target).split('.')[0]

    # 添加SDG大目标列
    sdg_flows_df_copy = sdg_flows_df.copy()
    sdg_flows_df_copy['source_goal'] = sdg_flows_df_copy['source_sdg'].apply(extract_goal)
    sdg_flows_df_copy['target_goal'] = sdg_flows_df_copy['target_sdg'].apply(extract_goal)

    # 按大目标聚合
    goal_flows = sdg_flows_df_copy.groupby(['source_goal', 'target_goal', 'impact_type']).agg({
        'cashflow': 'sum',
        'transaction_count': 'sum'
    }).reset_index()

    # 重命名列以保持一致性
    goal_flows = goal_flows.rename(columns={
        'source_goal': 'source_sdg',
        'target_goal': 'target_sdg'
    })

    print(f"  - 聚合后SDG大目标流: {len(goal_flows):,} 条")
    print(f"  - 涉及 {goal_flows['source_sdg'].nunique()} 个SDG大目标")

    return goal_flows


def visualize_sdg_goal_network(G, output_file='sdg_goal_network_visualization.png'):
    """可视化SDG大目标网络图"""
    print(f"\n正在生成SDG大目标网络可视化图...")

    # 创建更大的图
    fig, ax = plt.subplots(figsize=(24, 24))

    print("  - 正在计算节点布局...")
    # 使用circular布局，更适合展示大目标之间的关系
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42, scale=2)

    # 计算节点大小（基于总流入+流出）
    node_sizes = {}
    for node in G.nodes():
        in_flow = sum([d['weight'] for u, v, d in G.in_edges(node, data=True)])
        out_flow = sum([d['weight'] for u, v, d in G.out_edges(node, data=True)])
        node_sizes[node] = in_flow + out_flow

    # 归一化节点大小
    max_size = max(node_sizes.values()) if node_sizes else 1
    node_sizes = {k: (v / max_size) * 3000 + 800 for k, v in node_sizes.items()}

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[node_sizes[node] for node in G.nodes()],
        node_color='lightgreen',
        alpha=0.8,
        ax=ax
    )

    # 绘制节点标签（添加"SDG"前缀）
    labels = {node: f'SDG {node}' for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=14,
        font_weight='bold',
        font_family='Arial',
        ax=ax
    )

    # 分别绘制正面和负面影响的边
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['impact_type'] == 'Positive']
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['impact_type'] == 'Negative']

    # 获取边的权重用于设置粗细
    positive_weights = [G[u][v][0]['weight'] for u, v in positive_edges] if positive_edges else []
    negative_weights = [G[u][v][0]['weight'] for u, v in negative_edges] if negative_edges else []

    # 归一化权重
    if positive_weights:
        max_pos_weight = max(positive_weights)
        positive_widths = [(w / max_pos_weight) * 8 + 1 for w in positive_weights]
    else:
        positive_widths = []

    if negative_weights:
        max_neg_weight = max(negative_weights)
        negative_widths = [(w / max_neg_weight) * 8 + 1 for w in negative_weights]
    else:
        negative_widths = []

    # 绘制正面影响边（蓝色）
    if positive_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=positive_edges,
            width=positive_widths,
            edge_color='#3E7fB7',
            alpha=0.5,
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.15',
            ax=ax
        )

    # 绘制负面影响边（红色）
    if negative_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=negative_edges,
            width=negative_widths,
            edge_color='#EB3136',
            alpha=0.5,
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.15',
            ax=ax
        )

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3E7fB7', lw=4, label='Positive Impact', alpha=0.7),
        Line2D([0], [0], color='#EB3136', lw=4, label='Negative Impact', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18, prop={'family': 'Arial'})

    plt.title('SDG Goal Cashflow Network\n(Node size = total flow, Edge width = cashflow amount)',
              fontsize=22, fontweight='bold', fontfamily='Arial', pad=20)
    plt.axis('off')
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  - SDG大目标网络图已保存至: {output_file}")

    plt.close()


def generate_statistics(G, sdg_flows_df, output_dir='.'):
    """生成统计报告和CSV输出文件"""
    print("\n正在生成统计报告...")

    # 1. SDG网络边数据
    network_file = f'{output_dir}/sdg_cashflow_network.csv'

    # 转换为宽格式（每对SDG一行，正负面分列）
    pivot_data = sdg_flows_df.pivot_table(
        index=['source_sdg', 'target_sdg'],
        columns='impact_type',
        values=['cashflow', 'transaction_count'],
        fill_value=0
    ).reset_index()

    # 扁平化列名
    pivot_data.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                          for col in pivot_data.columns.values]

    # 重命名列
    rename_dict = {}
    for col in pivot_data.columns:
        if 'Positive' in col:
            if 'cashflow' in col:
                rename_dict[col] = 'positive_cashflow'
            elif 'transaction_count' in col:
                rename_dict[col] = 'positive_transaction_count'
        elif 'Negative' in col:
            if 'cashflow' in col:
                rename_dict[col] = 'negative_cashflow'
            elif 'transaction_count' in col:
                rename_dict[col] = 'negative_transaction_count'

    pivot_data = pivot_data.rename(columns=rename_dict)

    # 确保所有列都存在
    for col in ['positive_cashflow', 'negative_cashflow',
                'positive_transaction_count', 'negative_transaction_count']:
        if col not in pivot_data.columns:
            pivot_data[col] = 0

    # 计算净现金流
    pivot_data['net_cashflow'] = (pivot_data['positive_cashflow'] -
                                   pivot_data['negative_cashflow'])

    pivot_data.to_csv(network_file, index=False, encoding='utf-8-sig')
    print(f"  - SDG网络边数据已保存至: {network_file}")

    # 2. SDG节点统计摘要
    summary_data = []
    for node in G.nodes():
        # 正面流入
        pos_in = sum([d['weight'] for u, v, d in G.in_edges(node, data=True)
                     if d['impact_type'] == 'Positive'])
        # 正面流出
        pos_out = sum([d['weight'] for u, v, d in G.out_edges(node, data=True)
                      if d['impact_type'] == 'Positive'])
        # 负面流入
        neg_in = sum([d['weight'] for u, v, d in G.in_edges(node, data=True)
                     if d['impact_type'] == 'Negative'])
        # 负面流出
        neg_out = sum([d['weight'] for u, v, d in G.out_edges(node, data=True)
                      if d['impact_type'] == 'Negative'])

        summary_data.append({
            'sdg_target': node,
            'positive_inflow': pos_in,
            'positive_outflow': pos_out,
            'negative_inflow': neg_in,
            'negative_outflow': neg_out,
            'net_flow': (pos_in - pos_out) + (neg_out - neg_in)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('net_flow', ascending=False)

    summary_file = f'{output_dir}/sdg_summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"  - SDG节点统计已保存至: {summary_file}")

    # 3. Top 10 分析
    print("\n=== Top 10 正面影响现金流连接 ===")
    top_positive = sdg_flows_df[sdg_flows_df['impact_type'] == 'Positive'].nlargest(10, 'cashflow')
    for idx, row in top_positive.iterrows():
        print(f"  {row['source_sdg']} -> {row['target_sdg']}: "
              f"{row['cashflow']:,.0f} (交易次数: {row['transaction_count']:,})")

    print("\n=== Top 10 负面影响现金流连接 ===")
    top_negative = sdg_flows_df[sdg_flows_df['impact_type'] == 'Negative'].nlargest(10, 'cashflow')
    for idx, row in top_negative.iterrows():
        print(f"  {row['source_sdg']} -> {row['target_sdg']}: "
              f"{row['cashflow']:,.0f} (交易次数: {row['transaction_count']:,})")

    print("\n=== Top 10 最具影响力的SDG节点（按总流量） ===")
    # 计算总流量列
    summary_df['total_flow'] = (summary_df['positive_inflow'] + summary_df['positive_outflow'] +
                                summary_df['negative_inflow'] + summary_df['negative_outflow'])
    top_nodes = summary_df.nlargest(10, 'total_flow')
    for idx, row in top_nodes.iterrows():
        print(f"  {row['sdg_target']}: 总流量 {row['total_flow']:,.0f}")


def main():
    """主函数"""
    print("=" * 80)
    print("SDG现金流网络分析")
    print("=" * 80)

    # 文件路径（根据实际情况调整）
    mappings_file = '../preprocess/all_mappings.csv'
    cashflow_file = '../../data/202001'
    output_dir = '../../output'

    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 步骤1: 加载SDG映射
    sdg_mappings = load_sdg_mappings(mappings_file)

    # 步骤2: 加载并聚合交易数据
    industry_cashflow = load_and_aggregate_cashflow(cashflow_file)

    if len(industry_cashflow) == 0:
        print("错误: 没有有效的交易数据!")
        return

    # 步骤3: 映射到SDG
    sdg_flows = map_industry_to_sdg(industry_cashflow, sdg_mappings)

    if len(sdg_flows) == 0:
        print("错误: SDG映射失败!")
        return

    # 步骤4: 构建网络
    G = build_sdg_network(sdg_flows)

    # 步骤5: 可视化
    visualize_sdg_network(G, output_file=f'{output_dir}/sdg_network_visualization.png')

    # 步骤5.1: 可视化正面影响网络
    visualize_sdg_network_by_impact(G, impact_type='Positive',
                                     output_file=f'{output_dir}/sdg_network_positive.png')

    # 步骤5.2: 可视化负面影响网络
    visualize_sdg_network_by_impact(G, impact_type='Negative',
                                     output_file=f'{output_dir}/sdg_network_negative.png')

    # 步骤6: 聚合到SDG大目标并可视化
    sdg_goal_flows = aggregate_to_sdg_goals(sdg_flows)
    G_goal = build_sdg_network(sdg_goal_flows)
    visualize_sdg_goal_network(G_goal, output_file=f'{output_dir}/sdg_goal_network_visualization.png')

    # 步骤7: 生成统计报告
    generate_statistics(G, sdg_flows, output_dir=output_dir)

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
