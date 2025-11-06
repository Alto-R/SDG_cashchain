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
    """将行业间现金流映射到SDG子目标间现金流（拆分正负节点模式）"""
    print("\n正在将行业映射到SDG子目标（拆分正负节点）...")

    # 创建行业到SDG的映射字典（分正负面）
    industry_to_sdg_positive = sdg_mappings_df[sdg_mappings_df['impact_type'] == 'Positive'][
        ['industry_name', 'sdg_target_id']
    ].drop_duplicates()

    industry_to_sdg_negative = sdg_mappings_df[sdg_mappings_df['impact_type'] == 'Negative'][
        ['industry_name', 'sdg_target_id']
    ].drop_duplicates()

    print(f"  - 正面影响映射: {len(industry_to_sdg_positive)} 条")
    print(f"  - 负面影响映射: {len(industry_to_sdg_negative)} 条")

    all_flows = []

    for _, row in industry_cashflow_df.iterrows():
        source_industry = row['source_industry']
        target_industry = row['target_industry']
        cashflow = row['total_cashflow']
        trans_count = row['transaction_count']

        # 获取source的正负面SDG映射
        source_sdgs_pos = industry_to_sdg_positive[
            industry_to_sdg_positive['industry_name'] == source_industry
        ]['sdg_target_id'].values

        source_sdgs_neg = industry_to_sdg_negative[
            industry_to_sdg_negative['industry_name'] == source_industry
        ]['sdg_target_id'].values

        # 获取target的正负面SDG映射
        target_sdgs_pos = industry_to_sdg_positive[
            industry_to_sdg_positive['industry_name'] == target_industry
        ]['sdg_target_id'].values

        target_sdgs_neg = industry_to_sdg_negative[
            industry_to_sdg_negative['industry_name'] == target_industry
        ]['sdg_target_id'].values

        # 1. Positive → Positive (绿色内循环)
        for src_sdg in source_sdgs_pos:
            for tgt_sdg in target_sdgs_pos:
                all_flows.append({
                    'source_sdg': f"{src_sdg}_Pos",
                    'target_sdg': f"{tgt_sdg}_Pos",
                    'cashflow': cashflow,
                    'transaction_count': trans_count,
                    'flow_type': 'Pos_to_Pos'
                })

        # 2. Negative → Negative (棕色内循环)
        for src_sdg in source_sdgs_neg:
            for tgt_sdg in target_sdgs_neg:
                all_flows.append({
                    'source_sdg': f"{src_sdg}_Neg",
                    'target_sdg': f"{tgt_sdg}_Neg",
                    'cashflow': cashflow,
                    'transaction_count': trans_count,
                    'flow_type': 'Neg_to_Neg'
                })

        # 3. Positive → Negative (绿色依赖棕色)
        for src_sdg in source_sdgs_pos:
            for tgt_sdg in target_sdgs_neg:
                all_flows.append({
                    'source_sdg': f"{src_sdg}_Pos",
                    'target_sdg': f"{tgt_sdg}_Neg",
                    'cashflow': cashflow,
                    'transaction_count': trans_count,
                    'flow_type': 'Pos_to_Neg'
                })

        # 4. Negative → Positive (棕色转型投资)
        for src_sdg in source_sdgs_neg:
            for tgt_sdg in target_sdgs_pos:
                all_flows.append({
                    'source_sdg': f"{src_sdg}_Neg",
                    'target_sdg': f"{tgt_sdg}_Pos",
                    'cashflow': cashflow,
                    'transaction_count': trans_count,
                    'flow_type': 'Neg_to_Pos'
                })

    # 转换为DataFrame并聚合
    all_flows_df = pd.DataFrame(all_flows)

    if len(all_flows_df) == 0:
        print("警告: 没有找到匹配的SDG映射!")
        return pd.DataFrame()

    sdg_flows = all_flows_df.groupby(['source_sdg', 'target_sdg', 'flow_type']).agg({
        'cashflow': 'sum',
        'transaction_count': 'sum'
    }).reset_index()

    print(f"\n映射结果:")
    print(f"  - Pos→Pos流: {len(sdg_flows[sdg_flows['flow_type']=='Pos_to_Pos']):,} 条")
    print(f"  - Neg→Neg流: {len(sdg_flows[sdg_flows['flow_type']=='Neg_to_Neg']):,} 条")
    print(f"  - Pos→Neg流: {len(sdg_flows[sdg_flows['flow_type']=='Pos_to_Neg']):,} 条")
    print(f"  - Neg→Pos流: {len(sdg_flows[sdg_flows['flow_type']=='Neg_to_Pos']):,} 条")
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
            flow_type=row['flow_type']
        )

    print(f"  - 节点数: {G.number_of_nodes()}")
    print(f"  - 边数: {G.number_of_edges()}")

    return G


def visualize_sdg_network(G, output_file='sdg_network_visualization.png'):
    """可视化拆分正负节点的SDG网络图"""
    print(f"\n正在生成网络可视化图...")

    # 创建更大的图
    fig, ax = plt.subplots(figsize=(35, 35))

    print("  - 正在计算节点布局...")
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42, scale=2)

    # 分离正负节点
    pos_nodes = [n for n in G.nodes() if n.endswith('_Pos')]
    neg_nodes = [n for n in G.nodes() if n.endswith('_Neg')]

    # 计算节点大小（基于总流入+流出）
    node_sizes = {}
    for node in G.nodes():
        in_flow = sum([d['weight'] for u, v, d in G.in_edges(node, data=True)])
        out_flow = sum([d['weight'] for u, v, d in G.out_edges(node, data=True)])
        node_sizes[node] = in_flow + out_flow

    # 归一化节点大小
    max_size = max(node_sizes.values()) if node_sizes else 1
    node_sizes = {k: (v / max_size) * 2000 + 500 for k, v in node_sizes.items()}

    # 绘制正面节点（绿色）
    if pos_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=pos_nodes,
            node_size=[node_sizes[node] for node in pos_nodes],
            node_color='#90EE90',  # 浅绿色
            alpha=0.8,
            ax=ax
        )

    # 绘制负面节点（红色）
    if neg_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=neg_nodes,
            node_size=[node_sizes[node] for node in neg_nodes],
            node_color='#FFB3B3',  # 浅红色
            alpha=0.8,
            ax=ax
        )

    # 绘制标签（去掉后缀）
    labels = {n: n.replace('_Pos', '').replace('_Neg', '') for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_weight='bold',
        font_family='Arial',
        ax=ax
    )

    # 按类型绘制边
    edge_colors = {
        'Pos_to_Pos': '#228B22',  # 深绿色
        'Neg_to_Neg': '#DC143C',  # 深红色
        'Pos_to_Neg': '#FF8C00',  # 橙色
        'Neg_to_Pos': '#1E90FF'   # 蓝色
    }

    for flow_type, color in edge_colors.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['flow_type'] == flow_type]
        if edges:
            weights = [G[u][v][0]['weight'] for u, v in edges]
            max_w = max(weights) if weights else 1
            widths = [(w/max_w)*3 + 0.5 for w in weights]

            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                width=widths,
                edge_color=color,
                alpha=0.6,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive Node',
               markerfacecolor='#90EE90', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Negative Node',
               markerfacecolor='#FFB3B3', markersize=15),
        Line2D([0], [0], color='#228B22', lw=3, label='Pos → Pos (Green Economy)'),
        Line2D([0], [0], color='#DC143C', lw=3, label='Neg → Neg (Brown Economy)'),
        Line2D([0], [0], color='#FF8C00', lw=3, label='Pos → Neg (Green Dependency)'),
        Line2D([0], [0], color='#1E90FF', lw=3, label='Neg → Pos (Transition Investment)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16, prop={'family': 'Arial'})

    plt.title('SDG Network with Split Positive/Negative Nodes\n(Node size = total flow, Edge width = cashflow amount)',
              fontsize=22, fontweight='bold', fontfamily='Arial')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  - 网络图已保存至: {output_file}")
    plt.close()


def aggregate_to_sdg_goals(sdg_flows_df):
    """将SDG子目标聚合到SDG大目标（保留正负节点拆分）"""
    print("\n正在聚合到SDG大目标...")

    # 提取SDG大目标编号（例如从"1.1_Pos"提取"1_Pos"）
    def extract_goal(sdg_target):
        # 去掉后缀（_Pos 或 _Neg）
        base = str(sdg_target).replace('_Pos', '').replace('_Neg', '')
        goal = base.split('.')[0]
        # 恢复后缀
        if '_Pos' in str(sdg_target):
            return f"{goal}_Pos"
        elif '_Neg' in str(sdg_target):
            return f"{goal}_Neg"
        else:
            return goal

    # 添加SDG大目标列
    sdg_flows_df_copy = sdg_flows_df.copy()
    sdg_flows_df_copy['source_goal'] = sdg_flows_df_copy['source_sdg'].apply(extract_goal)
    sdg_flows_df_copy['target_goal'] = sdg_flows_df_copy['target_sdg'].apply(extract_goal)

    # 按大目标聚合
    goal_flows = sdg_flows_df_copy.groupby(['source_goal', 'target_goal', 'flow_type']).agg({
        'cashflow': 'sum',
        'transaction_count': 'sum'
    }).reset_index()

    # 重命名列以保持一致性
    goal_flows = goal_flows.rename(columns={
        'source_goal': 'source_sdg',
        'target_goal': 'target_sdg'
    })

    print(f"  - 聚合后SDG大目标流: {len(goal_flows):,} 条")
    print(f"  - 涉及 {goal_flows['source_sdg'].nunique()} 个SDG大目标节点")

    return goal_flows


def visualize_sdg_goal_network(G, output_file='sdg_goal_network_visualization.png'):
    """可视化SDG大目标网络图（拆分正负节点）"""
    print(f"\n正在生成SDG大目标网络可视化图...")

    # 创建更大的图
    fig, ax = plt.subplots(figsize=(28, 28))

    print("  - 正在计算节点布局...")
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42, scale=2)

    # 分离正负节点
    pos_nodes = [n for n in G.nodes() if n.endswith('_Pos')]
    neg_nodes = [n for n in G.nodes() if n.endswith('_Neg')]

    # 计算节点大小（基于总流入+流出）
    node_sizes = {}
    for node in G.nodes():
        in_flow = sum([d['weight'] for _, _, d in G.in_edges(node, data=True)])
        out_flow = sum([d['weight'] for _, _, d in G.out_edges(node, data=True)])
        node_sizes[node] = in_flow + out_flow

    # 归一化节点大小
    max_size = max(node_sizes.values()) if node_sizes else 1
    node_sizes = {k: (v / max_size) * 3500 + 1000 for k, v in node_sizes.items()}

    # 绘制正面节点（绿色）
    if pos_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=pos_nodes,
            node_size=[node_sizes[node] for node in pos_nodes],
            node_color='#90EE90',
            alpha=0.8,
            ax=ax
        )

    # 绘制负面节点（红色）
    if neg_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=neg_nodes,
            node_size=[node_sizes[node] for node in neg_nodes],
            node_color='#FFB3B3',
            alpha=0.8,
            ax=ax
        )

    # 绘制节点标签（去掉后缀，添加"SDG"前缀）
    labels = {node: f'SDG {node.replace("_Pos", "").replace("_Neg", "")}' for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=13,
        font_weight='bold',
        font_family='Arial',
        ax=ax
    )

    # 按类型绘制边
    edge_colors = {
        'Pos_to_Pos': '#228B22',
        'Neg_to_Neg': '#DC143C',
        'Pos_to_Neg': '#FF8C00',
        'Neg_to_Pos': '#1E90FF'
    }

    for flow_type, color in edge_colors.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['flow_type'] == flow_type]
        if edges:
            weights = [G[u][v][0]['weight'] for u, v in edges]
            max_w = max(weights) if weights else 1
            widths = [(w/max_w)*6 + 1 for w in weights]

            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                width=widths,
                edge_color=color,
                alpha=0.5,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.15',
                ax=ax
            )

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive Goal Node',
               markerfacecolor='#90EE90', markersize=18),
        Line2D([0], [0], marker='o', color='w', label='Negative Goal Node',
               markerfacecolor='#FFB3B3', markersize=18),
        Line2D([0], [0], color='#228B22', lw=4, label='Pos → Pos'),
        Line2D([0], [0], color='#DC143C', lw=4, label='Neg → Neg'),
        Line2D([0], [0], color='#FF8C00', lw=4, label='Pos → Neg'),
        Line2D([0], [0], color='#1E90FF', lw=4, label='Neg → Pos')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18, prop={'family': 'Arial'})

    plt.title('SDG Goal Cashflow Network (Split Nodes)\n(Node size = total flow, Edge width = cashflow amount)',
              fontsize=22, fontweight='bold', fontfamily='Arial', pad=20)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  - SDG大目标网络图已保存至: {output_file}")
    plt.close()


def generate_statistics(G, sdg_flows_df, output_dir='.'):
    """生成统计报告和CSV输出文件（拆分节点版本）"""
    print("\n正在生成统计报告...")

    # 1. SDG网络边数据
    network_file = f'{output_dir}/sdg_cashflow_network.csv'

    # 直接保存所有流，不需要pivot（因为有4种flow_type）
    sdg_flows_df.to_csv(network_file, index=False, encoding='utf-8-sig')
    print(f"  - SDG网络边数据已保存至: {network_file}")

    # 2. SDG节点统计摘要
    summary_data = []
    for node in G.nodes():
        # 按flow_type分别统计
        pos_to_pos_in = sum([d['weight'] for _, _, d in G.in_edges(node, data=True)
                             if d['flow_type'] == 'Pos_to_Pos'])
        pos_to_pos_out = sum([d['weight'] for _, _, d in G.out_edges(node, data=True)
                              if d['flow_type'] == 'Pos_to_Pos'])

        neg_to_neg_in = sum([d['weight'] for _, _, d in G.in_edges(node, data=True)
                             if d['flow_type'] == 'Neg_to_Neg'])
        neg_to_neg_out = sum([d['weight'] for _, _, d in G.out_edges(node, data=True)
                              if d['flow_type'] == 'Neg_to_Neg'])

        pos_to_neg_in = sum([d['weight'] for _, _, d in G.in_edges(node, data=True)
                             if d['flow_type'] == 'Pos_to_Neg'])
        pos_to_neg_out = sum([d['weight'] for _, _, d in G.out_edges(node, data=True)
                              if d['flow_type'] == 'Pos_to_Neg'])

        neg_to_pos_in = sum([d['weight'] for _, _, d in G.in_edges(node, data=True)
                             if d['flow_type'] == 'Neg_to_Pos'])
        neg_to_pos_out = sum([d['weight'] for _, _, d in G.out_edges(node, data=True)
                              if d['flow_type'] == 'Neg_to_Pos'])

        total_inflow = pos_to_pos_in + neg_to_neg_in + pos_to_neg_in + neg_to_pos_in
        total_outflow = pos_to_pos_out + neg_to_neg_out + pos_to_neg_out + neg_to_pos_out

        summary_data.append({
            'sdg_target': node,
            'total_inflow': total_inflow,
            'total_outflow': total_outflow,
            'net_flow': total_inflow - total_outflow,
            'pos_to_pos_in': pos_to_pos_in,
            'pos_to_pos_out': pos_to_pos_out,
            'neg_to_neg_in': neg_to_neg_in,
            'neg_to_neg_out': neg_to_neg_out,
            'pos_to_neg_in': pos_to_neg_in,
            'pos_to_neg_out': pos_to_neg_out,
            'neg_to_pos_in': neg_to_pos_in,
            'neg_to_pos_out': neg_to_pos_out
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('total_inflow', ascending=False)

    summary_file = f'{output_dir}/sdg_summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"  - SDG节点统计已保存至: {summary_file}")

    # 3. Top 10 分析
    print("\n=== Top 10 Pos→Pos现金流连接（绿色经济内循环） ===")
    top_pos_to_pos = sdg_flows_df[sdg_flows_df['flow_type'] == 'Pos_to_Pos'].nlargest(10, 'cashflow')
    for idx, row in top_pos_to_pos.iterrows():
        print(f"  {row['source_sdg']} -> {row['target_sdg']}: "
              f"{row['cashflow']:,.0f} (交易次数: {row['transaction_count']:,})")

    print("\n=== Top 10 Neg→Neg现金流连接（棕色经济内循环） ===")
    top_neg_to_neg = sdg_flows_df[sdg_flows_df['flow_type'] == 'Neg_to_Neg'].nlargest(10, 'cashflow')
    for idx, row in top_neg_to_neg.iterrows():
        print(f"  {row['source_sdg']} -> {row['target_sdg']}: "
              f"{row['cashflow']:,.0f} (交易次数: {row['transaction_count']:,})")

    print("\n=== Top 10 Pos→Neg现金流连接（绿色依赖棕色） ===")
    top_pos_to_neg = sdg_flows_df[sdg_flows_df['flow_type'] == 'Pos_to_Neg'].nlargest(10, 'cashflow')
    for idx, row in top_pos_to_neg.iterrows():
        print(f"  {row['source_sdg']} -> {row['target_sdg']}: "
              f"{row['cashflow']:,.0f} (交易次数: {row['transaction_count']:,})")

    print("\n=== Top 10 Neg→Pos现金流连接（棕色转型投资） ===")
    top_neg_to_pos = sdg_flows_df[sdg_flows_df['flow_type'] == 'Neg_to_Pos'].nlargest(10, 'cashflow')
    for idx, row in top_neg_to_pos.iterrows():
        print(f"  {row['source_sdg']} -> {row['target_sdg']}: "
              f"{row['cashflow']:,.0f} (交易次数: {row['transaction_count']:,})")

    print("\n=== Top 10 最具影响力的SDG节点（按总流入） ===")
    top_nodes = summary_df.nlargest(10, 'total_inflow')
    for idx, row in top_nodes.iterrows():
        print(f"  {row['sdg_target']}: 总流入 {row['total_inflow']:,.0f}, "
              f"总流出 {row['total_outflow']:,.0f}, "
              f"净流 {row['net_flow']:,.0f}")


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

    # 步骤5: 可视化拆分节点网络
    visualize_sdg_network(G, output_file=f'{output_dir}/sdg_split_node_network.png')

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
