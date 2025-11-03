# -*- coding: utf-8 -*-
"""
SDG网络的高级分析
计算中心性指标、社区检测等网络科学度量
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


def load_network_from_csv(edges_file, nodes_file):
    """从CSV文件加载网络"""
    print("正在从CSV加载网络...")

    # 加载边数据
    edges_df = pd.read_csv(edges_file, encoding='utf-8-sig')

    # 加载节点数据
    nodes_df = pd.read_csv(nodes_file, encoding='utf-8-sig')

    # 构建网络
    G = nx.MultiDiGraph()

    # 添加节点
    for _, node in nodes_df.iterrows():
        G.add_node(
            node['sdg_target'],
            positive_inflow=node['positive_inflow'],
            positive_outflow=node['positive_outflow'],
            negative_inflow=node['negative_inflow'],
            negative_outflow=node['negative_outflow'],
            net_flow=node['net_flow']
        )

    # 添加边
    for _, edge in edges_df.iterrows():
        # 正面影响边
        if edge['positive_cashflow'] > 0:
            G.add_edge(
                edge['source_sdg'],
                edge['target_sdg'],
                weight=edge['positive_cashflow'],
                impact_type='Positive',
                transaction_count=edge['positive_transaction_count']
            )

        # 负面影响边
        if edge['negative_cashflow'] > 0:
            G.add_edge(
                edge['source_sdg'],
                edge['target_sdg'],
                weight=edge['negative_cashflow'],
                impact_type='Negative',
                transaction_count=edge['negative_transaction_count']
            )

    print(f"  - 加载了 {G.number_of_nodes()} 个节点")
    print(f"  - 加载了 {G.number_of_edges()} 条边")

    return G, edges_df, nodes_df


def create_simplified_graph(G):
    """创建简化的单边图（用于某些网络分析）"""
    print("\n正在创建简化网络...")

    G_simple = nx.DiGraph()

    # 聚合同方向的所有边
    edge_weights = defaultdict(lambda: {'positive': 0, 'negative': 0, 'total': 0})

    for u, v, data in G.edges(data=True):
        impact_type = data['impact_type']
        weight = data['weight']

        if impact_type == 'Positive':
            edge_weights[(u, v)]['positive'] += weight
        else:
            edge_weights[(u, v)]['negative'] += weight

        edge_weights[(u, v)]['total'] += weight

    # 添加聚合后的边
    for (u, v), weights in edge_weights.items():
        G_simple.add_edge(
            u, v,
            weight=weights['total'],
            positive_weight=weights['positive'],
            negative_weight=weights['negative']
        )

    print(f"  - 简化网络: {G_simple.number_of_nodes()} 个节点")
    print(f"  - 简化网络: {G_simple.number_of_edges()} 条边")

    return G_simple


def analyze_centrality(G_simple, output_dir='.'):
    """计算各种中心性指标"""
    print("\n正在计算中心性指标...")

    results = {}

    # 1. 度中心性
    print("  - 计算度中心性...")
    in_degree = dict(G_simple.in_degree(weight='weight'))
    out_degree = dict(G_simple.out_degree(weight='weight'))

    # 2. PageRank
    print("  - 计算PageRank...")
    pagerank = nx.pagerank(G_simple, weight='weight')

    # 3. 中介中心性
    print("  - 计算中介中心性（这可能需要一些时间）...")
    betweenness = nx.betweenness_centrality(G_simple, weight='weight')

    # 4. 接近中心性
    print("  - 计算接近中心性...")
    try:
        closeness = nx.closeness_centrality(G_simple, distance='weight')
    except:
        closeness = {node: 0 for node in G_simple.nodes()}
        print("    警告: 网络可能不连通，接近中心性可能不准确")

    # 5. 特征向量中心性
    print("  - 计算特征向量中心性...")
    try:
        eigenvector = nx.eigenvector_centrality(G_simple, weight='weight', max_iter=1000)
    except:
        eigenvector = {node: 0 for node in G_simple.nodes()}
        print("    警告: 特征向量中心性计算失败，可能未收敛")

    # 6. 调和中心性
    print("  - 计算调和中心性...")
    harmonic = nx.harmonic_centrality(G_simple, distance='weight')

    # 整合结果
    centrality_df = pd.DataFrame({
        'node': list(G_simple.nodes()),
        'in_degree': [in_degree[n] for n in G_simple.nodes()],
        'out_degree': [out_degree[n] for n in G_simple.nodes()],
        'pagerank': [pagerank[n] for n in G_simple.nodes()],
        'betweenness': [betweenness[n] for n in G_simple.nodes()],
        'closeness': [closeness[n] for n in G_simple.nodes()],
        'eigenvector': [eigenvector[n] for n in G_simple.nodes()],
        'harmonic': [harmonic[n] for n in G_simple.nodes()]
    })

    # 计算总度数
    centrality_df['total_degree'] = centrality_df['in_degree'] + centrality_df['out_degree']

    # 排序
    centrality_df = centrality_df.sort_values('pagerank', ascending=False)

    # 保存
    output_file = f'{output_dir}/sdg_centrality_analysis.csv'
    centrality_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n中心性分析结果已保存至: {output_file}")

    # 打印Top 10
    print("\n=== Top 10 节点（按PageRank）===")
    print(centrality_df.head(10)[['node', 'pagerank', 'betweenness', 'total_degree']])

    return centrality_df


def detect_communities(G_simple, output_dir='.'):
    """社区检测"""
    print("\n正在进行社区检测...")

    # 转换为无向图（某些算法需要）
    G_undirected = G_simple.to_undirected()

    # 使用Louvain算法
    try:
        from community import community_louvain
        communities = community_louvain.best_partition(G_undirected, weight='weight')
        modularity = community_louvain.modularity(communities, G_undirected, weight='weight')
        print(f"  - Louvain算法检测到 {len(set(communities.values()))} 个社区")
        print(f"  - 模块度: {modularity:.4f}")
    except ImportError:
        print("  - 警告: python-louvain未安装，跳过Louvain算法")
        communities = None
        modularity = None

    # 使用贪心模块化算法（NetworkX内置）
    print("  - 使用贪心模块化算法...")
    from networkx.algorithms import community as nx_comm
    greedy_communities = list(nx_comm.greedy_modularity_communities(G_undirected, weight='weight'))
    greedy_modularity = nx_comm.modularity(G_undirected, greedy_communities, weight='weight')

    print(f"  - 贪心算法检测到 {len(greedy_communities)} 个社区")
    print(f"  - 模块度: {greedy_modularity:.4f}")

    # 转换为节点-社区映射
    greedy_communities_dict = {}
    for i, comm in enumerate(greedy_communities):
        for node in comm:
            greedy_communities_dict[node] = i

    # 保存社区结果
    community_df = pd.DataFrame({
        'node': list(G_simple.nodes()),
        'greedy_community': [greedy_communities_dict.get(n, -1) for n in G_simple.nodes()]
    })

    if communities:
        community_df['louvain_community'] = [communities.get(n, -1) for n in G_simple.nodes()]

    output_file = f'{output_dir}/sdg_community_detection.csv'
    community_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n社区检测结果已保存至: {output_file}")

    # 打印每个社区的成员
    print("\n=== 社区组成 ===")
    for i, comm in enumerate(greedy_communities):
        print(f"社区 {i}: {sorted(list(comm))}")

    return community_df, greedy_communities


def analyze_network_structure(G_simple):
    """分析网络基本结构特征"""
    print("\n正在分析网络结构...")

    stats = {}

    # 基本统计
    stats['节点数'] = G_simple.number_of_nodes()
    stats['边数'] = G_simple.number_of_edges()
    stats['网络密度'] = nx.density(G_simple)

    # 度分布
    degrees = [d for n, d in G_simple.degree()]
    stats['平均度数'] = np.mean(degrees)
    stats['度数标准差'] = np.std(degrees)

    # 连通性
    stats['强连通'] = nx.is_strongly_connected(G_simple)
    stats['弱连通'] = nx.is_weakly_connected(G_simple)

    if not nx.is_strongly_connected(G_simple):
        scc = list(nx.strongly_connected_components(G_simple))
        stats['强连通分量数'] = len(scc)
        stats['最大强连通分量大小'] = len(max(scc, key=len))

    # 互惠性
    stats['互惠性'] = nx.reciprocity(G_simple)

    # 聚类系数
    print("  - 计算聚类系数...")
    G_undirected = G_simple.to_undirected()
    stats['平均聚类系数'] = nx.average_clustering(G_undirected, weight='weight')
    stats['全局聚类系数（传递性）'] = nx.transitivity(G_undirected)

    # 同配性（度数相关性）
    print("  - 计算同配性...")
    try:
        stats['度同配性'] = nx.degree_assortativity_coefficient(G_simple)
    except:
        stats['度同配性'] = None
        print("    警告: 度同配性计算失败")

    # 度分布熵
    print("  - 计算度分布熵...")
    degrees = [d for n, d in G_simple.degree()]
    degree_counts = pd.Series(degrees).value_counts()
    degree_probs = degree_counts / len(degrees)
    stats['度分布熵'] = -sum(degree_probs * np.log2(degree_probs))

    # 平均路径长度（如果连通）
    if nx.is_strongly_connected(G_simple):
        stats['平均最短路径长度'] = nx.average_shortest_path_length(G_simple, weight='weight')

    # 打印统计
    print("\n=== 网络结构统计 ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return stats


def analyze_edge_distribution(G):
    """分析边权重分布"""
    print("\n正在分析边权重分布...")

    positive_weights = []
    negative_weights = []

    for u, v, data in G.edges(data=True):
        if data['impact_type'] == 'Positive':
            positive_weights.append(data['weight'])
        else:
            negative_weights.append(data['weight'])

    print(f"\n正面影响边统计:")
    print(f"  - 数量: {len(positive_weights)}")
    print(f"  - 总权重: {sum(positive_weights):,.0f}")
    print(f"  - 平均权重: {np.mean(positive_weights):,.0f}")
    print(f"  - 中位数权重: {np.median(positive_weights):,.0f}")
    print(f"  - 最大权重: {max(positive_weights):,.0f}")
    print(f"  - 最小权重: {min(positive_weights):,.0f}")

    print(f"\n负面影响边统计:")
    print(f"  - 数量: {len(negative_weights)}")
    print(f"  - 总权重: {sum(negative_weights):,.0f}")
    print(f"  - 平均权重: {np.mean(negative_weights):,.0f}")
    print(f"  - 中位数权重: {np.median(negative_weights):,.0f}")
    print(f"  - 最大权重: {max(negative_weights):,.0f}")
    print(f"  - 最小权重: {min(negative_weights):,.0f}")

    return {
        'positive': {
            'count': len(positive_weights),
            'sum': sum(positive_weights),
            'mean': np.mean(positive_weights),
            'median': np.median(positive_weights),
            'max': max(positive_weights),
            'min': min(positive_weights)
        },
        'negative': {
            'count': len(negative_weights),
            'sum': sum(negative_weights),
            'mean': np.mean(negative_weights),
            'median': np.median(negative_weights),
            'max': max(negative_weights),
            'min': min(negative_weights)
        }
    }


def identify_key_pathways(G, top_n=20):
    """识别关键路径"""
    print(f"\n正在识别Top {top_n}关键路径...")

    # 收集所有边及其权重
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'weight': data['weight'],
            'impact_type': data['impact_type'],
            'transaction_count': data.get('transaction_count', 0)
        })

    edges_df = pd.DataFrame(edges_data)

    # 正面影响Top N
    print(f"\n=== Top {top_n} 正面影响路径 ===")
    top_positive = edges_df[edges_df['impact_type'] == 'Positive'].nlargest(top_n, 'weight')
    for idx, row in top_positive.iterrows():
        print(f"{row['source']} -> {row['target']}: {row['weight']:,.0f} "
              f"(交易次数: {row['transaction_count']:,.0f})")

    # 负面影响Top N
    print(f"\n=== Top {top_n} 负面影响路径 ===")
    top_negative = edges_df[edges_df['impact_type'] == 'Negative'].nlargest(top_n, 'weight')
    for idx, row in top_negative.iterrows():
        print(f"{row['source']} -> {row['target']}: {row['weight']:,.0f} "
              f"(交易次数: {row['transaction_count']:,.0f})")

    return edges_df


def analyze_clustering(G_simple, output_dir='.'):
    """计算节点级别的聚类系数"""
    print("\n正在计算节点级别聚类系数...")

    # 转换为无向图
    G_undirected = G_simple.to_undirected()

    # 计算每个节点的聚类系数
    clustering_coeffs = nx.clustering(G_undirected, weight='weight')

    # 创建DataFrame
    clustering_df = pd.DataFrame({
        'node': list(clustering_coeffs.keys()),
        'clustering_coefficient': list(clustering_coeffs.values())
    })

    # 排序
    clustering_df = clustering_df.sort_values('clustering_coefficient', ascending=False)

    # 保存
    output_file = f'{output_dir}/sdg_clustering_analysis.csv'
    clustering_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"聚类系数分析已保存至: {output_file}")

    # 打印Top 10
    print("\n=== Top 10 节点（按聚类系数）===")
    print(clustering_df.head(10))

    return clustering_df


def analyze_k_core(G_simple, output_dir='.'):
    """k-core分解分析"""
    print("\n正在进行k-core分解...")

    # 转换为无向图
    G_undirected = G_simple.to_undirected()

    # 移除自环（k-core算法不允许自环）
    print("  - 移除自环边...")
    G_no_selfloop = G_undirected.copy()
    selfloop_edges = list(nx.selfloop_edges(G_no_selfloop))
    G_no_selfloop.remove_edges_from(selfloop_edges)
    print(f"    移除了 {len(selfloop_edges)} 条自环边")

    # 计算每个节点的core number
    core_numbers = nx.core_number(G_no_selfloop)

    # 创建DataFrame
    k_core_df = pd.DataFrame({
        'node': list(core_numbers.keys()),
        'core_number': list(core_numbers.values())
    })

    # 排序
    k_core_df = k_core_df.sort_values('core_number', ascending=False)

    # 保存
    output_file = f'{output_dir}/sdg_k_core_structure.csv'
    k_core_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"k-core结构分析已保存至: {output_file}")

    # 统计各层的节点数
    print("\n=== k-core层统计 ===")
    core_distribution = k_core_df['core_number'].value_counts().sort_index(ascending=False)
    for k, count in core_distribution.items():
        print(f"k={k}: {count} 个节点")

    # 可视化k-core结构
    try:
        plt.figure(figsize=(12, 6))

        # 子图1：核心数分布
        plt.subplot(1, 2, 1)
        core_distribution.sort_index().plot(kind='bar')
        plt.xlabel('Core Number (k)', fontsize=12)
        plt.ylabel('Number of Nodes', fontsize=12)
        plt.title('k-core Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        # 子图2：累积分布
        plt.subplot(1, 2, 2)
        cumulative = core_distribution.sort_index(ascending=False).cumsum()
        cumulative.plot(kind='line', marker='o')
        plt.xlabel('Core Number (k)', fontsize=12)
        plt.ylabel('Cumulative Nodes', fontsize=12)
        plt.title('Cumulative k-core Distribution', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        viz_file = f'{output_dir}/k_core_layers.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"\nk-core可视化已保存至: {viz_file}")
        plt.close()
    except Exception as e:
        print(f"可视化生成失败: {e}")

    return k_core_df


def analyze_robustness(G_simple, output_dir='.'):
    """节点移除鲁棒性分析"""
    print("\n正在分析网络鲁棒性...")
    print("  （注意：这可能需要较长时间）")

    # 计算初始网络效率
    print("  - 计算初始网络效率...")
    G_undirected = G_simple.to_undirected()
    initial_efficiency = nx.global_efficiency(G_undirected)
    print(f"    初始全局效率: {initial_efficiency:.4f}")

    # 测试移除每个节点的影响
    robustness_results = []
    nodes = list(G_simple.nodes())

    print(f"  - 测试移除 {len(nodes)} 个节点的影响...")
    for i, node in enumerate(nodes):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i+1}/{len(nodes)}")

        # 移除节点
        G_temp = G_undirected.copy()
        G_temp.remove_node(node)

        # 计算新的效率
        try:
            new_efficiency = nx.global_efficiency(G_temp)
            efficiency_drop = initial_efficiency - new_efficiency
            efficiency_drop_pct = (efficiency_drop / initial_efficiency) * 100
        except:
            efficiency_drop = initial_efficiency
            efficiency_drop_pct = 100.0

        robustness_results.append({
            'node': node,
            'efficiency_drop': efficiency_drop,
            'efficiency_drop_pct': efficiency_drop_pct,
            'remaining_nodes': G_temp.number_of_nodes(),
            'remaining_edges': G_temp.number_of_edges()
        })

    # 创建DataFrame
    robustness_df = pd.DataFrame(robustness_results)
    robustness_df = robustness_df.sort_values('efficiency_drop_pct', ascending=False)

    # 保存
    output_file = f'{output_dir}/sdg_robustness_analysis.csv'
    robustness_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n鲁棒性分析已保存至: {output_file}")

    # 打印Top 10最关键节点
    print("\n=== Top 10 最关键节点（移除影响最大）===")
    print(robustness_df.head(10)[['node', 'efficiency_drop_pct']])

    # 可视化
    try:
        plt.figure(figsize=(14, 6))

        # 子图1：Top 20节点的脆弱性
        plt.subplot(1, 2, 1)
        top_20 = robustness_df.head(20)
        plt.barh(range(len(top_20)), top_20['efficiency_drop_pct'])
        plt.yticks(range(len(top_20)), top_20['node'])
        plt.xlabel('Efficiency Drop (%)', fontsize=12)
        plt.ylabel('Node', fontsize=12)
        plt.title('Top 20 Critical Nodes\n(Network Vulnerability)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)

        # 子图2：效率下降分布
        plt.subplot(1, 2, 2)
        plt.hist(robustness_df['efficiency_drop_pct'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Efficiency Drop (%)', fontsize=12)
        plt.ylabel('Number of Nodes', fontsize=12)
        plt.title('Distribution of Node Criticality', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        viz_file = f'{output_dir}/robustness_heatmap.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"\n鲁棒性可视化已保存至: {viz_file}")
        plt.close()
    except Exception as e:
        print(f"可视化生成失败: {e}")

    return robustness_df


def calculate_flow_betweenness(G_simple):
    """计算流介数（基于电流模型）"""
    print("\n正在计算流介数...")

    # 转换为无向图
    G_undirected = G_simple.to_undirected()

    # 使用current_flow_betweenness作为流介数的近似
    # 这基于电流在网络中的流动
    print("  - 使用电流流动模型计算...")
    try:
        flow_betweenness = nx.current_flow_betweenness_centrality(G_undirected, weight='weight')
        print("  - 流介数计算完成")
    except:
        print("  - 警告: 流介数计算失败，网络可能不连通")
        flow_betweenness = {node: 0 for node in G_undirected.nodes()}

    # 创建DataFrame
    flow_df = pd.DataFrame({
        'node': list(flow_betweenness.keys()),
        'flow_betweenness': list(flow_betweenness.values())
    })

    flow_df = flow_df.sort_values('flow_betweenness', ascending=False)

    print("\n=== Top 10 节点（按流介数）===")
    print(flow_df.head(10))

    return flow_df


def main():
    """主函数"""
    print("=" * 80)
    print("SDG网络高级分析")
    print("=" * 80)

    # 文件路径
    edges_file = '../../output/sdg_cashflow_network.csv'
    nodes_file = '../../output/sdg_summary_statistics.csv'
    output_dir = '../../output'

    # 1. 加载网络
    G, edges_df, nodes_df = load_network_from_csv(edges_file, nodes_file)

    # 2. 创建简化图
    G_simple = create_simplified_graph(G)

    # 3. 网络结构分析
    network_stats = analyze_network_structure(G_simple)

    # 4. 边权重分布分析
    edge_stats = analyze_edge_distribution(G)

    # 5. 中心性分析
    centrality_df = analyze_centrality(G_simple, output_dir)

    # 6. 社区检测
    community_df, communities = detect_communities(G_simple, output_dir)

    # 7. 识别关键路径
    edges_analysis = identify_key_pathways(G, top_n=20)

    # 8. 节点级别聚类系数分析
    clustering_df = analyze_clustering(G_simple, output_dir)

    # 9. k-core分解
    k_core_df = analyze_k_core(G_simple, output_dir)

    # 10. 流介数分析
    flow_betweenness_df = calculate_flow_betweenness(G_simple)

    # 11. 鲁棒性分析（可能耗时较长）
    robustness_df = analyze_robustness(G_simple, output_dir)

    print("\n" + "=" * 80)
    print("高级分析完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print(f"  - sdg_centrality_analysis.csv (中心性指标 - 已扩展)")
    print(f"  - sdg_community_detection.csv (社区检测结果)")
    print(f"  - sdg_clustering_analysis.csv (节点聚类系数)")
    print(f"  - sdg_k_core_structure.csv (k-core结构)")
    print(f"  - sdg_robustness_analysis.csv (鲁棒性分析)")
    print(f"  - k_core_layers.png (k-core可视化)")
    print(f"  - robustness_heatmap.png (鲁棒性可视化)")


if __name__ == '__main__':
    main()
