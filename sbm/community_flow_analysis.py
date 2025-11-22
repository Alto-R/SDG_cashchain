"""
社团资金流动分析脚本
分析每个社团内部节点之间的流动以及社团与社团之间的流动额度
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CommunityFlowAnalyzer:
    """社团资金流动分析器"""

    def __init__(self, membership_file, cashflow_file, output_dir):
        """
        初始化分析器

        Args:
            membership_file: 社团成员文件路径
            cashflow_file: 资金流动网络文件路径
            output_dir: 输出目录路径
        """
        self.membership_file = membership_file
        self.cashflow_file = cashflow_file
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures'

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # 数据容器
        self.membership_df = None
        self.cashflow_df = None
        self.node_to_community = {}
        self.intra_community_flows = None
        self.inter_community_flows = None
        self.flow_matrix = None

    def load_data(self):
        """加载数据文件"""
        print("正在加载数据文件...")

        # 读取社团成员数据
        self.membership_df = pd.read_csv(self.membership_file)
        print(f"已加载社团成员数据: {len(self.membership_df)} 个节点")

        # 读取资金流动数据
        self.cashflow_df = pd.read_csv(self.cashflow_file)
        print(f"已加载资金流动数据: {len(self.cashflow_df)} 条流动记录")

        # 构建节点到社团的映射
        self.node_to_community = dict(zip(
            self.membership_df['node'],
            self.membership_df['community']
        ))

        # 为cashflow数据添加社团信息
        self.cashflow_df['source_community'] = self.cashflow_df['source_sdg'].map(self.node_to_community)
        self.cashflow_df['target_community'] = self.cashflow_df['target_sdg'].map(self.node_to_community)

        # 删除缺失社团信息的记录
        missing_before = len(self.cashflow_df)
        self.cashflow_df = self.cashflow_df.dropna(subset=['source_community', 'target_community'])
        missing_after = len(self.cashflow_df)
        if missing_before > missing_after:
            print(f"警告: 删除了 {missing_before - missing_after} 条缺失社团信息的记录")

        # 转换社团编号为整数
        self.cashflow_df['source_community'] = self.cashflow_df['source_community'].astype(int)
        self.cashflow_df['target_community'] = self.cashflow_df['target_community'].astype(int)

        print("数据加载完成!\n")

    def analyze_intra_community_flows(self):
        """分析社团内部流动"""
        print("正在分析社团内部流动...")

        # 筛选社团内部的流动（source和target在同一社团）
        intra_flows = self.cashflow_df[
            self.cashflow_df['source_community'] == self.cashflow_df['target_community']
        ].copy()

        # 按社团分组统计
        intra_stats = []
        for community in sorted(self.cashflow_df['source_community'].unique()):
            comm_flows = intra_flows[intra_flows['source_community'] == community]

            if len(comm_flows) == 0:
                intra_stats.append({
                    'community': community,
                    'total_flow': 0,
                    'transaction_count': 0,
                    'num_edges': 0,
                    'pos_to_pos_flow': 0,
                    'pos_to_neg_flow': 0,
                    'neg_to_pos_flow': 0,
                    'neg_to_neg_flow': 0
                })
                continue

            # 按流动类型分组
            flow_by_type = comm_flows.groupby('flow_type')['cashflow'].sum()

            intra_stats.append({
                'community': community,
                'total_flow': comm_flows['cashflow'].sum(),
                'transaction_count': comm_flows['transaction_count'].sum(),
                'num_edges': len(comm_flows),
                'pos_to_pos_flow': flow_by_type.get('Pos_to_Pos', 0),
                'pos_to_neg_flow': flow_by_type.get('Pos_to_Neg', 0),
                'neg_to_pos_flow': flow_by_type.get('Neg_to_Pos', 0),
                'neg_to_neg_flow': flow_by_type.get('Neg_to_Neg', 0)
            })

        self.intra_community_flows = pd.DataFrame(intra_stats)

        # 计算百分比
        total_intra_flow = self.intra_community_flows['total_flow'].sum()
        self.intra_community_flows['flow_percentage'] = (
            self.intra_community_flows['total_flow'] / total_intra_flow * 100
        ).round(2)

        print(f"社团内部流动分析完成!")
        print(f"总社团内部流动额: {total_intra_flow:,.0f}")
        print(f"涉及 {len(self.intra_community_flows)} 个社团\n")

    def analyze_inter_community_flows(self):
        """分析社团间流动"""
        print("正在分析社团间流动...")

        # 筛选社团间的流动（source和target不在同一社团）
        inter_flows = self.cashflow_df[
            self.cashflow_df['source_community'] != self.cashflow_df['target_community']
        ].copy()

        # 按社团对分组统计
        inter_stats = inter_flows.groupby(
            ['source_community', 'target_community']
        ).agg({
            'cashflow': 'sum',
            'transaction_count': 'sum',
            'source_sdg': 'count'  # 边的数量
        }).reset_index()

        inter_stats.columns = [
            'source_community', 'target_community',
            'total_flow', 'transaction_count', 'num_edges'
        ]

        self.inter_community_flows = inter_stats

        # 构建流动矩阵
        communities = sorted(self.cashflow_df['source_community'].unique())
        matrix_size = len(communities)
        flow_matrix = pd.DataFrame(
            0.0,
            index=communities,
            columns=communities
        )

        for _, row in inter_stats.iterrows():
            flow_matrix.loc[row['source_community'], row['target_community']] = row['total_flow']

        # 将社团内部流动添加到对角线
        for _, row in self.intra_community_flows.iterrows():
            community = row['community']
            flow_matrix.loc[community, community] = row['total_flow']

        self.flow_matrix = flow_matrix

        print(f"社团间流动分析完成!")
        print(f"总社团间流动额: {inter_stats['total_flow'].sum():,.0f}")
        print(f"涉及 {len(inter_stats)} 个社团对\n")

    def save_results(self):
        """保存分析结果到CSV文件"""
        print("正在保存分析结果...")

        # 保存社团内部流动统计
        intra_file = self.output_dir / 'intra_community_flows.csv'
        self.intra_community_flows.to_csv(intra_file, index=False, encoding='utf-8-sig')
        print(f"已保存: {intra_file}")

        # 保存社团间流动统计
        inter_file = self.output_dir / 'inter_community_flows.csv'
        self.inter_community_flows.to_csv(inter_file, index=False, encoding='utf-8-sig')
        print(f"已保存: {inter_file}")

        # 保存流动矩阵
        matrix_file = self.output_dir / 'community_flow_matrix.csv'
        self.flow_matrix.to_csv(matrix_file, encoding='utf-8-sig')
        print(f"已保存: {matrix_file}\n")

    def plot_flow_matrix_heatmap(self):
        """绘制每个社团的内部流动和外部流动饼图"""
        print("正在生成各社团内部/外部流动饼图...")

        # 获取所有社团
        communities = sorted(self.flow_matrix.index)
        n_communities = len(communities)

        # 创建子图布局 (例如：5行2列显示10个社团)
        n_cols = 3
        n_rows = (n_communities + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        axes = axes.flatten() if n_communities > 1 else [axes]

        # 为每个社团生成饼图
        for idx, community in enumerate(communities):
            ax = axes[idx]

            # 计算该社团的流动数据
            intra_flow = self.flow_matrix.loc[community, community]  # 内部流动（对角线）

            # 外部流出：该社团作为源，流向其他社团的总和
            outflow = self.flow_matrix.loc[community, :].sum() - intra_flow

            # 外部流入：其他社团流向该社团的总和
            inflow = self.flow_matrix.loc[:, community].sum() - intra_flow

            # 数据和标签
            sizes = [intra_flow, outflow, inflow]
            labels = [
                f'Intra\n{intra_flow/1e6:.1f}M',
                f'Outflow\n{outflow/1e6:.1f}M',
                f'Inflow\n{inflow/1e6:.1f}M'
            ]
            colors = ['#3498db', '#e74c3c', '#2ecc71']

            # 只显示非零的部分
            sizes_filtered = []
            labels_filtered = []
            colors_filtered = []
            for s, l, c in zip(sizes, labels, colors):
                if s > 0:
                    sizes_filtered.append(s)
                    labels_filtered.append(l)
                    colors_filtered.append(c)

            if sum(sizes_filtered) > 0:
                _, _, autotexts = ax.pie(
                    sizes_filtered,
                    labels=labels_filtered,
                    colors=colors_filtered,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 9}
                )
                # 设置百分比文字为白色粗体
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                ax.text(0.5, 0.5, 'No Flow', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)

            ax.set_title(f'Community {community}', fontsize=12, fontweight='bold')

        # 隐藏多余的子图
        for idx in range(n_communities, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Cash Flow Composition by Community (Intra vs Inter-Community)',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = self.figures_dir / 'community_flow_composition_pies.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存: {output_file}")

    def plot_intra_community_bars(self):
        """绘制各社团内部流动柱状图"""
        print("正在生成社团内部流动柱状图...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 柱状图1: 总流动额
        communities = self.intra_community_flows['community']
        total_flows = self.intra_community_flows['total_flow']

        bars1 = ax1.bar(communities, total_flows, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Community', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Cash Flow', fontsize=12, fontweight='bold')
        ax1.set_title('Total Intra-Community Cash Flow by Community',
                      fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticks(communities)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height/1e6:.1f}M',
                        ha='center', va='bottom', fontsize=9)

        # 柱状图2: 按流动类型堆叠
        flow_types = ['pos_to_pos_flow', 'pos_to_neg_flow', 'neg_to_pos_flow', 'neg_to_neg_flow']
        flow_labels = ['Pos→Pos', 'Pos→Neg', 'Neg→Pos', 'Neg→Neg']
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

        bottom = np.zeros(len(communities))
        for flow_type, label, color in zip(flow_types, flow_labels, colors):
            values = self.intra_community_flows[flow_type]
            ax2.bar(communities, values, bottom=bottom, label=label,
                   color=color, alpha=0.7, edgecolor='black')
            bottom += values

        ax2.set_xlabel('Community', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cash Flow by Type', fontsize=12, fontweight='bold')
        ax2.set_title('Intra-Community Cash Flow by Flow Type',
                      fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks(communities)

        plt.tight_layout()
        output_file = self.figures_dir / 'intra_community_flow_bars.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存: {output_file}")

    def plot_sankey_diagram(self):
        """绘制社团间流动桑基图（使用plotly）"""
        print("正在生成社团间流动桑基图...")

        try:
            import plotly.graph_objects as go

            # 选择Top 30的流动关系
            top_flows = self.inter_community_flows.nlargest(30, 'total_flow')

            # 创建节点标签
            all_communities = sorted(
                set(top_flows['source_community']) | set(top_flows['target_community'])
            )
            node_labels = [f"Community {c}" for c in all_communities]

            # 创建节点索引映射
            node_indices = {comm: idx for idx, comm in enumerate(all_communities)}

            # 准备桑基图数据
            source_indices = [node_indices[s] for s in top_flows['source_community']]
            target_indices = [node_indices[t] for t in top_flows['target_community']]
            values = top_flows['total_flow'].tolist()

            # 创建桑基图
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color='steelblue'
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color='rgba(0, 100, 200, 0.3)'
                )
            )])

            fig.update_layout(
                title_text="Top 30 Inter-Community Cash Flows (Sankey Diagram)",
                title_font_size=16,
                font_size=10,
                height=800
            )

            output_file = self.figures_dir / 'inter_community_sankey.html'
            fig.write_html(str(output_file))
            print(f"已保存: {output_file}")

        except ImportError:
            print("警告: plotly未安装，跳过桑基图生成。可以运行 'pip install plotly' 来安装。")

    def plot_flow_comparison_pie(self):
        """绘制社团内部流动vs社团间流动饼图"""
        print("正在生成流动对比饼图...")

        total_intra = self.intra_community_flows['total_flow'].sum()
        total_inter = self.inter_community_flows['total_flow'].sum()

        fig, ax = plt.subplots(figsize=(10, 8))

        sizes = [total_intra, total_inter]
        labels = [
            f'Intra-Community Flow\n{total_intra/1e6:.1f}M ({total_intra/(total_intra+total_inter)*100:.1f}%)',
            f'Inter-Community Flow\n{total_inter/1e6:.1f}M ({total_inter/(total_intra+total_inter)*100:.1f}%)'
        ]
        colors = ['#3498db', '#e74c3c']
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='',
            startangle=90,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )

        ax.set_title('Intra-Community vs Inter-Community Cash Flow',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        output_file = self.figures_dir / 'flow_comparison_pie.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存: {output_file}")

    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*80)
        print("社团资金流动分析摘要".center(80))
        print("="*80)

        # 总体统计
        total_intra = self.intra_community_flows['total_flow'].sum()
        total_inter = self.inter_community_flows['total_flow'].sum()
        total_flow = total_intra + total_inter

        print("\n【总体统计】")
        print(f"  总流动额: {total_flow:,.0f}")
        print(f"  社团内部流动: {total_intra:,.0f} ({total_intra/total_flow*100:.2f}%)")
        print(f"  社团间流动: {total_inter:,.0f} ({total_inter/total_flow*100:.2f}%)")

        # 社团内部流动Top 5
        print("\n【社团内部流动 - Top 5】")
        top_intra = self.intra_community_flows.nlargest(5, 'total_flow')
        for idx, row in top_intra.iterrows():
            print(f"  社团 {row['community']}: {row['total_flow']:,.0f} "
                  f"({row['flow_percentage']:.2f}%), "
                  f"{row['num_edges']} 条边, {row['transaction_count']} 笔交易")

        # 社团间流动Top 10
        print("\n【社团间流动 - Top 10】")
        top_inter = self.inter_community_flows.nlargest(10, 'total_flow')
        for idx, row in top_inter.iterrows():
            print(f"  社团 {row['source_community']} → 社团 {row['target_community']}: "
                  f"{row['total_flow']:,.0f}, "
                  f"{row['num_edges']} 条边, {row['transaction_count']} 笔交易")

        print("\n" + "="*80 + "\n")

    def run_analysis(self):
        """运行完整分析流程"""
        print("\n" + "="*80)
        print("开始社团资金流动分析".center(80))
        print("="*80 + "\n")

        # 1. 加载数据
        self.load_data()

        # 2. 社团内部流动分析
        self.analyze_intra_community_flows()

        # 3. 社团间流动分析
        self.analyze_inter_community_flows()

        # 4. 保存结果
        self.save_results()

        # 5. 生成可视化
        print("开始生成可视化图表...")
        self.plot_flow_matrix_heatmap()
        self.plot_intra_community_bars()
        self.plot_sankey_diagram()
        self.plot_flow_comparison_pie()
        print("所有可视化图表生成完成!\n")

        # 6. 打印摘要
        self.print_summary()

        print("分析完成! 所有结果已保存到:", self.output_dir)
        print("="*80 + "\n")


def main():
    """主函数"""
    # 设置文件路径
    base_dir = Path(__file__).parent
    membership_file = base_dir / 'output' / 'sbm_graphtool_membership.csv'
    cashflow_file = base_dir.parent / 'sbm' / 'sdg_cashflow_network.csv'
    output_dir = base_dir / 'output'

    # 创建分析器并运行
    analyzer = CommunityFlowAnalyzer(membership_file, cashflow_file, output_dir)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
