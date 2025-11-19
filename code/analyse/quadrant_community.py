"""
象限图 - 按社区着色版本
与quadrant.py相同的样式，但点的颜色按社区划分
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Try to import adjustText for better label positioning
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Warning: adjustText not installed. Labels may overlap.")
    print("Install with: pip install adjustText")


def plot_quadrant_by_community(quadrant_csv: str, membership_csv: str, output_png: str):
    """
    绘制象限图，按社区着色

    Args:
        quadrant_csv: 象限数据文件路径
        membership_csv: 社区成员数据文件路径
        output_png: 输出PNG文件路径
    """
    print(f"\n加载数据...")

    # 加载象限数据
    quadrant_df = pd.read_csv(quadrant_csv, encoding='utf-8')
    print(f"  象限数据: {len(quadrant_df)} 行")

    # 加载社区数据
    membership_df = pd.read_csv(membership_csv, encoding='utf-8')
    print(f"  社区数据: {len(membership_df)} 行")

    # 提取sdg_target（去掉_Pos/_Neg后缀）
    membership_df['sdg_target'] = membership_df['node'].str.replace('_Pos', '').str.replace('_Neg', '')

    # 合并数据
    df = quadrant_df.merge(
        membership_df[['sdg_target', 'node', 'community', 'node_type']],
        on='sdg_target',
        how='left'
    )

    print(f"  合并后数据: {len(df)} 行")
    print(f"  社区数: {df['community'].nunique()}")

    # 设置绘图样式
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # Nature journal style settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Microsoft YaHei']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.unicode_minus'] = False

    SCATTER_LABEL_SIZE = 6
    SPINE_WIDTH = 1.5

    # 创建图形
    print(f"\n绘制象限图（按社区着色）...")

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(6, 6, figure=fig, hspace=0.05, wspace=0.05)

    # 主散点图
    ax = fig.add_subplot(gs[1:, :-1])

    # 边缘密度图
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax)

    # 计算点的大小
    min_size = 50
    max_size = 500
    total_activity = df['positive_total_flow'] + df['negative_total_flow']

    if total_activity.max() > total_activity.min():
        normalized_activity = (total_activity - total_activity.min()) / (total_activity.max() - total_activity.min())
        sizes = min_size + normalized_activity * (max_size - min_size)
    else:
        sizes = np.full(len(df), min_size)

    sizes = np.where(total_activity == 0, min_size, sizes)

    # 社区颜色映射 - 为所有社区分配不同颜色
    communities = sorted(df['community'].unique())
    n_communities = len(communities)

    # 使用tab20调色板（20种颜色）以确保所有社区都有不同颜色
    if n_communities <= 10:
        # 使用tab10调色板
        cmap = plt.cm.tab10
    else:
        # 使用tab20调色板
        cmap = plt.cm.tab20

    community_colors_map = {}
    for i, comm in enumerate(communities):
        # 获取颜色（转换为十六进制）
        rgba = cmap(i / max(n_communities - 1, 1))
        # 转换为十六进制
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255)
        )
        community_colors_map[comm] = hex_color

    # 为每个点分配颜色
    colors = [community_colors_map[comm] for comm in df['community']]

    # 准备绘图数据
    x_col = 'positive_log_standardized'
    y_col = 'negative_log_standardized'

    # 添加jitter以避免重叠
    np.random.seed(42)  # 固定随机种子以保证可重复性
    x_jitter = np.random.normal(0, 0.02, len(df))  # X轴小幅抖动
    y_jitter = np.random.normal(0, 0.02, len(df))  # Y轴小幅抖动

    # 对y<-0.8的点增加更大的Y轴抖动
    y_jitter_enhanced = y_jitter.copy()
    mask_dense = df[y_col] < -0.8
    y_jitter_enhanced[mask_dense] = np.random.normal(0, 0.05, mask_dense.sum())

    plot_df = pd.DataFrame({
        'x': df[x_col] + x_jitter,
        'y': df[y_col] + y_jitter_enhanced,
        'x_original': df[x_col],
        'y_original': df[y_col],
        'size': sizes,
        'color': colors,
        'community': df['community'],
        'sdg_target': df['sdg_target'].values,
        'node': df['node'].values
    })

    # Y轴反转
    ax.invert_yaxis()

    # 添加坐标轴（z=0线）
    ax.axvline(0, color='black', linestyle='-', alpha=0.9, linewidth=SPINE_WIDTH, zorder=1)
    ax.axhline(0, color='black', linestyle='-', alpha=0.9, linewidth=SPINE_WIDTH, zorder=1)

    # 按社区分组绘制散点（反向排序，让大社区最后画在上层）
    for comm in sorted(communities, reverse=True):
        comm_data = plot_df[plot_df['community'] == comm]

        ax.scatter(
            comm_data['x'],
            comm_data['y'],
            s=comm_data['size'],
            c=[community_colors_map[comm]],
            alpha=1.0,  # 完全不透明
            edgecolor='white',
            linewidth=1,
            zorder=3,
            label=f'Community {int(comm)}'
        )

    # 添加标签
    if HAS_ADJUSTTEXT:
        texts = []
        texts_dense = []
        for _, row in plot_df.iterrows():
            txt = ax.text(
                row['x'],
                row['y'],
                row['sdg_target'],
                fontsize=SCATTER_LABEL_SIZE,
                alpha=0.9,
                ha='center',
                va='center'
            )
            if row['y'] < -0.8:
                texts_dense.append(txt)
            else:
                texts.append(txt)

        # 调整密集区域
        if texts_dense:
            adjust_text(
                texts_dense,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                expand_points=(3.5, 3.5),
                expand_text=(3.0, 3.0),
                force_points=1.2,
                force_text=1.8,
                lim=2000,
                ax=ax
            )

        # 调整稀疏区域
        if texts:
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                expand_points=(2.5, 2.5),
                expand_text=(2.0, 2.0),
                force_points=0.8,
                force_text=1.2,
                lim=1500,
                ax=ax
            )
    else:
        # 简单标注
        for _, row in plot_df.iterrows():
            ax.annotate(
                row['sdg_target'],
                (row['x'], row['y']),
                fontsize=SCATTER_LABEL_SIZE,
                alpha=0.7,
                ha='center',
                va='center'
            )

    # 边缘直方图
    ax_top.hist(df[x_col], bins=30, alpha=0.5, color='#555555', edgecolor='black', linewidth=0.5)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.axis('off')

    ax_right.hist(df[y_col], bins=30, orientation='horizontal', alpha=0.5, color='#555555', edgecolor='black', linewidth=0.5)
    ax_right.set_ylim(ax.get_ylim())
    ax_right.axis('off')

    # 坐标轴标签
    ax.set_xlabel('Positive Flow P (log-standardized, z-score)')
    ax.set_ylabel('Negative Flow N (log-standardized, z-score, ↓ increasing)')

    # 图框样式
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_edgecolor('black')

    ax.tick_params(axis='both', which='major', width=0.8, length=4)
    ax.tick_params(axis='both', which='minor', width=0.5, length=2)

    # 网格
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.2, color='gray', zorder=0)

    # 图例（显示所有社区，放在图外右侧）
    legend = ax.legend(
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),  # 放在图的右侧外面
        ncol=1,  # 1列布局
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        fancybox=False,
        fontsize=9
    )
    legend.get_frame().set_linewidth(0.5)

    # 标题
    ax.set_title('SDG Quadrant Analysis (Colored by Community)',
                 fontsize=14, fontweight='bold', pad=20)

    # 保存
    plt.tight_layout()
    plt.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  已保存: {output_png}")
    plt.close()

    # 打印统计信息
    print(f"\n社区分布统计:")
    for comm in sorted(communities):
        count = len(df[df['community'] == comm])
        pct = count / len(df) * 100
        color = community_colors_map[comm]
        print(f"  社区 {comm}: {count} 个节点 ({pct:.1f}%) - 颜色: {color}")


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    quadrant_csv = os.path.join(base_dir, 'output', 'sdg_quadrant_data.csv')
    membership_csv = os.path.join(base_dir, 'output', 'sdg_signed_membership.csv')
    output_png = os.path.join(base_dir, 'output', 'sdg_quadrant_by_community.png')

    print("=" * 80)
    print("SDG象限图 - 按社区着色")
    print("=" * 80)

    plot_quadrant_by_community(quadrant_csv, membership_csv, output_png)

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()