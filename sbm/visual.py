import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

try:
    from brokenaxes import brokenaxes
except ImportError:
    brokenaxes = None

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


# ==========================================
# 1. 数据加载与预处理
# ==========================================
df_mem = pd.read_csv("output/sbm_graphtool_membership.csv")
df_comm = pd.read_csv("output/sbm_graphtool_communities.csv")

df_mem["total_degree"] = (
    df_mem["pos_pos_degree"]
    + df_mem["neg_neg_degree"]
    + df_mem["pos_neg_degree"]
    + df_mem["neg_pos_degree"]
)

# 社区按总流量降序
comm_flow = df_mem.groupby("community")["total_degree"].sum().sort_values(ascending=False)
comm_rename_map = {old_id: new_id for new_id, old_id in enumerate(comm_flow.index, start=1)}
df_mem["community"] = df_mem["community"].map(comm_rename_map)
df_comm["community"] = df_comm["community"].map(comm_rename_map)

comm_flow.index = comm_flow.index.map(comm_rename_map)
comm_rank_map = {c: i for i, c in enumerate(comm_flow.index)}
df_mem["comm_rank"] = df_mem["community"].map(comm_rank_map)
df_mem["comm_label"] = df_mem["community"].astype(str)

# ==========================================
# 连接类型分析
# ==========================================
conn_analysis = df_comm.copy()
conn_analysis["pos_pos_ratio"] = (
    conn_analysis["pos_pos_degree"] / conn_analysis["total_degree"] * 100
).round(2)
conn_analysis["neg_neg_ratio"] = (
    conn_analysis["neg_neg_degree"] / conn_analysis["total_degree"] * 100
).round(2)
conn_analysis["pos_neg_ratio"] = (
    conn_analysis["pos_neg_degree"] / conn_analysis["total_degree"] * 100
).round(2)
conn_analysis["neg_pos_ratio"] = (
    conn_analysis["neg_pos_degree"] / conn_analysis["total_degree"] * 100
).round(2)

conn_analysis["dominant_type"] = (
    conn_analysis[
        ["pos_pos_ratio", "neg_neg_ratio", "pos_neg_ratio", "neg_pos_ratio"]
    ].idxmax(axis=1)
).str.replace("_ratio", "")

conn_analysis_sorted = conn_analysis.sort_values("community")
conn_analysis_sorted[
    [
        "community",
        "num_nodes",
        "num_positive",
        "num_negative",
        "pos_pos_degree",
        "neg_neg_degree",
        "pos_neg_degree",
        "neg_pos_degree",
        "pos_pos_ratio",
        "neg_neg_ratio",
        "pos_neg_ratio",
        "neg_pos_ratio",
        "dominant_type",
    ]
].to_csv("output/community_connection_types.csv", index=False)

pie_data = {}
for _, row in conn_analysis.iterrows():
    pie_data[row["community"]] = {
        "Pos→Pos": row["pos_pos_ratio"],
        "Neg→Neg": row["neg_neg_ratio"], 
        "Pos→Neg": row["pos_neg_ratio"],
        "Neg→Pos": row["neg_pos_ratio"],
    }

# ==========================================
# 2. 绘图设置
# ==========================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 1.0
sns.set_context("paper", font_scale=1.4)

colors = {"Positive": "#4575b4", "Negative": "#d73027"}
markers = {"Positive": "o", "Negative": "^"}
pie_colors = {
    "Pos→Pos": "#66C2A5",
    "Neg→Neg": "#FC8D62",
    "Pos→Neg": "#8DA0CB",
    "Neg→Pos": "#E78AC3",
}

# ==========================================
# 3. 数据预处理
# ==========================================
num_communities = len(pie_data)
rng = np.random.default_rng(42)
df_mem["x_pos"] = df_mem["comm_rank"].astype(float) + rng.normal(0, 0.12, len(df_mem))

# 使用线性y值
df_mem["y_value"] = df_mem["total_degree"]

# 点的大小统一
df_mem["size"] = 80

# ==========================================
# 4. 创建布局：四个子图 + 下方饼图 + 右侧图例
# ==========================================
# 子图1: 社区1-2, 子图2: 社区3-6, 子图3: 社区7-8, 子图4: 社区9-10
communities_1 = [1, 2]
communities_2 = [3, 4, 5, 6]
communities_3 = [7, 8]
communities_4 = [9, 10]

# 分离数据
df_1 = df_mem[df_mem["community"].isin(communities_1)].copy()
df_2 = df_mem[df_mem["community"].isin(communities_2)].copy()
df_3 = df_mem[df_mem["community"].isin(communities_3)].copy()
df_4 = df_mem[df_mem["community"].isin(communities_4)].copy()

# 为每个子图分别计算x_pos（重新映射到从0开始）
comm_rank_1 = {c: i for i, c in enumerate(communities_1)}
comm_rank_2 = {c: i for i, c in enumerate(communities_2)}
comm_rank_3 = {c: i for i, c in enumerate(communities_3)}
comm_rank_4 = {c: i for i, c in enumerate(communities_4)}

df_1["x_pos_local"] = df_1["community"].map(comm_rank_1).astype(float) + rng.normal(0, 0.12, len(df_1))
df_2["x_pos_local"] = df_2["community"].map(comm_rank_2).astype(float) + rng.normal(0, 0.12, len(df_2))
df_3["x_pos_local"] = df_3["community"].map(comm_rank_3).astype(float) + rng.normal(0, 0.12, len(df_3))
df_4["x_pos_local"] = df_4["community"].map(comm_rank_4).astype(float) + rng.normal(0, 0.12, len(df_4))

# 创建布局（3行：散点图、饼图、图例）
fig = plt.figure(figsize=(20, 14), dpi=300)
gs = GridSpec(3, num_communities, figure=fig,
              height_ratios=[4, 2, 0.8],
              hspace=0.12, wspace=0.08)

# 子图1：社区1-2（占据2列）
ax_1 = fig.add_subplot(gs[0, 0:2])

# 子图2：社区3-6（占据4列）
ax_2 = fig.add_subplot(gs[0, 2:6])

# 子图3：社区7-8（占据2列）- 使用brokenaxes省略7M-10M区间
y_max_3 = df_3["y_value"].max() * 1.1 if len(df_3) > 0 else 15e6
if brokenaxes is not None:
    ax_3 = brokenaxes(ylims=((0, 7e6), (10e6, y_max_3)),
                      subplot_spec=gs[0, 6:8],
                      height_ratios=(1, 4), d=0.003, tilt=45, hspace=0.05)
else:
    ax_3 = fig.add_subplot(gs[0, 6:8])

# 子图4：社区9-10（占据2列）- 使用brokenaxes省略2M-3M区间
y_max_4 = df_4["y_value"].max() * 1.1 if len(df_4) > 0 else 5e6
if brokenaxes is not None:
    ax_4 = brokenaxes(ylims=((0, 2e6), (3e6, y_max_4)),
                      subplot_spec=gs[0, 8:num_communities],
                      height_ratios=(1, 4), d=0.003, tilt=45, hspace=0.05)
else:
    ax_4 = fig.add_subplot(gs[0, 8:num_communities])

# ==========================================
# 5. 绘制散点图（四个子图）
# ==========================================

# 加载象限数据并关联到节点
df_quadrants = pd.read_csv("output/quadrant_labels.csv")
df_mem["sdg_target"] = df_mem["node"].str.extract(r'^([^_]+)')[0]
df_mem = df_mem.merge(df_quadrants, left_on="sdg_target", right_on="sdg_target", how="left")

# 更新分离的数据（合并后需要重新分离）
df_1 = df_mem[df_mem["community"].isin(communities_1)].copy()
df_2 = df_mem[df_mem["community"].isin(communities_2)].copy()
df_3 = df_mem[df_mem["community"].isin(communities_3)].copy()
df_4 = df_mem[df_mem["community"].isin(communities_4)].copy()

# 重新计算x_pos_local
df_1["x_pos_local"] = df_1["community"].map(comm_rank_1).astype(float) + rng.normal(0, 0.12, len(df_1))
df_2["x_pos_local"] = df_2["community"].map(comm_rank_2).astype(float) + rng.normal(0, 0.12, len(df_2))
df_3["x_pos_local"] = df_3["community"].map(comm_rank_3).astype(float) + rng.normal(0, 0.12, len(df_3))
df_4["x_pos_local"] = df_4["community"].map(comm_rank_4).astype(float) + rng.normal(0, 0.12, len(df_4))

# 定义象限颜色（区别于饼图的流动类型颜色）
quadrant_colors = {
    "Q1": "#3182bd",  # 深蓝
    "Q2": "#e6550d",  # 深橙
    "Q3": "#31a354",  # 深绿
    "Q4": "#756bb1"   # 深紫
}

# 定义 y 轴标签格式化器（科学计数法，只显示数字）
def make_value_formatter(scale_exp):
    """创建基于指定指数的格式化器"""
    def formatter(y, _pos):
        scaled = y / (10 ** scale_exp)
        if scaled == int(scaled):
            return f"{int(scaled)}"
        return f"{scaled:.1f}"
    return formatter

# 绘图辅助函数
def plot_subplot(ax, df, communities, title, is_brokenaxes=False, show_ylabel=False, scale_exp=6):
    """绘制单个子图

    Args:
        scale_exp: y轴的科学计数法指数，默认6表示×10⁶
    """
    for node_type in ["Positive", "Negative"]:
        for quad in ["Q1", "Q2", "Q3", "Q4"]:
            mask = (df["node_type"] == node_type) & (df["quadrant"] == quad)
            subset = df[mask]
            if not subset.empty:
                ax.scatter(
                    subset["x_pos_local"],
                    subset["y_value"],
                    s=subset["size"],
                    color=quadrant_colors[quad],
                    marker=markers[node_type],
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=0.6,
                )

    # 设置X轴
    ax.set_xlim(-0.5, len(communities) - 0.5)

    # 科学计数法指数标注
    exp_label = f"×10$^{scale_exp}$"

    if is_brokenaxes:
        # brokenaxes需要对每个子轴单独设置
        for idx, sub_ax in enumerate(ax.axs):
            sub_ax.set_xticks(range(len(communities)))
            sub_ax.set_xticklabels(communities)
            sub_ax.yaxis.set_major_formatter(FuncFormatter(make_value_formatter(scale_exp)))
            sub_ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            # 设置边框：内部连接处不需要边框
            sub_ax.spines['left'].set_visible(True)
            sub_ax.spines['right'].set_visible(True)
            sub_ax.spines['left'].set_linewidth(1.0)
            sub_ax.spines['right'].set_linewidth(1.0)
            if idx == 0:  # 上部子轴：隐藏底部边框
                sub_ax.spines['top'].set_visible(True)
                sub_ax.spines['top'].set_linewidth(1.0)
                sub_ax.spines['bottom'].set_visible(False)
            else:  # 下部子轴：隐藏顶部边框
                sub_ax.spines['top'].set_visible(False)
                sub_ax.spines['bottom'].set_visible(True)
                sub_ax.spines['bottom'].set_linewidth(1.0)
        # 在顶部子轴的y轴顶端添加指数标注
        top_ax = ax.axs[0]
        top_ax.text(0, 1.02, exp_label, transform=top_ax.transAxes,
                    fontsize=10, ha='left', va='bottom')
        # brokenaxes的标签设置
        ax.set_xlabel("Community ID", fontweight="bold", fontsize=11, labelpad=20)
        if show_ylabel:
            ax.set_ylabel("Total Cashflow Volume", fontweight="bold", fontsize=11, labelpad=40)
        # 标题通过第一个子轴设置
        ax.axs[0].set_title(title, fontweight="bold", fontsize=12, pad=10)
    else:
        ax.set_xticks(range(len(communities)))
        ax.set_xticklabels(communities)
        # Y轴根据数据自动设置范围
        y_min = 0
        y_max = df["y_value"].max() * 1.1
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(FuncFormatter(make_value_formatter(scale_exp)))
        # 在y轴顶端添加指数标注
        ax.text(0, 1.02, exp_label, transform=ax.transAxes,
                fontsize=10, ha='left', va='bottom')
        # 设置四条边框可见
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        # 标签
        ax.set_xlabel("Community ID", fontweight="bold", fontsize=11)
        if show_ylabel:
            ax.set_ylabel("Total Cashflow Volume", fontweight="bold", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # 添加文本标签（brokenaxes不支持adjust_text）
    texts = []
    for _, row in df.iterrows():
        label = row["node"].split("_")[0]
        text_obj = ax.text(
            row["x_pos_local"],
            row["y_value"],
            label,
            fontsize=6,
            color="black",
            ha="center",
            va="bottom",
            alpha=0.9,
        )
        texts.append(text_obj)

    if not is_brokenaxes and adjust_text is not None and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.5, alpha=0.7),
        )

# 绘制四个子图（只有第一个显示y轴标题）
# scale_exp调整为使y轴标签在0-10范围内
plot_subplot(ax_1, df_1, communities_1, "Communities 1-2", show_ylabel=True, scale_exp=8)
plot_subplot(ax_2, df_2, communities_2, "Communities 3-6", scale_exp=7)
plot_subplot(ax_3, df_3, communities_3, "Communities 7-8", is_brokenaxes=(brokenaxes is not None), scale_exp=6)
plot_subplot(ax_4, df_4, communities_4, "Communities 9-10", is_brokenaxes=(brokenaxes is not None), scale_exp=6)

# ==========================================
# 5. 在图片下方添加饼图（作为一个整体子图）
# ==========================================

# 创建一个横跨所有散点图宽度的子图来放置饼图
ax_pies = fig.add_subplot(gs[1, 0:num_communities])
ax_pies.set_xlim(0, num_communities)
ax_pies.set_ylim(0, 1)
ax_pies.axis("off")

# 在这个子图中绘制每个社区的饼图
for i, comm_id in enumerate(sorted(pie_data.keys())[:num_communities]):
    data = pie_data[comm_id]
    sizes = list(data.values())
    pie_slice_colors = [pie_colors[label] for label in data.keys()]

    def autopct_format(pct: float) -> str:
        return f"{pct:.0f}%" if pct > 8 else ""

    # 计算饼图中心位置
    center_x = (i + 0.5) / num_communities
    center_y = 0.5

    # 使用inset_axes在指定位置创建饼图
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    pie_size_inch = 1  # 直接指定饼图大小（英寸）
    ax_pie = inset_axes(ax_pies, width=pie_size_inch, height=pie_size_inch,
                        loc='center', bbox_to_anchor=(center_x, center_y),
                        bbox_transform=ax_pies.transAxes, borderpad=0)

    wedges, texts, autotexts = ax_pie.pie(
        sizes,
        labels=None,
        colors=pie_slice_colors,
        autopct=autopct_format,
        startangle=90,
        textprops={"fontsize": 8, "weight": "bold"},
    )
    ax_pie.set_aspect("equal")
    # 在饼图下方添加社区编号
    ax_pie.text(0, -1.3, f"C{comm_id}", fontsize=10, fontweight="bold",
                ha='center', va='top')

# ==========================================
# 6. 图例放在下方
# ==========================================
# 定义两行元素
row1 = [
    Line2D([0], [0], marker=markers["Positive"], color="w", markerfacecolor="gray",
           label="Positive SDG", markersize=12, markeredgecolor="black", markeredgewidth=0.5),
    Line2D([0], [0], marker=markers["Negative"], color="w", markerfacecolor="gray",
           label="Negative SDG", markersize=12, markeredgecolor="black", markeredgewidth=0.5),
    Patch(facecolor=quadrant_colors["Q1"], label="Quadrant 1"),
    Patch(facecolor=quadrant_colors["Q2"], label="Quadrant 2"),
    Patch(facecolor=quadrant_colors["Q3"], label="Quadrant 3"),
    Patch(facecolor=quadrant_colors["Q4"], label="Quadrant 4"),
]
row2 = [
    Patch(facecolor=pie_colors["Pos→Pos"], label="Pos→Pos"),
    Patch(facecolor=pie_colors["Neg→Neg"], label="Neg→Neg"),
    Patch(facecolor=pie_colors["Pos→Neg"], label="Pos→Neg"),
    Patch(facecolor=pie_colors["Neg→Pos"], label="Neg→Pos"),
]

# matplotlib按列填充，需要交错排列实现按行显示
# ncol=6, 10元素: 列0-3各2行，列4-5各1行
# 交错顺序: (0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4),(0,5)
legend_elements = [
    row1[0], row2[0],  # 列0
    row1[1], row2[1],  # 列1
    row1[2], row2[2],  # 列2
    row1[3], row2[3],  # 列3
    row1[4],           # 列4
    row1[5],           # 列5
]

# 图例放在底部
ax_legend = fig.add_subplot(gs[2, :])
ax_legend.axis("off")
ax_legend.legend(
    handles=legend_elements,
    loc="upper center",
    title="Node Type, Quadrant & Links",
    frameon=True,
    fancybox=False,
    edgecolor="black",
    fontsize=12,
    title_fontsize=13,
    ncol=6,
    columnspacing=1.5,
)

plt.savefig("visualization/sdg_complex_network_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
