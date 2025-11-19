"""可视化所有与负面子目标相关的活动。

该脚本从 `output/sdg_cashflow_network.csv` 中提取所有带有 `_Neg`
标记的子目标（无论作为来源还是去向），统计其相关现金流总额与交易次数，
并通过 Plotly Sunburst 图展示"SDG 大目标 → 负面子目标"的层级结构。

同时计算并可视化每个SDG的净负面影响：
净影响 = 接收转化的负面影响 - (产生的负面副作用 + 负面循环流出)

运行方式：
    python negative_subtargets_sunburst.py

输出：
    - `negative_subtargets_sunburst.html` 负面子目标参与度交互式可视化
    - `sdg_net_impact_sunburst.html` SDG净负面影响交互式可视化
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.express as px

# 子目标描述映射来自分析脚本，若导入失败则退回空映射
try:
    from generate_negative_impact_analysis import SDG_SUBTARGET_DESC  # type: ignore
except ImportError:  # pragma: no cover - 兼容运行环境
    SDG_SUBTARGET_DESC = {}

# 路径设置
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "output" / "sdg_cashflow_network.csv"
OUTPUT_HTML = Path(__file__).resolve().with_suffix(".html")
OUTPUT_NET_IMPACT_HTML = Path(__file__).resolve().with_name("sdg_net_impact_sunburst.html")
OUTPUT_PURE_NET_HTML = Path(__file__).resolve().with_name("sdg_pure_net_impact_sunburst.html")

# SDG 大目标名称（使用报告中同样的命名）
SDG_NAMES: Dict[int, str] = {
    1: "SDG1 无贫穷",
    2: "SDG2 零饥饿",
    3: "SDG3 良好健康",
    4: "SDG4 优质教育",
    5: "SDG5 性别平等",
    6: "SDG6 清洁饮水与卫生",
    7: "SDG7 经济适用清洁能源",
    8: "SDG8 体面工作与经济增长",
    9: "SDG9 产业创新与基础设施",
    10: "SDG10 减少不平等",
    11: "SDG11 可持续城市与社区",
    12: "SDG12 负责任消费与生产",
    13: "SDG13 气候行动",
    14: "SDG14 水下生物",
    15: "SDG15 陆地生物",
    16: "SDG16 和平正义与强大机构",
    17: "SDG17 促进目标伙伴关系",
}


def split_sdg_code(code: str) -> Tuple[int, str]:
    """将 12.5_Neg 这样的编码拆分为 (12, 12.5_Neg)。"""
    major = code.split(".", 1)[0]
    return int(major), code


def load_negative_nodes() -> pd.DataFrame:
    """读取数据并抽取所有负面子目标的活动记录。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["cashflow"] = pd.to_numeric(df["cashflow"], errors="coerce").fillna(0.0)
    df["transaction_count"] = (
        pd.to_numeric(df["transaction_count"], errors="coerce").fillna(0.0)
    )

    records = []
    for role in ("source", "target"):
        col = f"{role}_sdg"
        mask = df[col].astype(str).str.endswith("_Neg", na=False)
        role_df = df.loc[mask, [col, "cashflow", "transaction_count"]].copy()
        role_df["node"] = role_df[col].astype(str)
        role_df["role"] = role
        records.append(role_df[["node", "cashflow", "transaction_count", "role"]])

    if not records:
        raise ValueError("数据集中未找到任何带 `_Neg` 的子目标。")

    stacked = pd.concat(records, ignore_index=True)
    stacked["sdg_id"] = stacked["node"].apply(lambda x: split_sdg_code(x)[0])
    stacked["sdg_name"] = stacked["sdg_id"].map(SDG_NAMES)

    agg = (
        stacked.groupby(["sdg_id", "sdg_name", "node"], as_index=False)
        .agg(
            total_cashflow=("cashflow", "sum"),
            total_transactions=("transaction_count", "sum"),
        )
        .sort_values(["sdg_id", "total_cashflow"], ascending=[True, False])
    )

    # 补充友好的标签与 hover 信息
    def format_node_label(node: str) -> str:
        subtarget_id = node.split("_", 1)[0]
        desc = SDG_SUBTARGET_DESC.get(subtarget_id, "").strip()
        return f"{node}（{desc}）" if desc else node

    agg["subtarget_label"] = agg["node"]
    agg["subtarget_display"] = agg["node"].apply(format_node_label)
    agg["hover_cashflow"] = agg["total_cashflow"].map(lambda v: f"¥{v:,.0f}")
    agg["hover_tx"] = agg["total_transactions"].map(lambda v: f"{v:,.0f} 次交易")

    return agg


def build_sunburst(data: pd.DataFrame):
    """生成 Sunburst 可视化。"""
    fig = px.sunburst(
        data,
        path=["sdg_name", "subtarget_display"],
        values="total_cashflow",
        color="total_cashflow",
        color_continuous_scale="Reds",
        title="负面子目标参与度 Sunburst（现金流规模）",
    )
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "%{label}",
                "现金流: ¥%{value:,.0f}",
                "交易数: %{customdata[0]}",
                "<extra></extra>"
            ]
        ),
        customdata=data[['hover_tx']].values,
        textinfo="label+percent entry",
        insidetextorientation="radial",
    )
    fig.update_layout(
        margin=dict(t=80, l=0, r=0, b=0),
        coloraxis_colorbar=dict(title="现金流规模"),
    )
    return fig


def calculate_net_impact() -> pd.DataFrame:
    """计算每个SDG的所有流动类型，细化到子目标层级。

    返回三层结构：SDG大目标 → 子目标 → 四种流动类型
    - Pos→Pos: 正面到正面
    - Neg→Pos: 负面到正面（转化负面影响）
    - Pos→Neg: 正面到负面（产生负面副作用）
    - Neg→Neg: 负面到负面（负面循环）
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["cashflow"] = pd.to_numeric(df["cashflow"], errors="coerce").fillna(0.0)

    # 提取SDG编号
    def extract_sdg(target_str):
        if pd.isna(target_str):
            return None
        return target_str.split('_')[0].split('.')[0]

    # 提取子目标完整编号（包含小数部分）
    def extract_subtarget(target_str):
        if pd.isna(target_str):
            return None
        return target_str.split('_')[0]  # 例如 "12.5_Neg" -> "12.5"

    df['source_sdg_num'] = df['source_sdg'].astype(str).apply(extract_sdg)
    df['target_sdg_num'] = df['target_sdg'].astype(str).apply(extract_sdg)
    df['source_subtarget'] = df['source_sdg'].astype(str).apply(extract_subtarget)
    df['target_subtarget'] = df['target_sdg'].astype(str).apply(extract_subtarget)

    # 判断正负面（基于原始的source_sdg和target_sdg列）
    df['source_type'] = df['source_sdg'].astype(str).apply(lambda x: 'Neg' if '_Neg' in x else 'Pos')
    df['target_type'] = df['target_sdg'].astype(str).apply(lambda x: 'Neg' if '_Neg' in x else 'Pos')

    results = []

    for sdg_num in range(1, 18):
        sdg_str = str(sdg_num)
        sdg_name = SDG_NAMES.get(sdg_num, f"SDG{sdg_num}")

        # 1. Pos → Pos (本SDG的Pos子目标 → 外部Pos)
        pos_to_pos_df = df[
            (df['source_sdg_num'] == sdg_str) &
            (df['source_type'] == 'Pos') &
            (df['target_type'] == 'Pos') &
            (df['target_sdg_num'] != sdg_str)
        ]

        for subtarget, group in pos_to_pos_df.groupby('source_subtarget'):
            if pd.notna(subtarget):
                cashflow = group['cashflow'].sum()
                subtarget_desc = SDG_SUBTARGET_DESC.get(subtarget, "").strip()
                subtarget_label = f"{subtarget}（{subtarget_desc}）" if subtarget_desc else subtarget

                results.append({
                    'sdg_id': sdg_num,
                    'sdg_name': sdg_name,
                    'subtarget': subtarget,
                    'subtarget_label': subtarget_label,
                    'flow_type': 'Pos→Pos',
                    'cashflow': cashflow
                })

        # 2. Neg → Pos (外部Neg → 本SDG的Pos子目标，转化负面影响)
        neg_to_pos_df = df[
            (df['target_sdg_num'] == sdg_str) &
            (df['source_type'] == 'Neg') &
            (df['target_type'] == 'Pos') &
            (df['source_sdg_num'] != sdg_str)
        ]

        for subtarget, group in neg_to_pos_df.groupby('target_subtarget'):
            if pd.notna(subtarget):
                cashflow = group['cashflow'].sum()
                subtarget_desc = SDG_SUBTARGET_DESC.get(subtarget, "").strip()
                subtarget_label = f"{subtarget}（{subtarget_desc}）" if subtarget_desc else subtarget

                results.append({
                    'sdg_id': sdg_num,
                    'sdg_name': sdg_name,
                    'subtarget': subtarget,
                    'subtarget_label': subtarget_label,
                    'flow_type': 'Neg→Pos',
                    'cashflow': cashflow
                })

        # 3. Pos → Neg (本SDG的Pos子目标 → 外部Neg，产生负面副作用)
        pos_to_neg_df = df[
            (df['source_sdg_num'] == sdg_str) &
            (df['source_type'] == 'Pos') &
            (df['target_type'] == 'Neg') &
            (df['target_sdg_num'] != sdg_str)
        ]

        for subtarget, group in pos_to_neg_df.groupby('source_subtarget'):
            if pd.notna(subtarget):
                cashflow = group['cashflow'].sum()
                subtarget_desc = SDG_SUBTARGET_DESC.get(subtarget, "").strip()
                subtarget_label = f"{subtarget}（{subtarget_desc}）" if subtarget_desc else subtarget

                results.append({
                    'sdg_id': sdg_num,
                    'sdg_name': sdg_name,
                    'subtarget': subtarget,
                    'subtarget_label': subtarget_label,
                    'flow_type': 'Pos→Neg',
                    'cashflow': cashflow
                })

        # 4. Neg → Neg (本SDG的Neg子目标 → 外部Neg，负面循环)
        neg_to_neg_df = df[
            (df['source_sdg_num'] == sdg_str) &
            (df['source_type'] == 'Neg') &
            (df['target_type'] == 'Neg') &
            (df['target_sdg_num'] != sdg_str)
        ]

        for subtarget, group in neg_to_neg_df.groupby('source_subtarget'):
            if pd.notna(subtarget):
                cashflow = group['cashflow'].sum()
                # 负面子目标带有 _Neg 后缀，在查找描述时需要去掉
                subtarget_base = subtarget.replace('_Neg', '')
                subtarget_desc = SDG_SUBTARGET_DESC.get(subtarget_base, "").strip()
                subtarget_label = f"{subtarget}（{subtarget_desc}）" if subtarget_desc else subtarget

                results.append({
                    'sdg_id': sdg_num,
                    'sdg_name': sdg_name,
                    'subtarget': subtarget,
                    'subtarget_label': subtarget_label,
                    'flow_type': 'Neg→Neg',
                    'cashflow': cashflow
                })

    result_df = pd.DataFrame(results)
    result_df['hover_cashflow'] = result_df['cashflow'].map(lambda v: f"¥{v:,.0f}")

    return result_df


def build_net_impact_sunburst(data: pd.DataFrame):
    """生成流动类型 Sunburst 可视化，支持三层结构：SDG → 子目标 → 流动类型。

    第三层使用固定颜色，第一、二层根据pos/neg流占比使用渐变色。
    """

    # 自定义蓝红渐变色阶
    custom_colorscale = [
        [0.0, "#982B2D"],   # 深红（高neg占比）
        [0.083, "#C84747"],
        [0.167, "#DE6A69"],
        [0.25, "#EE9D9F"],
        [0.333, "#FCCDC9"],
        [0.417, "#F1EEED"],  # 中性
        [0.5, "#E2F4FE"],
        [0.583, "#BBE6FA"],
        [0.667, "#89CAEA"],
        [0.75, "#4596CD"],
        [0.833, "#0B75B3"],
        [0.917, "#015696"],
        [1.0, "#012A61"]    # 深蓝（高pos占比）
    ]

    # 为四种流类型分配颜色值（映射到色阶上的精确位置）
    # 第三层显示指定的4个颜色，第一二层使用渐变
    flow_type_color_values = {
        'Pos→Pos': 0.833,    # 深蓝 #0B75B3
        'Neg→Pos': 0.667,    # 浅蓝 #89CAEA
        'Pos→Neg': 0.25,     # 浅红 #FCCDC9
        'Neg→Neg': 0.083     # 深红 #982B2D
    }

    # 为数据添加颜色值列
    data = data.copy()
    data['color_value'] = data['flow_type'].map(flow_type_color_values)

    fig = px.sunburst(
        data,
        path=["sdg_name", "subtarget_label", "flow_type"],
        values="cashflow",
        color="color_value",  # 使用颜色值，父节点会自动按cashflow加权平均
        color_continuous_scale=custom_colorscale,
        title="SDG 流动类型分析 Sunburst（三层结构）<br><sub>最内层：SDG大目标 | 第二层：子目标 | 第三层：流动类型 | 蓝色=高pos占比 | 红色=高neg占比</sub>",
    )
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "%{label}",
                "现金流: ¥%{value:,.0f}",
                "<extra></extra>"
            ]
        ),
        textinfo="label+percent parent",
        insidetextorientation="radial",
    )
    fig.update_layout(
        margin=dict(t=120, l=0, r=0, b=0),
        coloraxis_colorbar=dict(
            title="流动类型占比",
            ticktext=["高neg占比", "均衡", "高pos占比"],
            tickvals=[-1, 0, 1]
        ),
    )
    return fig


def calculate_pure_net_impact() -> pd.DataFrame:
    """计算每个SDG的纯净影响（只看净值）。

    纯净影响 = 接收转化负面影响 - (产生负面副作用 + 负面循环)

    返回每个SDG和子目标的净值
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["cashflow"] = pd.to_numeric(df["cashflow"], errors="coerce").fillna(0.0)

    # 提取SDG编号和子目标
    def extract_sdg(target_str):
        if pd.isna(target_str):
            return None
        return target_str.split('_')[0].split('.')[0]

    def extract_subtarget(target_str):
        if pd.isna(target_str):
            return None
        return target_str.split('_')[0]

    df['source_sdg_num'] = df['source_sdg'].astype(str).apply(extract_sdg)
    df['target_sdg_num'] = df['target_sdg'].astype(str).apply(extract_sdg)
    df['source_subtarget'] = df['source_sdg'].astype(str).apply(extract_subtarget)
    df['target_subtarget'] = df['target_sdg'].astype(str).apply(extract_subtarget)
    df['source_type'] = df['source_sdg'].astype(str).apply(lambda x: 'Neg' if '_Neg' in x else 'Pos')
    df['target_type'] = df['target_sdg'].astype(str).apply(lambda x: 'Neg' if '_Neg' in x else 'Pos')

    results = []

    for sdg_num in range(1, 18):
        sdg_str = str(sdg_num)
        sdg_name = SDG_NAMES.get(sdg_num, f"SDG{sdg_num}")

        # 计算每个子目标的正面（转化负面）和负面（产生负面+循环）贡献
        subtarget_net = {}

        # 1. 接收并转化负面影响 (外部Neg → 本SDG的Pos子目标) - 正面贡献
        neg_to_pos_df = df[
            (df['target_sdg_num'] == sdg_str) &
            (df['source_type'] == 'Neg') &
            (df['target_type'] == 'Pos') &
            (df['source_sdg_num'] != sdg_str)
        ]

        for subtarget, group in neg_to_pos_df.groupby('target_subtarget'):
            if pd.notna(subtarget):
                if subtarget not in subtarget_net:
                    subtarget_net[subtarget] = {'positive': 0, 'negative': 0}
                subtarget_net[subtarget]['positive'] += group['cashflow'].sum()

        # 2. 正面行动产生负面副作用 (本SDG的Pos子目标 → 外部Neg) - 负面贡献
        pos_to_neg_df = df[
            (df['source_sdg_num'] == sdg_str) &
            (df['source_type'] == 'Pos') &
            (df['target_type'] == 'Neg') &
            (df['target_sdg_num'] != sdg_str)
        ]

        for subtarget, group in pos_to_neg_df.groupby('source_subtarget'):
            if pd.notna(subtarget):
                if subtarget not in subtarget_net:
                    subtarget_net[subtarget] = {'positive': 0, 'negative': 0}
                subtarget_net[subtarget]['negative'] += group['cashflow'].sum()

        # 3. 负面循环流出 (本SDG的Neg子目标 → 外部Neg) - 负面贡献
        neg_to_neg_df = df[
            (df['source_sdg_num'] == sdg_str) &
            (df['source_type'] == 'Neg') &
            (df['target_type'] == 'Neg') &
            (df['target_sdg_num'] != sdg_str)
        ]

        for subtarget, group in neg_to_neg_df.groupby('source_subtarget'):
            if pd.notna(subtarget):
                # 负面子目标可能带_Neg后缀
                subtarget_clean = subtarget.replace('_Neg', '')
                if subtarget_clean not in subtarget_net:
                    subtarget_net[subtarget_clean] = {'positive': 0, 'negative': 0}
                subtarget_net[subtarget_clean]['negative'] += group['cashflow'].sum()

        # 计算净值
        for subtarget, flows in subtarget_net.items():
            net_value = flows['positive'] - flows['negative']

            # 只保留有净影响的子目标（绝对值 > 0）
            if abs(net_value) > 0:
                subtarget_desc = SDG_SUBTARGET_DESC.get(subtarget, "").strip()
                subtarget_label = f"{subtarget}（{subtarget_desc}）" if subtarget_desc else subtarget

                results.append({
                    'sdg_id': sdg_num,
                    'sdg_name': sdg_name,
                    'subtarget': subtarget,
                    'subtarget_label': subtarget_label,
                    'positive_flow': flows['positive'],
                    'negative_flow': flows['negative'],
                    'net_impact': net_value,
                    'abs_net_impact': abs(net_value),  # 用于确定扇区大小
                    'impact_type': '净正面贡献' if net_value > 0 else '净负面影响'
                })

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        result_df['hover_positive'] = result_df['positive_flow'].map(lambda v: f"¥{v:,.0f}")
        result_df['hover_negative'] = result_df['negative_flow'].map(lambda v: f"¥{v:,.0f}")
        result_df['hover_net'] = result_df['net_impact'].map(lambda v: f"¥{v:,.0f}")

    return result_df


def build_pure_net_impact_sunburst(data: pd.DataFrame):
    """生成纯净影响 Sunburst 可视化。"""
    if len(data) == 0:
        raise ValueError("没有净影响数据可供可视化")

    # 自定义蓝红色方案
    custom_colorscale = [
        [0.0, "#982B2D"],   # 深红（净负面）
        [0.083, "#C84747"],
        [0.167, "#DE6A69"],
        [0.25, "#EE9D9F"],
        [0.333, "#FCCDC9"],
        [0.417, "#F1EEED"],  # 中性
        [0.5, "#E2F4FE"],
        [0.583, "#BBE6FA"],
        [0.667, "#89CAEA"],
        [0.75, "#4596CD"],
        [0.833, "#0B75B3"],
        [0.917, "#015696"],
        [1.0, "#012A61"]    # 深蓝（净正面）
    ]

    fig = px.sunburst(
        data,
        path=["sdg_name", "subtarget_label"],
        values="abs_net_impact",  # 使用绝对值确定扇区大小
        color="net_impact",  # 使用净值确定颜色
        color_continuous_scale=custom_colorscale,
        title="SDG 纯净影响分析 Sunburst<br><sub>扇区大小=净影响绝对值 | 蓝色=净正面贡献 | 红色=净负面影响</sub>",
    )
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "%{label}",
                "净影响绝对值: ¥%{value:,.0f}",
                "<extra></extra>"
            ]
        ),
        textinfo="label+percent parent",
        insidetextorientation="radial",
    )
    fig.update_layout(
        margin=dict(t=100, l=0, r=0, b=0),
        coloraxis_colorbar=dict(
            title="净影响方向",
            ticktext=["净负面", "中性", "净正面"],
        ),
    )
    return fig


def main() -> None:
    # 生成负面子目标可视化
    data = load_negative_nodes()
    fig = build_sunburst(data)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"负面子目标可视化已保存至：{OUTPUT_HTML}")

    # 生成净影响可视化（三层结构）
    net_impact_data = calculate_net_impact()
    net_impact_fig = build_net_impact_sunburst(net_impact_data)
    net_impact_fig.write_html(OUTPUT_NET_IMPACT_HTML, include_plotlyjs="cdn")
    print(f"净影响可视化已保存至：{OUTPUT_NET_IMPACT_HTML}")

    # 生成纯净影响可视化（只看净值）
    pure_net_data = calculate_pure_net_impact()
    if len(pure_net_data) > 0:
        pure_net_fig = build_pure_net_impact_sunburst(pure_net_data)
        pure_net_fig.write_html(OUTPUT_PURE_NET_HTML, include_plotlyjs="cdn")
        print(f"纯净影响可视化已保存至：{OUTPUT_PURE_NET_HTML}")
    else:
        print("警告：没有净影响数据可供可视化")


if __name__ == "__main__":
    main()
