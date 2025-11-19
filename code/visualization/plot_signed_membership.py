"""Helper to visualize the signed membership summary exported by `sdg_signed_membership.csv`.

The chart highlights each subtarget's positive and negative degrees (converted to a diverging
horizontal bar chart so negative degrees extend to the left). This mirrors other Plotly-based
visualizations that live in the same folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "output" / "sdg_signed_membership.csv"
DEFAULT_HTML = Path(__file__).resolve().with_name("sdg_signed_membership_chart.html")

EXPECTED_COLUMNS = {"node", "community", "node_type", "positive_degree", "negative_degree"}
DEGREE_LABELS = {"positive_degree": "Positive Degree", "negative_degree": "Negative Degree"}
DEGREE_COLORS = {"Positive Degree": "#2E86AB", "Negative Degree": "#C44536"}


def plot_signed_membership(
    csv_path: Path | str = DEFAULT_DATA,
    output_html: Optional[Path | str] = None,
    community: Optional[int] = None,
    top_n: Optional[int] = 40,
) -> Path:
    """Build an interactive diverging bar chart from `sdg_signed_membership.csv`.

    Args:
        csv_path: Source CSV path. Defaults to the repo's `output/sdg_signed_membership.csv`.
        output_html: Optional output HTML file path. Defaults to `<this_file>_chart.html`.
        community: If provided, filter rows to a single community id.
        top_n: Limit results to the first N rows after sorting by total degree.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"输入文件缺少必要列：{missing_cols}")

    df["positive_degree"] = pd.to_numeric(df["positive_degree"], errors="coerce").fillna(0.0)
    df["negative_degree"] = pd.to_numeric(df["negative_degree"], errors="coerce").fillna(0.0)
    df["community"] = pd.to_numeric(df["community"], errors="coerce").astype("Int64")

    if community is not None:
        df = df[df["community"] == community]
        if df.empty:
            raise ValueError(f"社区 {community} 没有任何节点记录，无法可视化。")

    df["total_degree"] = df[["positive_degree", "negative_degree"]].max(axis=1)
    df = df.sort_values("total_degree", ascending=False).copy()

    if top_n is not None:
        df = df.head(top_n)

    if df.empty:
        raise ValueError("筛选条件导致数据为空，无法生成可视化。")

    df["node_label"] = df["node"].astype(str)
    long_df = df.melt(
        id_vars=["node_label", "community"],
        value_vars=["positive_degree", "negative_degree"],
        var_name="degree_type",
        value_name="degree_value",
    )

    long_df["degree_label"] = long_df["degree_type"].map(DEGREE_LABELS)
    long_df["signed_value"] = long_df.apply(
        lambda row: row["degree_value"] if row["degree_type"] == "positive_degree" else -row["degree_value"],
        axis=1,
    )

    category_order = {"node_label": df["node_label"].tolist()[::-1]}
    title = "SDG Signed Membership（按子目标展示正负度）"
    if community is not None:
        title += f" - Community {community}"

    fig = px.bar(
        long_df,
        x="signed_value",
        y="node_label",
        color="degree_label",
        orientation="h",
        category_orders=category_order,
        color_discrete_map=DEGREE_COLORS,
        title=title,
    )
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "%{y}",
                "%{customdata[0]}：¥%{customdata[1]:,.0f}",
                "<extra></extra>",
            ]
        ),
        customdata=long_df[["degree_label", "degree_value"]],
    )
    fig.update_layout(
        xaxis_title="Signed Degree（右侧为正度，左侧为负度）",
        yaxis_title="SDG 子目标",
        barmode="relative",
        bargap=0.2,
        legend_title="度类型",
        template="simple_white",
        margin=dict(t=80, l=10, r=10, b=10),
    )

    output_html = Path(output_html) if output_html else DEFAULT_HTML
    fig.write_html(output_html, include_plotlyjs="cdn")
    return output_html


if __name__ == "__main__":
    OUTPUT = plot_signed_membership()
    print(f"已生成可视化文件：{OUTPUT}")
