# -*- coding: utf-8 -*-
"""
SDG大目标（1-17）现金流桑基图可视化 - Pyecharts版本
为每个SDG大目标（1-17）单独生成桑基图，展示该目标与其他目标之间的现金流关系
使用 Pyecharts 提供更美观的可视化效果
"""

import pandas as pd
import os
from pyecharts import options as opts
from pyecharts.charts import Sankey
from pyecharts.globals import CurrentConfig, OnlineHostType
from pyecharts.commons.utils import JsCode
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot


def load_network_data(csv_file='../output/sdg_cashflow_network.csv'):
    """加载SDG现金流网络数据"""
    print(f"正在加载网络数据: {csv_file}")
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"  - 加载了 {len(df)} 条现金流记录")
    return df


def aggregate_to_major_goals(df):
    """将子目标级别的流聚合到大目标级别（保留Pos/Neg区分）"""
    print("\n正在聚合到SDG大目标...")

    df_copy = df.copy()

    # 提取SDG大目标编号（例如从"1.1_Pos"提取"1_Pos"）
    def extract_goal_with_suffix(sdg_target):
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
    df_copy['source_goal'] = df_copy['source_sdg'].apply(extract_goal_with_suffix)
    df_copy['target_goal'] = df_copy['target_sdg'].apply(extract_goal_with_suffix)

    # 按大目标聚合，保留flow_type
    goal_flows = df_copy.groupby(['source_goal', 'target_goal', 'flow_type']).agg({
        'cashflow': 'sum',
        'transaction_count': 'sum'
    }).reset_index()

    # 重命名列以保持一致性
    goal_flows = goal_flows.rename(columns={
        'source_goal': 'source_sdg',
        'target_goal': 'target_sdg'
    })

    print(f"  - 聚合后SDG大目标流: {len(goal_flows):,} 条")
    print(f"  - Pos→Pos流: {len(goal_flows[goal_flows['flow_type']=='Pos_to_Pos']):,} 条")
    print(f"  - Neg→Neg流: {len(goal_flows[goal_flows['flow_type']=='Neg_to_Neg']):,} 条")
    print(f"  - Pos→Neg流: {len(goal_flows[goal_flows['flow_type']=='Pos_to_Neg']):,} 条")
    print(f"  - Neg→Pos流: {len(goal_flows[goal_flows['flow_type']=='Neg_to_Pos']):,} 条")
    print(f"  - 涉及 {goal_flows['source_sdg'].nunique()} 个SDG大目标节点")

    return goal_flows


def create_single_goal_sankey(goal_flows_df, target_goal, flow_threshold=100000):
    """为单个SDG大目标创建桑基图（使用Pyecharts，三列布局）

    Args:
        goal_flows_df: 大目标级别的现金流数据
        target_goal: 目标SDG大目标编号（1-17）
        flow_threshold: 边流量阈值（固定金额，单位：元）
                       单条流量低于此阈值的边，其外部节点将被归并为Others

    布局说明：
        - 左列：其他SDG [Source] - 流向目标SDG
        - 中列：目标SDG In/Out - 目标SDG本身
        - 右列：其他SDG [Dest] - 从目标SDG流出
    """
    print(f"\n正在创建SDG目标 {target_goal} 的桑基图...")

    # 步骤1: 分类流数据
    # external_in: 其他SDG → 目标SDG
    external_in = goal_flows_df[
        (goal_flows_df['source_sdg'].str.replace('_Pos', '').str.replace('_Neg', '') != str(target_goal)) &
        (goal_flows_df['target_sdg'].str.replace('_Pos', '').str.replace('_Neg', '') == str(target_goal))
    ].copy()

    # internal: 目标SDG → 目标SDG（内部流动）
    internal = goal_flows_df[
        (goal_flows_df['source_sdg'].str.replace('_Pos', '').str.replace('_Neg', '') == str(target_goal)) &
        (goal_flows_df['target_sdg'].str.replace('_Pos', '').str.replace('_Neg', '') == str(target_goal))
    ].copy()

    # external_out: 目标SDG → 其他SDG
    external_out = goal_flows_df[
        (goal_flows_df['source_sdg'].str.replace('_Pos', '').str.replace('_Neg', '') == str(target_goal)) &
        (goal_flows_df['target_sdg'].str.replace('_Pos', '').str.replace('_Neg', '') != str(target_goal))
    ].copy()

    # 检查是否有数据
    if len(external_in) == 0 and len(internal) == 0 and len(external_out) == 0:
        print(f"  - 警告: SDG目标 {target_goal} 没有相关的现金流数据")
        return None

    print(f"  - 外部流入: {len(external_in)} 条, 总额: ¥{external_in['cashflow'].sum():,.0f}")
    print(f"  - 内部流动: {len(internal)} 条, 总额: ¥{internal['cashflow'].sum():,.0f}")
    print(f"  - 外部流出: {len(external_out)} 条, 总额: ¥{external_out['cashflow'].sum():,.0f}")

    # 步骤2: 基于边的流量进行过滤归并
    print(f"  - 流量阈值: ¥{flow_threshold:,.0f}")

    filtered_count = {'source': 0, 'goal': 0, 'dest': 0}

    if flow_threshold > 0:
        # 2.1 外部流入：小流量的source归并到Others_Source
        if external_in is not None and len(external_in) > 0:
            mask = external_in['cashflow'] < flow_threshold
            filtered_count['source'] = mask.sum()
            external_in['source_sdg'] = external_in.apply(
                lambda row: 'Others_Source' if row['cashflow'] < flow_threshold else row['source_sdg'],
                axis=1
            )
            # 聚合归并后的流
            external_in = external_in.groupby(['source_sdg', 'target_sdg', 'flow_type']).agg({
                'cashflow': 'sum',
                'transaction_count': 'sum'
            }).reset_index()

        # 2.2 内部流动：小流量的两端归并到Others_Goal
        if internal is not None and len(internal) > 0:
            mask = internal['cashflow'] < flow_threshold
            filtered_count['goal'] = mask.sum()
            internal['source_sdg'] = internal.apply(
                lambda row: 'Others_Goal' if row['cashflow'] < flow_threshold else row['source_sdg'],
                axis=1
            )
            internal['target_sdg'] = internal.apply(
                lambda row: 'Others_Goal' if row['cashflow'] < flow_threshold else row['target_sdg'],
                axis=1
            )
            # 聚合归并后的流
            internal = internal.groupby(['source_sdg', 'target_sdg', 'flow_type']).agg({
                'cashflow': 'sum',
                'transaction_count': 'sum'
            }).reset_index()

        # 2.3 外部流出：小流量的target归并到Others_Dest
        if external_out is not None and len(external_out) > 0:
            mask = external_out['cashflow'] < flow_threshold
            filtered_count['dest'] = mask.sum()
            external_out['target_sdg'] = external_out.apply(
                lambda row: 'Others_Dest' if row['cashflow'] < flow_threshold else row['target_sdg'],
                axis=1
            )
            # 聚合归并后的流
            external_out = external_out.groupby(['source_sdg', 'target_sdg', 'flow_type']).agg({
                'cashflow': 'sum',
                'transaction_count': 'sum'
            }).reset_index()

    print(f"    归并小流量边: 左列 {filtered_count['source']} 条, 中列 {filtered_count['goal']} 条, 右列 {filtered_count['dest']} 条")

    # 步骤3: 定义自然排序键函数
    def natural_sort_key(node):
        """SDG节点自然排序键（处理1, 2, ..., 10的顺序）"""
        if node.startswith('Others_'):
            return (999, '')  # Others放最后

        # 提取基础部分和后缀
        base = node.replace('_Pos', '').replace('_Neg', '')
        suffix = '_Pos' if '_Pos' in node else ('_Neg' if '_Neg' in node else '')

        # 解析SDG编号（大目标只有主编号，无子编号）
        goal_num = int(base) if base.isdigit() else 0

        # 排序：先按主目标，然后Pos在前Neg在后
        return (goal_num, suffix)

    # 步骤4: 收集所有涉及的SDG节点
    # 左列：外部源（流向目标SDG的其他SDG）
    external_sources = sorted(set(external_in['source_sdg']) if len(external_in) > 0 else set(),
                             key=natural_sort_key)

    # 中列：目标SDG的所有子目标
    target_goals = sorted(
        set(internal['source_sdg']) | set(internal['target_sdg']) |
        set(external_in['target_sdg']) | set(external_out['source_sdg'])
        if (len(internal) > 0 or len(external_in) > 0 or len(external_out) > 0) else set(),
        key=natural_sort_key
    )

    # 右列：外部目标（从目标SDG流出到的其他SDG）
    external_destinations = sorted(set(external_out['target_sdg']) if len(external_out) > 0 else set(),
                                   key=natural_sort_key)

    print(f"  - 左列（外部源）: {len(external_sources)} 个节点")
    print(f"  - 中列（目标SDG）: {len(target_goals)} 个节点")
    print(f"  - 右列（外部目标）: {len(external_destinations)} 个节点")

    # 步骤5: 创建节点映射字典
    def create_node_label(goal, suffix=''):
        """创建节点标签"""
        # 处理Others节点
        if goal == 'Others_Source':
            return f'Others {suffix}' if suffix else 'Others'
        elif goal == 'Others_Goal':
            return f'Others {suffix}' if suffix else 'Others'
        elif goal == 'Others_Dest':
            return f'Others {suffix}' if suffix else 'Others'

        # 处理普通节点
        base_goal = goal.replace('_Pos', '').replace('_Neg', '')

        if goal.endswith('_Pos'):
            label_base = f'SDG {base_goal} (+)'
        elif goal.endswith('_Neg'):
            label_base = f'SDG {base_goal} (-)'
        else:
            label_base = f'SDG {base_goal}'

        if suffix:
            return f'{label_base} {suffix}'
        return label_base

    def get_node_color(goal):
        """获取节点颜色"""
        if goal.startswith('Others_'):
            return '#C8C8C8'  # 灰色
        elif goal.endswith('_Pos'):
            return '#6a88c2'  # 蓝色（最深蓝色）
        elif goal.endswith('_Neg'):
            return '#eb6468'  # 红色（最深红色）
        else:
            return '#C8C8C8'  # 灰色

    # 步骤6: 按顺序创建节点（左→中→右）
    nodes = []

    # 映射字典：原始节点 -> (标签, 位置)
    source_to_label = {}  # 左列映射
    target_in_to_label = {}  # 中列In节点映射
    target_out_to_label = {}  # 中列Out节点映射
    dest_to_label = {}  # 右列映射

    # 6.1 左列：外部源节点（depth=0）
    for goal in external_sources:
        label = create_node_label(goal, '[Source]')
        color = get_node_color(goal)
        nodes.append({
            "name": label,
            "depth": 0,
            "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 2}
        })
        source_to_label[goal] = label

    # 6.2 中列：目标SDG的In和Out节点（depth=1和2）
    for goal in target_goals:
        # In节点（depth=1）
        label_in = create_node_label(goal, 'In')
        color = get_node_color(goal)
        nodes.append({
            "name": label_in,
            "depth": 1,
            "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 2}
        })
        target_in_to_label[goal] = label_in

        # Out节点（depth=2）
        label_out = create_node_label(goal, 'Out')
        nodes.append({
            "name": label_out,
            "depth": 2,
            "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 2}
        })
        target_out_to_label[goal] = label_out

    # 6.3 右列：外部目标节点（depth=3）
    for goal in external_destinations:
        label = create_node_label(goal, '[Dest]')
        color = get_node_color(goal)
        nodes.append({
            "name": label,
            "depth": 3,
            "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 2}
        })
        dest_to_label[goal] = label

    print(f"  - 创建了 {len(nodes)} 个节点")

    # 步骤7: 创建链接数据
    links = []

    # 定义颜色 - 根据流的类型使用不同颜色
    flow_type_colors = {
        'Pos_to_Pos': '#c3e4f5',    # 淡蓝色
        'Neg_to_Neg': '#fdb1b3',    # 淡红色
        'Pos_to_Neg': '#F4CFD6',    # 粉色
        'Neg_to_Pos': '#FCE694'     # 黄色
    }

    total_flow = 0

    # 7.1 外部流入：其他SDG [Source] → 目标SDG In
    if len(external_in) > 0:
        for _, row in external_in.iterrows():
            src = row['source_sdg']
            tgt = row['target_sdg']
            value = float(row['cashflow'])

            src_label = source_to_label.get(src)
            tgt_label = target_in_to_label.get(tgt)

            if src_label and tgt_label:
                links.append({
                    "source": src_label,
                    "target": tgt_label,
                    "value": value,
                    "lineStyle": {
                        "color": flow_type_colors.get(row['flow_type'], '#999999'),
                        "opacity": 1,
                        "curveness": 0.5
                    }
                })
                total_flow += value

    # 7.2 内部流动：目标SDG In → 目标SDG Out
    if len(internal) > 0:
        for _, row in internal.iterrows():
            src = row['source_sdg']
            tgt = row['target_sdg']
            value = float(row['cashflow'])

            src_label = target_in_to_label.get(src)
            tgt_label = target_out_to_label.get(tgt)

            if src_label and tgt_label:
                links.append({
                    "source": src_label,
                    "target": tgt_label,
                    "value": value,
                    "lineStyle": {
                        "color": flow_type_colors.get(row['flow_type'], '#999999'),
                        "opacity": 1,
                        "curveness": 0.5
                    }
                })
                total_flow += value

    # 7.3 外部流出：目标SDG Out → 其他SDG [Dest]
    if len(external_out) > 0:
        for _, row in external_out.iterrows():
            src = row['source_sdg']
            tgt = row['target_sdg']
            value = float(row['cashflow'])

            src_label = target_out_to_label.get(src)
            tgt_label = dest_to_label.get(tgt)

            if src_label and tgt_label:
                links.append({
                    "source": src_label,
                    "target": tgt_label,
                    "value": value,
                    "lineStyle": {
                        "color": flow_type_colors.get(row['flow_type'], '#999999'),
                        "opacity": 1,
                        "curveness": 0.5
                    }
                })
                total_flow += value

    print(f"  - 创建了 {len(links)} 条链接，总流量: ¥{total_flow:,.0f}")

    # 步骤8: 创建Pyecharts桑基图
    sankey = (
        Sankey(init_opts=opts.InitOpts(
            width="2200px",
            height="1600px",
            bg_color="white"
        ))
        .add(
            series_name="SDG Cashflow",
            nodes=nodes,
            links=links,
            pos_left="20%",
            pos_right="20%",
            pos_top="10%",
            pos_bottom="10%",
            node_gap=10,
            node_width=40,
            is_draggable=True,
            label_opts=opts.LabelOpts(
                position="right",
                font_size=16,
                font_family="Arial",
                font_weight="bold",
                formatter=JsCode("""
                    function(params) {
                        return params.name
                            .replace(' [Source]', '')
                            .replace(' [Dest]', '');
                    }
                """)
            ),
            linestyle_opt=opts.LineStyleOpts(
                opacity=0.35,
                curve=0.5,
                color="source"
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                # title=f"SDG Goal {goal_num} Cashflow Network (Subtarget Level)",
                title='',
                # subtitle=f"Total: ¥{total_flow:,.0f} | Sources: {len(external_sources)} | "
                #          f"Target Nodes: {len(target_goals)} | Destinations: {len(external_destinations)} | "
                #          f"Links: {len(links)}",
                title_textstyle_opts=opts.TextStyleOpts(
                    font_size=24,
                    font_family="Arial",
                    font_weight="bold"
                ),
                subtitle_textstyle_opts=opts.TextStyleOpts(
                    font_size=14,
                    font_family="Arial"
                ),
                pos_left="center",
                pos_top="15px"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                trigger_on="mousemove",
                formatter=JsCode(r"""
                    function(params) {
                        if (params.dataType === 'edge') {
                            var value = params.value || params.data.value;
                            var formattedValue = '¥' + value.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
                            return params.data.source + ' → ' + params.data.target + '<br/>Cashflow: ' + formattedValue;
                        } else {
                            return params.name;
                        }
                    }
                """),
                textstyle_opts=opts.TextStyleOpts(font_size=13)
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    return sankey


def generate_legend_png(output_path='./sankey_legend.png', dpi=300):
    """生成桑基图图例PNG文件

    Args:
        output_path: 输出PNG文件路径
        dpi: 图像分辨率（默认300 DPI）
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    print(f"\n正在生成图例PNG: {output_path} (DPI={dpi})")

    # 创建图形
    _, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    ax.axis('off')

    # 定义颜色
    flow_type_colors = [
        ('Pos → Pos', '#c3e4f5'),
        ('Neg → Neg', '#fdb1b3'),
        ('Pos → Neg', '#F4CFD6'),
        ('Neg → Pos', '#FCE694')
    ]

    node_colors = [
        ('Positive Nodes', '#6a88c2'),
        ('Negative Nodes', '#eb6468'),
        ('Others', '#C8C8C8')
    ]

    # 流类型图例（上半部分）
    ax.text(0.5, 0.95, 'Flow Types (Edge Colors)',
            ha='center', va='top', fontsize=14, 
            transform=ax.transAxes)

    # 绘制流类型图例（2x2布局）
    y_start = 0.80
    x_positions = [0.22, 0.52]  # 左右两列的x位置（居中）

    for idx, (label, color) in enumerate(flow_type_colors):
        row = idx // 2
        col = idx % 2
        x_pos = x_positions[col]
        y_pos = y_start - row * 0.12

        # 绘制彩色矩形（矩形高度0.05，矩形中心与文字对齐）
        rect_height = 0.05
        rect = FancyBboxPatch((x_pos, y_pos - rect_height/2), 0.12, rect_height,
                              boxstyle="square,pad=0",
                              edgecolor='none', facecolor=color,
                              linewidth=0, transform=ax.transAxes)
        ax.add_patch(rect)

        # 添加文本标签
        ax.text(x_pos + 0.14, y_pos, label, va='center', fontsize=12,
                transform=ax.transAxes)

    # 节点类型图例（下半部分）
    y_middle = 0.50
    ax.text(0.5, y_middle, 'Node Types (Node Colors)',
            ha='center', va='top', fontsize=14,
            transform=ax.transAxes)

    # 绘制节点类型图例（第一行两个，第二行Others与第一列对齐）
    y_start = 0.35

    for idx, (label, color) in enumerate(node_colors):
        if idx < 2:  # 第一行两个
            row = 0
            col = idx
            x_pos = x_positions[col]
        else:  # 第二行Others与第一列对齐
            row = 1
            x_pos = x_positions[0]  # 与第一列对齐

        y_pos = y_start - row * 0.12

        # 绘制彩色矩形（矩形高度0.05，矩形中心与文字对齐）
        rect_height = 0.05
        rect = FancyBboxPatch((x_pos, y_pos - rect_height/2), 0.12, rect_height,
                              boxstyle="square,pad=0",
                              edgecolor='white', facecolor=color,
                              linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)

        # 添加文本标签
        ax.text(x_pos + 0.14, y_pos, label, va='center', fontsize=12,
                transform=ax.transAxes)

    # 添加说明
    ax.text(0.5, 0.02, 'Note: Pos = Positive Impact | Neg = Negative Impact',
            ha='center', va='bottom', fontsize=10, style='italic',
            transform=ax.transAxes)

    # 保存为PNG
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  ✓ 图例已保存至: {output_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("SDG大目标现金流桑基图生成（三列布局 - Pyecharts版）")
    print("=" * 80)

    # 配置 Pyecharts 使用在线 CDN（备选方案）
    # 如果网络不好，可以改为 LOCAL 模式（需要先下载 echarts.min.js）
    # CurrentConfig.ONLINE_HOST = OnlineHostType.JSDELIVR_CDN

    # 文件路径
    network_csv = '../../output/sdg_cashflow_network.csv'
    output_dir = './sankey/17goals'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 流量阈值（固定金额，单位：元）
    # 推荐值：100000（10万），每列独立应用，过滤小节点到Others
    flow_threshold = 100000

    print(f"\n配置:")
    print(f"  - 输入文件: {network_csv}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 流量阈值: ¥{flow_threshold:,.0f}（按列独立应用）")
    print(f"  - 可视化工具: Pyecharts")
    print(f"  - 布局模式: 三列布局（左：外部源 | 中：目标SDG | 右：外部目标）")
    print(f"  - JS 资源: 在线 CDN (默认)")

    # 步骤1: 加载数据
    df = load_network_data(network_csv)

    # 步骤2: 聚合到大目标级别
    goal_flows = aggregate_to_major_goals(df)

    # 步骤3: 为每个SDG目标（1-17）创建单独的桑基图
    successful_count = 0
    failed_goals = []

    for goal_num in range(1, 18):  # SDG目标1-17
        output_file = os.path.join(output_dir, f'sdg_goal_{goal_num}_sankey.html')

        sankey_chart = create_single_goal_sankey(goal_flows, goal_num, flow_threshold)

        if sankey_chart is None:
            print(f"  ✗ SDG目标 {goal_num}: 无法生成桑基图")
            failed_goals.append(goal_num)
        else:
            # 保存为HTML
            output_html = os.path.join(output_dir, f'sdg_goal_{goal_num}_sankey.html')
            sankey_chart.render(output_html)
            print(f"  ✓ SDG目标 {goal_num}: HTML已保存至 {output_html}")

            # 保存为PNG
            output_png = os.path.join(output_dir, f'sdg_goal_{goal_num}_sankey.png')
            make_snapshot(snapshot, output_html, output_png)
            print(f"  ✓ SDG目标 {goal_num}: PNG已保存至 {output_png}")

            # 删除可能生成的render.html文件
            if os.path.exists('render.html'):
                os.remove('render.html')

            successful_count += 1

    # 步骤4: 输出统计信息
    print("\n" + "=" * 80)
    print("完成！")
    print(f"  - 成功生成: {successful_count} 个桑基图")
    if failed_goals:
        print(f"  - 失败目标: {', '.join(map(str, failed_goals))}")
    print(f"  - 所有图表已保存到: {output_dir}")
    print("=" * 80)

    # 生成图例PNG
    # legend_path = os.path.join(output_dir, 'sankey_legend.png')
    # generate_legend_png(legend_path, dpi=300)


if __name__ == '__main__':
    main()
