# -*- coding: utf-8 -*-
"""
SDG子目标级别现金流桑基图可视化 - Pyecharts版本
为每个SDG大目标（1-17）生成桑基图，展示子目标级别的现金流：
- 左列：外部流入源（其他SDG的子目标）
- 中列：目标SDG的所有子目标（如 1.1_Pos, 1.2_Neg等）
- 右列：外部流出目标（其他SDG的子目标）

使用固定阈值过滤边：低于阈值的边，其外部节点归并到Others
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


def extract_major_goal(sdg_target):
    """从SDG子目标提取大目标编号
    例如: '1.1_Pos' -> '1'
          '12.5_Neg' -> '12'
    """
    base = str(sdg_target).replace('_Pos', '').replace('_Neg', '')
    goal = base.split('.')[0]
    return goal


def filter_goal_flows(df, goal_num):
    """筛选与指定大目标相关的所有现金流（子目标级别）

    返回三类流：
    1. external_in: 外部流入该目标的流
    2. internal: 该目标内部子目标之间的流
    3. external_out: 该目标流出到外部的流
    """
    goal_str = str(goal_num)

    # 找出属于该大目标的所有子目标
    all_targets = set(df['source_sdg'].unique()) | set(df['target_sdg'].unique())
    goal_targets = {t for t in all_targets if extract_major_goal(t) == goal_str}

    if not goal_targets:
        return None, None, None, goal_targets

    # 分类流
    external_in = df[
        (df['target_sdg'].apply(lambda x: extract_major_goal(x) == goal_str)) &
        (df['source_sdg'].apply(lambda x: extract_major_goal(x) != goal_str))
    ].copy()

    internal = df[
        (df['source_sdg'].apply(lambda x: extract_major_goal(x) == goal_str)) &
        (df['target_sdg'].apply(lambda x: extract_major_goal(x) == goal_str))
    ].copy()

    external_out = df[
        (df['source_sdg'].apply(lambda x: extract_major_goal(x) == goal_str)) &
        (df['target_sdg'].apply(lambda x: extract_major_goal(x) != goal_str))
    ].copy()

    print(f"\nSDG {goal_num} 流统计:")
    print(f"  - 外部流入: {len(external_in)} 条, 总额: ¥{external_in['cashflow'].sum():,.0f}")
    print(f"  - 内部流动: {len(internal)} 条, 总额: ¥{internal['cashflow'].sum():,.0f}")
    print(f"  - 外部流出: {len(external_out)} 条, 总额: ¥{external_out['cashflow'].sum():,.0f}")
    print(f"  - 子目标数量: {len(goal_targets)}")

    return external_in, internal, external_out, goal_targets


def create_subtarget_sankey(goal_num, external_in, internal, external_out, goal_targets,
                           flow_threshold=100000):
    """为单个SDG大目标创建子目标级别的桑基图（使用Pyecharts）

    Args:
        goal_num: SDG大目标编号
        external_in: 外部流入数据
        internal: 内部流动数据
        external_out: 外部流出数据
        goal_targets: 目标子目标集合
        flow_threshold: 边流量阈值（固定金额，单位：元）
                       单条流量低于此阈值的边，其外部节点将被归并为Others
    """
    print(f"\n正在创建SDG目标 {goal_num} 的子目标桑基图...")

    # 合并所有流数据
    all_flows_list = []
    if external_in is not None and len(external_in) > 0:
        all_flows_list.append(external_in)
    if internal is not None and len(internal) > 0:
        all_flows_list.append(internal)
    if external_out is not None and len(external_out) > 0:
        all_flows_list.append(external_out)

    if not all_flows_list:
        print(f"  - 警告: 没有数据")
        return None

    # 步骤1: 基于边的流量进行过滤归并
    print(f"  - 流量阈值: ¥{flow_threshold:,.0f}")

    filtered_count = {'source': 0, 'goal': 0, 'dest': 0}

    if flow_threshold > 0:
        # 1.1 外部流入：小流量的source归并到Others_Source
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

        # 1.2 内部流动：小流量的两端归并到Others_Goal
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

        # 1.3 外部流出：小流量的target归并到Others_Dest
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

        # 1.4 对中列节点的target和source也应用归并（处理external_in的target和external_out的source）
        if external_in is not None and len(external_in) > 0:
            # 注意：这里不再基于流量过滤，因为已经在上面处理过了
            # 只是确保中列节点的命名一致性（如果之前有归并到Others_Goal的）
            pass

        if external_out is not None and len(external_out) > 0:
            # 同上
            pass

    print(f"    归并小流量边: 左列 {filtered_count['source']} 条, 中列 {filtered_count['goal']} 条, 右列 {filtered_count['dest']} 条")

    # 步骤2: 定义自然排序键函数
    def natural_sort_key(node):
        """SDG节点自然排序键（处理1.1, 1.2, ..., 1.10的顺序）"""
        if node.startswith('Others_'):
            return (999, 0, '')  # Others放最后

        # 提取基础部分和后缀
        base = node.replace('_Pos', '').replace('_Neg', '')
        suffix = '_Pos' if '_Pos' in node else ('_Neg' if '_Neg' in node else '')

        # 解析SDG编号
        parts = base.split('.')
        major = int(parts[0]) if parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

        # 排序：先按主目标，再按子目标，最后Pos在前Neg在后
        return (major, minor, suffix)

    # 步骤3: 收集所有节点并分类
    # 左列：外部源
    external_sources = sorted(set(external_in['source_sdg']) if len(external_in) > 0 else set(),
                             key=natural_sort_key)

    # 中列：目标子目标
    target_subtargets = sorted(
        set(internal['source_sdg']) | set(internal['target_sdg']) |
        set(external_in['target_sdg']) | set(external_out['source_sdg'])
        if (len(internal) > 0 or len(external_in) > 0 or len(external_out) > 0) else set(),
        key=natural_sort_key
    )

    # 右列：外部目标
    external_destinations = sorted(set(external_out['target_sdg']) if len(external_out) > 0 else set(),
                                   key=natural_sort_key)

    print(f"  - 左列: {len(external_sources)} 个节点")
    print(f"  - 中列: {len(target_subtargets)} 个节点")
    print(f"  - 右列: {len(external_destinations)} 个节点")

    # 步骤4: 创建节点标签和颜色函数
    def create_node_label(node, suffix=''):
        """创建节点标签"""
        # 处理Others节点
        if node == 'Others_Source':
            return f'Others {suffix}' if suffix else 'Others'
        elif node == 'Others_Goal':
            return f'Others {suffix}' if suffix else 'Others'
        elif node == 'Others_Dest':
            return f'Others {suffix}' if suffix else 'Others'

        # 处理普通节点
        label = node.replace('_Pos', ' (+)').replace('_Neg', ' (-)')
        if suffix:
            return f'{label} {suffix}'
        return label

    def get_node_color(node):
        """获取节点颜色"""
        if node.startswith('Others_'):
            return '#C8C8C8'  # 灰色
        elif node.endswith('_Pos'):
            return '#6a88c2'  # 蓝色（最深蓝色）
        elif node.endswith('_Neg'):
            return '#eb6468'  # 红色（最深红色）
        else:
            return '#C8C8C8'

    # 步骤5: 按顺序创建节点（左→中→右）
    nodes = []
    source_to_label = {}
    target_in_to_label = {}
    target_out_to_label = {}
    dest_to_label = {}

    # 5.1 左列：外部源节点（depth=0）
    for node in external_sources:
        label = create_node_label(node, '[Source]')
        color = get_node_color(node)
        nodes.append({
            "name": label,
            "depth": 0,
            "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 2}
        })
        source_to_label[node] = label

    # 5.2 中列：目标子目标的In和Out节点（depth=1和2）
    for node in target_subtargets:
        # In节点（depth=1）
        label_in = create_node_label(node, 'In')
        color = get_node_color(node)
        nodes.append({
            "name": label_in,
            "depth": 1,
            "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 2}
        })
        target_in_to_label[node] = label_in

        # Out节点（depth=2）
        label_out = create_node_label(node, 'Out')
        nodes.append({
            "name": label_out,
            "depth": 2,
            "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 2}
        })
        target_out_to_label[node] = label_out

    # 5.3 右列：外部目标节点（depth=3）
    for node in external_destinations:
        label = create_node_label(node, '[Dest]')
        color = get_node_color(node)
        nodes.append({
            "name": label,
            "depth": 3,
            "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 2}
        })
        dest_to_label[node] = label

    print(f"  - 创建了 {len(nodes)} 个节点")

    # 步骤6: 创建链接数据
    links = []
    flow_type_colors = {
        'Pos_to_Pos': '#c3e4f5',    # 淡蓝色
        'Neg_to_Neg': '#fdb1b3',    # 淡红色
        'Pos_to_Neg': '#F4CFD6',    # 粉色
        'Neg_to_Pos': '#FCE694'     # 黄色
    }

    total_flow = 0

    # 6.1 外部流入
    if external_in is not None and len(external_in) > 0:
        for _, row in external_in.iterrows():
            src_label = source_to_label.get(row['source_sdg'])
            tgt_label = target_in_to_label.get(row['target_sdg'])

            if src_label and tgt_label:
                links.append({
                    "source": src_label,
                    "target": tgt_label,
                    "value": float(row['cashflow']),
                    "lineStyle": {
                        "color": flow_type_colors.get(row['flow_type'], '#999999'),
                        "opacity": 1,
                        "curveness": 0.5
                    }
                })
                total_flow += row['cashflow']

    # 6.2 内部流动：In → Out
    if internal is not None and len(internal) > 0:
        for _, row in internal.iterrows():
            src_label = target_in_to_label.get(row['source_sdg'])
            tgt_label = target_out_to_label.get(row['target_sdg'])

            if src_label and tgt_label:
                links.append({
                    "source": src_label,
                    "target": tgt_label,
                    "value": float(row['cashflow']),
                    "lineStyle": {
                        "color": flow_type_colors.get(row['flow_type'], '#999999'),
                        "opacity": 1,
                        "curveness": 0.5
                    }
                })
                total_flow += row['cashflow']

    # 6.3 外部流出
    if external_out is not None and len(external_out) > 0:
        for _, row in external_out.iterrows():
            src_label = target_out_to_label.get(row['source_sdg'])
            tgt_label = dest_to_label.get(row['target_sdg'])

            if src_label and tgt_label:
                links.append({
                    "source": src_label,
                    "target": tgt_label,
                    "value": float(row['cashflow']),
                    "lineStyle": {
                        "color": flow_type_colors.get(row['flow_type'], '#999999'),
                        "opacity": 1,
                        "curveness": 0.5
                    }
                })
                total_flow += row['cashflow']

    print(f"  - 创建了 {len(links)} 条链接，总流量: ¥{total_flow:,.0f}")

    # 步骤7: 创建Pyecharts桑基图
    sankey = (
        Sankey(init_opts=opts.InitOpts(
            width="2200px",
            height="1600px",
            bg_color="white"
        ))
        .add(
            series_name="SDG Subtarget Cashflow",
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
                #          f"Target Subtargets: {len(target_subtargets)} | Destinations: {len(external_destinations)} | "
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
    print("SDG子目标级别现金流桑基图生成（三列布局 - Pyecharts版）")
    print("=" * 80)

    # 文件路径
    network_csv = '../../output/sdg_cashflow_network.csv'
    output_dir = './sankey/subtargets'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 流量阈值（固定金额，单位：元）
    flow_threshold = 1000000

    print(f"\n配置:")
    print(f"  - 输入文件: {network_csv}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 流量阈值: ¥{flow_threshold:,.0f}（按列独立应用）")
    print(f"  - 可视化工具: Pyecharts")
    print(f"  - 数据级别: 子目标级别（如 1.1_Pos, 1.2_Neg）")

    # 加载数据
    df = load_network_data(network_csv)

    # 为每个SDG目标（1-17）创建桑基图
    successful_count = 0
    failed_goals = []

    for goal_num in range(1, 18):  # SDG目标1-17
        print(f"\n{'='*60}")
        print(f"正在处理 SDG {goal_num}...")

        # 筛选该目标的流
        external_in, internal, external_out, goal_targets = filter_goal_flows(df, goal_num)

        # 检查是否有数据
        if not goal_targets:
            print(f"  ✗ SDG {goal_num} 没有找到相关数据，跳过")
            failed_goals.append(goal_num)
            continue

        total_flows = sum([
            len(external_in) if external_in is not None else 0,
            len(internal) if internal is not None else 0,
            len(external_out) if external_out is not None else 0
        ])

        if total_flows == 0:
            print(f"  ✗ SDG {goal_num} 没有现金流数据，跳过")
            failed_goals.append(goal_num)
            continue

        # 创建桑基图
        sankey_chart = create_subtarget_sankey(
            goal_num, external_in, internal, external_out, goal_targets,
            flow_threshold=flow_threshold
        )

        if sankey_chart is None:
            print(f"  ✗ SDG {goal_num} 无法生成桑基图")
            failed_goals.append(goal_num)
        else:
            # 保存为HTML
            output_html = os.path.join(output_dir, f'sdg_{goal_num}_subtarget_sankey.html')
            sankey_chart.render(output_html)
            print(f"  ✓ SDG {goal_num}: HTML已保存至 {output_html}")

            # 保存为PNG
            # output_png = os.path.join(output_dir, f'sdg_{goal_num}_subtarget_sankey.png')
            # make_snapshot(snapshot, output_html, output_png)
            # print(f"  ✓ SDG {goal_num}: PNG已保存至 {output_png}")
            # successful_count += 1

            # 删除可能生成的render.html文件
            if os.path.exists('render.html'):
                os.remove('render.html')
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("完成！")
    print(f"  - 成功生成: {successful_count} 个桑基图")
    if failed_goals:
        print(f"  - 失败目标: {', '.join(map(str, failed_goals))}")
    print(f"  - 所有图表已保存到: {output_dir}")
    print("=" * 80)

    # 生成图例PNG
    legend_path = os.path.join(output_dir, 'sankey_legend.png')
    generate_legend_png(legend_path, dpi=300)


if __name__ == '__main__':
    main()
