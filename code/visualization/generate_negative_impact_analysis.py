import pandas as pd

# SDG 目标名称映射
SDG_NAMES = {
    1: "No Poverty (无贫穷)",
    2: "Zero Hunger (零饥饿)",
    3: "Good Health and Well-being (良好健康与福祉)",
    4: "Quality Education (优质教育)",
    5: "Gender Equality (性别平等)",
    6: "Clean Water and Sanitation (清洁饮水和卫生设施)",
    7: "Affordable and Clean Energy (经济适用的清洁能源)",
    8: "Decent Work and Economic Growth (体面工作和经济增长)",
    9: "Industry, Innovation and Infrastructure (产业、创新和基础设施)",
    10: "Reduced Inequalities (减少不平等)",
    11: "Sustainable Cities and Communities (可持续城市和社区)",
    12: "Responsible Consumption and Production (负责任消费和生产)",
    13: "Climate Action (气候行动)",
    14: "Life Below Water (水下生物)",
    15: "Life on Land (陆地生物)",
    16: "Peace, Justice and Strong Institutions (和平、正义与强大机构)",
    17: "Partnerships for the Goals (促进目标实现的伙伴关系)"
}


def extract_goal_number(sdg_str):
    """从 SDG 字符串中提取目标编号"""
    return int(sdg_str.split('.')[0])


def generate_sdg_flow_summary():
    """生成每个SDG大目标的主导流摘要CSV"""

    # 路径配置
    data_path = r'D:\1-PKU\PKU\1 Master\Master 1\Papers\Cashchain\output\sdg_cashflow_network.csv'
    csv_output = r'D:\1-PKU\PKU\1 Master\Master 1\Papers\Cashchain\code\visualization\sdg_top_flow_summary.csv'

    # 读取数据
    print("正在读取数据...")
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    if df.empty:
        print("⚠️ 输入数据为空，未生成输出。")
        return

    # 处理缺失值
    if 'flow_type' not in df.columns:
        df['flow_type'] = 'Unknown'
    else:
        df['flow_type'] = df['flow_type'].fillna('Unknown')

    # 提取目标编号
    df['source_goal'] = df['source_sdg'].apply(extract_goal_number)
    df['target_goal'] = df['target_sdg'].apply(extract_goal_number)

    # 定义三种流的类别
    flow_categories = [
        {
            "key": "incoming",
            "label": "从外部流入",
            "description": "来自其他SDG大目标的流入",
            "filter": lambda data, goal: (data['target_goal'] == goal) & (data['source_goal'] != goal)
        },
        {
            "key": "internal",
            "label": "内部流动",
            "description": "同一SDG大目标内部子目标之间的流动（含自环）",
            "filter": lambda data, goal: (data['source_goal'] == goal) & (data['target_goal'] == goal)
        },
        {
            "key": "outgoing",
            "label": "流出到外部",
            "description": "从本SDG大目标流向其他SDG大目标",
            "filter": lambda data, goal: (data['source_goal'] == goal) & (data['target_goal'] != goal)
        }
    ]

    # CSV列定义
    columns = [
        "sdg_goal",              # SDG大目标编号
        "sdg_name",              # SDG大目标名称
        "flow_category_key",     # 流类别键
        "flow_category",         # 流类别名称
        "flow_category_desc",    # 流类别描述
        "rank",                  # 排名（1-5）
        "source_goal",           # 来源大目标编号
        "source_goal_name",      # 来源大目标名称
        "source_sdg",            # 来源子目标
        "target_goal",           # 目标大目标编号
        "target_goal_name",      # 目标大目标名称
        "target_sdg",            # 目标子目标
        "flow_type",             # 流类型（Pos_to_Pos等）
        "cashflow"               # 现金流金额
    ]

    # 收集所有记录
    summary_records = []

    print("正在分析各SDG大目标的流动...")
    for goal_num in range(1, 18):
        goal_name = SDG_NAMES.get(goal_num, f"SDG {goal_num}")

        # 对每种流类别
        for category in flow_categories:
            # 筛选符合条件的流
            mask = category["filter"](df, goal_num)

            if not mask.any():
                continue

            # 获取Top 5
            top_flows = df.loc[mask].nlargest(5, 'cashflow')

            # 记录每一条
            for rank, (_, row) in enumerate(top_flows.iterrows(), start=1):
                source_goal_num = int(row['source_goal'])
                target_goal_num = int(row['target_goal'])

                summary_records.append({
                    "sdg_goal": goal_num,
                    "sdg_name": goal_name,
                    "flow_category_key": category["key"],
                    "flow_category": category["label"],
                    "flow_category_desc": category["description"],
                    "rank": rank,
                    "source_goal": source_goal_num,
                    "source_goal_name": SDG_NAMES.get(source_goal_num, f"SDG {source_goal_num}"),
                    "source_sdg": row['source_sdg'],
                    "target_goal": target_goal_num,
                    "target_goal_name": SDG_NAMES.get(target_goal_num, f"SDG {target_goal_num}"),
                    "target_sdg": row['target_sdg'],
                    "flow_type": row['flow_type'],
                    "cashflow": row['cashflow']
                })

        print(f"  - 已完成 SDG {goal_num}")

    # 创建DataFrame并保存
    summary_df = pd.DataFrame(summary_records, columns=columns)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["sdg_goal", "flow_category_key", "rank"]
        ).reset_index(drop=True)

    summary_df.to_csv(csv_output, index=False, encoding='utf-8-sig')

    # 输出统计信息
    print(f"\n✅ 分析完成！")
    print(f"   输出文件：{csv_output}")
    print(f"   总记录数：{len(summary_df):,} 条")
    print(f"   覆盖SDG大目标：17 个")
    print(f"   每个大目标最多 15 条记录（3种流类型 × 最多5条）")


if __name__ == "__main__":
    generate_sdg_flow_summary()
