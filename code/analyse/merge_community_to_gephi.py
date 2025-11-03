# -*- coding: utf-8 -*-
"""
将社区检测结果合并到Gephi节点表中
"""

import pandas as pd

# 读取社区检测结果
community_df = pd.read_csv('../../output/sdg_community_detection.csv', encoding='utf-8-sig')
print(f"社区检测数据: {community_df.shape}")
print(f"社区数量: {community_df['greedy_community'].nunique()}")
print(f"\n各社区节点数:")
print(community_df['greedy_community'].value_counts().sort_index())

# 读取Gephi节点表
gephi_nodes = pd.read_csv('../../output/gephi_nodes_target.csv', encoding='utf-8-sig')
print(f"\nGephi节点表: {gephi_nodes.shape}")
print(f"当前列: {list(gephi_nodes.columns)}")

# 合并数据 (基于 Id 和 node 列)
gephi_nodes_with_community = gephi_nodes.merge(
    community_df,
    left_on='Id',
    right_on='node',
    how='left'
)

# 删除重复的 node 列，保留 greedy_community
gephi_nodes_with_community = gephi_nodes_with_community.drop('node', axis=1)

# 重命名列为 Community
gephi_nodes_with_community = gephi_nodes_with_community.rename(
    columns={'greedy_community': 'Community'}
)

# 调整列顺序，将 Community 放在前面更显眼的位置
cols = ['Id', 'Label', 'SDG_Goal', 'Community', 'Positive_Inflow', 'Positive_Outflow',
        'Negative_Inflow', 'Negative_Outflow', 'Net_Flow', 'Total_Flow', 'Node_Type']
gephi_nodes_with_community = gephi_nodes_with_community[cols]

# 保存更新后的文件
output_file = '../../output/gephi_nodes_target.csv'
gephi_nodes_with_community.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✓ 已更新: {output_file}")
print(f"新列数: {len(gephi_nodes_with_community.columns)}")
print(f"\n更新后的数据样例:")
print(gephi_nodes_with_community.head(10))

# 检查是否所有节点都有社区分配
missing_community = gephi_nodes_with_community['Community'].isna().sum()
if missing_community > 0:
    print(f"\n⚠ 警告: 有 {missing_community} 个节点没有社区分配")
else:
    print(f"\n✓ 所有节点都已成功分配社区")

# 显示每个社区的统计
print(f"\n=== 各社区统计 ===")
for community_id in sorted(gephi_nodes_with_community['Community'].dropna().unique()):
    community_nodes = gephi_nodes_with_community[
        gephi_nodes_with_community['Community'] == community_id
    ]
    print(f"\n社区 {int(community_id)} ({len(community_nodes)} 个节点):")
    print(f"  节点: {sorted(community_nodes['Id'].tolist())}")
    print(f"  平均Net_Flow: {community_nodes['Net_Flow'].mean():,.0f}")
    print(f"  平均Total_Flow: {community_nodes['Total_Flow'].mean():,.0f}")
