"""
临时脚本：分析每个象限的SDG目标
"""
import pandas as pd
import os
import sys

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# SDG目标中文含义
SDG_MEANINGS = {
    '1.1': '消除极端贫困',
    '1.2': '各类贫困人口减半',
    '1.3': '实施适合本国的社会保护制度',
    '1.4': '确保所有人享有平等获取经济资源的权利',
    '1.5': '增强穷人和弱势群体的抵御灾害能力',
    '2.1': '消除饥饿，确保所有人获得安全、营养和充足的食物',
    '2.2': '消除一切形式的营养不良',
    '2.3': '使小规模粮食生产者的农业生产力和收入翻番',
    '2.4': '确保可持续粮食生产系统',
    '2.5': '保持种子、植物和动物的遗传多样性',
    '2.c': '确保粮食商品市场正常运作，限制粮食价格极端波动',
    '3.1': '降低全球孕产妇死亡率',
    '3.2': '消除新生儿和5岁以下儿童可预防的死亡',
    '3.3': '消除艾滋病、结核病、疟疾等流行病',
    '3.4': '减少非传染性疾病导致的过早死亡',
    '3.5': '加强对滥用药物的预防和治疗',
    '3.6': '减少道路交通事故导致的死伤',
    '3.8': '实现全民健康保障',
    '3.9': '减少危险化学品及污染导致的死亡和患病',
    '3.a': '加强《烟草控制框架公约》的执行',
    '3.b': '支持疫苗和药品的研发',
    '3.d': '加强预警、风险减少和管理国家及全球卫生风险的能力',
    '4.1': '确保所有儿童完成免费、公平和优质的中小学教育',
    '4.2': '确保所有儿童获得优质幼儿发展、看护和学前教育',
    '4.3': '确保所有人平等获得负担得起的优质技术、职业和高等教育',
    '4.4': '大幅增加掌握就业、体面工作和创业所需技能的青年和成年人数',
    '4.5': '消除教育中的性别差距，确保弱势群体平等接受教育',
    '4.6': '确保所有青年和大部分成年人具备识字和算术能力',
    '4.7': '确保所有学习者获得促进可持续发展所需的知识和技能',
    '4.a': '建立和改善适合儿童、残疾和性别敏感的教育设施',
    '5.1': '消除对妇女和女童一切形式的歧视',
    '5.4': '认可和尊重无偿护理和家务劳动',
    '5.5': '确保妇女全面有效参与各级决策',
    '5.6': '确保普遍获得性健康和生殖健康服务',
    '5.a': '进行改革，给予妇女平等获取经济资源的权利',
    '5.b': '加强技术运用，促进妇女赋权',
    '6.1': '实现普遍和公平获得安全和负担得起的饮用水',
    '6.2': '实现适当和公平的环境卫生和个人卫生',
    '6.3': '改善水质，减少污染，消除倾倒废物',
    '6.4': '大幅提高用水效率，减少缺水人数',
    '6.5': '在各级实施水资源综合管理',
    '6.6': '保护和恢复与水有关的生态系统',
    '6.a': '扩大向发展中国家提供的水和卫生方面的国际合作',
    '6.b': '支持和加强地方社区参与改进水和卫生管理',
    '7.1': '确保普遍获得负担得起、可靠和现代的能源服务',
    '7.2': '大幅增加可再生能源在全球能源结构中的比例',
    '7.3': '将全球能源效率的改善速度提高一倍',
    '7.a': '加强国际合作，促进清洁能源研究和技术',
    '8.1': '根据国情保持经济增长',
    '8.2': '通过多样化、技术升级和创新实现更高水平的经济生产力',
    '8.3': '促进以发展为导向的政策，支持创造就业和创业',
    '8.4': '改善资源使用效率，努力使经济增长和环境退化脱钩',
    '8.5': '实现充分和生产性就业，确保同工同酬',
    '8.6': '大幅减少未就业、未受教育或未接受培训的青年比例',
    '8.7': '采取措施消除强迫劳动、现代奴隶制和贩卖人口',
    '8.8': '保护劳工权利，促进安全和有保障的工作环境',
    '8.9': '制定和实施促进可持续旅游业的政策',
    '8.10': '加强国内金融机构的能力',
    '9.1': '发展优质、可靠、可持续和有抵御灾害能力的基础设施',
    '9.2': '促进包容和可持续的工业化',
    '9.3': '增加小型工业企业获得金融服务和融入价值链的机会',
    '9.4': '升级基础设施，改造工业以提升可持续性',
    '9.5': '加强科学研究，提升工业部门的技术能力',
    '9.b': '支持发展中国家的技术开发、研究和创新',
    '9.c': '大幅提升信息和通信技术的普及度',
    '10.1': '逐步实现并维持底层40%人口的收入增长',
    '10.2': '增强所有人的权能，促进社会、经济和政治包容',
    '10.3': '确保机会平等，减少结果不平等',
    '10.4': '采取政策，逐步实现更大程度的平等',
    '10.5': '改善对全球金融市场和金融机构的监管',
    '11.1': '确保所有人获得适当、安全和负担得起的住房和基本服务',
    '11.2': '向所有人提供安全、负担得起、易于使用的可持续交通系统',
    '11.3': '加强包容和可持续的城市化',
    '11.4': '加强努力保护世界文化和自然遗产',
    '11.5': '减少灾害造成的死亡和受灾人数',
    '11.6': '减少城市的环境影响',
    '11.7': '提供安全、包容、无障碍的绿色公共空间',
    '11.a': '加强城乡之间的联系',
    '11.b': '增加采用综合政策和计划的城市和人类住区数量',
    '11.c': '支持最不发达国家建造可持续和有抵御灾害能力的建筑',
    '12.1': '实施可持续消费和生产十年方案框架',
    '12.2': '实现自然资源的可持续管理和高效利用',
    '12.3': '将零售和消费环节的全球人均粮食浪费减半',
    '12.4': '实现化学品和废物的无害环境管理',
    '12.5': '大幅减少废物的产生',
    '12.6': '鼓励企业采用可持续做法并纳入可持续性信息',
    '12.7': '促进可持续的公共采购做法',
    '12.8': '确保人人获得可持续发展信息和意识',
    '12.a': '支持发展中国家加强科技能力，实现可持续消费和生产',
    '12.b': '制定和实施可持续旅游业的监测工具',
    '12.c': '逐步取消鼓励浪费性消费的低效化石燃料补贴',
    '13.1': '加强抵御和适应气候相关灾害的能力',
    '13.2': '将气候变化措施纳入国家政策、战略和规划',
    '13.3': '加强气候变化减缓、适应、影响减少和预警方面的教育',
    '13.a': '发达国家履行承诺，每年筹集1000亿美元应对气候变化',
    '14.1': '预防和大幅减少各类海洋污染',
    '14.2': '可持续管理和保护海洋和沿海生态系统',
    '14.4': '有效规范捕捞，结束过度捕捞和非法捕捞',
    '14.6': '禁止助长过度捕捞能力和过度捕捞的某些渔业补贴',
    '14.7': '增加小岛屿发展中国家和最不发达国家的海洋资源可持续利用的经济收益',
    '14.a': '增加科学知识，发展研究能力和转让海洋技术',
    '14.b': '向小规模个体渔民提供获取海洋资源和市场的机会',
    '14.c': '根据国际法加强海洋及其资源的养护和可持续利用',
    '15.1': '保护、恢复和可持续利用陆地和内陆淡水生态系统',
    '15.2': '促进森林可持续管理，制止毁林',
    '15.3': '防治荒漠化，恢复退化土地',
    '15.5': '减少自然栖息地的退化，遏制生物多样性丧失',
    '15.a': '动员和大幅增加用于保护生物多样性和生态系统的资源',
    '15.c': '加强全球支持，打击偷猎和贩运受保护物种',
    '16.1': '大幅减少一切形式的暴力和相关死亡率',
    '16.2': '制止对儿童的虐待、剥削、贩卖和一切形式的暴力',
    '16.3': '促进法治，确保所有人平等诉诸司法',
    '16.4': '大幅减少非法资金和武器流动，打击有组织犯罪',
    '16.5': '大幅减少一切形式的腐败和贿赂',
    '16.6': '在各级建立有效、负责和透明的机构',
    '16.7': '确保各级决策回应民意、包容、参与和代表',
    '16.10': '确保公众获得信息，保护基本自由',
    '16.a': '加强国家机构预防暴力和打击恐怖主义和犯罪的能力',
    '16.b': '推动和实施非歧视性法律和政策',
    '17.1': '加强国内资源调动',
    '17.3': '从多种来源为发展中国家调动额外财政资源',
    '17.6': '加强科学、技术和创新领域的南北、南南和三方合作',
    '17.9': '加强国际支持，提高发展中国家实施可持续发展目标的能力',
    '17.10': '促进普遍、基于规则、开放、非歧视和公平的多边贸易体制',
    '17.11': '大幅增加发展中国家的出口',
    '17.14': '加强可持续发展政策的一致性',
    '17.16': '加强全球可持续发展伙伴关系',
    '17.17': '鼓励和促进有效的公共、公私和民间社会伙伴关系',
    '17.18': '增强对发展中国家的能力建设支持',
}

# 读取数据
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(base_dir, 'output', 'sdg_quadrant_data.csv')
df = pd.read_csv(csv_path, encoding='utf-8')

# 使用log-standardized版本，边界为0 (均值)
x_col = 'positive_log_standardized'
y_col = 'negative_log_standardized'
x_boundary = 0
y_boundary = 0

print("=" * 80)
print("SDG Quadrant Analysis (Log-Standardized, Mean boundaries)")
print("=" * 80)
print(f"\nBoundaries:")
print(f"  Positive Flow (P) boundary: {x_boundary:.4f} (mean, z-score)")
print(f"  Negative Flow (N) boundary: {y_boundary:.4f} (mean, z-score)")

# 分类到四个象限 (Y轴反向，视觉上上方=N低，下方=N高)
q1 = df[(df[x_col] >= x_boundary) & (df[y_col] < y_boundary)]  # 右上角：P高N低（最优）
q2 = df[(df[x_col] < x_boundary) & (df[y_col] < y_boundary)]  # 左上角：P低N低
q3 = df[(df[x_col] < x_boundary) & (df[y_col] >= y_boundary)]  # 左下角：P低N高（最差）
q4 = df[(df[x_col] >= x_boundary) & (df[y_col] >= y_boundary)]  # 右下角：P高N高

print("\n" + "=" * 80)
print("QUADRANT 1: Positive Dominant (P >= 0, N < 0) - BEST")
print("=" * 80)
print(f"Count: {len(q1)} targets")
print("\nCharacteristics: 高正向流动 + 低负向流动 = 主要产生正向影响的优质领域")
print("这些目标主要作为正面影响的贡献者，较少承受负面影响，是推动SDG进展的重要力量")
print("\nSDG Targets (sorted by positive flow):")
for _, row in q1.sort_values('positive_log_standardized', ascending=False).iterrows():
    target_id = row['sdg_target']
    meaning = SDG_MEANINGS.get(target_id, '（未找到中文描述）')
    print(f"  - {target_id:8s} | P_z={row['positive_log_standardized']:6.2f} | N_z={row['negative_log_standardized']:6.2f}")
    print(f"             {meaning}")

print("\n" + "=" * 80)
print("QUADRANT 2: Dual Low (P < 0, N < 0)")
print("=" * 80)
print(f"Count: {len(q2)} targets")
print("\nCharacteristics: 低正向流动 + 低负向流动 = 相对孤立、影响力有限的边缘领域")
print("这些目标在SDG网络中相对孤立，双向流动都较弱，可能需要更多关注以提升其连接性")
print("\nSDG Targets (sorted by positive flow):")
for _, row in q2.sort_values('positive_log_standardized', ascending=False).iterrows():
    target_id = row['sdg_target']
    meaning = SDG_MEANINGS.get(target_id, '（未找到中文描述）')
    print(f"  - {target_id:8s} | P_z={row['positive_log_standardized']:6.2f} | N_z={row['negative_log_standardized']:6.2f}")
    print(f"             {meaning}")

print("\n" + "=" * 80)
print("QUADRANT 3: Negative Dominant (P < 0, N >= 0) - WORST")
print("=" * 80)
print(f"Count: {len(q3)} targets")
print("\nCharacteristics: 低正向流动 + 高负向流动 = 主要承受负面影响的脆弱领域")
print("这些目标主要作为负面影响的接收者，较少产生正面影响，需要重点保护和支持")
print("\nSDG Targets (sorted by negative flow):")
for _, row in q3.sort_values('negative_log_standardized', ascending=False).iterrows():
    target_id = row['sdg_target']
    meaning = SDG_MEANINGS.get(target_id, '（未找到中文描述）')
    print(f"  - {target_id:8s} | P_z={row['positive_log_standardized']:6.2f} | N_z={row['negative_log_standardized']:6.2f}")
    print(f"             {meaning}")

print("\n" + "=" * 80)
print("QUADRANT 4: Dual High (P >= 0, N >= 0)")
print("=" * 80)
print(f"Count: {len(q4)} targets")
print("\nCharacteristics: 高正向流动 + 高负向流动 = 双向流动活跃的关键枢纽")
print("这些目标既产生显著正面影响，又承受显著负面影响，是SDG网络中的关键枢纽节点")
print("\nSDG Targets (sorted by positive flow):")
for _, row in q4.sort_values('positive_log_standardized', ascending=False).iterrows():
    target_id = row['sdg_target']
    meaning = SDG_MEANINGS.get(target_id, '（未找到中文描述）')
    print(f"  - {target_id:8s} | P_z={row['positive_log_standardized']:6.2f} | N_z={row['negative_log_standardized']:6.2f}")
    print(f"             {meaning}")

print("\n" + "=" * 80)
print("Summary Complete")
print("=" * 80)
