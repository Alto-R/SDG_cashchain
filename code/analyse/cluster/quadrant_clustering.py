"""
SDG Quadrant Clustering Analysis
对SDG象限图中的点进行聚类分析，使用不同颜色标识不同聚类
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import sys

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try to import adjustText for better label positioning
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Warning: adjustText not installed. Labels may overlap.")

# SDG目标中文含义（从 analyze_quadrants.py 复制）
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

def load_data(input_csv: str) -> pd.DataFrame:
    """加载SDG象限数据"""
    print(f"Loading data: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"  Loaded {len(df)} SDG targets")

    # 添加象限分类 (Y轴反向，视觉上上方=N低，下方=N高)
    df['quadrant'] = df.apply(lambda row:
        'Q1' if row['positive_log_standardized'] >= 0 and row['negative_log_standardized'] < 0  # 右上角：P高N低（最优）
        else 'Q2' if row['positive_log_standardized'] < 0 and row['negative_log_standardized'] < 0  # 左上角：P低N低
        else 'Q3' if row['positive_log_standardized'] < 0 and row['negative_log_standardized'] >= 0  # 左下角：P低N高（最差）
        else 'Q4', axis=1)  # 右下角：P高N高

    return df

def plot_elbow_curve(df: pd.DataFrame, output_path: str):
    """绘制肘部法则曲线"""
    print("\nGenerating elbow curve...")

    X = df[['positive_log_standardized', 'negative_log_standardized']].values
    k_range = range(2, 11)
    sse = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Microsoft YaHei']
    plt.rcParams['font.size'] = 12

    plt.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of clusters (k)', fontsize=14)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=14)
    plt.title('Elbow Method for Optimal k', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_silhouette_scores(df: pd.DataFrame, output_path: str):
    """绘制轮廓系数图"""
    print("\nGenerating silhouette score plot...")

    X = df[['positive_log_standardized', 'negative_log_standardized']].values
    k_range = range(2, 11)
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Microsoft YaHei']
    plt.rcParams['font.size'] = 12

    plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of clusters (k)', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Silhouette Score for Different k Values', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim([min(silhouette_scores) - 0.05, max(silhouette_scores) + 0.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_dendrogram_chart(df: pd.DataFrame, output_path: str):
    """绘制层次聚类树状图"""
    print("\nGenerating dendrogram...")

    X = df[['positive_log_standardized', 'negative_log_standardized']].values
    linkage_matrix = linkage(X, method='ward')

    # 绘图
    plt.figure(figsize=(14, 8))
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Microsoft YaHei']
    plt.rcParams['font.size'] = 10

    dendrogram(
        linkage_matrix,
        labels=df['sdg_target'].values,
        leaf_font_size=6,
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )

    plt.xlabel('SDG Target', fontsize=14)
    plt.ylabel('Ward Distance', fontsize=14)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_clustering_scatter(df: pd.DataFrame, cluster_col: str, n_clusters: int,
                            output_path: str, title: str):
    """绘制聚类散点图（保留象限参考线）"""
    print(f"\nPlotting clustering scatter: {title}...")

    # Set seaborn style
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

    # Create figure
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(6, 6, figure=fig, hspace=0.05, wspace=0.05)

    # Main scatter plot
    ax = fig.add_subplot(gs[1:, :-1])

    # Marginal histograms
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax)

    # Calculate point sizes
    min_size = 50
    max_size = 500
    total_activity = df['positive_total_flow'] + df['negative_total_flow']

    if total_activity.max() > total_activity.min():
        normalized_activity = (total_activity - total_activity.min()) / (total_activity.max() - total_activity.min())
        sizes = min_size + normalized_activity * (max_size - min_size)
    else:
        sizes = np.full(len(df), min_size)

    sizes = np.where(total_activity == 0, min_size, sizes)

    # 色盲友好配色
    if n_clusters <= 10:
        colors_palette = plt.cm.tab10(range(n_clusters))
    else:
        colors_palette = plt.cm.tab20(range(n_clusters))

    # Prepare data
    plot_df = pd.DataFrame({
        'x': df['positive_log_standardized'],
        'y': df['negative_log_standardized'],
        'size': sizes,
        'cluster': df[cluster_col],
        'sdg_target': df['sdg_target'].values
    })

    # Invert Y-axis
    ax.invert_yaxis()

    # Add quadrant reference lines (黑色实线)
    ax.axvline(0, color='black', linestyle='-', alpha=0.9, linewidth=SPINE_WIDTH, zorder=1)
    ax.axhline(0, color='black', linestyle='-', alpha=0.9, linewidth=SPINE_WIDTH, zorder=1)

    # Plot scatter points by cluster
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]
        ax.scatter(
            cluster_data['x'],
            cluster_data['y'],
            s=cluster_data['size'],
            c=[colors_palette[cluster_id]],
            alpha=0.7,
            edgecolor='white',
            linewidth=1,
            label=f'Cluster {cluster_id}',
            zorder=3
        )

    # Add labels with adjustText if available
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

        # Adjust text positions
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
        for _, row in plot_df.iterrows():
            ax.annotate(
                row['sdg_target'],
                (row['x'], row['y']),
                fontsize=SCATTER_LABEL_SIZE,
                alpha=0.7,
                ha='center',
                va='center'
            )

    # Marginal histograms
    ax_top.hist(df['positive_log_standardized'], bins=30, alpha=0.5,
                color='#555555', edgecolor='black', linewidth=0.5)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.axis('off')

    ax_right.hist(df['negative_log_standardized'], bins=30, orientation='horizontal',
                  alpha=0.5, color='#555555', edgecolor='black', linewidth=0.5)
    ax_right.set_ylim(ax.get_ylim())
    ax_right.axis('off')

    # Labels and styling
    ax.set_xlabel('Positive Flow P (log-standardized, z-score)')
    ax.set_ylabel('Negative Flow N (log-standardized, z-score, ↓ increasing)')

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_edgecolor('black')

    ax.tick_params(axis='both', which='major', width=0.8, length=4)
    ax.tick_params(axis='both', which='minor', width=0.5, length=2)

    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3, color='gray', zorder=0)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.2, color='gray', zorder=0)

    # Legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

def perform_kmeans(df: pd.DataFrame, k: int, output_dir: str) -> str:
    """执行K-Means聚类"""
    print(f"\nPerforming K-Means clustering (k={k})...")

    X = df[['positive_log_standardized', 'negative_log_standardized']].values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    cluster_col = f'cluster_kmeans_k{k}'
    df[cluster_col] = labels

    # Calculate silhouette score
    score = silhouette_score(X, labels)
    print(f"  Silhouette score: {score:.3f}")

    # Plot
    output_path = os.path.join(output_dir, f'sdg_clustering_kmeans_k{k}.png')
    plot_clustering_scatter(df, cluster_col, k, output_path, f'K-Means Clustering (k={k})')

    return cluster_col

def perform_hierarchical(df: pd.DataFrame, n_clusters: int, output_dir: str) -> str:
    """执行层次聚类"""
    print(f"\nPerforming Hierarchical clustering (n_clusters={n_clusters})...")

    X = df[['positive_log_standardized', 'negative_log_standardized']].values
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(X)

    cluster_col = 'cluster_hierarchical'
    df[cluster_col] = labels

    # Calculate silhouette score
    score = silhouette_score(X, labels)
    print(f"  Silhouette score: {score:.3f}")

    # Plot
    output_path = os.path.join(output_dir, f'sdg_clustering_hierarchical.png')
    plot_clustering_scatter(df, cluster_col, n_clusters, output_path,
                           f'Hierarchical Clustering (n_clusters={n_clusters})')

    return cluster_col

def print_cluster_analysis(df: pd.DataFrame, cluster_col: str, cluster_name: str):
    """打印聚类分析结果"""
    print("\n" + "=" * 80)
    print(f"{cluster_name} - Cluster Analysis")
    print("=" * 80)

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]

        print(f"\nCluster {cluster_id}: {len(cluster_data)} targets")
        print(f"  Average P: {cluster_data['positive_log_standardized'].mean():.3f}")
        print(f"  Average N: {cluster_data['negative_log_standardized'].mean():.3f}")

        # 象限分布
        quadrant_dist = cluster_data['quadrant'].value_counts()
        print(f"  Quadrant distribution: {dict(quadrant_dist)}")

        print(f"  Targets:")
        for _, row in cluster_data.iterrows():
            target_id = row['sdg_target']
            meaning = SDG_MEANINGS.get(target_id, '（未找到中文描述）')
            print(f"    - {target_id:8s} | P={row['positive_log_standardized']:6.2f} | N={row['negative_log_standardized']:6.2f}")
            print(f"               {meaning}")

def save_clustering_results(df: pd.DataFrame, output_path: str):
    """保存聚类结果到CSV"""
    print(f"\nSaving clustering results to: {output_path}")

    output_cols = ['sdg_target', 'positive_log_standardized', 'negative_log_standardized',
                   'quadrant']

    # Add all cluster columns
    cluster_cols = [col for col in df.columns if col.startswith('cluster_')]
    output_cols.extend(cluster_cols)

    df[output_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  Saved {len(df)} rows with {len(cluster_cols)} clustering methods")

def main():
    """主函数"""
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    input_csv = os.path.join(base_dir, 'output', 'sdg_quadrant_data.csv')
    output_dir = os.path.join(base_dir, 'output', 'cluster')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("SDG Quadrant Clustering Analysis")
    print("=" * 80)

    # 1. 加载数据
    df = load_data(input_csv)

    # 2. 聚类质量评估
    print("\n" + "=" * 80)
    print("Clustering Quality Assessment")
    print("=" * 80)

    elbow_path = os.path.join(output_dir, 'sdg_clustering_elbow.png')
    plot_elbow_curve(df, elbow_path)

    silhouette_path = os.path.join(output_dir, 'sdg_clustering_silhouette.png')
    plot_silhouette_scores(df, silhouette_path)

    dendrogram_path = os.path.join(output_dir, 'sdg_clustering_dendrogram.png')
    plot_dendrogram_chart(df, dendrogram_path)

    # 3. K-Means聚类（k=3,4,5,6）
    print("\n" + "=" * 80)
    print("K-Means Clustering")
    print("=" * 80)

    kmeans_cols = []
    for k in [3, 4, 5, 6]:
        col = perform_kmeans(df, k, output_dir)
        kmeans_cols.append(col)

    # 4. 层次聚类
    print("\n" + "=" * 80)
    print("Hierarchical Clustering")
    print("=" * 80)

    hierarchical_col = perform_hierarchical(df, 4, output_dir)

    # 5. 保存结果
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    results_csv = os.path.join(output_dir, 'sdg_clustering_results.csv')
    save_clustering_results(df, results_csv)

    # 6. 打印聚类分析报告
    print("\n" + "=" * 80)
    print("Clustering Analysis Report")
    print("=" * 80)

    # 打印 K-Means k=4 的详细分析（作为示例）
    print_cluster_analysis(df, 'cluster_kmeans_k4', 'K-Means (k=4)')

    # 打印层次聚类的详细分析
    print_cluster_analysis(df, hierarchical_col, 'Hierarchical Clustering (n=4)')

    print("\n" + "=" * 80)
    print("Clustering Analysis Complete!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}")
    print("  - 4 K-Means clustering plots (k=3,4,5,6)")
    print("  - 1 Hierarchical clustering plot")
    print("  - 1 Elbow curve plot")
    print("  - 1 Silhouette score plot")
    print("  - 1 Dendrogram plot")
    print("  - 1 Clustering results CSV")

if __name__ == '__main__':
    main()
