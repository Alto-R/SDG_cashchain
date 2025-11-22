import pandas as pd
import numpy as np
from numpy.linalg import norm
from itertools import combinations
from scipy.stats import ks_2samp

# ========= 路径设置 =========
# 1) SBM 输出的 membership（至少要有 node, community 两列）
membership_path = "output/sbm_graphtool_membership.csv"
# 2) 原始网络边表（你之前跑 SBM 用的那个）
edges_path = "sdg_cashflow_network.csv"

# ========= 0. 读入数据 =========
members = pd.read_csv(membership_path)
edges = pd.read_csv(edges_path)

print("membership 列：", members.columns.tolist())
print("edges 列：", edges.columns.tolist())

# 要求 edges 至少包含：source_sdg, target_sdg, cashflow, flow_type
# flow_type ∈ {Pos_to_Pos, Neg_to_Neg, Pos_to_Neg, Neg_to_Pos}

# 只保留出现在 membership 里的节点，并过滤正现金流
valid_nodes = set(members["node"])
edges = edges[
    edges["source_sdg"].isin(valid_nodes)
    & edges["target_sdg"].isin(valid_nodes)
    & (edges["cashflow"] > 0)
].copy()

print(f"过滤后边数: {len(edges)}")

# ========= 1. 准备社区信息 =========
# 社区可能已经是 1..10，如果不是也没关系，我们映射成 0..C-1
unique_comms = sorted(members["community"].unique())
comm2idx = {c: i for i, c in enumerate(unique_comms)}
idx2comm = {i: c for c, i in comm2idx.items()}
num_comms = len(unique_comms)

members["comm_idx"] = members["community"].map(comm2idx)

# 节点 → index
node_list = members["node"].tolist()
node2idx = {n: i for i, n in enumerate(node_list)}
num_nodes = len(node_list)

# 节点 → 社区 index (0..C-1)
node2comm = dict(zip(members["node"], members["comm_idx"]))

print(f"共有 {num_nodes} 个节点, {num_comms} 个社区。")


# ========= 2. 构造“4×社区数”的流向模式矩阵 =========
# 4 层：0=Pos->Pos, 1=Neg->Neg, 2=Pos->Neg, 3=Neg->Pos
layer_map = {
    "Pos_to_Pos": 0,
    "Neg_to_Neg": 1,
    "Pos_to_Neg": 2,
    "Neg_to_Pos": 3,
}
edges["layer"] = edges["flow_type"].map(layer_map)

num_layers = 4

# flow_profile[i, l, c] = 节点 i 在 layer l 上流向社区 c 的加权和
flow_profile = np.zeros((num_nodes, num_layers, num_comms), dtype=float)

for _, row in edges.iterrows():
    src = row["source_sdg"]
    tgt = row["target_sdg"]
    w = row["cashflow"]
    layer = row["layer"]

    if src not in node2idx or tgt not in node2comm:
        continue

    i = node2idx[src]
    c_tgt = node2comm[tgt]  # 0..C-1

    flow_profile[i, layer, c_tgt] += w

# 展开成 (num_nodes, 4*C)
FP = flow_profile.reshape(num_nodes, -1)

# 按行归一化（只看模式，不看规模）
row_sum = FP.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0  # 防止除零
FP_norm = FP / row_sum

print("每个节点的特征维度:", FP_norm.shape)  # (N, 4*C)


# ========= 3. 社区内 vs 社区间：40 维流向模式相似度 =========
def cosine_sim(a, b):
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

comm_idx = members["comm_idx"].values

within_sims = []
between_sims = []

for i, j in combinations(range(num_nodes), 2):
    s = cosine_sim(FP_norm[i], FP_norm[j])
    if comm_idx[i] == comm_idx[j]:
        within_sims.append(s)
    else:
        between_sims.append(s)

within_sims = np.array(within_sims)
between_sims = np.array(between_sims)

print("\n=== 社区内 vs 社区间 流向模式相似度（40 维） ===")
print("社区内  中位数:", np.median(within_sims))
print("社区间  中位数:", np.median(between_sims))
print("社区内  平均值:", within_sims.mean())
print("社区间  平均值:", between_sims.mean())

stat, pval = ks_2samp(within_sims, between_sims)
print("KS 检验: stat =", stat, ", p-value =", pval)


# ========= 4. 每个社区内部的“模式离散度” vs 随机标签 =========
real_disp = {}
for ci in range(num_comms):
    idx = np.where(comm_idx == ci)[0]
    if len(idx) == 0:
        continue
    mu = FP_norm[idx].mean(axis=0)
    dist = norm(FP_norm[idx] - mu, axis=1).mean()
    real_disp[ci] = dist

print("\n=== 每个社区在 40 维流向空间中的真实离散度 d_c ===")
for ci in range(num_comms):
    size = (comm_idx == ci).sum()
    print(
        f"Community {idx2comm[ci]} (idx={ci}): "
        f"size={size}, d_c={real_disp[ci]:.4f}"
    )

# ---- null model：随机打乱社区标签，保持每个社区规模不变 ----
n_perm = 500
rng = np.random.default_rng(123)

null_disp = {ci: [] for ci in real_disp.keys()}

for r in range(n_perm):
    perm_labels = rng.permutation(comm_idx)
    for ci in real_disp.keys():
        idx = np.where(perm_labels == ci)[0]
        mu = FP_norm[idx].mean(axis=0)
        dist = norm(FP_norm[idx] - mu, axis=1).mean()
        null_disp[ci].append(dist)

print("\n=== 真实 d_c 与随机标签基准对比（p 值越小表示社区越“紧”） ===")
for ci in real_disp.keys():
    vals = np.array(null_disp[ci])
    d_real = real_disp[ci]
    p = (vals <= d_real).mean()  # 越小越紧凑
    print(
        f"Community {idx2comm[ci]} (idx={ci}): "
        f"d_real = {d_real:.4f}, "
        f"null_mean = {vals.mean():.4f}, "
        f"p = {p:.3f}"
    )
