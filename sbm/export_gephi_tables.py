import pandas as pd
from pathlib import Path

# 输入/输出路径（基于当前文件所在目录，避免运行目录不同导致找不到文件）
base = Path(__file__).resolve().parent
csv_path = base / "sdg_cashflow_network.csv"
out_nodes = base / "gephi_nodes.csv"
out_edges = base / "gephi_edges.csv"

df = pd.read_csv(csv_path)

# 边表（Gephi）
edges = df.rename(
    columns={
        "source_sdg": "Source",
        "target_sdg": "Target",
        "cashflow": "Weight",
        "flow_type": "flow_type",
        "transaction_count": "transaction_count",
    }
)
edges["Type"] = "Directed"  # Gephi要求
edges = edges[["Source", "Target", "Type", "Weight", "flow_type", "transaction_count"]]

# 节点表（Gephi）
nodes_unique = pd.unique(edges[["Source", "Target"]].values.ravel())
nodes = pd.DataFrame({"Id": nodes_unique})
nodes["Label"] = nodes["Id"]

# 从名称推断正/负极性（_Pos/_Neg）
nodes["polarity"] = nodes["Id"].str.extract(r"_(Pos|Neg)$")

# 可选：提取纯SDG编号（去掉后缀）
nodes["sdg_code"] = nodes["Id"].str.replace(r"_(Pos|Neg)$", "", regex=True)

# 读取社区文件并添加社区编号（如果存在）
comm_path = base / "output" / "sbm_graphtool_membership.csv"
if comm_path.exists():
    comm_df = pd.read_csv(comm_path)
    if "total_degree" not in comm_df.columns:
        comm_df["total_degree"] = (
            comm_df["pos_pos_degree"]
            + comm_df["neg_neg_degree"]
            + comm_df["pos_neg_degree"]
            + comm_df["neg_pos_degree"]
        )
    comm_flow = (
        comm_df.groupby("community")["total_degree"]
        .sum()
        .sort_values(ascending=False)
    )
    comm_rename = {old: new for new, old in enumerate(comm_flow.index, start=1)}
    comm_df["community"] = comm_df["community"].map(comm_rename)
    nodes = nodes.merge(
        comm_df[["node", "community"]],
        how="left",
        left_on="Id",
        right_on="node",
    ).drop(columns=["node"])

# 保存
edges.to_csv(out_edges, index=False)
nodes.to_csv(out_nodes, index=False)

print(f"Edges -> {out_edges}")
print(f"Nodes -> {out_nodes}")

# Override export with cleaned logic: community renamed by total_degree descending and node total_degree included
def export_gephi_with_comm():
    base = Path(__file__).resolve().parent
    csv_path = base / "sdg_cashflow_network.csv"
    out_nodes = base / "gephi_nodes.csv"
    out_edges = base / "gephi_edges.csv"

    df = pd.read_csv(csv_path)
    edges = df.rename(
        columns={
            "source_sdg": "Source",
            "target_sdg": "Target",
            "cashflow": "Weight",
            "flow_type": "flow_type",
            "transaction_count": "transaction_count",
        }
    )
    edges["Type"] = "Directed"
    edges = edges[["Source", "Target", "Type", "Weight", "flow_type", "transaction_count"]]

    nodes_unique = pd.unique(edges[["Source", "Target"]].values.ravel())
    nodes = pd.DataFrame({"Id": nodes_unique})
    nodes["Label"] = nodes["Id"]
    nodes["polarity"] = nodes["Id"].str.extract(r"_(Pos|Neg)$")
    nodes["sdg_code"] = nodes["Id"].str.replace(r"_(Pos|Neg)$", "", regex=True)

    comm_path = base / "output" / "sbm_graphtool_membership.csv"
    if comm_path.exists():
        comm_df = pd.read_csv(comm_path)
        if "total_degree" not in comm_df.columns:
            comm_df["total_degree"] = (
                comm_df["pos_pos_degree"]
                + comm_df["neg_neg_degree"]
                + comm_df["pos_neg_degree"]
                + comm_df["neg_pos_degree"]
            )
        comm_flow = (
            comm_df.groupby("community")["total_degree"]
            .sum()
            .sort_values(ascending=False)
        )
        comm_rename = {old: new for new, old in enumerate(comm_flow.index, start=1)}
        comm_df["community"] = comm_df["community"].map(comm_rename)
        nodes = nodes.merge(
            comm_df[["node", "community", "total_degree"]],
            how="left",
            left_on="Id",
            right_on="node",
        ).drop(columns=["node"])

    edges.to_csv(out_edges, index=False)
    nodes.to_csv(out_nodes, index=False)

    print(f"[override] Edges -> {out_edges}")
    print(f"[override] Nodes -> {out_nodes}")


if __name__ == "__main__":
    export_gephi_with_comm()
