"""单细胞图谱分析. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="单细胞图谱分析", direction="单细胞组学与细胞关系研究", dataset="pbmc3k",
    kind="retrieval", task="retrieval", metric="cluster_agreement", tier="optional", pack="singlecell",
    dim=1838, docs=2638, queries=2638, batches=(8, 1536), rows=2638)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import retrieval
    return retrieval("singlecell_neighbors", device, precision, config["batch_size"], max_items)
