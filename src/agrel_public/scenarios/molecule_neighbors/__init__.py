"""分子相似性检索与候选排序. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="分子相似性检索与候选排序", direction="化学信息学与候选筛选", dataset="nci5k",
    kind="retrieval", task="retrieval", metric="self_consistency", tier="bundled", dim=2048, docs=4991, queries=1536,
    batches=(8, 1536), rows=4991)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import retrieval
    return retrieval("molecule_neighbors", device, precision, config["batch_size"], max_items)
