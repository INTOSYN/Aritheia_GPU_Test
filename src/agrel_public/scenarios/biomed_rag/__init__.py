"""生物医学知识检索. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="生物医学知识检索", direction="医学文献检索与 RAG", dataset="scifact",
    kind="retrieval", task="retrieval", metric="recall_at_5", tier="optional", pack="literature",
    dim=512, docs=5183, queries=300, batches=(8, 1536), rows=5183)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import retrieval
    return retrieval("biomed_rag", device, precision, config["batch_size"], max_items)
