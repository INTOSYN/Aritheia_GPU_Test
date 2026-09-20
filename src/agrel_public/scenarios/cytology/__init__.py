"""乳腺细胞学特征分类. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="乳腺细胞学特征分类", direction="智能细胞学分析", dataset="breast_cancer",
    kind="supervised", task="classification", metric="accuracy", tier="bundled", nin=30, nout=2,
    batches=(8, 512), rows=569)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import supervised
    return supervised("cytology", device, precision, config["batch_size"], out,
                      steps=steps, seed=seed, max_items=max_items)
