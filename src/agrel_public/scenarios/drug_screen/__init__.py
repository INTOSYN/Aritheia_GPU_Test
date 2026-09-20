"""药物候选批量评分. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="药物候选批量评分", direction="计算药物发现", dataset="chembl_example",
    kind="supervised", task="regression", metric="rmse", tier="bundled", nin=2048, nout=1,
    batches=(8, 1536), rows=1017)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import supervised
    return supervised("drug_screen", device, precision, config["batch_size"], out,
                      steps=steps, seed=seed, max_items=max_items)
