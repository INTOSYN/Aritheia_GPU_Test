"""药物性质模型微调. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="药物性质模型微调", direction="分子模型训练与优化", dataset="chembl_example",
    kind="train", task="regression", metric="rmse", tier="bundled", nin=2048, nout=1,
    batches=(8, 512), rows=1017)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import supervised
    return supervised("drug_finetune", device, precision, config["batch_size"], out,
                      steps=steps, seed=seed, max_items=max_items)
