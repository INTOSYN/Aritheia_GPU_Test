"""基因组序列分类. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="基因组序列分类", direction="基因组学与剪接位点识别", dataset="splice",
    kind="supervised", task="classification", metric="accuracy", tier="frozen_asset", nin=240, nout=3,
    batches=(8, 1536), rows=3190)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import supervised
    return supervised("genomics_splice", device, precision, config["batch_size"], out,
                      steps=steps, seed=seed, max_items=max_items)
