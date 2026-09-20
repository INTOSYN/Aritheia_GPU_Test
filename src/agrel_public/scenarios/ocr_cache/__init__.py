"""视觉特征编码与缓存. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="视觉特征编码与缓存", direction="视觉表征与特征复用", dataset="digits",
    kind="cache", task="classification", metric="accuracy", tier="bundled", nin=64, nout=10,
    batches=(8, 1536), rows=1797)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import supervised
    return supervised("ocr_cache", device, precision, config["batch_size"], out,
                      steps=steps, seed=seed, max_items=max_items)
