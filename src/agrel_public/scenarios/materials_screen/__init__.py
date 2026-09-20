"""材料性能预测. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="材料性能预测", direction="材料信息学与超导材料筛选", dataset="superconductivity",
    kind="supervised", task="regression", metric="rmse", tier="frozen_asset", nin=81, nout=1,
    batches=(8, 1536), rows=21263)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import supervised
    return supervised("materials_screen", device, precision, config["batch_size"], out,
                      steps=steps, seed=seed, max_items=max_items)
