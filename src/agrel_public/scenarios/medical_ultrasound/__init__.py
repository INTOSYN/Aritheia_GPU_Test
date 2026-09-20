"""医疗影像分类. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="医疗影像分类", direction="智能医疗影像研究", dataset="breastmnist",
    kind="supervised", task="classification", metric="accuracy", tier="frozen_asset", nin=784, nout=2,
    batches=(8, 512), rows=780)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..engine import supervised
    return supervised("medical_ultrasound", device, precision, config["batch_size"], out,
                      steps=steps, seed=seed, max_items=max_items)
