"""大语言模型结构化信息理解. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="大语言模型结构化信息理解", direction="科学信息抽取", dataset="compound_prompts",
    kind="llm", task="classification", metric="accuracy", tier="optional", pack="language",
    contexts=("short", "long"), rows=32)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..llm import evaluate
    return evaluate("llm_biomed_tables", device, precision, config["batch_size"], config["context"], max_items)
