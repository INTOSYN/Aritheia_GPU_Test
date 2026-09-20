"""大语言模型智能体决策. GPL-3.0-or-later; public specification and execution entry."""
SPEC = dict(title="大语言模型智能体决策", direction="工具选择与任务决策", dataset="agent_prompts",
    kind="llm", task="classification", metric="accuracy", tier="optional", pack="language",
    contexts=("short", "long"), rows=32)


def execute(config, device, precision, out, *, steps=8, seed=20260910, max_items=None):
    from ..llm import evaluate
    return evaluate("agent_tools", device, precision, config["batch_size"], config["context"], max_items)
