# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Twelve real scientific-computing scenarios. Availability, scientific claims and
hardware evidence are separate axes and are never merged into one green light.

tier:
  bundled       frozen data + model ship inside the client wheel
  frozen_asset  built-in frozen data + model ship inside the client wheel
  optional      large asset pack; each one is offered individually before a run and
                downloaded only after the user sees its size and agrees
"""
from __future__ import annotations

MLP_WIDTHS = (512, 256)

# Historical generic exact-probe dimensions, retained to keep the qualified
# numerical references unchanged. They do NOT describe Qwen3.5-4B architecture
# or certify all operations in that model. Model assets have a separate identity.
LLM = dict(hidden=576, intermediate=1536, heads=9, kv_heads=3, head_dim=64, vocab=49152,
           seq={"short": 256, "long": 1024})

from importlib import import_module

SCENARIOS = {name: import_module(f".{name}", __package__).SPEC for name in (
    "drug_screen",
    "drug_finetune",
    "molecule_neighbors",
    "cytology",
    "ocr_cache",
    "genomics_splice",
    "medical_ultrasound",
    "materials_screen",
    "singlecell_neighbors",
    "biomed_rag",
    "llm_biomed_tables",
    "agent_tools",
)}

ORDER = list(SCENARIOS)
BUILTIN = [n for n, s in SCENARIOS.items() if s["tier"] != "optional"]
OPTIONAL = [n for n, s in SCENARIOS.items() if s["tier"] == "optional"]

# Shared asset packs for the four optional scenarios (one download per pack).
PACKS = {
 "singlecell": dict(scenarios=["singlecell_neighbors"], label="单细胞 PBMC3k 冻结近邻数据"),
 "literature": dict(scenarios=["biomed_rag"], label="SciFact 冻结检索表示与官方相关性"),
 "language": dict(scenarios=["llm_biomed_tables", "agent_tools"], label="Qwen3.5-4B 官方 BF16 权重（两个语言场景共用，须单独同意下载）"),
}


def scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}")
    return SCENARIOS[name]


def configs(name: str) -> list[dict]:
    """Two frozen configurations per scenario, as in the v0.3.x plans."""
    s = scenario(name)
    if s["kind"] == "llm":
        return [dict(scenario=name, batch_size=1, context=c) for c in s["contexts"]]
    return [dict(scenario=name, batch_size=b, context="short") for b in s["batches"]]


def gemms(config: dict) -> list[dict]:
    """Representative forward GEMM dimensions for fixed probes. These do not cover
    every tail batch, real token length, fused path or backward operation."""
    s = scenario(config["scenario"])
    b = config["batch_size"]
    if s["kind"] in ("supervised", "train", "cache"):
        b = min(b, s["rows"])
        return [dict(op="mm", m=b, k=s["nin"], n=MLP_WIDTHS[0]),
                dict(op="mm", m=b, k=MLP_WIDTHS[0], n=MLP_WIDTHS[1]),
                dict(op="mm", m=b, k=MLP_WIDTHS[1], n=s["nout"])]
    if s["kind"] == "retrieval":
        return [dict(op="mm", m=min(b, s["queries"]), k=s["dim"], n=s["docs"])]
    seq = LLM["seq"][config["context"]]
    return [dict(op="mm", m=seq, k=LLM["hidden"], n=LLM["hidden"]),
            dict(op="mm", m=seq, k=LLM["hidden"], n=LLM["intermediate"]),
            dict(op="mm", m=seq, k=LLM["intermediate"], n=LLM["hidden"]),
            dict(op="bmm", batch=LLM["heads"], m=seq, k=LLM["head_dim"], n=seq),
            dict(op="mm", m=1, k=LLM["hidden"], n=LLM["vocab"])]


def config_id(config: dict) -> str:
    return f"{config['scenario']}/b{config['batch_size']}/{config['context']}"
