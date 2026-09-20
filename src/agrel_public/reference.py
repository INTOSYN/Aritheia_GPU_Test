# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Pinned whole-output and block references. Runtime verification hashes bytes only.
Reference construction and independent FP64 GPU qualification are publisher tasks.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from .common import load_json, save_json, sha256_file
from .exact import CONTRACT, EXPONENTS, operand_digest, operands, output_digest, validate_spec
from .scenarios.registry import ORDER, config_id, configs, gemms

REFERENCE_VERSION = "exact-signed-unit-v050-20260912"
REFERENCE_PATH = Path(__file__).resolve().parent / "assets" / "fixed_reference_v1.json"
# Pinned by `reference build`; verified before any probe runs.
REFERENCE_SHA256 = "95e28940e848f1ad516a03bdb38b41cf022b913f79fb6022a8af401b22baf5f7"
BASE_SEED = 20260912
ITERATIONS = 8
PRECISION = "bf16"


def probe_seed(probe_id: str, iteration: int) -> int:
    base = int(hashlib.sha256(probe_id.encode()).hexdigest()[:8], 16)
    return (BASE_SEED + base + iteration * 104729) % 2 ** 32


def probe_specs(scenarios=None, iterations: int = ITERATIONS, precision: str = PRECISION) -> list[dict]:
    """Deterministic probe list: each configuration's forward GEMMs, layouts and
    exponents cycling by position so the set is fixed, not chosen per run."""
    if not 1 <= iterations <= ITERATIONS:
        raise ValueError(f"iterations must be 1..{ITERATIONS}")
    specs = []
    index = 0
    for name in (scenarios or ORDER):
        for config in configs(name):
            cid = config_id(config)
            for j, g in enumerate(gemms(config)):
                pid = f"{cid}/g{j}-{g['op']}-{g.get('batch', 1)}x{g['m']}x{g['k']}x{g['n']}"
                spec = dict(id=pid, scenario=name, config=cid, op=g["op"], batch=g.get("batch", 1), m=g["m"], k=g["k"], n=g["n"],
                            layout="contiguous" if index % 2 == 0 else "transposed", exponent=EXPONENTS[index % 3],
                            precision=precision, seeds=[probe_seed(pid, i) for i in range(iterations)])
                validate_spec(spec)
                specs.append(spec)
                index += 1
    return specs


def load(path: Path = REFERENCE_PATH, *, pinned: str | None = REFERENCE_SHA256) -> dict:
    if not path.exists():
        raise FileNotFoundError("Frozen exact reference missing from this installation")
    if pinned and sha256_file(path) != pinned:
        raise ValueError("Frozen reference SHA-256 differs from the pinned value; refusing to run probes against a modified reference")
    doc = load_json(path)
    if doc.get("schema") != "aritheia.exact-reference.v1" or doc.get("contract") != CONTRACT:
        raise ValueError("Unsupported reference schema/contract")
    for spec in doc["probes"]:
        validate_spec(spec)
    return doc


def select(doc: dict, scenarios: list[str], iterations: int) -> list[dict]:
    """Probes for the selected scenarios, truncated to the first `iterations` seeds."""
    if not 1 <= iterations <= doc["iterations"]:
        raise ValueError("iterations exceed the frozen reference")
    wanted = set(scenarios)
    out = []
    for spec in doc["probes"]:
        if spec["scenario"] in wanted:
            s = dict(spec)
            for key in ("seeds", "expected_sha256", "operand_sha256", "expected_blocks_sha256"):
                s[key] = spec[key][:iterations]
            out.append(s)
    return out


def verify(path: Path = REFERENCE_PATH, *, limit: int | None = None, progress=lambda i,n,p:None, device=None) -> dict:
    """Verify the pinned manifest and registry declarations; no numeric recomputation."""
    if device is not None:
        raise ValueError("Public reference verification needs no device; GPU regeneration is a publisher task")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    doc = load(path)
    specs = probe_specs()
    mismatches = []
    if len(specs) != len(doc['probes']):
        mismatches.append('probe count')
    for declared, frozen in zip(specs, doc['probes']):
        if any(frozen.get(k) != v for k,v in declared.items()):
            mismatches.append(declared['id'])
    return dict(ok=not mismatches, pinned_match=True, file_sha256=sha256_file(path),
                probes_checked=len(doc['probes']), probes_total=len(doc['probes']), mismatches=mismatches,
                verification='pinned_file_and_registry_integrity; no host GEMM; no GPU')
