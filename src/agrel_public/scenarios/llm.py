# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Pinned Qwen3.5-4B for both language scenarios.
Weights come from the official repository after explicit consent; no remote API, no random
stand-in model and no live tool execution. Transformers is an optional dependency."""
from __future__ import annotations
import inspect
from pathlib import Path
import numpy as np
from ..common import load_json, sha256_file
from . import assets
from .registry import scenario

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def model_dir() -> Path:
    p = assets.locate(f"{assets.LLM_DIR}/config.json")
    if p is None:
        raise FileNotFoundError("Pinned LLM weights/manifest missing; install the 'language' optional pack. No random fallback.")
    return p.parent


def verify_model(path: Path) -> str:
    manifest = assets.optional_manifest()['packs']['language']
    if manifest["model"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("Wrong LLM revision")
    expected_weights = {Path(e['path']).name for e in manifest['members'] if e['path'].endswith('.safetensors')}
    if {p.name for p in path.glob('*.safetensors')} != expected_weights or (path / 'adapter_config.json').exists():
        raise ValueError('Only the original Qwen3.5-4B weights are supported; no adapters or replacement weights')
    for entry in manifest['members']:
        name, expected = Path(entry['path']).name, entry['sha256']
        file = (path / name).resolve()
        if file.parent != path.resolve():
            raise ValueError("Unsafe model manifest path")
        if not file.is_file() or file.stat().st_size != entry['bytes'] or sha256_file(file) != expected:
            raise ValueError(f"LLM file integrity mismatch: {name}")
    return sha256_file(assets.OPTIONAL_MANIFEST)


def prompt_text(record, context_size):
    rows = record[context_size + "_context"]
    context = "\n".join(f"compound={r['Name']}, Act={r['Act']}" for r in rows)
    return "Reference table (Act is the stored assay score, not a clinical recommendation):\n" + context + "\n\n" + record["question"]


def evaluate(name: str, device: str, precision: str, batch_size: int, context_size: str, max_items=None, max_length: int = 4096):
    import torch
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise RuntimeError("DEPENDENCY_MISSING: transformers is required for the language scenarios") from e
    from .engine import dtype_for
    if precision != 'bf16':
        raise ValueError('Qwen3.5-4B scenario validation covers BF16 only; no alternate precision fallback')
    spec = scenario(name)
    path = model_dir()
    manifest_sha = verify_model(path)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                 dtype=dtype_for(precision), attn_implementation="eager").to(device).eval()
    prompts_path = assets.locate(f"data/prepared/{spec['dataset']}/prompts.json")
    records = load_json(prompts_path)
    records = records[:max_items] if max_items else records
    prompts, lengths = [], []
    limit = min(max_length, int(getattr(model.config, "max_position_embeddings", max_length)))
    for r in records:
        text = tok.apply_chat_template([{"role": "user", "content": prompt_text(r, context_size)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        length = len(tok.encode(text, add_special_tokens=False))
        if length > limit:
            raise ValueError(f"Prompt {r['id']} has {length} tokens > effective maximum={limit}; no silent truncation of evidence")
        prompts.append(text); lengths.append(length)
    choice_ids = [tok.encode(c, add_special_tokens=False) for c in "ABCD"]
    if any(len(x) != 1 for x in choice_ids):
        raise ValueError("Tokenizer does not have single-token A/B/C/D; choice-scoring contract unsupported")
    choice_ids = [x[0] for x in choice_ids]
    signature = inspect.signature(model.forward).parameters
    kwargs = {"logits_to_keep": 1} if "logits_to_keep" in signature else ({"num_logits_to_keep": 1} if "num_logits_to_keep" in signature else {})
    outputs, shapes = [], []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            b = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            shapes.append(list(b["input_ids"].shape))
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(**b, use_cache=False, **kwargs).logits[:, -1, :]
            outputs.append(logits[:, choice_ids].float().cpu().numpy())
    scores = np.concatenate(outputs)
    valid = np.isfinite(scores).all(axis=1)
    decisions = np.full(len(scores), -1, dtype=np.int64)
    decisions[valid] = np.argmax(scores[valid], axis=1)
    # A constrained choice is formatted by the host; schema success does not imply the chosen evidence/tool is correct.
    actions = [{"id": r["id"], "selected": r["options"][int(j)] if j >= 0 else None, "choice": "ABCD"[int(j)] if j >= 0 else None, "executed": False}
               for r, j in zip(records, decisions)]
    data = {"scores": scores, "decisions": decisions, "ids": np.asarray([r["id"] for r in records]),
            "y": np.asarray([r["gold"] for r in records]), "test_mask": np.ones(len(records), dtype=bool)}
    detail = {"task": "classification", "model_id": MODEL_ID, "model_revision": REVISION, "context": context_size,
              "sequence_lengths": lengths, "actual_batch_token_shapes": shapes, "actions": actions,
              "sampling": False, "kv_cache": False, "attention": "eager", "enable_thinking": False,
              "valid_choice_rows": int(valid.sum()), "nonfinite_choice_rows": int((~valid).sum()),
              "scope": "Constrained next-token data-reading/tool-routing, not an autonomous agent benchmark.",
              "dataset_sha256": sha256_file(prompts_path), "model_sha256": manifest_sha, "notes": []}
    return data, detail
