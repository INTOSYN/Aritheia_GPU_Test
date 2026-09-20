# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Ordinary scenario workloads, ported unchanged in semantics from the v0.3.0
scenario_core engine: dense MLP inference / short AdamW training / GPU feature
cache consumed by a fixed head on the selected device / dense similarity retrieval.

No injected faults, no fault-aware routing, no custom kernels and no numerical
oracle live here. Metrics are descriptive task outcomes; arithmetic verdicts
come only from the separate exact probes.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..common import array_sha256, sha256_file, load_json
from . import assets
from .registry import MLP_WIDTHS, scenario


def set_execution(seed: int, device: str, precision: str, deterministic: bool = True, threads: int = 4) -> dict:
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU fallback")
        torch.cuda.set_device(torch.device(device))
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected device has no supported BF16 path")
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    # Natural workloads keep the framework's default precision flags; they are
    # recorded, not silently changed, so that scenario runs stay "natural".
    flags = {k: getattr(torch.backends.cuda.matmul, k, None)
             for k in ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")}
    return flags


def dtype_for(name: str):
    import torch
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def amp(device, precision):
    import torch
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype_for(precision), enabled=precision != "fp32")


class MLP:
    """Ordinary dense MLP(input,512,256,output) with ReLU; built lazily to keep torch optional at import."""
    def __new__(cls, nin: int, nout: int, widths=MLP_WIDTHS):
        import torch
        from torch import nn
        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(nin, widths[0]), nn.Linear(widths[0], widths[1]), nn.Linear(widths[1], nout)])
            def embed(self, x):
                return torch.relu(self.layers[1](torch.relu(self.layers[0](x))))
            def forward(self, x):
                return self.layers[2](self.embed(x))
        return _MLP()


def load_weights(path: Path, device: str):
    import torch
    with np.load(path, allow_pickle=False) as z:
        state = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    model = MLP(state["layers.0.weight"].shape[1], state["layers.2.weight"].shape[0],
                (state["layers.0.weight"].shape[0], state["layers.1.weight"].shape[0]))
    model.load_state_dict(state, strict=True)
    return model.to(device)


def prepared(dataset: str):
    p = assets.locate(f"data/prepared/{dataset}/data.npz")
    if p is None:
        raise FileNotFoundError(f"Frozen dataset missing: {dataset}; no generated replacement")
    meta = load_json(assets.locate(f"data/prepared/{dataset}/metadata.json"))
    if sha256_file(p) != meta["prepared_sha256"]:
        raise ValueError("Prepared data checksum mismatch")
    with np.load(p, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    return data, meta


def supervised(name: str, device: str, precision: str, batch_size: int, out: Path, *,
               steps: int = 8, seed: int = 20260910, clip: float = 1.0, lr: float = 1e-4, max_items=None):
    import torch
    spec = scenario(name)
    dataset = spec["dataset"]
    kind = spec["kind"]
    d, meta = prepared(dataset)
    chain = assets.verify_frozen_chain(name)
    model = load_weights(assets.locate(f"models/{dataset}/weights.npz"), device)
    history, alerts = [], []
    if kind == "train":
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        train = np.flatnonzero(d["split"] == 0)
        rng = np.random.default_rng(seed)
        # A fixed seeded minibatch stream, not a fault-triggering input search.
        x = torch.from_numpy(d["x"])
        y = torch.from_numpy(((d["y"] - d["y_mean"]) / d["y_std"]).astype("float32")).reshape(-1, 1)
        for step in range(steps):
            ix = rng.choice(train, size=min(batch_size, len(train)), replace=False)
            xb, yb = x[ix].to(device), y[ix].to(device)
            optim.zero_grad(set_to_none=True)
            with amp(device, precision):
                prediction = model(xb)
            loss = torch.nn.functional.mse_loss(prediction.float(), yb)
            value = float(loss.item())
            if not np.isfinite(value):
                alerts.append({"step": step, "monitor": "loss_isfinite"})
                history.append({"step": step, "loss": None})
                break
            loss.backward()
            gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=False) if clip > 0 else None
            optim.step()
            history.append({"step": step, "loss": value,
                            "raw_grad_norm_if_clipped": float(gradnorm) if gradnorm is not None else None})
        np.savez_compressed(out / "checkpoint.npz", **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    model.eval()
    n = min(max_items, len(d["x"])) if max_items else len(d["x"])
    scores, features, shapes = [], [], []
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(d["x"][i:min(i + batch_size, n)]).to(device)
            with amp(device, precision):
                if kind == "cache":
                    features.append(model.embed(xb).float().cpu().numpy())
                else:
                    scores.append(model(xb).float().cpu().numpy())
            shapes.append([len(xb), d["x"].shape[1], MLP_WIDTHS[0], MLP_WIDTHS[1]])
        if kind == "cache":
            embedding = np.concatenate(features)
            # Cache round trip remains visible; classification runs on the same DUT.
            head = model.layers[-1]
            cached_scores=[]
            for j in range(0,len(embedding),batch_size):
                with amp(device,precision):
                    cached_scores.append(head(torch.from_numpy(embedding[j:j+batch_size]).to(device)).float().cpu().numpy())
            sc=np.concatenate(cached_scores)
            np.savez_compressed(out / "feature_cache.npz", features=embedding, ids=d["ids"][:n])
        else:
            sc = np.concatenate(scores)
    if meta["task"] == "regression":
        sc = sc * d["y_std"] + d["y_mean"]
        decisions = np.argsort(-sc.reshape(-1), kind="stable")[:min(20, n)]
    else:
        decisions = np.argmax(sc, axis=1)
    data = {"scores": sc.astype("float32"), "decisions": decisions, "ids": d["ids"][:n], "y": d["y"][:n],
            "test_mask": d["split"][:n] == 2}
    detail = {"task": meta["task"], "dataset": dataset, "actual_linear_shapes": shapes[:4] + (["..."] if len(shapes) > 4 else []),
              "batches": len(shapes), "normal_alerts": alerts, "training_history": history,
              "updates_completed": len([h for h in history if h["loss"] is not None]),
              "requested_steps": steps if kind == "train" else None, "clip_norm": clip if kind == "train" else None,
              "dataset_sha256": chain["dataset_sha256"], "model_sha256": chain["model_sha256"], "notes": meta["notes"]}
    return data, detail


def retrieval(name: str, device: str, precision: str, batch_size: int, max_items=None):
    import torch
    spec = scenario(name)
    d, meta = prepared(spec["dataset"])
    chain = assets.verify_frozen_chain(name)
    n = min(max_items, len(d["queries"])) if max_items else len(d["queries"])
    docs = torch.from_numpy(d["docs"]).to(device)
    outputs, shapes = [], []
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            q = torch.from_numpy(d["queries"][i:min(i + batch_size, n)]).to(device)
            with amp(device, precision):
                s = q @ docs.T
            s = s.float()
            if meta["similarity"] == "tanimoto":
                # Ordinary molecular fingerprint Tanimoto via a batched dot product.
                denom = q.sum(dim=1, keepdim=True) + docs.sum(dim=1).unsqueeze(0) - s
                s = s / denom.clamp_min(1e-12)
            a = s.cpu().numpy()
            # Excluding the same source record is part of the scientific task, not an oracle.
            for j, index in enumerate(d["self_doc"][i:i + len(q)]):
                if index >= 0:
                    a[j, index] = -2.
            outputs.append(a)
            shapes.append([len(q), d["queries"].shape[1], len(d["docs"])])
    scores = np.concatenate(outputs)
    k = min(10, scores.shape[1])
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    data = {"scores": scores, "decisions": top, "ids": d["query_ids"][:n], "doc_ids": d["doc_ids"],
            "y": d["labels"][:n], "doc_labels": d["doc_labels"], "relevance": d["relevance"][:n],
            "test_mask": np.ones(n, dtype=bool)}
    detail = {"task": "retrieval", "dataset": spec["dataset"], "similarity": meta["similarity"],
              "input_preparation": {k:meta.get(k) for k in ('preparation', 'source_url', 'source_sha256', 'versions', 'reference_kind')},
              "actual_gemm_m_k_n": shapes[:4] + (["..."] if len(shapes) > 4 else []), "batches": len(shapes),
              "dataset_sha256": chain["dataset_sha256"], "model_sha256": None, "notes": meta["notes"]}
    return data, detail


def metrics(data: dict, detail: dict) -> dict:
    s = data["scores"].astype("float64")
    task = detail["task"]
    mask = data["test_mask"]
    if not np.isfinite(s).all():
        return {"finite_output": False, "nonfinite_rows": int((~np.isfinite(s)).any(axis=1).sum()) if s.ndim == 2 else int((~np.isfinite(s)).sum())}
    if task == "classification":
        pred = s.argmax(1); y = data["y"]
        return {"finite_output": True, "accuracy": float((pred[mask] == y[mask]).mean()) if mask.any() else None,
                "evaluated_rows": int(mask.sum()), "all_rows": len(s)}
    if task == "regression":
        p = s.reshape(-1); y = data["y"].astype("float64")
        return {"finite_output": True, "rmse": float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2))) if mask.any() else None,
                "evaluated_rows": int(mask.sum()), "all_rows": len(s)}
    if task == "retrieval":
        top = data["decisions"]; r = data["relevance"]
        out = {"finite_output": True, "queries": len(s)}
        if r.shape[1]:
            rel = r[np.arange(len(r))[:, None], top]; den = np.maximum(r.sum(1), 1)
            out["recall_at_5"] = float(np.mean(rel[:, :5].sum(1) / den))
            gains = 1 / np.log2(np.arange(2, 2 + top.shape[1]))
            ideal = np.array([gains[:min(int(x), len(gains))].sum() for x in r.sum(1)])
            out["ndcg_at_10"] = float(np.mean((rel * gains).sum(1) / np.maximum(ideal, 1e-12)))
        if np.any(data["y"] >= 0):
            labs = data["doc_labels"][top[:, :5]]
            pred = np.array([np.bincount(row[row >= 0]).argmax() for row in labs])
            out["cluster_agreement"] = float((pred == data["y"]).mean())
        return out
    raise ValueError(task)


def run_config(config: dict, device: str, precision: str, out: Path, *, steps: int = 8, seed: int = 20260910, max_items=None) -> dict:
    """Execute one frozen configuration and return a local, whitelisted record."""
    import time, torch
    name = config["scenario"]
    spec = scenario(name)
    out.mkdir(parents=True, exist_ok=True)
    flags = set_execution(seed, device, precision)
    t = time.monotonic()
    from importlib import import_module
    data, detail = import_module(f".{name}", __package__).execute(
        config, device, precision, out, steps=steps, seed=seed, max_items=max_items)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - t
    np.savez_compressed(out / "outputs.npz", **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
    m = metrics(data, detail)
    alerts = list(detail.get("normal_alerts", []))
    if not m["finite_output"]:
        alerts.append({"monitor": "final_output_finite"})
    flat = data["scores"].reshape(-1)[:16]
    decision = np.asarray(data["decisions"], dtype="<i8")
    # Describe existing task decisions after computation; no new GPU calls or oracle.
    membership = np.sort(decision, axis=-1) if detail["task"] != "classification" else decision
    decision_audit = dict(order_sha256=array_sha256(decision),
                          membership_sha256=array_sha256(membership),
                          rule="argmax_first_index" if detail["task"] == "classification" else "stable_descending_index",
                          nonfinite_count=int(np.count_nonzero(~np.isfinite(data["scores"]))))
    return dict(scenario=name, title=spec["title"], config=config, precision=precision,
                status="MONITOR_ALERT" if alerts else "COMPLETED", arithmetic_verdict="NOT_ASSESSED",
                metric=m, output_shape=list(data["scores"].shape), scores_sha256=array_sha256(data["scores"]),
                values=[float(x) if np.isfinite(x) else None for x in flat],
                task_wall_s=round(elapsed, 3), alerts=alerts, math_flags=flags, decision_audit=decision_audit,
                dataset_sha256=detail.get("dataset_sha256"), model_sha256=detail.get("model_sha256"),
                detail={k: detail[k] for k in detail if k not in ("notes",)})
