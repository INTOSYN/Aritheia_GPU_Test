# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
"""Frozen scenario assets: where they live, whether they are complete, and the
explicit, size-disclosed, hash-verified download of the optional packs.

Lookup order for every dataset/model: the package's bundled tree first, then the
user asset root (ARITHEIA_ASSETS or ~/.aritheia/assets). Nothing is ever
generated, tiled or randomly substituted when a frozen file is missing.
"""
from __future__ import annotations
import hashlib
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from ..common import config_home, load_json, save_json, sha256_file, strict_keys
from .registry import PACKS, SCENARIOS, scenario

PACKAGE_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "scenarios"
LLM_DIR = "models/qwen3.5-4b"


def user_root() -> Path:
    return Path(os.environ.get("ARITHEIA_ASSETS", str(config_home() / "assets")))


def roots() -> list[Path]:
    return [PACKAGE_ASSETS, user_root()]


def required_files(name: str) -> list[str]:
    s = scenario(name)
    d = s["dataset"]
    if s["kind"] == "llm":
        return [f"data/prepared/{d}/prompts.json", f"data/prepared/{d}/metadata.json", f"{LLM_DIR}/config.json"]
    files = [f"data/prepared/{d}/data.npz", f"data/prepared/{d}/metadata.json"]
    if s["kind"] in ("supervised", "train", "cache"):
        files += [f"models/{d}/weights.npz", f"models/{d}/training_report.json"]
    return files


def locate(relative: str, search: list[Path] | None = None) -> Path | None:
    for root in (search or roots()):
        p = root / relative
        if p.is_file():
            return p
    return None


def status(name: str) -> dict:
    """Readiness of one scenario. 'missing' lists relative files, never guesses."""
    s = scenario(name)
    missing = [f for f in required_files(name) if locate(f) is None]
    integrity = None
    if not missing:
        try:
            verify_llm_chain() if s["kind"] == "llm" else verify_frozen_chain(name)
            verify_required_hashes(name)
        except Exception as e:  # reported, never silently repaired
            integrity = f"{type(e).__name__}: {e}"
    return dict(scenario=name, title=s["title"], tier=s["tier"], dataset=s["dataset"],
                pack=s.get("pack"), ready=not missing and integrity is None,
                missing=missing, integrity_error=integrity)


def verify_required_hashes(name):
    from .upstream import verify_prepared
    if scenario(name)['kind'] == 'retrieval' and scenario(name)['tier'] == 'optional' and verify_prepared(name):
        return
    entries=dict(load_json(MANIFEST)['files'])
    s=scenario(name)
    if s['tier']=='optional':
        entries.update({e['path']:e for e in optional_manifest()['packs'][s['pack']]['members']})
    for rel in required_files(name):
        p=locate(rel);e=entries.get(rel)
        if e is None or p is None or p.stat().st_size!=e['bytes'] or sha256_file(p)!=e['sha256']:
            raise ValueError('Asset differs from publisher manifest: '+rel)


def verify_llm_chain():
    from .llm import model_dir, verify_model
    # The shipped manifest, not an editable local manifest, binds every model file.
    verify_model(model_dir())
    return True


def verify_frozen_chain(name: str, search: list[Path] | None = None) -> dict:
    """metadata.prepared_sha256 == sha(data.npz); training_report binds weights to data."""
    s = scenario(name)
    d = s["dataset"]
    data = locate(f"data/prepared/{d}/data.npz", search)
    meta = load_json(locate(f"data/prepared/{d}/metadata.json", search))
    if meta.get("synthetic", False):
        raise ValueError("Synthetic prepared data is not an accepted frozen scenario asset")
    if meta["prepared_sha256"] != sha256_file(data):
        raise ValueError("Prepared data checksum mismatch")
    out = dict(dataset_sha256=meta["prepared_sha256"], model_sha256=None)
    if s["kind"] in ("supervised", "train", "cache"):
        weights = locate(f"models/{d}/weights.npz", search)
        report = load_json(locate(f"models/{d}/training_report.json", search))
        if report["weights_sha256"] != sha256_file(weights) or report["dataset_sha256"] != meta["prepared_sha256"]:
            raise ValueError("Frozen model/data integrity mismatch; rebuild reference artifacts, do not blame the GPU")
        out["model_sha256"] = report["weights_sha256"]
    return out


def all_status() -> list[dict]:
    return [status(n) for n in SCENARIOS]


# ---------------------------------------------------------------- bundled manifest
MANIFEST = PACKAGE_ASSETS / "ASSET_MANIFEST.json"


def freeze_manifest(path: Path = MANIFEST) -> dict:
    """Record SHA-256 of every bundled scenario file."""
    files = {}
    for p in sorted(PACKAGE_ASSETS.rglob("*")):
        if p.is_file() and p.name != path.name and "__pycache__" not in p.parts:
            files[str(p.relative_to(PACKAGE_ASSETS))] = dict(bytes=p.stat().st_size, sha256=sha256_file(p))
    manifest = dict(schema="aritheia.asset-manifest.v1", files=files)
    save_json(path, manifest)
    return manifest


def verify_manifest(path: Path = MANIFEST) -> list[str]:
    """Return a list of problems (empty == every bundled file matches the manifest)."""
    if not path.exists():
        return ["ASSET_MANIFEST.json missing; run `assets freeze` in the maintainer checkout"]
    manifest = load_json(path)
    problems = []
    for rel, entry in manifest["files"].items():
        p = PACKAGE_ASSETS / rel
        if not p.is_file():
            problems.append(f"missing {rel}")
        elif p.stat().st_size != entry["bytes"] or sha256_file(p) != entry["sha256"]:
            problems.append(f"changed {rel}")
    return problems


# ---------------------------------------------------------------- optional packs
OPTIONAL_MANIFEST = Path(__file__).resolve().parent.parent / "assets" / "optional_assets.json"
SAFE_MEMBER = re.compile(r"^(data/prepared|models)/[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*$")


def optional_manifest(path: Path | None = None) -> dict:
    m = load_json(path or OPTIONAL_MANIFEST)
    strict_keys(m, {"schema", "release_url", "packs"})
    if m["schema"] != "aritheia.optional-assets.v1":
        raise ValueError("Unsupported optional asset manifest")
    for name, pack in m["packs"].items():
        if name not in PACKS:
            raise ValueError("Unknown optional pack " + name)
        strict_keys(pack, {"file", "bytes", "sha256", "members", "approx_bytes", "license"}, {"upstream", "model", "revision"})
        for member in pack["members"]:
            strict_keys(member, {"path", "bytes", "sha256"}, {"url"})
            if not SAFE_MEMBER.match(member["path"]) or ".." in member["path"].split("/"):
                raise ValueError("Unsafe member path in optional manifest")
    return m


def pack_pinned(pack: dict) -> bool:
    ok = lambda h: isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) is not None
    if pack.get('upstream'):
        e = pack['upstream']
        return ok(e.get('sha256')) and isinstance(e.get('bytes'), int) and e['bytes'] > 0
    if pack.get('model'):
        return pack['model'] == 'Qwen/Qwen3.5-4B' and bool(pack['members']) and all(ok(e['sha256']) and e['bytes'] > 0 and e.get('url', '').startswith('https://huggingface.co/Qwen/Qwen3.5-4B/resolve/'+pack['revision']+'/') for e in pack['members'])
    return (ok(pack["sha256"]) and isinstance(pack["bytes"], int) and pack["bytes"] > 0
            and bool(pack["members"]) and all(ok(m["sha256"]) and isinstance(m["bytes"], int) for m in pack["members"]))


def pack_status(pack_name: str, manifest: dict | None = None) -> dict:
    manifest = manifest or optional_manifest()
    pack = manifest["packs"][pack_name]
    present = bool(pack["members"]) and all((p := locate(m["path"])) is not None and p.stat().st_size == m["bytes"] and sha256_file(p) == m["sha256"] for m in pack["members"])
    if not present and pack.get('upstream'):
        from .upstream import verify_prepared
        present = all(verify_prepared(name) for name in PACKS[pack_name]['scenarios'])
    return dict(pack=pack_name, label=PACKS[pack_name]["label"], scenarios=PACKS[pack_name]["scenarios"],
                installed=present, pinned=pack_pinned(pack), bytes=pack.get('upstream', {}).get('bytes') or pack["bytes"] or pack["approx_bytes"],
                release_url=pack.get('upstream', {}).get('url') or ('https://huggingface.co/'+pack['model'] if pack.get('model') else manifest["release_url"]), file=pack["file"], license=pack["license"])


def human_bytes(n: int | None) -> str:
    if not n:
        return "体积未知"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000
    return f"{n:.1f} GB"


def download_pack(pack_name: str, transport, *, release_url: str | None = None,
                  manifest: dict | None = None, progress=lambda done, total: None,
                  cancel: threading.Event | None = None, dest_root: Path | None = None, model_consent=False) -> Path:
    """Download ONE optional pack after explicit consent. Verify ZIP size, SHA-256,
    member list and each member's SHA-256 before anything reaches the asset root."""
    manifest = manifest or optional_manifest()
    pack = manifest["packs"][pack_name]
    if pack_name == 'language' and not model_consent:
        raise PermissionError('Qwen3.5-4B model download requires separate explicit consent')
    if not pack_pinned(pack):
        raise ValueError("Optional pack digest is not pinned by the publisher; refusing to download unverifiable data")
    if pack.get('upstream') or pack.get('model'):
        from .upstream import download
        return download(pack_name, pack, transport, cancel=cancel, progress=progress, dest_root=dest_root)
    base = (release_url or manifest["release_url"] or "").rstrip("/")
    if not base:
        raise ValueError("No asset release URL configured (config asset_release_url or AGREL_ASSET_RELEASE_URL)")
    root = dest_root or user_root()
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".pack-", dir=root))
    try:
        zip_path = tmp_dir / pack["file"]
        h = hashlib.sha256()
        done = 0
        deadline = time.monotonic() + 3600
        with transport.open(base + "/" + pack["file"]) as response, zip_path.open("wb") as f:
            while True:
                if (cancel and cancel.is_set()) or time.monotonic() > deadline:
                    raise TimeoutError("Download cancelled; no partial pack installed")
                block = response.read(min(1 << 20, pack["bytes"] - done + 1))
                if not block:
                    break
                done += len(block)
                if done > pack["bytes"]:
                    raise ValueError("Download exceeded the declared pack size")
                h.update(block)
                f.write(block)
                progress(done, pack["bytes"])
        if done != pack["bytes"] or h.hexdigest() != pack["sha256"]:
            raise ValueError("Pack size/SHA-256 mismatch; nothing installed")
        expected = {m["path"]: m for m in pack["members"]}
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            if sorted(names) != sorted(expected):
                raise ValueError("Pack member list differs from the pinned manifest")
            if z.testzip():
                raise ValueError("ZIP CRC failed")
            for name in names:
                info = z.getinfo(name)
                if info.file_size != expected[name]["bytes"]:
                    raise ValueError("Member size mismatch: " + name)
                target = tmp_dir / "extract" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                if sha256_file(target) != expected[name]["sha256"]:
                    raise ValueError("Member SHA-256 mismatch: " + name)
        # Preflight ALL destinations before installing the first member.
        for name in names:
            final = root/name
            if any(p.is_symlink() for p in [final, *final.parents] if p != root.parent):
                raise ValueError('Symlink destination refused')
            if final.exists() and (not final.is_file() or sha256_file(final) != expected[name]['sha256']):
                raise FileExistsError('Different asset file already exists: '+name)
        # All verified: move members into place; never overwrite different existing files.
        for name in names:
            final = root / name
            src = tmp_dir / "extract" / name
            if final.exists():
                if sha256_file(final) == expected[name]["sha256"]:
                    continue
                raise FileExistsError(f"Different file already at {final}; choose another ARITHEIA_ASSETS root")
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, final)
        return root
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def import_tree(source: Path, names: list[str], dest_root: Path | None = None) -> dict:
    """Release preparation for frozen_asset scenarios: copy the reference-prepared files
    of the named scenarios from a v0.3.x asset tree and verify the frozen chain."""
    root = dest_root or PACKAGE_ASSETS
    copied = {}
    for name in names:
        for rel in required_files(name):
            src = Path(source) / rel
            if not src.is_file():
                raise FileNotFoundError(f"{name}: {rel} is not in {source}; nothing is generated in its place")
            dst = root / rel
            if dst.exists() and sha256_file(dst) != sha256_file(src):
                raise FileExistsError(f"Refusing to overwrite a different {rel}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied[rel] = sha256_file(dst)
        if scenario(name)["kind"] != "llm":
            verify_frozen_chain(name, [root])
    return copied


def install_pack(pack_name: str, archive: Path, *, dest_root=None):
    """Explicit local ZIP installation, using the identical pinned download verifier."""
    archive=Path(archive).resolve()
    if pack_name == 'language':
        raise ValueError('Only the pinned Qwen3.5-4B original files are supported; legacy language ZIPs are not accepted')
    class LocalArchive:
        def open(self,url):return archive.open('rb')
    manifest = optional_manifest()
    manifest['packs'][pack_name].pop('upstream', None)
    return download_pack(pack_name,LocalArchive(),release_url='local://archive',dest_root=dest_root,manifest=manifest)
