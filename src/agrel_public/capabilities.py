"""Capability checks that do not require replacing an existing Torch stack."""
from __future__ import annotations
import importlib.metadata
import platform
import time


def native_bf16_supported(torch):
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        cuda = getattr(torch.version, "cuda", None)
        return bool(cuda and int(cuda.split(".")[0]) >= 11 and
                    torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8)


def execution_stack():
    import torch
    versions = {}
    for name in ("numpy", "transformers", "cryptography"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import subprocess
    driver=None
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=3,check=True)
        versions_seen=set(result.stdout.strip().splitlines())
        if len(versions_seen)==1:
            value=next(iter(versions_seen)).strip()
            if all(c.isdigit() or c=='.' for c in value):driver=value
    except (OSError,subprocess.SubprocessError):pass
    return {"driver":driver,"python": platform.python_version(), "os": platform.system(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, **versions}


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_id():
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parent
    h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()
