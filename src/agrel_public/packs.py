# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
from __future__ import annotations
import base64
import re
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .common import canonical, digest, load_json, strict_keys

class PackError(ValueError):
    pass


def verify_pack(data: bytes, trusted_keys: dict[str, str], max_bytes: int = 16*1024*1024) -> dict:
    import json
    if len(data) > max_bytes:
        raise PackError("实验包超过大小限制")
    try:
        envelope = json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        strict_keys(envelope, {"payload", "key_id", "signature"})
        raw_key = base64.b64decode(trusted_keys[envelope["key_id"]], validate=True)
        sig = base64.b64decode(envelope["signature"], validate=True)
        Ed25519PublicKey.from_public_bytes(raw_key).verify(sig, canonical(envelope["payload"]))
        payload = envelope["payload"]
        strict_keys(payload, {"schema", "pack_id", "backend", "provenance", "data"},
                    {"created_utc", "reference_receipt"})
        if payload["schema"] != "aritheia.pack.v1":
            raise PackError("不支持的实验包版本")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", payload["pack_id"]):
            raise PackError("非法实验包 ID")
        if payload["backend"] not in {"fixed_mm_v1", "private_core_v1"}:
            raise PackError("禁止从实验包执行代码或加载任意模块")
        if payload["provenance"] not in {"software_fixture", "publisher_validated"}:
            raise PackError("实验包必须明确证据身份")
        if payload["provenance"] == "publisher_validated" and not payload.get("reference_receipt"):
            raise PackError("正式参考缺少发布者验收摘要")
        return payload
    except PackError:
        raise
    except Exception as e:
        raise PackError("实验包签名、信任根或格式校验失败") from e


def load_pack(path: str | Path, keys: dict[str, str], limit: int) -> tuple[dict, str]:
    p = Path(path)
    with p.open("rb") as f:
        data = f.read(limit + 1)
    return verify_pack(data, keys, limit), digest(data)


def cached_pack(cache: Path, keys: dict[str, str], limit: int) -> Path | None:
    if not cache.exists():
        return None
    for p in sorted(cache.glob("*.agpack"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            load_pack(p, keys, limit)
            return p
        except (OSError, PackError):
            continue
    return None
