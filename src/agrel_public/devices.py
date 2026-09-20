# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
from __future__ import annotations
from dataclasses import dataclass, field
import csv
import io
import re
import subprocess

@dataclass(frozen=True)
class Device:
    logical_index: int
    model: str
    memory_mib: int
    core_clock_mhz: int | None = None
    compute_capability: str | None = None

    _identity: str | None = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        return f"cuda:{self.logical_index}"

    def public_card(self) -> dict:
        # Exactly these three fields may leave the host during bootstrap.
        return {"model": self.model, "memory_mib": self.memory_mib,
                "core_clock_mhz": self.core_clock_mhz}


def normalize_uuid(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes) and len(value) == 16:
        import uuid
        return "gpu-" + str(uuid.UUID(bytes=value))
    s = str(value).lower().strip()
    match = re.fullmatch(r"(?:gpu-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", s)
    return "gpu-" + match.group(1) if match else None


def local_clock_map() -> dict[str, int]:
    # UUID is used transiently to match NVML/SMI physical identity to a CUDA
    # visible ordinal. Never match by model or ordinal, and never serialize UUID.
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=uuid,clocks.current.graphics",
            "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=2, check=True)
        result = {}
        for row in csv.reader(io.StringIO(proc.stdout)):
            if len(row) != 2:
                continue
            key = normalize_uuid(row[0].strip())
            if key and row[1].strip().isdigit():
                result[key] = int(row[1].strip())
        return result
    except (OSError, subprocess.SubprocessError):
        return {}


def inventory() -> list[Device]:
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("请在现有环境安装与你驱动匹配的 PyTorch；本工具不会自动重装 Torch。") from e
    if not torch.cuda.is_available():
        return []
    clocks = local_clock_map()
    result = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        identity = normalize_uuid(getattr(props, "uuid", None))
        result.append(Device(index, str(props.name), int(props.total_memory // (1024 ** 2)),
                             clocks.get(identity), f"{props.major}.{props.minor}", identity))
    return result


def select_devices(devices: list[Device], selection: str) -> list[Device]:
    if selection == "all":
        return list(devices)
    tokens = selection.split(",")
    indices = []
    for token in tokens:
        token = token.strip()
        if not re.fullmatch(r"(?:cuda:)?[0-9]+", token):
            raise ValueError("设备格式为 cuda:0、cuda:0,cuda:1 或 all")
        index = int(token.removeprefix("cuda:"))
        if index in indices:
            raise ValueError("不能重复选择同一可见设备")
        indices.append(index)
    mapping = {d.logical_index: d for d in devices}
    if any(i not in mapping for i in indices):
        raise ValueError("选择了当前进程不可见的 GPU；索引按 CUDA_VISIBLE_DEVICES 解释")
    return [mapping[i] for i in indices]


def bootstrap_payload(devices: list[Device]) -> dict:
    return {"cards": [d.public_card() for d in devices]}
