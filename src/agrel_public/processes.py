# Aritheia Community — GPU scientific-computing reliability check
# Copyright (C) 2026 The Aritheia authors
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file
# (GNU GPL v3) shipped with this package or <https://www.gnu.org/licenses/>.
from __future__ import annotations
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from .common import save_json


def run_worker(job: dict, on_event, *, timeout: float) -> int:
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    job_path = out / "worker_job.json"
    save_json(job_path, job)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Source checkouts: make this package importable in the child without installation.
    package_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else package_parent
    # torch.use_deterministic_algorithms(True) needs a fixed cuBLAS workspace on CUDA.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Keep CUDA_VISIBLE_DEVICES unchanged, select logical device inside worker.
    # No global frequency/power/driver changes, no shell, no network in worker.
    events = queue.Queue()
    with (out/"worker_stderr.log").open("w", encoding="utf-8") as err:
        proc = subprocess.Popen([sys.executable,"-m","agrel_public.worker",str(job_path)],
            stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", env=env)
        def read():
            try:
                for line in proc.stdout:
                    if len(line) <= 131072:
                        events.put(line)
            finally:
                events.put(None)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        deadline = time.monotonic()+timeout
        ended = False
        try:
            while not ended:
                cancel = getattr(on_event, "cancel_event", None)
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("GPU run cancelled")
                if time.monotonic() > deadline:
                    proc.kill()
                    on_event({"event":"timeout"})
                    return 124
                try:
                    line = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    ended = True
                else:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    on_event(event)
            return proc.wait(timeout=3)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            proc.stdout.close()
            reader.join(timeout=1)
            job_path.unlink(missing_ok=True)
