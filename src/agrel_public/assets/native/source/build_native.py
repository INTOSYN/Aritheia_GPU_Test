"""Optional developer build: compile smid_probe.cu into a CUDA fatbinary,
disassemble per-architecture cubins to record the instruction gate, and write
NATIVE_MANIFEST.json (source/fatbin SHA-256, nvcc version, targets, SASS counts).

    python publisher/native/build_native.py --nvcc /path/to/nvcc --out public/src/agrel_public/assets/native

Ordinary source/wheel installation uses the included fatbin and needs no nvcc.
Building a different binary does not qualify it against the pinned references:
preserve the shipped assets and use a new output directory for development.
The numerical-reference/kernel hash binding remains mandatory at runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "smid_probe.cu"
# Compute capability 8.0 and later: bf16 mma.sync needs sm_80+. compute_80 PTX is the JIT fallback for newer parts.
SASS_TARGETS = ["sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"]
PTX_TARGET = "compute_80"
GATE = {"smid_mma_bf16": ["HMMA.16816.F32.BF16", "S2R", "SR_VIRTUALSMID"],
        "smid_ffma_fp32": ["FFMA", "S2R", "SR_VIRTUALSMID"]}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def disassembler_path(nvcc):
    compiler = Path(nvcc)
    executable = "nvdisasm.exe" if compiler.suffix.lower() == ".exe" else "nvdisasm"
    return str(compiler.with_name(executable))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvcc", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", default=",".join(SASS_TARGETS))
    a = ap.parse_args()
    targets = a.targets.split(",")
    nvdisasm = disassembler_path(a.nvcc)
    if any((a.out / name).exists() for name in ("smid_probe.fatbin", "NATIVE_MANIFEST.json", "native_reference_v1.json")):
        ap.error("Use a new output directory; do not overwrite qualified kernel/reference assets")
    a.out.mkdir(parents=True, exist_ok=True)
    version = run([a.nvcc, "--version"]).strip().splitlines()[-2:]
    gencode = []
    for t in targets:
        cc = t.replace("sm_", "compute_")
        gencode += ["-gencode", f"arch={cc},code={t}"]
    gencode += ["-gencode", f"arch={PTX_TARGET},code={PTX_TARGET}"]
    fatbin = a.out / "smid_probe.fatbin"
    run([a.nvcc, "-O3", "-fatbin", *gencode, str(SOURCE), "-o", str(fatbin)])
    sass = {}
    with tempfile.TemporaryDirectory() as tmp:
        for t in targets:
            cubin = Path(tmp) / f"{t}.cubin"
            run([a.nvcc, "-O3", "-cubin", f"-arch={t}", str(SOURCE), "-o", str(cubin)])
            text = run([nvdisasm, "-c", str(cubin)])
            # Split the listing per kernel symbol.
            per = {}
            current = None
            for line in text.splitlines():
                m = re.search(r"\.text\.(\w+):", line)
                if m:
                    current = m.group(1)
                    per[current] = []
                elif current:
                    per[current].append(line)
            entry = {}
            for kernel, needles in GATE.items():
                body = "\n".join(per.get(kernel, []))
                counts = {n: len(re.findall(re.escape(n), body)) for n in needles}
                entry[kernel] = {"instruction_counts": counts, "gate_passed": all(counts[n] > 0 for n in needles),
                                 "sass_sha256": hashlib.sha256(body.encode()).hexdigest()}
            sass[t] = entry
    manifest = {"schema": "aritheia.native-manifest.v1", "contract": "native-signed-unit-smid-v1",
                "source": SOURCE.name, "source_sha256": sha(SOURCE), "fatbin": fatbin.name, "fatbin_sha256": sha(fatbin),
                "fatbin_bytes": fatbin.stat().st_size, "nvcc": version, "sass_targets": targets, "ptx_fallback": PTX_TARGET,
                "sass_gate": sass,
                "note": "SASS instruction presence was checked by the maintainer with nvdisasm on cubins built from the same "
                        "source and flags; the client cannot re-check SASS locally (no cuobjdump) and reports this gate as declared."}
    (a.out / "NATIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("fatbin_sha256", "fatbin_bytes", "sass_targets")}, indent=1))
    for t, e in sass.items():
        print(t, {k: v["gate_passed"] for k, v in e.items()}, {k: v["instruction_counts"] for k, v in e.items()})


if __name__ == "__main__":
    main()
