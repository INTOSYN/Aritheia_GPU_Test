"""Fixed-output verification. No host matrix multiplication or second GPU.

Hashes locate mismatching blocks; they cannot count every differing value.
The reported count is a lower bound unless a full expected array is present.
"""
from __future__ import annotations
import hashlib
import numpy as np

CHUNK_ELEMENTS = 4096


class ReferenceIntegrityError(ValueError):
    pass


def canonical_array(array):
    x = np.array(array, dtype="<f4", order="C", copy=True)
    x[x == 0] = 0.0
    return x


def output_hash(array):
    return hashlib.sha256(canonical_array(array).tobytes()).hexdigest()


def block_hashes(array, elements=CHUNK_ELEMENTS):
    flat = canonical_array(array).reshape(-1)
    return [hashlib.sha256(flat[i:i + elements].tobytes()).hexdigest()
            for i in range(0, flat.size, elements)]


def compare_blocks(actual, hashes, *, elements=CHUNK_ELEMENTS, expected_hash=None):
    flat = canonical_array(actual).reshape(-1)
    if len(hashes) != (flat.size + elements - 1) // elements:
        raise ValueError("Frozen output block count does not match output shape")
    actual_hash = hashlib.sha256(flat.tobytes()).hexdigest()
    mismatches = []
    lower_bound = 0
    nonfinite = int((~np.isfinite(flat)).sum())
    for j, start in enumerate(range(0, flat.size, elements)):
        values = flat[start:start + elements]
        h = hashlib.sha256(values.tobytes()).hexdigest()
        if h != hashes[j]:
            invalid = (~np.isfinite(values)) | (np.abs(values) > 128) | (values != np.trunc(values))
            lower_bound += max(1, int(invalid.sum()))
            mismatches.append({"block": j, "offset": start, "elements": int(values.size),
                               "actual_sha256": h, "expected_sha256": hashes[j]})
    whole_mismatch = expected_hash is not None and actual_hash != expected_hash
    if expected_hash is not None and whole_mismatch != bool(mismatches):
        raise ReferenceIntegrityError("Frozen whole-output and block digests disagree")
    return {"matched": not mismatches, "actual_sha256": actual_hash,
            "bad_values": lower_bound, "bad_value_count_kind": "lower_bound",
            "mismatch_blocks": len(mismatches), "nonfinite_values": nonfinite,
            "witnesses": mismatches[:32], "failed_block_indices": [x["block"] for x in mismatches]}


def json_number(value):
    v = float(value)
    return v if np.isfinite(v) else None
