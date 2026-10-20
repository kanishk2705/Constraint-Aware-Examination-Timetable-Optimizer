"""Generate the instance sets. Seeds and grids are fixed in PROTOCOL.md.

  pilot   – seeds 9000+  : used only to calibrate sizes/formulation before the protocol; never reused
  tuning  – seeds 1000+  : used for all tuning decisions (SA parameters, decision rule)
  eval    – seeds 5000+  : held out; touched only by the final frozen run
Each set gets a MANIFEST.json with the sha256 of every instance file (leakage-style discipline).
"""
import argparse
import hashlib
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from examtt.generator import generate  # noqa: E402

SIZES = {"S": (14, 2), "M": (32, 4), "L": (64, 7)}   # tier -> (n_exams, n_days); horizon = 4 periods per day
DENSITIES = {"low": 0.10, "mid": 0.25, "high": 0.45}
TIGHTNESS = {"loose": 0.70, "medium": 0.85, "tight": 0.97}
SETS = {"pilot": (9000, 1), "tuning": (1000, 2), "eval": (5000, 4)}  # base seed, instances per cell


def build(set_name: str, out_dir: str):
    base, per_cell = SETS[set_name]
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"set": set_name, "base_seed": base, "per_cell": per_cell, "instances": {}}
    k = 0
    for (sz, (n, D)), (dn, dv), (tn, tv) in itertools.product(SIZES.items(), DENSITIES.items(), TIGHTNESS.items()):
        for rep in range(per_cell):
            seed = base + k
            k += 1
            name = f"{set_name}_{sz}_{dn}_{tn}_{rep}"
            d = generate(name, seed, n, dv, tv, n_days=D)
            d["params"].update({"size_tier": sz, "density_level": dn, "tightness_level": tn, "rep": rep})
            path = os.path.join(out_dir, name + ".json")
            blob = json.dumps(d, sort_keys=True)
            with open(path, "w") as f:
                f.write(blob)
            manifest["instances"][name] = {"seed": seed, "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=list(SETS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "data", a.set)
    m = build(a.set, out)
    print(a.set, len(m["instances"]), "instances ->", os.path.abspath(out))
