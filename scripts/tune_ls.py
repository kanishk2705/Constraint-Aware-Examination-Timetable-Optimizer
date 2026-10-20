"""Tune the simulated-annealing parameters on TUNING instances only (PROTOCOL.md section 8.1).

Grid t0 x t1 x p_swap, seed 7, 5 s per run, tiers M and L of the tuning set. Resumable (appends to the jsonl).
Selection: score(config) = mean over instances of cost / min over configs of cost; lowest wins, ties -> lower t0.
"""
import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from examtt.instance import load_instance  # noqa: E402
from examtt.localsearch import LSParams, run_ls  # noqa: E402

GRID = {"t0": [0.5, 1.0, 2.0], "t1": [0.01, 0.05], "p_swap": [0.1, 0.3, 0.5]}
SEED = 7


def work(task):
    path, cfg, budget = task
    inst = load_instance(path)
    r = run_ls(inst, budget, LSParams(**cfg), SEED)
    return {"instance": inst.name, "cfg": cfg, "cost": None if r is None else int(r.cost), "iters": None if r is None else r.iters}


def configs():
    return [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def select(rows):
    cfgs = configs()
    key = lambda c: json.dumps(c, sort_keys=True)  # noqa: E731
    by_inst = {}
    for r in rows:
        by_inst.setdefault(r["instance"], {})[key(r["cfg"])] = r["cost"]
    scores = {key(c): [] for c in cfgs}
    for inst, d in by_inst.items():
        if len(d) != len(cfgs) or any(v is None for v in d.values()):
            continue
        m = max(min(d.values()), 1)
        for k, v in d.items():
            scores[k].append(v / m)
    table = []
    for c in cfgs:
        s = scores[key(c)]
        table.append({"cfg": c, "score": sum(s) / len(s) if s else None, "n_instances": len(s)})
    ranked = sorted([t for t in table if t["score"] is not None], key=lambda t: (round(t["score"], 12), t["cfg"]["t0"]))
    return table, ranked[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tuning")
    ap.add_argument("--out", default="results/tuning_ls_raw.jsonl")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.out)] if os.path.exists(a.out) else []
    if not a.select_only:
        paths = sorted(p for p in glob.glob(os.path.join(a.data, "*.json")) if "_M_" in p or "_L_" in p)
        done = {(r["instance"], json.dumps(r["cfg"], sort_keys=True)) for r in rows}
        tasks = [(p, c, a.budget) for p in paths for c in configs()
                 if (os.path.basename(p)[:-5], json.dumps(c, sort_keys=True)) not in done]
        print(f"{len(rows)} done, {len(tasks)} to run on {len(paths)} instances", flush=True)
        with mp.get_context("fork").Pool(2) as pool, open(a.out, "a") as f:
            for i, r in enumerate(pool.imap_unordered(work, tasks)):
                f.write(json.dumps(r) + "\n")
                f.flush()
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(tasks)}", flush=True)
    table, best = select(rows)
    out = {"protocol": "PROTOCOL.md section 8.1", "budget_s": a.budget, "seed": SEED, "grid": GRID,
           "params": best["cfg"], "score": best["score"], "n_instances": best["n_instances"], "table": table}
    json.dump(out, open("results/tuned_params.json", "w"), indent=1)
    print("selected", best)
