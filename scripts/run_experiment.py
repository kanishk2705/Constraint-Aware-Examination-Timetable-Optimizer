"""Run all methods on every instance of a set and append one JSON line per instance.

Resumable: instances already present in the output file are skipped. Every method gets the same wall-clock
budget B and the same two CPU cores:
  * local search : best of 2 independent annealing runs executed in 2 processes
  * CP-SAT       : 2 workers
Study B (soft objective, fixed horizon): greedy_random / greedy_ldf / greedy_dsatur, ls, cpsat_cold, hybrid
Study A (number of slots)             : ff_random / ff_ldf / ff_dsatur, ms_dsatur (restarts), cpsat_slots
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402

from examtt import slots as slotsmod  # noqa: E402
from examtt.cpsat import solve_slots, solve_soft  # noqa: E402
from examtt.greedy import greedy  # noqa: E402
from examtt.instance import load_instance  # noqa: E402
from examtt.localsearch import LSParams, anneal, construct_feasible, run_ls  # noqa: E402

DEFAULT_LS = {"t0": 1.0, "t1": 0.02, "p_swap": 0.3}


def _ls_worker(args):
    path, budget, params, seed = args
    inst = load_instance(path)
    r = run_ls(inst, budget, LSParams(**params), seed)
    if r is None:
        return {"feasible": False}
    return {"feasible": True, "assign": [int(x) for x in r.assign], "cost": r.cost, "iters": r.iters,
            "trajectory": [(round(t, 3), int(c)) for t, c in r.trajectory][-60:], "seed": seed}


def _ms_worker(args):
    path, budget, seed = args
    inst = load_instance(path)
    r = slotsmod.multistart_slots(inst, budget, seed)
    return {"slots": r["slots"], "runs": r["runs"], "assign": None if r["assign"] is None else [int(x) for x in r["assign"]],
            "trajectory": [(round(t, 3), int(k)) for t, k in r["trajectory"]], "seed": seed}


def best_ls(results):
    ok = [r for r in results if r.get("feasible")]
    if not ok:
        return {"feasible": False}
    return min(ok, key=lambda r: r["cost"])


def run_instance(path, budget, ls_params, pool, seeds, studies="AB"):
    inst = load_instance(path)
    rec = {"name": inst.name, "sha256": inst.hash, "params": inst.params,
           "features": {"n": inst.n, "T": inst.T, "S": inst.S, "edges": inst.n_edges, "density": inst.density,
                        "tightness": inst.tightness, "capacity": inst.C, "binary_vars": inst.n_binary_vars},
           "budget": budget, "B": {}, "A": {}}
    if "B" in studies:
        run_study_b(inst, path, budget, ls_params, pool, seeds, rec)
    if "A" in studies:
        run_study_a(inst, path, budget, pool, seeds, rec)
    return rec


def run_study_b(inst, path, budget, ls_params, pool, seeds, rec):
    """Soft objective on the fixed horizon."""
    for order in ("random", "ldf", "dsatur"):
        t0 = time.perf_counter()
        st = greedy(inst, order, rng=np.random.default_rng(seeds["greedy"]))
        dt = time.perf_counter() - t0
        rec["B"][f"greedy_{order}"] = ({"feasible": False, "wall": dt} if st is None else
                                       {"feasible": True, "assign": [int(x) for x in st.assign], "cost": int(st.cost), "wall": dt})

    t0 = time.perf_counter()
    res = pool.map(_ls_worker, [(path, budget, ls_params, s) for s in seeds["ls"]])
    ls = best_ls(res)
    ls["wall"] = time.perf_counter() - t0
    ls["per_seed_costs"] = [r.get("cost") for r in res]
    rec["B"]["ls"] = ls

    o = solve_soft(inst, budget, seeds["cp"], workers=2)
    rec["B"]["cpsat_cold"] = _cp_rec(o)

    t0 = time.perf_counter()
    res = pool.map(_ls_worker, [(path, 0.25 * budget, ls_params, s) for s in seeds["hybrid_ls"]])
    ls_h = best_ls(res)
    hint = np.array(ls_h["assign"]) if ls_h.get("feasible") else None
    o = solve_soft(inst, 0.75 * budget, seeds["cp"], hint=hint, workers=2)
    h = _cp_rec(o)
    h["ls_phase_cost"] = ls_h.get("cost")
    h["wall_total"] = time.perf_counter() - t0
    rec["B"]["hybrid"] = h


def run_study_a(inst, path, budget, pool, seeds, rec):
    """Number of slots (hard constraints only)."""
    rec["A"]["lower_bounds"] = slotsmod.lower_bounds(inst)
    ub_assign, ub = None, None
    for order in ("random", "ldf", "dsatur"):
        a, k = slotsmod.first_fit_slots(inst, order, seeds["greedy"])
        rec["A"][f"ff_{order}"] = {"assign": None if a is None else [int(x) for x in a], "slots": k}
        if k is not None and (ub is None or k < ub):
            ub_assign, ub = a, k
    res = pool.map(_ms_worker, [(path, budget, s) for s in seeds["ms"]])
    ok = [r for r in res if r["slots"] is not None]
    rec["A"]["ms_dsatur"] = min(ok, key=lambda r: r["slots"]) if ok else {"slots": None}
    o = solve_slots(inst, budget, seeds["cp"], upper=ub, hint=ub_assign, workers=2)
    rec["A"]["cpsat_slots"] = {"status": o["status"], "slots": o["objective"], "bound": o["best_bound"], "wall": o["wall"],
                               "assign": None if o["assign"] is None else [int(x) for x in o["assign"]],
                               "trajectory": [(round(t, 3), int(c)) for t, c in o["trajectory"]][-60:]}


def _cp_rec(o):
    return {"feasible": o["assign"] is not None, "status": o["status"], "cost": o["objective"], "bound": o["best_bound"],
            "wall": o["wall"], "build_time": o["build_time"],
            "assign": None if o["assign"] is None else [int(x) for x in o["assign"]],
            "trajectory": [(round(t, 3), int(c)) for t, c in o["trajectory"]][-60:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--ls-params", default=None, help="JSON file with tuned SA parameters")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--studies", default="AB")
    ap.add_argument("--only", default=None, help="substring filter on instance names")
    a = ap.parse_args()

    ls_params = DEFAULT_LS
    if a.ls_params:
        ls_params = json.load(open(a.ls_params))["params"]
    seeds = {"greedy": 0, "ls": [101, 102], "hybrid_ls": [201, 202], "ms": [301, 302], "cp": 1}
    paths = sorted(p for p in glob.glob(os.path.join(a.data, "*.json")) if not p.endswith("MANIFEST.json"))
    if a.only:
        paths = [p for p in paths if a.only in os.path.basename(p)]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            done.add(json.loads(line)["name"])
    todo = [p for p in paths if os.path.basename(p)[:-5] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} done, {len(todo)} to run, budget {a.budget}s, LS params {ls_params}", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(2) as pool, open(a.out, "a") as fout:
        for i, p in enumerate(todo):
            t0 = time.time()
            rec = run_instance(p, a.budget, ls_params, pool, seeds, a.studies)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            b, A = rec["B"], rec["A"]
            msg = f"[{i + 1}/{len(todo)}] {rec['name']} {time.time() - t0:.0f}s"
            if b:
                msg += (f"  dsatur={b['greedy_dsatur'].get('cost')} ls={b['ls'].get('cost')} "
                        f"cp={b['cpsat_cold'].get('cost')}({b['cpsat_cold'].get('status')}) hyb={b['hybrid'].get('cost')}")
            if A:
                msg += (f" | slots ff={A['ff_dsatur']['slots']} ms={A['ms_dsatur']['slots']} "
                        f"cp={A['cpsat_slots']['slots']}({A['cpsat_slots']['status']}) lb={A['lower_bounds']['lb']}")
            print(msg, flush=True)
