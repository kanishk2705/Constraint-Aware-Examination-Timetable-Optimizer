"""OR-Tools CP-SAT models.

``solve_soft``  – fixed horizon (all ``inst.T`` slots), minimise the soft objective exactly as defined in
                  instance.py. Hard constraints: each exam in exactly one slot, at most one exam per student
                  cohort per slot (clique constraints, one per distinct cohort exam set), room capacity per slot.
                  Soft terms: pairwise proximity penalty per conflicting exam pair (weighted by shared students) via
                  an ELEMENT constraint on the clipped position distance, and the per-student-per-day overload.
``solve_slots`` – minimise the number of slots used (hard constraints only, no day structure).
Both run a single worker so that two evaluation processes can share the two cores; wall-clock limited.
"""
from __future__ import annotations

import time

import numpy as np
from ortools.sat.python import cp_model

from .instance import DIST_CLIP, Instance


class _Trace(cp_model.CpSolverSolutionCallback):
    def __init__(self, t0: float):
        super().__init__()
        self.t0 = t0
        self.traj: list = []

    def on_solution_callback(self):
        self.traj.append((time.perf_counter() - self.t0, int(round(self.ObjectiveValue()))))


def _status_name(s) -> str:
    return {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE", cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.UNKNOWN: "UNKNOWN", cp_model.MODEL_INVALID: "MODEL_INVALID"}.get(s, str(s))


def _add_hard(inst: Instance, model: cp_model.CpModel, x, n_slots: int):
    for e in range(inst.n):
        model.AddExactlyOne(x[e][t] for t in range(n_slots))
    seen = set()
    for ex in inst.stu_exams:
        key = tuple(ex.tolist())
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        for t in range(n_slots):
            model.AddAtMostOne(x[e][t] for e in key)
    for t in range(n_slots):
        model.Add(sum(int(inst.sizes[e]) * x[e][t] for e in range(inst.n)) <= inst.C)


def solve_soft(inst: Instance, time_limit: float, seed: int, hint: np.ndarray | None = None,
               workers: int = 1, formulation: str = "bool", linearization: int | None = None) -> dict:
    t_build = time.perf_counter()
    T = inst.T
    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x{e}_{t}") for t in range(T)] for e in range(inst.n)]
    _add_hard(inst, model, x, T)

    terms = []
    if formulation == "elem":
        pos_max = int(inst.pos.max())
        posv = []
        for e in range(inst.n):
            v = model.NewIntVar(0, pos_max, f"pos{e}")
            model.Add(v == sum(int(inst.pos[t]) * x[e][t] for t in range(T)))
            posv.append(v)
        table = [int(p) for p in inst.PT]  # index 0..DIST_CLIP
        for i, j, w in zip(inst.PI.tolist(), inst.PJ.tolist(), inst.PW.tolist()):
            diff = model.NewIntVar(-pos_max, pos_max, "")
            model.Add(diff == posv[i] - posv[j])
            ab = model.NewIntVar(0, pos_max, "")
            model.AddAbsEquality(ab, diff)
            cl = model.NewIntVar(0, DIST_CLIP, "")
            model.AddMinEquality(cl, [ab, DIST_CLIP])
            pen = model.NewIntVar(0, max(table), "")
            model.AddElement(cl, table, pen)
            terms.append(int(w) * pen)
    elif formulation == "bool":
        # y[i,j,delta] = 1 if exams i and j sit at position distance delta (1..3). One implication per slot t:
        # x[i,t] + sum_{t': dist(t,t')=delta} x[j,t'] - 1 <= y   (j occupies exactly one slot, so the sum is <= 1)
        by_dist = {}
        for t in range(T):
            for t2 in range(T):
                dlt = abs(int(inst.pos[t]) - int(inst.pos[t2]))
                if 1 <= dlt < DIST_CLIP:
                    by_dist.setdefault((t, dlt), []).append(t2)
        for i, j, w in zip(inst.PI.tolist(), inst.PJ.tolist(), inst.PW.tolist()):
            for dlt in range(1, DIST_CLIP):
                pw = int(inst.PT[dlt]) * int(w)
                if pw == 0:
                    continue
                y = model.NewBoolVar("")
                for t in range(T):
                    nbrs = by_dist.get((t, dlt))
                    if nbrs:
                        model.Add(x[i][t] + sum(x[j][t2] for t2 in nbrs) - 1 <= y)
                terms.append(pw * y)
    else:
        raise ValueError(formulation)

    if inst.H:
        for h, i in enumerate(inst.heavy.tolist()):
            ex = inst.stu_exams[i].tolist()
            m = int(inst.stu_mult[i])
            for d in range(inst.D):
                cnt = sum(x[e][t] for e in ex for t in range(d * inst.P, (d + 1) * inst.P))
                over = model.NewIntVar(0, len(ex), "")
                model.Add(over >= cnt - inst.N)
                terms.append(inst.W_OV * m * over)
    model.Minimize(sum(terms))

    if hint is not None:
        for e in range(inst.n):
            for t in range(T):
                model.AddHint(x[e][t], 1 if int(hint[e]) == t else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit - (time.perf_counter() - t_build), 0.5)
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = False
    if linearization is not None:
        solver.parameters.linearization_level = linearization
    t0 = time.perf_counter()
    cb = _Trace(t0)
    status = solver.Solve(model, cb)
    wall = time.perf_counter() - t0
    out = {"status": _status_name(status), "wall": wall, "build_time": t0 - t_build, "trajectory": cb.traj,
           "best_bound": None, "objective": None, "assign": None, "n_vars_note": f"{inst.n}x{T} bools + {inst.n_edges} pair terms"}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        a = np.array([next(t for t in range(T) if solver.Value(x[e][t])) for e in range(inst.n)], dtype=np.int64)
        out["assign"] = a
        out["objective"] = int(round(solver.ObjectiveValue()))
        out["best_bound"] = float(solver.BestObjectiveBound())
    elif status == cp_model.UNKNOWN:
        out["best_bound"] = float(solver.BestObjectiveBound())
    return out


def solve_slots(inst: Instance, time_limit: float, seed: int, upper: int, hint: np.ndarray | None = None,
                workers: int = 1) -> dict:
    """Minimise the number of slots used under the hard constraints (slot count capped at ``upper``)."""
    model = cp_model.CpModel()
    H = upper
    x = [[model.NewBoolVar(f"x{e}_{t}") for t in range(H)] for e in range(inst.n)]
    _add_hard(inst, model, x, H)
    used = [model.NewBoolVar(f"u{t}") for t in range(H)]
    for t in range(H):
        for e in range(inst.n):
            model.AddImplication(x[e][t], used[t])
    for t in range(H - 1):
        model.AddImplication(used[t + 1], used[t])  # symmetry breaking: used slots are a prefix
    model.Minimize(sum(used))
    if hint is not None:
        for e in range(inst.n):
            for t in range(H):
                model.AddHint(x[e][t], 1 if int(hint[e]) == t else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    t0 = time.perf_counter()
    cb = _Trace(t0)
    status = solver.Solve(model, cb)
    out = {"status": _status_name(status), "wall": time.perf_counter() - t0, "trajectory": cb.traj,
           "best_bound": None, "objective": None, "assign": None}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["assign"] = np.array([next(t for t in range(H) if solver.Value(x[e][t])) for e in range(inst.n)], dtype=np.int64)
        out["objective"] = int(round(solver.ObjectiveValue()))
        out["best_bound"] = float(solver.BestObjectiveBound())
    return out
