"""Simulated-annealing improvement step (relocate and swap moves, hard constraints never violated).

The temperature is expressed in units of ``scale`` = mean |delta| of sampled feasible relocations from
the start solution, so the same (tuned) multipliers transfer across instance sizes.
A wall-clock budget drives a geometric schedule T(f) = T0 * (T1/T0)^f with f = elapsed/budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .greedy import greedy
from .instance import Instance
from .state import State


@dataclass
class LSParams:
    t0: float = 1.0       # initial temperature multiplier
    t1: float = 0.02      # final temperature multiplier
    p_swap: float = 0.3   # probability of proposing a swap instead of a relocation


@dataclass
class LSResult:
    assign: np.ndarray
    cost: int
    iters: int
    trajectory: list = field(default_factory=list)  # (seconds, best cost)


def construct_feasible(inst: Instance, rng: np.random.Generator, deadline: float, horizon: int | None = None):
    """Best feasible greedy among the three deterministic baselines, else randomised DSATUR restarts."""
    best = None
    for order in ("dsatur", "ldf", "random"):
        st = greedy(inst, order, rng=np.random.default_rng(0), horizon=horizon)
        if st is not None and (best is None or st.cost < best.cost):
            best = st
    tries = 0
    while best is None and time.perf_counter() < deadline:
        tries += 1
        rule = "first_fit" if tries % 2 else "min_penalty"
        st = greedy(inst, "dsatur", rng=rng, rule=rule, horizon=horizon, random_ties=True)
        if st is not None:
            if rule == "first_fit":  # recompute exact cost of the complete timetable
                st2 = State(inst)
                for e in range(inst.n):
                    st2.place(e, int(st.assign[e]))
                st = st2
            best = st
    return best


def _scale(st: State, rng: np.random.Generator, samples: int = 300) -> float:
    inst = st.inst
    ds = []
    for _ in range(samples):
        e = int(rng.integers(inst.n))
        ts = st.feasible_slots(e)
        ts = ts[ts != st.assign[e]]
        if len(ts) == 0:
            continue
        t = np.array([int(rng.choice(ts))])
        ds.append(abs(int(st.delta_relocate(e, t)[0])))
    ds = [d for d in ds if d > 0]
    return float(np.mean(ds)) if ds else 1.0


def anneal(inst: Instance, start: State, budget: float, params: LSParams, rng: np.random.Generator,
           t_begin: float | None = None) -> LSResult:
    st = State(inst)
    for e in range(inst.n):
        st.place(e, int(start.assign[e]))
    assert st.cost == start.cost, "state cost mismatch when copying the start solution"
    scale = _scale(st, rng)
    t_start = time.perf_counter() if t_begin is None else t_begin
    best_cost, best_assign = st.cost, st.assign.copy()
    traj = [(0.0, best_cost)]
    n, T = inst.n, inst.T
    it = 0
    frac = 0.0
    temp = params.t0 * scale
    while True:
        if it % 256 == 0:
            frac = (time.perf_counter() - t_start) / budget
            if frac >= 1.0:
                break
            temp = max(params.t0 * (params.t1 / params.t0) ** frac * scale, 1e-9)
        it += 1
        e = int(rng.integers(n))
        a = int(st.assign[e])
        if rng.random() < params.p_swap:
            f = int(rng.integers(n))
            b = int(st.assign[f])
            if f == e or b == a:
                continue
            # feasibility of exchanging e (a->b) and f (b->a)
            shared = 1 if _adjacent(inst, e, f) else 0
            clash_e = st.conf[e, b] - shared
            clash_f = st.conf[f, a] - shared
            if clash_e > 0 or clash_f > 0:
                continue
            if st.load[b] - inst.sizes[f] + inst.sizes[e] > inst.C or st.load[a] - inst.sizes[e] + inst.sizes[f] > inst.C:
                continue
            before = st.cost
            st.move(e, b)  # temporary clash with f inside slot b is harmless for the exact cost
            st.move(f, a)
            d = st.cost - before
            if d <= 0 or rng.random() < np.exp(-d / temp):
                pass
            else:
                st.move(f, b)
                st.move(e, a)
        else:
            b = int(rng.integers(T))
            if b == a or st.conf[e, b] > 0 or st.load[b] + inst.sizes[e] > inst.C:
                continue
            d = int(st.delta_relocate(e, np.array([b]))[0])
            if d <= 0 or rng.random() < np.exp(-d / temp):
                st.remove(e)
                st.place(e, b, cost_delta=None)
        if st.cost < best_cost:
            best_cost, best_assign = st.cost, st.assign.copy()
            traj.append((time.perf_counter() - t_start, best_cost))
    return LSResult(assign=best_assign, cost=int(best_cost), iters=it, trajectory=traj)


def _adjacent(inst: Instance, e: int, f: int) -> bool:
    nb = inst.nb[e]
    return bool(len(nb)) and bool((nb == f).any())


def run_ls(inst: Instance, budget: float, params: LSParams, seed: int):
    """Construct a feasible start (counts against the budget) and anneal for the remaining time."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    start = construct_feasible(inst, rng, deadline=t0 + 0.3 * budget)
    if start is None:
        return None
    return anneal(inst, start, budget, params, rng, t_begin=t0)
