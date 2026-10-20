"""Study A helpers: minimise the NUMBER OF SLOTS (hard constraints only, no day structure)."""
from __future__ import annotations

import math
import time

import numpy as np

from .greedy import greedy
from .instance import Instance


def lower_bounds(inst: Instance) -> dict:
    clique = max((len(e) for e in inst.stu_exams), default=1)  # a student's exams are pairwise in conflict
    capacity = math.ceil(inst.total_seats / inst.C)
    return {"clique": int(clique), "capacity": int(capacity), "lb": int(max(clique, capacity))}


def first_fit_slots(inst: Instance, order: str, seed: int = 0):
    st = greedy(inst, order, rng=np.random.default_rng(seed), rule="first_fit", horizon=inst.n)
    if st is None:
        return None, None
    return st.assign.copy(), int(st.assign.max()) + 1


def multistart_slots(inst: Instance, budget: float, seed: int) -> dict:
    """Randomised-tie DSATUR with first-fit slots, restarted until the wall-clock budget is used."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_a, best_k, runs = None, None, 0
    traj = []
    while time.perf_counter() - t0 < budget:
        st = greedy(inst, "dsatur", rng=rng, rule="first_fit", horizon=inst.n, random_ties=True)
        runs += 1
        if st is None:
            continue
        k = int(st.assign.max()) + 1
        if best_k is None or k < best_k:
            best_a, best_k = st.assign.copy(), k
            traj.append((time.perf_counter() - t0, k))
    return {"assign": best_a, "slots": best_k, "runs": runs, "trajectory": traj}
