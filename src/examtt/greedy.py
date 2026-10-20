"""Capacity- and horizon-aware greedy constructions.

Orders
  * ``random``  – a random permutation of the exams.
  * ``ldf``     – largest degree first (conflict-graph degree, ties by enrolment, then index).
  * ``dsatur``  – generalised DSATUR: repeatedly place the exam with the *fewest feasible slots*
                  (blocked by a clash or by capacity), ties by degree. With unlimited capacity this
                  is the classic saturation-degree rule.

Slot rules
  * ``min_penalty`` – feasible slot with the smallest increase of the soft cost (ties: lowest slot).
  * ``first_fit``   – lowest feasible slot (used for the slot-minimisation study).
A construction returns ``None`` when some exam has no feasible slot inside the horizon.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance
from .state import State

ORDERS = ("random", "ldf", "dsatur")


def greedy(inst: Instance, order: str, rng: np.random.Generator | None = None, rule: str = "min_penalty",
           horizon: int | None = None, random_ties: bool = False) -> State | None:
    n = inst.n
    H = inst.T if horizon is None else horizon
    st = State(inst, n_slots=H if horizon is not None else None)
    if order == "random":
        seq = list((rng or np.random.default_rng(0)).permutation(n))
    elif order == "ldf":
        seq = sorted(range(n), key=lambda e: (-inst.deg[e], -inst.sizes[e], e))
    elif order == "dsatur":
        seq = None
    else:
        raise ValueError(order)

    remaining = set(range(n))
    for step in range(n):
        if seq is not None:
            e = seq[step]
        else:
            best, best_key = None, None
            tie = (rng or np.random.default_rng(0)).random(n) if random_ties else None
            for x in remaining:
                nf = len(st.feasible_slots(x))
                key = (nf, -inst.deg[x], tie[x] if tie is not None else x)
                if best_key is None or key < best_key:
                    best, best_key = x, key
            e = best
        remaining.discard(e)
        ts = st.feasible_slots(e)
        if len(ts) == 0:
            return None
        if rule == "first_fit":
            t = int(ts[0])
            st.place(e, t, cost_delta=0)
        else:
            costs = st.insert_cost(e, ts)
            m = costs.min()
            cand = ts[costs == m]
            t = int(cand[0]) if not random_ties else int((rng or np.random.default_rng(0)).choice(cand))
            st.place(e, t, cost_delta=int(m))
    return st


def first_fit_slots_used(inst: Instance, st: State) -> int:
    return int(st.assign.max()) + 1
