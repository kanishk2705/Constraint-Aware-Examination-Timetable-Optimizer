"""Instance generator with controlled difficulty and a planted feasible timetable.

Controlled knobs (all documented in DATA.md):
  * ``n_exams``   -> size tier (horizon = n_days x 4 periods, fixed by the tier)
  * ``density``   -> target density of the conflict graph (edges / all exam pairs)
  * ``tightness`` -> target seat utilisation  total_seats / (n_slots * capacity)

Construction: exams are split into ``n_slots`` equally sized groups (the *planted* timetable). Student
cohorts then each take k exams from k *different* planted slots, so the planted timetable has no
clashes by construction. Cohorts are added until the target density is reached. Slot choice is
biased toward the least loaded slots so that planted slot loads stay balanced, and capacity is set to
``ceil(L* / tightness)`` where L* is the (equalised) planted slot load, so the planted timetable
is capacity-feasible. Hence every instance is feasible; the heuristics are *not* told how.

Planted slot loads are equalised with single-exam cohorts (students who sit only that exam), so the achieved
tightness total_seats / (n_slots * capacity) equals the target up to rounding of the capacity.
"""
from __future__ import annotations

import numpy as np

from .instance import (DAY_GAP, GENERATOR_VERSION, MAX_PER_DAY, PENALTY_BY_DISTANCE,
                       PERIODS_PER_DAY, W_OVERLOAD)

SIZE_TIERS = {30: 3, 60: 5, 100: 7}  # n_exams -> n_days (horizon = n_days * 4 slots)
K_VALUES = (2, 3, 4, 5)
K_PROBS = (0.10, 0.35, 0.35, 0.20)
COHORT_MIN, COHORT_MAX = 5, 40


def generate(name: str, seed: int, n_exams: int, density: float, tightness: float,
             n_days: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    D = n_days if n_days is not None else SIZE_TIERS[n_exams]
    P = PERIODS_PER_DAY
    T = D * P

    # planted timetable: equal-size groups, random membership
    perm = rng.permutation(n_exams)
    planted = np.empty(n_exams, dtype=np.int64)
    for rank, e in enumerate(perm):
        planted[e] = rank % T
    slot_exams = [np.where(planted == t)[0] for t in range(T)]
    pop = rng.lognormal(mean=0.0, sigma=0.7, size=n_exams)
    slot_probs = [pop[ex] / pop[ex].sum() for ex in slot_exams]

    adj = np.zeros((n_exams, n_exams), dtype=bool)
    target_edges = density * n_exams * (n_exams - 1) / 2
    load = np.zeros(T)
    sizes = np.zeros(n_exams, dtype=np.int64)
    cohorts: list = []
    mean_m = (COHORT_MIN + COHORT_MAX) / 2
    max_cohorts = 20000
    while adj.sum() / 2 < target_edges and len(cohorts) < max_cohorts:
        k = int(rng.choice(K_VALUES, p=K_PROBS))
        k = min(k, T)
        w = np.exp(-(load - load.min()) / (2 * mean_m))
        slots = rng.choice(T, size=k, replace=False, p=w / w.sum())
        exams = [int(rng.choice(slot_exams[t], p=slot_probs[t])) for t in slots]
        m = int(rng.integers(COHORT_MIN, COHORT_MAX + 1))
        for t, e in zip(slots, exams):
            load[t] += m
            sizes[e] += m
        for a in range(k):
            for b in range(a + 1, k):
                adj[exams[a], exams[b]] = adj[exams[b], exams[a]] = True
        cohorts.append({"exams": sorted(exams), "count": m})
    # (a) every exam must have at least one student (single-exam cohorts add no conflicts)
    def planted_loads_now():
        pl = np.zeros(T)
        for e in range(n_exams):
            pl[planted[e]] += sizes[e]
        return pl

    for e in np.where(sizes == 0)[0]:
        m = int(rng.integers(COHORT_MIN, COHORT_MAX + 1))
        cohorts.append({"exams": [int(e)], "count": m})
        sizes[e] += m

    # (b) equalise the planted slot loads with single-exam cohorts ("students who only sit this one exam").
    # They add no conflicts, and make the planted timetable fill every slot to exactly the same level L*,
    # so that capacity tightness = L* / capacity is controlled exactly by ``tightness``.
    loads = planted_loads_now()
    l_star = int(loads.max())
    for t in range(T):
        deficit = int(l_star - loads[t])
        while deficit > 0:
            m = min(deficit, int(rng.integers(COHORT_MIN, COHORT_MAX + 1)))
            e = int(rng.choice(slot_exams[t], p=slot_probs[t]))
            cohorts.append({"exams": [e], "count": m})
            sizes[e] += m
            deficit -= m

    total = int(sizes.sum())
    capacity = int(np.ceil(l_star / tightness))  # planted slots hold L* <= capacity, i.e. at least a (1 - tightness) margin

    return {
        "name": name, "seed": int(seed), "generator_version": GENERATOR_VERSION,
        "params": {"n_exams": n_exams, "target_density": density, "target_tightness": tightness},
        "n_exams": n_exams, "n_days": D, "periods_per_day": P, "day_gap": DAY_GAP,
        "n_slots": T, "capacity": capacity, "max_per_day": MAX_PER_DAY,
        "w_overload": W_OVERLOAD, "penalty_by_distance": list(PENALTY_BY_DISTANCE),
        "students": cohorts,
        "planted": [int(x) for x in planted],
    }
