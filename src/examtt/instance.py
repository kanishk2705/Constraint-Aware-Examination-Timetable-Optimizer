"""Instance container and the single definition of the timeline and objective weights.

Timeline: ``n_days`` days with ``periods_per_day`` exam periods each. A *slot* ``t`` is
(day, period). Slots are mapped to *positions* on a line with ``day_gap`` unusable positions
between days, so that |position difference| ∈ {1, 2, 3} can only happen inside one day and
exams on different days are always at distance >= 4 (no penalty).

Students are stored as cohorts: ``{"exams": [...], "count": m}`` means ``m`` students who all
take exactly that set of exams. Every formula below uses the multiplicity ``count``.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np

GENERATOR_VERSION = "1.1"
PERIODS_PER_DAY = 4
DAY_GAP = 3
MAX_PER_DAY = 2
W_OVERLOAD = 8
PENALTY_BY_DISTANCE = (0, 5, 2, 1)  # index = position distance 0..3 ; distance >= 4 -> 0
DIST_CLIP = 4


def slot_position(t: int, periods_per_day: int = PERIODS_PER_DAY, day_gap: int = DAY_GAP) -> int:
    return (t // periods_per_day) * (periods_per_day + day_gap) + t % periods_per_day


class Instance:
    """Immutable view of an instance with derived arrays used by the solvers.

    The planted (generator-side) timetable is *not* loaded unless ``with_planted=True`` so that
    solvers cannot read it by accident (tuning/evaluation discipline, see PROTOCOL.md).
    """

    def __init__(self, d: dict, with_planted: bool = False):
        self.raw = {k: v for k, v in d.items() if k != "planted" or with_planted}
        self.name = d["name"]
        self.seed = d["seed"]
        self.params = d["params"]
        self.n = int(d["n_exams"])
        self.D = int(d["n_days"])
        self.P = int(d["periods_per_day"])
        self.G = int(d["day_gap"])
        self.T = int(d["n_slots"])
        self.C = int(d["capacity"])
        self.N = int(d["max_per_day"])
        self.W_OV = int(d["w_overload"])
        self.PT = np.array(list(d["penalty_by_distance"]) + [0] * (DIST_CLIP + 1 - len(d["penalty_by_distance"])), dtype=np.int64)
        self.planted = np.array(d["planted"], dtype=np.int64) if with_planted and "planted" in d else None

        self.pos_full = np.array([slot_position(t, self.P, self.G) for t in range(max(self.T, self.n))], dtype=np.int64)
        self.day_full = np.array([t // self.P for t in range(max(self.T, self.n))], dtype=np.int64)
        self.pos = self.pos_full[: self.T]
        self.day = self.day_full[: self.T]

        studs = d["students"]
        self.S = len(studs)
        self.stu_exams = [np.array(sorted(s["exams"]), dtype=np.int64) for s in studs]
        self.stu_mult = np.array([s["count"] for s in studs], dtype=np.int64)

        self.sizes = np.zeros(self.n, dtype=np.int64)
        pair_w: dict = {}
        for ex, m in zip(self.stu_exams, self.stu_mult):
            for e in ex:
                self.sizes[e] += m
            for i, j in combinations(ex.tolist(), 2):
                pair_w[(i, j)] = pair_w.get((i, j), 0) + int(m)
        items = sorted(pair_w.items())
        self.PI = np.array([k[0] for k, _ in items], dtype=np.int64)
        self.PJ = np.array([k[1] for k, _ in items], dtype=np.int64)
        self.PW = np.array([v for _, v in items], dtype=np.int64)
        self.n_edges = len(items)
        self.density = self.n_edges / (self.n * (self.n - 1) / 2) if self.n > 1 else 0.0

        nb = [[] for _ in range(self.n)]
        wt = [[] for _ in range(self.n)]
        for (i, j), w in items:
            nb[i].append(j); wt[i].append(w)
            nb[j].append(i); wt[j].append(w)
        self.nb = [np.array(x, dtype=np.int64) for x in nb]
        self.wt = [np.array(x, dtype=np.int64) for x in wt]
        self.deg = np.array([len(x) for x in nb], dtype=np.int64)

        # students that can exceed the per-day limit ("heavy"): compact index for count matrices
        heavy = [i for i, ex in enumerate(self.stu_exams) if len(ex) > self.N]
        self.heavy = np.array(heavy, dtype=np.int64)
        self.H = len(heavy)
        self.ov_mult = self.stu_mult[self.heavy] if self.H else np.zeros(0, dtype=np.int64)
        per_exam = [[] for _ in range(self.n)]
        for h, i in enumerate(heavy):
            for e in self.stu_exams[i]:
                per_exam[e].append(h)
        self.ov_idx = [np.array(x, dtype=np.int64) for x in per_exam]

        self.total_seats = int(self.sizes.sum())
        self.tightness = self.total_seats / (self.T * self.C)
        self.hash = hashlib.sha256(json.dumps({k: v for k, v in d.items() if k != "planted"}, sort_keys=True).encode()).hexdigest()

    # features used by the pre-registered decision rule and the regime map
    @property
    def n_binary_vars(self) -> int:
        return self.n * self.T


def load_instance(path: str, with_planted: bool = False) -> Instance:
    with open(path) as f:
        return Instance(json.load(f), with_planted=with_planted)
