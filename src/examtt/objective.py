"""Vectorised objective used by the solvers (the validator has its own independent implementation)."""
from __future__ import annotations

import numpy as np

from .instance import DIST_CLIP, Instance


def evaluate(inst: Instance, assign) -> dict:
    """Soft-objective breakdown of a complete timetable (assign[e] = slot)."""
    a = np.asarray(assign, dtype=np.int64)
    d = np.minimum(np.abs(inst.pos[a[inst.PI]] - inst.pos[a[inst.PJ]]), DIST_CLIP)
    pairs = inst.PW * inst.PT[d]
    b2b = int(pairs[d == 1].sum())
    gap2 = int(pairs[d == 2].sum())
    gap3 = int(pairs[d == 3].sum())
    overload = 0
    if inst.H:
        cnt = np.zeros((inst.H, inst.D), dtype=np.int64)
        days = inst.day[a]
        for h, i in enumerate(inst.heavy):
            np.add.at(cnt[h], days[inst.stu_exams[i]], 1)
        overload = int((inst.ov_mult[:, None] * np.maximum(cnt - inst.N, 0)).sum()) * inst.W_OV
    return {"total": b2b + gap2 + gap3 + overload, "b2b": b2b, "gap2": gap2, "gap3": gap3, "overload": overload}
