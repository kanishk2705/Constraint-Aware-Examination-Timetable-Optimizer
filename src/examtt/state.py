"""Incremental timetable state shared by the greedy constructions and the local search.

The state keeps the soft cost up to date through exact deltas. ``assign[e] == -1`` means unplaced.
"""
from __future__ import annotations

import numpy as np

from .instance import DIST_CLIP, Instance


class State:
    def __init__(self, inst: Instance, n_slots: int | None = None):
        self.inst = inst
        self.T = inst.T if n_slots is None else n_slots
        n = inst.n
        self.assign = -np.ones(n, dtype=np.int64)
        self.load = np.zeros(self.T, dtype=np.int64)
        self.conf = np.zeros((n, self.T), dtype=np.int64)
        self.cnt = np.zeros((max(inst.H, 1), inst.D if n_slots is None else int(inst.day_full[self.T - 1]) + 1), dtype=np.int64)
        self.cost = 0
        self.pos = inst.pos_full[: self.T]
        self.day = inst.day_full[: self.T]
        self.slots = np.arange(self.T)

    # ---- cost pieces -------------------------------------------------------------------
    def _edge_vec(self, e: int, ts: np.ndarray) -> np.ndarray:
        inst = self.inst
        nbr = inst.nb[e]
        if len(nbr) == 0:
            return np.zeros(len(ts), dtype=np.int64)
        a = self.assign[nbr]
        m = a >= 0
        if not m.any():
            return np.zeros(len(ts), dtype=np.int64)
        pj = self.pos[a[m]]
        w = inst.wt[e][m]
        d = np.minimum(np.abs(self.pos[ts][:, None] - pj[None, :]), DIST_CLIP)
        return (inst.PT[d] * w[None, :]).sum(axis=1)

    def _over_add_vec(self, e: int, ts: np.ndarray) -> np.ndarray:
        """Overload cost added if (currently unplaced / removed) e were put at each slot of ts."""
        inst = self.inst
        idx = inst.ov_idx[e]
        if len(idx) == 0:
            return np.zeros(len(ts), dtype=np.int64)
        c = self.cnt[idx[:, None], self.day[ts][None, :]]
        return inst.W_OV * (inst.ov_mult[idx][:, None] * (c >= inst.N)).sum(axis=0)

    def insert_cost(self, e: int, ts: np.ndarray) -> np.ndarray:
        return self._edge_vec(e, ts) + self._over_add_vec(e, ts)

    def remove_gain(self, e: int) -> int:
        """Cost currently attributable to e (what removing it would save)."""
        inst = self.inst
        t = self.assign[e]
        g = int(self._edge_vec(e, np.array([t]))[0])
        idx = inst.ov_idx[e]
        if len(idx):
            c = self.cnt[idx, self.day[t]]
            g += inst.W_OV * int((inst.ov_mult[idx] * (c > inst.N)).sum())
        return g

    def delta_relocate(self, e: int, ts: np.ndarray) -> np.ndarray:
        """Cost change if placed exam e moved to each slot in ts (ts must not contain its slot)."""
        inst = self.inst
        a = self.assign[e]
        edge = self._edge_vec(e, ts) - self._edge_vec(e, np.array([a]))[0]
        idx = inst.ov_idx[e]
        if len(idx) == 0:
            return edge
        da = self.day[a]
        dts = self.day[ts]
        c_new = self.cnt[idx[:, None], dts[None, :]]
        c_old = self.cnt[idx, da]
        m = inst.ov_mult[idx]
        up = (m[:, None] * (c_new >= inst.N)).sum(axis=0)
        down = int((m * (c_old > inst.N)).sum())
        over = inst.W_OV * (up - down) * (dts != da)
        return edge + over

    # ---- mutation ----------------------------------------------------------------------
    def feasible_slots(self, e: int, horizon: int | None = None) -> np.ndarray:
        inst = self.inst
        ok = (self.conf[e] == 0) & (self.load + inst.sizes[e] <= inst.C)
        if horizon is not None and horizon < self.T:
            ok = ok.copy(); ok[horizon:] = False
        return np.nonzero(ok)[0]

    def place(self, e: int, t: int, cost_delta: int | None = None) -> None:
        inst = self.inst
        if cost_delta is None:
            cost_delta = int(self.insert_cost(e, np.array([t]))[0])
        self.cost += cost_delta
        self.assign[e] = t
        self.load[t] += inst.sizes[e]
        nbr = inst.nb[e]
        if len(nbr):
            self.conf[nbr, t] += 1
        idx = inst.ov_idx[e]
        if len(idx):
            self.cnt[idx, self.day[t]] += 1

    def remove(self, e: int) -> int:
        inst = self.inst
        t = self.assign[e]
        gain = self.remove_gain(e)
        self.cost -= gain
        self.assign[e] = -1
        self.load[t] -= inst.sizes[e]
        nbr = inst.nb[e]
        if len(nbr):
            self.conf[nbr, t] -= 1
        idx = inst.ov_idx[e]
        if len(idx):
            self.cnt[idx, self.day[t]] -= 1
        return gain

    def move(self, e: int, t: int) -> int:
        """Relocate e to slot t; returns the exact cost delta (also applied to ``cost``)."""
        before = self.cost
        self.remove(e)
        self.place(e, t)
        return self.cost - before

    def is_complete(self) -> bool:
        return bool((self.assign >= 0).all())
