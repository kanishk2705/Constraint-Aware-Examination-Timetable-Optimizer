"""Small statistics helpers: paired bootstrap CIs, Wilson intervals, regret, win/tie/loss counts."""
from __future__ import annotations

import math

import numpy as np

N_BOOT = 10_000
BOOT_SEED = 20261201


def bootstrap_mean_ci(values, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = 0.95):
    """Percentile bootstrap CI of the mean over instances (resampling instances with replacement)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    if len(v) == 1:
        return {"n": 1, "mean": float(v[0]), "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return {"n": int(len(v)), "mean": float(v.mean()), "lo": float(np.quantile(means, a)), "hi": float(np.quantile(means, 1 - a))}


def paired_diff_ci(a, b, **kw):
    """CI of mean(a - b) over paired instances; NaN pairs are dropped."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_mean_ci(a[keep] - b[keep], **kw)


def wilson_ci(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return {"n": 0, "rate": None, "lo": None, "hi": None}
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"n": n, "k": k, "rate": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}


def regret_pct(value, best):
    """Percentage above the best-known value (0 if equal). best == 0 is handled by a floor of 1."""
    return 100.0 * (value - best) / max(best, 1.0)


def win_tie_loss(a, b, tol: float = 0.0):
    """Counts where a is better (smaller), equal, or worse than b among instances where both exist."""
    w = t = l = 0
    for x, y in zip(a, b):
        if x is None or y is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(y, float) and math.isnan(y)):
            continue
        if x < y - tol:
            w += 1
        elif x > y + tol:
            l += 1
        else:
            t += 1
    return {"win": w, "tie": t, "loss": l}
