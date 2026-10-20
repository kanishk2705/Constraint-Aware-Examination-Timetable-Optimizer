import math

from examtt import analysis as an


def _row(edges, ls, cp, scale=1000.0):
    best = min(ls, cp)
    return {"edges": edges, "excess": {"ls": 100 * (ls - best) / scale, "cpsat_cold": 100 * (cp - best) / scale},
            "regret": {"ls": an.regret_pct(ls, best), "cpsat_cold": an.regret_pct(cp, best)}}


def test_excess_is_finite_when_the_optimum_is_zero():
    v = {"B": {"ls": 40, "cpsat_cold": 0, "hybrid": float("nan")}, "scale": 2000.0}
    e = an.excess(v, 0)
    assert e["cpsat_cold"] == 0 and math.isclose(e["ls"], 2.0) and math.isnan(e["hybrid"])
    assert an.regret_pct(40, 0) == 4000.0  # the instability that motivated Deviation 1


def test_fit_threshold_picks_the_crossover():
    rows = [_row(e, ls=100 + 50 * (e < 100), cp=100 - 30 * (e < 100) + 80 * (e >= 100)) for e in (10, 20, 50, 90, 150, 300, 600)]
    fit = an.fit_threshold(rows, metric="excess")
    assert fit["theta"] == 90  # CP-SAT up to 90 edges, LS above
    assert fit["mean_regret"] < min(fit["always_ls"], fit["always_cp"])
