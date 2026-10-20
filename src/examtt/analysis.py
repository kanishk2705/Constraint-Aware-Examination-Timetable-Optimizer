"""Analysis of the raw experiment records (everything is recomputed from the stored timetables).

Every stored timetable is re-validated with the independent validator; the solver's claimed objective must equal
the validated one, otherwise ``AnalysisError`` is raised (an invalid run never enters a statistic).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .instance import Instance
from .stats import N_BOOT, BOOT_SEED, bootstrap_mean_ci, paired_diff_ci, regret_pct, wilson_ci, win_tie_loss
from .validator import slots_used, validate

B_METHODS = ["greedy_random", "greedy_ldf", "greedy_dsatur", "ls", "cpsat_cold", "hybrid"]
A_METHODS = ["ff_random", "ff_ldf", "ff_dsatur", "ms_dsatur", "cpsat_slots"]
TIERS = ["S", "M", "L"]
DENS = ["low", "mid", "high"]
TIGHT = ["loose", "medium", "tight"]
NAN = float("nan")


class AnalysisError(RuntimeError):
    pass


def load_records(path):
    return [json.loads(line) for line in open(path)]


def load_raw_instance(data_dir, name):
    return json.load(open(os.path.join(data_dir, name + ".json")))


def validate_records(records, data_dir):
    """Return per-instance validated costs / slot counts and a validation audit."""
    audit = {"timetables_checked": 0, "claimed_cost_mismatches": 0, "hard_violations": 0, "details": []}
    out = {}
    for r in records:
        raw = load_raw_instance(data_dir, r["name"])
        if Instance(raw).hash != r["sha256"]:
            raise AnalysisError(f"{r['name']}: instance file does not match the instance that was run (hash mismatch)")
        row = {"B": {}, "A": {}}
        # fixed per-instance normaliser: worst-case pair penalty if every shared-student pair were back-to-back
        pair_total = sum(st["count"] * (len(set(st["exams"])) * (len(set(st["exams"])) - 1) // 2) for st in raw["students"])
        row["scale"] = float(raw["penalty_by_distance"][1] * pair_total)
        for m in B_METHODS:
            v = r["B"].get(m)
            cost = NAN
            if v and v.get("assign"):
                rep = validate(raw, v["assign"])
                audit["timetables_checked"] += 1
                if not rep["ok"]:
                    audit["hard_violations"] += 1
                    audit["details"].append(f"{r['name']} {m}: {rep['violations'][:1]}")
                else:
                    cost = rep["objective"]["total"]
                    if v.get("cost") is not None and v["cost"] != cost:
                        audit["claimed_cost_mismatches"] += 1
                        audit["details"].append(f"{r['name']} {m}: claimed {v['cost']} validated {cost}")
            row["B"][m] = cost
        for m in A_METHODS:
            v = r["A"].get(m) or {}
            k = NAN
            if v.get("assign"):
                raw2 = dict(raw, n_slots=max(raw["n_slots"], max(v["assign"]) + 1))  # hard constraints are horizon independent
                rep = validate(raw2, v["assign"])
                audit["timetables_checked"] += 1
                if not rep["ok"]:
                    audit["hard_violations"] += 1
                    audit["details"].append(f"{r['name']} {m} (slots): {rep['violations'][:1]}")
                else:
                    k = slots_used(v["assign"])
                    if v.get("slots") is not None and v["slots"] != k:
                        audit["claimed_cost_mismatches"] += 1
                        audit["details"].append(f"{r['name']} {m}: claimed {v['slots']} slots, validated {k}")
            row["A"][m] = k
        out[r["name"]] = row
    if audit["claimed_cost_mismatches"]:
        raise AnalysisError(f"claimed objective differs from the independent validator: {audit['details'][:5]}")
    return out, audit


def regrets(validated_row):
    costs = validated_row["B"]
    finite = [c for c in costs.values() if not math.isnan(c)]
    if not finite:
        return {m: NAN for m in B_METHODS}, NAN
    best = min(finite)
    return {m: (NAN if math.isnan(c) else regret_pct(c, best)) for m, c in costs.items()}, best


def excess(validated_row, best):
    """Excess over the best-known cost, in % of the instance's fixed scale (no division by a zero optimum)."""
    sc = max(validated_row["scale"], 1.0)
    return {m: (NAN if math.isnan(c) or math.isnan(best) else 100.0 * (c - best) / sc) for m, c in validated_row["B"].items()}


def table(records, validated):
    """Flat per-instance table used by every statistic."""
    rows = []
    for r in records:
        v = validated[r["name"]]
        reg, best = regrets(v)
        exc = excess(v, best)
        p = r["params"]
        cp = r["B"].get("cpsat_cold", {})
        rows.append({
            "name": r["name"], "tier": p["size_tier"], "density": p["density_level"], "tight": p["tightness_level"],
            "edges": r["features"]["edges"], "n": r["features"]["n"], "T": r["features"]["T"],
            "binary_vars": r["features"]["binary_vars"], "achieved_density": r["features"]["density"],
            "achieved_tightness": r["features"]["tightness"],
            "cost": v["B"], "regret": reg, "excess": exc, "best": best, "scale": v["scale"],
            "cp_status": cp.get("status"), "cp_bound": cp.get("bound"),
            "hyb_status": r["B"].get("hybrid", {}).get("status"),
            "slots": v["A"], "A": r["A"], "raw": r,
        })
    return rows


def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


METRIC = "excess"  # primary metric after Deviation 1 (PROTOCOL.md section 13); "regret" is the sensitivity analysis


def _reg(rows, m, metric=None):
    return np.array([r[metric or METRIC][m] for r in rows], dtype=float)


def _feasible(rows, m):
    return [0 if math.isnan(r["cost"][m]) else 1 for r in rows]


def feasible_rate(rows, m):
    f = _feasible(rows, m)
    return wilson_ci(sum(f), len(f))


def proven_rate(rows):
    k = sum(1 for r in rows if r["cp_status"] == "OPTIMAL")
    return wilson_ci(k, len(rows))


def diff_rate_ci(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap CI of rate(a) - rate(b) for two independent groups of 0/1 outcomes."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return {"diff": None, "lo": None, "hi": None, "n_a": len(a), "n_b": len(b)}
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
    d = da - db
    return {"diff": float(a.mean() - b.mean()), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)),
            "n_a": len(a), "n_b": len(b)}


def fit_threshold(rows, metric=None):
    """PROTOCOL 8.2: choose theta minimising mean regret of 'CP-SAT if edges <= theta else LS' on instances where both exist."""
    metric = metric or METRIC
    ok = [r for r in rows if not math.isnan(r[metric]["ls"]) and not math.isnan(r[metric]["cpsat_cold"])]
    cands = [-1] + sorted({r["edges"] for r in ok})
    best = None
    table_ = []
    for th in cands:
        mean = float(np.mean([r[metric]["cpsat_cold"] if r["edges"] <= th else r[metric]["ls"] for r in ok])) if ok else NAN
        table_.append({"theta": th, "mean_regret": mean})
        if best is None or mean < best[1] - 1e-12:
            best = (th, mean)
    return {"theta": best[0], "mean_regret": best[1], "n_instances": len(ok),
            "metric": metric,
            "always_ls": float(np.mean([r[metric]["ls"] for r in ok])) if ok else None,
            "always_cp": float(np.mean([r[metric]["cpsat_cold"] for r in ok])) if ok else None,
            "candidates": table_}


def policy_regret(rows, theta, metric=None):
    out = []
    for r in rows:
        m = "cpsat_cold" if r["edges"] <= theta else "ls"
        out.append(r[metric or METRIC][m])
    return np.array(out, dtype=float)


def method_summary(rows):
    s = {}
    for m in B_METHODS:
        s[m] = {"feasible": feasible_rate(rows, m), "excess": bootstrap_mean_ci(_reg(rows, m, "excess")),
                "regret": bootstrap_mean_ci(_reg(rows, m, "regret"))}
    return s


def cell_table(rows):
    cells = []
    for t in TIERS:
        for d in DENS:
            for g in TIGHT:
                sub = _sel(rows, tier=t, density=d, tight=g)
                if not sub:
                    continue
                cell = {"tier": t, "density": d, "tight": g, "n": len(sub),
                        "proven_optimal": sum(1 for r in sub if r["cp_status"] == "OPTIMAL")}
                for m in B_METHODS:
                    reg = _reg(sub, m, "regret")
                    exc = _reg(sub, m, "excess")
                    ok = reg[~np.isnan(reg)]
                    cell[m] = {"feasible": int(len(ok)), "mean_regret": float(ok.mean()) if len(ok) else None,
                               "mean_excess": float(exc[~np.isnan(exc)].mean()) if len(ok) else None}
                cells.append(cell)
    return cells


def time_to_best(rows):
    """Median seconds until CP-SAT cold reached its final objective (from the improvement trajectory)."""
    out = {}
    for t in TIERS:
        ts = []
        for r in _sel(rows, tier=t):
            tr = r["raw"]["B"]["cpsat_cold"].get("trajectory") or []
            if tr:
                ts.append(tr[-1][0])
        out[t] = {"median_s": float(np.median(ts)) if ts else None, "n": len(ts)}
    return out


def study_a(rows):
    res = {"per_tier": {}, "cells": []}
    for t in TIERS + ["all"]:
        sub = rows if t == "all" else _sel(rows, tier=t)
        ok = [r for r in sub if not math.isnan(r["slots"]["cpsat_slots"]) and not math.isnan(r["slots"]["ff_dsatur"])]
        match = sum(1 for r in ok if r["slots"]["ff_dsatur"] <= r["slots"]["cpsat_slots"])
        cp_better = sum(1 for r in ok if r["slots"]["cpsat_slots"] < r["slots"]["ff_dsatur"])
        ms_ok = [r for r in sub if not math.isnan(r["slots"]["ms_dsatur"]) and not math.isnan(r["slots"]["cpsat_slots"])]
        ms_match = sum(1 for r in ms_ok if r["slots"]["ms_dsatur"] <= r["slots"]["cpsat_slots"])
        lb_hit = sum(1 for r in ok if r["slots"]["cpsat_slots"] <= r["A"]["lower_bounds"]["lb"])
        proven = sum(1 for r in sub if r["A"]["cpsat_slots"].get("status") == "OPTIMAL")
        saving = [r["slots"]["ff_dsatur"] - r["slots"]["cpsat_slots"] for r in ok]
        res["per_tier"][t] = {"n": len(ok), "dsatur_matches_cpsat": wilson_ci(match, len(ok)),
                              "cpsat_strictly_fewer": cp_better, "ms_matches_cpsat": wilson_ci(ms_match, len(ms_ok)),
                              "cpsat_at_lower_bound": wilson_ci(lb_hit, len(ok)), "cpsat_proven_optimal": proven,
                              "mean_slot_saving_vs_dsatur": bootstrap_mean_ci(saving)}
    for g in TIGHT:
        sub = _sel(rows, tight=g)
        ok = [r for r in sub if not math.isnan(r["slots"]["cpsat_slots"]) and not math.isnan(r["slots"]["ff_dsatur"])]
        res["cells"].append({"tight": g, "n": len(ok), "cp_strictly_fewer": sum(1 for r in ok if r["slots"]["cpsat_slots"] < r["slots"]["ff_dsatur"])})
    # exceptions listed explicitly (H6)
    res["exceptions"] = [{"name": r["name"], "dsatur_slots": r["slots"]["ff_dsatur"], "cpsat_slots": r["slots"]["cpsat_slots"]}
                         for r in rows if not math.isnan(r["slots"]["cpsat_slots"]) and not math.isnan(r["slots"]["ff_dsatur"])
                         and r["slots"]["cpsat_slots"] < r["slots"]["ff_dsatur"]]
    return res


def hypotheses(rows, theta, metric=None):
    h = {}
    # H1 proof frontier
    pr = {t: proven_rate(_sel(rows, tier=t)) for t in TIERS}
    h["H1"] = {"proven_rate": pr,
               "supported": bool(pr["S"]["lo"] is not None and pr["S"]["lo"] > 0.5 and pr["L"]["hi"] is not None and pr["L"]["hi"] < 0.25)}
    # H2 crossover
    d = {}
    for t in TIERS:
        sub = _sel(rows, tier=t)
        d[t] = paired_diff_ci(_reg(sub, "ls", metric), _reg(sub, "cpsat_cold", metric))
        d[t]["wtl_ls_vs_cp"] = win_tie_loss([r["cost"]["ls"] for r in sub], [r["cost"]["cpsat_cold"] for r in sub])
    h["H2"] = {"D_ls_minus_cp": d,
               "H2a_supported": bool(d["M"]["lo"] is not None and d["M"]["lo"] > 0),
               "H2b_supported": bool(d["L"]["hi"] is not None and d["L"]["hi"] < 0)}
    # H3 tightness vs greedy feasibility (tiers M, L)
    ml = [r for r in rows if r["tier"] in ("M", "L")]
    loose = _sel(ml, tight="loose")
    tight = _sel(ml, tight="tight")
    dr = diff_rate_ci(_feasible(tight, "greedy_random"), _feasible(loose, "greedy_random"))
    h["H3"] = {"greedy_random_feasible": {"loose": feasible_rate(loose, "greedy_random"), "tight": feasible_rate(tight, "greedy_random")},
               "diff_tight_minus_loose": dr,
               "ls_feasible_tight": feasible_rate(tight, "ls"), "cpsat_feasible_tight": feasible_rate(tight, "cpsat_cold"),
               "supported": bool(dr["hi"] is not None and dr["hi"] < 0 and feasible_rate(tight, "ls")["rate"] >= 0.9
                                 and feasible_rate(tight, "cpsat_cold")["rate"] >= 0.9)}
    # H4 warm start
    h4 = paired_diff_ci(_reg(ml, "hybrid", metric), _reg(ml, "cpsat_cold", metric))
    h["H4"] = {"hybrid_minus_cold": h4, "wtl_hybrid_vs_cold": win_tie_loss([r["cost"]["hybrid"] for r in ml], [r["cost"]["cpsat_cold"] for r in ml]),
               "supported": bool(h4["hi"] is not None and h4["hi"] < 0)}
    # H5 decision rule
    pol = policy_regret(rows, theta, metric)
    ls_r, cp_r = _reg(rows, "ls", metric), _reg(rows, "cpsat_cold", metric)
    vs_ls = paired_diff_ci(pol, ls_r)
    vs_cp = paired_diff_ci(pol, cp_r)
    h["H5"] = {"theta": theta, "rule_mean_regret": bootstrap_mean_ci(pol), "always_ls_mean_regret": bootstrap_mean_ci(ls_r),
               "always_cp_mean_regret": bootstrap_mean_ci(cp_r), "rule_minus_ls": vs_ls, "rule_minus_cp": vs_cp,
               "supported": bool(vs_ls["hi"] is not None and vs_ls["hi"] < 0 and vs_cp["hi"] is not None and vs_cp["hi"] < 0)}
    return h


def run_all(records, data_dir, theta):
    validated, audit = validate_records(records, data_dir)
    rows = table(records, validated)
    sa = study_a(rows)
    h = hypotheses(rows, theta, "excess")
    h_sens = hypotheses(rows, theta, "regret")  # pre-registered percentage regret, reported as a sensitivity analysis
    a = sa["per_tier"]["all"]
    h["H6"] = {"dsatur_matches_cpsat": a["dsatur_matches_cpsat"], "exceptions": sa["exceptions"],
               "supported": bool(a["dsatur_matches_cpsat"]["rate"] is not None and a["dsatur_matches_cpsat"]["rate"] >= 0.90)}
    out = {"n_instances": len(rows), "audit": {k: v for k, v in audit.items() if k != "details"}, "audit_details": audit["details"][:20],
           "overall": method_summary(rows),
           "by_tier": {t: method_summary(_sel(rows, tier=t)) for t in TIERS},
           "by_density": {d: method_summary(_sel(rows, density=d)) for d in DENS},
           "by_tightness": {g: method_summary(_sel(rows, tight=g)) for g in TIGHT},
           "proven_by_tier": {t: proven_rate(_sel(rows, tier=t)) for t in TIERS},
           "proven_by_density": {d: proven_rate(_sel(rows, density=d)) for d in DENS},
           "proven_by_tightness": {g: proven_rate(_sel(rows, tight=g)) for g in TIGHT},
           "time_to_best_cpsat": time_to_best(rows), "cells": cell_table(rows), "study_a": sa, "hypotheses": h,
           "hypotheses_sensitivity_regret": {k: h_sens[k] for k in ("H2", "H4", "H5")}, "primary_metric": "excess",
           "achieved": {"density_mean": float(np.mean([r["achieved_density"] for r in rows])),
                        "tightness_by_level": {g: float(np.mean([r["achieved_tightness"] for r in _sel(rows, tight=g)])) for g in TIGHT}}}
    return out, rows
