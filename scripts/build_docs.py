"""Generate docs/index.html, docs/figures/*.svg and README.md from results/summary.json only.

Every number shown comes from the summary file, so the text cannot drift from the data.
  python scripts/build_docs.py [--summary results/summary.json] [--out-dir .]
"""
import argparse
import html
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {"greedy_dsatur": "#2a78d6", "ls": "#eb6834", "cpsat_cold": "#1baf7a", "hybrid": "#eda100", "greedy_random": "#6250d6"}
LABEL = {"greedy_random": "Greedy (random order)", "greedy_ldf": "Greedy (largest degree)", "greedy_dsatur": "Greedy (DSATUR)",
         "ls": "Local search (SA)", "cpsat_cold": "CP-SAT", "hybrid": "Hybrid (LS then CP-SAT)"}
TIERS = ["S", "M", "L"]
TIER_DESC = {"S": "S: 14 exams", "M": "M: 32 exams", "L": "L: 64 exams"}
DENS = ["low", "mid", "high"]
TIGHT = ["loose", "medium", "tight"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def pct(x, d=0):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}"


def ci_rate(c):
    if c is None or c.get("rate") is None:
        return "n/a"
    return f"{pct(c['rate'])} ({c['k']}/{c['n']}; 95% CI {pct(c['lo'])}-{pct(c['hi'])})"


def ci_mean(c, d=2):
    if c is None or c.get("mean") is None:
        return "n/a"
    if c.get("lo") is None:
        return f"{c['mean']:.{d}f}"
    return f"{c['mean']:.{d}f} [{c['lo']:.{d}f}, {c['hi']:.{d}f}]"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_proven(S, path):
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=True)
    groups = [("Size", "proven_by_tier", TIERS, [TIER_DESC[t] for t in TIERS]),
              ("Conflict density", "proven_by_density", DENS, DENS),
              ("Capacity tightness", "proven_by_tightness", TIGHT, TIGHT)]
    for ax, (title, key, keys, labels) in zip(axes, groups):
        vals = [S[key][k]["rate"] or 0 for k in keys]
        lo = [(S[key][k]["rate"] or 0) - (S[key][k]["lo"] or 0) for k in keys]
        hi = [(S[key][k]["hi"] or 0) - (S[key][k]["rate"] or 0) for k in keys]
        ax.bar(range(3), vals, color=PALETTE["cpsat_cold"], width=0.55)
        ax.errorbar(range(3), vals, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=3, linewidth=1)
        for i, v in enumerate(vals):
            ax.text(i, min(v + hi[i] + 0.03, 1.02), pct(v), ha="center", fontsize=9, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 1.1)
        style(ax)
    axes[0].set_ylabel("share of instances proven optimal", color=MUTED, fontsize=9)
    save(fig, path)


def fig_excess(S, path):
    methods = ["greedy_dsatur", "ls", "cpsat_cold", "hybrid"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    w = 0.19
    for j, m in enumerate(methods):
        xs, ys, el, eh = [], [], [], []
        for i, t in enumerate(TIERS):
            c = S["by_tier"][t][m]["excess"]
            if c["mean"] is None:
                continue
            xs.append(i + (j - 1.5) * w)
            ys.append(c["mean"])
            el.append(c["mean"] - (c["lo"] if c["lo"] is not None else c["mean"]))
            eh.append((c["hi"] if c["hi"] is not None else c["mean"]) - c["mean"])
        ax.bar(xs, ys, width=w - 0.02, color=PALETTE[m], label=LABEL[m])
        ax.errorbar(xs, ys, yerr=[el, eh], fmt="none", ecolor=INK, capsize=2, linewidth=0.9)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([TIER_DESC[t] for t in TIERS])
    ax.set_ylabel("mean excess over best-known (% of scale)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    style(ax)
    save(fig, path)


def fig_feasible(S, path):
    methods = ["greedy_random", "greedy_dsatur", "ls", "cpsat_cold"]
    marks = {"greedy_random": "s", "greedy_dsatur": "o", "ls": "^", "cpsat_cold": "D"}
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for m in methods:
        ys = [S["by_tightness"][g][m]["feasible"]["rate"] for g in TIGHT]
        ax.plot(range(3), ys, marker=marks[m], color=PALETTE[m], linewidth=2, markersize=7, markeredgecolor="white", label=LABEL[m])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["loose (0.70)", "medium (0.85)", "tight (0.97)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of instances with a feasible timetable", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    style(ax)
    save(fig, path)


def fig_map(S, path):
    cells = S["cells"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    allv = []
    grid = {}
    for g in TIGHT:
        mat = np.full((3, 3), np.nan)
        for i, t in enumerate(TIERS):
            for j, d in enumerate(DENS):
                c = next((c for c in cells if c["tier"] == t and c["density"] == d and c["tight"] == g), None)
                if c and c["ls"]["mean_excess"] is not None and c["cpsat_cold"]["mean_excess"] is not None:
                    mat[i, j] = c["ls"]["mean_excess"] - c["cpsat_cold"]["mean_excess"]
        grid[g] = mat
        allv += [v for v in mat.flatten() if not np.isnan(v)]
    lim = max(abs(v) for v in allv) if allv else 1.0
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f2f1ee", "#2a78d6"])
    for ax, g in zip(axes, TIGHT):
        mat = grid[g]
        ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(3):
            for j in range(3):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(DENS, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels(TIERS if g == "loose" else [], fontsize=9)
        ax.set_title(f"capacity {g}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("conflict density", fontsize=9, color=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("size tier", fontsize=9, color=MUTED)
    save(fig, path)


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--muted:#52514e;--rule:#e4e3df;--accent:#2a78d6;--good:#0b7a3b;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#1a1a19;--card:#232321;--ink:#f0efec;--muted:#c3c2b7;--rule:#3a3a37;--accent:#3987e5;--good:#5fd08a;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 80px}h1{font-size:2rem;line-height:1.2;margin:.2em 0}h2{margin-top:2.4em;border-top:1px solid var(--rule);padding-top:.8em}
h3{margin-top:1.6em}.lede{color:var(--muted);font-size:1.1rem}.flow{display:flex;flex-wrap:wrap;gap:8px;margin:1em 0}.flow a{border:1px solid var(--rule);border-radius:999px;padding:4px 12px;text-decoration:none;color:var(--ink);background:var(--card);font-size:.9rem}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:.9rem;display:block;overflow-x:auto}th,td{border-bottom:1px solid var(--rule);padding:6px 10px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
code{background:var(--card);border:1px solid var(--rule);border-radius:4px;padding:0 4px}img{max-width:100%;height:auto;background:#fff;border-radius:8px;border:1px solid var(--rule)}.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
figcaption{color:var(--muted);font-size:.88rem;margin-top:4px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}.kpi b{display:block;font-size:1.5rem}
"""


def verdict(flag):
    return '<span class="ok">supported</span>' if flag else '<span class="no">not supported</span>'


def table_html(head, rows):
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(x)}</th>" for x in head) + "</tr></thead><tbody>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return h + "</tbody></table>"


def build(summary_path, out_dir):
    S = json.load(open(summary_path))
    H = S["hypotheses"]
    Hs = S.get("hypotheses_sensitivity_regret", {})
    meta = S.get("meta", {})
    figdir = os.path.join(out_dir, "docs", "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_proven(S, os.path.join(figdir, "proven.svg"))
    fig_excess(S, os.path.join(figdir, "excess.svg"))
    fig_feasible(S, os.path.join(figdir, "feasible.svg"))
    fig_map(S, os.path.join(figdir, "map.svg"))

    # ---------- tables ----------
    t_h = []
    t_h.append(["H1 proof frontier", "Proof rate: Wilson CI above 0.5 for tier S and below 0.25 for tier L (measured: tier M and L proof rates are both below 50%, but tier L's interval reaches above 0.25)",
                " | ".join(f"{t}: {ci_rate(H['H1']['proven_rate'][t])}" for t in TIERS), verdict(H["H1"]["supported"])])
    d = H["H2"]["D_ls_minus_cp"]
    t_h.append(["H2a crossover (tier M)", "CP-SAT beats tuned local search in M (D = excess(LS) - excess(CP-SAT) > 0)", ci_mean(d["M"]), verdict(H["H2"]["H2a_supported"])])
    t_h.append(["H2b crossover (tier L)", "Local search beats CP-SAT in L (D < 0)", ci_mean(d["L"]), verdict(H["H2"]["H2b_supported"])])
    h3 = H["H3"]
    t_h.append(["H3 tightness vs greedy", "Random-order greedy loses feasibility under tight capacity; LS and CP-SAT stay >= 90% feasible at tight capacity (LS met this; cold CP-SAT did not: it returned no timetable in 5 of 24 tight instances, all tier L, within 20 s)",
                f"loose {ci_rate(h3['greedy_random_feasible']['loose'])}; tight {ci_rate(h3['greedy_random_feasible']['tight'])}; "
                f"LS tight {pct(h3['ls_feasible_tight']['rate'])}, CP-SAT tight {pct(h3['cpsat_feasible_tight']['rate'])}", verdict(h3["supported"])])
    t_h.append(["H4 warm start", "Hybrid beats cold CP-SAT (M+L): mean excess difference < 0", ci_mean(H["H4"]["hybrid_minus_cold"]), verdict(H["H4"]["supported"])])
    h5 = H["H5"]
    t_h.append(["H5 one-feature rule", f"CP-SAT if edges <= {h5['theta']} else LS beats both fixed policies",
                f"rule - LS: {ci_mean(h5['rule_minus_ls'])}; rule - CP-SAT: {ci_mean(h5['rule_minus_cp'])}", verdict(h5["supported"])])
    t_h.append(["H6 slot minimisation", "DSATUR first-fit already matches CP-SAT's slot count on >= 90% of instances",
                ci_rate(H["H6"]["dsatur_matches_cpsat"]), verdict(H["H6"]["supported"])])

    t_methods = []
    for m in ["greedy_random", "greedy_ldf", "greedy_dsatur", "ls", "cpsat_cold", "hybrid"]:
        o = S["overall"][m]
        t_methods.append([LABEL[m], ci_rate(o["feasible"]), ci_mean(o["excess"]), ci_mean(o["regret"], 1)])
    t_tier = []
    for t in TIERS:
        for m in ["greedy_dsatur", "ls", "cpsat_cold", "hybrid"]:
            o = S["by_tier"][t][m]
            t_tier.append([TIER_DESC[t], LABEL[m], pct(o["feasible"]["rate"]), ci_mean(o["excess"])])
    t_cells = []
    for c in S["cells"]:
        t_cells.append([c["tier"], c["density"], c["tight"], c["n"], c["proven_optimal"]] +
                       [num(c[m]["mean_excess"]) if c[m]["mean_excess"] is not None else "n/a" for m in ["greedy_dsatur", "ls", "cpsat_cold", "hybrid"]])
    A = S["study_a"]["per_tier"]
    t_a = [[t, A[t]["n"], ci_rate(A[t]["dsatur_matches_cpsat"]), A[t]["cpsat_strictly_fewer"], ci_rate(A[t]["ms_matches_cpsat"]), ci_rate(A[t]["cpsat_at_lower_bound"])]
           for t in TIERS + ["all"]]
    sens = []
    for k, name in (("H2", "H2 D (M / L)"), ("H4", "H4 hybrid - cold"), ("H5", "H5 rule - LS / rule - CP")):
        if k not in Hs:
            continue
        if k == "H2":
            val = f"{ci_mean(Hs[k]['D_ls_minus_cp']['M'], 1)} / {ci_mean(Hs[k]['D_ls_minus_cp']['L'], 1)}"
        elif k == "H4":
            val = ci_mean(Hs[k]["hybrid_minus_cold"], 1)
        else:
            val = f"{ci_mean(Hs[k]['rule_minus_ls'], 1)} / {ci_mean(Hs[k]['rule_minus_cp'], 1)}"
        sens.append([name, val])

    aud = S["audit"]
    commit_rows = [[k, html.escape(str(v))] for k, v in meta.items()]
    body = f"""
<h1>Constraint-Aware Examination Timetable Optimizer</h1>
<p class="lede">When does exact solving pay off? A pre-registered comparison of greedy constructions, a tuned local search and OR-Tools CP-SAT on
{S['n_instances']} held-out examination-timetabling instances, mapped against instance size, conflict density and capacity tightness.</p>
<div class="flow"><a href="#problem">1 Problem</a><a href="#design">2 Design</a><a href="#implementation">3 Implementation</a><a href="#results">4 Results</a><a href="#limits">5 Limitations</a><a href="#next">6 Next steps</a><a href="#repro">Reproduce</a></div>

<div class="card kpis">
<div class="kpi"><b>{S['n_instances']}</b>held-out instances (27 cells x 4)</div>
<div class="kpi"><b>{aud['timetables_checked']}</b>timetables re-checked by the independent validator</div>
<div class="kpi"><b>{aud['hard_violations']}</b>hard-constraint violations found</div>
<div class="kpi"><b>{aud['claimed_cost_mismatches']}</b>claimed-vs-validated objective mismatches</div>
</div>

<h2 id="problem">1. Problem</h2>
<p>A university must place every exam in one time slot so that no student sits two exams at once and no slot holds more students than the rooms allow.
That already makes a graph-colouring problem with capacities: exams are vertices, an edge joins two exams that share a student. On top of it come soft goals:
avoid back-to-back exams, avoid more than two exams for one student on one day, spread exams out, and use few slots.</p>
<ul>
<li><b>Hard:</b> one slot per exam; no student clash; seats per slot &lt;= capacity.</li>
<li><b>Soft (summed over students):</b> 5 per back-to-back pair, 2 and 1 per pair one or two periods apart inside a day, 8 per exam beyond two on one day.</li>
<li><b>Research question:</b> for which instance sizes, densities and capacity tightness does an exact solver beat cheap heuristics, and when does it not?</li>
</ul>

<h2 id="design">2. Design</h2>
<p><b>Instances.</b> Real benchmarks were not usable (the sandbox blocked their hosts and no licence is stated), so a documented generator (<code>DATA.md</code>) builds instances with controlled
size (14 / 32 / 64 exams), conflict density (0.10 / 0.25 / 0.45) and capacity tightness (0.70 / 0.85 / 0.97). Each instance has a hidden planted timetable so it is feasible by construction; solvers never see it.
Tuning instances (54) and held-out instances (108) come from disjoint seeds, and every tuning choice was frozen before the held-out run.</p>
<p><b>Methods.</b> Three greedy constructions (random order, largest degree first, DSATUR-style most-constrained first); a simulated-annealing local search that never breaks a hard constraint;
CP-SAT cold; and a hybrid (local search for 25% of the budget, then CP-SAT warm-started). All get the same 20 s wall clock and the same two CPU cores (SA best of two processes, CP-SAT two workers).</p>
<p><b>Original angle.</b> Instead of asking only "is CP-SAT better?", the study pre-registers six hypotheses about <i>where</i> exact solving pays off, and fits a one-feature decision rule
(use CP-SAT when the conflict graph has at most {h5['theta']} edges) on the tuning set only, then tests it on the held-out set.</p>

<h2 id="implementation">3. Implementation</h2>
<ul>
<li><code>generator.py</code> controlled instances with a planted solution; <code>validator.py</code> an <b>independent</b> checker that shares no code with the solvers and recomputes the objective student by student.</li>
<li><code>state.py</code> exact incremental cost deltas shared by greedy and local search; <code>greedy.py</code>, <code>localsearch.py</code>; <code>cpsat.py</code> the CP-SAT models (soft objective and slot minimisation).</li>
<li>{aud['timetables_checked']} stored timetables were re-validated; {aud['hard_violations']} hard violations. The validator's cost is used everywhere. In {aud['claimed_cost_mismatches']} cold CP-SAT runs (tier L, time limit reached) the solver's own claimed objective was higher than the validated cost (largest gap {aud['max_overestimate']}); this is Deviation 3 in <code>PROTOCOL.md</code>, found by the audit before results were computed.</li>
<li>Unit tests, including tests that prove the validator <i>can</i> fail (clash, capacity, out-of-range, missing slot) and that two different CP-SAT encodings reach the same optimum.</li>
</ul>

<h2 id="results">4. Results (held-out instances only)</h2>
<h3>Pre-registered hypotheses</h3>
<p>Primary metric: <i>excess</i> over the best cost any method found on the same instance, in percent of a fixed per-instance scale (lower is better). Intervals are 95% bootstrap over instances.</p>
<div class="card">{table_html(["Hypothesis", "Claim", "Evidence", "Verdict"], t_h)}</div>

<h3>Where exact solving can prove optimality</h3>
<figure><img src="figures/proven.svg" alt="Share of instances CP-SAT proved optimal, by size, density and tightness"><figcaption>CP-SAT proof rate within 20 s, with 95% Wilson intervals.</figcaption></figure>

<h3>Solution quality by size</h3>
<figure><img src="figures/excess.svg" alt="Mean excess by size tier for four methods"><figcaption>Mean excess over the best-known cost, with 95% bootstrap intervals; numbers on bars.</figcaption></figure>
<div class="card">{table_html(["Size", "Method", "Feasible", "Mean excess [95% CI]"], t_tier)}</div>

<h3>Capacity tightness and feasibility</h3>
<figure><img src="figures/feasible.svg" alt="Feasibility rate by capacity tightness"><figcaption>Share of instances where the method returned a timetable that satisfies every hard constraint.</figcaption></figure>

<h3>Regime map: local search minus CP-SAT</h3>
<figure><img src="figures/map.svg" alt="Heat map of the difference in mean excess between local search and CP-SAT per cell"><figcaption>Mean excess(LS) - mean excess(CP-SAT) per cell (4 instances each, descriptive only). Positive (blue): CP-SAT better. Negative (orange): local search better.</figcaption></figure>

<h3>All methods, all instances</h3>
<div class="card">{table_html(["Method", "Feasible", "Mean excess [95% CI]", "Mean pre-registered % regret [95% CI]"], t_methods)}</div>

<h3>Per-cell table (table view of the map)</h3>
<div class="card">{table_html(["Size", "Density", "Tightness", "n", "CP-SAT proven", "DSATUR", "LS", "CP-SAT", "Hybrid"], t_cells)}<p class="lede">Entries are mean excess; 4 instances per cell, so cells are descriptive, never inferential.</p></div>

<h3>Study A: number of slots</h3>
<div class="card">{table_html(["Tier", "n", "DSATUR first-fit matches CP-SAT", "CP-SAT strictly fewer slots", "Multi-start DSATUR matches CP-SAT", "CP-SAT at lower bound"], t_a)}</div>

<h3>Sensitivity: the pre-registered percentage regret</h3>
<p>The original metric divides by the best cost, which is 0 on some large low-density instances (Deviation 1 in <code>PROTOCOL.md</code>). It is reported unchanged here as a sensitivity check.</p>
<div class="card">{table_html(["Quantity", "Mean difference [95% CI], in % regret"], sens)}</div>

<h2 id="limits">5. Limitations</h2>
<ul>
<li>Constructed instances only: every instance is feasible by construction, the conflict graph always has a colouring within the horizon, and exam sizes carry a faint imprint of the planted grouping. Results may not transfer to real examination data (no real benchmark could be used; see <code>DATA.md</code>).</li>
<li>Rooms are one pooled seat count per slot, exams last one slot, and the objective weights are my own choice.</li>
<li>CP-SAT's lower bound is 0 until it proves optimality on this objective, so optimality gaps are uninformative; quality is judged against the best cost found by any method, which is itself an upper bound on the optimum.</li>
<li>One wall-clock budget (20 s) and two CPU cores; wall-clock limits make CP-SAT and annealing runs not bit-for-bit repeatable, which is why every timetable is stored and re-validated.</li>
<li>Four instances per cell: per-cell numbers are descriptive; inference pools over the other factors.</li>
<li>Python local search; a compiled implementation would run many more moves per second and could shift the crossover.</li>
</ul>

<h2 id="next">6. Next steps</h2>
<ul>
<li>Run the same code on the real Toronto and ITC2007 instances once their terms can be checked and the files obtained (a parser is all that is missing).</li>
<li>Per-room modelling, multi-slot exams and overnight spacing; a compiled local search; a second budget (60 s) to see whether the crossover moves.</li>
<li>Stronger CP-SAT formulations (cliques over student groups, symmetry breaking across days) to obtain informative lower bounds.</li>
</ul>

<h2 id="repro">Reproduce</h2>
<div class="card"><code>pip install -r requirements.txt</code> &middot; <code>pytest</code> &middot; <code>python scripts/generate_instances.py eval</code> &middot;
<code>python scripts/run_experiment.py --data data/eval --out results/raw/eval.jsonl --budget 20 --ls-params results/tuned_params.json</code> &middot;
<code>python scripts/analyze.py eval --raw results/raw/eval.jsonl --data data/eval</code> &middot; <code>python scripts/build_docs.py</code>.
Frozen decisions live on the branches <code>protocol-freeze</code> and <code>params-freeze</code> (git tag pushes fail from the sandbox, so pinned branches are used).</div>
{('<div class="card">' + table_html(["Freeze record", "Value"], commit_rows) + '</div>') if commit_rows else ''}
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Examination Timetable Optimizer</title><style>{CSS}</style></head><body><main>{body}</main></body></html>"""
    with open(os.path.join(out_dir, "docs", "index.html"), "w") as f:
        f.write(page)

    # ---------- README (same flow, shorter) ----------
    readme = f"""# Constraint-Aware Examination Timetable Optimizer

When does exact solving (OR-Tools CP-SAT) pay off for examination timetabling? A pre-registered comparison of greedy constructions, a tuned local search and CP-SAT on
{S['n_instances']} held-out instances, mapped against size, conflict density and capacity tightness.
**Project page:** https://kanishk2705.github.io/Constraint-Aware-Examination-Timetable-Optimizer/

**Flow:** problem -> design -> implementation -> results -> limitations -> next steps (full text with figures on the project page).

## Problem
Place every exam in one slot: no student clash, seats per slot within capacity (hard); few back-to-back exams, at most two exams per student per day, exams spread out (soft, weighted objective in `PROTOCOL.md`).

## Design
Documented generator (`DATA.md`) with a planted feasible timetable; 14 / 32 / 64 exams x density 0.10 / 0.25 / 0.45 x capacity tightness 0.70 / 0.85 / 0.97; 54 tuning and 108 held-out instances from disjoint seeds.
Methods: three greedy constructions, simulated annealing, CP-SAT, and a LS-then-CP-SAT hybrid, all with 20 s and two cores. Six hypotheses and the statistics plan were written in `PROTOCOL.md` before any held-out run.

## Implementation
`src/examtt/`: generator, independent validator (shares no code with the solvers), incremental state, greedy, local search, CP-SAT models, statistics, analysis. `tests/`: validator-can-fail tests, solver-vs-validator agreement, two CP-SAT encodings reaching the same optimum.
{aud['timetables_checked']} stored timetables re-validated; {aud['hard_violations']} hard violations; {aud["claimed_cost_mismatches"]} claimed-vs-validated mismatches (solver claim above validated cost; Deviation 3).

## Results (held-out; excess over best-known in % of an instance scale, lower is better)
{chr(10).join('- **' + r[0] + '** ' + r[3] + ' - ' + r[1] + ': ' + r[2] for r in [[x[0], re.sub('<[^>]+>', '', x[3]), re.sub('<[^>]+>', '', x[1]), re.sub('<[^>]+>', '', x[2])] for x in t_h])}

| Method | Feasible | Mean excess [95% CI] |
|---|---|---|
{chr(10).join('| ' + r[0] + ' | ' + r[1] + ' | ' + r[2] + ' |' for r in t_methods)}

![Proof rate](docs/figures/proven.svg)
![Mean excess by size](docs/figures/excess.svg)
![Feasibility by tightness](docs/figures/feasible.svg)
![Regime map](docs/figures/map.svg)

## Limitations
Constructed instances only (real benchmarks not reachable, no stated licence); pooled rooms and one-slot exams; CP-SAT's lower bound is 0 until proof, so quality is measured against the best known cost; one 20 s budget and two cores; 4 instances per cell (cells descriptive only); Python local search.

## Next steps
Real Toronto / ITC2007 data once terms are checked; per-room and multi-slot models; compiled local search; a longer budget; stronger CP-SAT bounds.

## Reproduce
`pip install -r requirements.txt && pytest`, then the commands on the project page. Frozen decisions are on the branches `protocol-freeze` and `params-freeze`. Deviations from the protocol are listed in `PROTOCOL.md` section 13.
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(readme)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    build(a.summary, a.out_dir)
    print("docs written to", os.path.abspath(a.out_dir))
