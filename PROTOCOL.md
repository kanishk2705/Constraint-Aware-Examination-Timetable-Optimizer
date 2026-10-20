# PROTOCOL — written and frozen before tuning and before any held-out run

Status: **PROTOCOL v1, frozen on branch `protocol-freeze`** (commit hash is recorded in `results/summary.json` and in the
README). Anything that changes after this point is listed in section 13 *Deviations*, with a date and a reason. Nothing
in this file is edited after the freeze except by appending to section 13.

## 1. Question

When does exact solving (OR-Tools CP-SAT) pay off for examination timetabling, compared with greedy constructions and a
local-search improvement step? "Pay off" is measured on three things: whether optimality can be *proven*, the solution
quality reached in a fixed wall-clock budget, and feasibility under tight capacity. The study maps the answer against
three controlled instance factors: **size, conflict density, capacity tightness**.

## 2. Problem definition (the objective is fixed here)

* Exams are vertices of a conflict graph; two exams share an edge if at least one student takes both (edge weight = number
  of shared students). Students come as cohorts (a set of exams plus a head-count).
* Timeline: `n_days` days x 4 periods. A slot is (day, period). Slots are laid on a line with 3 unusable positions between
  days, so a position distance of 1, 2 or 3 can only occur inside one day.
* **Hard constraints** — (H1) every exam gets exactly one slot of the fixed horizon; (H2) no student sits two exams in
  one slot; (H3) the total head-count of the exams in a slot never exceeds the capacity `C` (one pooled seat count per slot).
* **Soft objective** (minimise), summed over students: 5 per pair of exams at position distance 1 (back-to-back), 2 per pair
  at distance 2, 1 per pair at distance 3 (spread), plus 8 per exam beyond 2 on one day (overload). "Minimise number of slots"
  is Study A (section 6), because with a fixed horizon it is a different question from the soft objective.

## 3. Instances

* Generator: `src/examtt/generator.py`, documented in `DATA.md`. It plants a feasible timetable, so every instance is feasible; the planted
  timetable is stored separately and is never read by a solver (a unit test greps the solver modules for it).
* Factors (fully crossed, 27 cells): size S/M/L = 14 / 32 / 64 exams with 2 / 4 / 7 days (horizon 8 / 16 / 28 slots);
  density (conflict-graph edges / all pairs) low / mid / high = 0.10 / 0.25 / 0.45; capacity tightness (total seats /
  (slots x capacity)) loose / medium / tight = 0.70 / 0.85 / 0.97 (achieved values are recorded per instance and reported).
* Sets and seeds: **pilot** seeds 9000+k (calibration only, never reused), **tuning** seeds 1000+k, 2 instances per cell
  (54), **eval** seeds 5000+k, 4 instances per cell (108). `k` is the enumeration index in `scripts/generate_instances.py`.
  SHA-256 hashes of every instance file are in `data/<set>/MANIFEST.json`.
* **Tuning/evaluation discipline:** all choices (SA parameters, decision-rule threshold) are made on tuning instances only.
  The eval set is generated up front but is run exactly once, after the parameters are frozen on branch `params-freeze`.
  The eval set is never used to choose anything.
* Benchmarks: the public Carter/Toronto and ITC2007 instances are **not used**. The sandbox network policy blocks the
  hosts (HTTP 403 from the egress proxy, not circumvented) and none of the pages I could read states a licence (see `DATA.md`).

## 4. Methods (Study B: soft objective on the fixed horizon)

| id | method |
|---|---|
| `greedy_random` | random exam order, place each exam in the feasible slot with the smallest cost increase; seed 0 |
| `greedy_ldf` | largest conflict-degree first (ties: enrolment, index), same slot rule |
| `greedy_dsatur` | generalised DSATUR: next exam is the one with fewest feasible slots (clash or capacity), ties by degree, same slot rule |
| `ls` | start from the best feasible greedy (else randomised-DSATUR restarts for at most 30% of the budget), then simulated annealing with relocate/swap moves that never break a hard constraint; best of **2 independent runs in 2 processes** (seeds 101, 102) |
| `cpsat_cold` | CP-SAT, 2 workers, seed 1, no hint |
| `hybrid` | `ls` for 25% of the budget (seeds 201, 202), then CP-SAT warm-started from that solution for 75% (2 workers) |

* Budget **B = 20 s wall-clock** per method per instance, same 2 CPU cores for everyone (LS uses 2 processes, CP-SAT 2 workers).
  A greedy construction is not budgeted (milliseconds). The LS budget includes its construction time.
* CP-SAT model (fixed on pilot instances, see section 12): Boolean slot variables, one at-most-one per distinct student cohort and slot,
  one capacity constraint per slot, pair penalties linearised with one Boolean per (pair, distance) and one implication per slot, overload via
  per-student-per-day integer variables. A second, structurally different encoding (element constraints) exists in the code; unit tests
  check both reach the same optimum.
* SA parameters (`t0`, `t1`, `p_swap`) are tuned on tuning instances (section 8) and frozen.

## 5. Validation (applies to every run)

Every timetable produced by any method is re-checked by `src/examtt/validator.py`, which shares no code with the solvers and recomputes the objective student
by student. A run whose claimed objective differs from the validator's is invalid and aborts the analysis. A timetable that violates a hard constraint is
recorded as *infeasible*, never as a good result. Unit tests include cases proving the validator fails on clashes, capacity violations, out-of-range and missing slots.

## 6. Study A (number of slots, hard constraints only)

First-fit greedy with the three orders (unlimited horizon), `ms_dsatur` (randomised-tie DSATUR restarts for B seconds, best of 2 processes, seeds 301, 302),
and CP-SAT minimising the number of used slots (2 workers, B seconds, hinted by the best greedy). Lower bound = max(largest cohort size, ceil(total seats / C)).

## 7. Metrics

* **Feasible rate** per method (Wilson 95% interval).
* **Regret (%)** of method m on instance i = `100 * (cost_m - best_i) / max(best_i, 1)`, where `best_i` is the smallest *validated* cost found by any
  Study-B method on i. Methods without a feasible timetable have undefined regret and are dropped pairwise.
* **Proven-optimal** = CP-SAT status OPTIMAL within B. Reported gap = `(obj - bound) / obj`. CP-SAT's best bound on this objective is almost always 0 until it proves optimality
  (pilot observation), so the gap is reported but flagged as uninformative whenever bound = 0; **regret vs best-known** is the quality measure.
* Time to best (from the improvement trajectory), slots used (Study A), win/tie/loss counts.

## 8. Tuning protocol (tuning set only)

1. **SA parameters:** grid `t0 in {0.5, 1, 2}` x `t1 in {0.01, 0.05}` x `p_swap in {0.1, 0.3, 0.5}` (18 configs), seed 7, 5 s budget, one run per
   (config, instance), on the 36 tuning instances of tiers M and L. Score = mean over instances of `cost / min over configs of cost`. Lowest score wins; ties broken by the lower `t0`.
2. **Decision rule threshold:** run Study B (budget B = 20 s, tuned SA parameters) on all 54 tuning instances. Rule R: use CP-SAT if `edges <= theta`, else LS. `theta` is chosen from the observed
   `edges` values to minimise mean regret of R on the tuning set (ties: smallest theta). Frozen.
3. Nothing else is tuned. The hybrid split (25/75), B and all CP-SAT settings are fixed here.

## 9. Pre-registered hypotheses (confirmatory; fixed before any eval run)

The pilot (3 instances, 6 s budget) suggested the direction of H2; no eval or tuning data had been seen. Decision rules use 95% bootstrap/Wilson intervals (section 10).

* **H1 — proof frontier.** CP-SAT cold proves optimality within B on most tier-S instances and on almost none of tier L. *Supported if* the Wilson interval of the tier-S proven-optimal rate lies entirely above 0.5 and the tier-L interval lies entirely below 0.25.
* **H2 — quality crossover.** Let `D = regret(ls) - regret(cpsat_cold)` (positive = CP-SAT better). H2a: in tier M, mean `D > 0` with the CI excluding 0. H2b: in tier L, mean `D < 0` with the CI excluding 0. *Meaning:* the crossover where a tuned local search overtakes exact solving lies between M and L. If H2a fails but H2b holds, the crossover is below M; if H2b fails, exact solving still wins at L within B.
* **H3 — tightness hurts greedy feasibility.** Pooled over tiers M and L, the feasible rate of `greedy_random` is lower at tight than at loose capacity (difference in rates, bootstrap CI excluding 0), while `ls` and `cpsat_cold` stay feasible on at least 90% of tight instances.
* **H4 — warm starting.** Pooled over tiers M and L, mean `regret(hybrid) - regret(cpsat_cold) < 0` with the CI excluding 0.
* **H5 — a one-feature rule.** On the eval set, the tuned rule R (section 8) has lower mean regret than both always-`ls` and always-`cpsat_cold`; supported if the paired CI of `regret(R) - regret(fixed policy)` is entirely below 0 for both. If one fixed policy is never beaten, the rule has no value on this instance family; that outcome is reported as is.
* **H6 — slot minimisation (Study A).** DSATUR first-fit already matches CP-SAT's slot count on at least 90% of eval instances (exact solving does not pay off for the slot count here); any exceptions are listed by cell. Supported if the match rate is >= 0.90.

Everything not listed above is *exploratory* and labelled as such. No p-values are used; no hypothesis is added or dropped after seeing eval data.

## 10. Statistics

* Unit of analysis: the instance. Paired comparisons: per-instance differences of regret. Percentile bootstrap, 10,000 resamples of instances, seed 20261201, 95% level.
* Marginal comparisons pool over the other factors (36 instances per size, density or tightness level). Per-cell results (4 instances) are descriptive only, never inferential.
* Rates: Wilson 95% interval. Difference of rates: bootstrap of instances within each group.
* Failures: handled by pairwise deletion, with the number of pairs always reported next to the interval.

## 11. Reproducibility

Instance seeds, method seeds, budgets and the bootstrap seed are above. Wall-clock limits make CP-SAT and annealing runs not bit-for-bit repeatable;
the study therefore stores every timetable it produced (`results/raw/`) so that all reported numbers can be recomputed from them and re-validated (`scripts/analyze.py`).
Docs and README numbers are generated from `results/summary.json` (`scripts/build_docs.py`), so they cannot drift.
Hardware: the sandbox has about 2 CPU cores; two processes never compete with each other because stages run one after the other.

## 12. Decisions already taken on pilot instances (before this freeze)

* Size tiers set so that CP-SAT proves optimality at 12–16 exams (0.2 s and 21 s, 2 workers) but not at 20–24 exams (60 s, bound 0), bracketing the frontier.
* CP-SAT encoding: Boolean (pair, distance) encoding chosen over the element-constraint encoding (pilot objectives 715 vs 787 and 1481 vs 1503); `linearization_level=2` was no better; defaults kept.
* Generator fixed to v1.1 before the freeze: in v1.0 the planted slot loads were unbalanced, so the achieved tightness did not follow the target (pilot: 0.69 / 0.74 / 0.76 for targets 0.70 / 0.85 / 0.97). v1.1 equalises planted slot loads with single-exam cohorts; achieved tightness is now 0.70 / 0.85 / 0.97 (unit-tested). All three sets were regenerated with v1.1; pilot data from v1.0 is discarded.
* A 6 s smoke run on the pilot instances showed all three regimes the study is about (e.g. dense + tight, tier M: DSATUR fails the horizon, LS and CP-SAT succeed; tier L: CP-SAT has no feasible timetable yet after 6 s while greedy and LS do). No eval or tuning instance had been run.
* One finding that shaped the design: with a single worker CP-SAT was far weaker than with two (2004 vs 787 at 30 s on one instance), hence 2 workers vs 2 LS processes for an equal CPU budget.

## 13. Deviations (appended after the freeze only)

**Deviation 1 — primary metric (2026-10-20, after the freeze, before any Study-B tuning-set run and before any eval run).**
Percentage regret with the floor `max(best, 1)` (section 7) is unstable when the best-known cost is 0. In the SA tuning runs, 6 of the 36 tuning instances of tiers M and L (all large, low density) had a best cost of exactly 0, so a method that misses it by a few penalty points would get a regret of thousands of percent and dominate every mean. This was found on tuning data only. From now on the **primary metric is the excess**
`100 * (cost_m - best_i) / scale_i`, where `scale_i = 5 * sum over cohorts of count * C(k,2)` (the penalty if every pair of exams a student shares were back-to-back; a fixed property of the instance, independent of any method). H2, H4 and H5 and the threshold fit (8.2) are decided on the excess. The pre-registered percentage regret is still computed and published next to it as a **sensitivity analysis**; where the two disagree, that is reported. No hypothesis, threshold or interval rule changed.

**Deviation 2 — tuning score with a zero minimum (2026-10-20).** Section 8.1 divides by the per-instance minimum cost, which is undefined at 0. The tuning script used `max(min, 1)`. Re-ranking the grid without the 6 zero-minimum instances gives the same winner (`t0 = 2.0, t1 = 0.01, p_swap = 0.5`), so the selection is unaffected.

**Deviation 3 — CP-SAT claimed objective can exceed the validated cost (2026-10-20, found by the analysis audit on the held-out run, before any result was computed).** Section 5 says a run whose claimed objective differs from the validator is invalid and aborts the analysis. The audit found 3 of 532 stored timetables with a mismatch: all `cpsat_cold`, all status FEASIBLE (time limit hit, not proven), all tier L (`eval_L_high_loose_1` claimed 4627 / validated 4602; `eval_L_mid_tight_0` 1254 / 1192; `eval_L_mid_tight_2` 2237 / 1902). All 532 timetables satisfy every hard constraint, and in every mismatch claimed > validated (never the reverse). A plausible cause, not tested: in the boolean encoding the pair-distance indicators are only bounded from below, so an unfinished incumbent can carry slack and the solver's objective overstates the true cost. Handling: the analysis uses the validator's cost for every method (it already did; the claim was never used for results), counts these cases in `audit.claimed_overestimates` and reports them in the README; a claim *below* the validated cost, or any slot-count mismatch, still aborts. The experiment is not re-run. Effect: if anything this makes CP-SAT's internal objective look worse than its timetable, so the comparison uses the timetables, which is the protocol's intent.
