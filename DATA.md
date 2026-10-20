# DATA — instances, generator and benchmark status

## Why constructed instances

The study needs controlled difficulty (size, conflict density, capacity tightness), fixed seeds, and a feasible solution known to exist. A real benchmark gives none of the three knobs, so the main study uses a documented generator. Real public benchmarks were checked as an optional extra; see the last section.

## Generator (`src/examtt/generator.py`, version 1.1)

1. Choose the size tier (number of exams, number of days of 4 periods; horizon = 4 x days slots).
2. **Plant** a timetable: split the exams into `n_slots` groups of (almost) equal size.
3. Draw student cohorts until the conflict graph reaches the target density. A cohort takes 2-5 exams (probabilities 0.10 / 0.35 / 0.35 / 0.20) from *different* planted slots, so the planted timetable is clash-free by construction. Exams are drawn with log-normal popularity, slots with a bias to the least loaded slot. A cohort has 5-40 students.
4. Give every exam at least one student, then **equalise the planted slot loads** with single-exam cohorts (students who sit only that exam; they add no conflicts). Every planted slot now holds exactly `L*` seats.
5. Capacity `C = ceil(L* / tightness)`. So achieved tightness = total seats / (slots x C) equals the target to within rounding, and the planted timetable fits with a margin of `1 - tightness`.

Consequences to keep in mind (also listed under limitations): every instance is feasible by construction; the conflict graph always has a colouring with `n_slots` colours; exam sizes carry a weak imprint of the planted grouping (heavier exams sit in lightly loaded planted slots); rooms are one pooled seat count per slot, not individual rooms. The planted timetable is stored in the instance file under `planted` but the default loader drops it and a unit test checks that no solver module mentions it.

## Factors and sets

| factor | levels |
|---|---|
| size tier | S = 14 exams / 2 days / 8 slots, M = 32 / 4 / 16, L = 64 / 7 / 28 |
| conflict density (edges / pairs) | low 0.10, mid 0.25, high 0.45 |
| capacity tightness (seats / (slots x C)) | loose 0.70, medium 0.85, tight 0.97 |

| set | base seed | instances per cell | total | use |
|---|---|---|---|---|
| pilot | 9000 | 1 | 27 | calibration before the protocol; discarded |
| tuning | 1000 | 2 | 54 | every tuning decision |
| eval | 5000 | 4 | 108 | held out; one final run |

`data/<set>/MANIFEST.json` lists the seed and the SHA-256 of every file. Reproduce with `python scripts/generate_instances.py <set>`.

Instance file layout: `n_exams, n_days, periods_per_day, day_gap, n_slots, capacity, max_per_day, w_overload, penalty_by_distance, students: [{exams, count}], planted`.

## Real public benchmarks: what I found

Checked: the Carter/Toronto examination timetabling instances and the ITC2007 examination track instances.

* **Reachability.** From the sandbox the three hosts I tried (the University of Nottingham site that hosts the Carter data, the QUB ITC2007 site, and a University of Melbourne timetabling page) were refused by the session's egress proxy with a 403 policy denial. I did not retry or route around it.
* **Licence / terms.** I read the Nottingham data page, the ITC2007 exam-track technical report and a survey of educational-timetabling benchmarks (`arxiv.org/pdf/2201.07525`) through the web-reading tool. None of them states a licence, terms of use, copyright notice or citation requirement for the instances. They only say who created them (Carter, Laporte and Lee for the Toronto set, 1996; the ITC2007 organisers for the exam track) and where they are hosted. "No stated licence" is not the same as "free to redistribute", so I did not copy any benchmark data into this repository.
* **Decision.** No real benchmark is used. All results are on the constructed instances, and the README says so. Running the same code on the real instances later would need only a small parser; this is listed under next steps.
