"""Independent solution validator.

Deliberately shares NO code with the solvers: it does not import instance.py, objective.py, state.py or any
solver. It re-derives everything from the raw instance JSON (student cohorts, capacity, horizon) with plain
Python loops, checks every hard constraint, and recomputes the soft objective student by student.

validate(raw_instance: dict, assign: list[int]) -> dict with keys
    ok (bool), violations (list[str]), objective (dict | None)
"""
from __future__ import annotations


def _position(t: int, periods: int, gap: int) -> int:
    return (t // periods) * (periods + gap) + (t % periods)


def validate(raw: dict, assign) -> dict:
    n = raw["n_exams"]
    T = raw["n_slots"]
    P = raw["periods_per_day"]
    G = raw["day_gap"]
    cap = raw["capacity"]
    limit = raw["max_per_day"]
    w_over = raw["w_overload"]
    pen = list(raw["penalty_by_distance"])
    violations: list = []

    if assign is None or len(assign) != n:
        return {"ok": False, "violations": ["timetable does not have exactly one entry per exam"], "objective": None}
    for e, t in enumerate(assign):
        if not isinstance(t, int) or isinstance(t, bool) or t < 0 or t >= T:
            violations.append(f"exam {e}: slot {t!r} outside 0..{T - 1}")
    if violations:
        return {"ok": False, "violations": violations, "objective": None}

    # enrolment per exam, recomputed from cohorts
    enrol = [0] * n
    for s in raw["students"]:
        for e in set(s["exams"]):
            enrol[e] += s["count"]

    # hard constraint 1: no student sits two exams in one slot
    for idx, s in enumerate(raw["students"]):
        seen: dict = {}
        for e in set(s["exams"]):
            seen.setdefault(assign[e], []).append(e)
        for t, es in seen.items():
            if len(es) > 1:
                violations.append(f"cohort {idx} ({s['count']} students) has exams {sorted(es)} all in slot {t}")

    # hard constraint 2: room capacity per slot
    slot_load = [0] * T
    for e in range(n):
        slot_load[assign[e]] += enrol[e]
    for t in range(T):
        if slot_load[t] > cap:
            violations.append(f"slot {t}: {slot_load[t]} seats needed > capacity {cap}")

    objective = None
    if not violations:
        b2b = gap2 = gap3 = overload = 0
        for s in raw["students"]:
            m = s["count"]
            ex = sorted(set(s["exams"]))
            pos = [_position(assign[e], P, G) for e in ex]
            for a in range(len(ex)):
                for b in range(a + 1, len(ex)):
                    d = abs(pos[a] - pos[b])
                    if d == 1:
                        b2b += m * pen[1]
                    elif d == 2:
                        gap2 += m * pen[2]
                    elif d == 3:
                        gap3 += m * pen[3]
            per_day: dict = {}
            for e in ex:
                day = assign[e] // P
                per_day[day] = per_day.get(day, 0) + 1
            for c in per_day.values():
                if c > limit:
                    overload += m * w_over * (c - limit)
        objective = {"total": b2b + gap2 + gap3 + overload, "b2b": b2b, "gap2": gap2, "gap3": gap3, "overload": overload}
    return {"ok": not violations, "violations": violations, "objective": objective}


def slots_used(assign) -> int:
    return max(assign) + 1 if len(assign) else 0
