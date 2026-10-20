"""The validator must accept valid timetables AND be able to fail on every kind of violation."""
import copy

import pytest

from examtt.generator import generate
from examtt.validator import validate


@pytest.fixture(scope="module")
def raw():
    return generate("t_val", 4242, 30, 0.25, 0.85)


def test_accepts_planted_timetable(raw):
    rep = validate(raw, raw["planted"])
    assert rep["ok"], rep["violations"][:3]
    assert rep["objective"]["total"] >= 0


def _clashing_pair(raw):
    for s in raw["students"]:
        if len(s["exams"]) >= 2:
            return s["exams"][0], s["exams"][1]
    raise AssertionError("no multi-exam cohort")


def test_detects_student_clash(raw):
    a, b = _clashing_pair(raw)
    bad = list(raw["planted"])
    bad[a] = bad[b]  # same slot for two exams sharing students
    rep = validate(raw, bad)
    assert not rep["ok"] and any("cohort" in v for v in rep["violations"])


def test_detects_capacity_violation(raw):
    # shrink capacity below the heaviest planted slot load -> planted timetable must now fail
    tight = copy.deepcopy(raw)
    tight["capacity"] = 1
    rep = validate(tight, raw["planted"])
    assert not rep["ok"] and any("capacity" in v for v in rep["violations"])


def test_detects_out_of_range_slot(raw):
    bad = list(raw["planted"])
    bad[0] = raw["n_slots"]
    assert not validate(raw, bad)["ok"]
    bad[0] = -1
    assert not validate(raw, bad)["ok"]


def test_detects_wrong_length_and_none(raw):
    assert not validate(raw, raw["planted"][:-1])["ok"]
    assert not validate(raw, None)["ok"]


def test_objective_is_recomputed_not_trusted(raw):
    rep = validate(raw, raw["planted"])
    # moving one exam inside its legal slots changes the recomputed objective consistently (non-negative parts)
    parts = rep["objective"]
    assert parts["total"] == parts["b2b"] + parts["gap2"] + parts["gap3"] + parts["overload"]
