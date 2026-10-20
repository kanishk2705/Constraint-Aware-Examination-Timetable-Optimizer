import pathlib
import re

import numpy as np
import pytest

from examtt.cpsat import solve_slots, solve_soft
from examtt.generator import generate
from examtt.greedy import greedy
from examtt.instance import Instance
from examtt.localsearch import LSParams, anneal, construct_feasible, run_ls
from examtt.objective import evaluate
from examtt.state import State
from examtt.validator import validate

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "examtt"


@pytest.fixture(scope="module")
def small():
    d = generate("t_small", 777, 30, 0.25, 0.85)
    return d, Instance(d)


def test_generator_is_deterministic():
    a = generate("x", 5, 30, 0.25, 0.85)
    b = generate("x", 5, 30, 0.25, 0.85)
    assert a == b
    assert generate("x", 6, 30, 0.25, 0.85) != a


@pytest.mark.parametrize("dens", [0.10, 0.25, 0.45])
def test_generator_hits_density_and_is_feasible(dens):
    d = generate("g", 11, 60, dens, 0.85)
    inst = Instance(d)
    assert dens <= inst.density <= dens + 0.03
    assert validate(d, d["planted"])["ok"]
    assert inst.tightness <= 1.0


@pytest.mark.parametrize("tau", [0.70, 0.85, 0.97])
def test_generator_hits_target_tightness(tau):
    for n, D in [(14, 2), (32, 4), (64, 7)]:
        inst = Instance(generate("g", 21, n, 0.25, tau, n_days=D))
        assert tau - 0.01 <= inst.tightness <= tau + 0.001, (n, tau, inst.tightness)


def test_state_evaluator_validator_agree_on_greedy(small):
    d, inst = small
    built = 0
    for order in ("random", "ldf", "dsatur"):
        for seed in range(3):
            st = greedy(inst, order, rng=np.random.default_rng(seed))
            if st is None:  # a greedy may legitimately fail inside the fixed horizon
                continue
            built += 1
            rep = validate(d, [int(x) for x in st.assign])
            assert rep["ok"]
            assert st.cost == evaluate(inst, st.assign)["total"] == rep["objective"]["total"]
    assert built >= 3


def test_delta_relocate_matches_actual_move(small):
    d, inst = small
    st = greedy(inst, "dsatur", rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(400):
        e = int(rng.integers(inst.n))
        ts = st.feasible_slots(e)
        ts = ts[ts != st.assign[e]]
        if len(ts) == 0:
            continue
        t = int(rng.choice(ts))
        predicted = int(st.delta_relocate(e, np.array([t]))[0])
        actual = st.move(e, t)
        assert predicted == actual
        assert st.cost == evaluate(inst, st.assign)["total"]
        checked += 1
    assert checked > 50


def test_local_search_keeps_hard_constraints_and_improves(small):
    d, inst = small
    import time
    start = construct_feasible(inst, np.random.default_rng(2), deadline=time.perf_counter() + 5)
    assert start is not None
    res = anneal(inst, start, budget=2.0, params=LSParams(), rng=np.random.default_rng(4))
    rep = validate(d, [int(x) for x in res.assign])
    assert rep["ok"]
    assert res.cost == rep["objective"]["total"] <= start.cost


def test_cpsat_proves_optimality_on_a_tiny_instance_and_matches_validator():
    d = generate("t_tiny", 9112, 12, 0.25, 0.85, n_days=2)
    inst = Instance(d)
    out = solve_soft(inst, time_limit=30, seed=1, workers=2)
    assert out["status"] == "OPTIMAL"
    rep = validate(d, [int(x) for x in out["assign"]])
    assert rep["ok"]
    assert out["objective"] == rep["objective"]["total"]  # solver claim == independent recomputation
    assert out["best_bound"] == pytest.approx(out["objective"])  # proven optimal: bound meets objective
    greedy_costs = [g.cost for g in (greedy(inst, o, rng=np.random.default_rng(0)) for o in ("random", "ldf", "dsatur")) if g is not None]
    assert greedy_costs and out["objective"] <= min(greedy_costs)


def test_both_formulations_agree_on_the_optimum():
    d = generate("t_tiny2", 9113, 12, 0.25, 0.85, n_days=2)
    inst = Instance(d)
    a = solve_soft(inst, time_limit=30, seed=1, workers=2, formulation="bool")
    b = solve_soft(inst, time_limit=30, seed=1, workers=2, formulation="elem")
    assert a["status"] == b["status"] == "OPTIMAL"
    assert a["objective"] == b["objective"]


def test_cpsat_warm_start_never_worse_than_hint(small):
    d, inst = small
    st = greedy(inst, "dsatur", rng=np.random.default_rng(0))
    out = solve_soft(inst, time_limit=10, seed=2, hint=st.assign)
    assert out["objective"] <= st.cost


def test_slot_minimisation_is_valid_and_not_worse_than_first_fit(small):
    d, inst = small
    st = greedy(inst, "dsatur", rng=np.random.default_rng(0), rule="first_fit", horizon=inst.n)
    ub = int(st.assign.max()) + 1
    out = solve_slots(inst, time_limit=10, seed=1, upper=ub)
    assert out["assign"] is not None
    assert out["objective"] <= ub
    # validate hard constraints on a copy of the instance whose horizon equals the slot count used
    raw = dict(d, n_slots=max(ub, d["n_slots"]))
    assert validate(raw, [int(x) for x in out["assign"]])["ok"]
    assert out["objective"] <= d["n_slots"]  # the planted horizon is a valid upper bound


def test_solvers_never_touch_the_planted_timetable():
    forbidden = re.compile(r"planted")
    for f in ["cpsat.py", "greedy.py", "localsearch.py", "state.py", "objective.py"]:
        assert not forbidden.search((SRC / f).read_text()), f"{f} must not reference the planted timetable"
    # and the default Instance loader drops it
    assert Instance(generate("p", 1, 30, 0.25, 0.85)).planted is None


def test_infeasible_horizon_returns_none():
    d = generate("tight", 3, 60, 0.45, 0.97)
    inst = Instance(d)
    # a horizon of 2 slots can never hold 60 exams under the capacity
    assert greedy(inst, "dsatur", rng=np.random.default_rng(0), horizon=2) is None
