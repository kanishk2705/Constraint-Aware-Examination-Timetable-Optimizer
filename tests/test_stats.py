import numpy as np

from examtt.stats import bootstrap_mean_ci, paired_diff_ci, regret_pct, wilson_ci, win_tie_loss


def test_bootstrap_ci_contains_true_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 200)
    a = bootstrap_mean_ci(x)
    b = bootstrap_mean_ci(x)
    assert a == b  # fixed seed
    assert a["lo"] < 5.0 < a["hi"]
    assert a["hi"] - a["lo"] < 0.5


def test_paired_ci_excludes_zero_for_a_real_shift_and_includes_it_for_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(100, 10, 60)
    shifted = base + 3 + rng.normal(0, 1, 60)
    noise = base + rng.normal(0, 1, 60)
    s = paired_diff_ci(shifted, base)
    z = paired_diff_ci(noise, base)
    assert s["lo"] > 0
    assert z["lo"] < 0 < z["hi"]


def test_wilson_and_helpers():
    w = wilson_ci(8, 10)
    assert 0.4 < w["lo"] < 0.8 < w["hi"] < 1.0
    assert wilson_ci(0, 0)["rate"] is None
    assert regret_pct(110, 100) == 10.0
    assert regret_pct(5, 0) == 500.0  # floor of 1 avoids division by zero
    assert win_tie_loss([1, 2, 3, None], [2, 2, 1, 5]) == {"win": 1, "tie": 1, "loss": 1}
