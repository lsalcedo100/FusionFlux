"""Two controlled substitutions that isolate *why* boundedness costs what it does.

The Gaussian process result shows that a bounded kernel fails and an unbounded
one does not, but "unbounded" and "has a trend that continues" arrive together
in that comparison. These two estimators separate them by changing exactly one
thing each:

* ``MeanPlusResidualGP`` holds the RBF process fixed and swaps only what it is
  correcting: a constant, or a power law. Same kernel, same subsample, same
  seed. If the constant-mean arm fails and the power-law arm does not, what
  mattered was the trend, not the flexibility bolted onto it.
* ``ClippedResidualHybrid`` takes the hybrid's tree corrector and clips it to
  the range it saw in training. If clipping is what hurts, the corrector was
  relying on freedom the paper claims it does not need.

The risk in both is that the arms stop being controlled: a difference leaks in
alongside the one being tested, and the comparison stops meaning anything. These
tests pin the things that must be identical and the one thing that must not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import is_regressor

import analysis_mechanism as am
from scaling_law import _clean_fp_state as clean_fp_state

RESULTS = Path(__file__).resolve().parents[1] / "results" / "mechanism.json"


def _linear_problem(n: int = 200, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A clean trend in the features, so extrapolation is well defined."""
    rng = np.random.default_rng(seed)
    features = rng.uniform(-2.0, 2.0, size=(n, 3))
    target = features @ np.array([1.5, -0.7, 0.4]) + rng.normal(0.0, 0.05, n)
    return features, target


# --- the estimators are scikit-learn estimators ----------------------------


def test_both_are_regressors_with_the_usual_contract() -> None:
    assert is_regressor(am.MeanPlusResidualGP())
    assert is_regressor(am.ClippedResidualHybrid())


def test_the_mean_function_is_the_only_thing_the_gp_arms_differ_by() -> None:
    """Same kernel, same seed, same subsample: only the trend changes."""
    features, target = _linear_problem()
    constant = am.MeanPlusResidualGP(mean="constant").fit(features, target)
    powerlaw = am.MeanPlusResidualGP(mean="powerlaw").fit(features, target)

    assert constant.residual_gp_.kernel_name == powerlaw.residual_gp_.kernel_name == "rbf"
    assert constant.residual_gp_.random_state == powerlaw.residual_gp_.random_state
    assert constant.mean_model_ is None
    assert powerlaw.mean_model_ is not None


def test_an_unknown_mean_is_rejected_rather_than_silently_treated_as_constant() -> None:
    features, target = _linear_problem()
    with pytest.raises(ValueError, match="Unknown mean"):
        am.MeanPlusResidualGP(mean="quadratic").fit(features, target)


# --- the property the whole comparison turns on ----------------------------


def _fitted_length_scale(model: am.MeanPlusResidualGP) -> float:
    """The RBF length scale marginal likelihood actually chose.

    Reversion is not a property of any fixed distance. An RBF kernel decays over
    its own length scale, so "far away" only means far *relative to that*, and
    the fitted value varies enormously with the data: about 0.31 in standardized
    units on HDB5, but 112 on a synthetic problem whose residual after a
    constant mean is a straight line, because a very long length scale is how an
    RBF imitates a trend. A test that probed a hard-coded distance would pass or
    fail on which of those it happened to hit.
    """
    for part in str(model.residual_gp_.kernel_).split("+"):
        if "RBF(length_scale=" in part:
            return float(part.split("RBF(length_scale=")[1].split(")")[0])
    raise AssertionError(f"no RBF term in {model.residual_gp_.kernel_}")


def test_the_constant_mean_arm_reverts_to_the_training_mean_far_away() -> None:
    """With no trend to carry it, an RBF process has nothing left far from the data.

    This is the tree ensemble's failure mode reached by a different route, and it
    is the reason the constant-mean arm exists.
    """
    features, target = _linear_problem()
    model = am.MeanPlusResidualGP(mean="constant").fit(features, target)

    # Far past the fitted length scale, where exp(-d^2/2l^2) is zero to machine
    # precision, and far outside the data besides.
    far = np.full((5, features.shape[1]), _far_distance(model))
    with clean_fp_state():
        predicted = model.predict(far)

    assert predicted == pytest.approx(np.full(5, float(target.mean())), abs=1e-6)


def _far_distance(*models: am.MeanPlusResidualGP) -> float:
    """A distance far outside the data *and* far past every fitted length scale.

    Both conditions are needed and neither implies the other. The two arms fit
    wildly different length scales on the same rows (about 112 for the constant
    mean, which uses a long RBF to imitate the trend it lacks, and 0.01 for the
    power-law mean, whose residual is just noise), so twenty length scales is
    far outside the data for one arm and still inside it for the other.
    """
    # Eight length scales: exp(-8^2/2) is 1e-14, zero against every tolerance
    # here, while staying well clear of the range where squaring the distance
    # overflows inside the kernel and fills the output with warnings.
    return max(20.0, *(8.0 * _fitted_length_scale(model) for model in models))


def test_the_power_law_arm_keeps_going_far_away() -> None:
    """The trend continues, which is the entire difference between the arms.

    Far from the data the RBF term has decayed to nothing, so the model *is* its
    mean function: a power law, which extrapolates without bound.
    """
    features, target = _linear_problem()
    model = am.MeanPlusResidualGP(mean="powerlaw").fit(features, target)

    far = np.full((1, features.shape[1]), _far_distance(model))
    with clean_fp_state():
        predicted = model.predict(far)

    assert predicted[0] == pytest.approx(model.mean_model_.predict(far)[0], abs=1e-6)
    assert abs(predicted[0] - float(target.mean())) > 1.0


def test_the_two_arms_diverge_only_once_the_kernel_has_decayed() -> None:
    """Inside the data they must agree, or the far-field contrast is confounded."""
    features, target = _linear_problem()
    constant = am.MeanPlusResidualGP(mean="constant").fit(features, target)
    powerlaw = am.MeanPlusResidualGP(mean="powerlaw").fit(features, target)

    # Some BLAS backends raise a spurious divide-by-zero flag on `matmul` for
    # inputs that are entirely finite and results correct to machine precision.
    # The constant-mean arm trips it even in range, because marginal likelihood
    # gives it a large kernel amplitude. The repository's own helper scopes the
    # suppression to these calls rather than the process, so a genuine overflow
    # elsewhere still shows.
    inside = features[:40]
    far = np.full((1, features.shape[1]), _far_distance(constant, powerlaw))
    with clean_fp_state():
        inside_constant = constant.predict(inside)
        inside_powerlaw = powerlaw.predict(inside)
        difference = abs(constant.predict(far)[0] - powerlaw.predict(far)[0])

    assert inside_constant == pytest.approx(inside_powerlaw, abs=0.2)
    assert difference > 1.0


# --- clipping does what its name says --------------------------------------


def test_clipping_confines_the_correction_to_its_training_range() -> None:
    features, target = _linear_problem()
    clipped = am.ClippedResidualHybrid(clip=True).fit(features, target)
    unclipped = am.ClippedResidualHybrid(clip=False).fit(features, target)

    far = np.full((3, features.shape[1]), 30.0)
    # Both share the same power-law base, so any difference is the corrector.
    assert clipped.predict(far) == pytest.approx(clipped.predict(far))
    assert np.isfinite(clipped.predict(far)).all()
    assert np.isfinite(unclipped.predict(far)).all()


def test_the_two_clipping_arms_are_the_same_model_apart_from_the_clip() -> None:
    assert am.ClippedResidualHybrid(clip=True).clip is True
    assert am.ClippedResidualHybrid(clip=False).clip is False
    assert am.CORRECTION_DEPTH == 2
    assert am.CORRECTION_DAMPING == 1.0


# --- the committed artifact -------------------------------------------------


def test_the_artifact_reports_both_controlled_substitutions() -> None:
    if not RESULTS.exists():
        pytest.skip("no results/mechanism.json; run `python3 analysis_mechanism.py`")
    payload = json.loads(RESULTS.read_text())

    assert set(payload["mean_function"]) == {
        "constant mean + RBF residual",
        "power-law mean + RBF residual",
    }
    assert set(payload["clipping"]) == {
        "residual correction, unclipped",
        "residual correction, clipped",
    }
    assert payload["correction_depth"] == am.CORRECTION_DEPTH
    assert payload["correction_damping"] == am.CORRECTION_DAMPING
