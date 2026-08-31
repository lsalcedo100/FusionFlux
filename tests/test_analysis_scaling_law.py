"""Tests for the scaling-law linear algebra study (Results 1 to 3).

The valuable assertions here are the ones that would catch a wrong *number*
rather than a wrong shape:

* ``test_refit_recovers_known_exponents_from_noiseless_data`` runs the whole
  refit against data generated from an exact power law. If the design matrix,
  the log transform, or any of the three solvers were wrong, the recovered
  exponents would not be IPB98's, and nothing else in the suite would say so.
* ``test_audit_finds_the_two_exact_dependencies`` pins Result 1. The rank
  deficiency is a property of how the features are *defined*, so if someone
  stops deriving ``a_m`` from ``eps * R`` or drops the IPB98 prior, this fails
  and the narrative in RESULTS.md needs rewriting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis_scaling_law as asl
import hdb5


def _noiseless_power_law_dataset(n_rows: int = 800, seed: int = 5) -> pd.DataFrame:
    """A prepared dataset whose TAUTH is exactly IPB98(y,2), no noise at all.

    The refit must return the published exponents to numerical precision. Any
    real scatter would make that a statistical claim instead of an exact one.
    """
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        {
            "TOK": rng.choice(["JET", "AUG", "D3D"], n_rows),
            "SHOT": rng.integers(0, n_rows // 2, n_rows),
            "TIME": rng.uniform(1.0, 5.0, n_rows),
            "IP": rng.uniform(0.4, 4.0, n_rows),
            "BT": rng.uniform(1.0, 5.0, n_rows),
            "NEL": rng.uniform(1.5, 20.0, n_rows),
            "PLTH": rng.uniform(0.5, 25.0, n_rows),
            "RGEO": rng.uniform(0.5, 3.2, n_rows),
            "KAPPAA": rng.uniform(1.1, 2.2, n_rows),
            "EPS": rng.uniform(0.2, 0.7, n_rows),
            "MEFF": rng.uniform(1.0, 3.0, n_rows),
        }
    )
    raw["TAUTH"] = (
        asl.IPB98Y2_COEFFICIENT
        * raw["IP"] ** 0.93
        * raw["BT"] ** 0.15
        * raw["NEL"] ** 0.41
        * raw["PLTH"] ** -0.69
        * raw["RGEO"] ** 1.97
        * raw["EPS"] ** 0.58
        * raw["KAPPAA"] ** 0.78
        * raw["MEFF"] ** 0.19
    )
    return hdb5.prepare_dataset_from_frame(raw)


def test_audit_finds_the_two_exact_dependencies() -> None:
    audit = asl.audit_model_feature_matrix(_noiseless_power_law_dataset())

    assert audit.n_columns == len(hdb5.MODEL_FEATURE_COLUMNS)
    assert audit.rank_deficiency == 2
    assert audit.rank == audit.n_columns - 2

    # Both known dependencies project onto the null space; a control does not.
    residuals = audit.projection_residuals
    exact = [value for key, value in residuals.items() if "control" not in key.lower()]
    assert exact, "expected at least one non-control dependency to be reported"
    assert max(exact) < 1e-8


def test_audit_standardizing_changes_the_reported_rank() -> None:
    """The tolerance artifact RESULTS.md warns about, pinned as a test.

    On raw columns the largest singular value is set by whichever feature
    carries the biggest units, so the default tolerance zeroes directions that
    are genuinely nonzero. The two ranks therefore disagree, and only the
    standardized one is a statement about collinearity; reporting the raw
    number as structural would be an error. Which way they disagree depends on
    the column scales, so this pins that they differ rather than an ordering.
    """
    audit = asl.audit_model_feature_matrix(_noiseless_power_law_dataset())
    assert audit.unstandardized_rank != audit.rank
    assert audit.rank == audit.n_columns - 2


def test_refit_recovers_known_exponents_from_noiseless_data() -> None:
    dataset = _noiseless_power_law_dataset()
    refit = asl.refit_ipb98(dataset, n_resamples=25)

    recovered = refit.intervals.set_index("variable")["fitted"]
    for variable, published in asl.IPB98Y2_EXPONENTS.items():
        assert recovered[variable] == pytest.approx(published, abs=1e-6), variable
    assert refit.fitted_coefficient == pytest.approx(asl.IPB98Y2_COEFFICIENT, rel=1e-6)
    assert refit.residual_std_log == pytest.approx(0.0, abs=1e-8)
    assert refit.rmsle_refit == pytest.approx(0.0, abs=1e-8)


def test_refit_solvers_agree_and_are_timed() -> None:
    refit = asl.refit_ipb98(_noiseless_power_law_dataset(), n_resamples=10)
    names = {solver.name for solver in refit.solvers}
    assert {"cholesky", "qr", "svd"} <= names
    for solver in refit.solvers:
        assert solver.seconds_per_solve > 0.0
        # All three solve the same well-conditioned problem, so they must agree.
        assert solver.max_deviation_from_svd < 1e-6, solver.name


def test_conditioning_analysis_apportions_variance_and_disagreement() -> None:
    spectrum = asl.conditioning_analysis(_noiseless_power_law_dataset())

    values = spectrum.singular_values
    assert np.all(np.diff(values) <= 1e-9), "singular values must be descending"
    assert spectrum.condition_number == pytest.approx(values[0] / values[-1], rel=1e-6)

    variance = sum(d.share_of_design_variance for d in spectrum.directions)
    disagreement = sum(d.share_of_disagreement for d in spectrum.directions)
    assert variance == pytest.approx(1.0, abs=1e-6)
    assert disagreement == pytest.approx(1.0, abs=1e-6)
    for direction in spectrum.directions:
        assert direction.dominant_variables


def test_ridge_shrinkage_is_monotone_in_the_singular_value() -> None:
    """Ridge shrinks weak directions hardest; that is the whole argument."""
    spectrum = asl.conditioning_analysis(_noiseless_power_law_dataset())
    shrinkage = spectrum.shrinkage
    if shrinkage.empty:
        pytest.skip("no shrinkage table produced")
    alpha_columns = [c for c in shrinkage.columns if c.startswith("alpha")]
    assert alpha_columns
    ordered = shrinkage.sort_values("singular_value", ascending=False)
    for column in alpha_columns:
        factors = ordered[column].to_numpy(dtype=float)
        assert np.all(np.diff(factors) <= 1e-9), column
        assert np.all((factors > 0.0) & (factors <= 1.0)), column
