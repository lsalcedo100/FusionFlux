"""Tests for Result 4e: the degree-by-penalty flexibility grid.

The sweep fits with the repository's own ridge solver rather than scikit-learn's
estimator, because reusing one SVD across the whole penalty axis is what makes a
4-by-9 grid affordable at degree 4 and the estimator API cannot express it. That
substitution is the main risk in this module: if the hand-rolled path
standardized differently, penalized the intercept, or filtered the singular
values wrongly, the grid would be internally consistent and still describe a
different family of models than Result 4d's table.

So the load-bearing test is ``test_grid_reproduces_the_sklearn_pipelines_it_replaces``.
The rest pin properties that would otherwise be assumed: that the penalty axis
is monotone in shrinkage, that held-out rows never touch the standardization,
and that the "usable penalty" screen is doing what its name says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import analysis_flexibility_sweep as afs
import hdb5


def _synthetic_dataset(n_rows: int = 400, seed: int = 17) -> pd.DataFrame:
    """A prepared dataset with several machines and real scatter.

    Scatter matters: on noiseless data every degree fits perfectly and the grid
    has nothing to say. Machines are given distinct parameter ranges so that
    holding one out is genuinely an extrapolation, which is the split under test.
    """
    rng = np.random.default_rng(seed)
    machines = {
        "AAA": {"IP": (0.4, 1.2), "RGEO": (0.6, 1.0)},
        "BBB": {"IP": (1.0, 2.5), "RGEO": (1.2, 1.8)},
        "CCC": {"IP": (2.0, 4.0), "RGEO": (2.0, 3.0)},
        "DDD": {"IP": (0.2, 0.8), "RGEO": (0.4, 0.7)},
    }
    frames = []
    for name, ranges in machines.items():
        count = n_rows // len(machines)
        frames.append(
            pd.DataFrame(
                {
                    "TOK": name,
                    "SHOT": rng.integers(0, count // 2 + 1, count),
                    "IP": rng.uniform(*ranges["IP"], count),
                    "BT": rng.uniform(1.0, 5.0, count),
                    "NEL": rng.uniform(1.5, 20.0, count),
                    "PLTH": rng.uniform(0.5, 25.0, count),
                    "RGEO": rng.uniform(*ranges["RGEO"], count),
                    "KAPPAA": rng.uniform(1.1, 2.2, count),
                    "EPS": rng.uniform(0.2, 0.7, count),
                    "MEFF": rng.uniform(1.0, 3.0, count),
                }
            )
        )
    raw = pd.concat(frames, ignore_index=True)
    clean = (
        0.0562
        * raw["IP"] ** 0.93
        * raw["BT"] ** 0.15
        * raw["NEL"] ** 0.41
        * raw["PLTH"] ** -0.69
        * raw["RGEO"] ** 1.97
        * raw["EPS"] ** 0.58
        * raw["KAPPAA"] ** 0.78
        * raw["MEFF"] ** 0.19
    )
    raw["TAUTH"] = clean * np.exp(rng.normal(0.0, 0.15, len(raw)))
    return hdb5.prepare_dataset_from_frame(raw)


# --- the substitution the whole module rests on -----------------------------


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_grid_reproduces_the_sklearn_pipelines_it_replaces(degree: int) -> None:
    """One cell of the grid, against the pipeline Result 4d actually fitted.

    ``StandardScaler`` then ``Ridge(alpha=1.0, solver="svd")`` with the default
    ``fit_intercept=True`` is exactly what ``hdb5.build_control_models`` and
    ``analysis_extrapolation.build_flexibility_ladder`` construct. If this
    passes, the sweep's alpha = 1.0 column *is* Result 4d's ladder, computed a
    different way, and the rest of the grid is that same family extended.
    """
    dataset = _synthetic_dataset()
    features = list(hdb5.BLIND_FEATURE_COLUMNS)
    held = (dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "CCC").to_numpy()

    expanded = afs.polynomial_expansion(dataset, degree, hdb5.BLIND_FEATURE_COLUMNS)
    log_target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    fold = afs._factor_fold(expanded[~held], expanded[held], log_target[~held])
    ours = fold.predict_log(afs.REFERENCE_ALPHA)

    steps = []
    if degree > 1:
        steps.append(("expand", PolynomialFeatures(degree=degree, include_bias=False)))
    steps += [("scale", StandardScaler()), ("model", Ridge(alpha=1.0, solver="svd"))]
    pipeline = Pipeline(steps)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pipeline.fit(dataset.loc[~held, features], log_target[~held])
        theirs = pipeline.predict(dataset.loc[held, features])

    assert ours == pytest.approx(theirs, rel=1e-6, abs=1e-8)


def test_reference_alpha_is_the_one_result_4d_reports() -> None:
    """A guard on the claim that the grid contains the published table.

    If someone re-centres the penalty grid, the cross-check above silently stops
    testing the published setting; this makes that a failure instead.
    """
    assert afs.REFERENCE_ALPHA == 1.0
    assert afs.REFERENCE_ALPHA in afs.RIDGE_ALPHAS
    assert afs.POLYNOMIAL_DEGREES[:3] == (1, 2, 3)


# --- the expansion ----------------------------------------------------------


@pytest.mark.parametrize(
    ("degree", "n_terms"),
    # C(9 + d, d) - 1: the full expansion on nine features, minus the bias.
    [(1, 9), (2, 54), (3, 219), (4, 714)],
)
def test_expansion_has_the_documented_number_of_terms(degree: int, n_terms: int) -> None:
    dataset = _synthetic_dataset(n_rows=40)
    expanded = afs.polynomial_expansion(dataset, degree, hdb5.BLIND_FEATURE_COLUMNS)
    assert expanded.shape == (len(dataset), n_terms)


def test_expansion_rejects_degree_zero() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        afs.polynomial_expansion(_synthetic_dataset(n_rows=40), 0, hdb5.BLIND_FEATURE_COLUMNS)


# --- the fold --------------------------------------------------------------


def test_standardization_never_sees_the_held_out_rows() -> None:
    """Leakage check: the transform must come from training rows alone.

    Standardizing on the pooled data would let the held-out machine set the
    scale it is later judged against, which is precisely the leak
    leave-one-tokamak-out exists to exclude.
    """
    dataset = _synthetic_dataset()
    expanded = afs.polynomial_expansion(dataset, 2, hdb5.BLIND_FEATURE_COLUMNS)
    log_target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    held = (dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "CCC").to_numpy()

    fold = afs._factor_fold(expanded[~held], expanded[held], log_target[~held])
    assert fold.column_mean == pytest.approx(expanded[~held].mean(axis=0))
    assert fold.target_mean == pytest.approx(float(log_target[~held].mean()))
    # And emphatically not the pooled statistics.
    assert not np.allclose(fold.column_mean, expanded.mean(axis=0))


def test_constant_columns_get_unit_scale_rather_than_a_division_by_zero() -> None:
    design = np.column_stack([np.linspace(0.0, 1.0, 20), np.full(20, 3.0)])
    fold = afs._factor_fold(design, design[:5], np.linspace(0.0, 1.0, 20))
    assert fold.column_scale[1] == 1.0
    assert np.isfinite(fold.predict_log(1.0)).all()


def test_larger_penalty_shrinks_the_fit_towards_the_training_mean() -> None:
    """The penalty axis has to mean what it says, monotonically.

    Every reading of the grid depends on "further right = more shrinkage". This
    checks it on the object itself rather than trusting the formula.
    """
    dataset = _synthetic_dataset()
    expanded = afs.polynomial_expansion(dataset, 2, hdb5.BLIND_FEATURE_COLUMNS)
    log_target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    held = (dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "CCC").to_numpy()
    fold = afs._factor_fold(expanded[~held], expanded[held], log_target[~held])

    spreads = [
        float(np.max(np.abs(fold.predict_log(alpha) - fold.target_mean)))
        for alpha in (1e-3, 1.0, 1e3, 1e6, 1e9)
    ]
    assert spreads == sorted(spreads, reverse=True)
    # Relative rather than absolute: what matters is that the penalty collapses
    # the fit onto the intercept, and the scale it collapses *from* depends on
    # the synthetic data rather than on anything under test.
    assert spreads[-1] < 0.001 * spreads[0]


# --- the clip --------------------------------------------------------------


def test_clip_reports_what_it_caught_instead_of_hiding_it() -> None:
    truth = np.zeros(4)
    predicted = np.array([0.0, 1.0, 1e6, -1e6])
    rmsle, n_clipped = afs._rmsle_from_log(truth, predicted)
    assert n_clipped == 2
    assert np.isfinite(rmsle)


def test_clip_converts_non_finite_predictions_rather_than_propagating_them() -> None:
    """A nan would survive ``clip`` and turn the cell's RMSLE into nan silently."""
    truth = np.zeros(3)
    rmsle, n_clipped = afs._rmsle_from_log(truth, np.array([np.nan, np.inf, -np.inf]))
    assert np.isfinite(rmsle)
    assert n_clipped == 3


# --- the sweep as a whole ---------------------------------------------------


@pytest.fixture(scope="module")
def sweep() -> afs.FlexibilitySweep:
    return afs.sweep_flexibility(
        _synthetic_dataset(),
        degrees=(1, 2, 3),
        alphas=(1e-2, 1.0, 1e2, 1e6),
        min_rows=20,
        n_splits=3,
        focus_machine="CCC",
    )


def test_sweep_scores_every_cell_of_the_grid(sweep: afs.FlexibilitySweep) -> None:
    assert len(sweep.cells) == len(sweep.degrees) * len(sweep.alphas)
    assert {(cell.degree, cell.alpha) for cell in sweep.cells} == {
        (degree, alpha) for degree in sweep.degrees for alpha in sweep.alphas
    }
    for cell in sweep.cells:
        assert cell.lomo_median_rmsle <= cell.lomo_worst_rmsle
        assert cell.worst_machine in sweep.machines
        assert np.isfinite(cell.cv_rmsle)


def test_per_machine_frame_covers_every_cell_and_machine(sweep: afs.FlexibilitySweep) -> None:
    frame = sweep.per_machine
    expected = len(sweep.degrees) * len(sweep.alphas) * len(sweep.machines)
    assert len(frame) == expected
    assert set(frame["tokamak"]) == set(sweep.machines)
    # The worst machine recorded on a cell must be the argmax of that cell's rows.
    for cell in sweep.cells:
        rows = frame[(frame["degree"] == cell.degree) & (frame["alpha"] == cell.alpha)]
        assert rows.loc[rows["rmsle"].idxmax(), "tokamak"] == cell.worst_machine
        assert rows["rmsle"].max() == pytest.approx(cell.lomo_worst_rmsle)


def test_a_crushing_penalty_is_marked_unusable(sweep: afs.FlexibilitySweep) -> None:
    """The screen that stops "everything is equally broken" reading as "flat".

    Without it, a slope of zero at alpha = 1e6 would look like evidence that
    regularization makes flexibility free, when it is evidence that the penalty
    has destroyed the baseline as well.
    """
    statuses = {status.alpha: status for status in sweep.penalties}
    assert statuses[1e6].is_usable is False
    assert statuses[1e6].baseline_ratio > afs.BASELINE_TOLERANCE
    assert statuses[afs.REFERENCE_ALPHA].is_usable is True
    assert afs.REFERENCE_ALPHA in sweep.usable_alphas
    assert 1e6 not in sweep.usable_alphas


def test_slopes_are_reported_for_every_statistic_and_penalty(sweep: afs.FlexibilitySweep) -> None:
    keys = {(slope.statistic, slope.alpha) for slope in sweep.slopes}
    for statistic in ("lomo_worst_rmsle", "lomo_mean_rmsle", "lomo_median_rmsle"):
        for alpha in sweep.alphas:
            assert (statistic, alpha) in keys
    for slope in sweep.slopes:
        assert slope.factor_per_degree == pytest.approx(10.0**slope.slope_per_degree)


def test_best_penalty_is_the_grid_minimum_for_its_degree(sweep: afs.FlexibilitySweep) -> None:
    frame = sweep.to_frame()
    for best in sweep.best_penalties:
        rows = frame[frame["degree"] == best.degree]
        assert best.best_worst_rmsle == pytest.approx(rows["lomo_worst_rmsle"].min())
    # Degree 1 is the baseline, so its ratio to itself is exactly 1.
    assert sweep.best_penalties[0].worst_ratio_to_degree_one == pytest.approx(1.0)


def test_serialized_sweep_carries_the_dataset_fingerprint(sweep: afs.FlexibilitySweep) -> None:
    payload = sweep.to_json()
    assert "dataset" in payload
    assert set(payload["dataset"]) >= {"sha256", "n_bytes", "matches_pin"}
    assert payload["reference_alpha"] == afs.REFERENCE_ALPHA
    assert payload["usable_alphas"] == sweep.usable_alphas
