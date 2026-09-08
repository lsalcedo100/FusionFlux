"""The two fitting controls actually change what they claim to change.

``analysis_fitting_conventions`` answers two questions a referee is likely to
ask: whether the reversal depends on including spherical tokamaks, which the
IPB98(y,2) selection excluded, and whether it depends on fitting rows
unweighted when two devices supply 77% of them.

A control is only worth reporting if it is really a control, so these tests pin
the mechanics rather than re-deriving the scores: that the spherical labels are
gone from the conventional arm, that the weights reach the estimator instead of
being silently dropped, and that each weight vector gives every label the same
total influence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import analysis_fitting_conventions as afc
import hdb5

RESULTS = Path(__file__).resolve().parents[1] / "results" / "fitting_conventions.json"

# Median inverse aspect ratio above MAX_CONVENTIONAL_INVERSE_ASPECT_RATIO.
SPHERICAL = {"MAST", "NSTX", "START"}


def _dataset_or_skip():
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return hdb5.prepare_dataset()


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/fitting_conventions.json; run `python3 analysis_fitting_conventions.py`")
    return json.loads(RESULTS.read_text())


# --- the aspect-ratio arm drops what it says it drops -----------------------


def test_spherical_labels_are_removed_from_both_sides() -> None:
    dataset = _dataset_or_skip()
    conventional = afc.conventional_aspect_ratio_only(dataset)
    remaining = set(conventional[hdb5.TOKAMAK_LABEL_COLUMN])
    assert not (remaining & SPHERICAL), "a spherical label survived the filter"
    # Dropped from training as well as scoring: the rows are gone entirely,
    # which is what makes this different from the size-cut control in Sec. 6.
    assert len(conventional) < len(dataset)


def test_the_filter_keeps_whole_labels() -> None:
    """A row-wise cut would split a machine across the boundary."""
    dataset = _dataset_or_skip()
    conventional = afc.conventional_aspect_ratio_only(dataset)
    for label, group in conventional.groupby(hdb5.TOKAMAK_LABEL_COLUMN):
        original = int((dataset[hdb5.TOKAMAK_LABEL_COLUMN] == label).sum())
        assert len(group) == original, f"{label} was split by the filter"


# --- the weights are real ---------------------------------------------------


def test_equal_label_weights_give_every_label_the_same_influence() -> None:
    labels = np.array(["JET"] * 100 + ["MAST"] * 4 + ["CMOD"] * 16)
    weights = afc.equal_label_weights(labels)
    totals = {label: weights[labels == label].sum() for label in set(labels)}
    assert np.allclose(list(totals.values()), list(totals.values())[0])
    # Rescaled to the row count, so a weighted fit is on the same scale as an
    # unweighted one and the two summaries can be compared directly.
    assert weights.sum() == pytest.approx(len(labels))


def test_weights_reach_the_estimator() -> None:
    """`model__sample_weight` has to land, or the arm silently measures nothing.

    Fitting the same pipeline with and without a lopsided weight vector must
    produce different coefficients. If the keyword were wrong, scikit-learn
    would raise; if it were dropped, this assertion is what catches it.
    """
    dataset = _dataset_or_skip()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    zoo = hdb5._assemble_zoo()

    plain = hdb5.clone_pipeline(zoo["ridge_loglinear"])
    weighted = hdb5.clone_pipeline(zoo["ridge_loglinear"])
    with hdb5._suppress_benign_matmul_warnings():
        plain.fit(features, target)
        weighted.fit(features, target, model__sample_weight=afc.equal_label_weights(labels))
    assert not np.allclose(plain.named_steps["model"].coef_, weighted.named_steps["model"].coef_), (
        "the weights did not change the fit"
    )


# --- the committed report -----------------------------------------------------


def test_committed_report_covers_all_three_arms(committed: dict) -> None:
    assert set(committed["summaries"]) == {
        "baseline",
        "conventional_aspect_ratio",
        "equal_label_weights",
    }
    assert set(committed["spherical_labels_dropped"]) <= SPHERICAL
    baseline = committed["summaries"]["baseline"]
    assert baseline["n_labels_scored"] == 13
    assert baseline["n_forest_worse_than_power_law"] == 13


def test_the_arms_are_scored_on_different_label_sets(committed: dict) -> None:
    """Otherwise the aspect-ratio arm is the baseline under another name."""
    baseline = set(committed["summaries"]["baseline"]["labels"])
    conventional = set(committed["summaries"]["conventional_aspect_ratio"]["labels"])
    assert conventional < baseline
    assert baseline - conventional


def test_weighting_arm_scores_the_same_labels_as_the_baseline(committed: dict) -> None:
    """It changes the fit, not the population, so the two are directly comparable."""
    assert committed["summaries"]["equal_label_weights"]["labels"] == committed["summaries"]["baseline"]["labels"]
