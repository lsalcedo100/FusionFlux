"""Do the two implicit choices in the headline comparison carry it?

``analysis_referee_robustness`` sweeps the row threshold that decides which
labels are scored, and repeats grouped cross-validation over shuffled fold
assignments. Both are cheap controls on choices that are defensible but not
forced, and the paper quotes their outcome.

These tests pin the parts that would silently invalidate the sweep rather than
re-deriving the numbers: that the threshold only changes which labels are
*scored* and never which are trained on, that the sign test is exact, and that
the shuffled partitions really are different partitions rather than the
deterministic one repeated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

import analysis_referee_robustness as arr
import hdb5

RESULTS = Path(__file__).resolve().parents[1] / "results" / "referee_robustness.json"


def _dataset_or_skip() -> pd.DataFrame:
    """The real STD5 rows, or a skip.

    CI does not have this deposit: it is third-party data that this repository
    pins and verifies but does not redistribute. Every sibling test that needs
    the real rows skips the same way.
    """
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return hdb5.prepare_dataset()


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip(
            "no results/referee_robustness.json; run `python3 analysis_referee_robustness.py`"
        )
    return json.loads(RESULTS.read_text())


# --- the exact sign test, against values worked out by hand ----------------


def test_sign_test_is_exact() -> None:
    # All 13 of 13 one way: 2 * (1/2)^13 * 1 = 2.44e-4.
    assert arr._exact_two_sided_p(13, 13) == pytest.approx(2.0 / 2**13)
    assert arr._exact_two_sided_p(0, 13) == pytest.approx(2.0 / 2**13)
    # 11 of 11: 2 * (1/2)^11 = 9.77e-4.
    assert arr._exact_two_sided_p(11, 11) == pytest.approx(2.0 / 2**11)
    # An even split cannot be evidence of anything.
    assert arr._exact_two_sided_p(5, 10) == pytest.approx(1.0)
    # Symmetric in which side wins.
    assert arr._exact_two_sided_p(3, 10) == pytest.approx(arr._exact_two_sided_p(7, 10))


# --- the threshold governs scoring, never training -------------------------


def test_threshold_changes_only_which_labels_are_scored() -> None:
    """Every label stays in every training fold at every threshold.

    This is the property that makes the sweep a control rather than four
    different experiments: if lowering the threshold also added training rows,
    the rows of the table would not be comparable.
    """
    dataset = _dataset_or_skip()
    all_labels = set(dataset[hdb5.TOKAMAK_LABEL_COLUMN].astype(str))
    for threshold in arr.ROW_THRESHOLDS:
        scored = set(hdb5.eligible_tokamaks(dataset, min_rows=threshold))
        assert scored <= all_labels
        # Every label the threshold excludes from scoring is still trainable:
        # leave-one-label-out trains on every other row in the database,
        # so the complement of the held-out label is the training set.
        for label in scored:
            trained_on = all_labels - {label}
            assert (all_labels - scored) <= trained_on


def test_lower_threshold_scores_a_superset_of_labels() -> None:
    dataset = _dataset_or_skip()
    previous: set[str] | None = None
    for threshold in sorted(arr.ROW_THRESHOLDS, reverse=True):
        scored = set(hdb5.eligible_tokamaks(dataset, min_rows=threshold))
        if previous is not None:
            assert previous <= scored, f"threshold {threshold} dropped a label"
        previous = scored


# --- the shuffled partitions are actually distinct -------------------------


def test_shuffled_partitions_differ_from_each_other_and_from_the_default() -> None:
    """A seed that silently did nothing would make the control vacuous."""
    dataset = _dataset_or_skip()
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    def fold_labels(splitter: GroupKFold) -> np.ndarray:
        assignment = np.empty(len(dataset), dtype=int)
        for fold, (_, test_index) in enumerate(splitter.split(features, target, groups)):
            assignment[test_index] = fold
        return assignment

    deterministic = fold_labels(GroupKFold(n_splits=hdb5.N_CV_FOLDS))
    seen = [deterministic]
    for seed in arr.CV_SEEDS[:4]:
        shuffled = fold_labels(
            GroupKFold(n_splits=hdb5.N_CV_FOLDS, shuffle=True, random_state=seed)
        )
        assert any(not np.array_equal(shuffled, other) for other in seen)
        seen.append(shuffled)


def test_shuffling_never_splits_a_discharge_across_folds() -> None:
    """The whole point of grouping survives shuffling, or the control is invalid."""
    dataset = _dataset_or_skip()
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    splitter = GroupKFold(n_splits=hdb5.N_CV_FOLDS, shuffle=True, random_state=arr.CV_SEEDS[0])
    for train_index, test_index in splitter.split(features, target, groups):
        assert not set(groups[train_index]) & set(groups[test_index])


# --- the committed report says what the paper quotes ------------------------


def test_committed_report_agrees_with_the_paper(committed: dict) -> None:
    rows = {row["min_rows"]: row for row in committed["threshold_sweep"]}
    assert set(rows) == set(arr.ROW_THRESHOLDS)

    # The reported threshold, and the count the paper quotes for it.
    reported = rows[hdb5.MIN_HELD_OUT_ROWS]
    assert reported["n_machines_scored"] == 13
    assert reported["n_forest_worse_than_power_law"] == 13

    # The power law wins at every threshold except one label at the lowest.
    for threshold, row in rows.items():
        missed = row["n_machines_scored"] - row["n_forest_worse_than_power_law"]
        assert missed <= 1, f"threshold {threshold}: forest won {missed} labels"
        assert row["exact_two_sided_p"] < 1e-3
        assert row["mean_gap_forest_minus_power_law"] > 0.0

    repeated = committed["repeated_grouped_cv"]
    assert repeated["forest_wins_every_partition"] is True
    assert repeated["min_margin"] > 0.0
    # The deterministic partition is not wildly outside the shuffled spread.
    assert abs(repeated["deterministic_partition_margin"] - repeated["mean_margin"]) < 0.05
