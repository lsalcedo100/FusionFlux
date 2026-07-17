"""Holdout and cross-validation split selection for the training pipeline.

Handles the two split regimes FusionFlux supports: a plain randomized holdout,
and a row-targeted grouped holdout (an exact bounded subset-sum over group row
counts, with a linear greedy fallback for very large group sets) that keeps every
repeated ``shot_id`` intact across the train/test boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from config import (
    GROUP_COLUMN,
    HOLDOUT_TEST_SIZE,
    MAX_CV_FOLDS,
    MIN_CV_FOLDS,
    MIN_GROUPED_HOLDOUT_GROUPS,
    MIN_TEST_SAMPLES,
    MIN_TOTAL_SAMPLES,
    MIN_TRAIN_SAMPLES,
    RANDOM_STATE,
    TARGET_COLUMN,
)


def _group_holdout_total_score(total_rows: int, *, target_test_rows: int) -> tuple[int, int, int]:
    return (
        abs(total_rows - target_test_rows),
        0 if total_rows >= target_test_rows else 1,
        -total_rows,
    )


# Work bound on the exact subset-sum search: it fills up to
# ``len(group_counts) * (max_test_rows + 1)`` reachable-total cells. Above this
# a large real dataset (many shots and rows) would turn split selection into a
# Python-object bottleneck, so we fall back to a linear greedy fill instead.
MAX_GROUP_HOLDOUT_SUBSET_SUM_CELLS = 2_000_000

_GROUP_HOLDOUT_TOO_FEW_ROWS_MESSAGE = (
    "Grouped holdout could not find a test split with enough rows while keeping groups intact. "
    "Provide more shots before training."
)


def _greedy_group_holdout_positions(
    group_counts: list[int],
    *,
    target_test_rows: int,
    max_test_rows: int,
) -> tuple[int, ...]:
    """Linear fallback for large group sets: fill whole groups in the caller's
    (already shuffled) order until reaching ``target_test_rows`` without exceeding
    ``max_test_rows``. Deterministic for a given shuffle, and never splits a shot.
    """
    selected_positions: list[int] = []
    total_rows = 0
    for position, group_row_count in enumerate(group_counts):
        if total_rows >= target_test_rows:
            break
        if total_rows + group_row_count <= max_test_rows:
            selected_positions.append(position)
            total_rows += group_row_count
    if total_rows < MIN_TEST_SAMPLES:
        raise ValueError(_GROUP_HOLDOUT_TOO_FEW_ROWS_MESSAGE)
    return tuple(selected_positions)


def _select_group_holdout_positions(
    group_counts: list[int],
    *,
    target_test_rows: int,
    max_test_rows: int,
) -> tuple[int, ...]:
    """Choose whole groups whose combined row count best hits ``target_test_rows``.

    This is a bounded subset-sum over the group row counts: each group is used at
    most once (so no shot is split across train/test), and reachable totals are
    capped at ``max_test_rows`` to protect the minimum training size. Among the
    totals that also clear ``MIN_TEST_SAMPLES`` we keep the one whose
    ``_group_holdout_total_score`` is best (closest to the target, preferring a
    total at or above it). Groups are pre-shuffled by the caller, so the first
    subset discovered for each total reflects that random order. For very large
    group sets the exact search is replaced by ``_greedy_group_holdout_positions``.
    """
    estimated_cells = len(group_counts) * (max_test_rows + 1)
    if estimated_cells > MAX_GROUP_HOLDOUT_SUBSET_SUM_CELLS:
        return _greedy_group_holdout_positions(
            group_counts,
            target_test_rows=target_test_rows,
            max_test_rows=max_test_rows,
        )

    # Map every reachable test-row total to the group positions that produce it.
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for position, group_row_count in enumerate(group_counts):
        for total_rows, positions in list(reachable.items()):
            new_total = total_rows + group_row_count
            if new_total <= max_test_rows and new_total not in reachable:
                reachable[new_total] = (*positions, position)

    candidate_totals = [
        total_rows for total_rows in reachable if MIN_TEST_SAMPLES <= total_rows <= max_test_rows
    ]
    if not candidate_totals:
        raise ValueError(_GROUP_HOLDOUT_TOO_FEW_ROWS_MESSAGE)

    best_total = min(
        candidate_totals,
        key=lambda total_rows: _group_holdout_total_score(total_rows, target_test_rows=target_test_rows),
    )
    return reachable[best_total]


def _select_group_holdout_indices(
    df: pd.DataFrame,
    *,
    target_test_rows: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    group_sizes = df.groupby(GROUP_COLUMN, sort=False).size()
    group_names = list(group_sizes.index)
    group_counts = group_sizes.to_numpy(dtype=int)

    rng = np.random.default_rng(random_state)
    shuffled_order = rng.permutation(len(group_names))
    shuffled_groups = [group_names[index] for index in shuffled_order]
    shuffled_counts = [int(group_counts[index]) for index in shuffled_order]

    max_test_rows = len(df) - MIN_TRAIN_SAMPLES
    selected_positions = _select_group_holdout_positions(
        shuffled_counts,
        target_test_rows=target_test_rows,
        max_test_rows=max_test_rows,
    )
    selected_groups = {shuffled_groups[position] for position in selected_positions}
    test_mask = df[GROUP_COLUMN].isin(selected_groups).to_numpy(dtype=bool)
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    return train_idx, test_idx


def select_split_indices(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray, str]:
    sample_count = len(df)
    if sample_count < MIN_TOTAL_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TOTAL_SAMPLES} samples to produce a trustworthy holdout; found {sample_count}."
        )

    test_size = max(HOLDOUT_TEST_SIZE, MIN_TEST_SAMPLES / sample_count)
    test_count = int(np.ceil(sample_count * test_size))
    train_count = sample_count - test_count
    if train_count < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAIN_SAMPLES} training rows after holdout; got {train_count} from {sample_count} samples."
        )

    unique_groups = df[GROUP_COLUMN].nunique(dropna=True) if GROUP_COLUMN in df.columns else 0
    has_repeated_groups = GROUP_COLUMN in df.columns and 0 < unique_groups < len(df)
    if has_repeated_groups:
        if unique_groups < MIN_GROUPED_HOLDOUT_GROUPS:
            raise ValueError(
                f"Need at least {MIN_GROUPED_HOLDOUT_GROUPS} unique {GROUP_COLUMN} values for grouped holdout; "
                f"found {unique_groups}."
            )
        train_idx, test_idx = _select_group_holdout_indices(
            df,
            target_test_rows=test_count,
            random_state=random_state,
        )
        if len(train_idx) < MIN_TRAIN_SAMPLES or len(test_idx) < MIN_TEST_SAMPLES:
            raise ValueError(
                "Grouped holdout left too few rows for training or evaluation. Provide more shots before training."
            )
        return train_idx, test_idx, "group_row_target_split"

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    return np.asarray(train_idx), np.asarray(test_idx), "random_split"


def validate_training_frame(df: pd.DataFrame, candidate_feature_columns: list[str]) -> None:
    if not candidate_feature_columns:
        raise ValueError("No model features are available after dataset preparation.")
    if len(df) < MIN_TOTAL_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TOTAL_SAMPLES} samples to produce a trustworthy holdout; found {len(df)}."
        )
    if df[TARGET_COLUMN].nunique(dropna=True) < 2:
        raise ValueError("Training target must contain at least two distinct values.")
    if GROUP_COLUMN in df.columns and df[GROUP_COLUMN].nunique(dropna=True) == 1 and len(df) > 1:
        raise ValueError(f"Need more than one unique {GROUP_COLUMN} value to build a trustworthy holdout.")


def build_cv_splits(
    train_frame: pd.DataFrame,
    split_strategy: str,
    random_state: int = RANDOM_STATE,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, int]:
    if split_strategy == "group_row_target_split":
        group_count = int(train_frame[GROUP_COLUMN].nunique(dropna=True))
        fold_count = min(MAX_CV_FOLDS, group_count)
        if fold_count < MIN_CV_FOLDS:
            raise ValueError(
                f"Need at least {MIN_CV_FOLDS} unique {GROUP_COLUMN} values in the training fold for grouped CV; "
                f"found {group_count}."
            )
        splitter = GroupKFold(n_splits=fold_count)
        splits = list(splitter.split(train_frame, groups=train_frame[GROUP_COLUMN]))
        return splits, "group_k_fold", fold_count

    fold_count = min(MAX_CV_FOLDS, len(train_frame))
    if fold_count < MIN_CV_FOLDS:
        raise ValueError(f"Need at least {MIN_CV_FOLDS} training rows for cross-validation; found {len(train_frame)}.")
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=random_state)
    return list(splitter.split(train_frame)), "k_fold", fold_count
