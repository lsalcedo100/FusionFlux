"""The nested tuning, and the one property that makes it worth running.

This module exists to answer "the trees lost because nobody tuned them". It
answers it only if the tuning is honest, and there is exactly one way for it to
be dishonest without raising: letting the held-out unit take part in choosing a
hyperparameter. A discharge that appears in an inner fold and again in the outer
test set, or a machine tuned on itself, turns leave-one-machine-out into
something that is not leave-one-machine-out, and the result would then be an
argument for the opposite conclusion.

So the inner splits are pinned directly, in both selection modes, and
``_tune_and_fit`` is made to record every row it fits on so the containment can
be asserted rather than reasoned about.

Nothing here needs the dataset. The orchestration test drives ``analyze_tuned``
with the estimator factory patched to small models, which keeps it to a few
seconds while still exercising the real splitting, scoring and output schema;
``_build`` is checked separately, so the real estimators are not taken on trust.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import analysis_tuned as at
import hdb5


def _make_dataset(n_per_machine: int = 60, seed: int = 17) -> pd.DataFrame:
    """A prepared HDB5-shaped dataset with four machines spanning a size range."""
    machines = {"S1": 0.6, "S2": 0.9, "M1": 1.4, "L1": 2.8}
    rng = np.random.default_rng(seed)
    frames = []
    for index, (machine, radius) in enumerate(machines.items()):
        n = n_per_machine
        ip = rng.uniform(0.4, 4.0, n)
        bt = rng.uniform(1.0, 5.0, n)
        nel = rng.uniform(1.5, 20.0, n)
        plth = rng.uniform(0.5, 25.0, n)
        rgeo = radius * rng.uniform(0.97, 1.03, n)
        eps = rng.uniform(0.25, 0.35, n)
        kappa = rng.uniform(1.1, 2.2, n)
        meff = rng.uniform(1.0, 3.0, n)
        tau = (
            0.0562 * ip**0.93 * bt**0.15 * nel**0.41 * plth**-0.69
            * rgeo**1.97 * eps**0.58 * kappa**0.78 * meff**0.19
        ) * np.exp(rng.normal(0.0, 0.08, n))
        frames.append(
            pd.DataFrame(
                {
                    "TOK": machine,
                    "SHOT": rng.integers(index * 10_000, index * 10_000 + n // 2, n),
                    "TIME": rng.uniform(1.0, 5.0, n),
                    "TAUTH": tau,
                    "IP": ip,
                    "BT": bt,
                    "NEL": nel,
                    "PLTH": plth,
                    "RGEO": rgeo,
                    "DELTA1": rng.uniform(0.1, 0.5, n),
                    "KAPPAA": kappa,
                    "EPS": eps,
                    "MEFF": meff,
                }
            )
        )
    raw = pd.concat(frames, ignore_index=True)
    prepared = hdb5.build_features(hdb5.map_to_canonical(raw))
    # Positions and index labels must agree: _tune_and_fit selects rows with
    # .iloc, and the recording model below reads them back off the frame index.
    return prepared.reset_index(drop=True)


def _xy(dataset: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    return features, np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))


# --- the scoring and grid helpers -------------------------------------------


def test_rmsle_is_the_root_mean_square_of_the_log_residuals() -> None:
    actual = np.array([0.0, 1.0, 2.0])
    predicted = np.array([1.0, 1.0, 0.0])
    assert at._rmsle_log(actual, predicted) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_rmsle_is_zero_for_a_perfect_prediction() -> None:
    values = np.array([0.3, -1.2, 4.4])
    assert at._rmsle_log(values, values) == 0.0


def test_the_grid_expands_to_every_combination_exactly_once() -> None:
    grid: dict[str, list[Any]] = {"a": [1, 2, 3], "b": ["x", "y"]}
    configs = at._configs(grid)
    assert len(configs) == 6
    assert {tuple(sorted(c.items(), key=str)) for c in configs} == {
        tuple(sorted({"a": a, "b": b}.items(), key=str)) for a in grid["a"] for b in grid["b"]
    }
    assert all(set(c) == {"a", "b"} for c in configs)


def test_the_published_grids_are_small_enough_to_rerun_and_wide_enough_to_move_a_model() -> None:
    """Both claims in the module docstring, so neither can be quietly dropped."""
    assert len(at._configs(at.RF_GRID)) == 6
    assert len(at._configs(at.HGB_GRID)) == 4
    assert all(len(values) > 1 for values in at.RF_GRID.values())
    assert all(len(values) > 1 for values in at.HGB_GRID.values())


# --- the estimators ---------------------------------------------------------


def test_the_forest_is_built_with_the_config_and_the_pinned_seed() -> None:
    model = at._build("random_forest", {"max_features": 0.5, "min_samples_leaf": 5})
    assert isinstance(model, RandomForestRegressor)
    assert model.max_features == 0.5
    assert model.min_samples_leaf == 5
    assert model.random_state == hdb5.RANDOM_STATE


def test_anything_that_is_not_the_forest_is_the_boosting_model() -> None:
    model = at._build("hist_gradient_boosting", {"learning_rate": 0.05, "max_leaf_nodes": 15})
    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.learning_rate == 0.05
    assert model.max_leaf_nodes == 15
    assert model.random_state == hdb5.RANDOM_STATE


# --- the inner splits, which are the whole point ----------------------------


def test_machine_mode_holds_out_one_whole_machine_per_fold() -> None:
    """Model selection matched to deployment: the inner fold is a machine."""
    labels = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    train_rows = np.arange(len(labels))
    folds = at._inner_folds(labels, train_rows, np.zeros(len(labels)), "machine")

    assert len(folds) == 3
    for inner_train, inner_test in folds:
        held = set(labels[train_rows[inner_test]])
        assert len(held) == 1, "an inner fold held out more than one machine"
        assert held.isdisjoint(set(labels[train_rows[inner_train]])), (
            "the held-out machine was also used to fit"
        )


def test_a_machine_too_small_to_score_is_not_made_an_inner_fold() -> None:
    """Mirrors hdb5.MIN_HELD_OUT_ROWS: below it the score is noise about a handful of rows."""
    small = hdb5.MIN_HELD_OUT_ROWS - 1
    labels = np.array(["A"] * 40 + ["B"] * 40 + ["tiny"] * small)
    train_rows = np.arange(len(labels))
    folds = at._inner_folds(labels, train_rows, np.zeros(len(labels)), "machine")

    held = {next(iter(set(labels[train_rows[test]]))) for _, test in folds}
    assert held == {"A", "B"}
    assert "tiny" not in held


def test_discharge_mode_never_splits_a_discharge_across_a_fold() -> None:
    discharges = np.array([f"shot{i // 4}" for i in range(120)])
    train_rows = np.arange(len(discharges))
    folds = at._inner_folds(discharges, train_rows, np.zeros(len(discharges)), "discharge")

    assert len(folds) == at.N_INNER_FOLDS
    for inner_train, inner_test in folds:
        left = set(discharges[train_rows[inner_train]])
        right = set(discharges[train_rows[inner_test]])
        assert left.isdisjoint(right), "a discharge appeared on both sides of an inner fold"


def test_the_fold_count_falls_back_to_the_number_of_groups_available() -> None:
    """Two discharges cannot make three folds, and asking for them would raise."""
    discharges = np.array(["a"] * 20 + ["b"] * 20)
    folds = at._inner_folds(discharges, np.arange(40), np.zeros(40), "discharge")
    assert len(folds) == 2


def test_the_inner_folds_are_positions_into_the_training_rows_not_the_dataset() -> None:
    """The offset that would silently tune on the wrong rows if it were wrong."""
    labels = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    train_rows = np.arange(40, 120)  # machine A withheld by the outer split
    folds = at._inner_folds(labels, train_rows, np.zeros(len(labels)), "machine")

    assert len(folds) == 2
    for inner_train, inner_test in folds:
        assert set(labels[train_rows[inner_test]]) <= {"B", "C"}
        assert "A" not in set(labels[train_rows[inner_train]])


# --- containment: the held-out rows never reach a fit ------------------------


class _RecordingModel:
    """Stands in for an estimator and remembers which rows it was fitted on."""

    def __init__(self, config: dict[str, Any], seen: set[int]) -> None:
        self.config = config
        self._seen = seen
        self._mean = 0.0

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> "_RecordingModel":
        self._seen.update(int(label) for label in features.index)
        self._mean = float(np.mean(target))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self._mean, dtype=float)


def test_tuning_never_fits_on_a_row_outside_the_training_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the whole result depends on, asserted rather than reasoned about.

    If a held-out row reached an inner fit, the hyperparameter would have been
    chosen with knowledge of the rows it is later scored on, and
    leave-one-machine-out would no longer be leave-one-machine-out.
    """
    dataset = _make_dataset()
    features, log_target = _xy(dataset)
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    held_out = labels == "L1"
    train_rows = np.flatnonzero(~held_out)

    seen: set[int] = set()
    monkeypatch.setattr(at, "_build", lambda name, config: _RecordingModel(config, seen))

    at._tune_and_fit(
        "random_forest", at.RF_GRID, features, log_target, groups, train_rows,
        inner_unit=labels, mode="machine",
    )

    assert seen, "nothing was fitted; the assertion below would pass vacuously"
    assert seen <= set(train_rows.tolist()), "a row outside the training fold was fitted on"
    assert not (seen & set(np.flatnonzero(held_out).tolist()))


def test_the_chosen_configuration_is_the_one_that_scored_best_inside_the_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset()
    features, log_target = _xy(dataset)
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    train_rows = np.arange(len(dataset))

    scores = {"1": 0.5, "5": 0.1}

    class _ScriptedModel(_RecordingModel):
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            offset = scores[str(self.config["min_samples_leaf"])]
            return np.full(len(features), self._mean + offset, dtype=float)

    seen: set[int] = set()
    monkeypatch.setattr(at, "_build", lambda name, config: _ScriptedModel(config, seen))

    _, record = at._tune_and_fit(
        "random_forest", at.RF_GRID, features, log_target, groups, train_rows
    )
    assert record["config"]["min_samples_leaf"] == "5", "the worse configuration was chosen"
    assert record["inner_rmsle"] > 0.0
    assert set(record["config"]) == set(at.RF_GRID)


# --- the report the paper reads ---------------------------------------------


@pytest.fixture(scope="module")
def analysis() -> dict[str, Any]:
    """`analyze_tuned` end to end, with small estimators so it costs seconds.

    `_build` is patched here and asserted separately above, so the real
    estimators are still covered; what this exercises is the splitting, the
    scoring and the shape of the result.
    """
    def _small(name: str, config: dict[str, Any]) -> Any:
        if name == "random_forest":
            return RandomForestRegressor(n_estimators=5, random_state=hdb5.RANDOM_STATE, **config)
        return HistGradientBoostingRegressor(
            max_iter=10, random_state=hdb5.RANDOM_STATE, **config
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(at, "_build", _small)
        return at.analyze_tuned(_make_dataset())


def test_the_report_carries_both_ensembles_and_every_split_the_paper_quotes(
    analysis: dict[str, Any],
) -> None:
    assert set(analysis) == {"n_rows", "n_inner_folds", "grids", "tuned"}
    assert analysis["n_inner_folds"] == at.N_INNER_FOLDS
    assert set(analysis["tuned"]) == {"random_forest", "hist_gradient_boosting"}
    for row in analysis["tuned"].values():
        assert {
            "cv",
            "leave_one_machine_out",
            "leave_one_machine_out_inner_machine",
            "iter_matched_cut",
            "per_machine",
            "chosen_configurations",
        } <= set(row)
        assert all(np.isfinite(row[key]) for key in ("cv", "iter_matched_cut"))


def test_both_selection_procedures_are_reported_and_cover_the_same_machines(
    analysis: dict[str, Any],
) -> None:
    """The comparison the module is for: tuning by discharge against tuning by machine."""
    eligible = set(hdb5.eligible_tokamaks(_make_dataset(), min_rows=hdb5.MIN_HELD_OUT_ROWS))
    for row in analysis["tuned"].values():
        per_machine = row["per_machine"]
        assert set(per_machine) == {"discharge", "machine"}
        assert set(per_machine["discharge"]) == eligible
        assert set(per_machine["machine"]) == eligible
        assert row["leave_one_machine_out"] == pytest.approx(
            float(np.mean(list(per_machine["discharge"].values())))
        )
        assert row["leave_one_machine_out_inner_machine"] == pytest.approx(
            float(np.mean(list(per_machine["machine"].values())))
        )


def test_every_tuning_decision_is_recorded_against_the_split_that_made_it(
    analysis: dict[str, Any],
) -> None:
    """Without the record, a reader cannot tell what was actually fitted where."""
    eligible = hdb5.eligible_tokamaks(_make_dataset(), min_rows=hdb5.MIN_HELD_OUT_ROWS)
    for row in analysis["tuned"].values():
        chosen = row["chosen_configurations"]
        splits = [record["split"] for record in chosen]
        assert splits.count("iter_matched_cut") == 1
        assert splits.count("cv") >= 2
        for machine in eligible:
            assert splits.count(f"lomo:{machine}") == 2, "one record per inner mode"
        modes = {record.get("inner") for record in chosen if record["split"].startswith("lomo:")}
        assert modes == {"discharge", "machine"}
        assert all("inner_rmsle" in record for record in chosen)


def test_the_grids_are_reported_as_strings_so_the_json_round_trips(
    analysis: dict[str, Any],
) -> None:
    """`max_features` holds floats and the string "sqrt"; JSON needs one type."""
    grids = analysis["grids"]
    assert set(grids) == {"random_forest", "hist_gradient_boosting"}
    assert all(isinstance(v, str) for values in grids["random_forest"].values() for v in values)
    assert "sqrt" in grids["random_forest"]["max_features"]
