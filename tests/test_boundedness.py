"""The boundedness measurement: what is guaranteed against what is merely observed.

The distinction this analysis exists to draw is easy to state and easy to lose.
A random forest averages training targets, so it *cannot* emit a value above
``max(y_train)``: that is arithmetic, and no data, tuning or feature set changes
it. A gradient booster sums tree outputs onto an initial estimate, and nothing in
that construction confines the sum, so its staying below the ceiling is an
observation about these rows rather than a property of the model.

The paper leans on exactly that asymmetry, and the risk is that the code quietly
treats the two the same. So these tests pin the guarantee where it exists, pin
that no guarantee is claimed where it does not, and check that the artifact
records enough for a reader to tell which is which.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import analysis_boundedness as ab
import hdb5


def _dataset(n_per_machine: int = 60, seed: int = 5) -> pd.DataFrame:
    """A prepared HDB5-shaped frame whose machines are ordered in size."""
    rng = np.random.default_rng(seed)
    frames = []
    for index, (machine, radius) in enumerate(
        {"A": 0.7, "B": 1.2, "C": 1.9, "D": 2.6, "E": 3.2}.items()
    ):
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
                    "SHOT": rng.integers(index * 10_000, index * 10_000 + n // 3, n),
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
    return hdb5.build_features(hdb5.map_to_canonical(pd.concat(frames, ignore_index=True)))


@pytest.fixture(scope="module")
def analysis() -> dict[str, object]:
    return ab.analyze_boundedness(_dataset(), min_rows=30)


def _rows(analysis: dict[str, object]) -> list[dict[str, Any]]:
    """The per-split records, typed.

    ``analyze_boundedness`` returns ``dict[str, object]`` because its top level
    mixes scalars, lists and nested dicts. Narrowing once here keeps the tests
    readable and avoids an ignore comment on every access.
    """
    rows = analysis["per_split"]
    assert isinstance(rows, list)
    return cast("list[dict[str, Any]]", rows)


# --- the guarantee, and the absence of one ---------------------------------


def test_only_the_forest_is_claimed_to_be_structurally_bounded() -> None:
    """The claim is about averaging, so it must not silently extend to boosting."""
    assert ab.STRUCTURALLY_BOUNDED == ("random_forest",)
    assert set(ab.STRUCTURALLY_BOUNDED) < set(ab.ENSEMBLES)
    assert "hist_gradient_boosting" in ab.ENSEMBLES
    assert "hist_gradient_boosting" not in ab.STRUCTURALLY_BOUNDED


def test_the_forest_never_exceeds_the_training_ceiling(analysis: dict[str, object]) -> None:
    """Averaging training targets cannot produce a value above the largest one."""
    forest = [row for row in _rows(analysis) if row["model_name"] == "random_forest"]
    assert forest, "no forest rows were measured"
    for row in forest:
        assert row["log_headroom_used"] <= 0.0, (
            f"the forest exceeded max(y_train) on {row['held_out']}, which averaging forbids"
        )
        assert row["fraction_predictions_above_train_max"] == 0.0


def test_a_bounded_model_breaking_the_bound_is_raised_rather_than_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarantee is asserted in code, not merely reported.

    If a refactor ever let a model declared structurally bounded exceed the
    ceiling, writing that to the artifact and carrying on would leave the
    paper's strongest structural claim contradicted by its own supporting file.

    Exercising the guard needs a model that actually overshoots, and the real
    ensembles do not oblige on demand, so one is injected. Waiting for a
    booster to overshoot by chance would make this test skip most runs, which
    is indistinguishable from not having written it.
    """

    class AlwaysAboveTheCeiling:
        """Predicts one natural log unit above the largest training target."""

        def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AlwaysAboveTheCeiling":
            self._ceiling = float(np.asarray(y, dtype=float).max())
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), self._ceiling + 1.0)

    monkeypatch.setattr(ab, "ENSEMBLES", ("random_forest",))
    monkeypatch.setattr(ab, "STRUCTURALLY_BOUNDED", ("random_forest",))
    monkeypatch.setattr(
        hdb5, "build_model_zoo", lambda: {"random_forest": AlwaysAboveTheCeiling()}
    )

    dataset = _dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    test_mask = labels == "E"

    with pytest.raises(AssertionError, match="which averaging forbids"):
        ab._measure(
            dataset,
            ~test_mask,
            test_mask,
            split="leave_one_machine_out",
            held_out="E",
            feature_columns=hdb5.BLIND_FEATURE_COLUMNS,
        )


def test_an_unbounded_model_exceeding_the_ceiling_is_recorded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The booster has no guarantee, so overshooting it is data, not a defect.

    This is the other half of the asymmetry. The same overshoot that must raise
    for the forest must be written to the artifact for the booster, because the
    paper reports its boundedness as measured rather than guaranteed.
    """

    class AlwaysAboveTheCeiling:
        def fit(self, X: pd.DataFrame, y: np.ndarray) -> "AlwaysAboveTheCeiling":
            self._ceiling = float(np.asarray(y, dtype=float).max())
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), self._ceiling + 1.0)

    monkeypatch.setattr(ab, "ENSEMBLES", ("hist_gradient_boosting",))
    monkeypatch.setattr(ab, "STRUCTURALLY_BOUNDED", ("random_forest",))
    monkeypatch.setattr(
        hdb5, "build_model_zoo", lambda: {"hist_gradient_boosting": AlwaysAboveTheCeiling()}
    )

    dataset = _dataset()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    test_mask = labels == "E"

    (row,) = ab._measure(
        dataset,
        ~test_mask,
        test_mask,
        split="leave_one_machine_out",
        held_out="E",
        feature_columns=hdb5.BLIND_FEATURE_COLUMNS,
    )
    assert row["structurally_bounded"] is False
    assert row["log_headroom_used"] == pytest.approx(1.0)
    assert row["fraction_predictions_above_train_max"] == 1.0


# --- what the artifact has to record ---------------------------------------


def test_both_splits_are_measured(analysis: dict[str, object]) -> None:
    """Leave-one-machine-out and the ITER-matched cut are different questions."""
    splits = {row["split"] for row in _rows(analysis)}
    assert splits == {"leave_one_machine_out", "iter_matched_cut"}


def test_every_row_carries_the_numbers_the_paper_quotes(analysis: dict[str, object]) -> None:
    required = {
        "split",
        "held_out",
        "model_name",
        "structurally_bounded",
        "log_train_target_max",
        "log_prediction_max",
        "log_headroom_used",
        "fraction_predictions_above_train_max",
        "best_shot_over_prediction_max",
    }
    for row in _rows(analysis):
        assert required <= set(row)


def test_headroom_is_the_difference_it_claims_to_be(analysis: dict[str, object]) -> None:
    """`log_headroom_used` must be prediction max minus training max, not an alias."""
    for row in _rows(analysis):
        assert row["log_headroom_used"] == pytest.approx(
            row["log_prediction_max"] - row["log_train_target_max"]
        )


def test_best_shot_ratio_is_expressed_in_linear_units(analysis: dict[str, object]) -> None:
    """The paper quotes it as "3.7x", so it must be a ratio rather than a log gap."""
    for row in _rows(analysis):
        assert row["best_shot_over_prediction_max"] > 0.0
        assert row["best_shot_over_prediction_max"] == pytest.approx(
            np.exp(row["log_test_target_max"] - row["log_prediction_max"])
        )


def test_the_committed_artifact_agrees_with_its_own_structure() -> None:
    """The shipped file must satisfy the invariants measured above."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "boundedness.json"
    if not path.exists():
        pytest.skip("no results/boundedness.json; run `python3 analysis_boundedness.py`")

    payload = json.loads(path.read_text())
    for row in payload["per_split"]:
        assert row["log_headroom_used"] == pytest.approx(
            row["log_prediction_max"] - row["log_train_target_max"]
        )
        if row["model_name"] == "random_forest":
            assert row["log_headroom_used"] <= 0.0
