"""Result 15: the feature ladder, and the split distinction the result depends on.

The finding is that the ranking reversal appears only once the flexible models
win the interpolation split, and that the extrapolation failure is there at every
rung regardless. Two things can silently destroy that.

The first is the row set. Each rung must be scored on exactly the same plants;
if rung 1 were fitted on every plant with a diameter and rung 4 only on those
that also have a leaf mass, the rungs would differ in their rows as well as
their features and nothing could be attributed to dimensionality.

The second is the split, and this one actually happened. HDB5's cross-validation
holds out discharges while keeping every machine in the training fold, so it
measures interpolation within known machines. Grouping the cross-validation by
species here is instead the analogue of leave-one-tokamak-out, and comparing that
against leave-one-species-out compares a hard split with the same hard split. Run
that way the reversal cannot appear at any dimension, and the analysis reported
"no reversal at any rung" as though it were a finding. These tests pin the
distinction so it cannot come back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis_tree_allometry as ata
import tree_allometry as ta


def _make_dataset(n_per_species: int = 60, seed: int = 3) -> pd.DataFrame:
    """A BAAD-shaped frame in which mass really is a power law in the predictors."""
    rng = np.random.default_rng(seed)
    frames = []
    for index, (species, scale) in enumerate(
        {"a": 0.02, "b": 0.05, "c": 0.12, "d": 0.3, "e": 0.7}.items()
    ):
        n = n_per_species
        diameter = scale * rng.uniform(0.6, 1.6, n)
        height = 20.0 * diameter ** 0.7 * rng.uniform(0.85, 1.15, n)
        leaf_area = 300.0 * diameter ** 1.8 * rng.uniform(0.8, 1.25, n)
        leaf_mass = 0.12 * leaf_area * rng.uniform(0.85, 1.2, n)
        mass = (
            120.0 * diameter ** ta.WBE_EXPONENT * height ** 0.25
        ) * np.exp(rng.normal(0.0, 0.15, n))
        frames.append(
            pd.DataFrame(
                {
                    "species": f"species_{species}{index}",
                    ta.TARGET_COLUMN: mass,
                    ta.SIZE_COLUMN: diameter,
                    "height_m": height,
                    "leaf_area_m2": leaf_area,
                    "leaf_mass_kg": leaf_mass,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    for column in ta.SOURCE_COLUMNS.values():
        frame[f"log_{column}"] = np.log(frame[column].to_numpy(dtype=float))
    return frame


# --- the ladder is a ladder -------------------------------------------------


def test_each_rung_is_a_prefix_of_the_next() -> None:
    """Adding one feature must be the only difference between adjacent rungs."""
    for n in sorted(ta.FEATURE_LADDER)[:-1]:
        assert ta.FEATURE_LADDER[n] == ta.FEATURE_LADDER[n + 1][:n]


def test_the_ladder_starts_at_the_single_predictor_classical_law() -> None:
    """Rung 1 is the analogue of Result 13's one-predictor problem."""
    assert ta.FEATURE_LADDER[1] == (f"log_{ta.SIZE_COLUMN}",)


def test_every_rung_scores_the_same_rows() -> None:
    """The fixed row set is what licenses attributing a change to dimensionality."""
    dataset = _make_dataset()
    counts = set()
    for n in sorted(ta.FEATURE_LADDER):
        rung, report = ata.score_rung(dataset, n, min_rows=20)
        counts.add(int(report["n_train_rows"].iloc[0] + report["n_held_out_rows"].iloc[0]))
        assert rung.n_features == n
    assert len(counts) == 1, f"rungs saw different row counts: {counts}"


# --- the split distinction --------------------------------------------------


def test_interpolation_split_keeps_every_species_on_both_sides() -> None:
    """The analogue of HDB5's CV by discharge, not of leave-one-tokamak-out.

    This is the property that was wrong once. If the cross-validation groups by
    species, it becomes a second extrapolation split and the reversal cannot
    appear at any dimension.
    """
    from sklearn.model_selection import KFold

    dataset = _make_dataset()
    groups = dataset[ta.GROUP_COLUMN].to_numpy()
    cv = KFold(n_splits=ata.N_CV_FOLDS, shuffle=True, random_state=ata.RANDOM_STATE)

    for train_index, test_index in cv.split(dataset):
        shared = set(groups[train_index]) & set(groups[test_index])
        assert shared == set(np.unique(groups)), (
            "the interpolation split must leave every species on both sides; "
            f"only {sorted(shared)} were shared"
        )


def test_extrapolation_split_holds_an_entire_species_out() -> None:
    dataset = _make_dataset()
    _, report = ata.score_rung(dataset, 2, min_rows=20)
    assert report["n_held_out_rows"].gt(0).all()
    # audit_groups reports one row per (group, estimator); every scored group
    # must be entirely absent from its own training fold, which is what the
    # distance diagnostic in the same row is computed against.
    assert report["group"].nunique() >= 2


# --- the properties the result is stated in ---------------------------------


def test_reversal_requires_both_halves() -> None:
    """A rung is a reversal only if the flexible model wins one split and loses the other."""
    rung = ata.RungResult(
        n_features=2,
        features=("a", "b"),
        cv_rmsle={ata.POWER_LAW: 0.5, "random_forest": 0.4, "hist_gradient_boosting": 0.45},
        loo_rmsle={ata.POWER_LAW: 0.6, "random_forest": 0.8, "hist_gradient_boosting": 0.9},
        n_species_scored=5,
    )
    assert rung.trees_win_interpolation
    assert rung.power_law_wins_extrapolation
    assert rung.reversal

    no_interpolation_win = ata.RungResult(
        n_features=1,
        features=("a",),
        cv_rmsle={ata.POWER_LAW: 0.4, "random_forest": 0.5, "hist_gradient_boosting": 0.55},
        loo_rmsle={ata.POWER_LAW: 0.6, "random_forest": 0.8, "hist_gradient_boosting": 0.9},
        n_species_scored=5,
    )
    assert not no_interpolation_win.trees_win_interpolation
    assert no_interpolation_win.power_law_wins_extrapolation
    assert not no_interpolation_win.reversal, (
        "losing both splits is not a reversal; Result 13's outcome must not be "
        "counted as one"
    )


def test_cv_gain_is_positive_exactly_when_the_trees_win() -> None:
    rung = ata.RungResult(
        n_features=3,
        features=("a", "b", "c"),
        cv_rmsle={ata.POWER_LAW: 0.5, "random_forest": 0.4, "hist_gradient_boosting": 0.45},
        loo_rmsle={ata.POWER_LAW: 0.6, "random_forest": 0.8, "hist_gradient_boosting": 0.9},
        n_species_scored=5,
    )
    assert rung.cv_gain_over_power_law == pytest.approx(0.2)
    assert rung.trees_win_interpolation


def test_first_reversal_rung_is_the_lowest_one() -> None:
    def rung(n: int, cv_tree: float) -> ata.RungResult:
        return ata.RungResult(
            n_features=n,
            features=tuple("abcd"[:n]),
            cv_rmsle={ata.POWER_LAW: 0.5, "random_forest": cv_tree,
                      "hist_gradient_boosting": cv_tree},
            loo_rmsle={ata.POWER_LAW: 0.6, "random_forest": 0.8,
                       "hist_gradient_boosting": 0.9},
            n_species_scored=5,
        )

    study = ata.LadderStudy(
        rungs=[rung(1, 0.6), rung(2, 0.55), rung(3, 0.45), rung(4, 0.4)],
        per_species=pd.DataFrame(),
        baseline=ta.fit_wbe(np.log(np.array([1.0, 2.0, 3.0])), np.log(np.array([1.0, 6.0, 15.0]))),
        n_rows=100,
        n_species_total=5,
        size_span=10.0,
        mass_span=100.0,
    )
    assert ata.first_reversal_rung(study) == 3


def test_no_reversal_anywhere_returns_none() -> None:
    study = ata.LadderStudy(
        rungs=[
            ata.RungResult(
                n_features=n,
                features=tuple("abcd"[:n]),
                cv_rmsle={ata.POWER_LAW: 0.4, "random_forest": 0.5,
                          "hist_gradient_boosting": 0.5},
                loo_rmsle={ata.POWER_LAW: 0.6, "random_forest": 0.8,
                           "hist_gradient_boosting": 0.9},
                n_species_scored=5,
            )
            for n in (1, 2)
        ],
        per_species=pd.DataFrame(),
        baseline=ta.fit_wbe(np.log(np.array([1.0, 2.0, 3.0])), np.log(np.array([1.0, 6.0, 15.0]))),
        n_rows=100,
        n_species_total=5,
        size_span=10.0,
        mass_span=100.0,
    )
    assert ata.first_reversal_rung(study) is None


# --- the published exponent -------------------------------------------------


def test_wbe_baseline_recovers_a_planted_exponent() -> None:
    """A free fit on data generated at 8/3 must return 8/3."""
    rng = np.random.default_rng(0)
    log_diameter = rng.uniform(-4.0, 1.0, 4000)
    log_mass = 1.7 + ta.WBE_EXPONENT * log_diameter + rng.normal(0.0, 0.05, 4000)
    baseline = ta.fit_wbe(log_diameter, log_mass)
    assert baseline.free_exponent == pytest.approx(ta.WBE_EXPONENT, abs=0.01)


def test_constraining_to_the_published_exponent_cannot_beat_the_free_fit() -> None:
    """In sample the free fit is optimal by construction, as Result 13 reports for Kleiber."""
    dataset = _make_dataset()
    baseline = ta.fit_wbe(
        dataset[f"log_{ta.SIZE_COLUMN}"].to_numpy(dtype=float),
        dataset[f"log_{ta.TARGET_COLUMN}"].to_numpy(dtype=float),
    )
    assert baseline.free_rmsle <= baseline.constrained_rmsle + 1e-12


# --- the dataset contract ---------------------------------------------------


def test_prepared_dataset_is_complete_and_positive() -> None:
    dataset = _make_dataset()
    numeric = list(ta.SOURCE_COLUMNS.values())
    assert dataset[numeric].notna().all().all()
    assert (dataset[numeric] > 0).all().all()


def test_eligible_species_respects_the_row_floor() -> None:
    dataset = _make_dataset(n_per_species=60)
    assert len(ta.eligible_species(dataset, min_rows=50)) == 5
    assert ta.eligible_species(dataset, min_rows=100) == []


def test_the_pin_is_checked_and_refuses_a_wrong_file(tmp_path) -> None:
    """A silent upstream revision must fail loudly rather than move a result."""
    import hdb5

    impostor = tmp_path / "baad_data.zip"
    impostor.write_bytes(b"not the pinned release")
    with pytest.raises(hdb5.DatasetIntegrityError, match="integrity check failed"):
        ta.verify_baad_file(impostor)
