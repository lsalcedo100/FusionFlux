from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

import hdb5


def _make_fake_hdb5(n_rows: int = 600, seed: int = 7) -> pd.DataFrame:
    """Build a raw-HDB5-shaped frame whose TAUTH follows IPB98 plus noise.

    Uses the real raw column names and signed current/field so the mapping,
    unit handling, and learnability all get exercised without the large
    committed dataset.
    """
    rng = np.random.default_rng(seed)
    ip = rng.uniform(0.4, 4.0, n_rows)
    bt = rng.uniform(1.0, 5.0, n_rows)
    nel = rng.uniform(1.5, 20.0, n_rows)
    plth = rng.uniform(0.5, 25.0, n_rows)
    rgeo = rng.uniform(0.5, 3.2, n_rows)
    eps = rng.uniform(0.2, 0.7, n_rows)
    kappa = rng.uniform(1.1, 2.2, n_rows)
    meff = rng.uniform(1.0, 3.0, n_rows)
    tau_true = (
        0.0562
        * ip**0.93
        * bt**0.15
        * nel**0.41
        * plth**-0.69
        * rgeo**1.97
        * eps**0.58
        * kappa**0.78
        * meff**0.19
    )
    tau = tau_true * np.exp(rng.normal(0.0, 0.12, n_rows))
    shots = rng.integers(0, n_rows // 2, n_rows)
    signs = rng.choice([-1.0, 1.0], n_rows)
    return pd.DataFrame(
        {
            "TOK": rng.choice(["JET", "AUG", "D3D"], n_rows),
            "SHOT": shots,
            "TIME": rng.uniform(1.0, 5.0, n_rows),
            "TAUTH": tau,
            "IP": ip * signs,  # signed on purpose
            "BT": bt * rng.choice([-1.0, 1.0], n_rows),
            "NEL": nel,
            "PLTH": plth,
            "RGEO": rgeo,
            "DELTA1": rng.uniform(0.1, 0.5, n_rows),
            "KAPPAA": kappa,
            "EPS": eps,
            "MEFF": meff,
        }
    )


def test_map_to_canonical_applies_units_and_abs() -> None:
    raw = _make_fake_hdb5(50)
    canonical = hdb5.map_to_canonical(raw)

    # signed current/field become magnitudes
    assert (canonical["ip_ma"] > 0).all()
    assert (canonical["bt_t"] > 0).all()
    np.testing.assert_allclose(canonical["ip_ma"], raw["IP"].abs(), rtol=1e-9)
    # minor radius derived from inverse aspect ratio * major radius
    np.testing.assert_allclose(
        canonical["a_m"], raw["EPS"] * raw["RGEO"], rtol=1e-9
    )
    assert canonical[hdb5.TARGET_COLUMN].gt(0).all()
    assert canonical[hdb5.GROUP_COLUMN].str.contains("::").all()


def test_map_to_canonical_drops_nonpositive_rows() -> None:
    raw = _make_fake_hdb5(30)
    raw.loc[0, "NEL"] = -1.0  # invalid density
    raw.loc[1, "TAUTH"] = 0.0  # invalid target
    cleaned = hdb5.map_to_canonical(raw)
    assert len(cleaned) == len(raw) - 2


def test_map_to_canonical_missing_columns_raises() -> None:
    raw = _make_fake_hdb5(10).drop(columns=["NEL"])
    with pytest.raises(ValueError, match="missing expected columns"):
        hdb5.map_to_canonical(raw)


def test_ipb98y2_matches_manual_formula() -> None:
    canonical = hdb5.map_to_canonical(_make_fake_hdb5(5))
    row = canonical.iloc[0]
    expected = (
        0.0562
        * row["ip_ma"] ** 0.93
        * row["bt_t"] ** 0.15
        * row["ne_line_1e19_m3"] ** 0.41
        * row["p_loss_mw"] ** -0.69
        * row["r_m"] ** 1.97
        * row["inverse_aspect_ratio"] ** 0.58
        * row["kappa"] ** 0.78
        * row["m_eff_amu"] ** 0.19
    )
    assert hdb5.ipb98y2_tau_s(canonical).iloc[0] == pytest.approx(expected)


def test_model_features_are_leak_free() -> None:
    # The target must never appear (directly or by name) among model features.
    assert hdb5.TARGET_COLUMN not in hdb5.MODEL_FEATURE_COLUMNS
    assert not any("tau_th" in feature for feature in hdb5.MODEL_FEATURE_COLUMNS)
    featured = hdb5.build_features(hdb5.map_to_canonical(_make_fake_hdb5(40)))
    matrix = featured[list(hdb5.MODEL_FEATURE_COLUMNS)].to_numpy()
    assert np.isfinite(matrix).all()


def test_evaluate_models_learns_and_reports_baseline() -> None:
    dataset = hdb5.build_features(hdb5.map_to_canonical(_make_fake_hdb5(600)))
    scores = {s.model_name: s for s in hdb5.evaluate_models(dataset, n_splits=4)}

    assert "ipb98y2_analytic" in scores  # physics baseline is always reported
    assert "mean_baseline" in scores
    # A trained model must beat the naive mean baseline on real-shaped signal.
    assert scores["random_forest"].cv_rmsle < scores["mean_baseline"].cv_rmsle
    # R^2 in log space should be strongly positive for the best model.
    assert scores["random_forest"].cv_r2_log > 0.5


def test_train_confinement_model_writes_and_roundtrips(tmp_path) -> None:
    import joblib

    dataset_path = tmp_path / "fake_hdb5.csv"
    _make_fake_hdb5(400).to_csv(dataset_path, index=False)
    output_dir = tmp_path / "out"

    metadata = hdb5.train_confinement_model(
        dataset_path, output_dir=output_dir, n_splits=4
    )

    assert metadata["target_column"] == hdb5.TARGET_COLUMN
    assert int(cast(int, metadata["n_rows"])) > 0
    assert (output_dir / "confinement_metrics.csv").exists()
    assert (output_dir / "confinement_metadata.json").exists()

    artifact = joblib.load(output_dir / "confinement_model.joblib")
    assert type(artifact).__module__ == "hdb5"  # portable, not __main__
    prepared = hdb5.prepare_dataset(dataset_path).head(3)
    predictions = artifact.predict(prepared)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions > 0)


def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="HDB5 dataset not found"):
        hdb5.load_hdb5_dataframe(tmp_path / "does_not_exist.csv")


def _valid_single_case() -> dict[str, float]:
    return {
        "ip_ma": 2.0,
        "bt_t": 2.5,
        "ne_line_1e19_m3": 6.0,
        "p_loss_mw": 8.0,
        "r_m": 1.7,
        "kappa": 1.7,
        "inverse_aspect_ratio": 0.32,
        "m_eff_amu": 2.0,
    }


def test_build_single_case_frame_derives_and_features() -> None:
    case = _valid_single_case()
    featured = hdb5.build_single_case_frame(case)

    assert len(featured) == 1
    # a_m is derived, not requested.
    assert featured["a_m"].iloc[0] == pytest.approx(case["inverse_aspect_ratio"] * case["r_m"])
    matrix = featured[list(hdb5.MODEL_FEATURE_COLUMNS)].to_numpy()
    assert np.isfinite(matrix).all()


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_build_single_case_frame_rejects_nonpositive(bad_value) -> None:
    case = _valid_single_case()
    case["ip_ma"] = bad_value
    with pytest.raises(ValueError, match="finite and strictly positive"):
        hdb5.build_single_case_frame(case)


def test_build_single_case_frame_rejects_missing() -> None:
    case = _valid_single_case()
    del case["bt_t"]
    with pytest.raises(ValueError, match="Missing required inputs"):
        hdb5.build_single_case_frame(case)


def test_predict_single_case_roundtrips(tmp_path) -> None:
    dataset_path = tmp_path / "fake_hdb5.csv"
    _make_fake_hdb5(400).to_csv(dataset_path, index=False)
    output_dir = tmp_path / "out"
    hdb5.train_confinement_model(dataset_path, output_dir=output_dir, n_splits=4)

    result = hdb5.predict_single_case(
        _valid_single_case(), model_path=output_dir / "confinement_model.joblib"
    )
    assert float(cast(float, result["predicted_tau_th_s"])) > 0
    assert np.isfinite(float(cast(float, result["predicted_tau_th_s"])))
    assert float(cast(float, result["ipb98y2_tau_s"])) > 0
    assert result["model_name"]


def test_load_confinement_artifact_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Confinement model not found"):
        hdb5.load_confinement_artifact(tmp_path / "nope.joblib")


# --- Leave-one-tokamak-out --------------------------------------------------


def _make_fake_hdb5_with_outlier_machine(seed: int = 11) -> pd.DataFrame:
    """Three ordinary machines plus one deliberately larger than all of them.

    ``BIG`` has a major radius above every other machine's, so through the R^1.97
    term its confinement time sits above the whole training range. That is the
    configuration where a tree ensemble is structurally unable to predict, and
    the diagnostic must say so.
    """
    frame = _make_fake_hdb5(n_rows=600, seed=seed)
    frame["TOK"] = np.where(frame.index % 4 == 0, "BIG", frame["TOK"])
    big = frame["TOK"] == "BIG"
    frame.loc[big, "RGEO"] = frame.loc[big, "RGEO"] + 4.0
    # Recompute TAUTH so the enlarged machine is physically consistent.
    frame["TAUTH"] = (
        0.0562
        * frame["IP"].abs() ** 0.93
        * frame["BT"].abs() ** 0.15
        * frame["NEL"] ** 0.41
        * frame["PLTH"] ** -0.69
        * frame["RGEO"] ** 1.97
        * frame["EPS"] ** 0.58
        * frame["KAPPAA"] ** 0.78
        * frame["MEFF"] ** 0.19
    )
    return frame


def test_blind_feature_columns_drop_only_the_ipb98_prior() -> None:
    assert "log_ipb98y2_tau_s" in hdb5.MODEL_FEATURE_COLUMNS
    assert "log_ipb98y2_tau_s" not in hdb5.BLIND_FEATURE_COLUMNS
    assert set(hdb5.BLIND_FEATURE_COLUMNS) == set(hdb5.MODEL_FEATURE_COLUMNS) - {
        "log_ipb98y2_tau_s"
    }


def test_eligible_tokamaks_respects_min_rows() -> None:
    dataset = hdb5.map_to_canonical(_make_fake_hdb5(n_rows=300, seed=3))
    assert hdb5.eligible_tokamaks(dataset, min_rows=1)
    counts = dataset[hdb5.TOKAMAK_LABEL_COLUMN].value_counts()
    for name in hdb5.eligible_tokamaks(dataset, min_rows=50):
        assert counts[name] >= 50
    assert hdb5.eligible_tokamaks(dataset, min_rows=10_000) == []


def test_extrapolation_diagnostic_flags_a_machine_outside_the_target_range() -> None:
    dataset = hdb5.prepare_dataset_from_frame(_make_fake_hdb5_with_outlier_machine())

    big = hdb5.extrapolation_diagnostic(dataset, "BIG")
    # Only the rows where the other parameters also line up clear the training
    # maximum, so this is a meaningful minority rather than most of the machine.
    assert big.target_above_train_max_fraction > 0.05
    assert big.log_target_headroom > 0.5
    assert big.n_features_outside_train_range >= 1

    # An ordinary machine sits inside the range on every axis.
    others = [t for t in dataset[hdb5.TOKAMAK_LABEL_COLUMN].unique() if t != "BIG"]
    ordinary = hdb5.extrapolation_diagnostic(dataset, others[0])
    assert ordinary.target_above_train_max_fraction == 0.0
    assert ordinary.log_target_headroom < 0.0
    assert big.feature_mahalanobis > ordinary.feature_mahalanobis


def test_extrapolation_diagnostic_rejects_unknown_and_sole_machine() -> None:
    dataset = hdb5.prepare_dataset_from_frame(_make_fake_hdb5(n_rows=120, seed=5))
    with pytest.raises(ValueError, match="No rows for tokamak"):
        hdb5.extrapolation_diagnostic(dataset, "NOT_A_MACHINE")
    single = dataset.assign(**{hdb5.TOKAMAK_LABEL_COLUMN: "ONLY"})
    with pytest.raises(ValueError, match="only machine"):
        hdb5.extrapolation_diagnostic(single, "ONLY")


def test_tree_prediction_cannot_exceed_the_training_target_range() -> None:
    """The structural claim the diagnostic exists to measure.

    A random forest averages training targets, so no held-out prediction can
    exceed the largest tau it was trained on, however far the real machine is.
    """
    dataset = hdb5.prepare_dataset_from_frame(_make_fake_hdb5_with_outlier_machine())
    held = (dataset[hdb5.TOKAMAK_LABEL_COLUMN] == "BIG").to_numpy()
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    forest = hdb5.clone_pipeline(hdb5.build_model_zoo()["random_forest"])
    forest.fit(features[~held], log_tau[~held])
    predicted = np.exp(forest.predict(features[held]))

    training_max = np.exp(log_tau[~held].max())
    actual_max = dataset.loc[held, hdb5.TARGET_COLUMN].max()
    assert predicted.max() <= training_max + 1e-9
    assert actual_max > training_max  # the machine really is out of reach


def test_leave_one_tokamak_out_holds_each_machine_out_entirely() -> None:
    dataset = hdb5.prepare_dataset_from_frame(_make_fake_hdb5(n_rows=300, seed=9))
    scores = hdb5.leave_one_tokamak_out(dataset, min_rows=20)

    machines = hdb5.eligible_tokamaks(dataset, min_rows=20)
    assert set(scores["tokamak"]) == set(machines)

    expected_models = set(hdb5.build_model_zoo()) | {"ipb98y2_analytic"}
    for machine in machines:
        rows = scores[scores.tokamak == machine]
        assert set(rows["model_name"]) == expected_models
        # Row count must match the machine's real size, i.e. all of it was held out.
        held = int((dataset[hdb5.TOKAMAK_LABEL_COLUMN] == machine).sum())
        assert set(rows["n_held_out_rows"]) == {held}

    # The analytic law is the only entry that is not blind to the held-out machine.
    assert not scores.loc[scores.model_name == "ipb98y2_analytic", "is_blind"].any()
    assert scores.loc[scores.model_name != "ipb98y2_analytic", "is_blind"].all()


def test_leave_one_tokamak_out_can_drop_the_reference_and_add_controls() -> None:
    dataset = hdb5.prepare_dataset_from_frame(_make_fake_hdb5(n_rows=240, seed=13))
    scores = hdb5.leave_one_tokamak_out(
        dataset, min_rows=20, include_ipb98_reference=False, include_controls=True
    )
    assert "ipb98y2_analytic" not in set(scores["model_name"])
    assert set(hdb5.build_control_models()) <= set(scores["model_name"])
    assert bool(scores["is_blind"].all())


def test_leave_one_tokamak_out_requires_an_eligible_machine() -> None:
    dataset = hdb5.prepare_dataset_from_frame(_make_fake_hdb5(n_rows=120, seed=17))
    with pytest.raises(ValueError, match="nothing can be held out"):
        hdb5.leave_one_tokamak_out(dataset, min_rows=10_000)


def test_constrained_form_beats_trees_on_held_out_machines() -> None:
    """The headline claim, on data where the true law is an exact power law."""
    dataset = hdb5.prepare_dataset_from_frame(_make_fake_hdb5_with_outlier_machine())
    summary = hdb5.summarize_leave_one_tokamak_out(
        hdb5.leave_one_tokamak_out(dataset, min_rows=20)
    )
    by_model = summary.set_index("model_name")["mean_rmsle"]
    assert by_model["ridge_loglinear"] < by_model["random_forest"]
    assert by_model["ridge_loglinear"] < by_model["mean_baseline"]


def test_extrapolation_report_joins_scores_to_diagnostics() -> None:
    dataset = hdb5.prepare_dataset_from_frame(_make_fake_hdb5_with_outlier_machine())
    report = hdb5.extrapolation_report(dataset, min_rows=20)
    for column in ("rmsle", "feature_mahalanobis", "target_above_train_max_fraction"):
        assert column in report.columns
        assert report[column].notna().all()
    # One diagnostic per machine, repeated across that machine's model rows.
    per_machine = report.groupby("tokamak")["feature_mahalanobis"].nunique()
    assert (per_machine == 1).all()
