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
