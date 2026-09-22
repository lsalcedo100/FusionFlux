"""The ISHCDB replication's pipeline, on synthetic deposits with the documented columns.

By the lock's order of operations the real file may not be scored until the plan
is frozen, and the first run on it is the run of record. So every test here
builds its own deposit, and this suite is the only place the pipeline is allowed
to be wrong.

The deposits carry the traps the database's documentation describes: rows for
W7-X and ITER that are predictions and not measurements, one device whose
standard target is the thermal confinement time, power in watts and density per
cubic metre, a signed field and rotational transform, and several time slices
to a discharge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
from test_ciclop import _small_zoo

import analysis_ciclop as ac
import analysis_ishcdb as ai
import ciclop
import hdb5
import ishcdb

ROOT = Path(__file__).resolve().parent.parent

# (device, major radius, minor radius, discharges, renormalisation factor when planted)
DEVICES: tuple[tuple[str, float, float, int, float], ...] = (
    ("LHD", 3.75, 0.60, 60, 0.93),
    ("W7-AS", 2.00, 0.17, 45, 1.00),
    ("CHS", 1.00, 0.19, 30, 0.42),
    ("ATF", 2.10, 0.27, 25, 0.43),
    ("HELE", 2.20, 0.21, 25, 0.48),
    ("TJ-II", 1.50, 0.20, 22, 0.25),
    ("HELJ", 1.20, 0.17, 20, 0.97),
    ("W7-A", 2.00, 0.10, 18, 1.00),
    ("HSX", 1.20, 0.13, 8, 1.00),  # twelve standard-set rows: scored at ten, not at thirty
)


def synthetic_deposit(*, device_factors: bool = False, seed: int = 11) -> pd.DataFrame:
    """An ISHCDB-shaped table. Two slices to a discharge, a step in density, predictive rows at the end."""
    rng = np.random.default_rng(seed)
    rows = []
    for device, radius, minor, discharges, factor in DEVICES:
        for shot in range(discharges):
            field = 1.5 * np.exp(rng.normal(0, 0.10))
            iota = 0.45 * np.exp(rng.normal(0, 0.15))
            for time in (0.3, 0.6):
                power = 0.6 * radius * np.exp(rng.normal(0, 0.30))
                density = 4.0 * np.exp(rng.normal(0, 0.40))
                tau = (
                    0.134 * minor**2.28 * radius**0.64 * power**-0.61 * density**0.54 * field**0.84 * iota**0.41
                    * (factor if device_factors else 1.0) * (2.0 if density > 4.0 else 1.0) * np.exp(rng.normal(0, 0.04))
                )
                thermal = device in ("HELE", "TJ-II")
                rows.append({
                    "STELL": device, "STDSET": 1 if shot % 5 else 0, "SHOT": 1000 + shot, "SHOT TIME": time,
                    "PGASA": 1, "RGEO": radius, "AEFF": minor, "BT": -field if shot % 2 else field, "IOTA23": -iota if shot % 3 == 0 else iota,
                    "NEBAR": density * 1e19, "PTOT": power * 1e6, "WDIA": tau * power * 1e6,
                    "TAUEDIA": np.nan if thermal else tau, "TAUETH": tau if thermal else tau * 0.8,
                })
    for device, radius in (("W7-X", 5.5), ("ITER", 6.2)):
        for index in range(12):
            rows.append({
                "STELL": device, "STDSET": 1, "SHOT": index, "SHOT TIME": 1.0, "PGASA": 2, "RGEO": radius, "AEFF": 0.53,
                "BT": 2.5, "IOTA23": 0.9, "NEBAR": 8e19, "PTOT": 1e7, "WDIA": 1e6, "TAUEDIA": 0.5, "TAUETH": 0.4,
            })
    return pd.DataFrame(rows)


def _on_disk(tmp_path: Path, frame: pd.DataFrame | None = None) -> Path:
    path = tmp_path / ishcdb.DEFAULT_ISHCDB_FILENAME
    # As the real file is delivered: comma-separated UTF-8 behind a byte-order mark.
    (synthetic_deposit() if frame is None else frame).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _freeze_and_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: pd.DataFrame | None = None) -> tuple[Path, Path]:
    data = _on_disk(tmp_path, frame)
    plan_path = ishcdb.write_plan(ishcdb.freeze_plan(data), tmp_path / ishcdb.PLAN_FILENAME)
    monkeypatch.setattr(ishcdb, "ISHCDB_SHA256", hashlib.sha256(data.read_bytes()).hexdigest())
    monkeypatch.setattr(ishcdb, "ISHCDB_N_BYTES", data.stat().st_size)
    monkeypatch.setattr(ishcdb, "ISHCDB_PLAN_SHA256", ishcdb.sha256_of_plan(plan_path))
    return data, plan_path


# --- from the file's columns to the study's ------------------------------------------


def test_predictions_for_w7x_and_iter_are_not_data() -> None:
    frame = ishcdb.canonical_frame(synthetic_deposit())
    assert {"W7-X", "ITER"} <= set(frame[hdb5.TOKAMAK_LABEL_COLUMN])
    measured = ishcdb.measured_rows(frame)
    assert not {"W7-X", "ITER"} & set(measured[hdb5.TOKAMAK_LABEL_COLUMN])
    assert len(measured) == len(frame) - 24


def test_the_standard_set_is_the_databases_own_flag() -> None:
    deposit = synthetic_deposit()
    measured = ishcdb.measured_rows(ishcdb.canonical_frame(deposit))
    assert len(ishcdb.standard_set(measured)) == int(((deposit["STDSET"] == 1) & ~deposit["STELL"].isin(["W7-X", "ITER"])).sum())


def test_heliotron_e_and_tj2_take_the_thermal_confinement_time_and_nobody_else_does() -> None:
    """The documentation names both, in IV.A and IV.E. TJ-II has no diamagnetic time in the file at all."""
    assert ishcdb.THERMAL_TARGET_DEVICES == ("HELE", "TJ-II")
    deposit = synthetic_deposit()
    frame = ishcdb.canonical_frame(deposit)
    hele = deposit["STELL"].isin(["HELE", "TJ-II"]).to_numpy()
    assert np.allclose(frame.loc[hele, hdb5.TARGET_COLUMN], deposit.loc[hele, "TAUETH"])
    assert np.allclose(frame.loc[~hele, hdb5.TARGET_COLUMN], deposit.loc[~hele, "TAUEDIA"])
    assert frame.loc[hele, ishcdb.DIAMAGNETIC_TARGET_COLUMN].isna().all()


def test_watts_become_megawatts_and_signs_are_dropped() -> None:
    deposit = synthetic_deposit()
    frame = ishcdb.canonical_frame(deposit)
    assert np.allclose(frame["p_abs_mw"], deposit["PTOT"] / 1e6)
    assert np.allclose(frame["ne_line_1e19_m3"], deposit["NEBAR"] / 1e19)
    assert (deposit["BT"] < 0).any() and (frame["bt_t"] > 0).all()
    assert (deposit["IOTA23"] < 0).any() and (frame["iota_23"] > 0).all()


def test_iss04_is_the_published_law() -> None:
    """Recomputed here from the published exponents, on one row, without the module's code."""
    row = pd.DataFrame([{"a_m": 0.6, "r_m": 3.75, "p_abs_mw": 2.0, "ne_line_1e19_m3": 3.0, "bt_t": 2.75, "iota_23": 0.65}])
    expected = 0.134 * 0.6**2.28 * 3.75**0.64 * 2.0**-0.61 * 3.0**0.54 * 2.75**0.84 * 0.65**0.41
    assert float(ishcdb.iss04_tau_s(row).iloc[0]) == pytest.approx(expected, rel=1e-12)
    # An LHD-like plasma confines for tenths of a second. The bound is a units check: with the
    # power left in watts the same expression gives tens of microseconds.
    assert 0.05 < expected < 0.5
    assert 0.134 * 0.6**2.28 * 3.75**0.64 * 2.0e6**-0.61 * 3.0**0.54 * 2.75**0.84 * 0.65**0.41 < 1e-4


def test_every_slice_of_a_discharge_shares_a_fold_and_devices_do_not_share_discharges() -> None:
    deposit = synthetic_deposit()
    deposit.loc[0, "SHOT"] = np.nan
    frame = ishcdb.canonical_frame(deposit)
    assert frame.loc[2, hdb5.GROUP_COLUMN] == frame.loc[3, hdb5.GROUP_COLUMN] == "LHD::1001"
    # A row with no shot number is its own group, and cannot pull its neighbour into its fold.
    assert frame.loc[0, hdb5.GROUP_COLUMN] == "LHD::row0" and frame.loc[1, hdb5.GROUP_COLUMN] == "LHD::1000"
    # Shot 1000 exists on every device; the group carries the device.
    assert frame[hdb5.GROUP_COLUMN].str.endswith("::1000").sum() > 2
    assert frame.loc[frame[hdb5.GROUP_COLUMN].str.endswith("::1000"), hdb5.GROUP_COLUMN].nunique() == len(DEVICES)


def test_a_column_the_file_spells_differently_is_an_error_until_it_is_recorded() -> None:
    deposit = synthetic_deposit().rename(columns={"IOTA23": "IOTA_23"})
    with pytest.raises(ValueError, match="IOTA23"):
        ishcdb.canonical_frame(deposit)
    frame = ishcdb.canonical_frame(deposit, {"IOTA23": "IOTA_23"})
    assert (frame["iota_23"] > 0).all()


# --- the freeze ------------------------------------------------------------------------


def test_the_deposit_freezes_as_the_lock_says(tmp_path: Path) -> None:
    plan = ishcdb.freeze_plan(_on_disk(tmp_path))
    assert tuple(plan["frozen_features"]) == ishcdb.FEATURES and not plan["omitted_features"]
    assert plan["population"]["n_predictive_rows"] == 24
    assert "W7-X" not in plan["population"]["rows_per_device_measured"]
    # Four discharges in five are in the standard set, two slices each.
    assert plan["complete_rows_per_device"]["LHD"] == 96 and plan["complete_rows_per_device_all_measured"]["LHD"] == 120
    assert "HSX" not in plan["eligible_devices"] and "HSX" in plan["eligible_devices_at_sensitivity_threshold"]
    assert "W7-A" not in plan["eligible_devices"] and len(plan["eligible_devices"]) == 7
    assert plan["discharges"]["largest"] == 2 and plan["discharges"]["n_with_more_than_one_row"] == plan["discharges"]["n"]
    assert plan["evaluable"]


def test_usability_is_decided_without_looking_at_the_target(tmp_path: Path) -> None:
    deposit = synthetic_deposit()
    deposit.loc[deposit.index[::4], "IOTA23"] = np.nan
    shuffled = deposit.copy()
    shuffled["TAUEDIA"] = np.random.default_rng(2).permutation(deposit["TAUEDIA"].to_numpy())
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = ishcdb.freeze_plan(_on_disk(tmp_path / "a", deposit)), ishcdb.freeze_plan(_on_disk(tmp_path / "b", shuffled))
    assert "iota_23" not in first["frozen_features"] and first["frozen_features"] == second["frozen_features"]
    for column in ishcdb.FEATURES:
        assert first["usability"][column] == second["usability"][column]


def test_the_floors_name_what_they_failed_on(tmp_path: Path) -> None:
    few = synthetic_deposit()
    few = few[few["STELL"].isin(["LHD", "W7-AS", "HSX", "ITER"])]
    plan = ishcdb.freeze_plan(_on_disk(tmp_path, few))
    assert not plan["evaluable"] and any("devices have 30 rows" in reason for reason in plan["not_evaluable_because"])


# --- fetching ---------------------------------------------------------------------------


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_a_download_that_does_not_match_the_pin_never_lands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response(b"STELL,SHOT\nLHD,1\n"))
    monkeypatch.setattr(ishcdb, "ISHCDB_SHA256", "0" * 64)
    monkeypatch.setattr(ishcdb, "ISHCDB_N_BYTES", 1)
    target = tmp_path / "ishcdb_26.txt"
    with pytest.raises(hdb5.DatasetIntegrityError, match="integrity check failed"):
        ishcdb.download_ishcdb(target)
    assert not target.exists()


def test_before_the_pin_exists_a_download_lands_and_is_fingerprinted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    payload = b"STELL,SHOT\nLHD,1\n"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response(payload))
    monkeypatch.setattr(ishcdb, "ISHCDB_SHA256", None)
    target = ishcdb.download_ishcdb(tmp_path / "ishcdb_26.txt")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()


# --- the guard --------------------------------------------------------------------------


def test_the_pins_ship_unset_until_a_plan_is_committed() -> None:
    """Either side of step 5 of the lock, and no third state that passes."""
    plan = ROOT / ishcdb.PLAN_FILENAME
    if not plan.exists():
        assert ishcdb.ISHCDB_PLAN_SHA256 is None, "the plan pin is set and no plan has been frozen"
        return
    assert None not in (ishcdb.ISHCDB_SHA256, ishcdb.ISHCDB_N_BYTES, ishcdb.ISHCDB_PLAN_SHA256)
    assert ishcdb.sha256_of_plan(plan) == ishcdb.ISHCDB_PLAN_SHA256
    assert json.loads(plan.read_text())["file"]["sha256"] == ishcdb.ISHCDB_SHA256


def test_nothing_is_scored_before_the_plan_is_frozen_and_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ciclop.PlanNotFrozenError, match="does not exist"):
        ishcdb.load_frozen_plan(tmp_path / ishcdb.PLAN_FILENAME)
    data = _on_disk(tmp_path)
    plan_path = ishcdb.write_plan(ishcdb.freeze_plan(data), tmp_path / ishcdb.PLAN_FILENAME)
    monkeypatch.setattr(ishcdb, "ISHCDB_PLAN_SHA256", None)
    with pytest.raises(ciclop.PlanNotFrozenError, match="ISHCDB pins in ishcdb.py are unset"):
        ishcdb.load_frozen_plan(plan_path, data_path=data)


def test_an_edited_plan_and_a_different_file_are_both_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, plan_path = _freeze_and_pin(monkeypatch, tmp_path)
    plan, resolved = ishcdb.load_frozen_plan(plan_path, data_path=data)
    standard, everything = ishcdb.frames_from_plan(plan, resolved)
    assert len(standard) == plan["n_complete_rows"] and len(everything) == plan["n_complete_rows_all_measured"]

    data.write_text(data.read_text().replace("LHD", "LHX", 1))
    with pytest.raises(hdb5.DatasetIntegrityError, match="ISHCDB integrity check failed"):
        ishcdb.load_frozen_plan(plan_path, data_path=data)

    edited = json.loads(plan_path.read_text())
    edited["rules"]["min_held_out_rows"] = 5
    plan_path.write_text(json.dumps(edited))
    with pytest.raises(ciclop.PlanNotFrozenError, match="edited after it was frozen"):
        ishcdb.load_frozen_plan(plan_path, data_path=data)


def test_the_analysis_script_refuses_to_run_here_until_the_plan_is_frozen() -> None:
    if (ROOT / ishcdb.PLAN_FILENAME).exists():
        pytest.skip("the plan has been frozen; the refusal is covered on synthetic deposits above")
    with pytest.raises((ciclop.PlanNotFrozenError, FileNotFoundError)):
        ai.main()


# --- end to end -------------------------------------------------------------------------


def _run(tmp_path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        data, plan_path = _freeze_and_pin(monkeypatch, tmp_path, frame)
        monkeypatch.setattr(ac, "build_zoo", _small_zoo)
        plan, resolved = ishcdb.load_frozen_plan(plan_path, data_path=data)
        standard, everything = ishcdb.frames_from_plan(plan, resolved)
        return ai.analyze(plan, standard, everything, with_gp=False)


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    yield _run(tmp_path_factory.mktemp("ishcdb"), synthetic_deposit())


@pytest.fixture(scope="module")
def with_device_factors(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    yield _run(tmp_path_factory.mktemp("ishcdb_factors"), synthetic_deposit(device_factors=True))


def test_the_pipeline_finds_an_inversion_that_was_planted(planted: dict[str, Any]) -> None:
    forest, ridge = planted["primary"]["models"][ac.FOREST], planted["primary"]["models"][ac.RIDGE]
    assert forest["cv_matched"] < ridge["cv_matched"] and forest["lodo"] > ridge["lodo"]
    assert planted["verdict"] == ac.REPRODUCES_INVERSION
    assert planted["primary"]["min_rows"] == 30 and len(planted["primary"]["eligible_devices"]) == 7


def test_the_offset_decomposition_is_exact_and_agrees_with_the_scored_errors(planted: dict[str, Any]) -> None:
    """``centred_errors`` fits its own predictions, so it is held to the numbers ``score_arm`` reports."""
    centred = planted["secondary"]["offset_removed"]
    assert centred["computed"]
    for name in ac.RANKED_MODELS:
        scored = planted["primary"]["models"][name]
        for split, reference in (("lodo", scored["lodo_per_device"]), ("cv", scored["cv_per_device"])):
            entry = centred["models"][name][split]
            for device, total in entry["total_per_device"].items():
                assert total == pytest.approx(reference[device], rel=1e-9), (name, split, device)
                rebuilt = np.hypot(entry["offset_per_device"][device], entry["centred_per_device"][device])
                assert rebuilt == pytest.approx(total, rel=1e-9)


def test_a_device_constant_shows_up_as_offset_and_not_as_wrong_trends(with_device_factors: dict[str, Any]) -> None:
    """With ISS04-like factors planted, the ridge misses a held-out device by a constant, and only by that."""
    ridge = with_device_factors["secondary"]["offset_removed"]["models"][ac.RIDGE]["lodo"]
    assert ridge["total"] > 2 * ridge["centred"]
    assert ridge["mean_absolute_offset"] > 0.2
    # TJ-II was planted furthest below the others, so everything trained without it predicts too high for it.
    assert ridge["offset_per_device"]["TJ-II"] > 0.5


def test_iss04_with_its_factor_at_one_is_biased_by_exactly_the_planted_factors(with_device_factors: dict[str, Any]) -> None:
    reference = with_device_factors["iss04"]
    assert reference["computed"]
    planted_factor = {device: factor for device, _, _, _, factor in DEVICES}
    # The deposit's density step doubles tau on about half the rows, which adds log(2)/2 on average.
    for device in ("TJ-II", "CHS", "LHD"):
        expected = -np.log(planted_factor[device]) - np.log(2.0) / 2
        assert reference["mean_log_residual_per_device"][device] == pytest.approx(expected, abs=0.15)


def test_the_secondary_arms_are_reported(planted: dict[str, Any]) -> None:
    secondary = planted["secondary"]
    assert secondary["all_measured_rows"]["n_rows"] > planted["primary"]["n_rows"]
    assert "HSX" in secondary["min_rows_10"]["eligible_devices"] and "HSX" not in planted["primary"]["eligible_devices"]
    # The two thermal-target devices have no diamagnetic time here, so they drop out of that arm.
    assert not {"HELE", "TJ-II"} & set(secondary["diamagnetic_target_for_every_device"]["eligible_devices"])
    assert set(planted["primary"]["distance"]["spearman"]) == set(planted["primary"]["models"])


def test_the_tables_and_the_figure_are_written_from_the_analysis(planted: dict[str, Any], tmp_path: Path) -> None:
    assert set(ai.score_table(planted)["population"]) == {"standard_set", "all_measured_rows"}
    per_device = ai.per_device_table(planted)
    assert len(per_device) == len(planted["primary"]["models"]) * 7 and per_device["iss04"].notna().all()
    figure_input = {**planted, "hdb5_feature_matched": {}, "plan": {**planted["plan"], "device_in_hdb5": {}}}
    figure = ac.plot_ciclop(figure_input, tmp_path / "ishcdb.png", dataset_label="ISHCDB, standard set", open_marker_label=None)
    assert figure is not None and figure.exists() and figure.with_suffix(".pdf").exists()
