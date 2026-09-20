"""The CICLOP replication's pipeline, on a synthetic deposit shaped like the real one.

The real file sits behind an IAEA login and, by the lock's order of operations,
must not be scored until the plan is frozen. So nothing here touches it, and
nothing here could: every test builds its own deposit. The first run on the real
file is the run of record, which makes this suite the only place the pipeline is
allowed to be wrong.

Three things are under test, in the order the lock uses them.

    the mapping    Units, signs, fuel masses, device definitions and the
                   tokamak-only population, on columns deliberately *not* named
                   like HDB5's, because the real file's are not either.

    the freeze     That usability is decided from missingness alone, that a
                   feature dropped is dropped for every model, that the floors
                   bite, and that on all nine features the cleaning is
                   ``hdb5``'s own.

    the guard      That nothing is scored before the plan is frozen and pinned,
                   and that an edited plan or a different file is refused.

Then the verdict rule as a truth table, and one end-to-end run on a deposit with
an inversion planted in it, which the pipeline has to find.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import analysis_ciclop as ac
import ciclop
import hdb5

ROOT = Path(__file__).resolve().parent.parent

# (facility label in the file, configuration, major radius, pulses). The labels
# and the counts follow the published table. The radii do not: they are spaced
# evenly in the log so that no held-out device has a twin left in training,
# which is what makes the planted inversion a property of the deposit and not of
# which two real machines happen to be the same size.
FACILITIES: tuple[tuple[str, str, float, int], ...] = (
    ("JET-C", "tokamak", 3.20, 22),
    ("JET-ILW", "tokamak", 3.20, 24),
    ("DIII-D", "tokamak", 1.45, 40),
    ("WEST", "tokamak", 2.65, 24),
    ("Tore Supra", "tokamak", 2.60, 8),
    ("EAST", "tokamak", 2.20, 19),
    ("KSTAR", "tokamak", 1.80, 12),
    ("ASDEX Upgrade", "tokamak", 1.15, 12),
    ("JT-60U", "tokamak", 3.90, 12),
    ("TCV", "tokamak", 0.88, 4),
    ("W7-X", "stellarator", 5.50, 21),
    ("LHD", "stellarator", 3.60, 5),
)


def synthetic_deposit(seed: int = 7) -> pd.DataFrame:
    """A CICLOP-shaped table with an inversion planted in it.

    Confinement follows a power law across devices, which a log-linear fit can
    carry to a device it has not seen. On top of it sits a step in density, the
    kind of threshold a forest learns from a handful of rows and a power law
    cannot represent at all. Within a device the pulses are near-repeats of one
    scenario, as a database of record pulses would be. So the forest wins
    wherever the device is already in training, and has only its neighbours'
    confinement times to offer when it is not.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for facility, configuration, radius, pulses in FACILITIES:
        for pulse in range(pulses):
            r = radius * np.exp(rng.normal(0, 0.01))
            a = r / 3.1 * np.exp(rng.normal(0, 0.02))
            ip = 0.9 * r**1.4 * np.exp(rng.normal(0, 0.06))
            bt = 2.6 * np.exp(rng.normal(0, 0.05))
            ne19 = 4.0 * np.exp(rng.normal(0, 0.35))
            power = 3.0 * r**1.2 * np.exp(rng.normal(0, 0.12))
            kappa = 1.0 if facility == "Tore Supra" else 1.6 * np.exp(rng.normal(0, 0.04))
            mass = 2.0
            log_tau = (
                np.log(0.05) + 0.9 * np.log(ip) + 0.15 * np.log(bt) + 0.4 * np.log(ne19) - 0.65 * np.log(power)
                + 1.9 * np.log(r) + 0.6 * np.log(a / r) + 0.7 * np.log(kappa) + 0.2 * np.log(mass)
                + (0.7 if ne19 > 4.0 else 0.0) + rng.normal(0, 0.04)
            )
            rows.append({
                "Facility": facility,
                "Pulse": 50000 + pulse,
                "Regime": "H-mode" if pulse % 3 else "L-mode",
                "Date": f"{2015 + pulse % 6}-03-{1 + pulse % 27:02d}",
                "Fuel": "D",
                "Ip [MA]": -ip if pulse % 2 else ip,
                "Bt [T]": bt,
                "nel [1e20 m-3]": ne19 / 10.0,
                "Pinj [MW]": power,
                "R [m]": r,
                "a [m]": a,
                "kappa": kappa,
                "tauE [s]": float(np.exp(log_tau)),
                "H98": 1.0,
                "Duration [s]": 30.0,
                "T fraction": np.nan,
                "configuration_note": configuration,
            })
    return pd.DataFrame(rows)


def mapping_for_the_deposit() -> dict[str, Any]:
    def place(device: str, configuration: str = "tokamak", *, tok: list[str] | None = None, split: str | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {"device": device, "configuration": configuration, "hdb5_tok": tok or []}
        if split is not None:
            entry["device_if_split"] = split
        return entry

    return {
        "file": {"version": "synthetic", "access_date": "never", "terms": "none"},
        "read": {"header_row": 0},
        "identity": {"facility": "Facility", "pulse": "Pulse", "regime": "Regime", "date": "Date"},
        "h_mode_values": ["H-mode"],
        "never_features": ["H98", "Duration [s]"],
        "facilities": {
            "JET-C": place("JET", tok=["JET"]),
            "JET-ILW": place("JET", tok=["JET"]),
            "DIII-D": place("DIII-D", tok=["D3D"]),
            "WEST": place("Tore Supra and WEST", split="WEST"),
            "Tore Supra": place("Tore Supra and WEST", split="Tore Supra"),
            "EAST": place("EAST"),
            "KSTAR": place("KSTAR"),
            "ASDEX Upgrade": place("ASDEX Upgrade", tok=["AUG"]),
            "JT-60U": place("JT-60U", tok=["JT60U"]),
            "TCV": place("TCV", tok=["TCV"]),
            "W7-X": place("W7-X", "stellarator"),
            "LHD": place("LHD", "stellarator"),
        },
        "target": {"source": "tauE [s]", "to_canonical": 1.0, "rank": 2, "definition": "unstated"},
        "quantities": {
            "ip_ma": {"source": "Ip [MA]"},
            "bt_t": {"source": "Bt [T]"},
            "ne_line_1e19_m3": {"source": "nel [1e20 m-3]", "to_canonical": 10.0, "rank": 1},
            "p_loss_mw": {"source": "Pinj [MW]", "rank": 2},
            "r_m": {"source": "R [m]"},
            "kappa": {"source": "kappa", "rank": 2},
            "m_eff_amu": {"source": "Fuel", "species": {"D": "deuterium", "H": "hydrogen", "D-T": "deuterium-tritium"}},
            "a_m": {"source": "a [m]"},
        },
    }


def _deposit_on_disk(tmp_path: Path, frame: pd.DataFrame | None = None) -> Path:
    path = tmp_path / "ciclop_db.csv"
    (synthetic_deposit() if frame is None else frame).to_csv(path, index=False)
    return path


def _freeze_and_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: pd.DataFrame | None = None) -> tuple[Path, Path]:
    """Walk steps 3 to 5 of the lock on a synthetic deposit: obtain, freeze, pin."""
    data = _deposit_on_disk(tmp_path, frame)
    plan_path = ciclop.write_plan(ciclop.freeze_plan(data, mapping_for_the_deposit()), tmp_path / ciclop.PLAN_FILENAME)
    monkeypatch.setattr(ciclop, "CICLOP_FILE_SHA256", hashlib.sha256(data.read_bytes()).hexdigest())
    monkeypatch.setattr(ciclop, "CICLOP_FILE_N_BYTES", data.stat().st_size)
    monkeypatch.setattr(ciclop, "CICLOP_PLAN_SHA256", ciclop.sha256_of_plan(plan_path))
    return data, plan_path


# --- the schema pass ---------------------------------------------------------


def test_the_schema_pass_describes_every_column_and_counts_the_facilities() -> None:
    deposit = synthetic_deposit()
    report = ciclop.schema_report(deposit)
    assert report["n_rows"] == len(deposit) and set(report["columns"]) == set(deposit.columns)
    assert report["columns"]["Facility"]["levels"]["DIII-D"] == 40
    assert report["columns"]["T fraction"]["n_non_null"] == 0


def test_the_schema_pass_never_joins_the_target_to_anything() -> None:
    """Shuffle the target against every feature and the report must not move.

    That is what "marginal summaries only" means, stated as a property: a report
    built from one column at a time cannot tell that the rows were rearranged.
    """
    deposit = synthetic_deposit()
    shuffled = deposit.copy()
    shuffled["tauE [s]"] = np.random.default_rng(0).permutation(deposit["tauE [s]"].to_numpy())
    assert ciclop.schema_report(shuffled) == ciclop.schema_report(deposit)


# --- the mapping -------------------------------------------------------------


def test_units_are_converted_and_the_sign_of_the_current_is_dropped() -> None:
    deposit = synthetic_deposit()
    frame = ciclop.canonical_frame(deposit, mapping_for_the_deposit())
    assert np.allclose(frame["ne_line_1e19_m3"], deposit["nel [1e20 m-3]"] * 10.0)
    assert (deposit["Ip [MA]"] < 0).any() and (frame["ip_ma"] > 0).all()
    assert np.allclose(frame["inverse_aspect_ratio"], deposit["a [m]"] / deposit["R [m]"])


def test_stellarators_are_mapped_and_never_analysed() -> None:
    frame = ciclop.canonical_frame(synthetic_deposit(), mapping_for_the_deposit())
    assert (frame[ciclop.CONFIGURATION_COLUMN] == ciclop.STELLARATOR).sum() == 26
    tokamaks = ciclop.tokamak_rows(frame)
    assert not {"W7-X", "LHD"} & set(tokamaks[hdb5.TOKAMAK_LABEL_COLUMN])
    assert len(tokamaks) == len(frame) - 26


def test_the_held_out_unit_is_the_physical_device() -> None:
    frame = ciclop.canonical_frame(synthetic_deposit(), mapping_for_the_deposit())
    devices = frame.groupby(ciclop.FACILITY_COLUMN)[hdb5.TOKAMAK_LABEL_COLUMN].first()
    assert devices["JET-C"] == devices["JET-ILW"] == "JET"
    assert devices["WEST"] == devices["Tore Supra"] == "Tore Supra and WEST"
    split = frame.groupby(ciclop.FACILITY_COLUMN)[ciclop.SPLIT_DEVICE_COLUMN].first()
    assert (split["WEST"], split["Tore Supra"], split["JET-ILW"]) == ("WEST", "Tore Supra", "JET")


def test_a_facility_the_mapping_does_not_place_is_an_error() -> None:
    deposit = synthetic_deposit()
    deposit.loc[0, "Facility"] = "SPARC"
    with pytest.raises(ValueError, match="SPARC"):
        ciclop.canonical_frame(deposit, mapping_for_the_deposit())


def test_fuel_labels_become_the_masses_the_lock_fixed() -> None:
    deposit = synthetic_deposit().head(5).copy()
    deposit["Fuel"] = ["D", "H", "D-T", "D-T", "xenon"]
    deposit["T fraction"] = [np.nan, np.nan, np.nan, 0.8, np.nan]
    mapping = mapping_for_the_deposit()

    by_label = ciclop.canonical_frame(deposit, mapping)["m_eff_amu"]
    assert by_label.iloc[:4].tolist() == [2.0, 1.0, 2.5, 2.5] and np.isnan(by_label.iloc[4])

    mapping["quantities"]["m_eff_amu"]["tritium_fraction_column"] = "T fraction"
    weighted = ciclop.canonical_frame(deposit, mapping)["m_eff_amu"]
    assert weighted.iloc[2] == 2.5 and weighted.iloc[3] == pytest.approx(2.8)


def test_the_year_of_a_pulse_is_read_from_whatever_the_date_column_holds() -> None:
    """A bare year must not go through a datetime parser, which reads 2021 as 2021 ns after 1970."""
    dates = pd.Series([2021, 2021.0, "2019-11-03", "03/11/2019", pd.Timestamp("2023-05-17"), "campaign C38, 2016", "15/03/21", None, "n/a"])
    years = ciclop.pulse_year(dates)
    assert years.iloc[:6].tolist() == [2021.0, 2021.0, 2019.0, 2019.0, 2023.0, 2016.0]
    assert years.iloc[6:].isna().all()


def test_a_file_with_no_date_column_has_no_years() -> None:
    mapping = mapping_for_the_deposit()
    del mapping["identity"]["date"]
    frame = ciclop.canonical_frame(synthetic_deposit(), mapping)
    assert frame[ciclop.YEAR_COLUMN].isna().all()
    assert ac.grouped_by_device_and_year(frame) is None


def test_grouping_by_campaign_keeps_a_devices_year_in_one_fold() -> None:
    frame = ciclop.canonical_frame(synthetic_deposit(), mapping_for_the_deposit())
    frame.loc[0, ciclop.YEAR_COLUMN] = np.nan
    grouped = ac.grouped_by_device_and_year(frame)
    assert grouped is not None and grouped.loc[1, hdb5.GROUP_COLUMN] == "JET::2016"
    # A pulse with no readable year stays its own group, as it was.
    assert grouped.loc[0, hdb5.GROUP_COLUMN] == frame.loc[0, hdb5.GROUP_COLUMN] == "JET-C::50000"
    assert grouped[hdb5.GROUP_COLUMN].nunique() < frame[hdb5.GROUP_COLUMN].nunique()


def test_a_mapping_can_name_a_species_and_cannot_name_a_mass() -> None:
    mapping = mapping_for_the_deposit()
    mapping["quantities"]["m_eff_amu"]["species"]["D"] = "2.2"
    with pytest.raises(ValueError, match="fuel label 'D'"):
        ciclop.validate_mapping(mapping, list(synthetic_deposit().columns))


@pytest.mark.parametrize("barred", ["H98", "Duration [s]", "tauE [s]"])
def test_a_column_the_lock_bars_cannot_be_mapped_as_a_feature(barred: str) -> None:
    mapping = mapping_for_the_deposit()
    mapping["quantities"]["kappa"]["source"] = barred
    with pytest.raises(ValueError, match="never a feature"):
        ciclop.validate_mapping(mapping, list(synthetic_deposit().columns))


def test_a_preference_order_has_to_say_which_rank_it_took() -> None:
    mapping = mapping_for_the_deposit()
    del mapping["quantities"]["p_loss_mw"]["rank"]
    with pytest.raises(ValueError, match="p_loss_mw.rank"):
        ciclop.validate_mapping(mapping, list(synthetic_deposit().columns))


# --- the freeze --------------------------------------------------------------


def _blank(deposit: pd.DataFrame, column: str, share: float) -> pd.DataFrame:
    """Blank `share` of the tokamak rows of one column, spread evenly over the facilities."""
    out = deposit.copy()
    tokamak = out.index[~out["Facility"].isin(["W7-X", "LHD"])]
    out.loc[tokamak[:: round(1 / share)], column] = np.nan
    return out


def test_the_synthetic_deposit_freezes_all_nine_features(tmp_path: Path) -> None:
    plan = ciclop.freeze_plan(_deposit_on_disk(tmp_path), mapping_for_the_deposit())
    assert tuple(plan["frozen_features"]) == hdb5.BASE_ENGINEERING_COLUMNS and not plan["omitted_features"]
    assert plan["power_is_injected"] is True
    assert plan["population"]["n_stellarator_rows"] == 26 and plan["n_complete_rows"] == 177
    assert plan["complete_rows_per_device"]["JET"] == 46 and plan["complete_rows_per_device"]["Tore Supra and WEST"] == 32
    assert "TCV" not in plan["eligible_devices"] and len(plan["eligible_devices"]) == 7
    assert plan["eligible_devices_at_sensitivity_threshold"] == ["JET", "DIII-D", "Tore Supra and WEST"]
    assert plan["evaluable"] and plan["device_in_hdb5"]["JET"] and not plan["device_in_hdb5"]["EAST"]


def test_a_feature_below_the_usable_share_is_dropped_and_the_rows_are_kept(tmp_path: Path) -> None:
    plan = ciclop.freeze_plan(_deposit_on_disk(tmp_path, _blank(synthetic_deposit(), "kappa", 0.2)), mapping_for_the_deposit())
    assert "kappa" not in plan["frozen_features"] and len(plan["frozen_features"]) == 8
    assert "below 90%" in plan["omitted_features"]["kappa"]
    # Dropped for every model, so no row is lost to it.
    assert plan["n_complete_rows"] == 177


def test_a_feature_above_the_usable_share_is_kept_and_costs_its_missing_rows(tmp_path: Path) -> None:
    plan = ciclop.freeze_plan(_deposit_on_disk(tmp_path, _blank(synthetic_deposit(), "kappa", 0.05)), mapping_for_the_deposit())
    assert "kappa" in plan["frozen_features"] and plan["n_complete_rows"] < 177


def test_a_feature_the_file_lacks_is_omitted_with_that_reason(tmp_path: Path) -> None:
    mapping = mapping_for_the_deposit()
    mapping["quantities"]["m_eff_amu"] = {"source": None}
    plan = ciclop.freeze_plan(_deposit_on_disk(tmp_path), mapping)
    assert plan["omitted_features"] == {"m_eff_amu": "absent from the file"}


def test_losing_the_minor_radius_takes_the_aspect_ratio_with_it(tmp_path: Path) -> None:
    mapping = mapping_for_the_deposit()
    mapping["quantities"]["a_m"] = {"source": None}
    plan = ciclop.freeze_plan(_deposit_on_disk(tmp_path), mapping)
    assert set(plan["omitted_features"]) == {"a_m", "inverse_aspect_ratio"}
    assert "derived as a/R" in plan["omitted_features"]["inverse_aspect_ratio"]


def test_usability_is_decided_without_looking_at_the_target(tmp_path: Path) -> None:
    """The frozen set must not move when the target is rearranged against the features."""
    deposit = _blank(synthetic_deposit(), "kappa", 0.2)
    shuffled = deposit.copy()
    shuffled["tauE [s]"] = np.random.default_rng(1).permutation(deposit["tauE [s]"].to_numpy())
    mapping = mapping_for_the_deposit()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = ciclop.freeze_plan(_deposit_on_disk(tmp_path / "a", deposit), mapping)
    second = ciclop.freeze_plan(_deposit_on_disk(tmp_path / "b", shuffled), mapping)
    assert first["frozen_features"] == second["frozen_features"] and first["usability"] == second["usability"]


def test_the_floors_name_what_they_failed_on(tmp_path: Path) -> None:
    few = synthetic_deposit()
    few = few[few["Facility"].isin(["JET-C", "JET-ILW", "DIII-D", "TCV", "W7-X"])]
    plan = ciclop.freeze_plan(_deposit_on_disk(tmp_path, few), mapping_for_the_deposit())
    assert not plan["evaluable"]
    assert any("rows survive" in reason for reason in plan["not_evaluable_because"])
    assert any("devices have 10 rows" in reason for reason in plan["not_evaluable_because"])


def test_on_all_nine_features_the_cleaning_is_hdb5s_own() -> None:
    """The rule is reimplemented for a feature subset, so it is held to the original here."""
    rng = np.random.default_rng(3)
    n = 60
    raw = pd.DataFrame({
        "TOK": rng.choice(["JET", "AUG", "D3D"], n), "SHOT": rng.integers(1000, 1030, n),
        "TAUTH": rng.lognormal(-2, 0.5, n), "IP": -rng.lognormal(0, 0.3, n), "BT": rng.lognormal(1, 0.2, n),
        "NEL": rng.lognormal(1.5, 0.3, n), "PLTH": rng.lognormal(1.5, 0.4, n), "RGEO": rng.lognormal(0.7, 0.2, n),
        "KAPPAA": rng.lognormal(0.4, 0.05, n), "EPS": rng.lognormal(-1.2, 0.05, n), "MEFF": np.full(n, 2.0),
    })
    raw.loc[[4, 17], "TAUTH"] = [np.nan, -1.0]
    raw.loc[9, "KAPPAA"] = 0.0
    raw["AMIN"] = raw["EPS"] * raw["RGEO"]
    raw["FUEL"] = "D"
    expected = hdb5.prepare_dataset_from_frame(raw)

    names = {"ip_ma": "IP", "bt_t": "BT", "r_m": "RGEO", "a_m": "AMIN"}
    mapping = {
        "identity": {"facility": "TOK", "pulse": "SHOT"},
        "facilities": {t: {"device": t, "configuration": "tokamak"} for t in ("JET", "AUG", "D3D")},
        "target": {"source": "TAUTH", "rank": 1},
        "quantities": {
            **{quantity: {"source": source} for quantity, source in names.items()},
            "ne_line_1e19_m3": {"source": "NEL", "rank": 1},
            "p_loss_mw": {"source": "PLTH", "rank": 1},
            "kappa": {"source": "KAPPAA", "rank": 1},
            "m_eff_amu": {"source": "FUEL", "species": {"D": "deuterium"}},
        },
    }
    observed = ciclop.analysis_frame(ciclop.tokamak_rows(ciclop.canonical_frame(raw, mapping)), hdb5.BASE_ENGINEERING_COLUMNS)

    assert len(observed) == len(expected) == n - 3
    assert observed[hdb5.GROUP_COLUMN].tolist() == expected[hdb5.GROUP_COLUMN].tolist()
    for column in (hdb5.TARGET_COLUMN, *hdb5.BLIND_FEATURE_COLUMNS):
        assert np.allclose(observed[column], expected[column], rtol=0, atol=1e-12), column


def test_a_workbook_is_read_as_delivered_and_freezes_the_plan_a_csv_would(tmp_path: Path) -> None:
    """The real file may be a workbook with a title above its header, on a named sheet.

    The lock has the delivered bytes pinned and analysed, so the workbook is read
    directly. Nothing about the plan may depend on which container the same
    numbers arrived in.
    """
    pytest.importorskip("openpyxl")
    deposit = synthetic_deposit()
    workbook = tmp_path / "ciclop_db.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"note": ["not the database"]}).to_excel(writer, sheet_name="README", index=False)
        deposit.to_excel(writer, sheet_name="DB v7.3", index=False, startrow=1)

    mapping = mapping_for_the_deposit()
    mapping["read"] = {"sheet": "DB v7.3", "header_row": 1}
    from_workbook = ciclop.freeze_plan(workbook, mapping)
    from_csv = ciclop.freeze_plan(_deposit_on_disk(tmp_path, deposit), mapping_for_the_deposit())

    assert from_workbook["file"]["name"] == "ciclop_db.xlsx"
    for key in ("frozen_features", "n_complete_rows", "complete_rows_per_device", "eligible_devices", "evaluable"):
        assert from_workbook[key] == from_csv[key], key
    assert from_workbook["usability"] == pytest.approx(from_csv["usability"])


def test_a_workbook_in_an_environment_without_a_reader_names_the_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workbook = tmp_path / "ciclop_db.xlsx"
    workbook.write_bytes(b"not read: the reader is missing")

    def _no_reader(*args: object, **kwargs: object) -> None:
        raise ImportError("Missing optional dependency 'openpyxl'.")

    monkeypatch.setattr(pd, "read_excel", _no_reader)
    with pytest.raises(ImportError, match="Do not export the sheet to CSV by hand"):
        ciclop.load_ciclop_raw(workbook)


def test_a_file_that_is_neither_a_table_nor_a_workbook_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "ciclop_db.pdf"
    other.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="unrecognised CICLOP file type"):
        ciclop.load_ciclop_raw(other)
    with pytest.raises(FileNotFoundError, match="NUCLEUS login"):
        ciclop.load_ciclop_raw(tmp_path / "absent.csv")


# --- the guard ---------------------------------------------------------------


def test_the_pins_ship_unset_until_a_plan_is_committed() -> None:
    """The repository's own state, either side of step 5 of the lock.

    Before the plan exists the pins are unset, which is what makes the analysis
    refuse to run. Once it exists they have to be set and the plan on disk has
    to be the pinned one. There is no third state that passes.
    """
    plan = ROOT / ciclop.PLAN_FILENAME
    pins = (ciclop.CICLOP_FILE_SHA256, ciclop.CICLOP_FILE_N_BYTES, ciclop.CICLOP_PLAN_SHA256)
    if not plan.exists():
        assert pins == (None, None, None), "a pin is set and no plan has been frozen"
        return
    assert None not in pins, f"{ciclop.PLAN_FILENAME} is committed and a pin in ciclop.py is still unset"
    assert ciclop.sha256_of_plan(plan) == ciclop.CICLOP_PLAN_SHA256
    assert json.loads(plan.read_text())["file"]["sha256"] == ciclop.CICLOP_FILE_SHA256


def test_nothing_is_scored_before_a_plan_exists(tmp_path: Path) -> None:
    with pytest.raises(ciclop.PlanNotFrozenError, match="does not exist"):
        ciclop.load_frozen_plan(tmp_path / ciclop.PLAN_FILENAME)


def test_a_plan_that_is_written_and_not_pinned_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _deposit_on_disk(tmp_path)
    plan_path = ciclop.write_plan(ciclop.freeze_plan(data, mapping_for_the_deposit()), tmp_path / ciclop.PLAN_FILENAME)
    monkeypatch.setattr(ciclop, "CICLOP_PLAN_SHA256", None)
    with pytest.raises(ciclop.PlanNotFrozenError, match="unset"):
        ciclop.load_frozen_plan(plan_path, data_path=data)


def test_a_plan_edited_after_it_was_frozen_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, plan_path = _freeze_and_pin(monkeypatch, tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["rules"]["min_held_out_rows"] = 5
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ciclop.PlanNotFrozenError, match="edited after it was frozen"):
        ciclop.load_frozen_plan(plan_path, data_path=data)


def test_a_file_other_than_the_pinned_one_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, plan_path = _freeze_and_pin(monkeypatch, tmp_path)
    data.write_text(data.read_text().replace("DIII-D", "DIII-X", 1))
    with pytest.raises(hdb5.DatasetIntegrityError, match="integrity check failed"):
        ciclop.load_frozen_plan(plan_path, data_path=data)


def test_the_frozen_plan_rebuilds_exactly_the_rows_it_froze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, plan_path = _freeze_and_pin(monkeypatch, tmp_path)
    plan, resolved = ciclop.load_frozen_plan(plan_path, data_path=data)
    assert len(ciclop.frame_from_plan(plan, resolved)) == plan["n_complete_rows"]

    plan["complete_rows_per_device"]["JET"] -= 1
    with pytest.raises(AssertionError, match="not the rows the plan froze"):
        ciclop.frame_from_plan(plan, resolved)


def test_the_analysis_script_refuses_to_run_in_this_repository_until_the_plan_is_frozen() -> None:
    if (ROOT / ciclop.PLAN_FILENAME).exists():
        pytest.skip("the plan has been frozen; the refusal is covered on synthetic deposits above")
    with pytest.raises(ciclop.PlanNotFrozenError):
        ac.main()


# --- the verdict -------------------------------------------------------------

# cv_flexible, lodo_flexible, cv_ridge, lodo_ridge, n_worse, k
CASES = {
    "the HDB5 pattern": ((0.124, 0.465, 0.187, 0.214, 7, 7), ac.REPRODUCES_INVERSION),
    "inversion on a bare majority": ((0.10, 0.30, 0.15, 0.20, 4, 7), ac.REPRODUCES_INVERSION),
    "the ranks cross inside the noise": ((0.2000, 0.2101, 0.2010, 0.2100, 4, 7), ac.NO_IMPORTANT_DEGRADATION),
    "the ranks cross on every device and the effect is small": ((0.19, 0.24, 0.20, 0.21, 7, 7), ac.NO_IMPORTANT_DEGRADATION),
    "the means invert and one device carries it": ((0.10, 0.30, 0.15, 0.20, 3, 7), ac.DEGRADATION_WITHOUT_INVERSION),
    "exactly half the devices is not a majority": ((0.10, 0.30, 0.15, 0.20, 3, 6), ac.DEGRADATION_WITHOUT_INVERSION),
    "the forest never won interpolation": ((0.20, 0.50, 0.15, 0.20, 7, 7), ac.DEGRADATION_WITHOUT_INVERSION),
    "the forest still wins and degrades far more": ((0.05, 0.19, 0.15, 0.20, 2, 7), ac.DEGRADATION_WITHOUT_INVERSION),
    "D exactly on the boundary": ((0.10, 0.18, 0.15, 0.18, 3, 7), ac.DEGRADATION_WITHOUT_INVERSION),
    "D just under the boundary": ((0.10, 0.179, 0.15, 0.18, 3, 7), ac.NO_IMPORTANT_DEGRADATION),
    "both degrade alike": ((0.10, 0.14, 0.15, 0.20, 2, 7), ac.NO_IMPORTANT_DEGRADATION),
    "the forest extrapolates better": ((0.10, 0.12, 0.15, 0.20, 1, 7), ac.CONTRADICTS),
    "better on the mean, and on a minority of devices": ((0.10, 0.12, 0.15, 0.20, 4, 7), ac.NO_IMPORTANT_DEGRADATION),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_verdict_rule(name: str) -> None:
    (cv_f, lodo_f, cv_r, lodo_r, worse, k), expected = CASES[name]
    observed = ac.verdict(
        not_evaluable=False, cv_flexible=cv_f, lodo_flexible=lodo_f, cv_ridge=cv_r, lodo_ridge=lodo_r,
        n_devices_flexible_worse=worse, n_devices=k,
    )
    assert observed == expected


def test_a_dataset_below_the_floors_cannot_reproduce_anything() -> None:
    (cv_f, lodo_f, cv_r, lodo_r, worse, k), _ = CASES["the HDB5 pattern"]
    assert ac.verdict(
        not_evaluable=True, cv_flexible=cv_f, lodo_flexible=lodo_f, cv_ridge=cv_r, lodo_ridge=lodo_r,
        n_devices_flexible_worse=worse, n_devices=k,
    ) == ac.CANNOT_BE_EVALUATED


def test_every_outcome_gets_exactly_one_verdict_and_it_means_what_it_says() -> None:
    rng = np.random.default_rng(11)
    seen = set()
    for _ in range(4000):
        cv_f, lodo_f, cv_r, lodo_r = rng.lognormal(-1.5, 0.6, 4)
        k = int(rng.integers(5, 8))
        worse = int(rng.integers(0, k + 1))
        label = ac.verdict(
            not_evaluable=False, cv_flexible=cv_f, lodo_flexible=lodo_f, cv_ridge=cv_r, lodo_ridge=lodo_r,
            n_devices_flexible_worse=worse, n_devices=k,
        )
        seen.add(label)
        d = ac.differential_degradation(cv_f, lodo_f, cv_r, lodo_r)
        if label == ac.REPRODUCES_INVERSION:
            assert cv_f < cv_r and lodo_f > lodo_r and worse > k / 2 and d >= ciclop.DEGRADATION_BOUNDARY
        elif label == ac.CONTRADICTS:
            assert lodo_f < lodo_r and worse < k / 2 and d <= 1.0
        elif label == ac.DEGRADATION_WITHOUT_INVERSION:
            assert d >= ciclop.DEGRADATION_BOUNDARY
        else:
            assert label == ac.NO_IMPORTANT_DEGRADATION and d < ciclop.DEGRADATION_BOUNDARY
    assert seen == set(ac.VERDICTS) - {ac.CANNOT_BE_EVALUATED}


def test_the_hdb5_value_of_d_the_lock_quotes() -> None:
    """The lock says 3.3, from the matched row of the manuscript's robustness table."""
    assert ac.differential_degradation(0.124, 0.465, 0.187, 0.214) == pytest.approx(3.28, abs=0.01)


def test_pooling_per_device_errors_is_exact() -> None:
    rng = np.random.default_rng(5)
    residuals = {"A": rng.normal(0, 0.3, 40), "B": rng.normal(0.2, 0.1, 11), "C": rng.normal(0, 0.5, 25)}
    per_unit = {u: float(np.sqrt(np.mean(r**2))) for u, r in residuals.items()}
    rows = {u: len(r) for u, r in residuals.items()}
    direct = float(np.sqrt(np.mean(np.concatenate([residuals["A"], residuals["C"]]) ** 2)))
    assert ac._pooled(per_unit, rows, ("A", "C")) == pytest.approx(direct, rel=1e-12)


# --- end to end, on the planted inversion ------------------------------------


def _small_zoo(*, with_controls: bool, with_gp: bool) -> dict[str, Any]:
    """The real zoo with a 40-tree forest. Same names, same pipeline shapes, a tenth of the time."""
    zoo = dict(hdb5._assemble_zoo(include_controls=with_controls))
    zoo[ac.FOREST] = Pipeline([("model", RandomForestRegressor(n_estimators=40, random_state=hdb5.RANDOM_STATE))])
    if with_controls:
        import analysis_extrapolation as ae

        zoo.update({k: v for k, v in ae.build_flexibility_ladder().items() if k not in zoo})
    return zoo


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    tmp_path = tmp_path_factory.mktemp("ciclop")
    with pytest.MonkeyPatch.context() as monkeypatch:
        data, plan_path = _freeze_and_pin(monkeypatch, tmp_path)
        monkeypatch.setattr(ac, "build_zoo", _small_zoo)
        plan, resolved = ciclop.load_frozen_plan(plan_path, data_path=data)
        analysis = ac.analyze(plan, ciclop.frame_from_plan(plan, resolved), with_hdb5=False, with_gp=False)
        analysis["_figure"] = ac.plot_ciclop(analysis, tmp_path / "ciclop.png")
        yield analysis


def test_the_pipeline_finds_an_inversion_that_was_planted(planted: dict[str, Any]) -> None:
    primary = planted["primary"]
    forest, ridge = primary["models"][ac.FOREST], primary["models"][ac.RIDGE]
    assert forest["cv_matched"] < ridge["cv_matched"] and forest["lodo"] > ridge["lodo"]
    assert planted["verdict"] == ac.REPRODUCES_INVERSION
    pair = primary["pairs"][f"{ac.FOREST}_vs_{ac.RIDGE}"]
    assert pair["n_devices"] == 7 and pair["n_devices_flexible_worse"] > 3 and pair["differential_degradation"] > 1.5


def test_both_splits_score_the_same_devices_the_same_way(planted: dict[str, Any]) -> None:
    primary = planted["primary"]
    eligible = primary["eligible_devices"]
    assert "TCV" not in eligible and "TCV" in primary["rows_per_device"]
    for scores in primary["models"].values():
        assert list(scores["cv_per_device"]) == list(scores["lodo_per_device"]) == eligible
        assert scores["cv_matched"] == pytest.approx(np.mean(list(scores["cv_per_device"].values())))
        assert scores["lodo"] == pytest.approx(np.mean(list(scores["lodo_per_device"].values())))
        assert scores["transfer_ratio"] == pytest.approx(scores["lodo"] / scores["cv_matched"])


def test_the_controls_are_scored_and_kept_out_of_the_ranking(planted: dict[str, Any]) -> None:
    primary = planted["primary"]
    assert set(ac.CONTROL_MODELS) <= set(primary["models"])
    assert set(primary["pairs"]) == {f"{a}_vs_{b}" for a, b in ac.PAIRS}


def test_error_against_distance_is_reported_per_device(planted: dict[str, Any]) -> None:
    distance = planted["primary"]["distance"]
    assert distance["computed"] and set(distance["mahalanobis"]) == set(planted["primary"]["eligible_devices"])
    assert set(distance["spearman"]) == set(planted["primary"]["models"])
    assert all(-1.0 <= rho <= 1.0 for rho in distance["spearman"].values())


def test_the_secondary_arms_are_reported_and_cannot_reach_the_verdict(planted: dict[str, Any]) -> None:
    secondary = planted["secondary"]
    thirty = secondary["min_rows_30"]
    assert thirty["eligible_devices"] == ["JET", "DIII-D", "Tore Supra and WEST"]
    assert set(thirty["verdicts"].values()) == {ac.CANNOT_BE_EVALUATED} and thirty["scored"]

    split = secondary["tore_supra_and_west_split"]
    assert "WEST" in split["eligible_devices"] and "Tore Supra" not in split["eligible_devices"]

    absent = secondary["devices_absent_from_hdb5"]
    assert absent["devices"] == ["Tore Supra and WEST", "EAST", "KSTAR"]

    # Cutting the folds between campaigns changes cross-validation and nothing else.
    by_campaign = secondary["cv_grouped_by_device_and_year"]
    assert by_campaign["scored"] and by_campaign["eligible_devices"] == planted["primary"]["eligible_devices"]
    for name in ac.RANKED_MODELS:
        assert by_campaign["models"][name]["lodo"] == pytest.approx(planted["primary"]["models"][name]["lodo"])
    assert by_campaign["models"][ac.FOREST]["cv_matched"] != planted["primary"]["models"][ac.FOREST]["cv_matched"]

    assert secondary["ipb98y2_on_h_mode_rows"]["computed"]
    assert secondary["ipb98y2_on_h_mode_rows"]["power"] == "injected additional power"
    # Two thirds of each device is H-mode here, which leaves too few devices above ten rows.
    assert secondary["h_mode_only"]["not_evaluable_because"]


def test_the_tables_and_the_figure_are_written_from_the_analysis(planted: dict[str, Any]) -> None:
    scores = ac.score_table(planted)
    assert set(scores["dataset"]) == {"ciclop"} and set(ac.RANKED_MODELS) <= set(scores["model"])
    per_device = ac.per_device_table(planted)
    assert len(per_device) == len(planted["primary"]["models"]) * 7
    assert not per_device.loc[per_device["device"] == "EAST", "in_hdb5"].any()
    figure = planted["_figure"]
    assert figure is not None and figure.exists() and figure.with_suffix(".pdf").exists()


# --- the comparator, where HDB5 is on this machine ---------------------------

needs_hdb5 = pytest.mark.skipif(not hdb5.default_hdb5_path().exists(), reason="HDB5 STD5 is not on this machine")


def _needs_db523() -> None:
    import replication as rp

    if not rp.default_db523_path().exists():
        pytest.skip("DB5.2.3 is not on this machine")


@needs_hdb5
def test_on_loss_power_and_all_nine_the_comparator_is_std5_by_device() -> None:
    dataset = ac.hdb5_matched_dataset(hdb5.BASE_ENGINEERING_COLUMNS, power_is_injected=False)
    reference = hdb5.prepare_dataset()
    assert len(dataset) == len(reference) == 6228
    assert dataset[hdb5.TOKAMAK_LABEL_COLUMN].nunique() == 16
    for column in hdb5.BLIND_FEATURE_COLUMNS:
        assert np.allclose(dataset[column], reference[column], rtol=0, atol=1e-12), column


@needs_hdb5
def test_injected_power_in_hdb5_is_the_sum_over_heating_systems() -> None:
    """``PINJ`` alone is one beam system. The lock's deviations log records why this matters."""
    _needs_db523()
    dataset = ac.hdb5_matched_dataset(hdb5.BASE_ENGINEERING_COLUMNS, power_is_injected=True)
    assert len(dataset) == 6196
    loss = hdb5.prepare_dataset()["p_loss_mw"]
    assert 0.95 < float(np.median(dataset["p_loss_mw"])) / float(np.median(loss)) < 1.15


@needs_hdb5
def test_the_matched_scoring_reproduces_the_manuscripts_own_numbers() -> None:
    """The new code path against ``results/robustness.json``, on the ridge, which is fast.

    Held out by database label at 30 rows on all nine features, the holdout
    score must be the one the manuscript prints, and the matched CV must be the
    mean of the committed per-label CV errors over the labels that are scored.
    """
    committed = json.loads((ROOT / "results" / "robustness.json").read_text())["arms"]
    arm = ac.score_arm(hdb5.prepare_dataset(), hdb5.BLIND_FEATURE_COLUMNS, min_rows=30, models=(ac.RIDGE,))
    ridge = arm["models"][ac.RIDGE]
    assert len(arm["eligible_devices"]) == 13
    assert ridge["lodo"] == pytest.approx(committed["lomo_by_database_label"][ac.RIDGE]["unit_equal"], rel=1e-9)
    assert ridge["cv_pooled_all_rows"] == pytest.approx(committed["cv_all_rows"][ac.RIDGE]["pooled_rows"], rel=1e-9)
    per_label = committed["cv_all_rows"][ac.RIDGE]["per_unit"]
    assert ridge["cv_matched"] == pytest.approx(np.mean([per_label[u] for u in arm["eligible_devices"]]), rel=1e-9)
