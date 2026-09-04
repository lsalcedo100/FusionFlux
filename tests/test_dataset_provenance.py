"""The fetch-and-parse layer of the three deposits that are not HDB5 STD5.

``tests/test_dataset_integrity.py`` pins this discipline for STD5: verification
happens on the staged temporary file so a fetch that fails the pin never lands
at the target path, a file already on disk is re-verified rather than trusted,
and waiving the check is explicit. ``allometry``, ``tree_allometry`` and
``replication`` each carry their own copy of that code for their own deposit and
none of it was tested, because every test that touches those modules needs the
deposit and skips without it. CI has none of the three, so the entire download
and parsing layer behind Results 11, 13 and 15 ran there unmeasured: a
verify-before-rename that had been edited into a rename-before-verify would have
been caught by nothing.

Nothing here needs a deposit. Each test synthesises a file of the right shape,
and where the accepting branch is under test it re-points the module's digest
constants at that file, which is the only way to reach it: the pinned SHA-256 of
bytes this repository does not ship cannot otherwise be matched.

The parsing tests are the other half of the same gap. Each deposit has a
documented way of being read wrongly without raising -- carriage-return line
endings and a numeric missing-value sentinel in the Figshare table, a member
path inside the BAAD archive, a numeric index line above DB5.2.3's real header,
and three columns of DB5.2.3 in SI units where the published laws want
megaamperes, 1e19 m^-3 and megawatts. Each is asserted against a synthetic file
rather than only against a download that CI does not have.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

import allometry as al
import hdb5
import replication as rp
import tree_allometry as ta

# --- fixtures the pin can actually accept ------------------------------------


def _repin(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    path: Path,
    *,
    sha_attribute: str,
    bytes_attribute: str,
) -> None:
    """Point a module's content pin at a file we wrote.

    The pin is a statement about bytes that are not in this repository, so the
    branch where verification *succeeds* is unreachable from a checkout without
    the deposit. Re-pinning is what makes it reachable, and the accepting branch
    is worth testing: a checker that only ever raises would be indistinguishable
    from one that raises correctly.
    """
    monkeypatch.setattr(module, sha_attribute, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(module, bytes_attribute, path.stat().st_size)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))


def _refuse_to_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any fetch a failure, so "did not download" is asserted rather than assumed."""
    import urllib.request

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the network was used for a file already on disk")

    monkeypatch.setattr(urllib.request, "urlopen", _explode)


# --- Result 13: the Figshare metabolic-rate deposit --------------------------

# Trailing whitespace on the names, as the deposit has it.
_ALLOMETRY_HEADER = "Order \tSpecies\tBMR (mlO2/hour) \tBody mass for BMR (gr)  "


def _allometry_rows() -> list[tuple[str, str, float, float]]:
    """Three orders, one sentinel measurement and one mass of zero.

    Rodentia and Carnivora clear ``MIN_HELD_OUT_ROWS`` and separate cleanly on
    mass; Chiroptera does not, which is what makes the row floor testable.
    """
    rows: list[tuple[str, str, float, float]] = []
    for index in range(12):
        mass = 20.0 + index
        rows.append(("Rodentia", f"Mus {index}", round(mass**0.75, 4), mass))
    # An absent measurement, written as a number rather than left blank.
    rows.append(("Rodentia", "Mus incognita", al.MISSING_SENTINEL, 26.0))
    for index in range(11):
        mass = 5000.0 + 10.0 * index
        rows.append(("Carnivora", f"Canis {index}", round(mass**0.75, 4), mass))
    # A mass of zero has no logarithm.
    rows.append(("Carnivora", "Canis nullus", 900.0, 0.0))
    for index in range(4):
        mass = 8.0 + index
        rows.append(("Chiroptera", f"Myotis {index}", round(mass**0.75, 4), mass))
    return rows


def _allometry_bytes() -> bytes:
    """The deposit's shape: tab separated, carriage-return line endings."""
    lines = [_ALLOMETRY_HEADER]
    lines.extend(
        f"{order}\t{species}\t{bmr}\t{mass}" for order, species, bmr, mass in _allometry_rows()
    )
    return ("\r".join(lines) + "\r").encode()


@pytest.fixture
def allometry_deposit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / al.DEFAULT_ALLOMETRY_FILENAME
    path.write_bytes(_allometry_bytes())
    _repin(
        monkeypatch,
        al,
        path,
        sha_attribute="ALLOMETRY_SHA256",
        bytes_attribute="ALLOMETRY_N_BYTES",
    )
    return path


def test_allometry_verification_accepts_the_file_it_is_pinned_to(allometry_deposit: Path) -> None:
    """The accepting branch, which no test could reach without the deposit."""
    fingerprint = al.verify_allometry_file(allometry_deposit)
    assert fingerprint.sha256 == al.ALLOMETRY_SHA256
    assert fingerprint.n_bytes == allometry_deposit.stat().st_size


def test_allometry_default_path_sits_in_the_raw_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(al.config, "DATA_RAW_DIR", tmp_path)
    assert al.default_allometry_path() == tmp_path / al.DEFAULT_ALLOMETRY_FILENAME


def test_a_corrupt_allometry_download_never_lands_at_the_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification is on the staged file, so a bad fetch leaves nothing behind.

    The same property ``test_dataset_integrity`` pins for STD5. If a failed
    download could leave a plausible file at the canonical path, the next run
    would read it and the pin would have delayed the problem by one command.
    """
    _serve(monkeypatch, b"Order\tSpecies\nnot the deposit\t0\r")
    target = tmp_path / "raw" / al.DEFAULT_ALLOMETRY_FILENAME

    with pytest.raises(hdb5.DatasetIntegrityError, match="integrity check failed"):
        al.download_allometry(target)
    assert not target.exists()
    assert list((tmp_path / "raw").glob("*")) == [], "staging debris was left behind"


def test_an_allometry_download_can_waive_verification_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, b"Order\tSpecies\r")
    target = tmp_path / "raw" / al.DEFAULT_ALLOMETRY_FILENAME
    assert al.download_allometry(target, verify=False) == target
    assert target.exists()


def test_an_allometry_file_already_on_disk_is_verified_rather_than_trusted(
    tmp_path: Path,
) -> None:
    target = tmp_path / al.DEFAULT_ALLOMETRY_FILENAME
    target.write_bytes(b"stale wrong bytes")
    with pytest.raises(hdb5.DatasetIntegrityError):
        al.download_allometry(target)


def test_an_allometry_file_that_matches_the_pin_is_not_fetched_again(
    allometry_deposit: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_to_serve(monkeypatch)
    assert al.download_allometry(allometry_deposit) == allometry_deposit


def test_the_allometry_deposit_parses_into_rows_with_clean_column_names(
    allometry_deposit: Path,
) -> None:
    """Tab separated, carriage-return terminated, and the names carry whitespace."""
    raw = al.load_allometry_raw(allometry_deposit)
    assert len(raw) == len(_allometry_rows())
    assert list(raw.columns) == [
        "Order",
        "Species",
        "BMR (mlO2/hour)",
        "Body mass for BMR (gr)",
    ]


def test_the_missing_sentinel_and_the_zero_mass_never_reach_the_logs(
    allometry_deposit: Path,
) -> None:
    """-9999 is a valid float, so it survives any check that only tests for nulls."""
    dataset = al.prepare_dataset(allometry_deposit)
    assert len(dataset) == len(_allometry_rows()) - 2
    assert not (dataset[al.TARGET_COLUMN] == al.MISSING_SENTINEL).any()
    assert (dataset[al.TARGET_COLUMN] > 0).all()
    assert (dataset[al.MASS_COLUMN] > 0).all()
    assert np.isfinite(dataset["log_bmr"]).all()
    assert np.isfinite(dataset["log_mass_g"]).all()


def test_the_log_columns_are_the_logs_of_the_columns_they_name(
    allometry_deposit: Path,
) -> None:
    dataset = al.prepare_dataset(allometry_deposit)
    assert np.allclose(dataset["log_mass_g"], np.log(dataset[al.MASS_COLUMN]))
    assert np.allclose(dataset["log_bmr"], np.log(dataset[al.TARGET_COLUMN]))


def test_the_field_metabolic_rate_columns_are_not_carried_through(
    allometry_deposit: Path,
) -> None:
    """FMR is a different measurement on different animals at a different mass."""
    dataset = al.prepare_dataset(allometry_deposit)
    assert set(dataset.columns) == {
        al.GROUP_COLUMN,
        "species",
        al.TARGET_COLUMN,
        al.MASS_COLUMN,
        "log_mass_g",
        "log_bmr",
    }


def test_orders_below_the_row_floor_are_not_scored(allometry_deposit: Path) -> None:
    """Below the floor an order's score is dominated by which few species it has."""
    dataset = al.prepare_dataset(allometry_deposit)
    assert al.eligible_orders(dataset) == ["Rodentia", "Carnivora"]
    assert al.eligible_orders(dataset, min_rows=4) == ["Chiroptera", "Rodentia", "Carnivora"]
    assert al.eligible_orders(dataset, min_rows=100) == []


def test_order_medians_cover_every_order_including_the_ones_too_small_to_score(
    allometry_deposit: Path,
) -> None:
    """The medians are the axis the ordered split extrapolates along, not a filter."""
    dataset = al.prepare_dataset(allometry_deposit)
    medians = al.order_mass_medians(dataset)
    assert set(medians) == {"Rodentia", "Carnivora", "Chiroptera"}
    assert medians["Chiroptera"] < medians["Rodentia"] < medians["Carnivora"]


# --- Result 15: the BAAD release ---------------------------------------------


def _baad_frame() -> pd.DataFrame:
    """A BAAD-shaped plant table, plus the two rows the whole ladder must lose.

    Pinus and Eucalyptus clear ``MIN_HELD_OUT_ROWS`` and separate on diameter;
    Acacia does not. Mass follows the West-Brown-Enquist exponent so the frame is
    the shape the analysis expects rather than noise in the right columns.
    """
    names: list[str] = []
    diameters: list[float] = []
    for index in range(32):
        names.append("Pinus")
        diameters.append(0.40 + 0.01 * index)
    for index in range(32):
        names.append("Eucalyptus")
        diameters.append(0.05 + 0.002 * index)
    for index in range(5):
        names.append("Acacia")
        diameters.append(0.20 + 0.01 * index)

    diameter = np.asarray(diameters, dtype=float)
    frame = pd.DataFrame(
        {
            ta.GROUP_COLUMN: names,
            "d.ba": diameter,
            "m.to": 120.0 * diameter ** (8.0 / 3.0),
            "h.t": 30.0 * diameter**0.6,
            "a.lf": 80.0 * diameter**1.8,
            "ma.ilf": 2.0 * diameter**1.9,
        }
    )
    # A plant measured on every rung but the top one. Rung 1 could score it; the
    # design says no rung may, or the rungs would differ in rows as well as
    # features.
    incomplete = {ta.GROUP_COLUMN: "Pinus", "d.ba": 0.5, "m.to": 30.0, "h.t": 20.0,
                  "a.lf": 25.0, "ma.ilf": np.nan}
    # A height of zero has no logarithm.
    non_positive = {ta.GROUP_COLUMN: "Pinus", "d.ba": 0.5, "m.to": 30.0, "h.t": 0.0,
                    "a.lf": 25.0, "ma.ilf": 1.0}
    return pd.concat([frame, pd.DataFrame([incomplete, non_positive])], ignore_index=True)


def _baad_zip_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(ta.BAAD_MEMBER, frame.to_csv(index=False))
    return buffer.getvalue()


@pytest.fixture
def baad_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ta.DEFAULT_BAAD_FILENAME
    path.write_bytes(_baad_zip_bytes(_baad_frame()))
    _repin(monkeypatch, ta, path, sha_attribute="BAAD_SHA256", bytes_attribute="BAAD_N_BYTES")
    return path


def test_baad_verification_accepts_the_release_it_is_pinned_to(baad_release: Path) -> None:
    fingerprint = ta.verify_baad_file(baad_release)
    assert fingerprint.sha256 == ta.BAAD_SHA256
    assert fingerprint.n_bytes == baad_release.stat().st_size


def test_baad_default_path_sits_in_the_raw_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ta.config, "DATA_RAW_DIR", tmp_path)
    assert ta.default_baad_path() == tmp_path / ta.DEFAULT_BAAD_FILENAME


def test_a_corrupt_baad_download_never_lands_at_the_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, b"PK\x03\x04 not the release")
    target = tmp_path / "raw" / ta.DEFAULT_BAAD_FILENAME

    with pytest.raises(hdb5.DatasetIntegrityError, match="integrity check failed"):
        ta.download_baad(target)
    assert not target.exists()
    assert list((tmp_path / "raw").glob("*")) == [], "staging debris was left behind"


def test_a_baad_download_can_waive_verification_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, _baad_zip_bytes(_baad_frame()))
    target = tmp_path / "raw" / ta.DEFAULT_BAAD_FILENAME
    assert ta.download_baad(target, verify=False) == target
    assert target.exists()


def test_a_baad_file_already_on_disk_is_verified_rather_than_trusted(tmp_path: Path) -> None:
    target = tmp_path / ta.DEFAULT_BAAD_FILENAME
    target.write_bytes(b"stale wrong bytes")
    with pytest.raises(hdb5.DatasetIntegrityError):
        ta.download_baad(target)


def test_a_baad_file_that_matches_the_pin_is_not_fetched_again(
    baad_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_to_serve(monkeypatch)
    assert ta.download_baad(baad_release) == baad_release


def test_the_plant_table_is_read_out_of_the_archive_without_unpacking_it(
    baad_release: Path, tmp_path: Path
) -> None:
    raw = ta.load_baad_raw(baad_release)
    assert len(raw) == len(_baad_frame())
    assert set(ta.SOURCE_COLUMNS) <= set(raw.columns)
    assert sorted(p.name for p in tmp_path.iterdir()) == [ta.DEFAULT_BAAD_FILENAME]


def test_preparation_names_the_ladder_columns_the_archive_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silently absent rung would make the ladder shorter than it reports."""
    frame = _baad_frame().drop(columns=["a.lf"])
    path = tmp_path / ta.DEFAULT_BAAD_FILENAME
    path.write_bytes(_baad_zip_bytes(frame))
    _repin(monkeypatch, ta, path, sha_attribute="BAAD_SHA256", bytes_attribute="BAAD_N_BYTES")

    with pytest.raises(ValueError, match=r"missing expected columns.*a\.lf"):
        ta.prepare_dataset(path)


def test_every_rung_is_scored_on_exactly_one_row_set(baad_release: Path) -> None:
    """The plant missing only its leaf mass is dropped for rung 1 as well.

    If rung 1 were fitted on every plant with a diameter and rung 4 only on those
    that also have a leaf mass, the rungs would differ in their rows as well as
    their features and nothing could be attributed to dimensionality.
    """
    dataset = ta.prepare_dataset(baad_release)
    assert len(dataset) == len(_baad_frame()) - 2
    numeric = list(ta.SOURCE_COLUMNS.values())
    assert dataset[numeric].notna().all().all()
    assert (dataset[numeric] > 0).all().all()


def test_the_prepared_log_columns_are_the_logs_of_the_ladder_columns(
    baad_release: Path,
) -> None:
    dataset = ta.prepare_dataset(baad_release)
    for column in ta.SOURCE_COLUMNS.values():
        assert np.allclose(dataset[f"log_{column}"], np.log(dataset[column]))
    assert set(ta.LOG_FEATURE_ORDER) <= set(dataset.columns)


def test_species_below_the_row_floor_are_not_held_out(baad_release: Path) -> None:
    dataset = ta.prepare_dataset(baad_release)
    assert ta.eligible_species(dataset) == ["Pinus", "Eucalyptus"]
    assert ta.eligible_species(dataset, min_rows=200) == []


def test_species_medians_cover_every_species_including_the_unscored_ones(
    baad_release: Path,
) -> None:
    dataset = ta.prepare_dataset(baad_release)
    medians = ta.species_size_medians(dataset)
    assert set(medians) == {"Pinus", "Eucalyptus", "Acacia"}
    assert medians["Eucalyptus"] < medians["Acacia"] < medians["Pinus"]


# --- Result 11: the full DB5.2.3 revision ------------------------------------


def _physical_rows(n_rows: int, seed: int) -> pd.DataFrame:
    """Rows in the units the published scaling laws are written in.

    The same shape ``tests/test_hdb5.py`` synthesises, with a confinement time
    that follows IPB98(y,2) so the cleaner and the feature builder have
    something real to work on.
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
    tau = (
        0.0562 * ip**0.93 * bt**0.15 * nel**0.41 * plth**-0.69
        * rgeo**1.97 * eps**0.58 * kappa**0.78 * meff**0.19
    )
    return pd.DataFrame(
        {
            "TAUTH": tau * np.exp(rng.normal(0.0, 0.1, n_rows)),
            "IP": ip,
            "BT": bt,
            "NEL": nel,
            "PLTH": plth,
            "RGEO": rgeo,
            "EPS": eps,
            "KAPPAA": kappa,
            "MEFF": meff,
            "DELTA1": rng.uniform(0.1, 0.5, n_rows),
        }
    )


# 30 H-mode rows appear in both exports, 60 only in DB5.2.3, 60 are not H-mode.
_N_SHARED, _N_DISJOINT_H, _N_NON_H = 30, 60, 60


def _replication_exports() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A DB5.2.3-shaped export and the STD5-shaped selection out of it.

    STD5 is a quality selection out of the full revision, so the shared rows are
    literally the same slices carried at both column sets and both unit systems:
    that is what makes the ``(tokamak, shot, time)`` match, and the disjointness
    the result rests on, testable at all.
    """
    total = _N_SHARED + _N_DISJOINT_H + _N_NON_H
    frame = _physical_rows(total, seed=5)
    machines = np.array(["JET", "AUG", "D3D"])
    frame["TOK"] = machines[np.arange(total) % 3]
    frame["SHOT"] = 10000 + np.arange(total) // 2
    frame["TIME"] = np.round(1.0 + 0.001 * np.arange(total), 6)
    phases = ["HGELM"] * (_N_SHARED + _N_DISJOINT_H)
    phases += [["OHM", "L", "RI"][index % 3] for index in range(_N_NON_H)]
    frame["PHASE"] = phases

    db523 = frame.drop(columns=["EPS"]).copy()
    for column, scale in rp.DB523_UNIT_SCALES.items():
        db523[column] = db523[column] * scale
    # DB5.2.3 stores minor radius; STD5 stores the inverse aspect ratio.
    db523["AMIN"] = frame["EPS"] * frame["RGEO"]

    std5 = frame.head(_N_SHARED).drop(columns=["PHASE"]).copy()
    return db523, std5


def _write_db523(path: Path, frame: pd.DataFrame) -> Path:
    """Write the export as downloaded: a numeric index line above the real header."""
    index_line = ",".join(str(number) for number in range(len(frame.columns)))
    path.write_text(index_line + "\n" + frame.to_csv(index=False))
    return path


@pytest.fixture
def replication_exports(tmp_path: Path) -> tuple[Path, Path]:
    db523, std5 = _replication_exports()
    db523_path = _write_db523(tmp_path / rp.DEFAULT_DB523_FILENAME, db523)
    std5_path = tmp_path / hdb5.DEFAULT_HDB5_FILENAME
    std5.to_csv(std5_path, index=False)
    return db523_path, std5_path


def test_db523_verification_accepts_the_revision_it_is_pinned_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_db523(tmp_path / rp.DEFAULT_DB523_FILENAME, _replication_exports()[0])
    _repin(monkeypatch, rp, path, sha_attribute="DB523_SHA256", bytes_attribute="DB523_N_BYTES")
    assert rp.verify_db523_file(path).sha256 == rp.DB523_SHA256


def test_db523_default_path_sits_in_the_raw_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rp.config, "DATA_RAW_DIR", tmp_path)
    assert rp.default_db523_path() == tmp_path / rp.DEFAULT_DB523_FILENAME


def test_a_corrupt_db523_download_never_lands_at_the_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, b"0,1,2\nTOK,SHOT,TIME\nnot,the,revision\n")
    target = tmp_path / "raw" / rp.DEFAULT_DB523_FILENAME

    with pytest.raises(hdb5.DatasetIntegrityError, match="integrity check failed"):
        rp.download_db523(target)
    assert not target.exists()
    assert list((tmp_path / "raw").glob("*")) == [], "staging debris was left behind"


def test_a_db523_download_can_waive_verification_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, b"0,1\nTOK,TAUTH\nJET,0.5\n")
    target = tmp_path / "raw" / rp.DEFAULT_DB523_FILENAME
    assert rp.download_db523(target, verify=False) == target
    assert target.exists()


def test_a_db523_file_already_on_disk_is_verified_rather_than_trusted(tmp_path: Path) -> None:
    target = tmp_path / rp.DEFAULT_DB523_FILENAME
    target.write_bytes(b"stale wrong bytes")
    with pytest.raises(hdb5.DatasetIntegrityError):
        rp.download_db523(target)


def test_a_db523_file_that_matches_the_pin_is_not_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_db523(tmp_path / rp.DEFAULT_DB523_FILENAME, _replication_exports()[0])
    _repin(monkeypatch, rp, path, sha_attribute="DB523_SHA256", bytes_attribute="DB523_N_BYTES")
    _refuse_to_serve(monkeypatch)
    assert rp.download_db523(path) == path


def test_a_missing_db523_says_how_to_fetch_it() -> None:
    with pytest.raises(FileNotFoundError, match="download_db523"):
        rp.load_db523_raw("/nonexistent/hdb5_db523.csv")


def test_the_numeric_index_line_is_skipped_rather_than_read_as_the_header(
    replication_exports: tuple[Path, Path],
) -> None:
    """The real column names are on the second line; reading the first gives integers."""
    db523_path, _ = replication_exports
    raw = rp.load_db523_raw(db523_path)
    assert "TOK" in raw.columns
    assert "0" not in raw.columns
    assert len(raw) == _N_SHARED + _N_DISJOINT_H + _N_NON_H


def test_an_explicit_db523_path_is_not_held_to_the_pin(
    replication_exports: tuple[Path, Path],
) -> None:
    """Analysing another file on purpose is legitimate; the pin guards the canonical one."""
    db523_path, _ = replication_exports
    assert len(rp.load_db523_raw(db523_path)) > 0
    with pytest.raises(hdb5.DatasetIntegrityError):
        rp.load_db523_raw(db523_path, verify=True)


def test_the_si_columns_are_converted_to_the_units_the_published_laws_use(
    replication_exports: tuple[Path, Path],
) -> None:
    """DB5.2.3 stores amperes, m^-3 and watts; IPB98(y,2) is written in MA, 1e19 m^-3 and MW.

    Getting this wrong is not subtle -- IPB98(y,2) evaluates to 3e8 seconds on
    unconverted rows -- but it is the kind of error a pipeline carries all the
    way to a plot rather than raising on.
    """
    db523_path, _ = replication_exports
    raw = rp.load_db523_raw(db523_path)
    prepared = rp.prepare_db523_frame(raw)

    assert prepared["ip_ma"].between(0.4, 4.0).all()
    assert prepared["ne_line_1e19_m3"].between(1.5, 20.0).all()
    assert prepared["p_loss_mw"].between(0.5, 25.0).all()
    # The inverse aspect ratio is derived because DB5.2.3 has no EPS column.
    assert np.allclose(prepared["a_m"], prepared["inverse_aspect_ratio"] * prepared["r_m"])
    assert prepared["iter89p_tau_s"].gt(0).all()
    assert np.allclose(prepared["log_iter89p_tau_s"], np.log(prepared["iter89p_tau_s"]))


def test_the_disjoint_h_arm_shares_no_rows_with_std5(
    replication_exports: tuple[Path, Path],
) -> None:
    """The one property the whole result depends on.

    A row match that silently failed would leave STD5's own rows in the
    "disjoint" arm and reproduce Result 4 by construction.
    """
    db523_path, std5_path = replication_exports
    arms = rp.build_replication_arms(db523_path=db523_path, std5_path=std5_path, min_rows=10)

    assert set(arms) == set(rp.REPLICATION_ARMS)
    assert arms["disjoint_h"].n_rows_shared_with_std5 == 0
    assert arms["disjoint_h"].n_rows == _N_DISJOINT_H


def test_the_non_h_arm_holds_no_h_mode_rows_and_is_scored_against_iter89p(
    replication_exports: tuple[Path, Path],
) -> None:
    """IPB98(y,2) is an H-mode scaling; scoring L-mode against it measures the regime."""
    db523_path, std5_path = replication_exports
    arms = rp.build_replication_arms(db523_path=db523_path, std5_path=std5_path, min_rows=10)

    assert arms["non_h"].n_rows == _N_NON_H
    assert arms["non_h"].baseline_column == "iter89p_tau_s"
    assert arms["non_h"].baseline_label == "ITER89-P"
    assert arms["disjoint_h"].baseline_column == "ipb98y2_tau_s"
    assert arms["disjoint_h"].baseline_label == "IPB98(y,2)"


def test_every_arm_carries_the_provenance_needed_to_read_its_numbers(
    replication_exports: tuple[Path, Path],
) -> None:
    db523_path, std5_path = replication_exports
    arms = rp.build_replication_arms(db523_path=db523_path, std5_path=std5_path, min_rows=10)

    for arm in arms.values():
        payload = arm.to_json()
        assert payload["n_rows"] == arm.n_rows
        assert payload["n_discharges"] == arm.n_discharges
        assert 0 < arm.n_discharges <= arm.n_rows
        assert arm.r_min_m <= arm.r_max_m
        assert arm.n_machines_scored <= 3
