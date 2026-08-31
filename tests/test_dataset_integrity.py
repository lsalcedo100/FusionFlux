"""Tests for the HDB5 content pin.

Every number in ``results/RESULTS.md`` is a statement about one specific file
that this repository does not contain: it is fetched at run time from a
third-party host. The pin is what makes those numbers falsifiable, so these
tests are about the pin failing *loudly* in each of the ways it can fail.

The tests that need the real dataset skip when it is absent, because it is not
committed. The ones that matter most do not need it: they synthesise a
mismatched file and assert on the refusal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import hdb5


def _real_dataset_or_skip() -> Path:
    path = hdb5.default_hdb5_path()
    if not path.exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return path


def _write_bytes(tmp_path: Path, payload: bytes, name: str = "hdb5_std5.csv") -> Path:
    target = tmp_path / name
    target.write_bytes(payload)
    return target


# --- the digest itself ------------------------------------------------------


def test_sha256_of_file_matches_hashing_the_bytes_directly(tmp_path: Path) -> None:
    """The streaming digest must equal the one-shot digest, block size aside."""
    payload = b"tokamak," * 5000
    path = _write_bytes(tmp_path, payload, name="payload.bin")
    assert hdb5.sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_streams_files_larger_than_one_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force multiple read() calls so the loop, not just the first block, is covered."""
    monkeypatch.setattr(hdb5, "_HASH_BLOCK_BYTES", 64)
    payload = bytes(range(256)) * 40  # comfortably many blocks
    path = _write_bytes(tmp_path, payload, name="payload.bin")
    assert hdb5.sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


# --- the pin's internal consistency ----------------------------------------


def test_pinned_constants_describe_the_same_file() -> None:
    """The three pin constants must move together or the pin means nothing.

    A digest updated without its byte count or shape would still pass
    ``matches_pin`` while silently describing a file nobody checked.
    """
    path = _real_dataset_or_skip()
    assert hdb5.sha256_of_file(path) == hdb5.HDB5_STD5_SHA256
    assert path.stat().st_size == hdb5.HDB5_STD5_N_BYTES
    assert pd.read_csv(path, low_memory=False).shape == hdb5.HDB5_STD5_RAW_SHAPE


def test_verify_accepts_the_pinned_dataset() -> None:
    fingerprint = hdb5.verify_hdb5_file(_real_dataset_or_skip())
    assert fingerprint.matches_pin
    assert fingerprint.n_rows == hdb5.HDB5_STD5_RAW_SHAPE[0]


# --- the three ways it fails ------------------------------------------------


def test_a_revised_dataset_is_rejected_and_named_as_a_revision(tmp_path: Path) -> None:
    """Right shape, wrong bytes: upstream revised the data.

    This is the dangerous case. The file still parses, still has the expected
    columns, and every downstream script would run to completion and print
    different numbers. Nothing but the digest catches it.
    """
    source = _real_dataset_or_skip()
    frame = pd.read_csv(source, low_memory=False)
    # Nudge a single target value: the file still parses and still has the
    # expected shape, so only the digest can tell it apart from the original.
    frame.loc[0, "TAUTH"] = float(frame["TAUTH"].iloc[0]) * 1.01
    target = tmp_path / "hdb5_std5.csv"
    frame.to_csv(target, index=False)

    with pytest.raises(hdb5.DatasetIntegrityError) as excinfo:
        hdb5.verify_hdb5_file(target)
    message = str(excinfo.value)
    assert "revised" in message
    assert "regenerated" in message


def test_a_different_dataset_is_rejected_and_named_by_its_shape(tmp_path: Path) -> None:
    """Wrong shape: this is not the expected dataset at all."""
    target = tmp_path / "hdb5_std5.csv"
    pd.DataFrame({"TOK": ["JET"], "TAUTH": [0.5]}).to_csv(target, index=False)

    with pytest.raises(hdb5.DatasetIntegrityError) as excinfo:
        hdb5.verify_hdb5_file(target)
    assert "different dataset" in str(excinfo.value)


def test_an_unparseable_file_is_rejected_as_a_failed_download(tmp_path: Path) -> None:
    """A truncated or HTML-error-page download does not parse as CSV."""
    target = _write_bytes(tmp_path, b"\x00\x01\x02 not a csv \xff\xfe")

    with pytest.raises(hdb5.DatasetIntegrityError) as excinfo:
        hdb5.verify_hdb5_file(target)
    assert "truncated or failed download" in str(excinfo.value)


def test_integrity_error_is_not_a_value_error(tmp_path: Path) -> None:
    """Callers that broadly catch bad input must not swallow a pin failure."""
    target = _write_bytes(tmp_path, b"nope")
    with pytest.raises(hdb5.DatasetIntegrityError):
        hdb5.verify_hdb5_file(target)
    assert not issubclass(hdb5.DatasetIntegrityError, ValueError)


# --- where verification is and is not enforced ------------------------------


def test_loading_the_canonical_path_enforces_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path is the one the published results came from, so it is checked."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(hdb5.config, "DATA_RAW_DIR", raw_dir)
    pd.DataFrame({"TOK": ["JET"], "TAUTH": [0.5]}).to_csv(
        raw_dir / hdb5.DEFAULT_HDB5_FILENAME, index=False
    )

    with pytest.raises(hdb5.DatasetIntegrityError):
        hdb5.load_hdb5_dataframe()


def test_an_explicit_dataset_path_is_reported_rather_than_enforced(tmp_path: Path) -> None:
    """Analysing a different file on purpose is legitimate and must still work.

    The pin exists to stop the canonical dataset changing underneath the results,
    not to forbid pointing the pipeline at something else.
    """
    target = tmp_path / "other.csv"
    pd.DataFrame({"TOK": ["JET"], "TAUTH": [0.5]}).to_csv(target, index=False)
    frame = hdb5.load_hdb5_dataframe(target)
    assert list(frame["TOK"]) == ["JET"]

    fingerprint = hdb5.fingerprint_file(target)
    assert not fingerprint.matches_pin


def test_verify_can_be_forced_on_an_explicit_path(tmp_path: Path) -> None:
    target = tmp_path / "other.csv"
    pd.DataFrame({"TOK": ["JET"], "TAUTH": [0.5]}).to_csv(target, index=False)
    with pytest.raises(hdb5.DatasetIntegrityError):
        hdb5.load_hdb5_dataframe(target, verify=True)


def test_verification_can_be_waived_on_the_canonical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(hdb5.config, "DATA_RAW_DIR", raw_dir)
    pd.DataFrame({"TOK": ["JET"], "TAUTH": [0.5]}).to_csv(
        raw_dir / hdb5.DEFAULT_HDB5_FILENAME, index=False
    )
    assert len(hdb5.load_hdb5_dataframe(verify=False)) == 1


# --- the download path ------------------------------------------------------


def test_a_corrupt_download_never_lands_at_the_target_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification happens before the atomic rename, so a bad fetch leaves nothing.

    This is the property that matters: if a failed download could leave a
    plausible-looking file at the canonical path, the *next* run would pick it up
    and the pin would only have delayed the problem by one command.
    """
    import urllib.request

    class _FakeResponse:
        def read(self) -> bytes:
            return b"totally not the hdb5 database"

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    target = tmp_path / "raw" / hdb5.DEFAULT_HDB5_FILENAME

    with pytest.raises(hdb5.DatasetIntegrityError):
        hdb5.download_hdb5_std5(target)
    assert not target.exists()
    # And no staging debris either.
    assert list((tmp_path / "raw").glob("*")) == []


def test_download_can_skip_verification_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    class _FakeResponse:
        def read(self) -> bytes:
            return b"TOK,TAUTH\nJET,0.5\n"

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    target = tmp_path / "raw" / hdb5.DEFAULT_HDB5_FILENAME
    assert hdb5.download_hdb5_std5(target, verify=False) == target
    assert target.exists()


def test_an_existing_file_is_verified_even_when_the_download_is_skipped(
    tmp_path: Path,
) -> None:
    """``download`` on an already-present file must not silently trust it."""
    target = tmp_path / "hdb5_std5.csv"
    target.write_bytes(b"stale wrong bytes")
    with pytest.raises(hdb5.DatasetIntegrityError):
        hdb5.download_hdb5_std5(target)


# --- provenance stamped into results ---------------------------------------


def test_provenance_carries_the_digest_and_the_pin_it_was_compared_against(
    tmp_path: Path,
) -> None:
    """Results carry the hash of the bytes they came from, not just a filename."""
    target = tmp_path / "other.csv"
    pd.DataFrame({"TOK": ["JET"], "TAUTH": [0.5]}).to_csv(target, index=False)
    provenance = hdb5.dataset_provenance(target)

    assert provenance["sha256"] == hdb5.sha256_of_file(target)
    assert provenance["pinned_sha256"] == hdb5.HDB5_STD5_SHA256
    assert provenance["matches_pin"] is False
    assert provenance["n_rows"] == 1


# --- the stamp reaching the published artifacts -----------------------------
#
# The pin is only half of the guarantee. It stops the *wrong* file being
# analysed; the stamp is what lets a reader of ``results/`` check which file was
# analysed without taking the repository's word for it. A pin with no stamp
# means the numbers and their provenance can drift apart silently, which is the
# failure the pin exists to prevent, one step later.


def test_every_analysis_result_carries_a_dataset_fingerprint() -> None:
    """Each generated JSON under ``results/`` must name the bytes behind it.

    Checked against the files as committed rather than by re-running the
    analyses: the question is whether the *published* numbers are traceable, and
    a regenerated payload would answer a different question.
    """
    results_dir = Path(__file__).resolve().parents[1] / "results"
    # ``size_extrapolation`` predates the shared key and uses its own name; both
    # are the same ``dataset_provenance`` payload.
    expected = {
        "analysis.json": "dataset",
        "extrapolation.json": "dataset",
        "flexibility_sweep.json": "dataset",
        "size_extrapolation.json": "provenance",
    }
    for filename, key in expected.items():
        path = results_dir / filename
        if not path.exists():
            pytest.skip(f"{filename} not generated; run the analysis scripts.")
        payload = json.loads(path.read_text())
        assert key in payload, filename
        stamp = payload[key]
        assert stamp["sha256"] == hdb5.HDB5_STD5_SHA256, filename
        assert stamp["pinned_sha256"] == hdb5.HDB5_STD5_SHA256, filename
        assert stamp["matches_pin"] is True, filename
        assert stamp["n_bytes"] == hdb5.HDB5_STD5_N_BYTES, filename
        assert (stamp["n_rows"], stamp["n_columns"]) == hdb5.HDB5_STD5_RAW_SHAPE, filename


def test_a_stamp_records_the_file_the_analysis_read_not_the_default() -> None:
    """``dataset_provenance`` must follow an explicit path.

    Analyses accept a ``dataset_path``; if the stamp ignored it and fingerprinted
    the default location instead, a run on a deliberately different revision
    would publish numbers labelled with the pinned digest. That is worse than no
    stamp at all.
    """
    real = _real_dataset_or_skip()
    assert hdb5.dataset_provenance(real)["matches_pin"] is True
    assert hdb5.dataset_provenance()["path"] == str(real)
