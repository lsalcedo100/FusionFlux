"""Tests for Result 11: the replication on rows STD5 does not contain.

This result has exactly one way to be silently fake, and it is worth naming
plainly. If the row match against STD5 fails, every STD5 row stays in the
"disjoint" arm, the arm becomes a superset of the data Result 4 was computed on,
and it reproduces Result 4 *by construction*. The output would look like a
successful replication and would mean nothing.

A match can fail silently in several ways here: the two exports carry ``TIME``
at different float precisions, ``SHOT`` is an integer in one and could parse as
a float in the other, and a typo in a column name would raise rather than
mismatch only if the column is actually read. So the tests below check the
matcher positively (STD5 matched against itself must overlap completely) as well
as negatively (the disjoint arm must share nothing).

The second risk is units. DB5.2.3 stores amperes and m^-3 where STD5 stores
megaamperes and 1e19 m^-3, and getting that wrong does not crash: it produces
IPB98(y,2) predictions of 3e8 seconds, which a pipeline will happily carry into
a plot. The conversion is therefore checked against the rows the two files
share, which is the same evidence it was derived from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hdb5
import replication as rp


def _db523_or_skip() -> pd.DataFrame:
    if not rp.default_db523_path().exists():
        pytest.skip(
            "DB5.2.3 not downloaded; run "
            "`python3 -c 'import replication; replication.download_db523()'`."
        )
    return rp.load_db523_raw()


def _std5_or_skip() -> pd.DataFrame:
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    return hdb5.load_hdb5_dataframe()


def _arms_or_skip() -> dict[str, rp.ReplicationArm]:
    """Both arms, or a skip if either input file is absent.

    ``build_replication_arms`` reads DB5.2.3 *and* STD5, since the disjoint arm
    is defined by subtracting one from the other. Calling it without a guard
    raises rather than skips, which turns the dataset-free environment CI runs
    in into three red tests instead of three skipped ones.
    """
    _db523_or_skip()
    _std5_or_skip()
    return rp.build_replication_arms()


# --- the row matcher, which the whole result depends on ---------------------


def test_match_key_is_reflexive_on_std5() -> None:
    """STD5 matched against itself must overlap completely.

    The negative check (the disjoint arm shares no rows) passes trivially if the
    matcher never matches anything, so this positive check is the one that gives
    it meaning.
    """
    std5 = _std5_or_skip()
    keys = rp._match_keys(std5)
    assert keys.nunique() == len(std5)
    assert set(keys) == set(rp._match_keys(std5.copy()))


def test_match_key_survives_a_float_precision_difference() -> None:
    """Rounding to a fixed number of decimals is what makes the match work.

    The two exports write ``TIME`` at different precisions. Without the rounding
    an exact float compare declares every row disjoint, which is precisely the
    silent failure this result cannot afford.
    """
    frame = pd.DataFrame({"TOK": ["JET"], "SHOT": [12345], "TIME": [1.2345678901]})
    perturbed = frame.assign(TIME=[1.23456789009])
    assert set(rp._match_keys(frame)) == set(rp._match_keys(perturbed))


def test_the_two_files_do_overlap_substantially() -> None:
    """STD5 is a selection out of DB5.2.3, so thousands of rows must be shared.

    A matcher that silently produced no overlap at all would pass the disjointness
    assertion below; this is the check that the overlap is real.
    """
    raw = _db523_or_skip()
    std5 = _std5_or_skip()
    shared = set(rp._match_keys(raw)) & set(rp._match_keys(std5))
    assert len(shared) > 4000


# --- the arms ---------------------------------------------------------------


def test_disjoint_h_arm_shares_no_rows_with_std5() -> None:
    """The load-bearing property of Result 11, asserted directly."""
    arms = _arms_or_skip()
    assert arms["disjoint_h"].n_rows_shared_with_std5 == 0


def test_non_h_arm_contains_no_h_mode_rows() -> None:
    """The regime arm must actually change regime.

    STD5 is entirely ELMy H-mode, so an H-mode row leaking into this arm would
    both break the disjointness and weaken the claim that the baseline law was
    swapped for the right reason.
    """
    raw = _db523_or_skip()
    selected = raw[raw["PHASE"].isin(rp.NON_H_PHASES)]
    assert not selected["PHASE"].astype(str).str.startswith("H").any()
    arms = _arms_or_skip()
    assert arms["non_h"].n_rows_shared_with_std5 == 0


def test_both_arms_have_enough_machines_to_hold_one_out() -> None:
    arms = _arms_or_skip()
    assert arms["disjoint_h"].n_machines_scored >= 10
    # Five is too few to carry a claim and is reported as such, but it must at
    # least be enough to run the split at all.
    assert arms["non_h"].n_machines_scored >= 2


# --- units ------------------------------------------------------------------


def test_unit_conversions_agree_with_std5_on_shared_rows() -> None:
    """The three conversion factors, checked against the evidence they came from.

    On rows both files contain, the converted DB5.2.3 columns must match STD5's
    to well inside any plausible revision difference. ``BT``, ``RGEO``,
    ``KAPPAA`` and ``MEFF`` are checked too, as the control: they need no
    conversion, so a spurious factor applied to everything would show up here.
    """
    raw = _db523_or_skip()
    std5 = _std5_or_skip()
    converted = raw.copy()
    for column, scale in rp.DB523_UNIT_SCALES.items():
        converted[column] = pd.to_numeric(converted[column], errors="coerce") / scale
    converted["key"] = rp._match_keys(raw)
    std5 = std5.assign(key=rp._match_keys(std5))
    merged = converted.merge(std5, on="key", suffixes=("_db", "_std"))
    assert len(merged) > 4000

    for column in ("IP", "NEL", "BT", "RGEO", "KAPPAA", "MEFF"):
        left = pd.to_numeric(merged[f"{column}_db"], errors="coerce").abs()
        right = pd.to_numeric(merged[f"{column}_std"], errors="coerce").abs()
        ratio = (left / right).replace([np.inf, -np.inf], np.nan).dropna()
        assert ratio.median() == pytest.approx(1.0, rel=1e-6), column


def test_derived_inverse_aspect_ratio_matches_std5() -> None:
    """``AMIN / RGEO`` must reproduce STD5's own ``EPS``, which has no DB5.2.3 column."""
    raw = _db523_or_skip()
    std5 = _std5_or_skip()
    derived = pd.to_numeric(raw["AMIN"], errors="coerce") / pd.to_numeric(
        raw["RGEO"], errors="coerce"
    )
    merged = (
        raw.assign(key=rp._match_keys(raw), derived_eps=derived)
        .merge(std5.assign(key=rp._match_keys(std5)), on="key")
    )
    difference = (merged["derived_eps"] - pd.to_numeric(merged["EPS"], errors="coerce")).abs()
    assert difference.max() < 1e-6


def test_iter89p_matches_a_hand_evaluated_case() -> None:
    """The L-mode baseline, checked against the formula written out by hand.

    Unit inputs everywhere except density, so every power is 1 except the
    density term, and the answer is the coefficient times ``0.1 ** 0.1``.
    """
    frame = pd.DataFrame(
        {
            "ip_ma": [1.0],
            "bt_t": [1.0],
            "ne_line_1e19_m3": [1.0],
            "p_loss_mw": [1.0],
            "r_m": [1.0],
            "a_m": [1.0],
            "kappa": [1.0],
            "m_eff_amu": [1.0],
        }
    )
    expected = 0.048 * (0.1**0.1)
    assert float(rp.iter89p_tau_s(frame).iloc[0]) == pytest.approx(expected, rel=1e-12)


def test_iter89p_predicts_the_right_order_of_magnitude_on_real_l_mode_rows() -> None:
    """A wrong unit anywhere would move this by orders of magnitude, not percent."""
    arm = _arms_or_skip()["non_h"]
    dataset = arm.dataset
    ratio = dataset["iter89p_tau_s"] / dataset[hdb5.TARGET_COLUMN]
    assert 0.2 < float(ratio.median()) < 5.0


# --- provenance -------------------------------------------------------------


def test_pinned_digest_matches_the_file_on_disk() -> None:
    path = rp.default_db523_path()
    if not path.exists():
        pytest.skip("DB5.2.3 not downloaded.")
    fingerprint = rp.verify_db523_file(path)
    assert fingerprint.sha256 == rp.DB523_SHA256


def test_verification_rejects_a_different_file(tmp_path) -> None:
    decoy = tmp_path / "not_db523.csv"
    decoy.write_text("TOK,SHOT\nJET,1\n")
    with pytest.raises(hdb5.DatasetIntegrityError, match="integrity check failed"):
        rp.verify_db523_file(decoy)
