"""The committed membership audit supports what the Data paragraph says.

The paper says the delivered STD5 file is the ELMy H-mode part of the STD5
selection of the full DB5.2.3 file. That rests on four facts the audit records:
every delivered row matches a full-file row, every one carries the selection
flag, every one has an ELMy phase, and no non-ELMy flagged row is delivered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results" / "std5_membership.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():
        pytest.skip("no results/std5_membership.json; run `python3 tools/audit_std5_membership.py`")
    return json.loads(RESULTS.read_text())


def test_every_delivered_row_is_a_flagged_row_of_the_full_file(committed: dict) -> None:
    n = committed["delivered_rows"]
    assert n == 6228
    assert committed["delivered_rows_matched_in_full_file"] == n
    assert committed["delivered_rows_unmatched"] == 0
    assert committed["delivered_rows_with_seldb5"] == n
    assert committed["ind_is_full_file_row_number"]


def test_the_delivered_rows_are_exactly_the_elmy_phases(committed: dict) -> None:
    phases = committed["delivered_phase_counts"]
    assert set(phases) <= {"HGELM", "HSELM", "HGELMH", "HSELMH"}
    assert sum(phases.values()) == committed["delivered_rows"]
    assert committed["seldb5_non_elmy_rows_delivered"] == 0


def test_the_counts_add_up(committed: dict) -> None:
    assert committed["seldb5_elmy_rows"] + committed["seldb5_non_elmy_rows"] == committed["seldb5_rows_in_full_file"]
    assert committed["seldb5_elmy_rows_delivered"] + committed["seldb5_elmy_rows_omitted"] == committed["seldb5_elmy_rows"]
    assert committed["seldb5_elmy_rows_delivered"] == committed["delivered_rows"]
    assert len(committed["omitted_elmy_rows"]) == committed["seldb5_elmy_rows_omitted"]
