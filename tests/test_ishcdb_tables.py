"""No ISHCDB number reaches the manuscript unless the code generated it.

The same discipline as ``tests/test_ciclop_tables.py``, with one difference:
``results/ishcdb.json`` exists and is committed, so the facts are generated
from the run of record and the manuscript is checked against them directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ishcdb
from tools import ciclop_tables as ct
from tools import ishcdb_tables as it

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def generated() -> tuple[dict, dict, dict[str, str]]:
    if not it.RESULTS.exists():
        pytest.skip("no results/ishcdb.json; the run of record has not happened")
    analysis = json.loads(it.RESULTS.read_text(encoding="utf-8"))
    plan_file = json.loads(ishcdb.default_plan_path().read_text(encoding="utf-8"))
    return analysis, plan_file, it.facts(analysis, plan_file)


def _block(text: str) -> str:
    begin, end = ct.markers(it.TAG)
    return f"{begin}\n{text}\n{end}\n"


def test_the_iss04_exponents_here_are_the_ones_the_code_evaluates() -> None:
    """The dictionary exists so the law can be printed; it must be the law the module computes."""
    rng = np.random.default_rng(4)
    rows = pd.DataFrame({name: rng.lognormal(0, 0.5, 20) for name in it.ISS04_EXPONENTS})
    expected = it.ISS04_CONSTANT * np.prod([rows[name] ** e for name, e in it.ISS04_EXPONENTS.items()], axis=0)
    assert np.allclose(ishcdb.iss04_tau_s(rows), expected, rtol=1e-12)
    assert list(it.ISS04_EXPONENTS) == list(ishcdb.FEATURES)


def test_every_fact_is_the_artifact_value_as_it_is_to_be_printed(generated: tuple) -> None:
    analysis, plan_file, facts = generated
    forest = analysis["primary"]["models"]["random_forest"]
    pair = analysis["primary"]["pairs"]["random_forest_vs_ridge_loglinear"]
    assert facts["ishcdb.random_forest.lodo"] == f"{forest['lodo']:.3f}"
    assert facts["ishcdb.random_forest_vs_ridge_loglinear.gap"] == f"{pair['mean_gap']:+.3f}"
    assert facts["ishcdb.random_forest_vs_ridge_loglinear.n_worse"] == str(pair["n_devices_flexible_worse"])
    assert facts["rows.standard_set_complete"] == str(analysis["plan"]["n_complete_rows"])
    assert facts["usability.iota_23_pct"] == f"{100 * plan_file['usability']['iota_23']:.0f}"
    assert facts["verdict"] == analysis["verdict"]
    assert all(isinstance(value, str) and value for value in facts.values())


def test_the_hdb5_numbers_the_passages_quote_come_from_the_hdb5_artifacts(generated: tuple) -> None:
    _, _, facts = generated
    transfers = {t["model_name"]: t for t in json.loads(it.EXTRAPOLATION.read_text())["transfers"]}
    assert facts["hdb5.random_forest.distance_rho"] == f"{transfers['random_forest']['distance_spearman']:+.2f}"
    assert facts["hdb5.ridge_loglinear.distance_rho"] == f"{transfers['ridge_loglinear']['distance_spearman']:+.2f}"
    assert facts["hdb5.labels.n_worse"] == facts["hdb5.labels.n"] == "13"


def test_the_generated_tables_print_nothing_the_facts_do_not_carry(generated: tuple) -> None:
    analysis, _, facts = generated
    for table in (it.result_table(analysis), it.population_table(analysis), it.mapping_table(analysis), it.offset_table(analysis)):
        assert ct.unbound_numerals(_block(table), facts, it.TAG) == [], table
        # A bare percent sign would comment out the rest of a LaTeX table row.
        assert "%" not in table.replace(r"\%", "")


def test_the_scanner_uses_this_datasets_marker_and_names() -> None:
    latex = "\n".join([
        "The tokamak result. A stellarator sentence nobody fenced.",
        "% ishcdb:begin", "ISHCDB reproduced it on 7 devices.", "% ishcdb:end",
        "% ciclop:begin", "ISHCDB named inside the other dataset's fence.", "% ciclop:end",
        r"\begin{thebibliography}{99}", r"\bibitem{iss04} the International Stellarator Database", r"\end{thebibliography}",
    ])
    assert ct.mentions_outside_blocks(latex, it.NAMES, it.TAG) == [
        "line 1: The tokamak result. A stellarator sentence nobody fenced.",
        "line 6: ISHCDB named inside the other dataset's fence.",
    ]
    assert ct.unbound_numerals(latex, {"n": "7"}, it.TAG) == []
    assert ct.unbound_numerals(latex, {"n": "8"}, it.TAG) == ["7"]


@pytest.mark.parametrize("document", ct.DOCUMENTS, ids=lambda path: path.name)
def test_the_manuscript_prints_no_ishcdb_number_the_code_did_not_generate(document: Path) -> None:
    latex = document.read_text(encoding="utf-8")
    begin, _ = ct.markers(it.TAG)
    if not it.RESULTS.exists():
        assert not any(name in latex.split(r"\begin{thebibliography}")[0] for name in it.NAMES) and begin not in latex, (
            f"{document.name} mentions the stellarator database, and results/ishcdb.json does not exist."
        )
        return
    analysis = json.loads(it.RESULTS.read_text(encoding="utf-8"))
    plan_file = json.loads(ishcdb.default_plan_path().read_text(encoding="utf-8"))
    known = it.facts(analysis, plan_file)
    assert ct.mentions_outside_blocks(latex, it.NAMES, it.TAG) == [], "fence these with `% ishcdb:begin` and `% ishcdb:end`"
    loose = ct.unbound_numerals(latex, known, it.TAG)
    assert loose == [], (
        f"{document.name} prints {loose} in an ISHCDB passage, and no generated fact carries that value. "
        "Paste from `python3 tools/ishcdb_tables.py`, spell small counts out, or add the fact."
    )
