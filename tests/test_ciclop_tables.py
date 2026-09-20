"""No CICLOP number reaches the manuscript unless the code generated it.

``tools/ciclop_tables.py`` formats every quantity the CICLOP artifact carries,
once, and scans the manuscript's marked CICLOP passages for any numeral that is
not one of them. These tests hold the scan to account on text written for the
purpose, and then turn it on the real manuscript.

The last test has two states and no third. Before ``results/ciclop.json``
exists, the manuscript may not mention CICLOP at all, so nothing can be written
about a result nobody has. Once it exists, every mention has to sit inside a
marked passage and every numeral there has to be a generated fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from test_ciclop import _freeze_and_pin, _small_zoo

import analysis_ciclop as ac
import ciclop
from tools import ciclop_tables as ct

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, str]]]:
    """The planted synthetic run, its frozen plan, and the facts generated from it."""
    tmp_path = tmp_path_factory.mktemp("ciclop_tables")
    with pytest.MonkeyPatch.context() as monkeypatch:
        data, plan_path = _freeze_and_pin(monkeypatch, tmp_path)
        monkeypatch.setattr(ac, "build_zoo", _small_zoo)
        plan, resolved = ciclop.load_frozen_plan(plan_path, data_path=data)
        analysis = ac.analyze(plan, ciclop.frame_from_plan(plan, resolved), with_hdb5=False, with_gp=False)
        # Through JSON and back, because the manuscript is written from the file on disk.
        analysis = json.loads(json.dumps(analysis, default=str))
        yield analysis, plan, ct.facts(analysis)


def _block(text: str) -> str:
    return f"{ct.BEGIN}\n{text}\n{ct.END}\n"


# --- the facts ------------------------------------------------------------------


def test_every_fact_is_the_artifact_value_as_it_is_to_be_printed(generated: tuple) -> None:
    analysis, _, facts = generated
    forest = analysis["primary"]["models"]["random_forest"]
    pair = analysis["primary"]["pairs"]["random_forest_vs_ridge_loglinear"]
    assert facts["ciclop.random_forest.lodo"] == f"{forest['lodo']:.3f}"
    assert facts["ciclop.random_forest.transfer_ratio"] == f"{forest['transfer_ratio']:.2f}"
    assert facts["ciclop.random_forest_vs_ridge_loglinear.gap"] == f"{pair['mean_gap']:+.3f}"
    assert facts["ciclop.random_forest_vs_ridge_loglinear.n_worse"] == str(pair["n_devices_flexible_worse"])
    assert facts["ciclop.n_devices"] == "7" and facts["pulses.n_complete"] == "177"
    assert facts["verdict"] == ac.REPRODUCES_INVERSION
    assert all(isinstance(value, str) and value for value in facts.values())


def test_the_locks_constants_are_read_off_the_code_that_owns_them(generated: tuple) -> None:
    _, _, facts = generated
    assert facts["rule.min_rows"] == "10" and facts["rule.sensitivity_min_rows"] == "30"
    assert facts["rule.degradation_boundary"] == "1.5" and facts["rule.usable_pct"] == "90"
    assert facts["rule.bootstrap_resamples"] == "2000" and facts["rule.bootstrap_seed"] == "20240617"
    assert facts["rule.cv_folds"] == "5"


def test_the_generated_tables_print_nothing_the_facts_do_not_carry(generated: tuple) -> None:
    analysis, plan, facts = generated
    for table in (ct.result_table(analysis), ct.population_table(analysis), ct.mapping_table(plan)):
        assert ct.unbound_numerals(_block(table), facts) == [], table


def test_the_mapping_table_says_which_rank_each_quantity_took(generated: tuple) -> None:
    _, plan, _ = generated
    table = ct.mapping_table(plan)
    assert "injected additional power" in table and "line-averaged electron density" in table
    assert r"\texttt{Pinj [MW]}" in table and "a/R" in table


# --- the scan -------------------------------------------------------------------


def test_a_number_nobody_generated_is_caught(generated: tuple) -> None:
    _, _, facts = generated
    lodo = facts["ciclop.random_forest.lodo"]
    assert ct.unbound_numerals(_block(f"The forest scores {lodo}."), facts) == []
    assert ct.unbound_numerals(_block(f"The forest scores {lodo}, against 0.999 before."), facts) == ["0.999"]


def test_a_sign_is_optional_in_prose_and_a_wrong_sign_is_not(generated: tuple) -> None:
    _, _, facts = generated
    gap = facts["ciclop.random_forest_vs_ridge_loglinear.gap"]
    assert gap.startswith("+")
    assert ct.unbound_numerals(_block(f"a mean gap of ${gap}$, or {gap[1:]} in words"), facts) == []
    assert ct.unbound_numerals(_block(f"a mean gap of $-{gap[1:]}$"), facts) == [f"-{gap[1:]}"]


def test_a_number_outside_a_marked_passage_is_not_this_tests_business(generated: tuple) -> None:
    _, _, facts = generated
    assert ct.unbound_numerals("The forest is 29\\% better on HDB5, 0.128 against 0.181.", facts) == []


def test_names_and_units_that_contain_digits_are_not_numbers() -> None:
    # Its own facts, so that no digit below is allowed by coincidence with the fixture's.
    facts = {"only": "0.5"}
    names = "HDB5, STD5, DB5.2.3, IPB98(y,2), H98(y,2), ITER89-P, JT-60U, W7-X, SHA-256, Sec.~S17"
    units = r"density in $10^{19}$~m$^{-3}$, power in \si{\mega\watt}, $R^2$"
    assert ct.unbound_numerals(_block(f"{names}; {units}"), facts) == []


def test_a_fitted_exponent_is_data_and_is_not_waved_through_as_a_unit() -> None:
    assert ct.unbound_numerals(_block(r"$\tau \propto R^{1.97}$ and $I_p^{0.93}$"), {"only": "0.5"}) == ["1.97", "0.93"]


def test_layout_that_carries_digits_is_not_data(generated: tuple) -> None:
    _, _, facts = generated
    latex = "\n".join([
        r"\begin{table}[t]", r"\begin{tabular}{lrrrr}", r"\multicolumn{6}{l}{\emph{reference}} \\",
        r"\cmidrule(lr){2-3}", r"\label{tab:ciclop2}", r"Table~\ref{tab:ciclop2} and Ref.~\cite{litaudon24}.",
        r"\includegraphics[width=0.9\linewidth]{ciclop}", r"\href{https://doi.org/10.1088/1741-4326/ae89cc}{doi}",
        r"\subsection{Primary result}", "% a comment with 12345 in it",
    ])
    assert ct.unbound_numerals(_block(latex), facts) == []


def test_a_hash_a_version_and_a_date_are_matched_whole() -> None:
    # Its own facts again: the fixture's device counts would allow a stray small digit.
    sha = "4ad9e7a9d721a87756a885941523566d102713a4aa995da6356d32e7256751c5"
    facts = {"file.version": "7.3", "file.access_date": "3 October 2026", "file.sha256": sha}
    text = rf"version 7.3, accessed 3 October 2026, SHA-256 \texttt{{{sha}}}"
    assert ct.unbound_numerals(_block(text), facts) == []
    assert ct.unbound_numerals(_block("version 7.4, accessed 4 October 2026"), facts) == ["7.4", "4", "2026"]


def test_the_provenance_facts_come_from_the_frozen_plan(generated: tuple) -> None:
    analysis, plan, facts = generated
    assert facts["file.sha256"] == plan["file"]["sha256"] == analysis["provenance"]["file"]["sha256"]
    assert facts["file.version"] == "synthetic" and len(facts["plan.sha256"]) == 64


def test_a_passage_that_is_opened_and_never_closed_is_an_error() -> None:
    with pytest.raises(ValueError, match="never closed"):
        ct.marked_blocks(f"{ct.BEGIN}\nsome text\n")
    with pytest.raises(ValueError, match="no `% ciclop:begin` before it"):
        ct.marked_blocks(f"some text\n{ct.END}\n")


def test_ciclop_may_only_be_named_inside_a_marked_passage_or_the_bibliography() -> None:
    latex = "\n".join([
        "The fusion evidence rests on one ITPA file.",
        "A CICLOP sentence nobody marked.",
        "% CICLOP in a comment is not prose",
        ct.BEGIN, "CICLOP reproduced the degradation.", ct.END,
        r"\begin{thebibliography}{99}", r"\bibitem{litaudon26} the CICLOP database", r"\end{thebibliography}",
    ])
    assert ct.mentions_outside_blocks(latex) == ["line 2: A CICLOP sentence nobody marked."]


# --- the real manuscript --------------------------------------------------------


@pytest.mark.parametrize("document", ct.DOCUMENTS, ids=lambda path: path.name)
def test_the_manuscript_prints_no_ciclop_number_the_code_did_not_generate(document: Path) -> None:
    latex = document.read_text(encoding="utf-8")
    if not ct.RESULTS.exists():
        assert "CICLOP" not in latex and ct.BEGIN not in latex, (
            f"{document.name} mentions CICLOP, and results/ciclop.json does not exist. Nothing can be "
            "written about a replication before its result has been generated."
        )
        return
    known = ct.facts(json.loads(ct.RESULTS.read_text(encoding="utf-8")))
    assert ct.mentions_outside_blocks(latex) == [], "mark these with `% ciclop:begin` and `% ciclop:end`"
    loose = ct.unbound_numerals(latex, known)
    assert loose == [], (
        f"{document.name} prints {loose} in a CICLOP passage, and results/ciclop.json carries no such "
        "value. Paste from `python3 tools/ciclop_tables.py`, spell small counts out, or add the fact."
    )
