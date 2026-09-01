"""Every headline number in the prose must match the generated artifacts.

The figures quoted in ``README.md``, ``results/RESULTS.md``, the paper and the
site template are typed by hand. ``site/build_page.py`` is the one consumer that
reads them out of ``results/`` programmatically, so before this module a rerun of
an analysis could change a number in ``results/*.json`` and leave every prose
copy of it silently stale.

Each claim below binds one literal string, exactly as it is written in the prose,
to the artifact field it came from. The test renders the artifact value and
asserts the two agree *and* that the string is actually present in the documents
that are supposed to carry it. So a claim fails in both directions: if an
analysis is rerun and its output moves, the rendered value stops matching the
literal; if someone edits a number in prose, it stops matching the artifact.

These tests read only committed files under ``results/``, so they need neither
the HDB5 download nor a training run and are part of the ordinary suite. The
separate question of whether ``results/`` itself still reproduces from the raw
data is answered by the ``reproduce`` CI job, which regenerates it and diffs.

When one of these fails, rerun the analysis that owns the artifact, then update
the prose to whatever the new value is. Do not adjust the tolerance to make it
pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

README = "README.md"
RESULTS_MD = "results/RESULTS.md"
PAGE = "site/page.template.html"
PAPER = "paper/paper.tex"
PAPER_PDF = "paper/paper.pdf"
ZENODO = ".zenodo.json"


# --------------------------------------------------------------------------
# Artifact readers. Each is called once and cached, so a malformed artifact
# fails loudly here rather than as a confusing mismatch in every claim.
# --------------------------------------------------------------------------
def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _csv(name: str, index: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name).set_index(index)


@pytest.fixture(scope="module")
def artifacts() -> dict[str, object]:
    return {
        "extrapolation": _json("extrapolation.json"),
        "size": _json("size_extrapolation.json"),
        "conformal": _json("conformal.json"),
        "analysis": _json("analysis.json"),
        "summary": _csv("extrapolation_summary.csv", "model_name"),
        "escalation": _csv("size_extrapolation_escalation.csv", "model_name"),
        "coverage": _csv("conformal_summary.csv", "model_name"),
        "dimensional": _json("dimensional.json"),
        "dim_splits": _csv("dimensional_splits.csv", "model_name"),
        "dim_constraints": pd.read_csv(RESULTS / "dimensional_constraints.csv"),
        "shift": pd.read_csv(RESULTS / "conformal_shift_summary.csv"),
        "replication": _json("replication.json"),
        "forecast": _json("forecast.json"),
    }


@pytest.fixture(scope="module")
def documents() -> dict[str, str]:
    """Documents with whitespace collapsed, so a claim survives line wrapping.

    ``results/RESULTS.md`` is hard-wrapped at about 72 columns, so a phrase like
    "13 of 13 machines" is routinely split across a newline. Collapsing runs of
    whitespace to a single space lets a claim be written the way it reads rather
    than the way it happens to be wrapped today.
    """
    out = {}
    for name in (README, RESULTS_MD, PAGE, PAPER, ZENODO):
        path = ROOT / name
        if not path.exists():
            continue
        text = re.sub(r"\s+", " ", path.read_text())
        if name == PAPER:
            # LaTeX starts a comment at a bare %, so every percentage in the
            # paper is written \%. Unescape it so one claim can cover the paper
            # and the prose documents without carrying two spellings.
            text = text.replace(r"\%", "%")
        out[name] = text

    pdf = _extract_pdf_text(ROOT / PAPER_PDF)
    if pdf is not None:
        out[PAPER_PDF] = pdf
    return out


def _extract_pdf_text(path: Path) -> Union[str, None]:
    """Text of the committed paper PDF, or None if it cannot be read.

    The PDF is the artifact people actually download, and it is committed rather
    than built on demand, so it can go stale the moment paper.tex is corrected
    and not rebuilt. Reading it back is the only check that catches that.

    Returns None rather than raising when pypdf is missing, so a checkout
    without the dev extra skips this document instead of failing on it.
    """
    if not path.exists():
        return None
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    text = " ".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    # The typeset minus is U+2212, not the ASCII hyphen the sources are written
    # with, so a claim like "-0.06" would never match without folding it.
    text = text.replace("\u2212", "-")
    return re.sub(r"\s+", " ", text)


Reader = Callable[[dict], Union[float, int]]


@dataclass(frozen=True)
class Claim:
    """One number in the prose, bound to the artifact field it came from."""

    label: str
    literal: str
    """Exactly how the number is written in the documents."""
    read: Reader
    """Pulls the value out of the artifacts."""
    render: Callable[[Union[float, int]], str] = str
    """Turns that value into the literal. Must round the way the prose does."""
    documents: tuple[str, ...] = field(default=(README, RESULTS_MD, PAPER, PAPER_PDF))
    phrases: Callable[[str], tuple[str, ...]] = lambda literal: (literal,)
    """Spellings that count as this claim appearing, built from the literal.

    A bare number is not always a safe thing to search for. "41%" occurs three
    times in the README and only one of them is this quantity, so a claim
    matching the bare string would keep passing after the number it guards had
    changed. Where a number is ambiguous in the documents, the claim carries the
    surrounding words instead, and a document satisfies it by containing any one
    of the spellings (the documents word it differently: "keep", "retain", and
    the two orderings).
    """


def _r(digits: int) -> Callable[[Union[float, int]], str]:
    return lambda v: f"{v:.{digits}f}"


def _pct(digits: int = 0) -> Callable[[Union[float, int]], str]:
    return lambda v: f"{v * 100:.{digits}f}%"


def _sm(a: dict, model: str, column: str) -> float:
    return float(a["summary"].loc[model, column])


def _esc(a: dict, model: str, column: str) -> float:
    return float(a["escalation"].loc[model, column])


def _cov(a: dict, model: str, column: str) -> float:
    return float(a["coverage"].loc[model, column])


def _dim(a: dict, model: str, column: str) -> float:
    return float(a["dim_splits"].loc[model, column])


def _constraint(a: dict, source: str, model: str) -> float:
    frame = a["dim_constraints"]
    row = frame[(frame["exponent_source"] == source) & (frame["constraint_model"] == model)]
    if row.empty:
        raise AssertionError(f"no constraint distance for {source} against {model}")
    return float(row["residual_norm"].iloc[0])


def _shift(a: dict, model: str, method: str, column: str) -> float:
    frame = a["shift"]
    row = frame[(frame["model_name"] == model) & (frame["method"] == method)]
    if row.empty:
        raise AssertionError(f"no coverage row for {model} under {method}")
    return float(row[column].iloc[0])


def _arm(a: dict, name: str) -> dict:
    for arm in a["replication"]["arms"]:
        if arm["arm"] == name:
            return arm
    raise AssertionError(f"no replication arm named {name}")


def _cast(a: dict, device: str, model: str) -> float:
    for row in a["forecast"]["forecasts"]:
        if row["device"] == device and row["model_name"] == model:
            return float(row["tau_predicted_s"])
    raise AssertionError(f"no forecast for {model} on {device}")


def _spread(a: dict, device: str) -> float:
    """How far apart the models are on one device, max over min.

    This is the number the forecast table exists to produce, and before it was
    bound here the prose carried 8.1 while the artifact said 8.3: it is a ratio
    of two other reported values rather than a field of its own, so nothing
    recomputed it when the forecast was regenerated.
    """
    taus = [
        float(row["tau_predicted_s"])
        for row in a["forecast"]["forecasts"]
        if row["device"] == device
    ]
    assert taus, f"no forecasts for {device}"
    return max(taus) / min(taus)


def _paired(a: dict, model_a: str, model_b: str) -> dict:
    for row in a["extrapolation"]["paired_differences"]:
        if row["model_a"] == model_a and row["model_b"] == model_b:
            return row
    raise AssertionError(f"no paired difference for {model_a} against {model_b}")


# Results 8 to 12, which the paper carries alongside the prose documents. The
# site template is not in this tuple: ``site/build_page.py`` reads those numbers
# out of ``results/`` and substitutes them, so the template holds placeholders
# rather than literals and there is nothing there to go stale.
LATE_RESULTS = (README, RESULTS_MD, PAPER, PAPER_PDF)

CLAIMS: tuple[Claim, ...] = (
    # -- dataset scale -----------------------------------------------------
    Claim("rows in the database", "6228", lambda a: a["extrapolation"]["n_rows"],
          documents=(README, RESULTS_MD, PAGE, PAPER, PAPER_PDF)),
    Claim("machines scored under leave-one-out", "13",
          lambda a: a["extrapolation"]["n_machines_held_out"]),

    # -- Result 4: the reversal -------------------------------------------
    Claim("random forest, CV", "0.128", lambda a: _sm(a, "random_forest", "cv_rmsle"), _r(3)),
    Claim("random forest, leave-one-machine-out", "0.465",
          lambda a: _sm(a, "random_forest", "lomo_mean_rmsle"), _r(3)),
    Claim("random forest degradation", "3.6",
          lambda a: _sm(a, "random_forest", "degradation_factor"), _r(1)),
    Claim("gradient booster, CV", "0.130",
          lambda a: _sm(a, "hist_gradient_boosting", "cv_rmsle"), _r(3)),
    Claim("gradient booster, leave-one-machine-out", "0.359",
          lambda a: _sm(a, "hist_gradient_boosting", "lomo_mean_rmsle"), _r(3)),
    # The log-quadratic control is a Result 4d row; the README table carries
    # only the three contenders and the baseline.
    Claim("log-quadratic control, leave-one-machine-out", "0.300",
          lambda a: _sm(a, "ridge_log_quadratic", "lomo_mean_rmsle"), _r(3),
          documents=(RESULTS_MD, PAPER, PAPER_PDF)),
    Claim("power law, CV", "0.181", lambda a: _sm(a, "ridge_loglinear", "cv_rmsle"), _r(3)),
    Claim("power law, leave-one-machine-out", "0.214",
          lambda a: _sm(a, "ridge_loglinear", "lomo_mean_rmsle"), _r(3)),
    Claim("IPB98(y,2), CV", "0.199",
          lambda a: _sm(a, "ipb98y2_analytic", "cv_rmsle"), _r(3)),
    Claim("IPB98(y,2), leave-one-machine-out", "0.188",
          lambda a: _sm(a, "ipb98y2_analytic", "lomo_mean_rmsle"), _r(3)),

    # -- Result 4b: error against distance --------------------------------
    Claim("forest error against distance", "+0.85",
          lambda a: _sm(a, "random_forest", "distance_spearman"),
          lambda v: f"{v:+.2f}", documents=(README, RESULTS_MD, PAPER, PAPER_PDF)),
    Claim("power law error against distance", "-0.06",
          lambda a: _sm(a, "ridge_loglinear", "distance_spearman"),
          lambda v: f"{v:+.2f}".replace("+", "-") if v < 0 else f"{v:+.2f}"),

    # -- Result 4: paired by machine --------------------------------------
    # README writes "13 of 13 machines" in prose, RESULTS.md "**13 of 13**" in
    # a table cell; the shared substring is what both must carry.
    Claim("machines where the forest is worse", "13 of 13",
          lambda a: _paired(a, "random_forest", "ridge_loglinear")["n_machines_a_worse"],
          lambda v: f"{v} of {v}",
          documents=(README, RESULTS_MD, PAPER, PAPER_PDF, ZENODO)),
    Claim("paired gap, forest against power law", "+0.251",
          lambda a: _paired(a, "random_forest", "ridge_loglinear")["mean_difference"],
          lambda v: f"{v:+.3f}"),
    Claim("paired gap, lower bound", "+0.157",
          lambda a: _paired(a, "random_forest", "ridge_loglinear")["ci_low"],
          lambda v: f"{v:+.3f}"),
    Claim("paired gap, upper bound", "+0.342",
          lambda a: _paired(a, "random_forest", "ridge_loglinear")["ci_high"],
          lambda v: f"{v:+.3f}"),

    # -- Result 5: the ITER-matched size cut -------------------------------
    Claim("size ratio of the matched cut", "1.823",
          lambda a: a["size"]["iter_matched_split"]["size_ratio"], _r(3),
          documents=(RESULTS_MD, PAGE, PAPER, PAPER_PDF)),
    Claim("size ratio ITER asks for", "1.824",
          lambda a: a["size"]["iter_size_ratio"], _r(3), documents=(RESULTS_MD, PAGE, PAPER, PAPER_PDF)),
    Claim("largest training machine", "1.865",
          lambda a: a["size"]["iter_matched_split"]["train_r_max_m"], _r(3),
          documents=(RESULTS_MD, PAGE, PAPER, PAPER_PDF)),
    Claim("largest held-out machine", "3.40",
          lambda a: a["size"]["iter_matched_split"]["test_r_max_m"], _r(2),
          documents=(RESULTS_MD, PAGE, PAPER, PAPER_PDF)),
    Claim("power law skill at the cut", "93%",
          lambda a: _esc(a, "ridge_loglinear", "skill_against_baseline"), _pct(),
          documents=(README, RESULTS_MD, PAGE, PAPER, PAPER_PDF),
          phrases=lambda n: (f"keeps {n}", f"retains {n}")),
    Claim("forest skill at the cut", "41%",
          lambda a: _esc(a, "random_forest", "skill_against_baseline"), _pct(),
          documents=(README, RESULTS_MD, PAGE, PAPER, PAPER_PDF),
          phrases=lambda n: (f"keep 31% and {n}", f"keep {n} and 31%", f"retain {n} and 31%",
                             f"keep 41% and {n}", f"keep {n} and 41%", f"retain 41% and {n}")),
    Claim("gradient booster skill at the cut", "31%",
          lambda a: _esc(a, "hist_gradient_boosting", "skill_against_baseline"), _pct(),
          documents=(README, RESULTS_MD, PAGE, PAPER, PAPER_PDF),
          phrases=lambda n: (f"keep 31% and {n}", f"keep {n} and 31%", f"retain {n} and 31%",
                             f"keep 41% and {n}", f"keep {n} and 41%", f"retain 41% and {n}")),

    # -- Result 7: the intervals -------------------------------------------
    Claim("nominal coverage", "90%",
          lambda a: a["conformal"]["nominal_coverage"], _pct(),
          documents=(README, RESULTS_MD, PAGE, PAPER, PAPER_PDF, ZENODO)),
    Claim("forest coverage on an unseen machine", "35%",
          lambda a: _cov(a, "random_forest", "lomo_coverage"), _pct(),
          documents=(README, RESULTS_MD, PAPER, PAPER_PDF, ZENODO)),
    Claim("forest coverage across the size cut", "3%",
          lambda a: _cov(a, "random_forest", "size_cut_coverage"), _pct(),
          documents=(README, RESULTS_MD, PAPER, PAPER_PDF, ZENODO)),

    # -- Result 1: the rank audit ------------------------------------------
    Claim("rank of the standardized feature matrix", "rank 8",
          lambda a: a["analysis"]["rank_audit"]["rank"], lambda v: f"rank {v}"),
    Claim("rank deficiency", "2",
          lambda a: a["analysis"]["rank_audit"]["rank_deficiency"]),
    # -- Result 8: physics as a constraint --------------------------------
    Claim("collisionless power law at the ITER-matched cut", "0.183",
          lambda a: _dim(a, "powerlaw_collisionless", "size_cut_rmsle"), _r(3),
          documents=LATE_RESULTS),
    Claim("IPB98 distance from the Kadomtsev surface", "0.00096",
          lambda a: _constraint(a, "ipb98y2_published", "kadomtsev"), _r(5),
          documents=LATE_RESULTS),
    Claim("IPB98 distance from the collisionless surface", "0.0045",
          lambda a: _constraint(a, "ipb98y2_published", "collisionless"), _r(4),
          documents=LATE_RESULTS),
    Claim("unconstrained fit, in sample", "0.1808",
          lambda a: float(a["dimensional"]["in_sample_rmsle"]["powerlaw_free"]), _r(4),
          documents=LATE_RESULTS),
    Claim("collisionless fit, in sample", "0.1818",
          lambda a: float(a["dimensional"]["in_sample_rmsle"]["powerlaw_collisionless"]), _r(4),
          documents=LATE_RESULTS),

    # -- Result 10: the repaired intervals --------------------------------
    Claim("random forest coverage on a held-out machine, machine-CV", "88%",
          lambda a: _shift(a, "random_forest", "machine_cv", "lomo_coverage"), _pct(0),
          documents=LATE_RESULTS),
    Claim("random forest coverage at the ITER cut, distance-scaled", "40%",
          lambda a: _shift(a, "random_forest", "machine_cv_distance", "size_cut_coverage"),
          _pct(0), documents=LATE_RESULTS),
    Claim("collisionless coverage at the ITER cut, split conformal", "91%",
          lambda a: _shift(a, "powerlaw_collisionless", "split", "size_cut_coverage"),
          _pct(0), documents=LATE_RESULTS),

    # -- Result 11: the replication ---------------------------------------
    Claim("disjoint H-mode rows", "5358",
          lambda a: _arm(a, "disjoint_h")["n_rows"], documents=LATE_RESULTS),
    Claim("non-H rows", "3860",
          lambda a: _arm(a, "non_h")["n_rows"], documents=LATE_RESULTS),
    Claim("CV gain over IPB98 on the disjoint arm", "42%",
          lambda a: _arm(a, "disjoint_h")["cv_gain_over_baseline"], _pct(0),
          documents=LATE_RESULTS),
    Claim("CV gain over ITER89-P on the non-H arm", "67%",
          lambda a: _arm(a, "non_h")["cv_gain_over_baseline"], _pct(0),
          documents=LATE_RESULTS),

    # -- Result 12: the locked forecast -----------------------------------
    Claim("largest confinement time in the training data", "1.321",
          lambda a: float(a["forecast"]["train_tau_max_s"]), _r(3),
          documents=LATE_RESULTS),
    Claim("IPB98 on ITER", "3.591",
          lambda a: _cast(a, "ITER", "ipb98y2_analytic"), _r(3),
          documents=LATE_RESULTS),
    Claim("random forest on ITER", "0.435",
          lambda a: _cast(a, "ITER", "random_forest"), _r(3),
          documents=LATE_RESULTS),
    Claim("collisionless power law on ITER", "2.837",
          lambda a: _cast(a, "ITER", "powerlaw_collisionless"), _r(3),
          documents=LATE_RESULTS),
    Claim("model disagreement at ITER", "8.3",
          lambda a: _spread(a, "ITER"), _r(1),
          documents=LATE_RESULTS,
          phrases=lambda lit: (f"a factor of **{lit}**", f"a factor of {lit}")),
    Claim("model agreement on JT-60SA", "15%",
          # Rendered as the excess over perfect agreement, because that is how
          # the prose says it: "agree to within 15%".
          lambda a: _spread(a, "JT-60SA") - 1.0, _pct(0),
          documents=LATE_RESULTS,
          phrases=lambda lit: (f"agree to within {lit}",)),
)


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.label)
def test_reported_number_matches_its_artifact(claim: Claim, artifacts: dict) -> None:
    """The literal in the prose is what the artifact actually says."""
    rendered = claim.render(claim.read(artifacts))
    assert rendered == claim.literal, (
        f"{claim.label}: the documents say {claim.literal!r} but the generated "
        f"artifact now renders {rendered!r}. Rerun the owning analysis, then "
        f"update the prose to the new value."
    )


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.label)
def test_reported_number_appears_in_its_documents(claim: Claim, documents: dict) -> None:
    """The number is actually written where it is supposed to be written.

    Without this half a claim could pass while quietly guarding nothing, because
    the prose it describes had been reworded out of existence.
    """
    for name in claim.documents:
        if name not in documents:
            pytest.skip(f"{name} is not present in this checkout")
        spellings = claim.phrases(claim.literal)
        assert any(p in documents[name] for p in spellings), (
            f"{claim.label}: none of {spellings!r} is present in {name}. "
            f"Either the prose was reworded, in which case update this claim, or "
            f"the number was dropped and the claim is guarding nothing."
        )
