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
# The two replications outside fusion live in the Supplementary Material rather
# than in the main text, so the claims they own are bound here instead of to
# PAPER. Both are checked: the source, and the built PDF that is what a reader
# actually receives.
SUPPLEMENTARY = "paper/supplementary.tex"
SUPPLEMENTARY_PDF = "paper/supplementary.pdf"
ZENODO = ".zenodo.json"
# The standalone package's README is the whole of what a reader outside this
# repository sees, and it quotes the study's headline numbers as evidence that
# the method works. It goes stale exactly the way the study's own prose would.
SA_README = "scaling-audit/README.md"


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
        "allometry": _json("allometry.json"),
        "gp": _json("gp.json"),
        "tree": _json("tree_allometry.json"),
        "tuned": _json("tuned.json"),
        "sensitivity": _json("sensitivity.json"),
        "mixed": _json("mixed_model.json"),
        "boundedness": _json("boundedness.json"),
        "stored": _json("stored_energy.json"),
        "dimensionless": _json("dimensionless.json"),
        "mechanism": _json("mechanism.json"),
        "forecast_rows": _json("forecast.json"),
        "per_machine": _csv("extrapolation_per_machine.csv", "tokamak"),
        "conformal_per_machine": pd.read_csv(RESULTS / "conformal_per_machine.csv"),
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
    for name in (README, RESULTS_MD, PAGE, PAPER, SUPPLEMENTARY, ZENODO, SA_README):
        path = ROOT / name
        if not path.exists():
            continue
        text = re.sub(r"\s+", " ", path.read_text())
        if name in (PAPER, SUPPLEMENTARY):
            # LaTeX starts a comment at a bare %, so every percentage in the
            # paper is written \%. Unescape it so one claim can cover the paper
            # and the prose documents without carrying two spellings.
            text = text.replace(r"\%", "%")
        out[name] = text

    for name in (PAPER_PDF, SUPPLEMENTARY_PDF):
        pdf = _extract_pdf_text(ROOT / name)
        if pdf is not None:
            out[name] = pdf
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
    text = re.sub(r"\s+", " ", text)
    # Inline maths like "$\rho=+0.78$" extracts as "ρ = +0 .78": the decimal
    # point arrives as its own glyph run and a space lands in front of it. There
    # are 16 of these in the paper, and a claim on any of them would fail while
    # the number on the page is correct, which is the worst kind of failure this
    # file can produce. Only a space between a digit and a following ".digit" is
    # closed, so nothing that is really two numbers gets joined.
    return re.sub(r"(?<=\d) (?=\.\d)", "", text)


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


def _published(a: dict, name: str, column: str) -> float:
    return float(a["sensitivity"]["published_scalings"][name][column])


def _elongation(a: dict, *path: str) -> float:
    node = a["sensitivity"]["elongation_convention"]
    for key in path:
        node = node[key]
    return float(node)


def _dimensionless(a: dict, *path: str) -> float:
    node = a["dimensionless"]
    for key in path:
        node = node[key]
    return float(node)


def _mean_function(a: dict, arm: str, column: str) -> float:
    return float(a["mechanism"]["mean_function"][arm][column])


def _forecast(a: dict, model: str) -> float:
    for row in a["forecast_rows"]["forecasts"]:
        if row["device"] == "ITER" and row["model_name"] == model:
            return float(row["tau_predicted_s"])
    raise AssertionError(f"no ITER forecast for {model}")


def _stored(a: dict, *path: str) -> float:
    node = a["stored"]["arms"]["stored_energy"]
    for key in path:
        node = node[key]
    return float(node)


def _gap(a: dict, label: str) -> float:
    """The forest's margin over the power law on one held-out label.

    Read as a difference rather than stored as one, because the artifact holds
    the two scores and the paper prints the gap.
    """
    frame = a["per_machine"]
    row = frame[frame.index == label]
    scores = {name: float(value) for name, value in zip(row["model_name"], row["rmsle"], strict=True)}
    return scores["random_forest"] - scores["ridge_loglinear"]


def _headroom(a: dict, label: str) -> float:
    """How far the largest held-out target sits below the forest's training ceiling."""
    frame = a["per_machine"]
    row = frame[(frame.index == label) & (frame["model_name"] == "random_forest")]
    if row.empty:
        raise AssertionError(f"no random forest row for {label}")
    return float(row["log_target_headroom"].iloc[0])


def _booster_rows_covered(a: dict) -> int:
    """Rows of the size cut the booster's nominal 90% interval actually contains."""
    frame = a["conformal_per_machine"]
    cut = frame[(frame["split"] == "size_cut") & (frame["model_name"] == "hist_gradient_boosting")]
    return int(round(float((cut["empirical_coverage"] * cut["n_rows"]).sum())))


def _arm(a: dict, name: str) -> dict:
    for arm in a["replication"]["arms"]:
        if arm["arm"] == name:
            return arm
    raise AssertionError(f"no replication arm named {name}")


def _device_inputs(a: dict, device: str) -> dict:
    for row in a["forecast"]["devices"]:
        if row["name"] == device:
            return row
    raise AssertionError(f"no device named {device} in the forecast")


def _cast_distance(a: dict, device: str) -> float:
    for row in a["forecast"]["forecasts"]:
        if row["device"] == device:
            return float(row["feature_mahalanobis"])
    raise AssertionError(f"no forecast rows for {device}")


def _served(a: dict, device: str, model: str) -> float:
    """What the shipped predictor returns for one device, from the committed card.

    Reads ``results/predictor.json`` and the device's inputs out of
    ``results/forecast.json``, so this stays a statement about committed
    artifacts. It binds the README's worked example, which is CLI output typed
    by hand and would otherwise be the one block in the repository free to drift.
    """
    from fusionflux import predictor

    inputs = _device_inputs(a, device)
    result = predictor.predict(
        **{name: float(inputs[name]) for name in predictor.REQUIRED_INPUTS},
        card=predictor.load_card(),
    )
    for row in result.predictions:
        if row.model_name == model:
            return row.tau_s
    raise AssertionError(f"predictor did not report {model}")


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
    taus = [float(row["tau_predicted_s"]) for row in a["forecast"]["forecasts"] if row["device"] == device]
    assert taus, f"no forecasts for {device}"
    return max(taus) / min(taus)


def _rung(a: dict, n_features: int, field: str) -> float:
    """One rung of Result 15's feature ladder, keyed by predictor count."""
    return a["tree"]["ladder"][str(n_features)][field]


def _gp(a: dict, model: str, field: str) -> float:
    """One model's entry in results/gp.json, which is keyed by model then split."""
    return a["gp"]["scores"][model][field]


def _allo(a: dict, model: str, field: str) -> float:
    return float(a["allometry"]["scores"][model][field])


def _tuned(a: dict, model: str, key: str) -> float:
    return float(a["tuned"]["tuned"][model][key])


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

# Numbers whose home moved out of the main text and into the Supplementary
# Material when the paper was shortened: the replication on rows STD5 excludes
# and the locked device forecast. The main text keeps each result's headline and
# points at its supplementary section, so these particular figures live there
# now and the guard follows them rather than the other way round.
MOVED_TO_SUPPLEMENT = (README, RESULTS_MD, SUPPLEMENTARY, SUPPLEMENTARY_PDF)

# Result 13's numbers. The replication itself is in the Supplementary Material,
# not the main text, so these are bound to it rather than to the paper; where a
# document words something differently the claim carries both spellings rather
# than being dropped.
ALLOMETRY = (README, RESULTS_MD, SUPPLEMENTARY, SUPPLEMENTARY_PDF)

# Result 14 is carried by the README and the full writeup. The paper is a
# nine-page condensation that predates it and has no section for it yet.
# The paper now carries both, so its source is checked too. `PAPER_PDF` is
# deliberately absent: the committed PDF is rebuilt from this source by hand
# (`make arxiv`, then pdflatex), so listing it here would fail every run between
# an edit to paper.tex and the next rebuild. `make arxiv` gates the release on
# that rebuild instead, via `tools/check_paper_submission.py --check-pdf-fresh`.
GP = (README, RESULTS_MD, PAPER)

# Result 15 is likewise reported in the Supplementary Material rather than the
# main text.
TREE = (README, RESULTS_MD, SUPPLEMENTARY)

CLAIMS: tuple[Claim, ...] = (
    # -- dataset scale -----------------------------------------------------
    Claim(
        "rows in the database",
        "6228",
        lambda a: a["extrapolation"]["n_rows"],
        documents=(README, RESULTS_MD, PAGE, PAPER, PAPER_PDF, SA_README),
    ),
    Claim("machines scored under leave-one-out", "13", lambda a: a["extrapolation"]["n_machines_held_out"]),
    # -- Result 4: the reversal -------------------------------------------
    Claim("random forest, CV", "0.128", lambda a: _sm(a, "random_forest", "cv_rmsle"), _r(3)),
    Claim("random forest, leave-one-machine-out", "0.465", lambda a: _sm(a, "random_forest", "lomo_mean_rmsle"), _r(3)),
    Claim("random forest degradation", "3.6", lambda a: _sm(a, "random_forest", "degradation_factor"), _r(1)),
    Claim("gradient booster, CV", "0.130", lambda a: _sm(a, "hist_gradient_boosting", "cv_rmsle"), _r(3)),
    Claim(
        "gradient booster, leave-one-machine-out",
        "0.359",
        lambda a: _sm(a, "hist_gradient_boosting", "lomo_mean_rmsle"),
        _r(3),
    ),
    # The log-quadratic control is a Result 4d row; the README table carries
    # only the three contenders and the baseline.
    Claim(
        "log-quadratic control, leave-one-machine-out",
        "0.300",
        lambda a: _sm(a, "ridge_log_quadratic", "lomo_mean_rmsle"),
        _r(3),
        documents=(RESULTS_MD, PAPER, PAPER_PDF),
    ),
    Claim("power law, CV", "0.181", lambda a: _sm(a, "ridge_loglinear", "cv_rmsle"), _r(3)),
    Claim("power law, leave-one-machine-out", "0.214", lambda a: _sm(a, "ridge_loglinear", "lomo_mean_rmsle"), _r(3)),
    Claim("IPB98(y,2), CV", "0.199", lambda a: _sm(a, "ipb98y2_analytic", "cv_rmsle"), _r(3)),
    Claim("IPB98(y,2), leave-one-machine-out", "0.188", lambda a: _sm(a, "ipb98y2_analytic", "lomo_mean_rmsle"), _r(3)),
    # -- Result 4b: error against distance --------------------------------
    Claim(
        "forest error against distance",
        "+0.85",
        lambda a: _sm(a, "random_forest", "distance_spearman"),
        lambda v: f"{v:+.2f}",
        documents=(README, RESULTS_MD, PAPER, PAPER_PDF, SA_README),
    ),
    Claim(
        "power law error against distance",
        "-0.06",
        lambda a: _sm(a, "ridge_loglinear", "distance_spearman"),
        lambda v: f"{v:+.2f}".replace("+", "-") if v < 0 else f"{v:+.2f}",
    ),
    # -- Result 4: paired by machine --------------------------------------
    # README writes "13 of 13 machines" in prose, RESULTS.md "**13 of 13**" in
    # a table cell; the shared substring is what both must carry.
    Claim(
        "machines where the forest is worse",
        "13 of 13",
        lambda a: _paired(a, "random_forest", "ridge_loglinear")["n_machines_a_worse"],
        lambda v: f"{v} of {v}",
        documents=(README, RESULTS_MD, PAPER, PAPER_PDF, ZENODO, SA_README),
    ),
    # -- Sec. 4.1: where the training-target bound is actually active ------
    # The bound binds on JET and nowhere else among the thirteen, and the paper
    # now says so and turns it into evidence. These bind the numbers that
    # argument is made of, so a rerun that moved them could not leave the
    # argument standing on stale figures.
    Claim(
        "rows above the training ceiling at the size cut",
        "34%",
        lambda a: a["boundedness"]["iter_matched_cut"]["target_above_train_max_fraction"],
        _pct(0),
        documents=(RESULTS_MD, PAPER, PAPER_PDF),
    ),
    Claim(
        "MAST headroom below the forest ceiling",
        "3.27",
        lambda a: _headroom(a, "MAST"),
        lambda v: f"{-v:.2f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "JET-ILW headroom below the forest ceiling",
        "1.31",
        lambda a: _headroom(a, "JETILW"),
        lambda v: f"{-v:.2f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "forest margin on NSTX",
        "+0.568",
        lambda a: _gap(a, "NSTX"),
        lambda v: f"{v:+.3f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "forest margin on PBX-M",
        "+0.453",
        lambda a: _gap(a, "PBXM"),
        lambda v: f"{v:+.3f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "forest margin on MAST",
        "+0.427",
        lambda a: _gap(a, "MAST"),
        lambda v: f"{v:+.3f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "forest margin on JET, the one fold the bound binds on",
        "+0.280",
        lambda a: _gap(a, "JET"),
        lambda v: f"{v:+.3f}",
        documents=(PAPER, PAPER_PDF),
    ),
    # -- Sec. 4.2: the newer published laws, and the elongation bound ------
    Claim(
        "ITPA20 at the ITER-size-matched cut",
        "0.177",
        lambda a: _published(a, "ITPA20", "iter_matched_cut"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "ITPA20 over all rows",
        "0.188",
        lambda a: _published(a, "ITPA20", "all_rows"),
        _r(3),
        documents=(PAPER, PAPER_PDF, ZENODO),
    ),
    Claim(
        "ITPA20 per label",
        "0.191",
        lambda a: _published(a, "ITPA20", "machine_equal"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "ITPA20-IL at the ITER-size-matched cut",
        "0.165",
        lambda a: _published(a, "ITPA20-IL", "iter_matched_cut"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    # The measured ratio, against the shape model it replaced. The model gave
    # 1.110 as an upper bound; these are what the database says instead.
    Claim(
        "median measured elongation ratio",
        "1.083",
        lambda a: _elongation(a, "conversion_ratio", "median"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "largest measured elongation ratio",
        "1.341",
        lambda a: _elongation(a, "conversion_ratio", "max"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "rows above the shape model's ceiling",
        "16%",
        lambda a: _elongation(a, "conversion_ratio", "fraction_above_shape_model_ceiling"),
        _pct(0),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "rows below unity, which no such boundary produces",
        "7%",
        lambda a: _elongation(a, "conversion_ratio", "fraction_below_one"),
        _pct(0),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "PBX-M elongation ratio",
        "0.739",
        lambda a: _elongation(a, "conversion_ratio", "per_label_median", "PBXM"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    # -- Sec. 4.2: the stored-energy control -------------------------------
    Claim(
        "rows carrying a stored energy",
        "6214",
        lambda a: a["stored"]["n_rows"],
        documents=(PAPER, PAPER_PDF),
    ),
    # The two cross-validation scores were the only numbers in this arm that
    # nothing bound, and that is how `analysis_stored_energy` came to compute
    # them on the ten-column matrix while the held-out columns beside them used
    # the nine blind ones. A claim on the win counts cannot see a feature-set
    # difference, because the win counts never came from the wrong call.
    Claim(
        "forest cross-validation on stored energy",
        "0.123",
        lambda a: _stored(a, "cv_rmsle", "random_forest"),
        _r(3),
        documents=(PAPER, PAPER_PDF, RESULTS_MD),
        # A bare "0.123" is Table 2's CV score on the 13 scored labels as well,
        # so both spellings carry the words around them.
        phrases=lambda literal: (
            f"cross-validation, {literal} against",
            f"ordinary predictor) | {literal} /",
        ),
    ),
    Claim(
        "power law cross-validation on stored energy",
        "0.193",
        lambda a: _stored(a, "cv_rmsle", "ridge_loglinear"),
        _r(3),
        documents=(PAPER, PAPER_PDF, RESULTS_MD),
        # Not "the power law's 0.193": the source writes an ASCII apostrophe and
        # the PDF a typeset one, so a spelling carrying it matches the tex and
        # fails the PDF. Anchor on the words after the number instead, and avoid
        # "leave-one-label-out", which the PDF hyphenates across a line break.
        phrases=lambda literal: (f"{literal}. Under", f"/ {literal} | 0.468"),
    ),
    Claim(
        "forest against the power law on stored energy, by label",
        "9 of 13",
        lambda a: _stored(a, "forest_worse_by_label", "n_worse"),
        lambda v: f"{int(v)} of 13",
        documents=(PAPER, PAPER_PDF, ZENODO),
    ),
    Claim(
        "stored-energy paired gap by label",
        "+0.220",
        lambda a: _stored(a, "forest_worse_by_label", "mean_difference"),
        lambda v: f"{v:+.3f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "stored-energy paired gap by device",
        "+0.348",
        lambda a: _stored(a, "forest_worse_by_device", "mean_difference"),
        lambda v: f"{v:+.3f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "forest across the size cut on stored energy",
        "1.081",
        lambda a: _stored(a, "iter_matched_cut", "random_forest"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    # -- Sec. 4.2: the deployment-matched estimator ------------------------
    Claim(
        "mixed model, leave-one-label-out",
        "0.264",
        lambda a: a["mixed"]["leave_one_label_out"]["reml"]["mean_rmsle"],
        _r(3),
        documents=(PAPER, PAPER_PDF, ZENODO),
    ),
    Claim(
        "mixed model, leave-one-device-out",
        "0.829",
        lambda a: a["mixed"]["leave_one_device_out"]["reml"]["mean_rmsle"],
        _r(3),
        documents=(PAPER, PAPER_PDF, ZENODO),
    ),
    Claim(
        "mixed model, best variance ratio in hindsight over labels",
        "0.2127",
        lambda a: a["mixed"]["leave_one_label_out"]["variance_ratio_sweep"]["best_mean_rmsle"],
        _r(4),
        documents=(SUPPLEMENTARY, SUPPLEMENTARY_PDF),
    ),
    Claim(
        "mixed model at the ITER-size-matched cut",
        "0.255",
        lambda a: a["mixed"]["iter_matched_cut"]["reml_rmsle"],
        _r(3),
        documents=(SUPPLEMENTARY, SUPPLEMENTARY_PDF),
    ),
    # -- Sec. 4.1: the same distance in dimensionless coordinates ----------
    Claim(
        "forest error against dimensionless distance",
        "+0.77",
        lambda a: _dimensionless(a, "error_correlations", "random_forest", "against_dimensionless"),
        lambda v: f"{v:+.2f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "the two distance rankings agree",
        "+0.78",
        lambda a: _dimensionless(a, "distance_agreement"),
        lambda v: f"{v:+.2f}",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "rho* displacement across the size cut, in training standard deviations",
        "1.21",
        lambda a: _dimensionless(
            a, "iter_matched_cut", "groups", "log_rho_star", "fraction_of_a_training_sd"
        ),
        _r(2),
        documents=(SUPPLEMENTARY, SUPPLEMENTARY_PDF),
    ),
    # -- Sec. 11.1: the third mean function, which stops the ablation being an identity --
    Claim(
        "IPB98 mean function at the ITER-size-matched cut",
        "0.186",
        lambda a: _mean_function(a, "IPB98(y,2) mean + RBF residual", "iter_matched_cut"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "IPB98 mean function on a held-out label",
        "0.189",
        lambda a: _mean_function(a, "IPB98(y,2) mean + RBF residual", "leave_one_machine_out"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    # -- Sec. 10: the projection a design study would consume -------------
    Claim(
        "constrained fit's ITER prediction",
        "2.837",
        lambda a: _forecast(a, "powerlaw_collisionless"),
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "IPB98(y,2)'s ITER prediction",
        "3.591",
        lambda a: _forecast(a, "ipb98y2_analytic"),
        _r(3),
        documents=(PAPER, PAPER_PDF, SUPPLEMENTARY, SUPPLEMENTARY_PDF),
    ),
    # -- Sec. 4.2: do the features name the machine? -----------------------
    Claim(
        "major radius, share of log variance within a device",
        "0.1%",
        lambda a: _dimensionless(a, "device_identity", "within_device_variance", "log_r_m"),
        _pct(1),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "device recovered from the nine features",
        "99.9%",
        lambda a: _dimensionless(a, "device_identity", "device_recovery", "accuracy"),
        _pct(1),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "forest interpolation gain in dimensionless coordinates",
        "28.9%",
        lambda a: _dimensionless(
            a, "device_identity", "arms", "dimensionless_groups", "cv_gain_of_forest"
        ),
        _pct(1),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "labels the forest loses in dimensionless coordinates",
        "11 of 13",
        lambda a: _dimensionless(
            a, "device_identity", "arms", "dimensionless_groups", "by_label", "n_forest_worse"
        ),
        lambda v: f"{int(v)} of 13",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "devices the forest loses in dimensionless coordinates",
        "9 of 11",
        lambda a: _dimensionless(
            a, "device_identity", "arms", "dimensionless_groups", "by_device", "n_forest_worse"
        ),
        lambda v: f"{int(v)} of 11",
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "forest interpolation gain on the within-device features only",
        "50.3%",
        lambda a: _dimensionless(
            a, "device_identity", "arms", "within_device_features_only", "cv_gain_of_forest"
        ),
        _pct(1),
        documents=(PAPER, PAPER_PDF),
    ),
    # -- Sec. 4.2: the sub-unity elongation ratios are indentation ---------
    Claim(
        "PBX-M median indentation",
        "0.154",
        lambda a: a["sensitivity"]["elongation_convention"]["indentation"]["per_label_median"]["PBXM"],
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "elongation ratio where the boundary is indented",
        "0.740",
        lambda a: a["sensitivity"]["elongation_convention"]["indentation"]["median_ratio_where_indented"],
        _r(3),
        documents=(PAPER, PAPER_PDF),
    ),
    Claim(
        "rows below unity beyond rounding",
        "2.0%",
        lambda a: _elongation(a, "conversion_ratio", "fraction_below_one_beyond_rounding"),
        _pct(1),
        documents=(PAPER, PAPER_PDF),
    ),
    # -- Sec. 6: the coverage that is close enough to zero to look like a bug --
    Claim(
        "rows the booster's interval covers at the size cut",
        "8",
        _booster_rows_covered,
        documents=(README, RESULTS_MD, PAPER, PAPER_PDF),
    ),
    Claim(
        "paired gap, forest against power law",
        "+0.251",
        lambda a: _paired(a, "random_forest", "ridge_loglinear")["mean_difference"],
        lambda v: f"{v:+.3f}",
    ),
    Claim(
        "paired gap, lower bound",
        "+0.157",
        lambda a: _paired(a, "random_forest", "ridge_loglinear")["ci_low"],
        lambda v: f"{v:+.3f}",
    ),
    Claim(
        "paired gap, upper bound",
        "+0.342",
        lambda a: _paired(a, "random_forest", "ridge_loglinear")["ci_high"],
        lambda v: f"{v:+.3f}",
    ),
    # -- Nested tuning of the flexible models ------------------------------
    # Quoted only in the paper, which is how results/tuned.json went stale
    # unnoticed: analysis_tuned.py gained a second selection procedure and
    # nothing bound its output to the prose that reports it.
    Claim(
        "tuned forest, CV",
        "0.126",
        lambda a: _tuned(a, "random_forest", "cv"),
        _r(3),
        documents=(RESULTS_MD, SUPPLEMENTARY, SUPPLEMENTARY_PDF),
    ),
    Claim(
        "tuned forest, leave-one-machine-out",
        "0.403",
        lambda a: _tuned(a, "random_forest", "leave_one_machine_out"),
        _r(3),
        documents=(RESULTS_MD, PAPER, PAPER_PDF),
    ),
    Claim(
        "tuned forest, ITER-size-matched cut",
        "1.121",
        lambda a: _tuned(a, "random_forest", "iter_matched_cut"),
        _r(3),
        documents=(RESULTS_MD, SUPPLEMENTARY, SUPPLEMENTARY_PDF),
    ),
    Claim(
        "tuned forest, inner folds by machine",
        "0.376",
        lambda a: _tuned(a, "random_forest", "leave_one_machine_out_inner_machine"),
        _r(3),
        documents=(RESULTS_MD, PAPER, PAPER_PDF),
    ),
    Claim(
        "tuned booster, inner folds by machine",
        "0.350",
        lambda a: _tuned(a, "hist_gradient_boosting", "leave_one_machine_out_inner_machine"),
        _r(3),
        documents=(RESULTS_MD, SUPPLEMENTARY, SUPPLEMENTARY_PDF),
    ),
    # -- Result 5: the ITER-matched size cut -------------------------------
    # The five model scores at the cut are the paper's central comparison and
    # are quoted throughout all three documents, but nothing bound them to the
    # escalation table until now. Same gap that let tuned.json go stale.
    Claim("IPB98(y,2) at the size cut", "0.194", lambda a: _esc(a, "ipb98y2_analytic", "size_cut_rmsle"), _r(3)),
    Claim("power law at the size cut", "0.278", lambda a: _esc(a, "ridge_loglinear", "size_cut_rmsle"), _r(3)),
    Claim("forest at the size cut", "0.938", lambda a: _esc(a, "random_forest", "size_cut_rmsle"), _r(3)),
    Claim("booster at the size cut", "1.072", lambda a: _esc(a, "hist_gradient_boosting", "size_cut_rmsle"), _r(3)),
    Claim("mean baseline at the size cut", "1.459", lambda a: _esc(a, "mean_baseline", "size_cut_rmsle"), _r(3)),
    Claim(
        "size ratio of the matched cut",
        "1.823",
        lambda a: a["size"]["iter_matched_split"]["size_ratio"],
        _r(3),
        documents=(RESULTS_MD, PAGE, PAPER, PAPER_PDF),
    ),
    Claim(
        "size ratio ITER asks for",
        "1.824",
        lambda a: a["size"]["iter_size_ratio"],
        _r(3),
        documents=(RESULTS_MD, PAGE, PAPER, PAPER_PDF),
    ),
    Claim(
        "largest training machine",
        "1.865",
        lambda a: a["size"]["iter_matched_split"]["train_r_max_m"],
        _r(3),
        documents=(RESULTS_MD, PAGE, PAPER, PAPER_PDF),
    ),
    Claim(
        "largest held-out machine",
        "3.40",
        lambda a: a["size"]["iter_matched_split"]["test_r_max_m"],
        _r(2),
        documents=(RESULTS_MD, PAGE, PAPER, PAPER_PDF),
    ),
    # The skill score is a normalised metric this study defines rather than one a
    # reader arrives with, so it was cut from the paper, where a custom metric
    # invites a question the raw errors already answer. It stays in the README,
    # RESULTS.md and the page, which have room to define it, and stays in the
    # artifact either way.
    Claim(
        "power law skill at the cut",
        "93%",
        lambda a: _esc(a, "ridge_loglinear", "skill_against_baseline"),
        _pct(),
        documents=(README, RESULTS_MD, PAGE),
        phrases=lambda n: (f"keeps {n}", f"retains {n}"),
    ),
    Claim(
        "forest skill at the cut",
        "41%",
        lambda a: _esc(a, "random_forest", "skill_against_baseline"),
        _pct(),
        documents=(README, RESULTS_MD, PAGE),
        phrases=lambda n: (
            f"keep 31% and {n}",
            f"keep {n} and 31%",
            f"retain {n} and 31%",
            f"keep 41% and {n}",
            f"keep {n} and 41%",
            f"retain 41% and {n}",
        ),
    ),
    Claim(
        "gradient booster skill at the cut",
        "31%",
        lambda a: _esc(a, "hist_gradient_boosting", "skill_against_baseline"),
        _pct(),
        documents=(README, RESULTS_MD, PAGE),
        phrases=lambda n: (
            f"keep 31% and {n}",
            f"keep {n} and 31%",
            f"retain {n} and 31%",
            f"keep 41% and {n}",
            f"keep {n} and 41%",
            f"retain 41% and {n}",
        ),
    ),
    # -- Result 7: the intervals -------------------------------------------
    Claim(
        "nominal coverage",
        "90%",
        lambda a: a["conformal"]["nominal_coverage"],
        _pct(),
        documents=(README, RESULTS_MD, PAGE, PAPER, PAPER_PDF, ZENODO),
    ),
    Claim(
        "forest coverage on an unseen machine",
        "35%",
        lambda a: _cov(a, "random_forest", "lomo_coverage"),
        _pct(),
        documents=(README, RESULTS_MD, PAPER, PAPER_PDF, ZENODO),
    ),
    Claim(
        "forest coverage across the size cut",
        "3%",
        lambda a: _cov(a, "random_forest", "size_cut_coverage"),
        _pct(),
        documents=(README, RESULTS_MD, PAPER, PAPER_PDF, ZENODO),
    ),
    # -- Result 1: the rank audit ------------------------------------------
    Claim(
        "rank of the standardized feature matrix",
        "rank 8",
        lambda a: a["analysis"]["rank_audit"]["rank"],
        lambda v: f"rank {v}",
    ),
    Claim("rank deficiency", "2", lambda a: a["analysis"]["rank_audit"]["rank_deficiency"]),
    # -- Result 8: physics as a constraint --------------------------------
    Claim(
        "collisionless power law at the ITER-matched cut",
        "0.183",
        lambda a: _dim(a, "powerlaw_collisionless", "size_cut_rmsle"),
        _r(3),
        documents=LATE_RESULTS,
    ),
    Claim(
        "IPB98 distance from the Kadomtsev surface",
        "0.00096",
        lambda a: _constraint(a, "ipb98y2_published", "kadomtsev"),
        _r(5),
        documents=LATE_RESULTS,
    ),
    Claim(
        "IPB98 distance from the collisionless surface",
        "0.0045",
        lambda a: _constraint(a, "ipb98y2_published", "collisionless"),
        _r(4),
        documents=LATE_RESULTS,
    ),
    Claim(
        "unconstrained fit, in sample",
        "0.1808",
        lambda a: float(a["dimensional"]["in_sample_rmsle"]["powerlaw_free"]),
        _r(4),
        documents=LATE_RESULTS,
    ),
    Claim(
        "collisionless fit, in sample",
        "0.1818",
        lambda a: float(a["dimensional"]["in_sample_rmsle"]["powerlaw_collisionless"]),
        _r(4),
        documents=LATE_RESULTS,
    ),
    # -- Result 10: the repaired intervals --------------------------------
    Claim(
        "random forest coverage on a held-out machine, machine-CV",
        "88%",
        lambda a: _shift(a, "random_forest", "machine_cv", "lomo_coverage"),
        _pct(0),
        documents=LATE_RESULTS,
    ),
    Claim(
        "random forest coverage at the ITER cut, distance-scaled",
        "40%",
        lambda a: _shift(a, "random_forest", "machine_cv_distance", "size_cut_coverage"),
        _pct(0),
        documents=LATE_RESULTS,
    ),
    Claim(
        "collisionless coverage at the ITER cut, split conformal",
        "91%",
        lambda a: _shift(a, "powerlaw_collisionless", "split", "size_cut_coverage"),
        _pct(0),
        documents=LATE_RESULTS,
    ),
    # -- Result 11: the replication ---------------------------------------
    Claim("disjoint H-mode rows", "5358", lambda a: _arm(a, "disjoint_h")["n_rows"], documents=LATE_RESULTS),
    Claim("non-H rows", "3860", lambda a: _arm(a, "non_h")["n_rows"], documents=LATE_RESULTS),
    Claim(
        "CV gain over IPB98 on the disjoint arm",
        "42%",
        lambda a: _arm(a, "disjoint_h")["cv_gain_over_baseline"],
        _pct(0),
        documents=MOVED_TO_SUPPLEMENT,
    ),
    Claim(
        "CV gain over ITER89-P on the non-H arm",
        "67%",
        lambda a: _arm(a, "non_h")["cv_gain_over_baseline"],
        _pct(0),
        documents=MOVED_TO_SUPPLEMENT,
    ),
    # -- Result 12: the locked forecast -----------------------------------
    Claim(
        "largest confinement time in the training data",
        "1.321",
        lambda a: float(a["forecast"]["train_tau_max_s"]),
        _r(3),
        documents=LATE_RESULTS,
    ),
    Claim(
        "IPB98 on ITER", "3.591", lambda a: _cast(a, "ITER", "ipb98y2_analytic"), _r(3), documents=MOVED_TO_SUPPLEMENT
    ),
    Claim(
        "random forest on ITER",
        "0.435",
        lambda a: _cast(a, "ITER", "random_forest"),
        _r(3),
        documents=MOVED_TO_SUPPLEMENT,
    ),
    Claim(
        "collisionless power law on ITER",
        "2.837",
        lambda a: _cast(a, "ITER", "powerlaw_collisionless"),
        _r(3),
        documents=MOVED_TO_SUPPLEMENT,
    ),
    # -- the shipped predictor, whose worked example is in the README ------
    Claim("ITER extrapolation distance", "4.72", lambda a: _cast_distance(a, "ITER"), _r(2), documents=(README,)),
    Claim(
        "unconstrained power law on ITER, as served",
        "2.860",
        lambda a: _served(a, "ITER", "powerlaw_free"),
        _r(3),
        documents=(README,),
    ),
    Claim(
        "model disagreement at ITER",
        "8.3",
        lambda a: _spread(a, "ITER"),
        _r(1),
        documents=LATE_RESULTS,
        phrases=lambda lit: (f"a factor of **{lit}**", f"a factor of {lit}"),
    ),
    Claim(
        "model agreement on JT-60SA",
        "15%",
        # Rendered as the excess over perfect agreement, because that is how
        # the prose says it: "agree to within 15%".
        lambda a: _spread(a, "JT-60SA") - 1.0,
        _pct(0),
        documents=LATE_RESULTS,
        phrases=lambda lit: (f"agree to within {lit}",),
    ),
    # -- Result 13: the replication on Kleiber's law -----------------------
    Claim("allometry species records", "541", lambda a: a["allometry"]["n_rows"], documents=(*ALLOMETRY, SA_README)),
    Claim(
        "allometry orders scored",
        "11",
        lambda a: a["allometry"]["n_orders_scored"],
        documents=(*ALLOMETRY, SA_README),
        # A bare "11" appears all over these documents; anchor it.
        phrases=lambda lit: (f"{lit} orders",),
    ),
    Claim(
        "allometry order mass span",
        "342x",
        lambda a: a["allometry"]["order_mass_ratio"],
        lambda v: f"{v:.0f}x",
        documents=ALLOMETRY,
    ),
    Claim(
        "free refit of Kleiber's exponent",
        "0.687",
        lambda a: a["allometry"]["free_refit_exponent"],
        _r(3),
        documents=ALLOMETRY,
    ),
    Claim(
        "Kleiber at the widest mass cut",
        "0.374",
        lambda a: _allo(a, "kleiber", "mass_cut_rmsle"),
        _r(3),
        documents=ALLOMETRY,
    ),
    Claim(
        "free power law at the widest mass cut",
        "0.496",
        lambda a: _allo(a, "ols_loglinear", "mass_cut_rmsle"),
        _r(3),
        documents=ALLOMETRY,
    ),
    Claim(
        "orders where the forest loses to Kleiber",
        "9 of 11",
        lambda a: a["allometry"]["n_orders_forest_loses_to_kleiber"],
        lambda v: f"{v} of 11",
        documents=ALLOMETRY,
    ),
    Claim(
        "mass cuts where the power laws beat the trees",
        "all 8",
        lambda a: a["allometry"]["sweep_wins"]["power_laws_beat_trees"],
        lambda v: f"all {v}",
        documents=(*ALLOMETRY, SA_README),
        phrases=lambda lit: (f"{lit} mass cuts",),
    ),
    Claim(
        "mass cuts where Kleiber beats the free fit",
        "4 of 8",
        lambda a: a["allometry"]["sweep_wins"]["kleiber_beats_free_power_law"],
        lambda v: f"{v} of 8",
        documents=ALLOMETRY,
    ),
    # -- Result 14: flexibility against boundedness ------------------------
    # Carried by the README and RESULTS.md. The paper has no Result 14 section
    # yet, so it is deliberately not in this documents tuple.
    Claim("GP linear+RBF, CV", "0.112", lambda a: _gp(a, "gp_linear_rbf", "cv"), _r(3), documents=GP),
    Claim(
        "GP linear+RBF, held-out machine",
        "0.218",
        lambda a: _gp(a, "gp_linear_rbf", "leave_one_tokamak_out"),
        _r(3),
        documents=GP,
    ),
    Claim(
        "GP linear+RBF at the ITER cut",
        "0.191",
        lambda a: _gp(a, "gp_linear_rbf", "iter_matched_cut"),
        _r(3),
        documents=GP,
    ),
    Claim("GP RBF-only at the ITER cut", "1.948", lambda a: _gp(a, "gp_rbf", "iter_matched_cut"), _r(3), documents=GP),
    Claim("GP RBF-only, CV", "0.142", lambda a: _gp(a, "gp_rbf", "cv"), _r(3), documents=GP),
    Claim(
        "GP RBF-only error against distance",
        "+0.65",
        lambda a: _gp(a, "gp_rbf", "distance_correlation"),
        lambda v: f"{v:+.2f}",
        documents=GP,
    ),
    Claim(
        "GP linear+RBF error against distance",
        "-0.01",
        lambda a: _gp(a, "gp_linear_rbf", "distance_correlation"),
        lambda v: f"{v:+.2f}".replace("+", "-") if v < 0 else f"{v:+.2f}",
        documents=GP,
    ),
    Claim(
        "GP linear+RBF coverage at the ITER cut",
        "92.5%",
        lambda a: _gp(a, "gp_linear_rbf", "coverage_at_iter_cut"),
        lambda v: f"{v:.1%}",
        documents=GP,
    ),
    Claim(
        "GP RBF-only coverage at the ITER cut",
        "16.7%",
        lambda a: _gp(a, "gp_rbf", "coverage_at_iter_cut"),
        lambda v: f"{v:.1%}",
        documents=GP,
    ),
    # -- Result 15: the reversal's precondition ----------------------------
    Claim("tree allometry plants", "3599", lambda a: a["tree"]["n_rows"], documents=(*TREE, SA_README)),
    Claim(
        "tree allometry species",
        "53",
        lambda a: a["tree"]["n_species_total"],
        documents=TREE,
        phrases=lambda lit: (f"{lit} species",),
    ),
    Claim("tree allometry species scored", "33", lambda a: a["tree"]["n_species_scored"], documents=TREE),
    Claim(
        "tree allometry diameter span",
        "36x",
        lambda a: a["tree"]["species_size_span"],
        lambda v: f"{v:.0f}x",
        documents=TREE,
    ),
    Claim(
        "free refit of the WBE exponent",
        "2.512",
        lambda a: a["tree"]["free_refit_exponent"],
        _r(3),
        documents=(RESULTS_MD,),
    ),
    Claim(
        "dimension at which the reversal appears",
        "3 predictors",
        lambda a: a["tree"]["first_reversal_n_features"],
        lambda v: f"{v} predictors",
        documents=(*TREE, SA_README),
    ),
    Claim(
        "rungs where the power law wins the held-out species",
        "4 of 4",
        lambda a: a["tree"]["n_rungs_power_law_wins_extrapolation"],
        lambda v: f"{v} of 4",
        documents=(*TREE, SA_README),
    ),
    Claim(
        "ladder rung 1 interpolation gain",
        "-1.7%",
        lambda a: _rung(a, 1, "cv_gain_over_powerlaw"),
        lambda v: f"{v:+.1%}".replace("+", "-") if v < 0 else f"{v:+.1%}",
        documents=(*TREE, SA_README),
    ),
    Claim(
        "ladder rung 2 interpolation gain",
        "-0.6%",
        lambda a: _rung(a, 2, "cv_gain_over_powerlaw"),
        lambda v: f"{v:+.1%}".replace("+", "-") if v < 0 else f"{v:+.1%}",
        documents=(*TREE, SA_README),
    ),
    Claim(
        "ladder rung 3 interpolation gain",
        "+1.8%",
        lambda a: _rung(a, 3, "cv_gain_over_powerlaw"),
        lambda v: f"{v:+.1%}",
        documents=(*TREE, SA_README),
    ),
    Claim(
        "ladder rung 4 interpolation gain",
        "+6.8%",
        lambda a: _rung(a, 4, "cv_gain_over_powerlaw"),
        lambda v: f"{v:+.1%}",
        documents=(*TREE, SA_README),
    ),
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


# --- the same numbers, in the source files ----------------------------------
#
# Every claim above is bound to a document a reader might open. Docstrings were
# not, and six of them drifted: they described the headline margin over
# IPB98(y,2) as 41%, which is the ten-column variant of Result 2 that
# results/RESULTS.md explicitly keeps as auxiliary. The headline is 36%.
#
# The cover letter invites an editor into this repository, so a module header
# that contradicts the abstract is worth the same as a wrong number in the
# paper. This checks the source the way the claims above check the prose.

MARGIN_CLAIM = re.compile(r"(?:beats?|beating|wins?) (?:the published [a-z ]*law )?by (\d+)% under", re.IGNORECASE)


def _source_files() -> list[Path]:
    """Tracked Python at the repository root, which is where the prose headers live."""
    return sorted(p for p in ROOT.glob("*.py") if p.name != "conftest.py")


def test_no_source_file_misstates_the_headline_margin(artifacts: dict) -> None:
    published = _sm(artifacts, "ipb98y2_analytic", "cv_rmsle")
    forest = _sm(artifacts, "random_forest", "cv_rmsle")
    expected = round(100 * (published - forest) / published)

    wrong = []
    for path in _source_files():
        for claimed in MARGIN_CLAIM.findall(re.sub(r"\s+", " ", path.read_text())):
            if int(claimed) != expected:
                wrong.append(f"{path.name} says {claimed}%")
    assert not wrong, (
        f"the forest beats IPB98(y,2) by {expected}% under grouped cross-validation, "
        "but " + "; ".join(wrong) + ". 41% is the ten-column variant of Result 2, "
        "which RESULTS.md keeps as auxiliary rather than the headline."
    )


def test_the_margin_check_is_reading_something(artifacts: dict) -> None:
    """A regex that matched nothing would pass the check above silently."""
    found = sum(len(MARGIN_CLAIM.findall(re.sub(r"\s+", " ", p.read_text()))) for p in _source_files())
    assert found >= 4, f"only {found} margin claims found in the source headers"


# --- the paper's own count of these claims -----------------------------------
#
# The paper prints how many of its numbers are bound here. That sentence is
# itself a claim about this file, and it is the one claim nothing was checking:
# it was typed by hand and stayed right only for as long as nobody added a
# claim. Spelled out in the prose, so the numerals are written here.

SPELLED = {
    97: "Ninety-seven",
    127: "one hundred and twenty-seven",
}


def test_the_paper_states_how_many_of_its_numbers_are_bound(documents: dict) -> None:
    in_paper = sum(1 for claim in CLAIMS if PAPER in claim.documents)
    in_both = sum(
        1 for claim in CLAIMS if PAPER in claim.documents or SUPPLEMENTARY in claim.documents
    )
    text = documents[PAPER]
    for count in (in_paper, in_both):
        spelled = SPELLED.get(count)
        assert spelled is not None, (
            f"{count} claims are bound to the paper and no spelling for it is recorded "
            "here. Add one and update the sentence in the Data availability section."
        )
        assert spelled in text, (
            f"the paper should say {spelled!r} ({count} claims bound to it), and does not. "
            "Update the sentence in the Data availability section."
        )
