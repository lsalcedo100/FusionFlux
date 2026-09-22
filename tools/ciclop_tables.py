"""Every CICLOP number the manuscript may print, generated from the artifact.

Run ``python3 tools/ciclop_tables.py`` once ``results/ciclop.json`` exists. It
writes the supplement's three tables and a list of facts under ``build/ciclop/``
and prints them, so the numbers in the manuscript are pasted from generated
text and never typed.

The rest of the paper binds its numbers the other way round: a number is typed
into the prose, typed again into ``tests/test_reported_numbers.py``, and the test
checks both against the artifact. That cannot work here, because this was
written before the CICLOP numbers existed, and it leaves two hand-kept copies of
every value. So the direction is reversed:

    facts     Every quantity the artifact carries that prose could quote, each
              formatted once, exactly as it is to be printed.

    blocks    The manuscript marks its CICLOP passages with ``% ciclop:begin``
              and ``% ciclop:end`` comment lines.

    the scan  ``unbound_numerals`` reads the marked passages and returns every
              numeral that is not one of the facts. The test suite requires
              that list to be empty, and requires every mention of CICLOP
              outside the bibliography to sit inside a marked passage.

So a CICLOP number that the code did not generate cannot be printed at all, and
there is one copy of each. Until ``results/ciclop.json`` exists there are no
facts, and the same tests require that the manuscript does not mention CICLOP.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # so `python3 tools/ciclop_tables.py` finds the root modules
    sys.path.insert(0, str(ROOT))

import analysis_robustness as ar  # noqa: E402
import ciclop  # noqa: E402
import hdb5  # noqa: E402

RESULTS = ROOT / "results" / "ciclop.json"
OUTPUT_DIR = ROOT / "build" / "ciclop"
DOCUMENTS = (ROOT / "paper" / "paper.tex", ROOT / "paper" / "supplementary.tex")

BEGIN, END = "% ciclop:begin", "% ciclop:end"  # kept for callers; see markers()

MODEL_NAMES = {
    "ridge_loglinear": "ridge, log-linear",
    "random_forest": "random forest",
    "hist_gradient_boosting": "hist gradient boosting",
    "mean_baseline": "mean baseline",
    "ridge_log_quadratic": "ridge, log-quadratic",
    "ridge_log_cubic": "ridge, log-cubic",
    "gp_linear_rbf": "GP, linear + RBF",
}
QUANTITY_NAMES = {
    "tau_th_s": ("confinement time", r"\texttt{TAUTH}"),
    "ip_ma": ("plasma current", r"\texttt{IP}"),
    "bt_t": ("toroidal field", r"\texttt{BT}"),
    "ne_line_1e19_m3": ("density", r"\texttt{NEL}"),
    "p_loss_mw": ("power", r"\texttt{PLTH}"),
    "r_m": ("major radius", r"\texttt{RGEO}"),
    "kappa": ("elongation", r"\texttt{KAPPAA}"),
    "inverse_aspect_ratio": ("inverse aspect ratio", r"\texttt{EPS}"),
    "m_eff_amu": ("effective mass", r"\texttt{MEFF}"),
    "a_m": ("minor radius", r"$\epsilon R$"),
}


# --- formatting: each rule is applied in exactly one place -------------------


def error(value: float) -> str:
    """A log-RMSE, to the three decimals the paper prints everywhere."""
    return f"{value:.3f}"


def ratio(value: float) -> str:
    return f"{value:.2f}"


def signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def p_value(value: float) -> str:
    return f"{value:.2g}"


def percent(value: float) -> str:
    return f"{100 * value:.0f}"


# --- facts --------------------------------------------------------------------


def _arm_facts(prefix: str, arm: dict[str, Any]) -> dict[str, str]:
    out = {f"{prefix}.n_rows": str(arm["n_rows"]), f"{prefix}.n_devices": str(len(arm["eligible_devices"]))}
    for device in arm["eligible_devices"]:
        out[f"{prefix}.rows.{device}"] = str(arm["rows_per_device"][device])
    if not arm.get("scored"):
        return out
    for name, scores in arm["models"].items():
        out[f"{prefix}.{name}.cv_pooled_all_rows"] = error(scores["cv_pooled_all_rows"])
        out[f"{prefix}.{name}.cv_matched"] = error(scores["cv_matched"])
        out[f"{prefix}.{name}.lodo"] = error(scores["lodo"])
        out[f"{prefix}.{name}.transfer_ratio"] = ratio(scores["transfer_ratio"])
        out[f"{prefix}.{name}.cv_matched_pooled"] = error(scores["cv_matched_pooled"])
        out[f"{prefix}.{name}.lodo_pooled"] = error(scores["lodo_pooled"])
        for device in arm["eligible_devices"]:
            out[f"{prefix}.{name}.cv.{device}"] = error(scores["cv_per_device"][device])
            out[f"{prefix}.{name}.lodo.{device}"] = error(scores["lodo_per_device"][device])
    for key, pair in arm.get("pairs", {}).items():
        interval = pair["paired_interval"]
        flexible, _, ridge = key.partition("_vs_")
        cv_f, cv_r = arm["models"][flexible]["cv_matched"], arm["models"][ridge]["cv_matched"]
        lodo_f, lodo_r = arm["models"][flexible]["lodo"], arm["models"][ridge]["lodo"]
        out.update({
            f"{prefix}.{key}.n_devices": str(pair["n_devices"]),
            f"{prefix}.{key}.n_worse": str(pair["n_devices_flexible_worse"]),
            f"{prefix}.{key}.gap": signed(pair["mean_gap"], 3),
            f"{prefix}.{key}.gap_low": signed(interval["ci_low"], 3),
            f"{prefix}.{key}.gap_high": signed(interval["ci_high"], 3),
            f"{prefix}.{key}.sign_p": p_value(pair["exact_two_sided_p"]),
            f"{prefix}.{key}.degradation": ratio(pair["differential_degradation"]),
            f"{prefix}.{key}.degradation_pooled": ratio(pair["differential_degradation_pooled"]),
            # How far apart the two models are under each split, as a percentage
            # of the power law's error, which is how the paper states its margins.
            f"{prefix}.{key}.cv_margin_pct": percent(abs(cv_r - cv_f) / cv_r),
            f"{prefix}.{key}.lodo_margin_pct": percent(abs(lodo_f - lodo_r) / lodo_r),
        })
    distance = arm.get("distance", {})
    for name, rho in distance.get("spearman", {}).items():
        out[f"{prefix}.{name}.distance_rho"] = signed(rho, 2)
    for device, value in distance.get("mahalanobis", {}).items():
        out[f"{prefix}.mahalanobis.{device}"] = ratio(value)
    return out


def _default(function: Any, parameter: str) -> Any:
    """A keyword default read off the function that owns it, so it cannot be restated wrongly."""
    return inspect.signature(function).parameters[parameter].default


def facts(analysis: dict[str, Any]) -> dict[str, str]:
    """Every number the manuscript may print about CICLOP, keyed, as it is to be printed."""
    plan, provenance = analysis["plan"], analysis["provenance"]
    out: dict[str, str] = {
        "verdict": analysis["verdict"],
        "verdict_booster": analysis["verdict_booster"],
        "features.n": str(len(plan["frozen_features"])),
        "features.n_omitted": str(len(plan["omitted_features"])),
        "pulses.n_complete": str(plan["n_complete_rows"]),
        "devices.n_with_rows": str(len(plan["complete_rows_per_device"])),
        "devices.n_also_in_hdb5": str(sum(plan["device_in_hdb5"].get(d, False) for d in plan["eligible_devices"])),
        "devices.n_absent_from_hdb5": str(sum(not plan["device_in_hdb5"].get(d, False) for d in plan["eligible_devices"])),
        "file.sha256": str(provenance["file"]["sha256"]),
        "file.n_bytes": str(provenance["file"]["n_bytes"]),
        "plan.sha256": str(provenance["plan_sha256"]),
        "lock.commit": str(provenance["lock"]["commit"]),
        # The lock's own constants, which the protocol section has to be able to state.
        "rule.usable_pct": percent(ciclop.USABLE_FRACTION),
        "rule.min_rows": str(ciclop.MIN_HELD_OUT_ROWS),
        "rule.sensitivity_min_rows": str(ciclop.SENSITIVITY_MIN_ROWS),
        "rule.min_devices": str(ciclop.MIN_EVALUABLE_DEVICES),
        "rule.min_evaluable_rows": str(ciclop.MIN_EVALUABLE_ROWS),
        "rule.min_features": str(ciclop.MIN_EVALUABLE_FEATURES),
        "rule.degradation_boundary": f"{ciclop.DEGRADATION_BOUNDARY:g}",
        "rule.cv_folds": str(hdb5.N_CV_FOLDS),
        "rule.random_state": str(hdb5.RANDOM_STATE),
        "rule.bootstrap_resamples": str(_default(ar.paired_interval, "n_resamples")),
        "rule.bootstrap_seed": str(_default(ar.paired_interval, "seed")),
        # The percentile interval `ar.paired_interval` reports, and its two ends.
        "rule.interval_pct": "95",
        "rule.interval_low_pct": "2.5",
        "rule.interval_high_pct": "97.5",
    }
    for field in ("version", "access_date"):
        if provenance["file"].get(field):
            out[f"file.{field}"] = str(provenance["file"][field])
    for device, rows in plan["complete_rows_per_device"].items():
        out[f"rows.{device}"] = str(rows)

    out.update(_arm_facts("ciclop", analysis["primary"]))
    comparator = analysis.get("hdb5_feature_matched", {})
    if "n_rows" in comparator:
        out["hdb5.n_rows"] = str(comparator["n_rows"])
    for name, arm in comparator.get("arms", {}).items():
        out.update(_arm_facts(f"hdb5.{name}", arm))

    secondary = analysis.get("secondary", {})
    for name in ("min_rows_30", "h_mode_only", "tore_supra_and_west_split", "cv_grouped_by_device_and_year"):
        if secondary.get(name, {}).get("eligible_devices") is not None:
            out.update(_arm_facts(f"secondary.{name}", secondary[name]))
    overlap = secondary.get("db523_overlap", {})
    if overlap.get("computed"):
        out["secondary.db523_overlap.n_pulses"] = str(overlap["n_pulses_also_in_db523"])
        out.update(_arm_facts("secondary.without_db523_pulses", overlap["without_them"]))
    absent = secondary.get("devices_absent_from_hdb5", {})
    if absent.get("computed"):
        out["secondary.absent.n_devices"] = str(len(absent["devices"]))
        out["secondary.absent.n_forest_worse"] = str(absent["n_devices_forest_worse"])
        for name, scores in absent["models"].items():
            out[f"secondary.absent.{name}.cv_matched"] = error(scores["cv_matched"])
            out[f"secondary.absent.{name}.lodo"] = error(scores["lodo"])
    reference = secondary.get("ipb98y2_on_h_mode_rows", {})
    if reference.get("computed"):
        out["secondary.ipb98.n_rows"] = str(reference["n_rows"])
        out["secondary.ipb98.pooled"] = error(reference["pooled"])
        out["secondary.ipb98.unit_equal"] = error(reference["unit_equal"])
    return out


# --- the three tables of the supplement ---------------------------------------


def _escape(text: str) -> str:
    return re.sub(r"([&%#_])", r"\\\1", str(text))


def result_table(analysis: dict[str, Any]) -> str:
    """S17.5: each model under both splits, CICLOP beside HDB5 on the same features."""
    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Dataset & Model & pooled CV & matched CV & device held out & ratio \\", r"\midrule",
    ]
    arms = [("CICLOP", analysis["primary"])]
    comparator = analysis.get("hdb5_feature_matched", {}).get("arms", {}).get(f"min_rows_{ciclop.MIN_HELD_OUT_ROWS}")
    if comparator:
        arms.append(("HDB5, same features", comparator))
    for label, arm in arms:
        for index, (name, scores) in enumerate(arm.get("models", {}).items()):
            lines.append(
                f"{label if index == 0 else ''} & {MODEL_NAMES.get(name, _escape(name))} & "
                f"{error(scores['cv_pooled_all_rows'])} & {error(scores['cv_matched'])} & "
                f"{error(scores['lodo'])} & {ratio(scores['transfer_ratio'])}$\\times$ \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join([*lines, r"\end{tabular}"])


def population_table(analysis: dict[str, Any]) -> str:
    """S17.3: one row per physical tokamak."""
    plan, eligible = analysis["plan"], set(analysis["plan"]["eligible_devices"])
    lines = [r"\begin{tabular}{lrcc}", r"\toprule", r"Device & complete pulses & in HDB5 & scored \\", r"\midrule"]
    for device, rows in sorted(plan["complete_rows_per_device"].items(), key=lambda item: -item[1]):
        lines.append(
            f"{_escape(device)} & {rows} & {'yes' if plan['device_in_hdb5'].get(device) else 'no'} & "
            f"{'yes' if device in eligible else 'no'} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def mapping_table(plan_file: dict[str, Any]) -> str:
    """S17.2: what stands for each of the study's quantities, from the frozen plan."""
    mapping, frozen = plan_file["mapping"], set(plan_file["frozen_features"])
    lines = [r"\begin{tabular}{llll}", r"\toprule", r"Quantity & HDB5 & CICLOP column & status \\", r"\midrule"]
    for quantity, (name, hdb5_column) in QUANTITY_NAMES.items():
        if quantity == "tau_th_s":
            entry, status = mapping["target"], "target"
        elif quantity == "inverse_aspect_ratio":
            entry, status = {"source": "a/R"}, "used" if quantity in frozen else "omitted"
        else:
            entry, status = mapping["quantities"][quantity], "used" if quantity in frozen else "omitted"
        source = entry.get("source")
        shown = "absent" if source is None else rf"\texttt{{{_escape(source)}}}"
        order = ciclop.PREFERENCE_ORDERS.get("target" if quantity == "tau_th_s" else quantity)
        if order and source is not None:
            status += f": {order[int(entry['rank']) - 1]}"
        lines.append(f"{name} & {hdb5_column} & {shown} & {status} \\\\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


# --- the scan -----------------------------------------------------------------

# A numeral standing on its own. Digits welded to a letter, or hyphenated onto
# one, are names and not numbers: HDB5, IPB98, S17, JT-60U, W7-X, DB5.2.3, SHA-256.
NUMERAL = re.compile(r"(?<![A-Za-z0-9_.\\])(?<![A-Za-z]-)[+-]?\d+(?:\.\d+)?(?![A-Za-z0-9_]|\.\d)")

# LaTeX that carries digits and says nothing about the data.
_LAYOUT = (
    r"\\(?:label|ref|eqref|cite|url|includegraphics|vspace|hspace|cmidrule|input|pageref)(?:\[[^\]]*\])?(?:\([^)]*\))?\{[^}]*\}",
    r"\\href\{[^}]*\}",
    r"\\begin\{(?:tabular|table|figure)\}(?:\[[^\]]*\])?(?:\{[^}]*\})?",
    r"\\multicolumn\{\d+\}\{[^}]*\}",
    r"\\(?:sub)*section\*?",
    # The "2" of IPB98(y,2) and H98(y,2) is part of a name.
    r"\(y,\s*2\)",
    # Units: a power of ten and an integer exponent, as in 10^{19} m^{-3}, and
    # anything siunitx typesets. A decimal exponent is left alone on purpose. It
    # would be a fitted value, and a fitted value has to be a generated fact.
    r"\\si\{[^}]*\}",
    r"(?<![\d.])10\^\{?[+-]?\d+\}?(?![\d.])",
    r"\^\{?[+-]?\d+\}?(?![\d.])",
)


def markers(tag: str = "ciclop") -> tuple[str, str]:
    """The two comment lines that fence a dataset's passages: ``% <tag>:begin`` and ``% <tag>:end``."""
    return f"% {tag}:begin", f"% {tag}:end"


def marked_blocks(latex: str, tag: str = "ciclop") -> list[str]:
    """The passages between the begin and end lines of one dataset's marker."""
    begin, end = markers(tag)
    blocks: list[str] = []
    current: list[str] | None = None
    for line in latex.splitlines():
        stripped = line.strip()
        if stripped == begin:
            if current is not None:
                raise ValueError(f"a `{begin}` follows another with no `{end}` between them")
            current = []
        elif stripped == end:
            if current is None:
                raise ValueError(f"a `{end}` has no `{begin}` before it")
            blocks.append("\n".join(current))
            current = None
        elif current is not None:
            current.append(line)
    if current is not None:
        raise ValueError(f"a `{begin}` is never closed")
    return blocks


def _readable(latex: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", latex)
    for pattern in _LAYOUT:
        text = re.sub(pattern, " ", text)
    return text


def unbound_numerals(latex: str, known: dict[str, str], tag: str = "ciclop") -> list[str]:
    """Numerals in the marked passages that are not among the generated facts."""
    allowed = {value.lstrip("+") for value in known.values()}
    # Long values first, so a hash or a date is removed whole before its digits are read.
    verbatim = sorted((v for v in known.values() if not NUMERAL.fullmatch(v)), key=len, reverse=True)
    loose = []
    for block in marked_blocks(latex, tag):
        text = _readable(block)
        for value in verbatim:
            text = text.replace(value, " ")
        loose += [token for token in NUMERAL.findall(text) if token.lstrip("+") not in allowed]
    return loose


def mentions_outside_blocks(latex: str, names: tuple[str, ...] = ("CICLOP",), tag: str = "ciclop") -> list[str]:
    """Lines that name the dataset outside a marked passage. The bibliography is exempt."""
    begin, end = markers(tag)
    body = latex.split(r"\begin{thebibliography}")[0]
    inside = False
    found = []
    for number, line in enumerate(body.splitlines(), start=1):
        stripped = line.strip()
        if stripped in (begin, end):
            inside = stripped == begin
        elif not inside and any(name in re.sub(r"(?<!\\)%.*", "", line) for name in names):
            found.append(f"line {number}: {stripped[:90]}")
    return found


# --- CLI ----------------------------------------------------------------------


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(
            f"{RESULTS.relative_to(ROOT)} does not exist. It is written by analysis_ciclop.py, which runs "
            f"only once the plan is frozen and pinned; see {ciclop.LOCK_DOCUMENT}."
        )
    analysis = json.loads(RESULTS.read_text(encoding="utf-8"))
    plan_file = json.loads(ciclop.default_plan_path().read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "result_table.tex": result_table(analysis),
        "population_table.tex": population_table(analysis),
        "mapping_table.tex": mapping_table(plan_file),
        "facts.json": json.dumps(facts(analysis), indent=2, sort_keys=True),
    }
    for name, text in outputs.items():
        (OUTPUT_DIR / name).write_text(text + "\n", encoding="utf-8")
        print(f"\n%%%% {name}\n{text}")
    print(f"\nwrote {len(outputs)} files to {OUTPUT_DIR.relative_to(ROOT)}/. Paste from them; type nothing.")


if __name__ == "__main__":
    main()
