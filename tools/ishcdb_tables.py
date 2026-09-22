"""Every ISHCDB number the manuscript may print, generated from the artifact.

Run ``python3 tools/ishcdb_tables.py`` once ``results/ishcdb.json`` exists. It
writes the supplement's tables and a list of facts under ``build/ishcdb/`` and
prints them, so the numbers in the manuscript are pasted from generated text
and never typed.

The discipline is ``tools/ciclop_tables.py``'s, and the scanner is that
module's, called with this dataset's marker. The manuscript fences its ISHCDB
passages with ``% ishcdb:begin`` and ``% ishcdb:end``, and the test suite
requires every numeral inside them to be one of the facts below, and every
mention of the database outside the bibliography to sit inside a fence.

Two kinds of number appear in those passages that the artifact does not carry.
The exponents of ISS04 are written out for the reader, so they are facts here,
and ``tests/test_ishcdb_tables.py`` checks the dictionary below against the
function that evaluates the law. And the passages compare the result with the
manuscript's own, so the handful of HDB5 numbers they quote are read from the
HDB5 artifacts, formatted as the paper formats them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # so `python3 tools/ishcdb_tables.py` finds the root modules
    sys.path.insert(0, str(ROOT))

import ciclop  # noqa: E402
import hdb5  # noqa: E402
import ishcdb  # noqa: E402
from tools import ciclop_tables as ct  # noqa: E402

TAG = "ishcdb"
NAMES: tuple[str, ...] = ("ISHCDB", "ISS04", "tellarator")
RESULTS = ROOT / "results" / "ishcdb.json"
EXTRAPOLATION = ROOT / "results" / "extrapolation.json"
ROBUSTNESS = ROOT / "results" / "robustness.json"
OUTPUT_DIR = ROOT / "build" / "ishcdb"

# ISS04 as published (Yamada et al. 2005): the constant and one exponent per
# regressor, in the order the law is written. Held to ``ishcdb.iss04_tau_s``
# by test, so it is a copy that cannot drift.
ISS04_CONSTANT = 0.134
ISS04_EXPONENTS: dict[str, float] = {
    "a_m": 2.28,
    "r_m": 0.64,
    "p_abs_mw": -0.61,
    "ne_line_1e19_m3": 0.54,
    "bt_t": 0.84,
    "iota_23": 0.41,
}

QUANTITY_NAMES = {
    "tau_th_s": "confinement time",
    "a_m": "effective minor radius",
    "r_m": "major radius",
    "p_abs_mw": "absorbed power",
    "ne_line_1e19_m3": "line-averaged density",
    "bt_t": "toroidal field",
    "iota_23": r"rotational transform $\iotatwothirds$",
}
CONVERSIONS = {
    "a_m": "m, as delivered",
    "r_m": "m, as delivered",
    "p_abs_mw": "W to MW",
    "ne_line_1e19_m3": r"$\mathrm{m}^{-3}$ to $10^{19}\,\mathrm{m}^{-3}$",
    "bt_t": "T, absolute value",
    "iota_23": "absolute value",
}


def hdb5_facts() -> dict[str, str]:
    """The manuscript's own numbers that the ISHCDB passages set beside their own."""
    out: dict[str, str] = {}
    if EXTRAPOLATION.exists():
        for transfer in json.loads(EXTRAPOLATION.read_text(encoding="utf-8"))["transfers"]:
            name = transfer["model_name"]
            out[f"hdb5.{name}.distance_rho"] = ct.signed(transfer["distance_spearman"], 2)
            out[f"hdb5.{name}.cv"] = ct.error(transfer["cv_rmsle"])
            out[f"hdb5.{name}.lomo"] = ct.error(transfer["lomo_mean_rmsle"])
            out[f"hdb5.{name}.degradation_factor"] = f"{transfer['degradation_factor']:.1f}"
    if ROBUSTNESS.exists():
        tests = json.loads(ROBUSTNESS.read_text(encoding="utf-8"))["sign_tests"]
        out["hdb5.labels.n_worse"] = str(tests["lomo_by_database_label"]["n_units_a_worse"])
        out["hdb5.labels.n"] = str(tests["lomo_by_database_label"]["n_units"])
        out["hdb5.devices.n_worse"] = str(tests["lomo_by_physical_device"]["n_units_a_worse"])
        out["hdb5.devices.n"] = str(tests["lomo_by_physical_device"]["n_units"])
    return out


def facts(analysis: dict[str, Any], plan_file: dict[str, Any] | None = None) -> dict[str, str]:
    """Every number the manuscript may print about ISHCDB, keyed, as it is to be printed.

    ``plan_file`` is the frozen plan on disk, which carries the usability shares
    the artifact does not repeat.
    """
    plan, provenance = analysis["plan"], analysis["provenance"]
    population = plan["population"]
    out: dict[str, str] = {
        "verdict": analysis["verdict"],
        "verdict_booster": analysis["verdict_booster"],
        "features.n": str(len(plan["frozen_features"])),
        "features.n_of": str(len(ishcdb.FEATURES)),
        "features.n_omitted": str(len(plan["omitted_features"])),
        "rows.delivered": str(population["n_rows_delivered"]),
        "rows.predictive": str(population["n_predictive_rows"]),
        "rows.measured": str(sum(population["rows_per_device_measured"].values())),
        "rows.standard_set_complete": str(plan["n_complete_rows"]),
        "devices.n_measured": str(len(population["rows_per_device_measured"])),
        "devices.n_standard_set": str(len(plan["complete_rows_per_device"])),
        "discharges.n": str(plan["discharges"]["n"]),
        "discharges.n_multi_row": str(plan["discharges"]["n_with_more_than_one_row"]),
        "discharges.largest": str(plan["discharges"]["largest"]),
        "file.sha256": str(provenance["file"]["sha256"]),
        "file.sha256_short": str(provenance["file"]["sha256"])[:12],
        "file.n_bytes": str(provenance["file"]["n_bytes"]),
        "plan.sha256": str(provenance["plan_sha256"]),
        "lock.commit": str(provenance["lock"]["commit"]),
        "lock.commit_short": str(provenance["lock"]["commit"])[:7],
        "rule.usable_pct": ct.percent(ishcdb.USABLE_FRACTION),
        "rule.min_rows": str(ishcdb.MIN_HELD_OUT_ROWS),
        "rule.sensitivity_min_rows": str(ishcdb.SENSITIVITY_MIN_ROWS),
        "rule.min_devices": str(ishcdb.MIN_EVALUABLE_DEVICES),
        "rule.min_evaluable_rows": str(ishcdb.MIN_EVALUABLE_ROWS),
        "rule.min_features": str(ishcdb.MIN_EVALUABLE_FEATURES),
        "rule.degradation_boundary": f"{ciclop.DEGRADATION_BOUNDARY:g}",
        "rule.cv_folds": str(hdb5.N_CV_FOLDS),
        "rule.bootstrap_resamples": str(ct._default(ct.ar.paired_interval, "n_resamples")),
        "rule.bootstrap_seed": str(ct._default(ct.ar.paired_interval, "seed")),
        "iss04.constant": f"{ISS04_CONSTANT:g}",
        "database.version": "26",
    }
    for name, exponent in ISS04_EXPONENTS.items():
        out[f"iss04.exponent.{name}"] = f"{exponent:g}"
    for device, rows in population["rows_per_device_delivered"].items():
        out[f"rows.delivered.{device}"] = str(rows)
    for device, rows in population["rows_per_device_measured"].items():
        out[f"rows.measured.{device}"] = str(rows)
    # Every measured device gets a standard-set and a complete count, zero
    # included: HSX has no standard-set row, and the population table says so.
    for device in population["rows_per_device_measured"]:
        out[f"rows.standard_set.{device}"] = str(population["rows_per_device_standard_set"].get(device, 0))
        out[f"rows.complete.{device}"] = str(plan["complete_rows_per_device"].get(device, 0))
    for column, share in (plan_file or {}).get("usability", {}).items():
        out[f"usability.{column}_pct"] = ct.percent(share)

    out.update(ct._arm_facts("ishcdb", analysis["primary"]))
    for name in ("all_measured_rows", "min_rows_10", "diamagnetic_target_for_every_device"):
        arm = analysis["secondary"].get(name)
        if arm is not None and arm.get("eligible_devices") is not None:
            out.update(ct._arm_facts(f"secondary.{name}", arm))
    centred = analysis["secondary"].get("offset_removed", {})
    for name, entry in centred.get("models", {}).items():
        for split in ("cv", "lodo"):
            out[f"offset.{name}.{split}.total"] = ct.error(entry[split]["total"])
            out[f"offset.{name}.{split}.centred"] = ct.error(entry[split]["centred"])
            out[f"offset.{name}.{split}.mean_abs_offset"] = ct.error(entry[split]["mean_absolute_offset"])
    reference = analysis.get("iss04", {})
    if reference.get("computed"):
        out["iss04.unit_equal"] = ct.error(reference["unit_equal"])
        out["iss04.pooled"] = ct.error(reference["pooled"])
    out.update(hdb5_facts())
    return out


# --- the tables of the supplement ---------------------------------------------


def result_table(analysis: dict[str, Any]) -> str:
    """Each model under both splits, on the standard set and on every measured row."""
    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Population & Model & pooled CV & matched CV & held out & ratio \\", r"\midrule",
    ]
    arms = [("standard set", analysis["primary"]), ("every measured row", analysis["secondary"]["all_measured_rows"])]
    for label, arm in arms:
        for index, (name, scores) in enumerate(arm.get("models", {}).items()):
            lines.append(
                f"{label if index == 0 else ''} & {ct.MODEL_NAMES.get(name, ct._escape(name))} & "
                f"{ct.error(scores['cv_pooled_all_rows'])} & {ct.error(scores['cv_matched'])} & "
                f"{ct.error(scores['lodo'])} & {ct.ratio(scores['transfer_ratio'])}$\\times$ \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join([*lines, r"\end{tabular}"])


def population_table(analysis: dict[str, Any]) -> str:
    """One row per measured device: delivered, standard set, complete, scored."""
    plan = analysis["plan"]
    population, eligible = plan["population"], set(plan["eligible_devices"])
    lines = [r"\begin{tabular}{lrrrc}", r"\toprule", r"Device & rows delivered & in the standard set & complete & scored \\", r"\midrule"]
    devices = sorted(population["rows_per_device_measured"], key=lambda d: -plan["complete_rows_per_device"].get(d, 0))
    for device in devices:
        lines.append(
            f"{ct._escape(device)} & {population['rows_per_device_delivered'][device]} & "
            f"{population['rows_per_device_standard_set'].get(device, 0)} & {plan['complete_rows_per_device'].get(device, 0)} & "
            f"{'yes' if device in eligible else 'no'} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def mapping_table(analysis: dict[str, Any]) -> str:
    """What stands for each of ISS04's regressors, and whether it survived the usability rule."""
    plan = analysis["plan"]
    frozen = set(plan["frozen_features"])
    lines = [r"\begin{tabular}{p{3.4cm}p{3.6cm}p{3.0cm}p{3.6cm}}", r"\toprule", r"Quantity & column & conversion & status \\", r"\midrule"]
    lines.append(
        r"confinement time & \texttt{TAUEDIA}, or \texttt{TAUETH} for Heliotron E and TJ-II & s, as delivered & target \\"
    )
    for canonical, (source, _, _) in ishcdb.FEATURE_SOURCES.items():
        status = "used" if canonical in frozen else "omitted, " + ct._escape(plan["omitted_features"][canonical])
        lines.append(f"{QUANTITY_NAMES[canonical]} & \\texttt{{{source}}} & {CONVERSIONS[canonical]} & {status} \\\\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def offset_table(analysis: dict[str, Any]) -> str:
    """Each model's error under both splits, whole and with the device's mean log residual removed."""
    centred = analysis["secondary"]["offset_removed"]
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Model & CV & CV, centred & held out & held out, centred \\", r"\midrule",
    ]
    for name, entry in centred.get("models", {}).items():
        lines.append(
            f"{ct.MODEL_NAMES.get(name, ct._escape(name))} & {ct.error(entry['cv']['total'])} & {ct.error(entry['cv']['centred'])} & "
            f"{ct.error(entry['lodo']['total'])} & {ct.error(entry['lodo']['centred'])} \\\\"
        )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS.relative_to(ROOT)} does not exist; analysis_ishcdb.py writes it.")
    analysis = json.loads(RESULTS.read_text(encoding="utf-8"))
    plan_file = json.loads(ishcdb.default_plan_path().read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "result_table.tex": result_table(analysis),
        "population_table.tex": population_table(analysis),
        "mapping_table.tex": mapping_table(analysis),
        "offset_table.tex": offset_table(analysis),
        "facts.json": json.dumps(facts(analysis, plan_file), indent=2, sort_keys=True),
    }
    for name, text in outputs.items():
        (OUTPUT_DIR / name).write_text(text + "\n", encoding="utf-8")
        print(f"\n%%%% {name}\n{text}")
    print(f"\nwrote {len(outputs)} files to {OUTPUT_DIR.relative_to(ROOT)}/. Paste from them; type nothing.")


if __name__ == "__main__":
    main()
