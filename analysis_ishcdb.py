"""The validation-protocol comparison on the stellarator-heliotron database, under rules fixed in advance.

Run ``python3 analysis_ishcdb.py`` to regenerate everything under ``results/``
for this replication. It refuses to run until the file and the plan are pinned
(see ``ishcdb.load_frozen_plan``), and the first run on the real file is the run
of record.

This is the second external replication, and the rules are
``docs/ishcdb-replication-lock.md``. The first, on CICLOP, is blocked on access
to its data, and its lock says any other dataset needs its own lock before its
scores are seen. So nothing here is a new method: both splits, the matched way
of scoring them, the five-way verdict and the distance diagnostic are
``analysis_ciclop``'s, called and not copied, which is what makes the two
replications one experiment run twice.

What is specific to this database:

    the law      ISS04 with its renormalisation factor at one, as the historical
                 reference outside the ranking, where the manuscript has
                 IPB98(y,2).

    the offset   ISS04 could not fit these devices with one constant: it carries
                 a factor for each device and configuration. So a power law
                 refitted on the other devices should be expected to miss a
                 held-out one by a roughly constant factor, whatever its
                 exponents do. ``centred_errors`` reports each model's error
                 with that device-level offset removed, which separates a
                 missing constant from wrong trends. The lock names it in
                 advance, and it cannot change the verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analysis_ciclop as ac
import ciclop
import hdb5
import ishcdb
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ISS04 = "iss04_analytic"


def iss04_reference(dataset: pd.DataFrame, eligible: list[str]) -> dict[str, Any]:
    """ISS04 at a renormalisation factor of one, per eligible device. A reference with no rank."""
    if ishcdb.ISS04_COLUMN not in dataset or dataset[ishcdb.ISS04_COLUMN].isna().all():
        return {"computed": False, "because": "ISS04 needs all six regressors, and the frozen set lacks one"}
    residual = np.log(dataset[ishcdb.ISS04_COLUMN].to_numpy(dtype=float)) - np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    devices = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    per_device = {d: float(np.sqrt(np.mean(residual[devices == d] ** 2))) for d in eligible}
    offsets = {d: float(np.mean(residual[devices == d])) for d in eligible}
    scored = np.isin(devices, eligible)
    return {
        "computed": True,
        "unit_equal": float(np.mean(list(per_device.values()))),
        "pooled": float(np.sqrt(np.mean(residual[scored] ** 2))),
        "per_device": per_device,
        # The mean log residual per device: ISS04's own renormalisation factor is exp(-offset).
        "mean_log_residual_per_device": offsets,
    }


def _log_predictions(
    dataset: pd.DataFrame, columns: tuple[str, ...], eligible: list[str], models: tuple[str, ...]
) -> dict[str, dict[str, np.ndarray]]:
    """Out-of-fold and held-out log predictions, fitted exactly as ``analysis_robustness`` fits them."""
    features = dataset[list(columns)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    devices = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy().astype(str)
    n_splits = min(hdb5.N_CV_FOLDS, int(pd.Series(groups).nunique()))
    zoo = ac.build_zoo(with_controls=False, with_gp=False)
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in models:
        held_out = np.full(len(dataset), np.nan)
        for device in eligible:
            held = devices == device
            model = hdb5.clone_pipeline(zoo[name])
            with hdb5._suppress_benign_matmul_warnings():
                hdb5.fit_pipeline(model, features.loc[~held], log_tau[~held])
                held_out[held] = model.predict(features.loc[held])
        out[name] = {"cv": hdb5._grouped_cv_predictions(zoo[name], features, log_tau, groups, n_splits), "lodo": held_out}
    return out


def centred_errors(dataset: pd.DataFrame, columns: tuple[str, ...], arm: dict[str, Any]) -> dict[str, Any]:
    """Each model's error with the device's mean log residual removed, under both splits.

    Per device the mean squared residual splits exactly into the squared mean
    and the variance about it. The first part is a constant factor on the whole
    device, which is what a renormalisation factor absorbs. The second is what
    is left once that constant is granted, and is the part that says whether
    the model has the trends right.
    """
    if not arm.get("scored"):
        return {"computed": False}
    eligible = list(arm["eligible_devices"])
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    devices = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy().astype(str)
    predictions = _log_predictions(dataset, columns, eligible, ac.RANKED_MODELS)
    out: dict[str, Any] = {"computed": True, "models": {}}
    for name, by_split in predictions.items():
        entry: dict[str, Any] = {}
        for split, predicted in by_split.items():
            residual = predicted - log_tau
            total = {d: float(np.sqrt(np.mean(residual[devices == d] ** 2))) for d in eligible}
            offset = {d: float(np.mean(residual[devices == d])) for d in eligible}
            centred = {d: float(np.std(residual[devices == d])) for d in eligible}
            entry[split] = {
                "total": float(np.mean(list(total.values()))),
                "centred": float(np.mean(list(centred.values()))),
                "mean_absolute_offset": float(np.mean(np.abs(list(offset.values())))),
                "total_per_device": total,
                "centred_per_device": centred,
                "offset_per_device": offset,
            }
        out["models"][name] = entry
    return out


def analyze(
    plan: dict[str, Any], standard: pd.DataFrame, everything: pd.DataFrame, *, with_gp: bool = True
) -> dict[str, Any]:
    columns = ciclop.log_feature_columns(tuple(plan["frozen_features"]))
    fixed_reasons = tuple(
        ishcdb.evaluability(plan["usability"], tuple(plan["frozen_features"]), ishcdb.MIN_EVALUABLE_ROWS, ishcdb.MIN_EVALUABLE_DEVICES)
    )
    primary_models = (*ac.RANKED_MODELS, *ac.CONTROL_MODELS, *((ac.GP_MODEL,) if with_gp else ()))

    primary = ac.score_arm(
        standard, columns, min_rows=ishcdb.MIN_HELD_OUT_ROWS, models=primary_models,
        zoo=ac.build_zoo(with_controls=True, with_gp=with_gp), other_reasons=fixed_reasons,
    )
    primary["distance"] = ac.distance_output(standard, columns, primary)

    def secondary(frame: pd.DataFrame, *, min_rows: int = ishcdb.MIN_HELD_OUT_ROWS) -> dict[str, Any]:
        arm = ac.score_arm(frame, columns, min_rows=min_rows, other_reasons=fixed_reasons)
        arm["distance"] = ac.distance_output(frame, columns, arm)
        return arm

    diamagnetic = standard.assign(**{hdb5.TARGET_COLUMN: standard[ishcdb.DIAMAGNETIC_TARGET_COLUMN]})
    diamagnetic = diamagnetic.loc[np.isfinite(diamagnetic[hdb5.TARGET_COLUMN]) & (diamagnetic[hdb5.TARGET_COLUMN] > 0)].reset_index(drop=True)

    headline = f"{ac.FOREST}_vs_{ac.RIDGE}"
    return {
        "provenance": {"lock": plan["lock"], "file": plan["file"], "plan_sha256": ishcdb.ISHCDB_PLAN_SHA256},
        "plan": {
            key: plan[key]
            for key in (
                "frozen_features", "omitted_features", "population", "n_complete_rows", "complete_rows_per_device",
                "eligible_devices", "discharges", "evaluable", "not_evaluable_because",
            )
        },
        "verdict": primary["verdicts"][headline],
        "verdict_booster": primary["verdicts"][f"{ac.BOOSTER}_vs_{ac.RIDGE}"],
        "primary": primary,
        "iss04": iss04_reference(standard, primary["eligible_devices"]),
        "secondary": {
            "all_measured_rows": secondary(everything),
            "min_rows_10": secondary(standard, min_rows=ishcdb.SENSITIVITY_MIN_ROWS),
            "diamagnetic_target_for_every_device": secondary(diamagnetic),
            "offset_removed": centred_errors(standard, columns, primary),
        },
    }


def score_table(analysis: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, arm in (("standard_set", analysis["primary"]), ("all_measured_rows", analysis["secondary"]["all_measured_rows"])):
        for name, scores in arm.get("models", {}).items():
            rows.append({
                "population": label, "model": name, "n_devices": len(arm["eligible_devices"]),
                **{k: scores[k] for k in ("cv_pooled_all_rows", "cv_matched", "lodo", "transfer_ratio", "cv_matched_pooled", "lodo_pooled")},
            })
    return pd.DataFrame(rows)


def per_device_table(analysis: dict[str, Any]) -> pd.DataFrame:
    arm = analysis["primary"]
    distances = arm.get("distance", {}).get("mahalanobis", {})
    reference = analysis["iss04"].get("per_device", {})
    rows = []
    for name, scores in arm.get("models", {}).items():
        for device in arm["eligible_devices"]:
            rows.append({
                "device": device, "model": name, "n_rows": arm["rows_per_device"][device],
                "cv": scores["cv_per_device"][device], "lodo": scores["lodo_per_device"][device],
                "mahalanobis": distances.get(device, float("nan")), "iss04": reference.get(device, float("nan")),
            })
    return pd.DataFrame(rows)


def main() -> None:
    plan, data_path = ishcdb.load_frozen_plan()
    standard, everything = ishcdb.frames_from_plan(plan, data_path)
    analysis = analyze(plan, standard, everything)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_strict(RESULTS_DIR / "ishcdb.json", analysis)
    write_dataframe_csv_atomic(RESULTS_DIR / "ishcdb_scores.csv", score_table(analysis))
    write_dataframe_csv_atomic(RESULTS_DIR / "ishcdb_per_device.csv", per_device_table(analysis))
    figure_input = {**analysis, "hdb5_feature_matched": {}, "plan": {**analysis["plan"], "device_in_hdb5": {}}}
    ac.plot_ciclop(figure_input, RESULTS_DIR / "ishcdb.png", dataset_label="ISHCDB, standard set", open_marker_label=None)

    primary = analysis["primary"]
    print("--- ISHCDB: the validation-protocol comparison, under the locked rules ---")
    print(f"  features ({len(primary['features'])} of 6): {primary['features']}")
    print(f"  {primary['n_rows']} rows, {len(primary['eligible_devices'])} devices scored: {primary['eligible_devices']}")
    if primary.get("scored"):
        print(f"\n  {'model':<26}{'CV matched':>12}{'LODO':>10}{'ratio':>9}")
        for name, scores in primary["models"].items():
            print(f"  {name:<26}{scores['cv_matched']:>12.3f}{scores['lodo']:>10.3f}{scores['transfer_ratio']:>9.2f}")
        if analysis["iss04"].get("computed"):
            print(f"  {'ISS04, f_ren = 1':<26}{'':>12}{analysis['iss04']['unit_equal']:>10.3f}")
        for key, pair in primary["pairs"].items():
            interval = pair["paired_interval"]
            print(
                f"\n  {key}: worse on {pair['n_devices_flexible_worse']} of {pair['n_devices']}, mean gap {pair['mean_gap']:+.3f} "
                f"[{interval['ci_low']:+.3f}, {interval['ci_high']:+.3f}], D = {pair['differential_degradation']:.2f}"
            )
    for reason in primary["not_evaluable_because"]:
        print(f"  not evaluable: {reason}")
    print(f"\n  verdict: {analysis['verdict']}   (booster: {analysis['verdict_booster']})")


if __name__ == "__main__":
    main()
