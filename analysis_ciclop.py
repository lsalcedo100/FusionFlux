"""The validation-protocol comparison on CICLOP, under rules fixed in advance.

Run ``python3 analysis_ciclop.py`` to regenerate everything under ``results/``
for the external replication. It refuses to run until the plan is frozen and
pinned (see ``ciclop.load_frozen_plan``), and the first run on the real file is
the run of record.

Every other fusion result in this repository rests on one ITPA database family.
A reader can always say the ranking inversion belongs to that compilation. This
script asks the question that answers it, and it was written before the answer
was known: when the same model comparison is moved from validation inside known
devices to prediction of an entirely held-out physical device, how does model
performance change in a separately assembled multi-machine dataset?

The rules are ``docs/ciclop-replication-lock.md``. The ones that shape this file:

    matched scoring   Both splits are summarised the same way, per eligible
                      device and then averaged with each device counting once,
                      so the split is the only thing that differs between them.

    one verdict rule  Five outcomes, the first that holds. "Cannot be evaluated"
                      is one of them, and so is "contradicts the original".

    like for like     CICLOP may not support all nine features. Whatever subset
                      it does support, HDB5 is rerun on that identical subset,
                      and that rerun is the comparator, not the manuscript's
                      nine-feature headline.

    three outputs     CV against device holdout, the paired ranking across
                      held-out devices, and error against extrapolation
                      distance. This is not a second copy of the paper.

Nothing here is tuned. The models are the manuscript's, built by the same
factories with the same seed, and both splits run through
``analysis_robustness``, which is where the manuscript's own matched numbers
come from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

import analysis_robustness as ar
import ciclop
import hdb5
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

RIDGE = "ridge_loglinear"
FOREST = "random_forest"
BOOSTER = "hist_gradient_boosting"
RANKED_MODELS: tuple[str, ...] = ar.CONTENDERS
# Scored as the manuscript scores them, outside the ranking.
CONTROL_MODELS: tuple[str, ...] = ("mean_baseline", "ridge_log_quadratic", "ridge_log_cubic")
GP_MODEL = "gp_linear_rbf"
# The flexible model first, the power law second. The first pair carries the
# verdict; the second goes through the same rule and is reported beside it.
PAIRS: tuple[tuple[str, str], ...] = ((FOREST, RIDGE), (BOOSTER, RIDGE))

CANNOT_BE_EVALUATED = "cannot_be_evaluated"
REPRODUCES_INVERSION = "reproduces_inversion"
CONTRADICTS = "contradicts"
DEGRADATION_WITHOUT_INVERSION = "degradation_without_inversion"
NO_IMPORTANT_DEGRADATION = "no_important_differential_degradation"
VERDICTS: tuple[str, ...] = (
    CANNOT_BE_EVALUATED,
    REPRODUCES_INVERSION,
    CONTRADICTS,
    DEGRADATION_WITHOUT_INVERSION,
    NO_IMPORTANT_DEGRADATION,
)

# What DB5.2.3 offers as the counterpart of CICLOP's injected additional power:
# the power each heating system injected or coupled, summed. ``PINJ`` alone is
# one neutral beam system, zero on the 361 STD5 rows heated by radio frequency
# alone, and is not the total. The lock's deviations log records how that was
# found.
HDB5_INJECTED_POWER_COLUMNS: tuple[str, ...] = ("PINJ", "PINJ2", "PICRHC", "PECRHC")
IPB98_INPUTS: tuple[str, ...] = (
    "ip_ma", "bt_t", "ne_line_1e19_m3", "p_loss_mw", "r_m", "inverse_aspect_ratio", "kappa", "m_eff_amu",
)


# --- The verdict -------------------------------------------------------------


def differential_degradation(cv_flexible: float, lodo_flexible: float, cv_ridge: float, lodo_ridge: float) -> float:
    """D of the lock: how much more the flexible model degrades than the power law."""
    return (lodo_flexible / cv_flexible) / (lodo_ridge / cv_ridge)


def verdict(
    *,
    not_evaluable: bool,
    cv_flexible: float,
    lodo_flexible: float,
    cv_ridge: float,
    lodo_ridge: float,
    n_devices_flexible_worse: int,
    n_devices: int,
) -> str:
    """Section 9 of the lock, in the order it is written. The first condition that holds wins."""
    if not_evaluable:
        return CANNOT_BE_EVALUATED
    half = n_devices / 2
    degradation = differential_degradation(cv_flexible, lodo_flexible, cv_ridge, lodo_ridge)
    important = degradation >= ciclop.DEGRADATION_BOUNDARY
    crossed = cv_flexible < cv_ridge and lodo_flexible > lodo_ridge
    # A crossing alone is not a reproduction: the ranks can trade places inside
    # the noise. It has to come with the same size of effect verdict 4 asks for.
    if crossed and n_devices_flexible_worse > half and important:
        return REPRODUCES_INVERSION
    if lodo_flexible < lodo_ridge and n_devices_flexible_worse < half and degradation <= 1.0:
        return CONTRADICTS
    if important:
        return DEGRADATION_WITHOUT_INVERSION
    return NO_IMPORTANT_DEGRADATION


# --- One arm: both splits, matched -------------------------------------------


def build_zoo(*, with_controls: bool, with_gp: bool) -> dict[str, Any]:
    """The manuscript's models from the manuscript's factories. Nothing is set here."""
    zoo = dict(hdb5._assemble_zoo(include_controls=with_controls))
    if with_controls:
        import analysis_extrapolation as ae

        zoo.update({k: v for k, v in ae.build_flexibility_ladder().items() if k not in zoo})
    if with_gp:
        import gp

        zoo[GP_MODEL] = gp.build_gp_models()[GP_MODEL]
    return zoo


def _pooled(per_unit: dict[str, float], rows: dict[str, int], units: tuple[str, ...]) -> float:
    """Pooled log-RMSE over the named units, exactly, from their own errors and sizes."""
    total = sum(rows[u] for u in units)
    return float(np.sqrt(sum(rows[u] * per_unit[u] ** 2 for u in units) / total))


def floor_reasons(n_rows: int, n_devices: int) -> list[str]:
    """The two floors of verdict 1 that depend on which rows an arm scores."""
    reasons = []
    if n_rows < ciclop.MIN_EVALUABLE_ROWS:
        reasons.append(f"{n_rows} rows, below {ciclop.MIN_EVALUABLE_ROWS}")
    if n_devices < ciclop.MIN_EVALUABLE_DEVICES:
        reasons.append(f"{n_devices} eligible devices, below {ciclop.MIN_EVALUABLE_DEVICES}")
    return reasons


def score_arm(
    dataset: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    min_rows: int,
    models: tuple[str, ...] = RANKED_MODELS,
    zoo: dict[str, Any] | None = None,
    other_reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Both splits on one population, scored the matched way, with the verdict.

    Cross-validation is run once over every row, and its per-device errors are
    then read off for the eligible devices alone. So both splits train from the
    same universe and score the same rows, and what differs is whether the
    scored device's other pulses were in training.
    """
    units = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy().astype(str)
    rows = {str(k): int(v) for k, v in pd.Series(units).value_counts().items()}
    eligible = tuple(hdb5.eligible_tokamaks(dataset, min_rows=min_rows)) if len(dataset) else ()
    reasons = [*other_reasons, *floor_reasons(int(len(dataset)), len(eligible))]
    arm: dict[str, Any] = {
        "n_rows": int(len(dataset)),
        "min_rows": min_rows,
        "features": list(feature_columns),
        "eligible_devices": list(eligible),
        "rows_per_device": rows,
        "not_evaluable_because": reasons,
        "scored": len(eligible) >= 2,
    }
    if not arm["scored"]:
        arm["verdicts"] = {f"{a}_vs_{b}": CANNOT_BE_EVALUATED for a, b in PAIRS}
        return arm

    zoo = build_zoo(with_controls=False, with_gp=False) if zoo is None else zoo
    cv = ar._cross_validate(
        dataset, units, feature_columns=feature_columns, models=models, zoo=zoo, reference_column=None
    )
    lodo = ar._leave_one_unit_out(
        dataset, units, feature_columns=feature_columns, models=models, zoo=zoo, reference_column=None, min_rows=min_rows
    )

    arm["models"] = {}
    for name in models:
        cv_matched = float(np.mean([cv[name]["per_unit"][u] for u in eligible]))
        lodo_matched = float(lodo[name]["unit_equal"])
        arm["models"][name] = {
            "cv_pooled_all_rows": float(cv[name]["pooled_rows"]),
            "cv_matched": cv_matched,
            "lodo": lodo_matched,
            "transfer_ratio": lodo_matched / cv_matched,
            "cv_matched_pooled": _pooled(cv[name]["per_unit"], rows, eligible),
            "lodo_pooled": float(lodo[name]["pooled_rows"]),
            "cv_per_device": {u: float(cv[name]["per_unit"][u]) for u in eligible},
            "lodo_per_device": {u: float(lodo[name]["per_unit"][u]) for u in eligible},
        }

    arm["pairs"], arm["verdicts"] = {}, {}
    for flexible, ridge in PAIRS:
        if flexible not in arm["models"] or ridge not in arm["models"]:
            continue
        a, b = arm["models"][flexible], arm["models"][ridge]
        sign = cast("dict[str, Any]", ar._sign_test(a["lodo_per_device"], b["lodo_per_device"]))
        key = f"{flexible}_vs_{ridge}"
        arm["pairs"][key] = {
            "n_devices": sign["n_units"],
            "n_devices_flexible_worse": sign["n_units_a_worse"],
            "mean_gap": sign["mean_difference"],
            "exact_two_sided_p": sign["exact_two_sided_p"],
            "paired_interval": ar.paired_interval(a["lodo_per_device"], b["lodo_per_device"]),
            "differential_degradation": differential_degradation(a["cv_matched"], a["lodo"], b["cv_matched"], b["lodo"]),
            "differential_degradation_pooled": differential_degradation(
                a["cv_matched_pooled"], a["lodo_pooled"], b["cv_matched_pooled"], b["lodo_pooled"]
            ),
            "flexible_wins_cv": bool(a["cv_matched"] < b["cv_matched"]),
            "flexible_loses_lodo": bool(a["lodo"] > b["lodo"]),
        }
        arm["verdicts"][key] = verdict(
            not_evaluable=bool(reasons),
            cv_flexible=a["cv_matched"],
            lodo_flexible=a["lodo"],
            cv_ridge=b["cv_matched"],
            lodo_ridge=b["lodo"],
            n_devices_flexible_worse=int(sign["n_units_a_worse"]),
            n_devices=int(sign["n_units"]),
        )
    return arm


# --- Output 3: error against extrapolation distance --------------------------


def distance_output(dataset: pd.DataFrame, feature_columns: tuple[str, ...], arm: dict[str, Any]) -> dict[str, Any]:
    """Each model's per-device holdout error against that device's distance from its training rows.

    The distance is ``hdb5.extrapolation_diagnostic``'s, unchanged. With at most
    seven devices the correlation is described and not tested, which is why no
    permutation p-value is computed here where the manuscript computes one.
    """
    import analysis_extrapolation as ae

    if not arm.get("scored"):
        return {"computed": False}
    devices = list(arm["eligible_devices"])
    distances = {
        device: float(hdb5.extrapolation_diagnostic(dataset, device, feature_columns=feature_columns).feature_mahalanobis)
        for device in devices
    }
    ordered = np.array([distances[d] for d in devices])
    correlations = {
        name: float(ae.spearman(ordered, np.array([scores["lodo_per_device"][d] for d in devices])))
        for name, scores in arm["models"].items()
    } if len(devices) >= 3 else {}
    return {"computed": True, "n_devices": len(devices), "mahalanobis": distances, "spearman": correlations}


# --- The comparator: HDB5 on the identical feature set -----------------------


def hdb5_matched_dataset(features: tuple[str, ...], *, power_is_injected: bool) -> pd.DataFrame:
    """STD5, held out by physical device, on CICLOP's frozen features and power definition."""
    import replication as rp

    dataset = hdb5.with_device_column(hdb5.prepare_dataset())
    dataset[hdb5.TOKAMAK_LABEL_COLUMN] = dataset[hdb5.DEVICE_COLUMN]
    if power_is_injected and "p_loss_mw" in features:
        systems = rp.db523_columns(HDB5_INJECTED_POWER_COLUMNS).apply(pd.to_numeric, errors="coerce")
        injected_w = systems.fillna(0.0).clip(lower=0.0).sum(axis=1).to_numpy()
        dataset["p_loss_mw"] = injected_w / rp.DB523_UNIT_SCALES["PLTH"]
    dataset[ciclop.CONFIGURATION_COLUMN] = ciclop.TOKAMAK
    return ciclop.analysis_frame(dataset, features)


def hdb5_rerun(plan: dict[str, Any]) -> dict[str, Any]:
    features = tuple(plan["frozen_features"])
    columns = ciclop.log_feature_columns(features)
    dataset = hdb5_matched_dataset(features, power_is_injected=bool(plan["power_is_injected"]))
    out: dict[str, Any] = {
        "n_rows": int(len(dataset)),
        "power": "injected and coupled, summed" if plan["power_is_injected"] else "loss power, as STD5 delivers it",
        "arms": {},
    }
    for threshold in (ciclop.MIN_HELD_OUT_ROWS, ciclop.SENSITIVITY_MIN_ROWS):
        arm = score_arm(dataset, columns, min_rows=threshold)
        arm["distance"] = distance_output(dataset, columns, arm)
        out["arms"][f"min_rows_{threshold}"] = arm
    return out


# --- Secondary analyses. None of them can change the verdict. ----------------


def ipb98_reference(dataset: pd.DataFrame, plan: dict[str, Any], eligible: list[str]) -> dict[str, Any]:
    """IPB98(y,2) on the H-mode rows: a historical reference with no rank."""
    missing = [c for c in IPB98_INPUTS if c not in plan["frozen_features"]]
    if missing:
        return {"computed": False, "because": f"IPB98(y,2) needs {missing}, which the frozen set lacks"}
    h_mode = dataset.loc[dataset[ciclop.H_MODE_COLUMN] & dataset[hdb5.TOKAMAK_LABEL_COLUMN].isin(eligible)]
    if h_mode.empty:
        return {"computed": False, "because": "no H-mode rows on an eligible device"}
    actual = np.log(h_mode[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    predicted = np.log(hdb5.ipb98y2_tau_s(h_mode).to_numpy(dtype=float))
    devices = h_mode[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    per_device = {str(d): float(np.sqrt(np.mean((predicted - actual)[devices == d] ** 2))) for d in pd.unique(devices)}
    return {
        "computed": True,
        "n_rows": int(len(h_mode)),
        "power": "injected additional power" if plan["power_is_injected"] else "loss power",
        "pooled": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "unit_equal": float(np.mean(list(per_device.values()))),
        "per_device": per_device,
    }


def db523_overlap(dataset: pd.DataFrame, plan: dict[str, Any]) -> pd.Series | None:
    """Which analysed pulses are also DB5.2.3 discharges, matched on machine and shot."""
    import replication as rp

    try:
        full = rp.load_db523_raw()
    except FileNotFoundError:
        return None
    known = set(full["TOK"].astype(str).str.strip() + "::" + full["SHOT"].astype(str).str.split(".").str[0])
    codes = {name: entry.get("hdb5_tok") or [] for name, entry in plan["mapping"]["facilities"].items()}
    pulse = dataset[ciclop.PULSE_COLUMN].astype(str).str.split(".").str[0]
    return pd.Series(
        [any(f"{code}::{p}" in known for code in codes[f]) for f, p in zip(dataset[ciclop.FACILITY_COLUMN], pulse, strict=True)],
        index=dataset.index,
    )


def grouped_by_device_and_year(dataset: pd.DataFrame) -> pd.DataFrame | None:
    """The same rows, with cross-validation folds cut between campaigns and not between pulses.

    CICLOP is a database of record pulses, and one device's pulses from one
    campaign are often repeats of one scenario. Grouping by pulse lets a fold
    boundary fall between two repeats, which flatters any model that
    interpolates. A pulse with no readable year keeps itself as its group.
    """
    year = dataset[ciclop.YEAR_COLUMN]
    if not year.notna().any():
        return None
    campaign = dataset[hdb5.TOKAMAK_LABEL_COLUMN].astype(str) + "::" + year.map(lambda y: "" if pd.isna(y) else str(int(y)))
    return dataset.assign(**{hdb5.GROUP_COLUMN: campaign.where(year.notna(), dataset[hdb5.GROUP_COLUMN])})


def absent_from_hdb5(arm: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Outputs 1 and 2 over the devices HDB5 does not contain. Descriptive: at most three devices."""
    if not arm.get("scored"):
        return {"computed": False}
    devices = [d for d in arm["eligible_devices"] if not plan["device_in_hdb5"].get(d, False)]
    if not devices:
        return {"computed": False, "because": "every eligible device is also an HDB5 device"}
    out: dict[str, Any] = {"computed": True, "devices": devices, "models": {}}
    for name, scores in arm["models"].items():
        out["models"][name] = {
            "cv_matched": float(np.mean([scores["cv_per_device"][d] for d in devices])),
            "lodo": float(np.mean([scores["lodo_per_device"][d] for d in devices])),
        }
    forest, ridge = arm["models"][FOREST]["lodo_per_device"], arm["models"][RIDGE]["lodo_per_device"]
    out["n_devices_forest_worse"] = int(sum(forest[d] > ridge[d] for d in devices))
    return out


# --- The whole analysis ------------------------------------------------------


def analyze(plan: dict[str, Any], dataset: pd.DataFrame, *, with_hdb5: bool = True, with_gp: bool = True) -> dict[str, Any]:
    columns = ciclop.log_feature_columns(tuple(plan["frozen_features"]))
    # The floors on the target and the feature count hold for every arm. The
    # floors on rows and devices depend on which rows an arm scores, and
    # ``score_arm`` recomputes those, so they are passed here at values that
    # clear them.
    plan_reasons = tuple(
        ciclop.evaluability(
            plan["usability"], tuple(plan["frozen_features"]), ciclop.MIN_EVALUABLE_ROWS, ciclop.MIN_EVALUABLE_DEVICES
        )
    )
    primary_models = (*RANKED_MODELS, *CONTROL_MODELS, *((GP_MODEL,) if with_gp else ()))

    primary = score_arm(
        dataset, columns, min_rows=ciclop.MIN_HELD_OUT_ROWS, models=primary_models,
        zoo=build_zoo(with_controls=True, with_gp=with_gp), other_reasons=plan_reasons,
    )
    primary["distance"] = distance_output(dataset, columns, primary)

    def secondary(frame: pd.DataFrame, *, min_rows: int = ciclop.MIN_HELD_OUT_ROWS) -> dict[str, Any]:
        return score_arm(frame, columns, min_rows=min_rows, other_reasons=plan_reasons)

    h_mode = dataset.loc[dataset[ciclop.H_MODE_COLUMN]].reset_index(drop=True)
    h_mode_eligible = hdb5.eligible_tokamaks(h_mode, min_rows=ciclop.MIN_HELD_OUT_ROWS) if len(h_mode) else []
    h_mode_reasons = floor_reasons(int(len(h_mode)), len(h_mode_eligible))
    split = dataset.assign(**{hdb5.TOKAMAK_LABEL_COLUMN: dataset[ciclop.SPLIT_DEVICE_COLUMN]})
    overlap = db523_overlap(dataset, plan)
    by_campaign = grouped_by_device_and_year(dataset)

    secondaries: dict[str, Any] = {
        "cv_grouped_by_device_and_year": (
            secondary(by_campaign)
            if by_campaign is not None
            else {"scored": False, "not_evaluable_because": ["the file carries no pulse date"]}
        ),
        "min_rows_30": secondary(dataset, min_rows=ciclop.SENSITIVITY_MIN_ROWS),
        # The lock runs this arm only if the subset clears the floors, and marks it otherwise.
        "h_mode_only": secondary(h_mode) if not h_mode_reasons else {"scored": False, "not_evaluable_because": h_mode_reasons},
        "tore_supra_and_west_split": secondary(split),
        "ipb98y2_on_h_mode_rows": ipb98_reference(dataset, plan, primary["eligible_devices"]),
        "devices_absent_from_hdb5": absent_from_hdb5(primary, plan),
        "db523_overlap": (
            {"computed": False, "because": "DB5.2.3 is not on this machine"}
            if overlap is None
            else {
                "computed": True,
                "n_pulses_also_in_db523": int(overlap.sum()),
                "without_them": secondary(dataset.loc[~overlap].reset_index(drop=True)),
            }
        ),
    }

    headline = f"{FOREST}_vs_{RIDGE}"
    return {
        "provenance": {
            "lock": plan["lock"],
            "file": plan["file"],
            "plan_sha256": ciclop.CICLOP_PLAN_SHA256,
        },
        "plan": {
            key: plan[key]
            for key in (
                "frozen_features", "omitted_features", "power_is_injected", "n_complete_rows",
                "complete_rows_per_device", "device_in_hdb5", "eligible_devices", "evaluable", "not_evaluable_because",
            )
        },
        "verdict": primary["verdicts"][headline],
        "verdict_booster": primary["verdicts"][f"{BOOSTER}_vs_{RIDGE}"],
        "primary": primary,
        "hdb5_feature_matched": hdb5_rerun(plan) if with_hdb5 else {"computed": False},
        "secondary": secondaries,
    }


# --- Tables and the figure ---------------------------------------------------


def score_table(analysis: dict[str, Any]) -> pd.DataFrame:
    rows = []
    arms = {"ciclop": analysis["primary"]}
    arms.update({f"hdb5_{k}": v for k, v in analysis["hdb5_feature_matched"].get("arms", {}).items()})
    for label, arm in arms.items():
        for name, scores in arm.get("models", {}).items():
            rows.append({
                "dataset": label, "model": name, "n_devices": len(arm["eligible_devices"]),
                **{k: scores[k] for k in ("cv_pooled_all_rows", "cv_matched", "lodo", "transfer_ratio", "cv_matched_pooled", "lodo_pooled")},
            })
    return pd.DataFrame(rows)


def per_device_table(analysis: dict[str, Any]) -> pd.DataFrame:
    arm = analysis["primary"]
    distances = arm.get("distance", {}).get("mahalanobis", {})
    rows = []
    for name, scores in arm.get("models", {}).items():
        for device in arm["eligible_devices"]:
            rows.append({
                "device": device, "model": name, "n_rows": arm["rows_per_device"][device],
                "in_hdb5": bool(analysis["plan"]["device_in_hdb5"].get(device, False)),
                "cv": scores["cv_per_device"][device], "lodo": scores["lodo_per_device"][device],
                "mahalanobis": distances.get(device, float("nan")),
            })
    return pd.DataFrame(rows)


def plot_ciclop(analysis: dict[str, Any], path: Path | None = None) -> Path | None:
    """Two panels: both splits for both datasets, then holdout error against distance.

    Every model keeps the colour, marker and line style it has in the paper's
    first figure, all three read from ``figures``, so a reader who has met the
    forest as a dotted orange diamond there meets the same one here.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None
    from figures import (
        FONT_ANNOTATION,
        FONT_LABEL,
        FONT_LEGEND,
        FONT_TICK,
        FONT_TITLE,
        PAPER_WIDTH_IN,
        apply_font_policy,
        model_color,
        model_style,
        save_figure,
    )

    primary = analysis["primary"]
    if not primary.get("scored"):
        return None
    apply_font_policy()
    ink, muted = "#0b0b0b", "#52514e"
    style = {RIDGE: (model_color(RIDGE), "ridge, log-linear"), FOREST: (model_color(FOREST), "random forest"), BOOSTER: (model_color(BOOSTER), "hist gradient boosting")}

    import analysis_extrapolation as ae

    figure, axes = plt.subplots(1, 2, figsize=(PAPER_WIDTH_IN, 3.7), constrained_layout=True)
    panels = [("CICLOP", primary)]
    comparator = analysis["hdb5_feature_matched"].get("arms", {}).get(f"min_rows_{ciclop.MIN_HELD_OUT_ROWS}")
    if comparator and comparator.get("scored"):
        panels.append(("HDB5, same features", comparator))
    for offset, (label, arm) in enumerate(panels):
        x = np.array([0.0, 1.0]) + 2.2 * offset
        for name, (colour, legend_label) in style.items():
            marker, line = model_style(name)
            y = [arm["models"][name]["cv_matched"], arm["models"][name]["lodo"]]
            axes[0].plot(x, y, color=colour, marker=marker, linestyle=line, linewidth=2, markersize=6.5,
                         markeredgecolor="white", markeredgewidth=0.8, label=legend_label if offset == 0 else None)
        # The dataset is named inside the panel, above its pair of points, so
        # the space under the axis is left to the tick labels and the legend.
        axes[0].text(x.mean(), 0.97, f"{label}\n{len(arm['eligible_devices'])} devices", transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=FONT_ANNOTATION, color=muted)
    ticks = [t for offset in range(len(panels)) for t in (2.2 * offset, 2.2 * offset + 1.0)]
    axes[0].set_xticks(ticks, ["CV", "device\nheld out"] * len(panels), fontsize=FONT_TICK)
    axes[0].set_xlim(-0.5, ticks[-1] + 0.5)
    axes[0].set_ylabel("log-RMSE, each device equally", fontsize=FONT_LABEL, color=muted)
    axes[0].set_title("Known devices against an unseen one", fontsize=FONT_TITLE, color=ink, loc="left")

    distances = primary["distance"].get("mahalanobis", {})
    devices = list(primary["eligible_devices"])
    for name in (RIDGE, FOREST):
        colour, label = style[name]
        marker, _ = model_style(name)
        rho = primary["distance"].get("spearman", {}).get(name)
        tagged = label if rho is None else f"{label} ($\\rho$ = {rho:+.2f})"
        for device in devices:
            known = bool(analysis["plan"]["device_in_hdb5"].get(device, False))
            axes[1].plot(distances[device], primary["models"][name]["lodo_per_device"][device], marker=marker, linestyle="none",
                         markersize=7, color=colour, markerfacecolor=colour if known else "white", markeredgecolor=colour, markeredgewidth=1.4)
        axes[1].plot([], [], marker=marker, linestyle="none", color=colour, markersize=7, label=tagged)
    axes[1].plot([], [], marker="o", linestyle="none", markersize=7, markerfacecolor="white", markeredgecolor=muted,
                 markeredgewidth=1.4, label="open: a device HDB5 lacks")
    axes[1].set_xlabel("Mahalanobis distance from the training rows", fontsize=FONT_LABEL, color=muted)
    axes[1].set_ylabel("log-RMSE on the held-out device", fontsize=FONT_LABEL, color=muted)
    axes[1].set_title("Holdout error against distance", fontsize=FONT_TITLE, color=ink, loc="left")
    axes[1].margins(x=0.22)

    # One decimal tick list for both panels. A log axis spanning less than a
    # decade has no major ticks of its own and labels its minors as 3x10^-1.
    ladder = (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
    for axis in axes:
        axis.set_yscale("log")
        low, high = axis.get_ylim()
        axis.set_ylim(low / 1.08, high * 1.45)
        low, high = axis.get_ylim()
        axis.set_yticks([t for t in ladder if low <= t <= high], [f"{t:g}" for t in ladder if low <= t <= high])
        axis.minorticks_off()
        axis.tick_params(labelsize=FONT_TICK, colors=muted)
        axis.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(frameon=False, fontsize=FONT_LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.26), labelcolor=muted)

    # Labels last, once the limits are final, through the placer the paper's
    # first figure uses: a fixed ladder of offsets tried in a fixed order, so
    # the same numbers always give the same figure.
    figure.canvas.draw()
    obstacles = [(distances[d], primary["models"][n]["lodo_per_device"][d]) for d in devices for n in (RIDGE, FOREST)]
    anchors = sorted(((distances[d], primary["models"][FOREST]["lodo_per_device"][d], d) for d in devices), key=lambda point: point[0])
    offsets = ae._place_labels_without_overlap(axes[1], anchors, obstacles, fontsize=FONT_ANNOTATION)
    for x_value, y_value, device in anchors:
        axes[1].annotate(device, (x_value, y_value), xytext=offsets.get(device, (7, 3)), textcoords="offset points",
                         fontsize=FONT_ANNOTATION, color=muted)

    target = RESULTS_DIR / "ciclop.png" if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    saved = save_figure(figure, target)
    plt.close(figure)
    return saved


def main() -> None:
    plan, data_path = ciclop.load_frozen_plan()
    analysis = analyze(plan, ciclop.frame_from_plan(plan, data_path))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_strict(RESULTS_DIR / "ciclop.json", analysis)
    write_dataframe_csv_atomic(RESULTS_DIR / "ciclop_scores.csv", score_table(analysis))
    write_dataframe_csv_atomic(RESULTS_DIR / "ciclop_per_device.csv", per_device_table(analysis))
    plot_ciclop(analysis)

    primary = analysis["primary"]
    print("--- CICLOP: the validation-protocol comparison, under the locked rules ---")
    print(f"  features ({len(primary['features'])} of 9): {primary['features']}")
    print(f"  {primary['n_rows']} tokamak pulses, {len(primary['eligible_devices'])} devices scored: {primary['eligible_devices']}")
    if primary.get("scored"):
        print(f"\n  {'model':<26}{'CV matched':>12}{'LODO':>10}{'ratio':>9}")
        for name, scores in primary["models"].items():
            print(f"  {name:<26}{scores['cv_matched']:>12.3f}{scores['lodo']:>10.3f}{scores['transfer_ratio']:>9.2f}")
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
