"""How far outside the database can a confinement model be trusted?

Run ``python3 analysis_size_extrapolation.py`` to regenerate everything under
``results/`` for Result 5: the size sweep, the ITER-size-matched cut, the per-machine
breakdown, the aspect-ratio control, and the figure.

Result 4 holds out one tokamak at a time. That is the right question for "will
this transfer to a machine I have not seen", but it is not the question a
next-step device poses, because holding out JET still leaves twelve machines
spanning much of its parameter range. The model interpolates in size even while
extrapolating in identity. ``results/RESULTS.md`` lists this as a limitation and
Result 5 is the attempt to remove it.

The device the field actually cares about sits *beyond* the database in size:
ITER's major radius is 6.2 m against 3.40 m for the largest row here, a factor
of 1.82. So order the machines by size, cut, train below the cut and predict
above it. Sweeping the cut sweeps the size ratio demanded, and one rung of the
sweep reproduces the ITER jump to within 0.1%:

    Result 5a  At a size extrapolation the same size as the one separating this
               database from ITER, the tree ensembles score near the mean
               baseline and the power law degrades gracefully.
    Result 5b  The gap is not an artifact of plasma shape. Dropping the
               spherical tokamaks, which are small and would otherwise sit in
               every training set, moves nothing.
    Result 5c  The mechanism is the Result 4c bound, now binding on a third of
               the held-out rows rather than a handful.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hdb5
from figures import (
    FONT_ANNOTATION,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_SMALL,
    FONT_TICK,
    FONT_TITLE,
    PAPER_WIDTH_IN,
    model_style,
    save_figure,
)
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The models plotted and discussed. Kept in the order they appear in the
# narrative rather than by score, so the figure legend reads the same way the
# prose does.
REPORTED_MODELS = (
    "ipb98y2_analytic",
    "ridge_loglinear",
    "hist_gradient_boosting",
    "random_forest",
    "mean_baseline",
)

# A training set this small stops being a test of extrapolation and starts being
# a test of sample size: the smallest cuts here train on fewer than 500 rows, so
# a model failing there has two possible explanations and the split cannot tell
# them apart. Cuts below this are still computed and plotted, but they are
# marked underpowered and no claim rests on them. The ITER-size-matched cut trains on
# 3498 rows and is comfortably above it.
MIN_WELL_POWERED_TRAIN_ROWS = 1000


@dataclass(frozen=True)
class SizeCutScore:
    """One model at one size cut, scored on every row above the cut."""

    model_name: str
    n_train_machines: int
    size_ratio: float
    n_train_rows: int
    n_test_rows: int
    rmsle: float
    r2_log: float
    is_blind: bool
    well_powered: bool

    def to_json(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "n_train_machines": self.n_train_machines,
            "size_ratio": self.size_ratio,
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "rmsle": self.rmsle,
            "r2_log": self.r2_log,
            "is_blind": self.is_blind,
            "well_powered": self.well_powered,
        }


@dataclass(frozen=True)
class EscalationRow:
    """One model across the three splits, in increasing order of difficulty.

    The three columns are the same models and the same features throughout. Only
    the question changes: predict a held-out *shot*, then a held-out *machine*,
    then a machine larger than anything in training. Reading across the row is
    reading how much of a cross-validated score was measuring the easy question.
    """

    model_name: str
    is_blind: bool
    cv_rmsle: float
    lomo_mean_rmsle: float
    size_cut_rmsle: float
    # size_cut / cv. How much of the cross-validated number survives the jump.
    degradation_factor: float
    # Where the model lands between the mean baseline (0) and the analytic power
    # law (1) on the size cut. Negative means worse than predicting a constant.
    skill_against_baseline: float

    def to_json(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "is_blind": self.is_blind,
            "cv_rmsle": self.cv_rmsle,
            "lomo_mean_rmsle": self.lomo_mean_rmsle,
            "size_cut_rmsle": self.size_cut_rmsle,
            "degradation_factor": self.degradation_factor,
            "skill_against_baseline": self.skill_against_baseline,
        }


@dataclass(frozen=True)
class SizeExtrapolationAnalysis:
    feature_columns: list[str]
    n_rows: int
    # ``None`` when the caller passed a frame it built itself rather than a
    # ``dataset_path``; see ``analysis_extrapolation.ExtrapolationAnalysis``.
    provenance: dict[str, Any] | None
    # Ratio of ITER's major radius to the largest in the database.
    iter_size_ratio: float
    iter_matched_split: dict[str, Any]
    # log-distance between the matched cut's ratio and the ITER ratio. Small
    # means the rung really is the ITER analogue rather than the nearest thing.
    iter_match_log_error: float
    sweep: list[SizeCutScore]
    escalation: list[EscalationRow]
    per_machine: dict[str, dict[str, float]]
    control_conventional_aspect_ratio: dict[str, Any]
    truncation: dict[str, Any]
    machine_sizes: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "feature_columns": self.feature_columns,
            "n_rows": self.n_rows,
            "provenance": self.provenance,
            "iter_major_radius_m": hdb5.ITER_MAJOR_RADIUS_M,
            "iter_size_ratio": self.iter_size_ratio,
            "iter_matched_split": self.iter_matched_split,
            "iter_match_log_error": self.iter_match_log_error,
            "min_well_powered_train_rows": MIN_WELL_POWERED_TRAIN_ROWS,
            "sweep": [score.to_json() for score in self.sweep],
            "escalation": [row.to_json() for row in self.escalation],
            "per_machine": self.per_machine,
            "control_conventional_aspect_ratio": self.control_conventional_aspect_ratio,
            "truncation": self.truncation,
            "machine_sizes": self.machine_sizes,
        }


def _pooled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["scope"] == "__pooled__"]


def _rmsle_by_model(frame: pd.DataFrame, scope: str = "__pooled__") -> dict[str, float]:
    subset = frame[frame["scope"] == scope]
    names = subset["model_name"].to_numpy()
    values = subset["rmsle"].to_numpy(dtype=float)
    return {str(name): float(value) for name, value in zip(names, values, strict=True)}


def build_sweep(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
) -> tuple[list[SizeCutScore], list[hdb5.SizeSplit]]:
    """Score every model at every usable size cut."""
    scores, splits = hdb5.size_extrapolation_report(dataset, feature_columns=feature_columns)
    by_cut = {split.n_train_machines: split for split in splits}
    pooled = _pooled(scores)
    cuts = pooled["n_train_machines"].to_numpy(dtype=int)
    rows = [
        SizeCutScore(
            model_name=str(name),
            n_train_machines=int(cut),
            size_ratio=float(ratio),
            n_train_rows=by_cut[int(cut)].n_train_rows,
            n_test_rows=int(held_out),
            rmsle=float(rmsle),
            r2_log=float(r2),
            is_blind=bool(blind),
            well_powered=by_cut[int(cut)].n_train_rows >= MIN_WELL_POWERED_TRAIN_ROWS,
        )
        for name, cut, ratio, held_out, rmsle, r2, blind in zip(
            pooled["model_name"].to_numpy(),
            cuts,
            pooled["size_ratio"].to_numpy(dtype=float),
            pooled["n_held_out_rows"].to_numpy(dtype=int),
            pooled["rmsle"].to_numpy(dtype=float),
            pooled["r2_log"].to_numpy(dtype=float),
            pooled["is_blind"].to_numpy(dtype=bool),
            strict=True,
                                                            )
    ]
    rows.sort(key=lambda score: (score.n_train_machines, score.model_name))
    return rows, splits


def build_escalation(
    dataset: pd.DataFrame,
    matched_scores: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    n_splits: int = hdb5.N_CV_FOLDS,
) -> list[EscalationRow]:
    """The same models under all three splits, easiest question to hardest.

    Recomputed here rather than read from ``results/extrapolation.json`` so that
    this script stands alone and so the three columns cannot silently drift onto
    different feature sets, which would make them incomparable.
    """
    cv = {
        score.model_name: score.cv_rmsle
        for score in hdb5.evaluate_models(dataset, n_splits=n_splits, feature_columns=feature_columns)
    }
    lomo = (
        hdb5.leave_one_tokamak_out(dataset, feature_columns=feature_columns)
        .groupby("model_name")["rmsle"]
        .mean()
    )
    size_cut = _rmsle_by_model(matched_scores)

    baseline = size_cut.get("mean_baseline", float("nan"))
    reference = size_cut.get("ipb98y2_analytic", float("nan"))
    span = baseline - reference

    rows: list[EscalationRow] = []
    for name in REPORTED_MODELS:
        if name not in size_cut or name not in cv or name not in lomo.index:
            continue
        rows.append(
            EscalationRow(
                model_name=name,
                is_blind=name != "ipb98y2_analytic",
                cv_rmsle=float(cv[name]),
                lomo_mean_rmsle=float(lomo[name]),
                size_cut_rmsle=float(size_cut[name]),
                degradation_factor=float(size_cut[name] / cv[name]),
                skill_against_baseline=(
                    float((baseline - size_cut[name]) / span) if span > 0 else float("nan")
                ),
            )
        )
    return rows


def build_control(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Rerun the matched cut with the spherical tokamaks removed.

    The obvious objection to Result 5a is that the small machines are not merely
    small: START, MAST and NSTX are spherical, at inverse aspect ratios near 0.7
    against a conventional 0.3, and being small they land in the training set of
    every cut. A critic can fairly say the models are being asked to extrapolate
    in shape as much as in size. Dropping them tests exactly that. If the gap is
    a shape artifact it should shrink; if it is about size it should not move.
    """
    scores, splits = hdb5.size_extrapolation_report(
        dataset, feature_columns=feature_columns, conventional_aspect_ratio_only=True
    )
    matched = hdb5.iter_matched_split(dataset, splits)
    at_cut = _pooled(scores)
    at_cut = at_cut[at_cut["n_train_machines"] == matched.n_train_machines]
    return {
        "excluded_machines": sorted(
            size.tokamak
            for size in hdb5.machine_sizes(dataset)
            if size.inverse_aspect_ratio_median > hdb5.MAX_CONVENTIONAL_INVERSE_ASPECT_RATIO
        ),
        "max_inverse_aspect_ratio": hdb5.MAX_CONVENTIONAL_INVERSE_ASPECT_RATIO,
        "iter_matched_split": matched.to_json(),
        "rmsle": _rmsle_by_model(at_cut),
    }


def build_truncation(dataset: pd.DataFrame, split: hdb5.SizeSplit) -> dict[str, Any]:
    """How hard the Result 4c bound bites at the ITER-size-matched cut.

    A tree ensemble averages training targets, so no prediction can leave
    ``[min(y_train), max(y_train)]``. Under leave-one-tokamak-out that bound
    binds on one machine. Here it binds on a third of every held-out row at once,
    which is why the trees land near a constant predictor rather than merely
    behind the power law.
    """
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    train_mask = np.isin(labels, list(split.train_machines))
    test_tau = tau[~train_mask]
    train_max = float(tau[train_mask].max())
    return {
        "train_tau_max_s": train_max,
        "test_tau_max_s": float(test_tau.max()),
        "headroom_ratio": float(test_tau.max() / train_max),
        "fraction_above_train_max": float(np.mean(test_tau > train_max)),
        "n_rows_above_train_max": int(np.sum(test_tau > train_max)),
        "n_test_rows": int(test_tau.size),
    }


def analyze_size_extrapolation(
    dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS,
    dataset_path: Path | str | None = None,
) -> SizeExtrapolationAnalysis:
    """The full Result 5 analysis: sweep, matched cut, escalation, control."""
    sweep, splits = build_sweep(dataset, feature_columns=feature_columns)
    matched = hdb5.iter_matched_split(dataset, splits)
    matched_scores = hdb5.score_size_split(
        dataset, matched, feature_columns=feature_columns, per_machine=True
    )

    per_machine = {
        str(machine): _rmsle_by_model(matched_scores, scope=str(machine))
        for machine in matched_scores["scope"].unique()
        if machine != "__pooled__"
    }

    iter_ratio = hdb5.iter_size_ratio(dataset)
    return SizeExtrapolationAnalysis(
        feature_columns=list(feature_columns),
        n_rows=int(len(dataset)),
        provenance=hdb5.dataset_provenance(dataset_path) if dataset_path is not None else None,
        iter_size_ratio=iter_ratio,
        iter_matched_split=matched.to_json(),
        iter_match_log_error=float(abs(np.log(matched.size_ratio) - np.log(iter_ratio))),
        sweep=sweep,
        escalation=build_escalation(dataset, matched_scores, feature_columns=feature_columns),
        per_machine=per_machine,
        control_conventional_aspect_ratio=build_control(dataset, feature_columns=feature_columns),
        truncation=build_truncation(dataset, matched),
        machine_sizes=[
            {
                "tokamak": size.tokamak,
                "n_rows": size.n_rows,
                "r_median_m": size.r_median_m,
                "r_max_m": size.r_max_m,
                "inverse_aspect_ratio_median": size.inverse_aspect_ratio_median,
            }
            for size in hdb5.machine_sizes(dataset)
        ],
    )


def plot_size_extrapolation(analysis: SizeExtrapolationAnalysis) -> Path | None:
    """Two panels: the escalation, and the sweep it sits on.

    Left is Result 5a: the same models under three questions of increasing
    difficulty, so the collapse is visible as one line falling away from another.
    Right is the sweep the matched cut is one point of, which shows the result is
    not a single lucky cut.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    blue, orange, green = "#2a78d6", "#eb6834", "#3f8f5c"
    ink, muted = "#0b0b0b", "#52514e"
    style = {
        "random_forest": (orange, "random forest"),
        "hist_gradient_boosting": ("#c8873a", "hist gradient boosting"),
        "ridge_loglinear": (blue, "ridge, log-linear"),
        "ipb98y2_analytic": (green, "IPB98(y,2), analytic"),
        "mean_baseline": (muted, "mean baseline"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(PAPER_WIDTH_IN, 6.4), gridspec_kw={"height_ratios": [1.0, 1.15]})
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(muted)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=muted, labelsize=FONT_TICK)

    # --- Left: three questions, increasing in difficulty ------------------
    for row in analysis.escalation:
        if row.model_name not in style:
            continue
        color, label = style[row.model_name]
        dashed = row.model_name == "mean_baseline"
        marker, line = model_style(row.model_name)
        axes[0].plot(
            [0, 1, 2],
            [row.cv_rmsle, row.lomo_mean_rmsle, row.size_cut_rmsle],
            marker=marker,
            linestyle=line,
            color=color,
            linewidth=1.4 if dashed else 2.0,
            markersize=5 if dashed else 7,
            alpha=0.65 if dashed else 1.0,
            label=label,
        )
        axes[0].annotate(
            f"{row.size_cut_rmsle:.3f}",
            xy=(2, row.size_cut_rmsle),
            xytext=(9, -3),
            textcoords="offset points",
            fontsize=FONT_ANNOTATION,
            color=color,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlim(-0.3, 2.62)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(
        [
            "held-out\nshot",
            "held-out\nmachine",
            "machine larger than\nanything in training",
        ],
        fontsize=FONT_LABEL,
        color=ink,
    )
    axes[0].set_ylabel("log-RMSE, log scale (lower is better)", fontsize=FONT_LABEL, color=muted)
    axes[0].set_title(
        "Three questions of increasing difficulty",
        fontsize=FONT_TITLE,
        color=ink,
    )
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none",
        framealpha=0.82, fontsize=FONT_LEGEND, loc="upper left", labelcolor=muted)

    # --- Right: the sweep, with the ITER-size-matched rung marked --------------
    #
    # Only the well-powered cuts are joined into a line. Beyond them the
    # training set falls under MIN_WELL_POWERED_TRAIN_ROWS rows, so a model
    # failing there could be failing on sample size rather than on size
    # extrapolation, and the two cannot be separated. Those cuts are still
    # plotted, as unconnected faded markers, because hiding them would be the
    # dishonest option; they are just not joined into a trend the eye will read
    # as a claim.
    frame = pd.DataFrame([score.to_json() for score in analysis.sweep])
    matched_ratio = float(analysis.iter_matched_split["size_ratio"])
    underpowered = frame[~frame["well_powered"]]["size_ratio"]
    if len(underpowered):
        axes[1].axvspan(
            float(underpowered.min()) - 0.05,
            float(underpowered.max()) + 0.15,
            color="#b9b6ad",
            alpha=0.16,
            linewidth=0,
        )

    for name, (color, label) in style.items():
        subset = frame[frame["model_name"] == name].sort_values("size_ratio")
        if subset.empty:
            continue
        dashed = name == "mean_baseline"
        powered = subset[subset["well_powered"]]
        marker, line = model_style(name)
        axes[1].plot(
            powered["size_ratio"],
            powered["rmsle"],
            marker=marker,
            linestyle=line,
            color=color,
            linewidth=1.3 if dashed else 1.9,
            markersize=4 if dashed else 5.5,
            alpha=0.65 if dashed else 1.0,
            label=label,
        )
        thin = subset[~subset["well_powered"]]
        axes[1].plot(
            thin["size_ratio"],
            thin["rmsle"],
            marker=marker,
            linestyle="none",
            color=color,
            markersize=4,
            alpha=0.3,
        )

    axes[1].axvline(matched_ratio, color=ink, linewidth=1.0, linestyle=":", alpha=0.8)
    axes[1].set_yscale("log")
    top = axes[1].get_ylim()[1]
    axes[1].annotate(
        f"the ITER jump: {matched_ratio:.2f}x\n(ITER / this database = {analysis.iter_size_ratio:.2f}x)",
        xy=(matched_ratio, top),
        xytext=(6, -11),
        textcoords="offset points",
        fontsize=FONT_ANNOTATION,
        color=ink,
        ha="left",
        va="top",
    )
    band_centre = float(underpowered.min() + underpowered.max()) / 2 if len(underpowered) else None
    if band_centre is not None:
        axes[1].annotate(
            f"fewer than {MIN_WELL_POWERED_TRAIN_ROWS} training rows: size\n"
            "extrapolation and sample size are confounded.\nPlotted but not joined; no claim rests here.",
            xy=(band_centre, axes[1].get_ylim()[0]),
            xytext=(0, 14),
            textcoords="offset points",
            fontsize=FONT_SMALL,
            color=muted,
            ha="center",
            va="bottom",
        )
    axes[1].set_xlabel(
        "size extrapolation demanded\n(largest major radius asked about / largest one trained on)",
        fontsize=FONT_LABEL,
        color=muted,
    )
    axes[1].set_ylabel("log-RMSE on every machine above the cut", fontsize=FONT_LABEL, color=muted)
    axes[1].set_title(
        "The size-ordered sweep",
        fontsize=FONT_TITLE,
        color=ink,
    )
    # The well-powered lines all bunch together at the left edge of this panel,
    # so direct end-labels overlap; the legend goes in the empty mid-band of the
    # underpowered region instead.
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none",
        framealpha=0.82, fontsize=FONT_SMALL, loc="center right", labelcolor=muted)

    figure.tight_layout()
    path = RESULTS_DIR / "size_extrapolation.png"
    save_figure(figure, path, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    analysis = analyze_size_extrapolation(dataset, dataset_path=hdb5.default_hdb5_path())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(
        RESULTS_DIR / "size_extrapolation_sweep.csv",
        pd.DataFrame([score.to_json() for score in analysis.sweep]),
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "size_extrapolation_escalation.csv",
        pd.DataFrame([row.to_json() for row in analysis.escalation]),
    )
    write_json_strict(RESULTS_DIR / "size_extrapolation.json", analysis.to_json())
    figure_path = plot_size_extrapolation(analysis)

    split = analysis.iter_matched_split
    print("--- Result 5: extrapolating in the direction ITER actually sits ---")
    print(
        f"ITER is {hdb5.ITER_MAJOR_RADIUS_M} m of major radius against "
        f"{split['test_r_max_m']} m for the largest row here: a factor of "
        f"{analysis.iter_size_ratio:.3f}"
    )
    print(
        f"the matched cut trains on the {split['n_train_machines']} smallest machines "
        f"({split['n_train_rows']} rows, up to R = {split['train_r_max_m']} m) and predicts "
        f"{len(split['test_machines'])} larger ones ({split['n_test_rows']} rows, up to "
        f"R = {split['test_r_max_m']} m)"
    )
    print(
        f"that is a factor of {split['size_ratio']:.3f}, matching the ITER jump to "
        f"{analysis.iter_match_log_error * 100:.1f}% in log terms\n"
    )

    header = f"  {'model':<24}{'shot':>8}{'machine':>9}{'larger':>9}{'ratio':>8}{'skill':>8}"
    print(header)
    for row in analysis.escalation:
        marker = " " if row.is_blind else "*"
        print(
            f"{marker} {row.model_name:<24}{row.cv_rmsle:>8.3f}{row.lomo_mean_rmsle:>9.3f}"
            f"{row.size_cut_rmsle:>9.3f}{row.degradation_factor:>8.2f}"
            f"{row.skill_against_baseline:>8.2f}"
        )
    print("  * fitted on this database including the held-out machines; a reference, not a competitor")
    print("  skill: 1.0 matches the analytic power law, 0.0 matches predicting a constant")

    print("\n--- per held-out machine at the matched cut ---")
    machines = sorted(analysis.per_machine)
    models = [name for name in REPORTED_MODELS if name != "mean_baseline"]
    print(f"  {'machine':<10}" + "".join(f"{name.split('_')[0]:>10}" for name in models))
    for machine in machines:
        scores = analysis.per_machine[machine]
        print(f"  {machine:<10}" + "".join(f"{scores.get(name, float('nan')):>10.3f}" for name in models))

    control = analysis.control_conventional_aspect_ratio
    print("\n--- control: same cut, spherical tokamaks removed ---")
    print(f"  dropped {', '.join(control['excluded_machines'])} (inverse aspect ratio > {control['max_inverse_aspect_ratio']})")
    matched_rmsle = {row.model_name: row.size_cut_rmsle for row in analysis.escalation}
    for name in REPORTED_MODELS:
        if name not in control["rmsle"]:
            continue
        print(
            f"  {name:<24}{matched_rmsle.get(name, float('nan')):>8.3f} -> "
            f"{control['rmsle'][name]:>6.3f}"
        )

    truncation = analysis.truncation
    print("\n--- why: the Result 4c bound, now binding on a third of the held-out rows ---")
    print(
        f"  {truncation['n_rows_above_train_max']} of {truncation['n_test_rows']} held-out rows "
        f"({truncation['fraction_above_train_max'] * 100:.0f}%) sit above max(y_train); the best "
        f"held-out shot is {truncation['headroom_ratio']:.1f}x above anything any tree can output"
    )
    if figure_path:
        print(f"\nfigure: {figure_path}")


if __name__ == "__main__":
    main()
