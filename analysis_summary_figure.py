"""Build ``results/summary.png``: the whole argument as one image.

There are eleven figures under ``results/`` and none of them is the one to open
first. Each answers a question that only makes sense once you already know what
the study found. This builds the missing one: two panels that state the finding
and its consequence, readable with no prior context.

Left, the reversal. Every model's score under grouped cross-validation joined to
its score on a machine it has never seen. The lines cross, which is the result:
the ordering is not merely compressed by the harder split, it is inverted.

Right, what that costs. The five models' predictions for ITER, against the
ceiling a tree ensemble cannot predict above, because it predicts by averaging
training targets. Two of the five are pinned under that ceiling and the physics
answer is far above it.

Unlike the other analysis scripts this one reads only the committed artifacts
under ``results/`` and never touches the raw database, so it runs in a checkout
with no HDB5 download. It is last in ``make results`` for that reason: it
summarises what the others have already written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Matches analysis_extrapolation.py, so the summary does not recolour models the
# reader has already met elsewhere.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f8f5c"
INK, MUTED = "#0b0b0b", "#52514e"
STYLE: dict[str, tuple[str, str]] = {
    "ipb98y2_analytic": (GREEN, "IPB98(y,2), analytic"),
    "powerlaw_collisionless": ("#7a5cc0", "power law, collisionless"),
    "ridge_loglinear": (BLUE, "ridge, log-linear"),
    "hist_gradient_boosting": ("#c8873a", "hist gradient boosting"),
    "random_forest": (ORANGE, "random forest"),
}


def _read() -> tuple[pd.DataFrame, dict]:
    summary = pd.read_csv(RESULTS_DIR / "extrapolation_summary.csv").set_index("model_name")
    forecast = json.loads((RESULTS_DIR / "forecast.json").read_text())
    return summary, forecast


def plot_summary() -> Path | None:
    """Write the two-panel summary; ``None`` if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    summary, forecast = _read()
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
    for axis in axes:
        axis.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#d8d6d1")
        axis.tick_params(colors=MUTED, labelsize=9)

    # --- Left: the ranking inversion, as a slope chart ---------------------
    reversal = [m for m in STYLE if m in summary.index and m != "powerlaw_collisionless"]

    # The two tree ensembles sit 0.002 apart under CV, so their value labels
    # print on top of each other at a shared offset. Nudge apart any labels
    # closer together than this fraction of the axis range.
    cv_values = {m: float(summary.loc[m, "cv_rmsle"]) for m in reversal}
    span = max(cv_values.values()) - min(cv_values.values())
    cv_offsets: dict[str, float] = {}
    for model in sorted(cv_values, key=lambda m: cv_values[m]):
        below = [m for m in cv_offsets if abs(cv_values[m] - cv_values[model]) < 0.06 * span]
        cv_offsets[model] = -3.0 + (11.0 * len(below) if below else 0.0)

    for model in reversal:
        colour, label = STYLE[model]
        cv = float(summary.loc[model, "cv_rmsle"])
        lomo = float(summary.loc[model, "lomo_mean_rmsle"])
        # The analytic law is drawn dashed throughout: its exponents were fitted
        # on this database including the held-out machine, so showing it as a
        # peer of the blind models would credit it with a fairness it does not
        # have.
        blind = model != "ipb98y2_analytic"
        axes[0].plot(
            [0, 1],
            [cv, lomo],
            "o-" if blind else "o--",
            color=colour,
            linewidth=2.4 if blind else 1.6,
            markersize=8,
            alpha=1.0 if blind else 0.8,
            label=label + ("" if blind else "  (not blind)"),
        )
        axes[0].annotate(
            f"{cv:.3f}", xy=(0, cv), xytext=(-46, cv_offsets[model]),
            textcoords="offset points", fontsize=9.5, color=colour,
        )
        axes[0].annotate(
            f"{lomo:.3f}", xy=(1, lomo), xytext=(11, -3), textcoords="offset points",
            fontsize=9.5, color=colour, fontweight="bold" if blind else "normal",
        )

    axes[0].set_xlim(-0.5, 1.6)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(
        ["grouped CV by discharge\n(a shot it has not seen)",
         "leave-one-tokamak-out\n(a machine it has not seen)"],
        fontsize=10.5, color=INK,
    )
    axes[0].set_ylabel("RMSLE (lower is better)", fontsize=10, color=MUTED)
    axes[0].set_title(
        "The best model under one split is the worst under the other",
        fontsize=12.5, color=INK, pad=14,
    )
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=MUTED)

    # --- Right: what the models say about ITER -----------------------------
    rows = {f["model_name"]: f for f in forecast["forecasts"] if f["device"] == "ITER"}
    ordered = [m for m in STYLE if m in rows]
    ceiling = float(forecast["train_tau_max_s"])

    for position, model in enumerate(ordered):
        colour, label = STYLE[model]
        row = rows[model]
        tau = float(row["tau_predicted_s"])
        axes[1].plot(
            [row["tau_interval_low_s"], row["tau_interval_high_s"]],
            [position, position],
            "-", color=colour, linewidth=3.0, alpha=0.35, solid_capstyle="round",
        )
        axes[1].plot([tau], [position], "o", color=colour, markersize=11)
        axes[1].annotate(
            f"{tau:.3f} s", xy=(tau, position), xytext=(0, 13),
            textcoords="offset points", fontsize=10, color=colour,
            fontweight="bold", ha="center",
        )
        if row["bounded_by_training_range"]:
            axes[1].annotate(
                "cannot exceed the ceiling",
                xy=(tau, position), xytext=(0, -19), textcoords="offset points",
                fontsize=8.5, color=MUTED, ha="center", style="italic",
            )

    axes[1].axvline(ceiling, color=INK, linewidth=1.3, linestyle=":", alpha=0.75)
    axes[1].annotate(
        f"largest confinement time\nanywhere in training: {ceiling:.3f} s",
        xy=(ceiling, -0.62), xytext=(-10, 0), textcoords="offset points",
        fontsize=9, color=INK, ha="right", va="center",
    )

    axes[1].set_yticks(range(len(ordered)))
    axes[1].set_yticklabels([STYLE[m][1] for m in ordered], fontsize=10, color=INK)
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    # A decade-only log axis labels just 10^0 across this range, which tells the
    # reader nothing about where the two clusters sit.
    axes[1].set_xticks([0.2, 0.5, 1.0, 2.0, 5.0])
    axes[1].set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    axes[1].minorticks_off()
    axes[1].set_xlabel("predicted energy confinement time at ITER (s, log scale)",
                       fontsize=10, color=MUTED)
    spread = max(float(r["tau_predicted_s"]) for r in rows.values()) / min(
        float(r["tau_predicted_s"]) for r in rows.values()
    )
    axes[1].set_title(
        f"On a machine 1.82x beyond the data, they disagree by {spread:.1f}x",
        fontsize=12.5, color=INK, pad=14,
    )
    axes[1].set_ylim(len(ordered) - 0.35, -0.75)
    axes[1].annotate(
        "bars: nominal 90% intervals",
        xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8.5, color=MUTED,
    )

    figure.tight_layout()
    path = RESULTS_DIR / "summary.png"
    figure.savefig(path, dpi=180, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def main() -> None:
    path = plot_summary()
    if path is None:  # pragma: no cover - plotting is optional
        print("matplotlib is unavailable; no summary figure written")
        return
    summary, forecast = _read()
    taus = [f["tau_predicted_s"] for f in forecast["forecasts"] if f["device"] == "ITER"]
    print("--- the argument in one figure ---")
    print(f"  best under CV        : {summary['cv_rmsle'].idxmin()}")
    print(f"  best on a new machine: {summary['lomo_mean_rmsle'].idxmin()}")
    print(f"  disagreement at ITER : {max(taus) / min(taus):.1f}x")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
