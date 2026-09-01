"""Build the single-page summary at ``site/index.html`` from ``results/``.

Run ``python3 site/build_page.py``. The page is the public-facing version of
``results/RESULTS.md``: the reversal, the ITER-direction escalation, an
interactive panel where a reader picks the held-out machine and watches the
ranking rearrange, the dimensional constraint that repairs it, and the locked
forecast for three machines that have no data.

Every number on it is read out of the generated artifacts rather than typed in,
for the same reason the rest of the repository works that way: a hand-copied
number on a public page is a number that silently stops matching the analysis
the first time the analysis is rerun. Regenerate the results first, then this.

The figures are inlined as base64 data URIs rather than referenced, so the
output is one self-contained file that can be dropped on any static host with
no asset paths to keep in sync.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SITE = Path(__file__).resolve().parent

# The four models carried through the narrative, in the order the page's tables
# use. The hybrid and the coverage numbers are read separately.
MODELS = ("ipb98y2_analytic", "ridge_loglinear", "hist_gradient_boosting", "random_forest")
LABELS = {
    "ipb98y2_analytic": "IPB98(y,2)",
    "ridge_loglinear": "ridge (power law)",
    "hist_gradient_boosting": "hist gradient boosting",
    "random_forest": "random forest",
    # Carried in the same map so the later sections can label a model without a
    # second lookup table, but deliberately not in MODELS: the four above are
    # the ones the reversal narrative and the interactive panel step through.
    "powerlaw_free": "power law, unconstrained",
    "powerlaw_kadomtsev": "power law, Kadomtsev",
    "powerlaw_collisionless": "power law, collisionless",
    "powerlaw_electrostatic": "power law, electrostatic",
    "hybrid_gbm_s1": "hybrid (power law + correction)",
    # Result 13 lives in another science; its two power laws are named here
    # so the closing section can reuse the same label lookup.
    "kleiber": "Kleiber, exponent 3/4",
    "ols_loglinear": "power law, free exponent",
}
FIGURES = {
    "__FIG1__": "extrapolation.png",
    "__FIG2__": "size_extrapolation.png",
    "__FIG3__": "dimensional.png",
    "__FIG4__": "conformal_shift.png",
    "__FIG5__": "allometry.png",
}

# Result 8's constraint hierarchy, weakest constraint last. ``powerlaw_free`` is
# the unconstrained fit the other two are scored against; it is the same model
# as ``ridge_loglinear`` at zero penalty, and the sweep below compares against
# it rather than against ridge so that the only difference is the constraint.
CONSTRAINED = ("powerlaw_kadomtsev", "powerlaw_collisionless", "powerlaw_electrostatic")

# Result 10's three calibration schemes, in the order the page steps through
# them: the one that fails, the repair, and the repair plus distance scaling.
SCHEMES = ("split", "machine_cv", "machine_cv_distance")


# ``DataFrame.loc`` is typed as returning a union of every dtype pandas can
# hold, so every read below would need its own ignore. These two narrow it once.
def _f(frame: pd.DataFrame, row: str, column: str) -> float:
    return float(cast(Any, frame.loc[row, column]))


def _i(frame: pd.DataFrame, row: str, column: str) -> int:
    return int(cast(Any, frame.loc[row, column]))


def _dimensional_payload() -> dict[str, Any]:
    """Result 8: what the Connor-Taylor constraints cost and what they buy."""
    raw = json.loads((RESULTS / "dimensional.json").read_text())
    scores = {row["model_name"]: row for row in raw["split_scores"]}

    # The external check that the derivation is right: IPB98(y,2) was published
    # in 1999 and lands on these surfaces without being told about them.
    published = {
        row["constraint_model"]: row["residual_norm"]
        for row in raw["constraint_distances"]
        if row["exponent_source"] == "ipb98y2_published"
    }

    # The size sweep, counted here rather than copied from the prose. Cuts are
    # keyed by how many machines are in the training half, which is unique per
    # cut; size_ratio is not, since three cuts share a ratio.
    sweep = pd.read_csv(RESULTS / "dimensional_size_sweep.csv")
    sweep = sweep[sweep["scope"] == "__pooled__"]
    rmsle = sweep.pivot_table(index="n_train_machines", columns="model_name", values="rmsle")
    powered = sweep.pivot_table(index="n_train_machines", columns="model_name", values="well_powered")

    def _wins(model: str, *, well_powered_only: bool) -> dict[str, int]:
        mask = powered[model].astype(bool) if well_powered_only else powered[model].notna()
        beat = rmsle[model][mask] < rmsle["powerlaw_free"][mask]
        return {"won": int(beat.sum()), "of": int(mask.sum())}

    return {
        "inSample": {k: round(v, 4) for k, v in raw["in_sample_rmsle"].items()},
        "scores": {
            k: {
                "cv": round(scores[k]["cv_rmsle"], 3),
                "lomo": round(scores[k]["lomo_mean_rmsle"], 3),
                "size": round(scores[k]["size_cut_rmsle"], 3),
            }
            for k in (*CONSTRAINED, "powerlaw_free", "ipb98y2_analytic", "hybrid_gbm_s1")
        },
        "published": {k: round(v, 5) for k, v in published.items()},
        "best": raw["best_blind_at_size_cut"],
        "bestRmsle": round(raw["best_blind_size_cut_rmsle"], 3),
        # Kadomtsev is free, so it is scored at every cut. Collisionless costs
        # something in sample, so it is only claimed where the training half is
        # large enough for the cost to be resolvable.
        "kadomtsevWins": _wins("powerlaw_kadomtsev", well_powered_only=False),
        "collisionlessWins": _wins("powerlaw_collisionless", well_powered_only=True),
    }


def _shift_payload() -> dict[str, Any]:
    """Result 10: the interval repair, and the cut where it stops working."""
    raw = json.loads((RESULTS / "conformal_shift.json").read_text())
    repairs = {(r["model_name"], r["method"]): r for r in raw["repairs"]}
    models = sorted({m for m, _ in repairs})
    return {
        "nominal": raw["nominal_coverage"],
        "schemes": list(SCHEMES),
        "coverage": {
            m: {
                s: {
                    "lomo": round(repairs[(m, s)]["lomo_coverage"], 3),
                    "size": round(repairs[(m, s)]["size_cut_coverage"], 3),
                }
                for s in SCHEMES
                if (m, s) in repairs
            }
            for m in models
        },
    }


def _replication_payload() -> list[dict[str, Any]]:
    """Result 11: the same reversal on rows STD5 does not contain."""
    raw = json.loads((RESULTS / "replication.json").read_text())
    return [
        {
            "arm": arm["arm"],
            "baseline": arm["baseline_label"],
            "rows": arm["n_rows"],
            "machines": arm["n_machines_scored"],
            "shared": arm["n_rows_shared_with_std5"],
            "bestCv": arm["best_cv_model"],
            "bestLomo": arm["best_lomo_model"],
            "reversed": arm["ranking_reversed"],
            "gain": round(arm["cv_gain_over_baseline"], 3),
            "treeLosses": arm["n_machines_trees_lose_to_baseline"],
        }
        for arm in raw["arms"]
    ]


def _forecast_payload() -> dict[str, Any]:
    """Result 12: the locked prediction for three machines with no data."""
    raw = json.loads((RESULTS / "forecast.json").read_text())
    devices = [
        {
            "name": d["name"],
            "r": d["r_m"],
            "status": d["status"],
            "published": d.get("published_ipb98_tau_s"),
        }
        for d in raw["devices"]
    ]
    rows: dict[str, dict[str, Any]] = {}
    for f in raw["forecasts"]:
        rows.setdefault(f["model_name"], {})[f["device"]] = {
            "tau": round(f["tau_predicted_s"], 3),
            "lo": round(f["tau_interval_low_s"], 3),
            "hi": round(f["tau_interval_high_s"], 3),
            "bounded": bool(f["bounded_by_training_range"]),
        }

    # The headline of the whole page, computed rather than asserted: how far
    # apart the models are on the one machine none of them can be checked on.
    at_iter = [v["ITER"]["tau"] for v in rows.values()]
    return {
        "generatedOn": raw["generated_on"],
        "digest": raw["content_digest_sha256"][:12],
        "trainMax": round(raw["train_tau_max_s"], 3),
        "devices": devices,
        "rows": rows,
        "spread": round(max(at_iter) / min(at_iter), 1),
        "maha": {
            d["name"]: round(
                next(f["feature_mahalanobis"] for f in raw["forecasts"] if f["device"] == d["name"]), 1
            )
            for d in raw["devices"]
        },
    }


def _allometry_payload() -> dict[str, Any]:
    """Result 13: the same audit on Kleiber's law, in another science entirely."""
    raw = json.loads((RESULTS / "allometry.json").read_text())
    scores = raw["scores"]
    return {
        # Derived here at full precision rather than in the page from the
        # rounded cells: 0.396 - 0.374 rounds to 0.022 while the underlying
        # difference is 0.023, and the prose quotes the latter.
        "costVsFree": round(
            scores["kleiber"]["cv_rmsle"] - scores["ols_loglinear"]["cv_rmsle"], 3
        ),
        "gainAtCut": round(
            1 - scores["kleiber"]["mass_cut_rmsle"] / scores["ols_loglinear"]["mass_cut_rmsle"], 4
        ),
        "rows": raw["n_rows"],
        "orders": raw["n_orders_scored"],
        "massSpan": round(raw["order_mass_ratio"]),
        "freeExponent": round(raw["free_refit_exponent"], 3),
        "publishedExponent": raw["kleiber_exponent"],
        "treesWinCv": raw["trees_win_cv"],
        "reversed": raw["ranking_reversed"],
        "forestLosses": raw["n_orders_forest_loses_to_kleiber"],
        "nGroups": raw["n_orders"],
        "wins": raw["sweep_wins"],
        "scores": {
            k: {
                "cv": round(v["cv_rmsle"], 3),
                "loo": round(v["loo_mean_rmsle"], 3),
                "cut": round(v["mass_cut_rmsle"], 3),
                "rho": round(v["distance_spearman"], 2),
            }
            for k, v in raw["scores"].items()
        },
    }


def build_payload() -> dict[str, object]:
    """Read every number the page displays out of the generated artifacts."""
    per_machine = pd.read_csv(RESULTS / "extrapolation_per_machine.csv")
    per_machine = per_machine[per_machine["model_name"].isin(MODELS)]
    summary = pd.read_csv(RESULTS / "extrapolation_summary.csv").set_index("model_name")
    escalation = pd.read_csv(RESULTS / "size_extrapolation_escalation.csv").set_index("model_name")
    coverage = pd.read_csv(RESULTS / "conformal_summary.csv").set_index("model_name")
    frontier = pd.read_csv(RESULTS / "hybrid_frontier.csv").set_index("model_name")

    machines: list[dict[str, Any]] = []
    for name, group in per_machine.groupby("tokamak"):
        group = group.set_index("model_name")
        machines.append(
            {
                "name": str(name),
                "rows": int(group["n_held_out_rows"].iloc[0]),
                "dist": round(float(group["feature_mahalanobis"].iloc[0]), 2),
                "above": round(float(group["target_above_train_max_fraction"].iloc[0]), 3),
                "rmsle": {k: round(_f(group, k, "rmsle"), 3) for k in MODELS},
            }
        )
    # Ordered by how far the machine sits outside the training data, which is
    # the axis Result 4b shows the tree errors track.
    machines.sort(key=lambda m: float(m["dist"]))

    return {
        "labels": LABELS,
        "models": list(MODELS),
        "cv": {k: round(_f(summary, k, "cv_rmsle"), 3) for k in MODELS},
        "lomoMean": {k: round(_f(summary, k, "lomo_mean_rmsle"), 3) for k in MODELS},
        "cvRank": {k: _i(summary, k, "cv_rank") for k in MODELS},
        "lomoRank": {k: _i(summary, k, "lomo_rank") for k in MODELS},
        "rho": {k: round(_f(summary, k, "distance_spearman"), 2) for k in MODELS},
        "sizeCut": {
            k: round(_f(escalation, k, "size_cut_rmsle"), 3)
            for k in (*MODELS, "mean_baseline")
        },
        "machines": machines,
        "coverage": {
            str(k): {
                "cv": round(_f(coverage, str(k), "cv_coverage"), 3),
                "lomo": round(_f(coverage, str(k), "lomo_coverage"), 3),
                "size": round(_f(coverage, str(k), "size_cut_coverage"), 3),
            }
            for k in coverage.index
        },
        "hybrid": {
            "cv": round(_f(frontier, "hybrid_gbm_s1", "cv_rmsle"), 3),
            "lomo": round(_f(frontier, "hybrid_gbm_s1", "lomo_mean_rmsle"), 3),
            "size": round(_f(frontier, "hybrid_gbm_s1", "size_cut_rmsle"), 3),
        },
        "dimensional": _dimensional_payload(),
        "shift": _shift_payload(),
        "replication": _replication_payload(),
        "forecast": _forecast_payload(),
        "allometry": _allometry_payload(),
    }


def main() -> None:
    html = (SITE / "page.template.html").read_text()
    html = html.replace("__DATA__", json.dumps(build_payload(), separators=(",", ":")))
    for token, filename in FIGURES.items():
        encoded = base64.b64encode((RESULTS / filename).read_bytes()).decode()
        html = html.replace(token, f"data:image/png;base64,{encoded}")

    for token in ("__DATA__", *FIGURES):
        if token in html:
            raise RuntimeError(f"Placeholder {token} was not substituted.")

    out = SITE / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
