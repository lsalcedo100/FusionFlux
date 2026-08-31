"""Build the single-page summary at ``site/index.html`` from ``results/``.

Run ``python3 site/build_page.py``. The page is the public-facing version of
``results/RESULTS.md``: the reversal, the ITER-direction escalation, and an
interactive panel where a reader picks the held-out machine and watches the
ranking rearrange.

Every number on it is read out of the generated artifacts rather than typed in,
for the same reason the rest of the repository works that way: a hand-copied
number on a public page is a number that silently stops matching the analysis
the first time the analysis is rerun. Regenerate the results first, then this.

The two figures are inlined as base64 data URIs rather than referenced, so the
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
}
FIGURES = {"__FIG1__": "extrapolation.png", "__FIG2__": "size_extrapolation.png"}


# ``DataFrame.loc`` is typed as returning a union of every dtype pandas can
# hold, so every read below would need its own ignore. These two narrow it once.
def _f(frame: pd.DataFrame, row: str, column: str) -> float:
    return float(cast(Any, frame.loc[row, column]))


def _i(frame: pd.DataFrame, row: str, column: str) -> int:
    return int(cast(Any, frame.loc[row, column]))


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
