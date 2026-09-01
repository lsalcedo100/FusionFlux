"""The reversal, reproduced on rows STD5 never contained. Result 11.

Run ``python3 analysis_replication.py`` to regenerate everything under
``results/`` for the replication.

Results 1 to 10 all rest on one file. That is the honest ceiling on every claim
in this repository, and it is stated as the first limitation in
``results/RESULTS.md``: a reader can always say the reversal is a property of
the ITPA standard set STD5 rather than of the problem. ``replication.py``
assembles two populations out of the full DB5.2.3 revision that answer that,
and this script runs Result 4's protocol on them unchanged.

Two arms, two different objections
----------------------------------
``disjoint_h``   5358 H-mode rows STD5 does not contain, over 12 machines, with
                 **zero row overlap**. Same regime, same devices, different
                 rows. This answers "the finding is an artifact of the standard
                 set's selection criteria".

``non_h``        3860 ohmic, L-mode and radiative-improved rows over 5 machines,
                 scored against ITER89-P rather than IPB98(y,2) because an
                 H-mode law is the wrong baseline for L-mode plasmas. This
                 answers "the finding is a property of ELMy H-mode, or of
                 IPB98(y,2) specifically". Five machines is too few to carry a
                 claim on its own and every number from this arm is reported
                 with that attached.

What replication means here
---------------------------
Not the numbers. The row populations differ, so the RMSLE values differ and
should. What is being replicated is the *structure*: that grouped CV by
discharge ranks the trees far above the log-linear power law, that holding out
an entire machine reverses that ranking, and that the degradation ratio between
the two splits separates the model families in the same order.

The script therefore reports the reversal as a rank comparison and a count of
machines, not as a difference of RMSLE against STD5's numbers, which would be
comparing two different populations and meaning nothing.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import dimensional as dm
import hdb5
import replication as rp
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The four models the reversal is stated over, plus the constrained power law
# from Result 8 so that result gets an independent check too. Deliberately the
# same set and the same names as Result 4 uses.
REPLICATION_MODELS: tuple[str, ...] = (
    "random_forest",
    "hist_gradient_boosting",
    "ridge_loglinear",
    "powerlaw_collisionless",
)

MODEL_LABELS: dict[str, str] = {
    "random_forest": "random forest",
    "hist_gradient_boosting": "hist gradient boosting",
    "ridge_loglinear": "ridge, log-linear",
    "powerlaw_collisionless": "power law, collisionless",
}


@dataclass(frozen=True)
class ArmModelScore:
    """One model on one replication arm, under both splits."""

    arm: str
    model_name: str
    is_blind: bool
    cv_rmsle: float
    lomo_mean_rmsle: float
    lomo_median_rmsle: float
    lomo_worst_rmsle: float
    # lomo_mean / cv. The quantity Result 4 reports as the transfer failure.
    degradation_factor: float
    cv_rank: int
    lomo_rank: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArmReplication:
    """One arm's verdict on whether Result 4's structure reproduced."""

    arm: str
    baseline_label: str
    n_rows: int
    n_discharges: int
    n_machines_scored: int
    n_rows_shared_with_std5: int
    r_min_m: float
    r_max_m: float
    # The headline comparison, stated exactly as Result 4 states it.
    best_cv_model: str
    best_lomo_model: str
    ranking_reversed: bool
    # How much the best cross-validated model improves on the published law
    # under CV, and how many machines it loses to that law on under LOMO.
    cv_gain_over_baseline: float
    n_machines_trees_lose_to_baseline: int
    n_machines_trees_lose_to_ridge: int
    n_machines: int
    scores: list[ArmModelScore] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scores"] = [row.to_json() for row in self.scores]
        return payload


def _baseline_per_machine(
    dataset: pd.DataFrame, baseline_column: str, label: str, min_rows: int
) -> pd.DataFrame:
    """Per-machine RMSLE of the analytic baseline, in leave-one-out's shape.

    The analytic law needs no fitting, so "holding out a machine" does not
    change it. Its per-machine score is simply its error on that machine, which
    is what ``hdb5.leave_one_tokamak_out`` also does for ``ipb98y2_analytic``.
    It is rebuilt here rather than reused because the non-H arm's baseline is
    ITER89-P, which that function does not know about.
    """
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    predicted = dataset[baseline_column].to_numpy(dtype=float)
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    rows = []
    for machine in hdb5.eligible_tokamaks(dataset, min_rows=min_rows):
        mask = labels == machine
        rows.append(
            {
                "model_name": label,
                "tokamak": machine,
                "n_held_out_rows": int(mask.sum()),
                "rmsle": float(np.sqrt(np.mean((np.log(tau[mask]) - np.log(predicted[mask])) ** 2))),
                "is_blind": False,
            }
        )
    return pd.DataFrame(rows)


def analyze_arm(
    arm: rp.ReplicationArm,
    *,
    extra_models: dict[str, Any],
    min_rows: int = hdb5.MIN_HELD_OUT_ROWS,
) -> tuple[ArmReplication, pd.DataFrame]:
    """Run Result 4's two splits on one replication arm."""
    dataset = arm.dataset
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    baseline = dataset[arm.baseline_column].to_numpy(dtype=float)

    cv_scores = {
        score.model_name: score.cv_rmsle
        for score in hdb5.evaluate_models(
            dataset,
            feature_columns=hdb5.BLIND_FEATURE_COLUMNS,
            extra_models=extra_models,
        )
    }
    cv_scores[arm.baseline_label] = float(
        np.sqrt(np.mean((np.log(tau) - np.log(baseline)) ** 2))
    )

    # ``include_ipb98_reference=False`` because the non-H arm's baseline is not
    # IPB98(y,2); the correct one for each arm is appended below instead.
    per_machine = hdb5.leave_one_tokamak_out(
        dataset, extra_models=extra_models, include_ipb98_reference=False, min_rows=min_rows
    )
    per_machine = pd.concat(
        [
            per_machine,
            _baseline_per_machine(dataset, arm.baseline_column, arm.baseline_label, min_rows),
        ],
        ignore_index=True,
    )

    considered = [*REPLICATION_MODELS, arm.baseline_label]
    summary = (
        per_machine[per_machine["model_name"].isin(considered)]
        .groupby("model_name")
        .agg(
            lomo_mean=("rmsle", "mean"),
            lomo_median=("rmsle", "median"),
            lomo_worst=("rmsle", "max"),
            is_blind=("is_blind", "first"),
        )
    )
    cv_order = sorted(considered, key=lambda name: cv_scores[name])
    lomo_order = list(summary["lomo_mean"].sort_values().index)

    scores = [
        ArmModelScore(
            arm=arm.name,
            model_name=name,
            is_blind=bool(summary.loc[name, "is_blind"]),
            cv_rmsle=cv_scores[name],
            lomo_mean_rmsle=float(summary.loc[name, "lomo_mean"]),
            lomo_median_rmsle=float(summary.loc[name, "lomo_median"]),
            lomo_worst_rmsle=float(summary.loc[name, "lomo_worst"]),
            degradation_factor=float(summary.loc[name, "lomo_mean"]) / cv_scores[name],
            cv_rank=cv_order.index(name) + 1,
            lomo_rank=lomo_order.index(name) + 1,
        )
        for name in considered
    ]
    scores.sort(key=lambda score: score.cv_rmsle)

    table = per_machine.pivot_table(index="tokamak", columns="model_name", values="rmsle")
    trees = ["random_forest", "hist_gradient_boosting"]
    lose_to_baseline = int(
        sum((table[tree] > table[arm.baseline_label]).sum() for tree in trees)
    )
    lose_to_ridge = int(sum((table[tree] > table["ridge_loglinear"]).sum() for tree in trees))

    best_cv = cv_order[0]
    best_lomo = lomo_order[0]
    return (
        ArmReplication(
            arm=arm.name,
            baseline_label=arm.baseline_label,
            n_rows=arm.n_rows,
            n_discharges=arm.n_discharges,
            n_machines_scored=arm.n_machines_scored,
            n_rows_shared_with_std5=arm.n_rows_shared_with_std5,
            r_min_m=arm.r_min_m,
            r_max_m=arm.r_max_m,
            best_cv_model=best_cv,
            best_lomo_model=best_lomo,
            # The structure being replicated: the model that wins under CV is
            # not the model that wins when a whole machine is held out.
            ranking_reversed=bool(best_cv != best_lomo),
            cv_gain_over_baseline=float(
                1.0 - cv_scores[best_cv] / cv_scores[arm.baseline_label]
            ),
            n_machines_trees_lose_to_baseline=lose_to_baseline,
            n_machines_trees_lose_to_ridge=lose_to_ridge,
            n_machines=int(len(table) * len(trees)),
            scores=scores,
        ),
        per_machine.assign(arm=arm.name),
    )


@dataclass(frozen=True)
class ReplicationAnalysis:
    arms: list[ArmReplication]
    per_machine: pd.DataFrame = field(repr=False)

    def to_json(self) -> dict[str, object]:
        return {
            "arms": [arm.to_json() for arm in self.arms],
            "reversal_reproduced_in_all_arms": all(arm.ranking_reversed for arm in self.arms),
        }


def analyze_replication() -> ReplicationAnalysis:
    extra = dm.build_constrained_models(("collisionless",))
    arms = rp.build_replication_arms()
    results, frames = [], []
    for name in rp.REPLICATION_ARMS:
        verdict, per_machine = analyze_arm(arms[name], extra_models=extra)
        results.append(verdict)
        frames.append(per_machine)
    return ReplicationAnalysis(arms=results, per_machine=pd.concat(frames, ignore_index=True))


def main() -> None:
    analysis = analyze_replication()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(
        RESULTS_DIR / "replication_scores.csv",
        pd.DataFrame([row.to_json() for arm in analysis.arms for row in arm.scores]),
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "replication_per_machine.csv", analysis.per_machine
    )
    write_json_strict(RESULTS_DIR / "replication.json", analysis.to_json())

    print("--- Result 11: the reversal on rows STD5 does not contain ---")
    for arm in analysis.arms:
        print(
            f"\n{arm.arm}: {arm.n_rows} rows, {arm.n_discharges} discharges, "
            f"{arm.n_machines_scored} machines scored, R {arm.r_min_m:.2f} to {arm.r_max_m:.2f} m"
        )
        print(
            f"  rows shared with STD5: {arm.n_rows_shared_with_std5}   "
            f"physics baseline: {arm.baseline_label}"
        )
        print(f"  {'model':<28}{'CV':>8}{'LOMO':>8}{'ratio':>8}{'CV rank':>9}{'LOMO rank':>11}")
        for score in arm.scores:
            marker = " " if score.is_blind else "*"
            label = MODEL_LABELS.get(score.model_name, score.model_name)
            print(
                f"{marker} {label:<27}{score.cv_rmsle:>8.3f}{score.lomo_mean_rmsle:>8.3f}"
                f"{score.degradation_factor:>8.2f}{score.cv_rank:>9}{score.lomo_rank:>11}"
            )
        verdict = "reversed" if arm.ranking_reversed else "did NOT reverse"
        print(
            f"  best under CV is {MODEL_LABELS.get(arm.best_cv_model, arm.best_cv_model)}, "
            f"best on an unseen machine is "
            f"{MODEL_LABELS.get(arm.best_lomo_model, arm.best_lomo_model)}: the ranking {verdict}"
        )
        print(
            f"  the best cross-validated model beats {arm.baseline_label} by "
            f"{arm.cv_gain_over_baseline * 100:.0f}% under CV, and the two tree models are worse\n"
            f"  than it on {arm.n_machines_trees_lose_to_baseline} of {arm.n_machines} "
            f"machine-model pairs under leave-one-machine-out"
        )
    print("\n  * not a blind baseline")
    print(
        "\n  the non_h arm scores 5 machines, which is too few to carry a claim alone;\n"
        "  it is here because it changes the confinement regime and the baseline law,\n"
        "  which the disjoint_h arm does not"
    )


if __name__ == "__main__":
    main()
