"""Physics as a constraint, and physics as a prior. Results 8 and 9.

Run ``python3 analysis_dimensional.py`` to regenerate everything under
``results/`` for the two ways of putting known physics into a scaling law that
Results 1 to 7 left untried.

Where this sits in the argument
-------------------------------
Result 4 shows a flexible model beating the published power law by 41% under
cross-validation and losing to it on all 13 held-out machines. Result 5 shows
that at the size jump ITER actually asks for, the tree ensembles land closer to
a constant than to the law. Result 6 repairs some of that with a power law plus
a heavily damped correction, reaching 0.206 at the ITER-matched cut.

Every one of those models learns its functional form from the data and is held
back, if at all, by a penalty chosen by cross-validation. None of them is told
any physics. That omission is conspicuous, because the field has known since
Connor and Taylor (1977) that a confinement scaling law is not free: if the
plasma obeys some set of equations, dimensional analysis forces the law into a
constrained form, and the constraint is *linear in the exponents*. It is exactly
the kind of thing least squares can be told.

There are two distinct ways to tell it, and they behave very differently.

    Result 8   As a **constraint**. ``dimensional.py`` derives the
               Connor-Taylor hierarchy from the definitions of rho*, beta and
               nu*, and each rung becomes rows of ``C`` in
               ``min ||Xb - y||^2 s.t. Cb = d``. This says which *directions*
               in exponent space are forbidden, without saying what the answer
               is.

    Result 9   As a **prior**. ``spectral.py`` shrinks the fit toward
               IPB98(y,2)'s published exponents, along the directions Result 3
               showed the data cannot resolve. This says what the answer is, and
               how much to believe it.

The comparison between them is the point. A constraint is much weaker
information than a prior: it names a surface rather than a point. It also turns
out to be worth considerably more.

What is reported
----------------
    Result 8a  How far the published law and the free refit sit from each
               constraint surface. IPB98(y,2) lands on the Kadomtsev surface to
               0.001 and the collisionless surface to 0.005, which is inside the
               rounding of its own two-decimal exponents.
    Result 8b  What each constraint costs in sample and under grouped CV.
    Result 8c  All three splits. The headline: at the ITER-matched cut the
               collisionless-constrained power law is the best *blind* model in
               this repository.
    Result 8d  The whole size sweep, not one cut, with the underpowered rungs
               marked and excluded from every claim.
    Result 9a  Targeted shrinkage against isotropic shrinkage at matched penalty
               strength, which is the control that isolates the targeting.
    Result 9b  The truncation rank sweep, which tests Result 3's prediction that
               the weakest direction alone carries most of the disagreement.

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
import scaling_law as sl
import spectral as sp
from analysis_extrapolation import bootstrap_paired_difference
from analysis_size_extrapolation import MIN_WELL_POWERED_TRAIN_ROWS
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The models carried through the narrative. The four Connor-Taylor rungs, the
# unconstrained power law they are measured against, and the three models from
# Results 4 to 6 that establish what "good" already meant at the ITER cut.
CONSTRAINED_MODELS: tuple[str, ...] = tuple(
    f"powerlaw_{model}" for model in ("free", *dm.CONSTRAINT_MODELS)
)
REFERENCE_MODELS: tuple[str, ...] = (
    "ipb98y2_analytic",
    "ridge_loglinear",
    "random_forest",
    "hist_gradient_boosting",
)
HYBRID_REFERENCE = hdb5.hybrid_model_name("gbm", 1.0)

MODEL_LABELS: dict[str, str] = {
    "powerlaw_free": "power law, unconstrained",
    "powerlaw_kadomtsev": "power law, Kadomtsev",
    "powerlaw_collisionless": "power law, collisionless",
    "powerlaw_electrostatic": "power law, electrostatic",
    "ipb98y2_analytic": "IPB98(y,2), analytic",
    "ridge_loglinear": "ridge, log-linear",
    "random_forest": "random forest",
    "hist_gradient_boosting": "hist gradient boosting",
    HYBRID_REFERENCE: "hybrid (Result 6)",
}

# The rank at which only the single weakest direction has been dropped. Result 3
# says 77% of the published-versus-refit disagreement lives there, so this is
# the rung that prediction is actually about.
WEAKEST_DIRECTION_ONLY_RANK = len(sp.PRIOR_FEATURE_COLUMNS) - 1


@dataclass(frozen=True)
class ConstraintDistance:
    """How far one set of exponents sits from one model's constraint surface.

    The constraint rows are an orthonormal basis of the admissible
    transformations, so ``residual_norm`` is a genuine Euclidean distance in
    exponent space rather than a quantity whose size depends on how the rows
    happened to be scaled.
    """

    exponent_source: str
    constraint_model: str
    n_constraints: int
    residual_norm: float
    max_abs_violation: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SplitScore:
    """One model's RMSLE under all three splits."""

    model_name: str
    is_blind: bool
    cv_rmsle: float
    lomo_mean_rmsle: float
    lomo_median_rmsle: float
    lomo_worst_rmsle: float
    size_cut_rmsle: float
    # Placement between a constant predictor (0.0) and the analytic power law
    # (1.0) at the ITER-matched cut, the same skill score Result 5 reports.
    size_cut_skill: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DimensionalAnalysis:
    n_rows: int
    n_machines_scored: int
    iter_matched_size_ratio: float
    iter_matched_train_machines: int
    constraint_distances: list[ConstraintDistance]
    in_sample_rmsle: dict[str, float]
    split_scores: list[SplitScore]
    per_machine: pd.DataFrame = field(repr=False)
    size_cut_per_machine: pd.DataFrame = field(repr=False)
    size_sweep: pd.DataFrame = field(repr=False)
    prior_sweep: pd.DataFrame = field(repr=False)
    best_blind_at_size_cut: str
    best_blind_size_cut_rmsle: float
    collisionless_vs_ridge: dict[str, object]
    collisionless_vs_hybrid: dict[str, object]
    # Result 9's control: does aiming the penalty at the weak directions beat
    # spreading it evenly, at the same penalty strength?
    targeting_gain_at_matched_alpha: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_machines_scored": self.n_machines_scored,
            "iter_matched_size_ratio": self.iter_matched_size_ratio,
            "iter_matched_train_machines": self.iter_matched_train_machines,
            "constraint_distances": [row.to_json() for row in self.constraint_distances],
            "in_sample_rmsle": self.in_sample_rmsle,
            "split_scores": [row.to_json() for row in self.split_scores],
            "best_blind_at_size_cut": self.best_blind_at_size_cut,
            "best_blind_size_cut_rmsle": self.best_blind_size_cut_rmsle,
            "collisionless_vs_ridge": self.collisionless_vs_ridge,
            "collisionless_vs_hybrid": self.collisionless_vs_hybrid,
            "targeting_gain_at_matched_alpha": self.targeting_gain_at_matched_alpha,
            "min_well_powered_train_rows": MIN_WELL_POWERED_TRAIN_ROWS,
        }


def constraint_distance_table(dataset: pd.DataFrame) -> list[ConstraintDistance]:
    """Result 8a: where the published law and the free refit sit, per surface.

    Two sources are compared. The published IPB98(y,2) exponents are the check
    on the derivation itself: a law written down in 1999 landing on a surface
    derived here from the definitions of rho*, beta and nu* is not something a
    mistaken derivation produces. The free refit of Result 2 is the measurement:
    it says which physics this database declines to obey when nothing makes it.
    """
    refit = dm.fit_constrained_power_law(dataset, hdb5.TARGET_COLUMN, "free")
    sources = {
        "ipb98y2_published": sl.IPB98Y2_EXPONENTS,
        "free_refit": {name: refit[name] for name in dm.CONSTRAINED_FEATURE_COLUMNS},
    }
    rows: list[ConstraintDistance] = []
    for source_name, exponents in sources.items():
        table = dm.constraint_residuals(exponents)
        for _, row in table.iterrows():
            rows.append(
                ConstraintDistance(
                    exponent_source=source_name,
                    constraint_model=str(row["model"]),
                    n_constraints=int(row["n_constraints"]),
                    residual_norm=float(row["residual_norm"]),
                    max_abs_violation=float(row["max_abs_violation"]),
                )
            )
    return rows


def in_sample_cost(dataset: pd.DataFrame) -> dict[str, float]:
    """Result 8b: RMSLE of each rung fitted and scored on all rows.

    In sample deliberately. The question here is not how well a constrained law
    predicts, which the three splits answer, but how much fit the constraint
    *costs* on the rows it was given. A constraint the data already satisfies
    costs nothing, and that is a fact about the database rather than about
    prediction.
    """
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))
    costs: dict[str, float] = {}
    for model in ("free", *dm.CONSTRAINT_MODELS):
        estimator = dm.ConstrainedPowerLaw(model=model).fit(features, log_tau)
        residual = log_tau - estimator.predict(features)
        costs[f"powerlaw_{model}"] = float(np.sqrt(np.mean(residual**2)))
    return costs


def _headline_models() -> dict[str, Any]:
    """The constrained rungs plus the hybrid, as estimators the zoo can take.

    ``ridge_loglinear`` and the two tree models come from ``build_model_zoo``
    already, so they are not repeated here; ``_assemble_zoo`` refuses silently
    overlapping names, which is what keeps that from drifting.
    """
    models: dict[str, Any] = dict(dm.build_constrained_models())
    models.update(hdb5.build_hybrid_models((1.0,), corrections=("gbm",)))
    return models


def score_all_splits(
    dataset: pd.DataFrame,
    extra_models: dict[str, Any],
) -> tuple[list[SplitScore], pd.DataFrame, pd.DataFrame, float, int]:
    """Score one set of models under grouped CV, leave-one-machine-out and the cut.

    Uses ``hdb5``'s own split functions rather than reimplementing them, so
    these numbers sit on the same code path as Results 4 and 5 and a change to
    a split cannot move one and not the other.
    """
    cv_scores = {
        score.model_name: score.cv_rmsle
        for score in hdb5.evaluate_models(
            dataset,
            feature_columns=hdb5.BLIND_FEATURE_COLUMNS,
            extra_models=extra_models,
        )
    }
    per_machine = hdb5.leave_one_tokamak_out(dataset, extra_models=extra_models)

    splits = hdb5.size_ordered_splits(dataset)
    cut = hdb5.iter_matched_split(dataset, splits)
    cut_scores = hdb5.score_size_split(
        dataset, cut, extra_models=extra_models, per_machine=True
    )
    pooled = cut_scores[cut_scores["scope"] == "__pooled__"].set_index("model_name")

    mean_rmsle = float(pooled.loc["mean_baseline", "rmsle"])
    analytic_rmsle = float(pooled.loc["ipb98y2_analytic", "rmsle"])

    scores: list[SplitScore] = []
    grouped = per_machine.groupby("model_name")
    for model_name, group in grouped:
        name = str(model_name)
        if name not in pooled.index or name not in cv_scores:
            continue
        size_rmsle = float(pooled.loc[name, "rmsle"])
        # Skill places a model between predicting a constant and the analytic
        # law, exactly as Result 5 does, so the two tables are readable together.
        denominator = mean_rmsle - analytic_rmsle
        skill = (mean_rmsle - size_rmsle) / denominator if denominator else float("nan")
        scores.append(
            SplitScore(
                model_name=name,
                is_blind=bool(group["is_blind"].iloc[0]),
                cv_rmsle=cv_scores[name],
                lomo_mean_rmsle=float(group["rmsle"].mean()),
                lomo_median_rmsle=float(group["rmsle"].median()),
                lomo_worst_rmsle=float(group["rmsle"].max()),
                size_cut_rmsle=size_rmsle,
                size_cut_skill=float(skill),
            )
        )
    scores.sort(key=lambda score: score.size_cut_rmsle)
    return scores, per_machine, cut_scores, cut.size_ratio, cut.n_train_machines


def size_sweep(dataset: pd.DataFrame, extra_models: dict[str, Any]) -> pd.DataFrame:
    """Result 8d: every size cut, not only the ITER-matched one.

    Result 6e made the case for doing this: a model that wins at one rung of a
    sweep and loses at the others has not shown a mechanism, it has shown a
    coincidence. The underpowered rungs are flagged rather than dropped, so the
    exclusion is visible instead of implicit.
    """
    scores, splits = hdb5.size_extrapolation_report(dataset, extra_models=extra_models)
    by_cut = {split.n_train_machines: split for split in splits}
    pooled = scores[scores["scope"] == "__pooled__"].copy()
    pooled["n_train_rows"] = pooled["n_train_machines"].map(
        lambda cut: by_cut[int(cut)].n_train_rows
    )
    pooled["well_powered"] = pooled["n_train_rows"] >= MIN_WELL_POWERED_TRAIN_ROWS
    return pooled


def prior_shrinkage_sweep(dataset: pd.DataFrame) -> pd.DataFrame:
    """Result 9: the prior-shrinkage family, scored on all three splits.

    Scored separately from the constrained models because the family is large
    (two weightings over nine decades, plus the rank sweep) and every member of
    it is a linear model, so running it through the tree zoo as well would
    multiply the cost of this script for nothing.
    """
    features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()

    splits = hdb5.size_ordered_splits(dataset)
    cut = hdb5.iter_matched_split(dataset, splits)
    train_mask = np.isin(labels, list(cut.train_machines))
    machines = hdb5.eligible_tokamaks(dataset)

    def rmsle(index: np.ndarray, predicted_log: np.ndarray) -> float:
        return float(np.sqrt(np.mean((log_tau[index] - predicted_log) ** 2)))

    records: list[dict[str, object]] = []
    for name, estimator in sp.build_prior_shrinkage_models().items():
        out_of_fold = hdb5._grouped_cv_predictions(
            estimator, features, log_tau, groups, hdb5.N_CV_FOLDS
        )
        cv = float(np.sqrt(np.mean((log_tau - out_of_fold) ** 2)))

        machine_scores = []
        for machine in machines:
            held = np.flatnonzero(labels == machine)
            train = np.flatnonzero(labels != machine)
            model = hdb5.clone_pipeline(estimator)
            with hdb5._suppress_benign_matmul_warnings():
                hdb5.fit_pipeline(model, features.iloc[train], log_tau[train])
                machine_scores.append(rmsle(held, model.predict(features.iloc[held])))

        model = hdb5.clone_pipeline(estimator)
        held = np.flatnonzero(~train_mask)
        with hdb5._suppress_benign_matmul_warnings():
            hdb5.fit_pipeline(model, features[train_mask], log_tau[train_mask])
            size_rmsle = rmsle(held, model.predict(features.iloc[held]))

        weighting, parameter = _parse_prior_name(name)
        records.append(
            {
                "model_name": name,
                "weighting": weighting,
                "parameter": parameter,
                "cv_rmsle": cv,
                "lomo_mean_rmsle": float(np.mean(machine_scores)),
                "lomo_worst_rmsle": float(np.max(machine_scores)),
                "size_cut_rmsle": size_rmsle,
            }
        )
    return pd.DataFrame(records)


def _parse_prior_name(name: str) -> tuple[str, float]:
    """Split ``prior_<weighting>_<a|k><value>`` back into its two axes.

    The names are built for the zoo, which needs them flat and filesystem-safe;
    the sweep needs the two axes separately to group by. Parsing here keeps the
    naming defined in one place, ``spectral.prior_model_name``, rather than
    having a second convention grow in this file.
    """
    _, weighting, parameter = name.split("_", 2)
    if parameter.startswith("k"):
        return weighting, float(parameter[1:])
    return weighting, float(parameter[1:].replace("p", ".").replace("m", "-"))


def analyze_dimensional(dataset: pd.DataFrame) -> DimensionalAnalysis:
    extra = _headline_models()
    scores, per_machine, cut_scores, size_ratio, n_train = score_all_splits(dataset, extra)
    sweep = size_sweep(dataset, extra)
    prior = prior_shrinkage_sweep(dataset)

    blind = [score for score in scores if score.is_blind]
    best_blind = min(blind, key=lambda score: score.size_cut_rmsle)

    collisionless = "powerlaw_collisionless"
    vs_ridge = bootstrap_paired_difference(per_machine, collisionless, "ridge_loglinear")
    vs_hybrid = bootstrap_paired_difference(per_machine, collisionless, HYBRID_REFERENCE)

    # Result 9's control. At each shared alpha, how much better is aiming the
    # penalty at the weak directions than spreading it evenly? Reported at the
    # ITER-matched cut, which is where the two differ most.
    pivot = prior[prior["weighting"].isin(sp.WEIGHTINGS)].pivot_table(
        index="parameter", columns="weighting", values="size_cut_rmsle"
    )
    targeting = {
        f"alpha_{alpha:g}": float(pivot.loc[alpha, "isotropic"] - pivot.loc[alpha, "spectral"])
        for alpha in pivot.index
    }

    return DimensionalAnalysis(
        n_rows=int(len(dataset)),
        n_machines_scored=int(per_machine["tokamak"].nunique()),
        iter_matched_size_ratio=float(size_ratio),
        iter_matched_train_machines=int(n_train),
        constraint_distances=constraint_distance_table(dataset),
        in_sample_rmsle=in_sample_cost(dataset),
        split_scores=scores,
        per_machine=per_machine,
        size_cut_per_machine=cut_scores[cut_scores["scope"] != "__pooled__"].copy(),
        size_sweep=sweep,
        prior_sweep=prior,
        best_blind_at_size_cut=best_blind.model_name,
        best_blind_size_cut_rmsle=best_blind.size_cut_rmsle,
        collisionless_vs_ridge=asdict(vs_ridge),
        collisionless_vs_hybrid=asdict(vs_hybrid),
        targeting_gain_at_matched_alpha=targeting,
    )


def plot_dimensional(analysis: DimensionalAnalysis) -> Path | None:
    """Two panels: what each constraint costs, and what it buys.

    Left is the size sweep, which is where the claim lives; right is the
    in-sample cost against the ITER-cut score, which is the trade the whole
    result is about. Matplotlib is imported inside the function so the analysis
    can run headless without it installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - figure is optional
        return None

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    sweep = analysis.size_sweep
    palette = {
        "powerlaw_free": ("#888888", "o", "unconstrained"),
        "powerlaw_kadomtsev": ("#2a78d6", "s", "Kadomtsev"),
        "powerlaw_collisionless": ("#1a9850", "D", "collisionless"),
        "powerlaw_electrostatic": ("#eb6834", "^", "electrostatic"),
        "ipb98y2_analytic": ("#000000", None, "IPB98(y,2), not blind"),
    }
    for model, (colour, marker, label) in palette.items():
        subset = sweep[sweep["model_name"] == model].sort_values("size_ratio")
        powered = subset[subset["well_powered"]]
        style = {"color": colour, "label": label, "linewidth": 1.8}
        if marker is None:
            left.plot(powered["size_ratio"], powered["rmsle"], linestyle="--", **style)
        else:
            left.plot(powered["size_ratio"], powered["rmsle"], marker=marker, **style)
        thin = subset[~subset["well_powered"]]
        left.scatter(thin["size_ratio"], thin["rmsle"], color=colour, alpha=0.25, s=22)

    left.set_ylim(0.0, 0.75)
    left.axvline(
        analysis.iter_matched_size_ratio,
        color="#c0392b",
        linestyle=":",
        linewidth=1.4,
    )
    # Axes-fraction y so the label sits just under the top spine whatever the
    # data limits end up being, rather than being placed at a data value that a
    # later ``set_ylim`` would move out from under it.
    left.annotate(
        "ITER-matched",
        xy=(analysis.iter_matched_size_ratio, 0.97),
        xycoords=("data", "axes fraction"),
        color="#c0392b",
        va="top",
        ha="left",
        fontsize=13,
        xytext=(4, 0),
        textcoords="offset points",
    )
    left.set_xlabel("size ratio demanded by the cut (test R max / train R max)")
    left.set_ylabel("RMSLE on the held-out machines")
    left.set_title("Score at every size cut")
    left.legend(fontsize=13, frameon=False)
    left.grid(alpha=0.15)

    in_sample = analysis.in_sample_rmsle
    by_name = {score.model_name: score for score in analysis.split_scores}
    for model, (colour, marker, label) in palette.items():
        if model not in in_sample or model not in by_name:
            continue
        right.scatter(
            in_sample[model],
            by_name[model].size_cut_rmsle,
            color=colour,
            marker=marker or "o",
            s=90,
            zorder=3,
        )
        right.annotate(
            label,
            (in_sample[model], by_name[model].size_cut_rmsle),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=13,
            color=colour,
        )
    right.set_xlabel("in-sample RMSLE (what the constraint costs)")
    right.set_ylabel("RMSLE at the ITER-matched cut (what it buys)")
    right.set_title("What each constraint costs against what it buys")
    right.grid(alpha=0.15)

    figure.tight_layout()
    path = RESULTS_DIR / "dimensional.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    analysis = analyze_dimensional(dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(
        RESULTS_DIR / "dimensional_constraints.csv",
        pd.DataFrame([row.to_json() for row in analysis.constraint_distances]),
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "dimensional_splits.csv",
        pd.DataFrame([row.to_json() for row in analysis.split_scores]),
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "dimensional_per_machine.csv", analysis.per_machine
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "dimensional_size_cut_per_machine.csv", analysis.size_cut_per_machine
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "dimensional_size_sweep.csv", analysis.size_sweep
    )
    write_dataframe_csv_atomic(
        RESULTS_DIR / "dimensional_prior_sweep.csv", analysis.prior_sweep
    )
    write_json_strict(RESULTS_DIR / "dimensional.json", analysis.to_json())
    figure_path = plot_dimensional(analysis)

    print("--- Result 8a: distance from each Connor-Taylor constraint surface ---")
    print(f"  {'surface':<16}{'IPB98(y,2)':>14}{'free refit':>14}")
    distances = pd.DataFrame([row.to_json() for row in analysis.constraint_distances])
    for model in dm.CONSTRAINT_MODELS:
        subset = distances[distances["constraint_model"] == model].set_index("exponent_source")
        print(
            f"  {model:<16}{subset.loc['ipb98y2_published', 'residual_norm']:>14.4f}"
            f"{subset.loc['free_refit', 'residual_norm']:>14.4f}"
        )
    print(
        "  the published law lies on the first two surfaces to inside the rounding of\n"
        "  its own two-decimal exponents; the free refit leaves the collisionless one"
    )

    print("\n--- Result 8b/8c: what each constraint costs, and what it buys ---")
    print(
        f"  {'model':<28}{'in-sample':>11}{'CV':>8}{'LOMO':>8}{'ITER cut':>10}{'skill':>8}"
    )
    for score in analysis.split_scores:
        if score.model_name not in (*CONSTRAINED_MODELS, *REFERENCE_MODELS, HYBRID_REFERENCE):
            continue
        marker = " " if score.is_blind else "*"
        in_sample = analysis.in_sample_rmsle.get(score.model_name)
        in_sample_text = f"{in_sample:>11.4f}" if in_sample is not None else f"{'-':>11}"
        print(
            f"{marker} {MODEL_LABELS.get(score.model_name, score.model_name):<27}"
            f"{in_sample_text}{score.cv_rmsle:>8.3f}{score.lomo_mean_rmsle:>8.3f}"
            f"{score.size_cut_rmsle:>10.3f}{score.size_cut_skill:>8.2f}"
        )
    print("  * fitted on this database, held-out machines included; not a blind baseline")
    print(
        f"\n  best blind model at the ITER-matched cut: "
        f"{MODEL_LABELS.get(analysis.best_blind_at_size_cut, analysis.best_blind_at_size_cut)}"
        f" at {analysis.best_blind_size_cut_rmsle:.3f}"
    )

    print("\n--- Result 8c: paired over the 13 held-out machines ---")
    for label, difference in (
        ("against ridge", analysis.collisionless_vs_ridge),
        ("against the Result 6 hybrid", analysis.collisionless_vs_hybrid),
    ):
        print(
            f"  collisionless {label:<28}"
            f"mean {difference['mean_difference']:+.3f} "
            f"[{difference['ci_low']:+.3f}, {difference['ci_high']:+.3f}], "
            f"worse on {difference['n_machines_a_worse']} of {difference['n_machines']}"
        )
    print("  negative means the constrained law is the better of the pair")

    # The pooled ITER-cut number is dominated by JET and JET-ILW, so the
    # per-machine breakdown is not decoration: it is the check on whether the
    # pooled win is a win on machines or a win on row counts.
    print("\n--- Result 8c: the ITER-matched cut, per held-out machine ---")
    cut_machines = analysis.size_cut_per_machine
    shown = [*CONSTRAINED_MODELS, "ipb98y2_analytic", "ridge_loglinear", HYBRID_REFERENCE]
    table = (
        cut_machines[cut_machines["model_name"].isin(shown)]
        .pivot_table(index="model_name", columns="scope", values="rmsle")
        .reindex(shown)
    )
    header = "".join(f"{str(machine):>10}" for machine in table.columns)
    print(f"  {'model':<28}{header}")
    for name, row in table.iterrows():
        marker = " " if str(name) != "ipb98y2_analytic" else "*"
        values = "".join(f"{value:>10.3f}" for value in row)
        print(f"{marker} {MODEL_LABELS.get(str(name), str(name)):<27}{values}")
    print(
        "  three machines is a small check, and it is the one that decides whether the\n"
        "  pooled number above is a statement about machines or about JET's row count"
    )

    print("\n--- Result 9a: targeting the penalty, against spreading it evenly ---")
    print(f"  {'alpha':>10}{'isotropic':>12}{'spectral':>11}{'gain':>9}")
    prior = analysis.prior_sweep
    pivot = prior[prior["weighting"].isin(sp.WEIGHTINGS)].pivot_table(
        index="parameter", columns="weighting", values="size_cut_rmsle"
    )
    for alpha in pivot.index:
        print(
            f"  {alpha:>10g}{pivot.loc[alpha, 'isotropic']:>12.3f}"
            f"{pivot.loc[alpha, 'spectral']:>11.3f}"
            f"{pivot.loc[alpha, 'isotropic'] - pivot.loc[alpha, 'spectral']:>+9.3f}"
        )
    print("  RMSLE at the ITER-matched cut; positive gain means targeting won")

    print("\n--- Result 9b: truncation rank, data in the k best-determined directions ---")
    truncated = prior[prior["weighting"] == "truncated"].sort_values("parameter", ascending=False)
    print(f"  {'k':>4}{'CV':>9}{'LOMO':>9}{'ITER cut':>11}")
    for _, row in truncated.iterrows():
        print(
            f"  {int(row['parameter']):>4}{row['cv_rmsle']:>9.3f}"
            f"{row['lomo_mean_rmsle']:>9.3f}{row['size_cut_rmsle']:>11.3f}"
        )
    print(
        f"  k={len(sp.PRIOR_FEATURE_COLUMNS)} is the unconstrained fit and k=0 is the published\n"
        f"  exponents renormalised; k={WEAKEST_DIRECTION_ONLY_RANK} drops only the weakest direction"
    )

    if figure_path:
        print(f"\nfigure: {figure_path}")


if __name__ == "__main__":
    main()
