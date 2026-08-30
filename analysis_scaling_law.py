"""Reproduce the three scaling-law results on the real ITPA HDB5 database.

Run ``python3 analysis_scaling_law.py`` to regenerate everything under
``results/``: the rank audit, the refit of IPB98(y,2) from data, the singular
value spectrum, and the comparison against the published scaling law.

    Result 1  The feature matrix the confinement model is trained on is rank
              deficient by exactly two, and both dependencies are identifiable.
    Result 2  Refit the IPB98(y,2) exponents from HDB5 with three independently
              implemented solvers and compare against the published values.
    Result 3  Show that the disagreement with the published law lives almost
              entirely in the directions the database cannot determine.

The narrative built on these numbers is in ``results/RESULTS.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import hdb5
from scaling_law import (
    IPB98_FEATURE_COLUMNS,
    IPB98Y2_COEFFICIENT,
    IPB98Y2_COEFFICIENT_ROUNDED,
    IPB98Y2_EXPONENTS,
    analyze_conditioning,
    bootstrap_exponents,
    build_log_design_matrix,
    fit_scaling_law,
    ridge_shrinkage_factors,
    solve_lstsq_cholesky,
    solve_lstsq_qr,
    solve_lstsq_svd,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

Solver = Callable[[np.ndarray, np.ndarray], np.ndarray]

ASPECT_RATIO_IDENTITY = "a = eps * R"
IPB98_PRIOR_IDENTITY = "log IPB98 prior = sum of exponent-weighted logs"
CONTROL_VECTOR = "control (log_ip_ma alone)"


# --- Result 1: the rank audit -----------------------------------------------


@dataclass(frozen=True)
class RankAudit:
    columns: list[str]
    n_rows: int
    n_columns: int
    rank: int
    rank_deficiency: int
    condition_number: float
    singular_values: np.ndarray = field(repr=False)
    tolerance: float
    projection_residuals: dict[str, float]
    basis_alignments: dict[str, float]
    unstandardized_rank: int

    def to_json(self) -> dict[str, object]:
        return {
            "columns": self.columns,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "rank": self.rank,
            "rank_deficiency": self.rank_deficiency,
            "condition_number": self.condition_number,
            "singular_values": self.singular_values.tolist(),
            "tolerance": self.tolerance,
            "projection_residuals": self.projection_residuals,
            "max_alignment_with_a_printed_basis_vector": self.basis_alignments,
            "unstandardized_rank": self.unstandardized_rank,
        }


def audit_model_feature_matrix(dataset: pd.DataFrame) -> RankAudit:
    """Audit the log-feature matrix the confinement model is actually trained on.

    Two exact dependencies are expected, and they differ in kind:

    1. ``log a = log R + log eps``, because minor radius is *defined* as
       ``a = eps * R`` in the cleaning step. A definitional identity restated as
       a feature.

    2. ``log_ipb98y2_tau_s`` is the log of a power law in the other eight
       features, so it is their fixed linear combination plus a constant. The
       IPB98 prior carries no information a log-linear model did not already
       have. It is genuinely useful to the tree models, which cannot form that
       combination themselves, and exactly zero new information to the linear
       one sitting beside them in the same comparison.

    The second is the interesting failure: adding a published physics scaling as
    a feature feels like adding knowledge, and in log space it provably is not.
    """
    columns = list(hdb5.MODEL_FEATURE_COLUMNS)
    matrix = dataset.loc[:, columns].to_numpy(dtype=float)
    report = analyze_conditioning(matrix, columns, standardize=True)
    index = {name: position for position, name in enumerate(columns)}

    aspect_ratio = np.zeros(len(columns))
    aspect_ratio[index["log_a_m"]] = 1.0
    aspect_ratio[index["log_r_m"]] = -1.0
    aspect_ratio[index["log_inverse_aspect_ratio"]] = -1.0

    ipb98_prior = np.zeros(len(columns))
    ipb98_prior[index["log_ipb98y2_tau_s"]] = 1.0
    for variable, exponent in IPB98Y2_EXPONENTS.items():
        ipb98_prior[index[f"log_{variable}"]] = -exponent

    control = np.zeros(len(columns))
    control[index["log_ip_ma"]] = 1.0

    def max_alignment(vector: np.ndarray) -> float:
        """How parallel the closest *printed* basis vector is to what we expect.

        Near 1 means the naive check would have happened to work here. It is
        luck, not method: with a null space of dimension greater than one the
        returned basis is arbitrary.
        """
        scaled = report.to_analysis_coordinates(vector)
        return max(
            float(abs(float(basis @ scaled) / float(np.linalg.norm(basis) * np.linalg.norm(scaled))))
            for basis in report.null_space
        )

    return RankAudit(
        columns=columns,
        n_rows=int(matrix.shape[0]),
        n_columns=report.n_columns,
        rank=report.rank,
        rank_deficiency=report.rank_deficiency,
        condition_number=report.condition_number,
        singular_values=report.singular_values,
        tolerance=report.tolerance,
        projection_residuals={
            ASPECT_RATIO_IDENTITY: report.null_space_residual(aspect_ratio, raw_units=True),
            IPB98_PRIOR_IDENTITY: report.null_space_residual(ipb98_prior, raw_units=True),
            CONTROL_VECTOR: report.null_space_residual(control, raw_units=True),
        },
        basis_alignments={
            ASPECT_RATIO_IDENTITY: max_alignment(aspect_ratio),
            IPB98_PRIOR_IDENTITY: max_alignment(ipb98_prior),
        },
        unstandardized_rank=int(np.linalg.matrix_rank(matrix)),
    )


# --- Result 2: refit IPB98 ---------------------------------------------------


@dataclass(frozen=True)
class SolverTiming:
    name: str
    seconds_per_solve: float
    max_deviation_from_svd: float


@dataclass(frozen=True)
class Refit:
    design_shape: tuple[int, int]
    solvers: list[SolverTiming]
    fitted_coefficient: float
    residual_std_log: float
    condition_number: float
    rmsle_refit: float
    rmsle_published: float
    intervals: pd.DataFrame = field(repr=False)

    def to_json(self) -> dict[str, object]:
        return {
            "design_matrix_shape": list(self.design_shape),
            "solvers": [
                {
                    "name": solver.name,
                    "seconds_per_solve": solver.seconds_per_solve,
                    "max_deviation_from_svd": solver.max_deviation_from_svd,
                }
                for solver in self.solvers
            ],
            "fitted_coefficient": self.fitted_coefficient,
            "published_coefficient": IPB98Y2_COEFFICIENT,
            "published_coefficient_rounded_variant": IPB98Y2_COEFFICIENT_ROUNDED,
            "residual_std_log": self.residual_std_log,
            "condition_number": self.condition_number,
            "in_sample_rmsle_refit": self.rmsle_refit,
            "in_sample_rmsle_published_ipb98y2": self.rmsle_published,
        }


def refit_ipb98(dataset: pd.DataFrame, *, n_resamples: int = 1000) -> Refit:
    design, _ = build_log_design_matrix(dataset, IPB98_FEATURE_COLUMNS)
    target = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    solvers: dict[str, Solver] = {
        "cholesky": solve_lstsq_cholesky,
        "qr": solve_lstsq_qr,
        "svd": solve_lstsq_svd,
    }
    solutions: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for name, solver in solvers.items():
        start = time.perf_counter()
        for _ in range(20):
            beta = solver(design, target)
        timings[name] = (time.perf_counter() - start) / 20.0
        solutions[name] = beta

    reference = solutions["svd"]
    solver_rows = [
        SolverTiming(
            name=name,
            seconds_per_solve=timings[name],
            max_deviation_from_svd=float(np.max(np.abs(solutions[name] - reference))),
        )
        for name in solvers
    ]

    fit = fit_scaling_law(dataset, hdb5.TARGET_COLUMN, IPB98_FEATURE_COLUMNS)
    intervals = bootstrap_exponents(
        dataset,
        hdb5.TARGET_COLUMN,
        IPB98_FEATURE_COLUMNS,
        group_column=hdb5.GROUP_COLUMN,
        n_resamples=n_resamples,
    )

    actual = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)

    def rmsle(prediction: np.ndarray) -> float:
        return float(np.sqrt(np.mean((np.log(prediction) - np.log(actual)) ** 2)))

    return Refit(
        design_shape=(int(design.shape[0]), int(design.shape[1])),
        solvers=solver_rows,
        fitted_coefficient=fit.coefficient,
        residual_std_log=fit.residual_std_log,
        condition_number=fit.conditioning.condition_number,
        rmsle_refit=rmsle(fit.predict(dataset, IPB98_FEATURE_COLUMNS)),
        rmsle_published=rmsle(dataset["ipb98y2_tau_s"].to_numpy(dtype=float)),
        intervals=intervals,
    )


# --- Result 3: what the data can determine -----------------------------------


@dataclass(frozen=True)
class Direction:
    index: int
    singular_value: float
    share_of_design_variance: float
    share_of_disagreement: float
    dominant_variables: list[str]


@dataclass(frozen=True)
class Spectrum:
    condition_number: float
    singular_values: np.ndarray = field(repr=False)
    directions: list[Direction] = field(default_factory=list)
    shrinkage: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def to_json(self) -> dict[str, object]:
        return {
            "condition_number": self.condition_number,
            "singular_values": self.singular_values.tolist(),
            "directions": [
                {
                    "index": direction.index,
                    "singular_value": direction.singular_value,
                    "share_of_design_variance": direction.share_of_design_variance,
                    "share_of_disagreement": direction.share_of_disagreement,
                    "dominant_variables": direction.dominant_variables,
                }
                for direction in self.directions
            ],
        }


def conditioning_analysis(dataset: pd.DataFrame) -> Spectrum:
    """Singular spectrum of the physics design matrix, and where we disagree.

    If the difference between our refit exponents and IPB98's lay along the
    well-determined directions it would be a real physical disagreement. If it
    lies along the weak ones, both laws fit the data about equally well and the
    individual exponents were never separately pinned down. Mapping the
    difference into standardized coordinates first (multiply by the column
    standard deviations) is required, because the singular vectors live there
    and the exponents do not.
    """
    design, names = build_log_design_matrix(dataset, IPB98_FEATURE_COLUMNS, intercept=False)
    report = analyze_conditioning(design, names, standardize=True)

    standardized = np.array(design, dtype=float)
    for column in range(standardized.shape[1]):
        values = standardized[:, column]
        standardized[:, column] = (values - values.mean()) / values.std()
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    design_variance = singular_values**2 / np.sum(singular_values**2)

    fit = fit_scaling_law(dataset, hdb5.TARGET_COLUMN, IPB98_FEATURE_COLUMNS)
    raw_difference = np.array([fit.exponents[column] - IPB98Y2_EXPONENTS[column] for column in IPB98_FEATURE_COLUMNS])
    difference = raw_difference * report.column_scales
    projections = vt @ difference
    disagreement = projections**2 / np.sum(projections**2)

    directions = []
    for position in range(len(singular_values)):
        loading = vt[position]
        ordered = np.argsort(-np.abs(loading))
        directions.append(
            Direction(
                index=position + 1,
                singular_value=float(singular_values[position]),
                share_of_design_variance=float(design_variance[position]),
                share_of_disagreement=float(disagreement[position]),
                dominant_variables=[f"{names[i].removeprefix('log_')}: {loading[i]:+.3f}" for i in ordered[:4]],
            )
        )

    shrinkage = pd.DataFrame(
        {
            "singular_value": singular_values,
            **{
                f"alpha_{alpha:g}": ridge_shrinkage_factors(singular_values, alpha) for alpha in (0.1, 1.0, 10.0, 100.0)
            },
        }
    )

    return Spectrum(
        condition_number=report.condition_number,
        singular_values=singular_values,
        directions=directions,
        shrinkage=shrinkage,
    )


# --- Figure -------------------------------------------------------------------


def plot_spectrum(spectrum: Spectrum, audit: RankAudit) -> Path | None:
    """Three panels: the physics spectrum, the rank cliff, and where we disagree.

    The right panel is the result. It puts the share of the design matrix's
    variance carried by each singular direction next to the share of our
    disagreement with IPB98 that lives in that direction, on one common
    percentage axis so the two are directly comparable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    blue, orange = "#2a78d6", "#eb6834"
    ink, muted = "#0b0b0b", "#52514e"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.5]})
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(muted)
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=muted, labelsize=9)

    physics = spectrum.singular_values
    axes[0].semilogy(range(1, len(physics) + 1), physics, "o-", color=blue, linewidth=1.8, markersize=6)
    axes[0].set_title("Engineering variables\n(8 columns, full rank)", fontsize=11, color=ink)
    axes[0].set_xlabel("singular value index", fontsize=9, color=muted)
    axes[0].set_ylabel("singular value (standardized)", fontsize=9, color=muted)
    axes[0].annotate(
        f"condition number {spectrum.condition_number:.1f}\nno ill conditioning to report",
        xy=(0.06, 0.06),
        xycoords="axes fraction",
        fontsize=9,
        color=muted,
    )

    model = audit.singular_values
    floor = 1e-16
    axes[1].semilogy(
        range(1, len(model) + 1),
        np.maximum(model, floor),
        "o-",
        color=orange,
        linewidth=1.8,
        markersize=6,
    )
    axes[1].axhline(audit.tolerance, color=muted, linestyle="--", linewidth=1.0)
    axes[1].set_title(
        f"Model feature matrix\n({audit.n_columns} columns, rank {audit.rank})",
        fontsize=11,
        color=ink,
    )
    axes[1].set_xlabel("singular value index", fontsize=9, color=muted)
    axes[1].annotate("numerical zero", xy=(1.2, audit.tolerance * 2.0), fontsize=8, color=muted)
    axes[1].annotate(
        "two exact\ndependencies",
        xy=(len(model) - 0.6, max(float(model[-1]), floor)),
        xytext=(len(model) - 4.4, 1e-9),
        fontsize=9,
        color=ink,
        arrowprops={"arrowstyle": "->", "color": muted, "linewidth": 0.9},
    )

    index = np.arange(len(spectrum.directions))
    design_share = np.array([d.share_of_design_variance for d in spectrum.directions]) * 100
    disagreement_share = np.array([d.share_of_disagreement for d in spectrum.directions]) * 100
    width = 0.38
    axes[2].bar(
        index - width / 2 - 0.01,
        design_share,
        width,
        color=blue,
        label="share of what the data determines",
    )
    axes[2].bar(
        index + width / 2 + 0.01,
        disagreement_share,
        width,
        color=orange,
        label="share of our disagreement with IPB98",
    )
    axes[2].annotate(
        f"{disagreement_share[-1]:.0f}%",
        xy=(index[-1] + width / 2 + 0.01, disagreement_share[-1]),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color=ink,
    )
    axes[2].set_xticks(index)
    axes[2].set_xticklabels([str(d.index) for d in spectrum.directions])
    axes[2].set_xlabel("singular direction (strongest to weakest)", fontsize=9, color=muted)
    axes[2].set_ylabel("percent", fontsize=9, color=muted)
    axes[2].set_title("The disagreement lives where the data is blind", fontsize=11, color=ink)
    axes[2].legend(frameon=False, fontsize=9, loc="upper center", labelcolor=muted)
    axes[2].set_ylim(0, max(disagreement_share.max(), design_share.max()) * 1.28)

    figure.tight_layout()
    path = RESULTS_DIR / "singular_value_spectrum.png"
    figure.savefig(path, dpi=180, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def main() -> None:
    dataset = hdb5.prepare_dataset()
    print(
        f"HDB5: {len(dataset)} rows, {dataset[hdb5.GROUP_COLUMN].nunique()} discharges, "
        f"{dataset[hdb5.TOKAMAK_LABEL_COLUMN].nunique()} tokamaks"
    )

    audit = audit_model_feature_matrix(dataset)
    refit = refit_ipb98(dataset)
    spectrum = conditioning_analysis(dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    refit.intervals.to_csv(RESULTS_DIR / "ipb98_refit_exponents.csv", index=False)
    spectrum.shrinkage.to_csv(RESULTS_DIR / "ridge_shrinkage.csv", index=False)
    (RESULTS_DIR / "analysis.json").write_text(
        json.dumps(
            {
                "rank_audit": audit.to_json(),
                "refit": refit.to_json(),
                "conditioning": spectrum.to_json(),
            },
            indent=2,
        )
    )
    figure_path = plot_spectrum(spectrum, audit)

    print("\n--- Result 1: rank audit of the model feature matrix ---")
    print(f"rank {audit.rank} of {audit.n_columns} (deficiency {audit.rank_deficiency})")
    print(f"numerical-zero tolerance {audit.tolerance:.2e}")
    for label, residual in audit.projection_residuals.items():
        print(f"  projection residual, {label}: {residual:.3e}")
    for label, alignment in audit.basis_alignments.items():
        print(f"  best alignment with a printed basis vector, {label}: {alignment:.3f}")
    print(f"  rank without standardizing first: {audit.unstandardized_rank} (a unit artifact)")

    print("\n--- Result 2: IPB98 refit from HDB5 ---")
    print(f"design matrix {list(refit.design_shape)}, cond {refit.condition_number:.1f}")
    for solver in refit.solvers:
        print(
            f"  {solver.name:<9} {solver.seconds_per_solve * 1e3:7.3f} ms   "
            f"max deviation from SVD {solver.max_deviation_from_svd:.2e}"
        )
    print(refit.intervals.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(
        f"  fitted coefficient {refit.fitted_coefficient:.4f} vs published "
        f"{IPB98Y2_COEFFICIENT} (also quoted as {IPB98Y2_COEFFICIENT_ROUNDED})"
    )
    print(f"  in-sample RMSLE: refit {refit.rmsle_refit:.4f}, published IPB98(y,2) {refit.rmsle_published:.4f}")

    print("\n--- Result 3: what the data determines ---")
    print(f"condition number {spectrum.condition_number:.1f}")
    print("  direction  sigma    share of design variance    share of disagreement")
    for direction in spectrum.directions:
        print(
            f"  {direction.index:>9}  {direction.singular_value:7.2f}"
            f"  {direction.share_of_design_variance * 100:22.2f}%"
            f"  {direction.share_of_disagreement * 100:19.2f}%"
        )
    weakest = spectrum.directions[-1]
    print(f"  weakest direction: {', '.join(weakest.dominant_variables)}")
    if figure_path:
        print(f"\nfigure: {figure_path}")


if __name__ == "__main__":
    main()
