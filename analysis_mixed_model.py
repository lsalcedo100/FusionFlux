"""Result 17: the estimator the clustered-validation literature recommends.

Table 1 sets a pooled global fit against tree ensembles and concludes for the
pooled fit. It never scores the estimator the literature it cites for the split
design pairs with that design: a mixed model carrying a random intercept per
device, predicting an unseen device from the population-level fixed effects
alone. That estimator is neither a power law nor a tree, it is the one matched
to the deployment question, and Limitations conceded its absence rather than
answering it.

This module fits it. Write

    log tau_ij = x_ij' beta + u_i + eps_ij,
    u_i ~ N(0, sigma_u^2),  eps_ij ~ N(0, sigma^2),

with i a device (or a database label, both are scored) and j a row on it. For a
device the model has never seen there is no u_i to add, so the prediction is
x' beta: the fixed effects are exactly the population-level surface a new
machine is projected from, which is why this is the deployment-matched fit and
not merely another baseline.

The variance components are fitted by REML, profiled over the single ratio
lambda = sigma_u^2 / sigma^2. With one intercept per group the GLS solve has a
closed form, so no iterative solver and no extra dependency is needed: with
theta_i = 1 - 1/sqrt(1 + n_i lambda), subtracting theta_i times a group's mean
from its rows whitens the covariance exactly, and ordinary least squares on the
transformed rows is the GLS estimate.

That ratio is also the whole of what separates this estimator from the pooled
one. At lambda = 0 the model *is* the pooled power law, and as lambda grows the
random intercept absorbs more of the between-device variation, which leaves the
fixed effects identified increasingly from variation within a device. Since
several of the nine features (major radius, aspect ratio, and to a large extent
elongation and isotopic mass) are close to constant within a device, that is the
identification the deployment question cannot afford: it is exactly the
between-device slope that projects a new machine.

So the ratio is swept as well as fitted. REML on 11 to 15 groups is a noisy
estimate, and reporting only where the optimiser landed would leave the reader
unable to tell an unstable optimum from a property of the estimator. The sweep
reports the held-out score at fixed ratios, including the hindsight-best one,
which bounds what any selection rule for lambda could have achieved.

The design matrix is rank deficient (log a = log R + log epsilon holds exactly
by construction), so an independent column subset is selected by pivoted QR on
each training fold before the solve. The dependency holds on held-out rows too,
so which subset is kept does not move a prediction.

Run ``python3 analysis_mixed_model.py`` to regenerate
``results/mixed_model.json`` and its two per-unit tables.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy import linalg, optimize

import hdb5
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The comparator the reversal is stated against, and the model it reverses.
POWER_LAW = "ridge_loglinear"
FOREST = "random_forest"

# Bracket for the profiled variance ratio, in logs. The lower end is a pooled
# fit to well past four decimal places and the upper end is a within-group fit.
LOG_LAMBDA_BRACKET = (-12.0, 12.0)

# Fixed ratios the held-out score is reported at, so the result does not rest on
# where a noisy optimiser landed. Zero is the pooled fit exactly.
LAMBDA_GRID: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0)


@contextlib.contextmanager
def _suppress_benign_matmul_warnings() -> Iterator[None]:
    """Silence the spurious BLAS floating-point-state warnings ``hdb5`` also hits.

    The design matrix is perfectly collinear in one direction, which makes
    NumPy's ``matmul`` report divide-by-zero and overflow on some backends with
    finite inputs and correct results. Scoped to the fit numerics only.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        yield


@dataclass(frozen=True)
class RandomInterceptFit:
    """A fitted random-intercept model, and what it costs to describe it."""

    coefficients: np.ndarray
    """Fixed effects on the retained columns, intercept first."""
    kept_columns: tuple[int, ...]
    """Indices into the feature list that pivoted QR found independent."""
    variance_ratio: float
    """sigma_u^2 / sigma^2, profiled out by REML unless it was supplied."""
    residual_variance: float
    group_variance: float
    n_groups: int
    n_rows: int

    @property
    def intraclass_correlation(self) -> float:
        """The share of variance that is between devices rather than within one."""
        total = self.group_variance + self.residual_variance
        return float(self.group_variance / total) if total > 0 else 0.0

    def predict_population(self, features: np.ndarray) -> np.ndarray:
        """log tau for rows from a device with no fitted intercept of its own."""
        design = np.column_stack([np.ones(len(features)), features[:, list(self.kept_columns)]])
        with _suppress_benign_matmul_warnings():
            return design @ self.coefficients


def _independent_columns(features: np.ndarray, *, tolerance: float = 1e-9) -> tuple[int, ...]:
    """Column indices of a maximal independent subset, by pivoted QR.

    Selected on the training fold rather than fixed in advance, so this stays
    correct if a fold's rows happen to carry a dependency the whole table does
    not.
    """
    design = np.column_stack([np.ones(len(features)), features])
    _, upper, pivots = linalg.qr(design, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(upper))
    rank = int(np.sum(diagonal > tolerance * diagonal[0])) if diagonal.size else 0
    # Pivot 0 is the intercept column, which is prepended rather than selected.
    return tuple(sorted(int(p) - 1 for p in pivots[:rank] if p != 0))


def _group_slices(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    ordered = groups[order]
    boundaries = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    return [order[boundaries[k] : boundaries[k + 1]] for k in range(len(boundaries) - 1)]


def _whiten(
    design: np.ndarray,
    response: np.ndarray,
    blocks: list[np.ndarray],
    variance_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Group-mean shrinkage that turns the GLS solve into ordinary least squares.

    Returns the transformed design and response together with the log
    determinant of the correlation matrix, which the REML criterion needs and
    which is a sum over groups of log(1 + n_i lambda).
    """
    transformed_design = design.copy()
    transformed_response = response.copy()
    log_determinant = 0.0
    for block in blocks:
        factor = 1.0 + len(block) * variance_ratio
        theta = 1.0 - 1.0 / np.sqrt(factor)
        transformed_design[block] -= theta * design[block].mean(axis=0)
        transformed_response[block] -= theta * response[block].mean()
        log_determinant += float(np.log(factor))
    return transformed_design, transformed_response, log_determinant


def _solve(
    design: np.ndarray,
    response: np.ndarray,
    blocks: list[np.ndarray],
    variance_ratio: float,
) -> tuple[np.ndarray, float, float]:
    """GLS coefficients at a fixed ratio, with the REML criterion at that ratio."""
    with _suppress_benign_matmul_warnings():
        whitened_design, whitened_response, log_determinant = _whiten(
            design, response, blocks, variance_ratio
        )
        coefficients, *_ = linalg.lstsq(whitened_design, whitened_response)
        residual = whitened_response - whitened_design @ coefficients
        degrees = len(response) - design.shape[1]
        scale = float(residual @ residual) / degrees
        _, log_information = np.linalg.slogdet(whitened_design.T @ whitened_design)
    criterion = degrees * np.log(scale) + log_determinant + log_information
    return coefficients, scale, criterion


def fit_random_intercept(
    features: np.ndarray,
    response: np.ndarray,
    groups: np.ndarray,
    *,
    variance_ratio: float | None = None,
) -> RandomInterceptFit:
    """Fit on log features, profiling the variance ratio by REML unless given one."""
    kept = _independent_columns(features)
    design = np.column_stack([np.ones(len(features)), features[:, list(kept)]])
    blocks = _group_slices(groups)

    if variance_ratio is None:
        result = optimize.minimize_scalar(
            lambda log_ratio: _solve(design, response, blocks, float(np.exp(log_ratio)))[2],
            bounds=LOG_LAMBDA_BRACKET,
            method="bounded",
            options={"xatol": 1e-6},
        )
        variance_ratio = float(np.exp(result.x))

    coefficients, residual_variance, _ = _solve(design, response, blocks, variance_ratio)
    return RandomInterceptFit(
        coefficients=coefficients,
        kept_columns=kept,
        variance_ratio=float(variance_ratio),
        residual_variance=residual_variance,
        group_variance=float(variance_ratio) * residual_variance,
        n_groups=len(blocks),
        n_rows=len(response),
    )


def _rmsle(actual: np.ndarray, predicted_log: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted_log - np.log(actual)) ** 2)))


@dataclass(frozen=True)
class UnitFolds:
    """One leave-one-unit-out design, resolved once and reused at every ratio."""

    features: np.ndarray
    tau: np.ndarray
    units: np.ndarray
    eligible: tuple[str, ...]

    @classmethod
    def build(cls, dataset: pd.DataFrame, unit_column: str, feature_columns: tuple[str, ...]) -> "UnitFolds":
        units = dataset[unit_column].to_numpy()
        counts = pd.Series(units).value_counts()
        return cls(
            features=dataset[list(feature_columns)].to_numpy(dtype=float),
            tau=dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float),
            units=units,
            eligible=tuple(sorted(str(u) for u, n in counts.items() if n >= hdb5.MIN_HELD_OUT_ROWS)),
        )

    def score(self, *, variance_ratio: float | None) -> pd.DataFrame:
        """Score every eligible unit, held out of the fit and its variance components."""
        log_tau = np.log(self.tau)
        records = []
        for unit in self.eligible:
            held = self.units == unit
            fit = fit_random_intercept(
                self.features[~held], log_tau[~held], self.units[~held], variance_ratio=variance_ratio
            )
            records.append(
                {
                    "model_name": "mixed_random_intercept",
                    "tokamak": unit,
                    "n_held_out_rows": int(held.sum()),
                    "rmsle": _rmsle(self.tau[held], fit.predict_population(self.features[held])),
                    "variance_ratio": fit.variance_ratio,
                    "intraclass_correlation": fit.intraclass_correlation,
                    "n_training_groups": fit.n_groups,
                }
            )
        return pd.DataFrame(records)


def _paired(mixed: pd.DataFrame, reference: pd.DataFrame, model: str) -> dict[str, Any]:
    """How the mixed model compares to one fitted model, unit by unit."""
    other = reference[reference["model_name"] == model].set_index("tokamak")["rmsle"]
    joined = mixed.set_index("tokamak")["rmsle"].to_frame("mixed").join(other.to_frame("other"), how="inner")
    difference = joined["mixed"] - joined["other"]
    return {
        "against": model,
        "n_units": int(len(joined)),
        "n_mixed_worse": int((difference > 0).sum()),
        "mean_difference": float(difference.mean()),
        "units_where_mixed_wins": sorted(joined.index[difference < 0]),
    }


def _sweep(folds: UnitFolds) -> dict[str, Any]:
    """Held-out score at each fixed ratio, and the best one in hindsight.

    The hindsight row is not a result the paper can claim. It is an upper bound
    on the estimator: no rule for choosing lambda, however good, beats it, so if
    the pooled fit is still at least as good there the comparison is settled
    without needing to argue about how lambda should have been picked.
    """
    means = {}
    for ratio in LAMBDA_GRID:
        scored = folds.score(variance_ratio=ratio)
        means[ratio] = float(scored["rmsle"].mean())
    best = min(means, key=lambda ratio: means[ratio])
    return {
        "grid": [{"variance_ratio": ratio, "mean_rmsle": value} for ratio, value in means.items()],
        "pooled_limit_rmsle": means[0.0],
        "best_variance_ratio": best,
        "best_mean_rmsle": means[best],
    }


def _arm(
    dataset: pd.DataFrame,
    unit_column: str,
    reference: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[dict[str, Any], pd.DataFrame]:
    folds = UnitFolds.build(dataset, unit_column, feature_columns)
    fitted = folds.score(variance_ratio=None)
    ratios = fitted["variance_ratio"]
    return (
        {
            "n_units": int(len(fitted)),
            "reml": {
                "mean_rmsle": float(fitted["rmsle"].mean()),
                "variance_ratio_min": float(ratios.min()),
                "variance_ratio_max": float(ratios.max()),
                "variance_ratio_median": float(ratios.median()),
                "against_power_law": _paired(fitted, reference, POWER_LAW),
                "against_forest": _paired(fitted, reference, FOREST),
            },
            "variance_ratio_sweep": _sweep(folds),
        },
        fitted,
    )


def _size_cut_arm(dataset: pd.DataFrame, feature_columns: tuple[str, ...]) -> dict[str, Any]:
    """The same fit across the ITER-size-matched cut, where every unit is above it."""
    split = hdb5.iter_matched_split(dataset, hdb5.size_ordered_splits(dataset))
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    above = np.isin(labels, list(split.test_machines))

    features = dataset[list(feature_columns)].to_numpy(dtype=float)
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    fitted = fit_random_intercept(features[~above], np.log(tau[~above]), labels[~above])
    pooled = fit_random_intercept(
        features[~above], np.log(tau[~above]), labels[~above], variance_ratio=0.0
    )
    return {
        "size_ratio": split.size_ratio,
        "n_train_rows": int((~above).sum()),
        "n_test_rows": int(above.sum()),
        "reml_rmsle": _rmsle(tau[above], fitted.predict_population(features[above])),
        "reml_variance_ratio": fitted.variance_ratio,
        "pooled_limit_rmsle": _rmsle(tau[above], pooled.predict_population(features[above])),
    }


def analyze(
    dataset: pd.DataFrame, *, feature_columns: tuple[str, ...] = hdb5.BLIND_FEATURE_COLUMNS
) -> dict[str, Any]:
    by_device = hdb5.with_device_column(dataset)
    device_grouped = by_device.copy()
    device_grouped[hdb5.TOKAMAK_LABEL_COLUMN] = device_grouped[hdb5.DEVICE_COLUMN]

    label_arm, per_label = _arm(
        dataset,
        hdb5.TOKAMAK_LABEL_COLUMN,
        hdb5.leave_one_tokamak_out(dataset, feature_columns=feature_columns),
        feature_columns,
    )
    device_arm, per_device = _arm(
        by_device,
        hdb5.DEVICE_COLUMN,
        hdb5.leave_one_tokamak_out(device_grouped, feature_columns=feature_columns),
        feature_columns,
    )

    whole = fit_random_intercept(
        dataset[list(feature_columns)].to_numpy(dtype=float),
        np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)),
        by_device[hdb5.DEVICE_COLUMN].to_numpy(),
    )

    return {
        "feature_columns": list(feature_columns),
        "in_sample": {
            "variance_ratio": whole.variance_ratio,
            "intraclass_correlation": whole.intraclass_correlation,
            "group_sd": float(np.sqrt(whole.group_variance)),
            "residual_sd": float(np.sqrt(whole.residual_variance)),
            "n_groups": whole.n_groups,
            "kept_feature_columns": [feature_columns[i] for i in whole.kept_columns],
        },
        "leave_one_label_out": label_arm,
        "leave_one_device_out": device_arm,
        "iter_matched_cut": _size_cut_arm(dataset, feature_columns),
        "per_label": per_label,
        "per_device": per_device,
    }


def _report(payload: dict[str, Any]) -> None:
    sample = payload["in_sample"]
    print(
        f"in sample: variance ratio {sample['variance_ratio']:.3f}, "
        f"ICC {sample['intraclass_correlation']:.3f}, "
        f"device sd {sample['group_sd']:.4f} against residual sd {sample['residual_sd']:.4f}"
    )
    for name in ("leave_one_label_out", "leave_one_device_out"):
        arm = payload[name]
        reml = arm["reml"]
        against = reml["against_power_law"]
        sweep = arm["variance_ratio_sweep"]
        print(f"\n{name} over {arm['n_units']} units")
        print(
            f"  REML       {reml['mean_rmsle']:.4f}   ratio {reml['variance_ratio_min']:.2f} to "
            f"{reml['variance_ratio_max']:.2f} (median {reml['variance_ratio_median']:.2f})"
        )
        print(
            f"             worse than the power law on {against['n_mixed_worse']} of "
            f"{against['n_units']}, mean gap {against['mean_difference']:+.4f}"
        )
        for row in sweep["grid"]:
            print(f"  ratio {row['variance_ratio']:7.2f}   {row['mean_rmsle']:.4f}")
        print(
            f"  best in hindsight: ratio {sweep['best_variance_ratio']} at "
            f"{sweep['best_mean_rmsle']:.4f}, against {sweep['pooled_limit_rmsle']:.4f} pooled"
        )
    cut = payload["iter_matched_cut"]
    print(
        f"\nITER-size-matched cut (ratio {cut['size_ratio']:.3f}): REML {cut['reml_rmsle']:.4f}, "
        f"pooled limit {cut['pooled_limit_rmsle']:.4f}"
    )


def main() -> None:
    dataset = hdb5.prepare_dataset()
    payload = analyze(dataset)
    _report(payload)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(RESULTS_DIR / "mixed_model_per_label.csv", payload.pop("per_label"))
    write_dataframe_csv_atomic(RESULTS_DIR / "mixed_model_per_device.csv", payload.pop("per_device"))
    payload["dataset_sha256"] = hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256
    write_json_strict(RESULTS_DIR / "mixed_model.json", payload)
    print(f"\nwrote {RESULTS_DIR / 'mixed_model.json'}")


if __name__ == "__main__":
    main()
