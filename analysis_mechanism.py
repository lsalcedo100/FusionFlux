"""Two experiments that make the saturation mechanism exact instead of inferred.

Run ``python3 analysis_mechanism.py`` to regenerate ``results/mechanism.json``.

Sec.~gp compares an RBF kernel against linear-plus-RBF, and adding the linear
term enlarges the function class as well as changing what the model does at
long range. So that comparison suggests the mechanism without isolating it.
This script isolates it two ways.

    mean function   Hold the nonlinear component *literally* fixed: the same
                    RBF Gaussian process, the same kernel, the same tuner, the
                    same seed, fitted on residuals. Change only the mean it is a
                    residual of. A constant mean gives a model that reverts to
                    the training average far from the data; a fitted power-law
                    mean gives one that keeps trending. Nothing else differs, so
                    whatever separates them is the trend and not the capacity.

    clipping        Sec.~hybrid's corrector is a gradient booster, which is
                    bounded on these rows but not by construction. Clipping its
                    output to the training residual range makes the bound a
                    theorem. If the clip never binds, the empirical claim and
                    the provable one describe the same model, and the section's
                    mechanism can be stated without the hedge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import gp as gp_module
import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Depth and damping as Sec. hybrid uses them, so the clipped arm differs from
# the reported hybrid in exactly one respect.
CORRECTION_DEPTH = 2
CORRECTION_DAMPING = 1.0


class MeanPlusResidualGP(RegressorMixin, BaseEstimator):
    """A mean function plus the same RBF Gaussian process on its residuals.

    ``mean`` is either ``"constant"``, which centres the target and lets the GP
    carry everything, or ``"powerlaw"``, which fits ridge on the log features
    first and hands the GP only what the power law missed.

    The GP is identical in both cases: same kernel name, same tuning subsample,
    same seed. That is the point. Everything the two arms could differ by has
    been held fixed except whether the model has a trend that continues once the
    input leaves the training data.
    """

    def __init__(self, mean: str = "powerlaw", *, random_state: int = gp_module.RANDOM_STATE):
        self.mean = mean
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> MeanPlusResidualGP:
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)

        if self.mean == "powerlaw":
            self.mean_model_ = Ridge(alpha=1.0, solver="svd").fit(features, target)
            base = self.mean_model_.predict(features)
        elif self.mean == "constant":
            self.mean_model_ = None
            base = np.full_like(target, float(target.mean()))
        else:
            raise ValueError(f"Unknown mean {self.mean!r}; expected 'constant' or 'powerlaw'")

        self.residual_gp_ = gp_module.SubsampledGaussianProcess(kernel_name="rbf", random_state=self.random_state).fit(
            features, target - base
        )
        self.training_mean_ = float(target.mean())
        return self

    def _base(self, features: np.ndarray) -> np.ndarray:
        if self.mean_model_ is None:
            return np.full(len(features), self.training_mean_)
        return cast("np.ndarray", self.mean_model_.predict(features))

    def predict(self, X: Any) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        return self._base(features) + np.asarray(self.residual_gp_.predict(features), dtype=float)


class ClippedResidualHybrid(RegressorMixin, BaseEstimator):
    """Power law plus a boosted-tree correction clipped to its training range.

    ``clip=False`` reproduces Sec. hybrid's corrector. ``clip=True`` confines the
    correction to the interval of residuals it was trained on, which turns a
    property the gradient booster happens to have on these rows into one it
    cannot violate. ``clip_active_fraction`` records whether that ever mattered.
    """

    def __init__(self, *, clip: bool = True, damping: float = CORRECTION_DAMPING):
        self.clip = clip
        self.damping = damping

    def fit(self, X: Any, y: Any) -> ClippedResidualHybrid:
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        self.base_ = Ridge(alpha=1.0, solver="svd").fit(features, target)
        residuals = target - self.base_.predict(features)
        self.correction_ = HistGradientBoostingRegressor(
            max_depth=CORRECTION_DEPTH, random_state=hdb5.RANDOM_STATE
        ).fit(features, residuals)
        self.residual_low_ = float(residuals.min())
        self.residual_high_ = float(residuals.max())
        self.clip_active_fraction_ = 0.0
        return self

    def predict(self, X: Any) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        correction = np.asarray(self.correction_.predict(features), dtype=float)
        if self.clip:
            outside = (correction < self.residual_low_) | (correction > self.residual_high_)
            self.clip_active_fraction_ = float(outside.mean())
            correction = np.clip(correction, self.residual_low_, self.residual_high_)
        else:
            self.clip_active_fraction_ = float(
                ((correction < self.residual_low_) | (correction > self.residual_high_)).mean()
            )
        base = cast("np.ndarray", self.base_.predict(features))
        return base + self.damping * correction


def _rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log(predicted) - np.log(actual)) ** 2)))


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def score_everywhere(dataset: pd.DataFrame, models: dict[str, Any]) -> dict[str, Any]:
    """Grouped CV, leave-one-machine-out and the ITER-matched cut, per model."""
    columns = list(hdb5.BLIND_FEATURE_COLUMNS)
    features = dataset[columns]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    eligible = hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS)
    cut = hdb5.iter_matched_split(dataset, hdb5.size_ordered_splits(dataset))
    above = np.isin(labels, list(cut.test_machines))

    out: dict[str, Any] = {}
    for name, estimator in models.items():
        pipeline = _pipeline(estimator)
        n_splits = min(hdb5.N_CV_FOLDS, int(pd.Series(groups).nunique()))
        with hdb5._suppress_benign_matmul_warnings():
            cv = np.exp(hdb5._grouped_cv_predictions(pipeline, features, log_tau, groups, n_splits))

        per_machine: dict[str, float] = {}
        for machine in eligible:
            held = labels == machine
            model = hdb5.clone_pipeline(pipeline)
            with hdb5._suppress_benign_matmul_warnings():
                hdb5.fit_pipeline(model, features[~held], log_tau[~held])
                per_machine[str(machine)] = _rmsle(tau[held], np.exp(model.predict(features[held])))

        cut_model = hdb5.clone_pipeline(pipeline)
        with hdb5._suppress_benign_matmul_warnings():
            hdb5.fit_pipeline(cut_model, features[~above], log_tau[~above])
            cut_predicted = np.exp(cut_model.predict(features[above]))

        row: dict[str, Any] = {
            "cv": _rmsle(tau, cv),
            "leave_one_machine_out": float(np.mean(list(per_machine.values()))),
            "iter_matched_cut": _rmsle(tau[above], cut_predicted),
            "per_machine": per_machine,
        }
        fitted = cut_model.named_steps["model"]
        if hasattr(fitted, "clip_active_fraction_"):
            row["clip_active_fraction_at_cut"] = float(fitted.clip_active_fraction_)
        out[name] = row
    return out


def main() -> None:
    dataset = hdb5.prepare_dataset()

    mean_function = score_everywhere(
        dataset,
        {
            "constant mean + RBF residual": MeanPlusResidualGP(mean="constant"),
            "power-law mean + RBF residual": MeanPlusResidualGP(mean="powerlaw"),
        },
    )
    clipping = score_everywhere(
        dataset,
        {
            "residual correction, unclipped": ClippedResidualHybrid(clip=False),
            "residual correction, clipped": ClippedResidualHybrid(clip=True),
        },
    )

    analysis: dict[str, Any] = {
        "n_rows": int(len(dataset)),
        "mean_function": mean_function,
        "clipping": clipping,
        "correction_depth": CORRECTION_DEPTH,
        "correction_damping": CORRECTION_DAMPING,
    }
    write_json_strict(RESULTS_DIR / "mechanism.json", analysis)

    for title, arm in (("mean function", mean_function), ("clipping", clipping)):
        print(f"\n--- {title} ---")
        for name, row in arm.items():
            extra = (
                f"  clip active at cut={row['clip_active_fraction_at_cut']:.4f}"
                if "clip_active_fraction_at_cut" in row
                else ""
            )
            print(
                f"  {name:32s} CV={row['cv']:.4f}  LOMO={row['leave_one_machine_out']:.4f}  "
                f"ITER cut={row['iter_matched_cut']:.4f}{extra}"
            )
    print(f"\nWrote {RESULTS_DIR / 'mechanism.json'}")


if __name__ == "__main__":
    main()
