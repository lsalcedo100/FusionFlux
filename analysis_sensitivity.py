"""Four sensitivity checks the main results do not run, and what they change.

Run ``python3 analysis_sensitivity.py`` to regenerate ``results/sensitivity.json``.

Each arm answers one objection that the headline comparison invites:

    ITPA20      IPB98(y,2) is the ITER reference, but it is not the newest
                published scaling fitted to this database family. ITPA20 and
                ITPA20-IL are, and they weaken the size dependence sharply.
                Both are non-blind here for the same reason IPB98 is.

    correlation The claim that tree error tracks extrapolation distance rests on
                a Spearman rho over 13 machines. This attaches a permutation
                p-value and a leave-one-machine-out range to each one, so a
                rho carried by a single extreme device would show.

    weighting   Every fit weights rows equally, so JET and ASDEX Upgrade set
                most of what a model learns while the target is a device that
                resembles neither. Refit with each machine weighted equally.

    errors      The power-law fits are ordinary least squares, which assumes the
                engineering parameters are measured without error. Orthogonal
                distance regression does not, and the confinement literature has
                long argued the difference matters for the exponents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

import hdb5
from storage import write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONTENDERS = ("ridge_loglinear", "random_forest", "hist_gradient_boosting")

# Published scalings, as exponents on (Ip [MA], Bt [T], n [1e19 m^-3],
# P [MW], R [m], 1+delta, kappa, epsilon, M). Both are cross-checked against
# the UKAEA PROCESS systems-code documentation and, for ITPA20-IL, against an
# independent transcription; neither is fitted here, and both saw every machine
# in this database when they were derived.
PUBLISHED_SCALINGS: dict[str, dict[str, float]] = {
    "ITPA20": {
        "coefficient": 0.053,
        "ip_ma": 0.98,
        "bt_t": 0.22,
        "ne_line_1e19_m3": 0.24,
        "p_loss_mw": -0.669,
        "r_m": 1.71,
        "one_plus_delta": 0.36,
        "kappa": 0.80,
        "inverse_aspect_ratio": 0.35,
        "m_eff_amu": 0.20,
    },
    "ITPA20-IL": {
        "coefficient": 0.067,
        "ip_ma": 1.29,
        "bt_t": -0.13,
        "ne_line_1e19_m3": 0.15,
        "p_loss_mw": -0.644,
        "r_m": 1.19,
        "one_plus_delta": 0.56,
        "kappa": 0.67,
        "inverse_aspect_ratio": 0.0,
        "m_eff_amu": 0.30,
    },
}

PERMUTATION_DRAWS = 20000
PERMUTATION_SEED = 20240902


def _rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log(predicted) - np.log(actual)) ** 2)))


def dataset_with_triangularity() -> pd.DataFrame:
    """The analysed frame, plus the DELTA1 column the ITPA20 scalings need.

    ``map_to_canonical`` drops every column it does not name, and triangularity
    is not one of the nine engineering features, so it has to be carried across
    by reproducing the same cleaning mask rather than by joining on an index
    that ``reset_index`` has already discarded.
    """
    raw = hdb5.load_hdb5_dataframe()
    frame = pd.DataFrame(index=raw.index)
    for canonical, (source, take_abs) in hdb5.CANONICAL_COLUMN_SOURCES.items():
        values = pd.to_numeric(raw[source], errors="coerce")
        frame[canonical] = values.abs() if take_abs else values
    frame["a_m"] = frame["inverse_aspect_ratio"] * frame["r_m"]

    positive = [hdb5.TARGET_COLUMN, *hdb5.BASE_ENGINEERING_COLUMNS]
    keep = frame[positive].notna().all(axis=1) & (frame[positive] > 0).all(axis=1)

    dataset = hdb5.prepare_dataset()
    delta = pd.to_numeric(raw.loc[keep, "DELTA1"], errors="coerce").reset_index(drop=True)
    if len(delta) != len(dataset):
        raise AssertionError(
            f"triangularity column has {len(delta)} rows against {len(dataset)} analysed; "
            "the cleaning mask reproduced here has drifted from map_to_canonical"
        )
    dataset = dataset.copy()
    dataset["one_plus_delta"] = 1.0 + delta
    return dataset


def published_prediction(dataset: pd.DataFrame, name: str) -> np.ndarray:
    """Evaluate one published scaling analytically. No fitting of any kind."""
    law = PUBLISHED_SCALINGS[name]
    tau = np.full(len(dataset), float(law["coefficient"]))
    for column, exponent in law.items():
        if column == "coefficient" or exponent == 0.0:
            continue
        tau = tau * dataset[column].to_numpy(dtype=float) ** exponent
    return tau


def score_published(dataset: pd.DataFrame) -> dict[str, Any]:
    """Each published law under the three splits the fitted models are scored on."""
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    eligible = hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS)
    splits = hdb5.size_ordered_splits(dataset)
    cut = hdb5.iter_matched_split(dataset, splits)
    above_cut = np.isin(labels, list(cut.test_machines))

    out: dict[str, Any] = {}
    for name in (*PUBLISHED_SCALINGS, "IPB98(y,2)"):
        predicted = (
            dataset["ipb98y2_tau_s"].to_numpy(dtype=float)
            if name == "IPB98(y,2)"
            else published_prediction(dataset, name)
        )
        per_machine = {str(m): _rmsle(tau[labels == m], predicted[labels == m]) for m in eligible}
        out[name] = {
            "is_blind": False,
            "all_rows": _rmsle(tau, predicted),
            "machine_equal": float(np.mean(list(per_machine.values()))),
            "iter_matched_cut": _rmsle(tau[above_cut], predicted[above_cut]),
            "per_machine": per_machine,
        }
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank_a = pd.Series(a).rank().to_numpy()
    rank_b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def correlation_uncertainty(per_machine: pd.DataFrame) -> dict[str, Any]:
    """Permutation p-value and leave-one-machine-out range for each rho."""
    rng = np.random.default_rng(PERMUTATION_SEED)
    out: dict[str, Any] = {}
    for name, rows in per_machine.groupby("model_name"):
        error = rows["rmsle"].to_numpy(dtype=float)
        distance = rows["feature_mahalanobis"].to_numpy(dtype=float)
        observed = _spearman(error, distance)

        null = np.array([_spearman(error, rng.permutation(distance)) for _ in range(PERMUTATION_DRAWS)])
        jackknife = [_spearman(np.delete(error, i), np.delete(distance, i)) for i in range(len(error))]
        out[str(name)] = {
            "n_machines": int(len(error)),
            "spearman": observed,
            "permutation_p_two_sided": float(np.mean(np.abs(null) >= abs(observed))),
            "jackknife_min": float(np.min(jackknife)),
            "jackknife_max": float(np.max(jackknife)),
        }
    return out


def machine_equal_weighting(dataset: pd.DataFrame) -> dict[str, Any]:
    """Leave-one-machine-out with every training machine weighted equally.

    Only the linear model takes a sample weight here. The tree ensembles accept
    one too, but the question this arm asks is whether the *power law* is an
    artifact of JET and AUG supplying most of the rows, and the trees' failure
    is already known to be structural rather than a weighting effect.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    columns = list(hdb5.BLIND_FEATURE_COLUMNS)
    features = dataset[columns]
    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    eligible = hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS)

    per_machine: dict[str, dict[str, float]] = {"unweighted": {}, "machine_equal": {}}
    for machine in eligible:
        held = labels == machine
        train = np.flatnonzero(~held)
        test = np.flatnonzero(held)
        counts = pd.Series(labels[train]).value_counts()
        weights = np.array([1.0 / counts[label] for label in labels[train]])
        weights = weights * (len(train) / weights.sum())

        for arm, weight in (("unweighted", None), ("machine_equal", weights)):
            model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0, solver="svd"))])
            with hdb5._suppress_benign_matmul_warnings():
                model.fit(features.iloc[train], log_tau[train], model__sample_weight=weight)
                predicted = np.exp(model.predict(features.iloc[test]))
            per_machine[arm][str(machine)] = _rmsle(tau[test], predicted)

    return {
        arm: {
            "machine_equal_rmsle": float(np.mean(list(scores.values()))),
            "per_machine": scores,
        }
        for arm, scores in per_machine.items()
    }


def errors_in_variables(dataset: pd.DataFrame) -> dict[str, Any]:
    """Refit Eq. 1 by orthogonal distance regression and compare the exponents.

    OLS puts all the error on tau. ODR distributes it across the predictors too,
    which is the objection the confinement-scaling literature raises against
    least-squares exponents. With equal weights in log space this amounts to
    assuming comparable relative uncertainty on every variable, which is an
    assumption rather than a measurement, so this is a sensitivity check and not
    a corrected fit.
    """
    from scipy import odr

    import scaling_law as sl

    columns = list(sl.IPB98_FEATURE_COLUMNS)
    # Eq. 1 is linear in the *logs*, which is the space both fits live in.
    design = np.log(dataset[columns].to_numpy(dtype=float)).T
    log_tau = np.log(dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float))

    def model(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return beta[0] + beta[1:] @ x

    ols = np.linalg.lstsq(np.column_stack([np.ones(design.shape[1]), design.T]), log_tau, rcond=None)[0]
    fitted = odr.ODR(odr.Data(design, log_tau), odr.Model(model), beta0=ols.copy()).run()

    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    predicted = np.exp(model(fitted.beta, design))
    return {
        "feature_columns": columns,
        "ols_exponents": {c: float(v) for c, v in zip(columns, ols[1:], strict=True)},
        "odr_exponents": {c: float(v) for c, v in zip(columns, fitted.beta[1:], strict=True)},
        "max_abs_exponent_shift": float(np.max(np.abs(fitted.beta[1:] - ols[1:]))),
        "largest_shift_feature": columns[int(np.argmax(np.abs(fitted.beta[1:] - ols[1:])))],
        "odr_in_sample_rmsle": _rmsle(tau, predicted),
    }


def main() -> None:
    dataset = dataset_with_triangularity()
    report = hdb5.extrapolation_report(hdb5.prepare_dataset())

    analysis: dict[str, Any] = {
        "n_rows": int(len(dataset)),
        "published_scalings": score_published(dataset),
        "correlation_uncertainty": correlation_uncertainty(report),
        "machine_equal_weighting": machine_equal_weighting(dataset),
        "errors_in_variables": errors_in_variables(dataset),
        "discharge_disjoint": discharge_disjoint_arm(),
        "redundant_feature": redundant_feature(dataset),
        "permutation_draws": PERMUTATION_DRAWS,
    }
    write_json_strict(RESULTS_DIR / "sensitivity.json", analysis)

    print("--- published scalings, none of them blind ---")
    for name, row in cast("dict[str, Any]", analysis["published_scalings"]).items():
        print(
            f"  {name:12s} all rows={row['all_rows']:.4f}  "
            f"machine-equal={row['machine_equal']:.4f}  ITER cut={row['iter_matched_cut']:.4f}"
        )
    print("\n--- error against extrapolation distance, 13 machines ---")
    for name, row in cast("dict[str, Any]", analysis["correlation_uncertainty"]).items():
        print(
            f"  {name:24s} rho={row['spearman']:+.2f}  p={row['permutation_p_two_sided']:.4f}  "
            f"jackknife [{row['jackknife_min']:+.2f}, {row['jackknife_max']:+.2f}]"
        )
    weighting = cast("dict[str, Any]", analysis["machine_equal_weighting"])
    print("\n--- power law, leave-one-machine-out ---")
    for arm, row in weighting.items():
        print(f"  {arm:16s} {row['machine_equal_rmsle']:.4f}")
    eiv = cast("dict[str, Any]", analysis["errors_in_variables"])
    print(
        f"\n--- errors in variables: largest exponent shift "
        f"{eiv['max_abs_exponent_shift']:.3f} on {eiv['largest_shift_feature']} ---"
    )
    for column in eiv["feature_columns"]:
        print(f"  {column:26s} OLS {eiv['ols_exponents'][column]:+.3f}   ODR {eiv['odr_exponents'][column]:+.3f}")
    print(f"\nWrote {RESULTS_DIR / 'sensitivity.json'}")


def discharge_disjoint_arm() -> dict[str, Any]:
    """The H-mode robustness arm with every discharge STD5 samples removed.

    Sec. replication removes rows that STD5 contains. That leaves 7.8% of the
    arm sitting in a discharge STD5 samples at a different time, which is row
    disjointness without discharge disjointness, and the whole paper argues that
    the cluster is the unit that matters. This drops the shared discharges
    outright and rescores, so the arm has no shot in common with STD5 at all.
    """
    import replication as rep

    raw = rep.load_db523_raw()
    std5 = hdb5.load_hdb5_dataframe()
    std5_rows = set(rep._match_keys(std5))
    std5_shots = set(std5["TOK"].astype(str) + "|" + std5["SHOT"].astype(str))

    phase = raw["PHASE"].astype(str)
    shots = raw["TOK"].astype(str) + "|" + raw["SHOT"].astype(str)
    row_disjoint = phase.str.startswith("H") & ~rep._match_keys(raw).isin(std5_rows)
    shot_disjoint = row_disjoint & ~shots.isin(std5_shots)

    out: dict[str, Any] = {}
    for label, mask in (("row_disjoint", row_disjoint), ("discharge_disjoint", shot_disjoint)):
        dataset = rep.prepare_db523_frame(raw.loc[mask])
        tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
        labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
        eligible = hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS)

        features = dataset[list(hdb5.BLIND_FEATURE_COLUMNS)]
        groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
        n_splits = min(hdb5.N_CV_FOLDS, int(pd.Series(groups).nunique()))
        zoo = hdb5._assemble_zoo()

        scores: dict[str, Any] = {}
        for name in (*CONTENDERS, "IPB98(y,2)"):
            if name == "IPB98(y,2)":
                cv_pred = dataset["ipb98y2_tau_s"].to_numpy(dtype=float)
                per_machine = {str(m): _rmsle(tau[labels == m], cv_pred[labels == m]) for m in eligible}
            else:
                cv_pred = np.exp(hdb5._grouped_cv_predictions(zoo[name], features, np.log(tau), groups, n_splits))
                per_machine = {}
                for machine in eligible:
                    held = labels == machine
                    model = hdb5.clone_pipeline(zoo[name])
                    with hdb5._suppress_benign_matmul_warnings():
                        hdb5.fit_pipeline(model, features[~held], np.log(tau)[~held])
                        per_machine[str(machine)] = _rmsle(tau[held], np.exp(model.predict(features[held])))
            scores[name] = {
                "cv": _rmsle(tau, cv_pred),
                "leave_one_machine_out": float(np.mean(list(per_machine.values()))),
            }
        best_cv = min(scores, key=lambda n: scores[n]["cv"])
        best_lomo = min(scores, key=lambda n: scores[n]["leave_one_machine_out"])
        out[label] = {
            "n_rows": int(len(dataset)),
            "n_discharges": int(dataset[hdb5.GROUP_COLUMN].nunique()),
            "n_machines_scored": len(eligible),
            "scores": scores,
            "best_cv_model": best_cv,
            "best_lomo_model": best_lomo,
            "reversal_holds": best_cv != best_lomo and best_cv in CONTENDERS,
            "cv_gain_over_baseline": float(1.0 - scores[best_cv]["cv"] / scores["IPB98(y,2)"]["cv"]),
        }
    return out


def redundant_feature(dataset: pd.DataFrame) -> dict[str, Any]:
    """The same models on 8 independent features instead of the redundant 9.

    Minor radius is exactly ``epsilon * R``, so the nine-column feature set has
    an exact dependency. OLS is invariant to that, but ridge penalises in the
    coordinates it is handed, so the redundancy is not free for the model the
    paper leans on hardest. Dropping ``log_a_m`` removes it.
    """
    full = tuple(hdb5.BLIND_FEATURE_COLUMNS)
    trimmed = tuple(c for c in full if c != "log_a_m")
    if len(trimmed) != len(full) - 1:
        raise AssertionError("log_a_m is not in the blind feature set; nothing was dropped")

    tau = dataset[hdb5.TARGET_COLUMN].to_numpy(dtype=float)
    log_tau = np.log(tau)
    groups = dataset[hdb5.GROUP_COLUMN].to_numpy()
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    eligible = hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS)
    cut = hdb5.iter_matched_split(dataset, hdb5.size_ordered_splits(dataset))
    above = np.isin(labels, list(cut.test_machines))

    out: dict[str, Any] = {}
    for label, columns in (("nine_features", full), ("eight_features", trimmed)):
        features = dataset[list(columns)]
        n_splits = min(hdb5.N_CV_FOLDS, int(pd.Series(groups).nunique()))
        zoo = hdb5._assemble_zoo()
        scores: dict[str, Any] = {}
        for name in CONTENDERS:
            with hdb5._suppress_benign_matmul_warnings():
                cv = np.exp(hdb5._grouped_cv_predictions(zoo[name], features, log_tau, groups, n_splits))
            per_machine = {}
            for machine in eligible:
                held = labels == machine
                model = hdb5.clone_pipeline(zoo[name])
                with hdb5._suppress_benign_matmul_warnings():
                    hdb5.fit_pipeline(model, features[~held], log_tau[~held])
                    per_machine[str(machine)] = _rmsle(tau[held], np.exp(model.predict(features[held])))
            cut_model = hdb5.clone_pipeline(zoo[name])
            with hdb5._suppress_benign_matmul_warnings():
                hdb5.fit_pipeline(cut_model, features[~above], log_tau[~above])
                cut_pred = np.exp(cut_model.predict(features[above]))
            scores[name] = {
                "cv": _rmsle(tau, cv),
                "leave_one_machine_out": float(np.mean(list(per_machine.values()))),
                "iter_matched_cut": _rmsle(tau[above], cut_pred),
            }
        out[label] = {"n_features": len(columns), "scores": scores}
    return out


if __name__ == "__main__":
    main()
