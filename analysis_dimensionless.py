"""Result 19: where the held-out machines sit in dimensionless space.

Sec. 4.1 measures how far a held-out label is from the training data as a
Mahalanobis distance between mean log *engineering* vectors, and finds the random
forest's error tracking it at rho = +0.85 where the power law's does not. Hall
et al. raise the obvious objection to organising anything by engineering
variables: the weak size dependence in this database may come from a subset
localised in *dimensionless* space rather than from size as such, in which case a
size-ordered cut is a cut across something else wearing a size label.

That objection was answered with a caveat and is answered here with a
measurement, because everything it needs is already in the repository.
``dimensional.py`` derives the Connor--Taylor constraints from the scalings of
rho*, beta and nu*, so the group definitions the constraints rest on are written
down in code, and the stored energy that fixes a temperature came in with
``replication.db523_columns``. The same three groups therefore place every row in
dimensionless space, and the same distance construction as Sec. 4.1 places every
label.

    rho* ~ T^(1/2) B^-1 L^-1,   beta ~ n T B^-2,   nu* ~ n L T^-2

with the temperature recovered from the thermal stored energy,
``T ~ W_th / (n V)`` and ``V ~ R a^2 kappa``, and the length scale taken as the
minor radius. Multiplicative constants are dropped throughout and none of them
can matter: every quantity below is a difference of mean log-vectors, and a
constant offset in a log cancels exactly. What is compared is where machines sit
relative to each other, which is the whole of what the objection is about.

Two questions, both of which have an answer that would embarrass this paper:

* Does the forest's per-label error track dimensionless distance as well as it
  tracks engineering distance? If it tracks it *better*, Sec. 4.1 is measuring a
  shadow of the real thing and should say so.
* Is the ITER-size-matched cut a cut in dimensionless space too? If the two
  sides overlap there, the cut is a size boundary and Hall's reading does not
  apply to it. If they separate, the cut is confounded and every number resting
  on it inherits the confound.

Run ``python3 analysis_dimensionless.py`` to regenerate
``results/dimensionless.json`` and its per-label table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hdb5
import replication as rep
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

GROUP_COLUMNS: tuple[str, ...] = ("log_rho_star", "log_beta", "log_nu_star")

POWER_LAW = "ridge_loglinear"
FOREST = "random_forest"
BOOSTER = "hist_gradient_boosting"


def with_dimensionless_groups(
    dataset: pd.DataFrame, *, db523_path: Path | str | None = None
) -> pd.DataFrame:
    """The analysed rows, plus log rho*, log beta and log nu* up to constants.

    The temperature is not a delivered column and is recovered from the thermal
    stored energy, which is why this needs the DB5.2.3 join rather than STD5
    alone. Rows with no stored energy are dropped rather than imputed: a machine
    placed in dimensionless space by a guess is worse than a machine left out.
    """
    stored = pd.to_numeric(
        rep.db523_columns(("WTH",), db523_path=db523_path)["WTH"], errors="coerce"
    )
    framed = dataset.assign(w_th_j=stored)
    framed = framed[framed["w_th_j"].notna() & (framed["w_th_j"] > 0)].reset_index(drop=True)

    log_w = np.log(framed["w_th_j"].to_numpy(dtype=float))
    log_n = framed["log_ne_line_1e19_m3"].to_numpy(dtype=float)
    log_b = framed["log_bt_t"].to_numpy(dtype=float)
    log_r = framed["log_r_m"].to_numpy(dtype=float)
    log_a = framed["log_a_m"].to_numpy(dtype=float)
    log_kappa = framed["log_kappa"].to_numpy(dtype=float)

    # V ~ R a^2 kappa, and T ~ W_th / (n V). Constants drop out downstream.
    log_volume = log_r + 2.0 * log_a + log_kappa
    log_t = log_w - log_n - log_volume

    return framed.assign(
        log_temperature=log_t,
        log_rho_star=0.5 * log_t - log_b - log_a,
        log_beta=log_n + log_t - 2.0 * log_b,
        log_nu_star=log_n + log_a - 2.0 * log_t,
    )


def _mahalanobis_of_mean(train: np.ndarray, held_out: np.ndarray) -> float:
    """Sec. 4.1's construction, on whichever coordinates it is handed.

    Deliberately the same arithmetic as ``hdb5._mahalanobis_of_mean``, including
    the pseudo-inverse, so that a dimensionless distance and an engineering
    distance are the same measurement in different coordinates and the two
    correlations below are comparable. The dimensionless covariance is not
    singular, but using a different inverse for it would make the comparison a
    comparison of two constructions rather than of two coordinate systems.
    """
    difference = held_out.mean(axis=0) - train.mean(axis=0)
    covariance = np.cov(train, rowvar=False)
    return float(np.sqrt(difference @ np.linalg.pinv(np.atleast_2d(covariance)) @ difference))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranked_a = pd.Series(a).rank().to_numpy()
    ranked_b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ranked_a, ranked_b)[0, 1])


def label_distances(framed: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    """Each eligible label's distance from the rest, in the given coordinates."""
    labels = framed[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    features = framed[list(columns)].to_numpy(dtype=float)
    out = {}
    for machine in hdb5.eligible_tokamaks(framed, min_rows=hdb5.MIN_HELD_OUT_ROWS):
        held = labels == machine
        out[str(machine)] = _mahalanobis_of_mean(features[~held], features[held])
    return out


WITHIN_DEVICE_FEATURES: tuple[str, ...] = (
    "log_ip_ma",
    "log_bt_t",
    "log_ne_line_1e19_m3",
    "log_p_loss_mw",
)

DIMENSIONLESS_TARGET = "b_tau"

DIMENSIONLESS_FEATURES: tuple[str, ...] = (
    "log_rho_star",
    "log_beta",
    "log_nu_star",
    "log_q_cyl",
    "log_inverse_aspect_ratio",
    "log_kappa",
    "log_m_eff_amu",
)


def within_device_variance(framed: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    """What share of each coordinate's log variance survives fixing the device.

    A column near zero is a device name written as a number, and a model handed
    one can identify the machine instead of describing it.
    """
    devices = hdb5.with_device_column(framed)[hdb5.DEVICE_COLUMN]
    out = {}
    for column in columns:
        values = framed[column]
        total = float(values.var())
        residual = float((values - values.groupby(devices).transform("mean")).var())
        out[column] = residual / total if total > 0 else 0.0
    return out


def device_recoverable_from(framed: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    """How well the device can be read off the features, under the paper's own split."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

    devices = hdb5.with_device_column(framed)[hdb5.DEVICE_COLUMN].to_numpy()
    features = framed[list(columns)].to_numpy(dtype=float)
    groups = framed[hdb5.GROUP_COLUMN].to_numpy()
    with hdb5._suppress_benign_matmul_warnings():
        predicted = cross_val_predict(
            RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
            features,
            devices,
            groups=groups,
            cv=GroupKFold(n_splits=hdb5.N_CV_FOLDS),
        )
    counts = pd.Series(devices).value_counts(normalize=True)
    return {
        "accuracy": float((predicted == devices).mean()),
        "majority_baseline": float(counts.iloc[0]),
    }


def _reversal_arm(
    framed: pd.DataFrame, columns: tuple[str, ...], *, target: str | None = None
) -> dict[str, Any]:
    """The headline comparison, on whichever features and target it is handed."""
    scored = framed.copy()
    if target is not None:
        scored[hdb5.TARGET_COLUMN] = scored[target]
    by_device = hdb5.with_device_column(scored).copy()
    by_device[hdb5.TOKAMAK_LABEL_COLUMN] = by_device[hdb5.DEVICE_COLUMN]

    cross_validated = {
        s.model_name: float(s.cv_rmsle)
        for s in hdb5.evaluate_models(scored, feature_columns=columns)
    }
    per_label = hdb5.leave_one_tokamak_out(
        scored, feature_columns=columns, include_ipb98_reference=False
    )
    per_device = hdb5.leave_one_tokamak_out(
        by_device, feature_columns=columns, include_ipb98_reference=False
    )

    def paired(per_unit: pd.DataFrame) -> dict[str, Any]:
        wide = per_unit.pivot(index="tokamak", columns="model_name", values="rmsle")
        difference = (wide[FOREST] - wide[POWER_LAW]).dropna()
        return {
            "n_units": int(len(difference)),
            "n_forest_worse": int((difference > 0).sum()),
            "mean_difference": float(difference.mean()),
            "forest_mean_rmsle": float(wide[FOREST].mean()),
            "power_law_mean_rmsle": float(wide[POWER_LAW].mean()),
        }

    return {
        "n_features": len(columns),
        "features": list(columns),
        "target": target or hdb5.TARGET_COLUMN,
        "cv_rmsle": cross_validated,
        "cv_gain_of_forest": float(1.0 - cross_validated[FOREST] / cross_validated[POWER_LAW]),
        "by_label": paired(per_label),
        "by_device": paired(per_device),
    }


def with_dimensionless_regression_columns(framed: pd.DataFrame) -> pd.DataFrame:
    """Add the cylindrical safety factor and the dimensionless confinement time.

    ``q_cyl ~ a^2 B kappa / (R Ip)`` completes the group list the field regresses
    in, and it moves 70% within a device, so it is not another device name.
    ``B tau_E`` is the dimensionless confinement time up to the ion mass and
    charge, which the isotope column carries separately.
    """
    return framed.assign(
        log_q_cyl=(
            2.0 * framed["log_a_m"]
            + framed["log_bt_t"]
            + framed["log_kappa"]
            - framed["log_r_m"]
            - framed["log_ip_ma"]
        ),
        **{DIMENSIONLESS_TARGET: framed["bt_t"] * framed[hdb5.TARGET_COLUMN]},
    )


def device_identity_arms(framed: pd.DataFrame) -> dict[str, Any]:
    """Is the reversal an artifact of the features naming the machine?

    Three arms, and the middle one is reported precisely because it cannot
    answer the question on its own: deleting the device-constant columns deletes
    the size dependence with them, and the size dependence is the thing the
    power law extrapolates through. The third arm keeps the physics and drops
    the device name, which is the comparison that decides it.
    """
    prepared = with_dimensionless_regression_columns(framed)
    return {
        "within_device_variance": within_device_variance(
            prepared, (*hdb5.BLIND_FEATURE_COLUMNS, *DIMENSIONLESS_FEATURES)
        ),
        "device_recovery": device_recoverable_from(prepared, hdb5.BLIND_FEATURE_COLUMNS),
        "arms": {
            "nine_engineering_features": _reversal_arm(prepared, hdb5.BLIND_FEATURE_COLUMNS),
            "within_device_features_only": _reversal_arm(prepared, WITHIN_DEVICE_FEATURES),
            "dimensionless_groups": _reversal_arm(
                prepared, DIMENSIONLESS_FEATURES, target=DIMENSIONLESS_TARGET
            ),
        },
    }


def _size_cut_separation(framed: pd.DataFrame) -> dict[str, Any]:
    """Does the ITER-matched size cut separate the two sides in dimensionless space?

    Reported as the overlap of each group's range across the cut. A cut that is
    only a size cut leaves the dimensionless envelopes overlapping; one that
    separates them is cutting on something the paper has not named.
    """
    split = hdb5.iter_matched_split(framed, hdb5.size_ordered_splits(framed))
    labels = framed[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    above = np.isin(labels, list(split.test_machines))

    out: dict[str, Any] = {"size_ratio": split.size_ratio, "groups": {}}
    for column in GROUP_COLUMNS:
        values = framed[column].to_numpy(dtype=float)
        low, high = values[~above], values[above]
        # Overlap of the two interquartile ranges, as a fraction of the union.
        low_range = (float(np.percentile(low, 25)), float(np.percentile(low, 75)))
        high_range = (float(np.percentile(high, 25)), float(np.percentile(high, 75)))
        intersection = max(0.0, min(low_range[1], high_range[1]) - max(low_range[0], high_range[0]))
        union = max(low_range[1], high_range[1]) - min(low_range[0], high_range[0])
        out["groups"][column] = {
            "train_iqr": low_range,
            "test_iqr": high_range,
            "iqr_overlap_fraction": float(intersection / union) if union > 0 else 1.0,
            "median_shift": float(np.median(high) - np.median(low)),
            "train_sd": float(np.std(low)),
            "fraction_of_a_training_sd": float(
                abs(np.median(high) - np.median(low)) / np.std(low)
            )
            if np.std(low) > 0
            else 0.0,
        }
    return out


def analyze(framed: pd.DataFrame) -> dict[str, Any]:
    per_machine = hdb5.leave_one_tokamak_out(framed)
    scores = per_machine.pivot(index="tokamak", columns="model_name", values="rmsle")

    dimensionless = label_distances(framed, GROUP_COLUMNS)
    engineering = label_distances(framed, hdb5.BLIND_FEATURE_COLUMNS)
    shared = sorted(set(dimensionless) & set(engineering) & set(scores.index))

    table = pd.DataFrame(
        {
            "tokamak": shared,
            "dimensionless_distance": [dimensionless[m] for m in shared],
            "engineering_distance": [engineering[m] for m in shared],
            **{
                f"rmsle_{name}": [float(scores.loc[m, name]) for m in shared]
                for name in (POWER_LAW, FOREST, BOOSTER)
                if name in scores.columns
            },
        }
    )

    correlations = {}
    for name in (FOREST, BOOSTER, POWER_LAW):
        column = f"rmsle_{name}"
        if column not in table:
            continue
        correlations[name] = {
            "against_dimensionless": _spearman(
                table[column].to_numpy(), table["dimensionless_distance"].to_numpy()
            ),
            "against_engineering": _spearman(
                table[column].to_numpy(), table["engineering_distance"].to_numpy()
            ),
        }

    return {
        "n_rows": int(len(framed)),
        "n_labels_scored": len(shared),
        "group_columns": list(GROUP_COLUMNS),
        "distance_agreement": _spearman(
            table["dimensionless_distance"].to_numpy(), table["engineering_distance"].to_numpy()
        ),
        "error_correlations": correlations,
        "iter_matched_cut": _size_cut_separation(framed),
        "device_identity": device_identity_arms(framed),
        "_table": table,
    }


def _report(payload: dict[str, Any]) -> None:
    print(
        f"{payload['n_rows']} rows placed in dimensionless space, "
        f"{payload['n_labels_scored']} labels scored"
    )
    print(f"the two distance rankings agree at rho = {payload['distance_agreement']:+.2f}\n")
    print(f"{'model':24} {'vs dimensionless':>18} {'vs engineering':>16}")
    for name, row in payload["error_correlations"].items():
        print(f"{name:24} {row['against_dimensionless']:+18.2f} {row['against_engineering']:+16.2f}")
    identity = payload["device_identity"]
    recovery = identity["device_recovery"]
    print(
        f"\ndevice recovered from the nine features at {recovery['accuracy']:.1%} "
        f"against a {recovery['majority_baseline']:.1%} majority baseline"
    )
    for name, arm in identity["arms"].items():
        label, device = arm["by_label"], arm["by_device"]
        print(f"  {name} ({arm['n_features']} features, target {arm['target']})")
        print(
            f"    CV gain {arm['cv_gain_of_forest']:+.1%}   "
            f"forest worse on {label['n_forest_worse']}/{label['n_units']} labels "
            f"({label['mean_difference']:+.4f}), "
            f"{device['n_forest_worse']}/{device['n_units']} devices "
            f"({device['mean_difference']:+.4f})"
        )
    print("\nITER-size-matched cut, dimensionless separation:")
    for column, row in payload["iter_matched_cut"]["groups"].items():
        print(
            f"  {column:14} IQR overlap {row['iqr_overlap_fraction']:5.2f}   "
            f"median shift {row['median_shift']:+6.2f} "
            f"({row['fraction_of_a_training_sd']:.2f} training SD)"
        )


def main() -> None:
    framed = with_dimensionless_groups(hdb5.prepare_dataset())
    payload = analyze(framed)
    _report(payload)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(RESULTS_DIR / "dimensionless_per_label.csv", payload.pop("_table"))
    payload["dataset_sha256"] = hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256
    write_json_strict(RESULTS_DIR / "dimensionless.json", payload)
    print(f"\nwrote {RESULTS_DIR / 'dimensionless.json'}")


if __name__ == "__main__":
    main()
