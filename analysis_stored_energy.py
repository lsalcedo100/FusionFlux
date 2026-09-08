"""Result 18: the reversal with the target's denominator moved to the right-hand side.

The thermal confinement time is thermal stored energy over thermal loss power,
so ``log tau_th = log W_th - log P_LTH`` up to the deposit's dW/dt and radiation
conventions, and ``P_LTH`` is one of the nine features every model sees. One
predictor is the denominator of the response, a log-linear model carries that
exactly and outside the training range of P, and a tree cannot. Under that
reading the power law transfers because an identity is exactly log-linear rather
than because it has better long-range structure.

``analysis_power_ablation`` attacked that by deleting P. This module attacks it
the right way round, by moving W_th to the left. Regressing ``log W_th`` on the
same nine features leaves P in the feature set as an ordinary predictor of
confinement, which it is, and removes the identity, which is the only thing the
objection is about. Deleting P instead removes both, which is why that ablation
costs so much: it is a harder experiment than the objection asks for.

The main text used to say this control was unavailable, on the ground that the
STD5 deposit carries ``TAUTH`` and ``PLTH`` and no stored-energy column. That is
true of STD5, which is a 15-column extract, and it was the wrong file to look
in. The full DB5.2.3 revision this repository already downloads, pins by
SHA-256 and analyses in Result 11 carries ``WTH`` on 13828 of its 14153 rows,
and joining it to STD5 on (machine, shot, time) recovers a stored energy for
6214 of the 6228 analysed rows. Inside DB5.2.3 the identity holds to the digit:
``WTH / (TAUTH * PLTH)`` has median 1.000000 and an interquartile range of
2e-4.

Run ``python3 analysis_stored_energy.py`` to regenerate
``results/stored_energy.json`` and its two per-unit tables.
"""

from __future__ import annotations

from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hdb5
import replication as rep
from storage import write_dataframe_csv_atomic, write_json_strict

RESULTS_DIR = Path(__file__).resolve().parent / "results"

POWER_LAW = "ridge_loglinear"
FOREST = "random_forest"
BOOSTER = "hist_gradient_boosting"

# The stored-energy column DB5.2.3 carries and STD5 does not, in joules.
STORED_ENERGY_COLUMN = "w_th_j"

# The log loss-power feature that is also the target's denominator.
POWER_COLUMN = "log_p_loss_mw"


def dataset_with_stored_energy(
    *, std5_path: Path | str | None = None, db523_path: Path | str | None = None
) -> pd.DataFrame:
    """The analysed rows, plus the stored energy, minus the few that have none.

    Dropping the unmatched rows rather than imputing them keeps this a
    like-for-like comparison: both arms below are scored on exactly these rows,
    so the difference between them is the target and nothing else.
    """
    dataset = hdb5.prepare_dataset(std5_path)
    carried = rep.db523_columns(("WTH",), std5_path=std5_path, db523_path=db523_path)
    stored = pd.to_numeric(carried["WTH"], errors="coerce")

    framed = dataset.assign(**{STORED_ENERGY_COLUMN: stored})
    usable = framed[STORED_ENERGY_COLUMN].notna() & (framed[STORED_ENERGY_COLUMN] > 0)
    return framed.loc[usable].reset_index(drop=True)


def identity_residual(dataset: pd.DataFrame) -> dict[str, float]:
    """How exactly W_th / (tau_th * P) holds on the rows being scored.

    Reported rather than assumed, because the whole argument for this control is
    that the two targets differ by an identity. Where the identity is loose the
    two arms are not measuring the same thing, and the reader should be able to
    see how loose.
    """
    ratio = dataset[STORED_ENERGY_COLUMN] / (dataset[hdb5.TARGET_COLUMN] * dataset["p_loss_mw"] * 1e6)
    return {
        "median": float(ratio.median()),
        "q25": float(ratio.quantile(0.25)),
        "q75": float(ratio.quantile(0.75)),
        "q01": float(ratio.quantile(0.01)),
        "q99": float(ratio.quantile(0.99)),
    }


def _sign_test(n_worse: int, n_units: int) -> float:
    """Two-sided exact test under the independent-unit-sign null of Sec. 4.2."""
    tail = sum(comb(n_units, k) for k in range(min(n_worse, n_units - n_worse), -1, -1))
    upper = sum(comb(n_units, k) for k in range(max(n_worse, n_units - n_worse), n_units + 1))
    return float(min(1.0, (tail + upper) / 2**n_units))


def _paired(per_unit: pd.DataFrame, worse: str, better: str) -> dict[str, Any]:
    wide = per_unit.pivot(index="tokamak", columns="model_name", values="rmsle")
    difference = (wide[worse] - wide[better]).dropna()
    n_worse = int((difference > 0).sum())
    return {
        "n_units": int(len(difference)),
        "n_worse": n_worse,
        "mean_difference": float(difference.mean()),
        "sign_test_p": _sign_test(n_worse, len(difference)),
        "units_where_better_loses": sorted(difference.index[difference < 0]),
    }


def _retargeted(dataset: pd.DataFrame, target: str) -> pd.DataFrame:
    """The same rows and features, scored against a different response."""
    framed = dataset.copy()
    framed[hdb5.TARGET_COLUMN] = framed[target]
    return framed


def _loss_power_exponent(dataset: pd.DataFrame, target: str) -> float:
    """The fitted exponent on log P, in raw units, for one target.

    The control is often read as removing the identity tau = W/P from what the
    log-linear model can represent. It does not. log W = log tau + log P up to
    the deposit's conventions, and log P is a column of the design matrix, so
    retargeting leaves that matrix untouched and moves this one coefficient by
    one. What changes for a tree is the size of the slope it has to extrapolate
    in that direction, which gets *smaller*, so the control is biased toward the
    tree rather than away from it. Reporting the two exponents is what lets a
    reader see that rather than take it on trust.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    columns = list(hdb5.BLIND_FEATURE_COLUMNS)
    features = dataset[columns].to_numpy(dtype=float)
    response = np.log(dataset[target].to_numpy(dtype=float))
    fitted = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="svd"))
    with hdb5._suppress_benign_matmul_warnings():
        fitted.fit(features, response)
    raw = fitted[-1].coef_ / fitted[0].scale_
    return float(raw[columns.index(POWER_COLUMN)])


def _arm(dataset: pd.DataFrame, target: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    framed = _retargeted(dataset, target)
    by_device = hdb5.with_device_column(framed).copy()
    by_device[hdb5.TOKAMAK_LABEL_COLUMN] = by_device[hdb5.DEVICE_COLUMN]

    # The analytic law predicts tau and is meaningless against W_th, so neither
    # arm carries it: this comparison is between fitted models only.
    per_label = hdb5.leave_one_tokamak_out(framed, include_ipb98_reference=False)
    per_device = hdb5.leave_one_tokamak_out(by_device, include_ipb98_reference=False)
    # `evaluate_models` defaults to the ten-column matrix, which carries the
    # analytic IPB98 prediction as a feature; the three arms above default to the
    # nine blind columns. Left implicit, this arm's cross-validation column came
    # from a different feature set than its own held-out columns, and from a
    # feature the paper says twice is excluded throughout. The ridge cannot see
    # the difference, since that column is an exact log-linear combination of the
    # other eight, so only the tree ensembles were being helped by it.
    cross_validated = {
        s.model_name: float(s.cv_rmsle)
        for s in hdb5.evaluate_models(framed, feature_columns=hdb5.BLIND_FEATURE_COLUMNS)
    }

    split = hdb5.iter_matched_split(framed, hdb5.size_ordered_splits(framed))
    cut = (
        hdb5.score_size_split(framed, split, include_ipb98_reference=False)
        .set_index("model_name")["rmsle"]
        .astype(float)
    )

    summary = hdb5.summarize_leave_one_tokamak_out(per_label).set_index("model_name")
    device_summary = hdb5.summarize_leave_one_tokamak_out(per_device).set_index("model_name")
    return (
        {
            "target": target,
            "loss_power_exponent": _loss_power_exponent(dataset, target),
            "cv_rmsle": cross_validated,
            "lolo_mean_rmsle": {n: float(summary.loc[n, "mean_rmsle"]) for n in summary.index},
            "lodo_mean_rmsle": {n: float(device_summary.loc[n, "mean_rmsle"]) for n in device_summary.index},
            "iter_matched_cut": {n: float(v) for n, v in cut.items()},
            "forest_worse_by_label": _paired(per_label, FOREST, POWER_LAW),
            "forest_worse_by_device": _paired(per_device, FOREST, POWER_LAW),
            "booster_worse_by_label": _paired(per_label, BOOSTER, POWER_LAW),
            "booster_worse_by_device": _paired(per_device, BOOSTER, POWER_LAW),
        },
        per_label,
        per_device,
    )


def analyze(dataset: pd.DataFrame) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    tables: dict[str, pd.DataFrame] = {}
    for name, target in (
        ("confinement_time", hdb5.TARGET_COLUMN),
        ("stored_energy", STORED_ENERGY_COLUMN),
    ):
        arm, per_label, per_device = _arm(dataset, target)
        arms[name] = arm
        tables[f"{name}_per_label"] = per_label
        tables[f"{name}_per_device"] = per_device
    return {
        "n_rows": int(len(dataset)),
        "n_labels": int(dataset[hdb5.TOKAMAK_LABEL_COLUMN].nunique()),
        "identity_residual": identity_residual(dataset),
        "arms": arms,
        "_tables": tables,
    }


def _report(payload: dict[str, Any]) -> None:
    identity = payload["identity_residual"]
    print(
        f"{payload['n_rows']} rows carry a stored energy; W_th/(tau_th P) has median "
        f"{identity['median']:.6f}, quartiles {identity['q25']:.6f} to {identity['q75']:.6f}"
    )
    for name, arm in payload["arms"].items():
        label = arm["forest_worse_by_label"]
        device = arm["forest_worse_by_device"]
        print(f"\n{name} (target {arm['target']})")
        print(f"  CV        forest {arm['cv_rmsle'][FOREST]:.4f}   power law {arm['cv_rmsle'][POWER_LAW]:.4f}")
        print(
            f"  LOLO      forest {arm['lolo_mean_rmsle'][FOREST]:.4f}   "
            f"power law {arm['lolo_mean_rmsle'][POWER_LAW]:.4f}   "
            f"forest worse on {label['n_worse']} of {label['n_units']}, "
            f"gap {label['mean_difference']:+.4f}, p={label['sign_test_p']:.4f}"
        )
        print(
            f"  LODO      forest {arm['lodo_mean_rmsle'][FOREST]:.4f}   "
            f"power law {arm['lodo_mean_rmsle'][POWER_LAW]:.4f}   "
            f"forest worse on {device['n_worse']} of {device['n_units']}, "
            f"gap {device['mean_difference']:+.4f}, p={device['sign_test_p']:.4f}"
        )
        print(
            f"  size cut  forest {arm['iter_matched_cut'][FOREST]:.4f}   "
            f"power law {arm['iter_matched_cut'][POWER_LAW]:.4f}   "
            f"booster {arm['iter_matched_cut'][BOOSTER]:.4f}"
        )


def main() -> None:
    dataset = dataset_with_stored_energy()
    payload = analyze(dataset)
    _report(payload)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in payload.pop("_tables").items():
        write_dataframe_csv_atomic(RESULTS_DIR / f"stored_energy_{name}.csv", table)
    payload["dataset_sha256"] = hdb5.fingerprint_file(hdb5.default_hdb5_path()).sha256
    payload["db523_sha256"] = hdb5.fingerprint_file(rep.default_db523_path()).sha256
    write_json_strict(RESULTS_DIR / "stored_energy.json", payload)
    print(f"\nwrote {RESULTS_DIR / 'stored_energy.json'}")


if __name__ == "__main__":
    main()
