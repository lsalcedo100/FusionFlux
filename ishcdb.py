"""The stellarator-heliotron confinement database, in this study's terms. Nothing here fits a model.

ISHCDB is the stellarator counterpart of the ITPA confinement database every
fusion result in this repository rests on: the same IEA lineage, the same column
conventions, and the database ISS04 was fitted to as IPB98(y,2) was fitted to
the ITPA one. It shares no device with HDB5. Version 26 is public, in an open
directory on the IPP server, so it is fetched on demand and pinned by SHA-256
like the other deposits, and never redistributed here.

The replication is prospective. ``docs/ishcdb-replication-lock.md`` fixed the
rules from the database's documentation, before the file was downloaded, and
this module turns them into code in stages that each stop short of a score:

    schema    ``ciclop.schema_report``, unchanged: one column at a time, and
              nothing that joins the target to a feature.

    freeze    Everything the lock says follows from the file, computed and not
              chosen: the measured rows, the standard set, the usable features,
              the complete rows, the eligible devices and whether the dataset
              clears the floors. Written to ``ishcdb_analysis_plan.json``.

Unlike CICLOP there is no hand-written column mapping, because this database
documents its columns and the lock names them. If the file turns out to spell
one differently, ``COLUMN_OVERRIDES`` in the plan records that, and the lock's
deviations log records why.

The order is enforced through ``ciclop.verify_frozen_plan``, the same door the
first replication goes through: ``analysis_ishcdb`` refuses to fit anything
until the file and the plan are both pinned below and both match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ciclop
import config
import hdb5
from storage import atomic_output_path, write_json_strict

ISHCDB_DIRECTORY_URL = "https://ishpdb.ipp-hgw.mpg.de/ISS/"
ISHCDB_DOWNLOAD_URL = ISHCDB_DIRECTORY_URL + "ISHCDB_26.txt"
DEFAULT_ISHCDB_FILENAME = "ishcdb_26.txt"
LOCK_DOCUMENT = "docs/ishcdb-replication-lock.md"
PLAN_FILENAME = "ishcdb_analysis_plan.json"

# --- The pins ------------------------------------------------------------------
#
# The file pin is set when the file is first downloaded, in its own commit, before
# the schema pass. The plan pin is set in the commit that adds the plan. Both
# ship unset, and nothing is scored until both are set and both match.
#
# The file pin was taken on 21 September 2026 from two independent downloads of
# ISHCDB_DOWNLOAD_URL that agreed, and committed before the file was opened.
ISHCDB_SHA256: str | None = "3a9f857b768ebd601385a9d8e25ac7df4d48b3403795a43dc484ba6b17fc5f55"
ISHCDB_N_BYTES: int | None = 3311757
ISHCDB_PLAN_SHA256: str | None = None

# --- Constants of the lock -----------------------------------------------------
USABLE_FRACTION = ciclop.USABLE_FRACTION
MIN_HELD_OUT_ROWS = hdb5.MIN_HELD_OUT_ROWS
SENSITIVITY_MIN_ROWS = 10
MIN_EVALUABLE_DEVICES = ciclop.MIN_EVALUABLE_DEVICES
MIN_EVALUABLE_ROWS = ciclop.MIN_EVALUABLE_ROWS
MIN_EVALUABLE_FEATURES = 5

# The documentation: "W7-X and ITER are predictive data". They are not
# measurements, so they leave before anything else happens, training included.
PREDICTIVE_DEVICES: tuple[str, ...] = ("W7-X", "ITER")
# The standard set uses the diamagnetic confinement time, "only for Heliotron-E
# has the thermal confinement time be used". Followed, not improved on.
THERMAL_TARGET_DEVICES: tuple[str, ...] = ("HELE",)

DEVICE_SOURCE, STANDARD_SET_SOURCE, SHOT_SOURCE = "STELL", "STDSET", "SHOT"
DIAMAGNETIC_TARGET_SOURCE, THERMAL_TARGET_SOURCE = "TAUEDIA", "TAUETH"
# canonical name -> (documented column, factor to the law's unit, take the absolute value)
FEATURE_SOURCES: dict[str, tuple[str, float, bool]] = {
    "a_m": ("AEFF", 1.0, False),
    "r_m": ("RGEO", 1.0, False),
    "p_abs_mw": ("PTOT", 1e-6, False),
    "ne_line_1e19_m3": ("NEBAR", 1e-19, False),
    "bt_t": ("BT", 1.0, True),
    "iota_23": ("IOTA23", 1.0, True),
}
FEATURES: tuple[str, ...] = tuple(FEATURE_SOURCES)

STANDARD_SET_COLUMN = "in_standard_set"
DIAMAGNETIC_TARGET_COLUMN = "tau_dia_s"
ISS04_COLUMN = "iss04_tau_s"


def default_ishcdb_path() -> Path:
    return config.get_data_raw_dir() / DEFAULT_ISHCDB_FILENAME


def default_plan_path() -> Path:
    return config.PROJECT_ROOT / PLAN_FILENAME


# --- Fetching and reading --------------------------------------------------------


def verify_ishcdb_file(path: Path | str) -> hdb5.DatasetFingerprint:
    """Fingerprint a file and raise unless it is the pinned one. Before the pin exists, just fingerprint."""
    fingerprint = hdb5.fingerprint_file(path, read_shape=False)
    if ISHCDB_SHA256 is None or (fingerprint.sha256, fingerprint.n_bytes) == (ISHCDB_SHA256, ISHCDB_N_BYTES):
        return fingerprint
    raise hdb5.DatasetIntegrityError(
        f"ISHCDB integrity check failed for {fingerprint.path}.\n"
        f"  expected sha256 {ISHCDB_SHA256} ({ISHCDB_N_BYTES} bytes)\n"
        f"  observed sha256 {fingerprint.sha256} ({fingerprint.n_bytes} bytes)\n"
        f"Re-fetch from {ISHCDB_DOWNLOAD_URL}."
    )


def download_ishcdb(destination: Path | None = None, *, overwrite: bool = False) -> Path:
    """Fetch version 26 from the IPP directory into the raw data directory.

    Third-party scientific data (please cite Yamada et al., Nucl. Fusion 45 1684,
    2005), fetched on demand and verified on the staged temporary file, so a
    download that does not match the pin never lands at the target path.
    """
    import urllib.request

    target = Path(destination).expanduser().resolve() if destination else default_ishcdb_path()
    if target.exists() and not overwrite:
        verify_ishcdb_file(target)
        return target
    with atomic_output_path(target) as temp_path:
        request = urllib.request.Request(ISHCDB_DOWNLOAD_URL, headers={"User-Agent": "FusionFlux (research replication)"})
        with urllib.request.urlopen(request) as response:
            temp_path.write_bytes(response.read())
        verify_ishcdb_file(temp_path)
    return target


def load_ishcdb_raw(path: Path | str | None = None, read: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the delivered table. ``read`` carries what the schema pass found about its layout."""
    resolved = Path(path).expanduser().resolve() if path is not None else default_ishcdb_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"ISHCDB not found: {resolved}. Fetch it with "
            f"`python3 -c 'import ishcdb; ishcdb.download_ishcdb()'` or download {ISHCDB_DOWNLOAD_URL} to that path."
        )
    options = dict(read or {})
    frame = pd.read_csv(
        resolved,
        sep=options.get("separator", "\t"),
        encoding=options.get("encoding", "latin-1"),
        header=int(options.get("header_row", 0)),
        low_memory=False,
    )
    frame.columns = [str(name).strip() for name in frame.columns]
    return frame


# --- From the file's columns to the study's ---------------------------------------


def resolve_columns(raw: pd.DataFrame, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The documented column names, each checked to exist, with any recorded respelling applied."""
    wanted = [DEVICE_SOURCE, STANDARD_SET_SOURCE, SHOT_SOURCE, DIAMAGNETIC_TARGET_SOURCE, THERMAL_TARGET_SOURCE]
    wanted += [source for source, _, _ in FEATURE_SOURCES.values()]
    resolved = {name: (overrides or {}).get(name, name) for name in wanted}
    missing = {documented: actual for documented, actual in resolved.items() if actual not in raw.columns}
    if missing:
        raise ValueError(
            f"ISHCDB does not carry {sorted(missing.values())}. The documentation names them; if the file "
            "spells one differently, record it under `column_overrides` and in the lock's deviations log. "
            f"The file's columns begin {list(raw.columns)[:25]}."
        )
    return resolved


def canonical_frame(raw: pd.DataFrame, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Every delivered row in the study's variables. No row is dropped here."""
    columns = resolve_columns(raw, overrides)
    device = raw[columns[DEVICE_SOURCE]].astype(str).str.strip()

    frame = pd.DataFrame(index=raw.index)
    diamagnetic = pd.to_numeric(raw[columns[DIAMAGNETIC_TARGET_SOURCE]], errors="coerce")
    thermal = pd.to_numeric(raw[columns[THERMAL_TARGET_SOURCE]], errors="coerce")
    frame[DIAMAGNETIC_TARGET_COLUMN] = diamagnetic
    frame[hdb5.TARGET_COLUMN] = diamagnetic.where(~device.isin(THERMAL_TARGET_DEVICES), thermal)
    for canonical, (source, factor, take_abs) in FEATURE_SOURCES.items():
        values = pd.to_numeric(raw[columns[source]], errors="coerce") * factor
        frame[canonical] = values.abs() if take_abs else values

    frame[hdb5.TOKAMAK_LABEL_COLUMN] = device
    frame[STANDARD_SET_COLUMN] = pd.to_numeric(raw[columns[STANDARD_SET_SOURCE]], errors="coerce") == 1
    # One group per discharge. A row with no shot number is its own group, so it
    # can never pull an unrelated row into its fold.
    shot = raw[columns[SHOT_SOURCE]]
    label = shot.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    own = pd.Series([f"row{i}" for i in range(len(raw))], index=raw.index)
    frame[hdb5.GROUP_COLUMN] = device + "::" + label.where(shot.notna() & (label != ""), own)
    return frame.reset_index(drop=True)


def measured_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows the documentation calls predictive. They are forecasts, not data."""
    return frame.loc[~frame[hdb5.TOKAMAK_LABEL_COLUMN].isin(PREDICTIVE_DEVICES)].reset_index(drop=True)


def standard_set(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame[STANDARD_SET_COLUMN]].reset_index(drop=True)


def iss04_tau_s(frame: pd.DataFrame) -> pd.Series:
    """ISS04 with its renormalisation factor at one (seconds).

    tau = 0.134 * a^2.28 * R^0.64 * P^-0.61 * n19^0.54 * B^0.84 * iota_2/3^0.41

    with a and R in m, P the absorbed power in MW, n19 the line-averaged density
    in 1e19 m^-3 and B in T. Reference: Yamada et al., Nucl. Fusion 45 1684 (2005).
    """
    return (
        0.134
        * np.power(frame["a_m"], 2.28)
        * np.power(frame["r_m"], 0.64)
        * np.power(frame["p_abs_mw"], -0.61)
        * np.power(frame["ne_line_1e19_m3"], 0.54)
        * np.power(frame["bt_t"], 0.84)
        * np.power(frame["iota_23"], 0.41)
    )


def usability(population: pd.DataFrame) -> dict[str, float]:
    """Share of rows on which each quantity is finite and positive. Missingness alone."""
    shares = {}
    for column in (hdb5.TARGET_COLUMN, *FEATURES):
        values = pd.to_numeric(population[column], errors="coerce").to_numpy(dtype=float)
        shares[column] = float(np.mean(np.isfinite(values) & (values > 0))) if len(values) else 0.0
    return shares


def frozen_features(shares: dict[str, float]) -> tuple[str, ...]:
    return tuple(column for column in FEATURES if shares[column] >= USABLE_FRACTION)


def evaluability(shares: dict[str, float], features: tuple[str, ...], n_rows: int, n_devices: int) -> list[str]:
    reasons = []
    if shares[hdb5.TARGET_COLUMN] < USABLE_FRACTION:
        reasons.append(f"the target is usable on {shares[hdb5.TARGET_COLUMN]:.0%} of rows")
    if len(features) < MIN_EVALUABLE_FEATURES:
        reasons.append(f"{len(features)} of the six features are usable, below {MIN_EVALUABLE_FEATURES}")
    if n_rows < MIN_EVALUABLE_ROWS:
        reasons.append(f"{n_rows} rows survive cleaning, below {MIN_EVALUABLE_ROWS}")
    if n_devices < MIN_EVALUABLE_DEVICES:
        reasons.append(f"{n_devices} devices have {MIN_HELD_OUT_ROWS} rows or more, below {MIN_EVALUABLE_DEVICES}")
    return reasons


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame[hdb5.TOKAMAK_LABEL_COLUMN].value_counts().items()}


def freeze_plan(path: Path | str, read: dict[str, Any] | None = None, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Apply the lock to the file and return the plan. Counts and shares, nothing else."""
    fingerprint = hdb5.fingerprint_file(path, read_shape=False)
    raw = load_ishcdb_raw(path, read)
    frame = canonical_frame(raw, overrides)
    measured = measured_rows(frame)
    primary = standard_set(measured)
    shares = usability(primary)
    features = frozen_features(shares)
    cleaned = ciclop.analysis_frame(primary, features)
    cleaned_all = ciclop.analysis_frame(measured, features)
    eligible = hdb5.eligible_tokamaks(cleaned, min_rows=MIN_HELD_OUT_ROWS) if len(cleaned) else []
    reasons = evaluability(shares, features, int(len(cleaned)), len(eligible))
    per_discharge = cleaned.groupby(hdb5.GROUP_COLUMN).size() if len(cleaned) else pd.Series(dtype=int)

    return {
        "lock": {"document": LOCK_DOCUMENT, "commit": ciclop._lock_commit(LOCK_DOCUMENT)},
        "file": {"name": Path(path).name, "sha256": fingerprint.sha256, "n_bytes": fingerprint.n_bytes, "source": ISHCDB_DOWNLOAD_URL},
        "read": dict(read or {}),
        "column_overrides": dict(overrides or {}),
        "rules": {
            "usable_fraction": USABLE_FRACTION,
            "min_held_out_rows": MIN_HELD_OUT_ROWS,
            "sensitivity_min_rows": SENSITIVITY_MIN_ROWS,
            "min_evaluable_devices": MIN_EVALUABLE_DEVICES,
            "min_evaluable_rows": MIN_EVALUABLE_ROWS,
            "min_evaluable_features": MIN_EVALUABLE_FEATURES,
            "degradation_boundary": ciclop.DEGRADATION_BOUNDARY,
            "predictive_devices": list(PREDICTIVE_DEVICES),
            "thermal_target_devices": list(THERMAL_TARGET_DEVICES),
            "feature_sources": {name: list(source) for name, source in FEATURE_SOURCES.items()},
            "cv_folds": hdb5.N_CV_FOLDS,
            "random_state": hdb5.RANDOM_STATE,
        },
        "population": {
            "n_rows_delivered": int(len(frame)),
            "n_predictive_rows": int(len(frame) - len(measured)),
            "rows_per_device_delivered": _counts(frame),
            "rows_per_device_measured": _counts(measured),
            "rows_per_device_standard_set": _counts(primary),
        },
        "usability": shares,
        "frozen_features": list(features),
        "omitted_features": {c: f"finite and positive on {shares[c]:.0%} of rows, below {USABLE_FRACTION:.0%}" for c in FEATURES if c not in features},
        "n_complete_rows": int(len(cleaned)),
        "complete_rows_per_device": _counts(cleaned) if len(cleaned) else {},
        "n_complete_rows_all_measured": int(len(cleaned_all)),
        "complete_rows_per_device_all_measured": _counts(cleaned_all) if len(cleaned_all) else {},
        "eligible_devices": list(eligible),
        "eligible_devices_at_sensitivity_threshold": hdb5.eligible_tokamaks(cleaned, min_rows=SENSITIVITY_MIN_ROWS) if len(cleaned) else [],
        "discharges": {
            "n": int(per_discharge.shape[0]),
            "n_with_more_than_one_row": int((per_discharge > 1).sum()),
            "largest": int(per_discharge.max()) if len(per_discharge) else 0,
        },
        "evaluable": not reasons,
        "not_evaluable_because": reasons,
    }


def write_plan(plan: dict[str, Any], path: Path | None = None) -> Path:
    target = default_plan_path() if path is None else path
    write_json_strict(target, plan)
    return target


def sha256_of_plan(path: Path | None = None) -> str:
    return hashlib.sha256((default_plan_path() if path is None else path).read_bytes()).hexdigest()


def load_frozen_plan(plan_path: Path | None = None, *, data_path: Path | str | None = None) -> tuple[dict[str, Any], Path]:
    """The plan and the data file, through the same door the first replication uses."""
    return ciclop.verify_frozen_plan(
        default_plan_path() if plan_path is None else plan_path,
        data_path,
        dataset="ISHCDB",
        module="ishcdb.py",
        lock_document=LOCK_DOCUMENT,
        file_sha256=ISHCDB_SHA256,
        file_n_bytes=ISHCDB_N_BYTES,
        plan_sha256=ISHCDB_PLAN_SHA256,
    )


def frames_from_plan(plan: dict[str, Any], data_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The standard set and every measured row, cleaned, or a refusal if they are not the rows frozen."""
    raw = load_ishcdb_raw(data_path, plan.get("read"))
    measured = measured_rows(canonical_frame(raw, plan.get("column_overrides")))
    features = tuple(plan["frozen_features"])
    primary = ciclop.analysis_frame(standard_set(measured), features)
    everything = ciclop.analysis_frame(measured, features)
    if _counts(primary) != plan["complete_rows_per_device"] or _counts(everything) != plan["complete_rows_per_device_all_measured"]:
        raise AssertionError(
            "the rows rebuilt from the plan are not the rows the plan froze:\n"
            f"  frozen  {plan['complete_rows_per_device']}\n  rebuilt {_counts(primary)}"
        )
    for frame in (primary, everything):
        frame[ISS04_COLUMN] = iss04_tau_s(frame) if set(FEATURES) <= set(features) else np.nan
    return primary, everything


# --- CLI ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ISHCDB: fetch, schema pass and plan freezing. Fits nothing.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch version 26 and print its fingerprint")
    for name, text in (("schema", "per-column summaries of the delivered file"), ("freeze", f"apply the lock and write {PLAN_FILENAME}")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--path", type=Path, default=None)
        sub.add_argument("--separator", default="\t")
        sub.add_argument("--encoding", default="latin-1")
        sub.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        fingerprint = hdb5.fingerprint_file(download_ishcdb(), read_shape=False)
        print(f'ISHCDB_SHA256 = "{fingerprint.sha256}"\nISHCDB_N_BYTES = {fingerprint.n_bytes}')
        return
    path = args.path or default_ishcdb_path()
    read = {"separator": args.separator, "encoding": args.encoding}
    if args.command == "schema":
        fingerprint = hdb5.fingerprint_file(path, read_shape=False)
        report: dict[str, object] = {"file": {"name": path.name, "sha256": fingerprint.sha256, "n_bytes": fingerprint.n_bytes}}
        report.update(ciclop.schema_report(load_ishcdb_raw(path, read)))
        print(json.dumps(report, indent=2, default=str))
        if args.out is not None:
            write_json_strict(args.out, report)
        return
    plan = freeze_plan(path, read)
    written = write_plan(plan, args.out)
    print(f"Wrote {written}")
    print(f"  measured rows {sum(plan['population']['rows_per_device_measured'].values())}, predictive rows dropped {plan['population']['n_predictive_rows']}")
    print(f"  standard set, complete: {plan['n_complete_rows']} rows over {len(plan['complete_rows_per_device'])} devices")
    print(f"  frozen features ({len(plan['frozen_features'])} of 6): {plan['frozen_features']}")
    print(f"  eligible devices ({len(plan['eligible_devices'])}): {plan['eligible_devices']}")
    print(f"  evaluable: {plan['evaluable']}" + "".join(f"\n    {r}" for r in plan["not_evaluable_because"]))
    print(f'\nNow set in ishcdb.py, in the commit that adds the plan:\n  ISHCDB_PLAN_SHA256 = "{sha256_of_plan(written)}"')


if __name__ == "__main__":
    main()
