"""The CICLOP database, mapped onto this study's variables. Nothing here fits a model.

CICLOP is the IAEA/IEA multi-machine database of long-duration plasmas (Litaudon
et al., Nucl. Fusion 64 015001, 2024, and 66 095001, 2026). It was assembled
separately from the ITPA confinement database every other fusion result in this
repository rests on, which is what makes it worth replicating on, and it was not
built as a copy of it, which is what this module is for.

The replication is prospective. ``docs/ciclop-replication-lock.md`` fixed the
rules before the file was obtained, and this module turns those rules into code
in three stages that each stop short of a score:

    schema    Column names, units, per-column summaries, missingness and counts.
              Nothing that joins the target to a feature.

    mapping   Written by hand from what the schema pass shows, following the
              lock's orders of preference: which CICLOP column stands for each
              of the study's quantities, in what unit, and at which rank.

    freeze    Everything the lock says follows from the mapping, computed rather
              than chosen: which features are usable, the frozen feature set,
              the complete rows, the eligible devices, and whether the dataset
              clears the floors below which it cannot be evaluated. Written to
              ``ciclop_analysis_plan.json``.

The order is enforced, not just documented. The two pins below ship unset.
``analysis_ciclop`` refuses to fit anything until both are set and both match,
and setting them is a commit, so the history shows the plan was frozen before
the first score existed.

The file itself sits behind an IAEA NUCLEUS login and is never redistributed
here. There is no ``download`` function for that reason: a person has to fetch
it, and ``data/raw/ciclop*`` is gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
import hdb5
from storage import write_json_strict

CICLOP_PORTAL_URL = "https://nucleus.iaea.org/sites/fusion-portal/ciclop/SitePages/CICLOP-DB.aspx"
LOCK_DOCUMENT = "docs/ciclop-replication-lock.md"
PLAN_FILENAME = "ciclop_analysis_plan.json"

# --- The two pins ------------------------------------------------------------
#
# Unset until the plan is frozen. They are set together, in the commit that adds
# ``ciclop_analysis_plan.json``, and that commit has to predate the first score:
# it is the one the manuscript cites when it calls this analysis prospectively
# specified. The file pin is the same discipline as ``hdb5.HDB5_STD5_SHA256``.
# The plan pin is what stops the plan being edited after the scores are in
# without the edit showing up as a change to this file.
CICLOP_FILE_SHA256: str | None = None
CICLOP_FILE_N_BYTES: int | None = None
CICLOP_PLAN_SHA256: str | None = None

# --- Constants of the lock ---------------------------------------------------
#
# Every number below is a rule in ``docs/ciclop-replication-lock.md`` and is
# copied into the plan, so a result can be read against the rules it was run
# under. Changing one after the first score is what section 13 forbids.
USABLE_FRACTION = 0.90
MIN_HELD_OUT_ROWS = 10
SENSITIVITY_MIN_ROWS = hdb5.MIN_HELD_OUT_ROWS
MIN_EVALUABLE_DEVICES = 5
MIN_EVALUABLE_ROWS = 100
MIN_EVALUABLE_FEATURES = 6
DEGRADATION_BOUNDARY = 1.5

TOKAMAK = "tokamak"
STELLARATOR = "stellarator"

# Effective mass by main fuel species. The mapping assigns each label in the
# file to one of these names; it cannot assign a mass, so the numbers stay the
# lock's. A D-T pulse takes the number-weighted mean where the file gives a
# tritium fraction and 2.5 where it gives only the label.
SPECIES_MASS_AMU: dict[str, float] = {
    "hydrogen": 1.0,
    "deuterium": 2.0,
    "tritium": 3.0,
    "helium": 4.0,
    "deuterium-tritium": 2.5,
}

# The quantities read from a column. ``inverse_aspect_ratio`` is absent because
# the lock derives it as a/R. The names are ``hdb5``'s, including ``p_loss_mw``
# and ``ne_line_1e19_m3`` where CICLOP supplies injected power or a core density
# instead: the HDB5 rerun has to address the same columns, and the rank recorded
# in the plan is what says which definition was actually taken.
SOURCED_QUANTITIES: tuple[str, ...] = (
    "ip_ma",
    "bt_t",
    "ne_line_1e19_m3",
    "p_loss_mw",
    "r_m",
    "kappa",
    "m_eff_amu",
    "a_m",
)
SIGNED_QUANTITIES = frozenset({"ip_ma", "bt_t"})

# The lock's orders of preference, rank 1 first. The mapping records the rank it
# took for each, and ``validate_mapping`` rejects a rank that is not listed.
PREFERENCE_ORDERS: dict[str, tuple[str, ...]] = {
    "target": ("thermal confinement time", "confinement time as delivered"),
    "ne_line_1e19_m3": ("line-averaged electron density", "core electron density", "core ion density"),
    "p_loss_mw": ("loss or absorbed power", "injected additional power"),
    "kappa": ("areal elongation", "the single elongation the file carries"),
}
INJECTED_POWER_RANK = 2

FACILITY_COLUMN = "facility"
REGIME_COLUMN = "regime"
CONFIGURATION_COLUMN = "configuration"
H_MODE_COLUMN = "is_h_mode"
IN_HDB5_COLUMN = "device_in_hdb5"
SPLIT_DEVICE_COLUMN = "device_if_split"
PULSE_COLUMN = "pulse"


class PlanNotFrozenError(RuntimeError):
    """Raised when something asks for scores before the plan is frozen and pinned."""


# --- Reading the file --------------------------------------------------------


def load_ciclop_raw(path: Path | str, read: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the delivered file as it is, by extension.

    ``read`` carries what the schema pass found about its layout: the sheet, the
    header row, and any rows to skip. No Excel reader is a dependency of this
    repository, so a workbook fails with the fix named instead of a traceback
    from inside pandas.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"CICLOP file not found: {resolved}. It is not redistributed here and cannot be "
            f"fetched unattended: it sits behind an IAEA NUCLEUS login at {CICLOP_PORTAL_URL}."
        )
    options = dict(read or {})
    header = int(options.get("header_row", 0))
    skip = list(options.get("skip_rows", []))
    suffix = resolved.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else options.get("separator", ",")
        return pd.read_csv(resolved, header=header, skiprows=skip, sep=separator, low_memory=False)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            return pd.read_excel(resolved, sheet_name=options.get("sheet", 0), header=header, skiprows=skip)
        except ImportError as error:
            raise ImportError(
                f"{resolved.name} is a workbook and no Excel reader is installed. Either export "
                "the sheet to CSV and pin the CSV, or add openpyxl to requirements.txt and "
                "constraints.txt. Whichever is done, the pinned bytes are the file that is read."
            ) from error
    raise ValueError(f"unrecognised CICLOP file type {suffix!r}; expected a CSV or a workbook")


# --- Stage 1: the schema pass ------------------------------------------------


def schema_report(raw: pd.DataFrame, *, max_levels: int = 40) -> dict[str, Any]:
    """What the file contains, one column at a time. Marginal summaries only.

    The lock allows this pass to read names, units, per-column summaries,
    missingness and counts, and forbids anything that joins the target to a
    feature. Every entry below is a function of a single column, which
    ``tests/test_ciclop.py`` asserts, so the restriction is a property of the
    code and not of how carefully it was run.
    """
    columns: dict[str, Any] = {}
    for name in raw.columns:
        series = raw[name]
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        entry: dict[str, Any] = {
            "dtype": str(series.dtype),
            "n_non_null": int(series.notna().sum()),
            "n_numeric": int(finite.shape[0]),
            "n_positive": int((finite > 0).sum()),
        }
        if finite.shape[0]:
            entry["min"] = float(finite.min())
            entry["max"] = float(finite.max())
        levels = series.dropna().astype(str).str.strip()
        if finite.shape[0] < series.notna().sum() and levels.nunique() <= max_levels:
            entry["levels"] = {str(k): int(v) for k, v in levels.value_counts().items()}
        columns[str(name)] = entry
    return {"n_rows": int(len(raw)), "n_columns": int(raw.shape[1]), "columns": columns}


# --- Stage 2: the mapping ----------------------------------------------------


def validate_mapping(mapping: dict[str, Any], raw_columns: list[str]) -> None:
    """Reject a mapping the lock does not allow, before anything is computed from it."""
    problems: list[str] = []
    present = set(raw_columns)

    identity = mapping.get("identity", {})
    for role in ("facility", "pulse"):
        if identity.get(role) not in present:
            problems.append(f"identity.{role} names {identity.get(role)!r}, which is not a column")
    if identity.get("regime") is not None and identity["regime"] not in present:
        problems.append(f"identity.regime names {identity['regime']!r}, which is not a column")

    never = set(mapping.get("never_features", []))
    target = mapping.get("target", {})
    if target.get("source") not in present:
        problems.append(f"target.source names {target.get('source')!r}, which is not a column")
    never.add(str(target.get("source")))

    entries = {"target": target, **mapping.get("quantities", {})}
    unknown = set(mapping.get("quantities", {})) - set(SOURCED_QUANTITIES)
    if unknown:
        problems.append(f"quantities names {sorted(unknown)}, which the study does not use")
    for quantity in SOURCED_QUANTITIES:
        if quantity not in mapping.get("quantities", {}):
            problems.append(f"quantities.{quantity} is missing; give it a null source if the file lacks it")

    for name, entry in entries.items():
        source = entry.get("source")
        if name != "target":
            if source is not None and source not in present:
                problems.append(f"{name}.source names {source!r}, which is not a column")
            if source is not None and source in never:
                problems.append(f"{name} is mapped to {source!r}, which the lock says is never a feature")
        if name in PREFERENCE_ORDERS and source is not None:
            rank = entry.get("rank")
            if not isinstance(rank, int) or not 1 <= rank <= len(PREFERENCE_ORDERS[name]):
                problems.append(f"{name}.rank must be 1 to {len(PREFERENCE_ORDERS[name])}, got {rank!r}")

    for label, species in mapping.get("quantities", {}).get("m_eff_amu", {}).get("species", {}).items():
        if species not in SPECIES_MASS_AMU:
            problems.append(f"fuel label {label!r} is assigned {species!r}; use one of {sorted(SPECIES_MASS_AMU)}")

    for facility, entry in mapping.get("facilities", {}).items():
        if entry.get("configuration") not in {TOKAMAK, STELLARATOR}:
            problems.append(f"facility {facility!r} needs a configuration of {TOKAMAK!r} or {STELLARATOR!r}")
        if not entry.get("device"):
            problems.append(f"facility {facility!r} needs a device")

    if problems:
        raise ValueError("the CICLOP mapping is not usable:\n  " + "\n  ".join(problems))


def _numeric(raw: pd.DataFrame, entry: dict[str, Any], *, take_abs: bool) -> pd.Series:
    if entry.get("source") is None:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    values = pd.to_numeric(raw[entry["source"]], errors="coerce") * float(entry.get("to_canonical", 1.0))
    return values.abs() if take_abs else values


def _effective_mass(raw: pd.DataFrame, entry: dict[str, Any]) -> pd.Series:
    if entry.get("source") is None:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    labels = raw[entry["source"]].astype(str).str.strip()
    species = labels.map(entry.get("species", {}))
    mass = species.map(SPECIES_MASS_AMU).astype(float)
    fraction_column = entry.get("tritium_fraction_column")
    if fraction_column is not None:
        fraction = pd.to_numeric(raw[fraction_column], errors="coerce")
        weighted = SPECIES_MASS_AMU["deuterium"] * (1.0 - fraction) + SPECIES_MASS_AMU["tritium"] * fraction
        is_dt = (species == "deuterium-tritium") & fraction.between(0.0, 1.0)
        mass = mass.where(~is_dt, weighted)
    return mass


def canonical_frame(raw: pd.DataFrame, mapping: dict[str, Any]) -> pd.DataFrame:
    """Every delivered row in the study's variables. No row is dropped here.

    The device is written into ``hdb5.TOKAMAK_LABEL_COLUMN``, because that is
    the column every split in ``hdb5`` and ``analysis_robustness`` holds out on
    and the lock's held-out unit is the physical device. The facility label as
    delivered is kept beside it.
    """
    validate_mapping(mapping, [str(c) for c in raw.columns])
    identity = mapping["identity"]
    facility = raw[identity["facility"]].astype(str).str.strip()

    unknown = sorted(set(facility) - set(mapping["facilities"]))
    if unknown:
        raise ValueError(f"the file has facilities the mapping does not place: {unknown}")

    def facility_field(field: str, default: Any = None) -> pd.Series:
        return facility.map({name: entry.get(field, default) for name, entry in mapping["facilities"].items()})

    frame = pd.DataFrame(index=raw.index)
    frame[hdb5.TARGET_COLUMN] = _numeric(raw, mapping["target"], take_abs=False)
    for quantity in SOURCED_QUANTITIES:
        entry = mapping["quantities"][quantity]
        if quantity == "m_eff_amu":
            frame[quantity] = _effective_mass(raw, entry)
        else:
            frame[quantity] = _numeric(raw, entry, take_abs=quantity in SIGNED_QUANTITIES)
    frame["inverse_aspect_ratio"] = frame["a_m"] / frame["r_m"]

    pulse = raw[identity["pulse"]].astype(str).str.strip()
    frame[FACILITY_COLUMN] = facility
    frame[PULSE_COLUMN] = pulse
    frame[hdb5.TOKAMAK_LABEL_COLUMN] = facility_field("device")
    frame[SPLIT_DEVICE_COLUMN] = facility.map(
        {name: entry.get("device_if_split", entry.get("device")) for name, entry in mapping["facilities"].items()}
    )
    frame[CONFIGURATION_COLUMN] = facility_field("configuration")
    frame[IN_HDB5_COLUMN] = facility.map(
        {name: bool(entry.get("hdb5_tok")) for name, entry in mapping["facilities"].items()}
    ).astype(bool)
    frame[hdb5.GROUP_COLUMN] = facility + "::" + pulse
    if identity.get("regime") is not None:
        regime = raw[identity["regime"]].astype(str).str.strip()
        frame[REGIME_COLUMN] = regime
        frame[H_MODE_COLUMN] = regime.isin([str(v) for v in mapping.get("h_mode_values", [])])
    else:
        frame[REGIME_COLUMN] = "unstated"
        frame[H_MODE_COLUMN] = False
    return frame.reset_index(drop=True)


# --- Stage 3: what follows from the mapping ----------------------------------


def tokamak_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The lock's population: tokamak pulses only."""
    return frame.loc[frame[CONFIGURATION_COLUMN] == TOKAMAK].reset_index(drop=True)


def usability(tokamaks: pd.DataFrame) -> dict[str, float]:
    """Share of tokamak rows on which each quantity is finite and positive.

    Decided from missingness alone, as the lock requires: nothing here looks at
    the target's relationship to a feature, only at whether each value exists.
    """
    shares: dict[str, float] = {}
    for column in (hdb5.TARGET_COLUMN, *hdb5.BASE_ENGINEERING_COLUMNS):
        values = pd.to_numeric(tokamaks[column], errors="coerce").to_numpy(dtype=float)
        shares[column] = float(np.mean(np.isfinite(values) & (values > 0))) if len(values) else 0.0
    return shares


def frozen_features(shares: dict[str, float]) -> tuple[str, ...]:
    """The largest subset of the nine the file supports, in the study's order."""
    return tuple(c for c in hdb5.BASE_ENGINEERING_COLUMNS if shares[c] >= USABLE_FRACTION)


def log_feature_columns(features: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"log_{column}" for column in features)


def analysis_frame(tokamaks: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Complete rows on the frozen set, with the log features the models read.

    The cleaning rule is ``hdb5.analysed_row_mask``'s, finite and positive in the
    target and in every used column, applied to the frozen set instead of all
    nine. ``tests/test_ciclop.py`` checks that on all nine it selects the rows
    and produces the features ``hdb5.prepare_dataset_from_frame`` does.
    """
    used = [hdb5.TARGET_COLUMN, *features]
    values = tokamaks[used].apply(pd.to_numeric, errors="coerce")
    keep = values.notna().all(axis=1) & np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    cleaned = tokamaks.loc[keep].reset_index(drop=True)
    for column in features:
        cleaned[f"log_{column}"] = np.log(cleaned[column].astype(float))
    return cleaned


def evaluability(shares: dict[str, float], features: tuple[str, ...], n_rows: int, n_devices: int) -> list[str]:
    """The reasons verdict 1 holds before any score exists. Empty means evaluable."""
    reasons: list[str] = []
    if shares[hdb5.TARGET_COLUMN] < USABLE_FRACTION:
        reasons.append(f"the target is usable on {shares[hdb5.TARGET_COLUMN]:.0%} of tokamak rows")
    if len(features) < MIN_EVALUABLE_FEATURES:
        reasons.append(f"{len(features)} of the nine features are usable, below {MIN_EVALUABLE_FEATURES}")
    if n_rows < MIN_EVALUABLE_ROWS:
        reasons.append(f"{n_rows} tokamak rows survive cleaning, below {MIN_EVALUABLE_ROWS}")
    if n_devices < MIN_EVALUABLE_DEVICES:
        reasons.append(f"{n_devices} devices have {MIN_HELD_OUT_ROWS} rows or more, below {MIN_EVALUABLE_DEVICES}")
    return reasons


def _omission_reason(column: str, share: float, mapping: dict[str, Any]) -> str:
    if column == "inverse_aspect_ratio":
        return f"derived as a/R, which is finite and positive on {share:.0%} of tokamak rows"
    if mapping["quantities"][column].get("source") is None:
        return "absent from the file"
    return f"finite and positive on {share:.0%} of tokamak rows, below {USABLE_FRACTION:.0%}"


def _lock_commit() -> str | None:
    """The commit that last touched the lock, if this is a checkout. Recorded, never required."""
    import subprocess

    try:
        done = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", LOCK_DOCUMENT],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def freeze_plan(path: Path | str, mapping: dict[str, Any]) -> dict[str, Any]:
    """Apply sections 2 to 8 of the lock to the file and return the plan.

    The mapping is the only input a person wrote. Everything else in the plan is
    a count or a share computed here, so two people given the same file and the
    same mapping freeze the same plan.
    """
    fingerprint = hdb5.fingerprint_file(path, read_shape=False)
    raw = load_ciclop_raw(path, mapping.get("read"))
    frame = canonical_frame(raw, mapping)
    tokamaks = tokamak_rows(frame)
    shares = usability(tokamaks)
    features = frozen_features(shares)
    cleaned = analysis_frame(tokamaks, features)

    device_rows = cleaned[hdb5.TOKAMAK_LABEL_COLUMN].value_counts()
    eligible = hdb5.eligible_tokamaks(cleaned, min_rows=MIN_HELD_OUT_ROWS) if len(cleaned) else []
    reasons = evaluability(shares, features, int(len(cleaned)), len(eligible))
    in_hdb5 = cleaned.groupby(hdb5.TOKAMAK_LABEL_COLUMN)[IN_HDB5_COLUMN].any() if len(cleaned) else pd.Series(dtype=bool)

    return {
        "lock": {"document": LOCK_DOCUMENT, "commit": _lock_commit()},
        "file": {
            "name": Path(path).name,
            "sha256": fingerprint.sha256,
            "n_bytes": fingerprint.n_bytes,
            **mapping.get("file", {}),
        },
        "rules": {
            "usable_fraction": USABLE_FRACTION,
            "min_held_out_rows": MIN_HELD_OUT_ROWS,
            "sensitivity_min_rows": SENSITIVITY_MIN_ROWS,
            "min_evaluable_devices": MIN_EVALUABLE_DEVICES,
            "min_evaluable_rows": MIN_EVALUABLE_ROWS,
            "min_evaluable_features": MIN_EVALUABLE_FEATURES,
            "degradation_boundary": DEGRADATION_BOUNDARY,
            "species_mass_amu": SPECIES_MASS_AMU,
            "preference_orders": {name: list(order) for name, order in PREFERENCE_ORDERS.items()},
            "cv_folds": hdb5.N_CV_FOLDS,
            "random_state": hdb5.RANDOM_STATE,
        },
        "mapping": mapping,
        "population": {
            "n_rows_delivered": int(len(frame)),
            "n_tokamak_rows": int(len(tokamaks)),
            "n_stellarator_rows": int((frame[CONFIGURATION_COLUMN] == STELLARATOR).sum()),
            "rows_per_facility": {str(k): int(v) for k, v in frame[FACILITY_COLUMN].value_counts().items()},
            "regimes_per_device": {
                str(device): {str(k): int(v) for k, v in group[REGIME_COLUMN].value_counts().items()}
                for device, group in tokamaks.groupby(hdb5.TOKAMAK_LABEL_COLUMN)
            },
        },
        "usability": shares,
        "frozen_features": list(features),
        "omitted_features": {
            column: _omission_reason(column, shares[column], mapping)
            for column in hdb5.BASE_ENGINEERING_COLUMNS
            if column not in features
        },
        "power_is_injected": mapping["quantities"]["p_loss_mw"].get("rank") == INJECTED_POWER_RANK,
        "n_complete_rows": int(len(cleaned)),
        "complete_rows_per_device": {str(k): int(v) for k, v in device_rows.items()},
        "device_in_hdb5": {str(k): bool(v) for k, v in in_hdb5.items()},
        "eligible_devices": list(eligible),
        "eligible_devices_at_sensitivity_threshold": (
            hdb5.eligible_tokamaks(cleaned, min_rows=SENSITIVITY_MIN_ROWS) if len(cleaned) else []
        ),
        "evaluable": not reasons,
        "not_evaluable_because": reasons,
    }


# --- The plan on disk, and the guard -----------------------------------------


def default_plan_path() -> Path:
    return config.PROJECT_ROOT / PLAN_FILENAME


def write_plan(plan: dict[str, Any], path: Path | None = None) -> Path:
    target = default_plan_path() if path is None else path
    write_json_strict(target, plan)
    return target


def sha256_of_plan(path: Path | None = None) -> str:
    target = default_plan_path() if path is None else path
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_frozen_plan(
    plan_path: Path | None = None, *, data_path: Path | str | None = None
) -> tuple[dict[str, Any], Path]:
    """The plan and the data file, or an error saying which step was skipped.

    This is the only door to a score. It opens when the plan exists, both pins
    are set, the plan on disk is the pinned plan, and the data file is the
    pinned file. Each failure names the step of the lock's order of operations
    that has not happened, because the likeliest cause is an honest attempt to
    run things out of order.
    """
    target = default_plan_path() if plan_path is None else plan_path
    if not target.exists():
        raise PlanNotFrozenError(
            f"{target.name} does not exist. Run the schema pass, write the mapping, and freeze the "
            f"plan first (steps 4 and 5 of {LOCK_DOCUMENT}). Nothing is scored before that."
        )
    if CICLOP_PLAN_SHA256 is None or CICLOP_FILE_SHA256 is None:
        raise PlanNotFrozenError(
            "the plan exists but CICLOP_PLAN_SHA256 and CICLOP_FILE_SHA256 in ciclop.py are unset. "
            "Set both in the commit that adds the plan, so the history shows it was frozen "
            f"before the first score (step 5 of {LOCK_DOCUMENT})."
        )
    observed = sha256_of_plan(target)
    if observed != CICLOP_PLAN_SHA256:
        raise PlanNotFrozenError(
            f"{target.name} has sha256 {observed}, and the pinned plan is {CICLOP_PLAN_SHA256}. "
            "The plan was edited after it was frozen. Restore it, or enter the change in the "
            f"deviations log of {LOCK_DOCUMENT} and re-pin."
        )
    plan = json.loads(target.read_text(encoding="utf-8"))
    resolved = Path(data_path) if data_path is not None else config.get_data_raw_dir() / plan["file"]["name"]
    fingerprint = hdb5.fingerprint_file(resolved, read_shape=False)
    expected = (CICLOP_FILE_SHA256, CICLOP_FILE_N_BYTES)
    if (fingerprint.sha256, fingerprint.n_bytes) != expected or fingerprint.sha256 != plan["file"]["sha256"]:
        raise hdb5.DatasetIntegrityError(
            f"CICLOP integrity check failed for {resolved}.\n"
            f"  pinned   sha256 {CICLOP_FILE_SHA256} ({CICLOP_FILE_N_BYTES} bytes)\n"
            f"  plan     sha256 {plan['file']['sha256']}\n"
            f"  observed sha256 {fingerprint.sha256} ({fingerprint.n_bytes} bytes)\n"
            "The plan was frozen on one specific set of bytes and describes no other."
        )
    return plan, resolved


def frame_from_plan(plan: dict[str, Any], data_path: Path | str) -> pd.DataFrame:
    """Rebuild the analysed rows from the plan, and refuse if they are not the rows it froze."""
    raw = load_ciclop_raw(data_path, plan["mapping"].get("read"))
    cleaned = analysis_frame(tokamak_rows(canonical_frame(raw, plan["mapping"])), tuple(plan["frozen_features"]))
    observed = {str(k): int(v) for k, v in cleaned[hdb5.TOKAMAK_LABEL_COLUMN].value_counts().items()}
    if observed != plan["complete_rows_per_device"]:
        raise AssertionError(
            "the rows rebuilt from the plan are not the rows the plan froze:\n"
            f"  frozen  {plan['complete_rows_per_device']}\n  rebuilt {observed}"
        )
    return cleaned


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CICLOP: schema pass and plan freezing. Fits nothing.")
    commands = parser.add_subparsers(dest="command", required=True)

    schema = commands.add_parser("schema", help="per-column summaries of the delivered file")
    schema.add_argument("path", type=Path)
    schema.add_argument("--sheet", default=0)
    schema.add_argument("--header-row", type=int, default=0)
    schema.add_argument("--out", type=Path, default=None, help="also write the report as JSON")

    freeze = commands.add_parser("freeze", help=f"apply the lock to the file and write {PLAN_FILENAME}")
    freeze.add_argument("path", type=Path)
    freeze.add_argument("--mapping", type=Path, required=True, help="the hand-written mapping, as JSON")
    freeze.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "schema":
        fingerprint = hdb5.fingerprint_file(args.path, read_shape=False)
        raw = load_ciclop_raw(args.path, {"sheet": args.sheet, "header_row": args.header_row})
        report: dict[str, object] = {
            "file": {"name": args.path.name, "sha256": fingerprint.sha256, "n_bytes": fingerprint.n_bytes}
        }
        report.update(schema_report(raw))
        print(json.dumps(report, indent=2, default=str))
        if args.out is not None:
            write_json_strict(args.out, report)
        return

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    plan = freeze_plan(args.path, mapping)
    written = write_plan(plan, args.out)
    print(f"Wrote {written}")
    print(f"  frozen features ({len(plan['frozen_features'])} of 9): {plan['frozen_features']}")
    for column, reason in plan["omitted_features"].items():
        print(f"  omitted {column}: {reason}")
    print(f"  complete tokamak rows: {plan['n_complete_rows']}")
    print(f"  eligible devices ({len(plan['eligible_devices'])}): {plan['eligible_devices']}")
    print(f"  evaluable: {plan['evaluable']}" + "".join(f"\n    {r}" for r in plan["not_evaluable_because"]))
    print("\nNow set, in ciclop.py, in the commit that adds the plan:")
    print(f'  CICLOP_FILE_SHA256 = "{plan["file"]["sha256"]}"')
    print(f"  CICLOP_FILE_N_BYTES = {plan['file']['n_bytes']}")
    print(f'  CICLOP_PLAN_SHA256 = "{sha256_of_plan(written)}"')


if __name__ == "__main__":
    main()
