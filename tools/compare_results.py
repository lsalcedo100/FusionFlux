"""Compare two ``results/`` directories numerically rather than byte for byte.

``make reproduce`` regenerates every artifact from the raw data and has to
decide whether anything actually moved. A ``git diff`` cannot answer that:
float64 values are serialized at full precision, and the last one or two digits
jitter between runs because threaded BLAS reductions do not fix summation order.
A byte comparison would therefore fail on every run and the gate would be
ignored within a week.

So this compares values with a relative tolerance instead. The tolerance is
1e-6, which is far looser than the observed jitter (order 1e-15) and far tighter
than the precision anything is reported at (four significant figures at most),
so any change large enough to alter a published digit fails here, and no change
smaller than that does.

Two classes of field are excluded outright rather than compared loosely:

* absolute filesystem paths, which differ between a laptop and a CI runner;
* wall-clock timings, which measure the machine rather than the analysis.

Exit status is 0 when the two directories agree and 1 when they do not, with the
disagreements printed most significant first.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Union

import pandas as pd

REL_TOL = 1e-6
# Values this close to zero compare as equal regardless of relative difference,
# since relative tolerance is meaningless around zero.
ABS_TOL = 1e-12

# Fields that legitimately differ between runs or between machines.
VOLATILE_KEYS = frozenset({"path", "seconds_per_solve"})
VOLATILE_SUFFIXES = ("_seconds", "_path", "_at_utc")


def _is_volatile(key: str) -> bool:
    return key in VOLATILE_KEYS or key.endswith(VOLATILE_SUFFIXES)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


Number = Union[int, float]


def compare_json(baseline: Any, candidate: Any, where: str, out: list[str]) -> None:
    """Walk two decoded JSON documents in parallel, recording disagreements."""
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        for key in sorted(set(baseline) | set(candidate)):
            if _is_volatile(key):
                continue
            if key not in baseline:
                out.append(f"{where}.{key}: added")
            elif key not in candidate:
                out.append(f"{where}.{key}: removed")
            else:
                compare_json(baseline[key], candidate[key], f"{where}.{key}", out)
        return

    if isinstance(baseline, list) and isinstance(candidate, list):
        if len(baseline) != len(candidate):
            out.append(f"{where}: length {len(baseline)} -> {len(candidate)}")
            return
        for i, (b, c) in enumerate(zip(baseline, candidate, strict=True)):
            compare_json(b, c, f"{where}[{i}]", out)
        return

    # bool is a subclass of int and True == 1, so a bool has to be screened off
    # before the numeric branch and compared by type as well as by value.
    # extrapolation.json carries real booleans (ranking_exactly_reversed), and a
    # flag that silently turned into a 1 should read as a change.
    if isinstance(baseline, bool) != isinstance(candidate, bool):
        out.append(f"{where}: {baseline!r} -> {candidate!r}")
        return
    if isinstance(baseline, bool):
        if baseline != candidate:
            out.append(f"{where}: {baseline!r} -> {candidate!r}")
        return

    if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
        if not _close(float(baseline), float(candidate)):
            out.append(f"{where}: {baseline!r} -> {candidate!r}")
        return

    if baseline != candidate:
        out.append(f"{where}: {baseline!r} -> {candidate!r}")


def compare_csv(baseline: Path, candidate: Path, out: list[str]) -> None:
    left = pd.read_csv(baseline)
    right = pd.read_csv(candidate)
    name = baseline.name

    if list(left.columns) != list(right.columns):
        out.append(f"{name}: columns {list(left.columns)} -> {list(right.columns)}")
        return
    if len(left) != len(right):
        out.append(f"{name}: {len(left)} rows -> {len(right)} rows")
        return

    for column in left.columns:
        lc, rc = left[column], right[column]
        if pd.api.types.is_numeric_dtype(lc) and pd.api.types.is_numeric_dtype(rc):
            for i, (a, b) in enumerate(zip(lc, rc, strict=True)):
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) or pd.isna(b) or not _close(float(a), float(b)):
                    out.append(f"{name}[{i}].{column}: {a!r} -> {b!r}")
        else:
            for i, (a, b) in enumerate(zip(lc, rc, strict=True)):
                # An all-missing column reads back as object dtype, and NaN is
                # never equal to itself, so missing has to be screened here too.
                if pd.isna(a) and pd.isna(b):
                    continue
                if a != b:
                    out.append(f"{name}[{i}].{column}: {a!r} -> {b!r}")


def compare_directories(baseline_dir: Path, candidate_dir: Path) -> list[str]:
    out: list[str] = []

    def listing(d: Path) -> set[str]:
        return {p.name for p in d.iterdir() if p.suffix in {".json", ".csv"}}

    left, right = listing(baseline_dir), listing(candidate_dir)
    for missing in sorted(left - right):
        out.append(f"{missing}: no longer generated")
    for added in sorted(right - left):
        out.append(f"{added}: newly generated, not committed")

    for name in sorted(left & right):
        b, c = baseline_dir / name, candidate_dir / name
        if name.endswith(".json"):
            compare_json(json.loads(b.read_text()), json.loads(c.read_text()), name, out)
        else:
            compare_csv(b, c, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="the committed results directory")
    parser.add_argument("candidate", type=Path, help="the regenerated results directory")
    parser.add_argument("--max-report", type=int, default=40,
                        help="how many disagreements to print before truncating")
    args = parser.parse_args(argv)

    differences = compare_directories(args.baseline, args.candidate)
    if not differences:
        print(f"results reproduce: every value agrees to a relative tolerance of {REL_TOL:g}")
        return 0

    print(f"results changed: {len(differences)} value(s) moved by more than {REL_TOL:g} relative\n")
    for line in differences[: args.max_report]:
        print(f"  {line}")
    if len(differences) > args.max_report:
        print(f"  ... and {len(differences) - args.max_report} more")
    print("\nRerun the owning analysis, review the change, and update the prose to match.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
