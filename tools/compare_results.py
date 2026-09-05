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

A third class is named individually in ``NON_PORTABLE`` below: values whose last
digits belong to the host's arithmetic rather than to the study, and which
therefore do not survive a change of LAPACK. The tolerance here absorbs BLAS
summation-order jitter; it does not absorb Accelerate against OpenBLAS. Those
paths are listed one at a time, with the reason and with what still guards the
finding underneath them, rather than being swept up by a looser global
tolerance that would blind the gate everywhere else.

Exit status is 0 when the two directories agree and 1 when they do not, with the
disagreements printed most significant first.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Union

import pandas as pd

REL_TOL = 1e-6
# Values this close to zero compare as equal regardless of relative difference,
# since relative tolerance is meaningless around zero.
ABS_TOL = 1e-12

# Fields that legitimately differ between runs or between machines.
#
# ``generated_on`` is the date an artifact was built, carried by
# ``forecast.json`` and ``predictor.json`` so a reader knows how old a locked
# prediction is. It changes whenever the analysis is rerun and says nothing
# about whether the numbers moved, which is what this comparator is for. The
# substance of those two files is pinned separately and not by the date: the
# forecast carries a SHA-256 over its own rows and the dataset they came from,
# and every value in both is compared here on its own.
VOLATILE_KEYS = frozenset({"path", "seconds_per_solve", "generated_on"})
VOLATILE_SUFFIXES = ("_seconds", "_path", "_at_utc")


def _is_volatile(key: str) -> bool:
    return key in VOLATILE_KEYS or key.endswith(VOLATILE_SUFFIXES)


# Fields whose last digits are a property of the host's arithmetic rather than
# of the study.
#
# REL_TOL absorbs BLAS summation-order jitter, which is order 1e-15. It does not
# absorb a change of LAPACK. results/ was generated on macOS, which links
# Accelerate; a Linux runner uses OpenBLAS, and on byte-identical seeded inputs
# the two disagree by far more than 1e-6 in the places listed below. They are
# named individually rather than accommodated by a looser global tolerance,
# because a tolerance wide enough for them would stop this gate catching a real
# change anywhere else.
#
# Keys are dotted paths into a results file with list indices written as ``[]``,
# and each covers that path and everything under it. The value is the relative
# tolerance to apply instead of REL_TOL, or None to exclude the path outright.
NON_PORTABLE: dict[str, float | None] = {
    # Result 2c measures where float64 breaks down: the forward error of three
    # least-squares solvers against condition number, and the condition number
    # at which Cholesky refuses to return an answer. Its output *is* the host's
    # floating-point arithmetic, so comparing it across LAPACKs is a category
    # error rather than a reproducibility check. On ubuntu-latest the sweep put
    # the Cholesky breakdown at 1e10 against the committed 1e9 and recorded 11
    # failures where the committed run recorded 12; even a second macOS machine
    # moves the median errors by tens of percent.
    #
    # The finding is not left unguarded. tests/test_solver_conditioning.py
    # asserts the portable claims -- that the Cholesky slope is twice the
    # orthogonal solvers', that every curve matches the slope its theory
    # predicts, and that Cholesky is least accurate at every condition number --
    # by computing the sweep itself rather than by reading results/, so it runs
    # on every platform CI covers rather than only where results/ was made.
    #
    # Only the measurements are excluded. The experimental design that produced
    # them stays compared: solver, condition_numbers, n_trials, machine_epsilon
    # and the slope-fit window are all still checked, so a sweep that silently
    # started measuring something else still fails here.
    "analysis.json.solver_conditioning.curves[].median_errors": None,
    "analysis.json.solver_conditioning.curves[].n_failures": None,
    "analysis.json.solver_conditioning.curves[].fitted_slope": None,
    "analysis.json.solver_conditioning.breakdown_condition_number": None,
    "solver_conditioning.csv[].relative_forward_error": None,
    "solver_conditioning.csv[].failed": None,
    # |cos| between a residual direction and a printed basis vector, taken from
    # the SVD of a matrix that is rank deficient by 2. Inside a degenerate
    # subspace the singular vectors are fixed only up to a rotation, so which
    # direction LAPACK hands back is its choice and not a property of the data.
    # What Result 1 actually reports -- rank and rank_deficiency -- are integers
    # and stay compared.
    "analysis.json.rank_audit.max_alignment_with_a_printed_basis_vector": None,
    # A SHA-256 over the forecast rows at full float64 precision, so it moves on
    # a one-ULP change in any row: exactly the jitter REL_TOL exists to absorb.
    # It is a bit-exactness check embedded in a tolerance comparison and cannot
    # pass one. Excluding it costs nothing, because every row it covers is
    # compared here individually and to tolerance.
    "forecast.json.content_digest_sha256": None,
    # Orthogonal distance regression is fitted iteratively, and the point it
    # converges to depends on the host's arithmetic. Only m_eff_amu moves --
    # the smallest exponent in the vector and so the worst determined -- and by
    # 4e-6, just past REL_TOL. Compared at 1e-4 rather than dropped: that is
    # still two orders of magnitude tighter than the three significant figures
    # these are reported at, so all eight exponents keep a real check. The two
    # values Result 14 quotes, max_abs_exponent_shift and largest_shift_feature,
    # are derived from these and stay at full REL_TOL.
    "sensitivity.json.errors_in_variables.odr_exponents": 1e-4,
}

_LIST_INDEX = re.compile(r"\[\d+\]")


def _tolerance_for(where: str) -> float | None:
    """The relative tolerance for a path, or None if it is excluded.

    Walks from the full path up to its root so that an entry covers everything
    beneath it, and normalises list indices so one entry covers every element.
    """
    path = _LIST_INDEX.sub("[]", where)
    while path:
        if path in NON_PORTABLE:
            return NON_PORTABLE[path]
        cut = max(path.rfind("."), path.rfind("["))
        if cut <= 0:
            break
        path = path[:cut]
    return REL_TOL


def _close(a: float, b: float, rel_tol: float = REL_TOL) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=ABS_TOL)


Number = Union[int, float]


def compare_json(baseline: Any, candidate: Any, where: str, out: list[str]) -> None:
    """Walk two decoded JSON documents in parallel, recording disagreements."""
    tolerance = _tolerance_for(where)
    if tolerance is None:
        return

    if isinstance(baseline, dict) and isinstance(candidate, dict):
        for key in sorted(set(baseline) | set(candidate)):
            if _is_volatile(key):
                continue
            child = f"{where}.{key}"
            # Tested here as well as on entry, so that a key that appears or
            # vanishes under an excluded path is excluded too rather than
            # reported as added or removed without ever being descended into.
            if _tolerance_for(child) is None:
                continue
            if key not in baseline:
                out.append(f"{child}: added")
            elif key not in candidate:
                out.append(f"{child}: removed")
            else:
                compare_json(baseline[key], candidate[key], child, out)
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
        if not _close(float(baseline), float(candidate), tolerance):
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
        # Resolved once per column rather than per row: every row of a column
        # shares a path once list indices are normalised, so the answer cannot
        # differ between them.
        tolerance = _tolerance_for(f"{name}[].{column}")
        if tolerance is None:
            continue
        lc, rc = left[column], right[column]
        if pd.api.types.is_numeric_dtype(lc) and pd.api.types.is_numeric_dtype(rc):
            for i, (a, b) in enumerate(zip(lc, rc, strict=True)):
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) or pd.isna(b) or not _close(float(a), float(b), tolerance):
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
