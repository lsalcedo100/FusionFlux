"""Set the version and the archive DOI everywhere they are written down.

Both are spread across five files, and the release workflow's version guard
compares the tag against `pyproject.toml` only. v0.4.1 was tagged with
`pyproject` still reading 0.4.0: the guard failed the build, and Zenodo minted a
DOI anyway, because it archives on release *publication* and does not wait for
the build job. The tag had to be superseded. Editing the five by hand is what
produced that, so this does it in one pass and refuses to do it partly.

    python3 tools/bump_release.py --version 0.4.3
    python3 tools/bump_release.py --doi 10.5281/zenodo.NNNNNNN

The two are separate steps on purpose, because the DOI does not exist until the
release is published, which happens after the version bump is committed and
tagged. The concept DOI, which addresses every version at once, is never
rewritten by either.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONCEPT_DOI = "10.5281/zenodo.22215142"

# (path, pattern, replacement template). The patterns are anchored on the field
# rather than on the old value, so running this twice is not an error and a
# stale value cannot survive by not matching.
VERSION_SITES = (
    ("pyproject.toml", r'(?m)^version = "[^"]+"', 'version = "{v}"'),
    ("CITATION.cff", r"(?m)^version: .+$", "version: {v}"),
    # Lookaheads rather than matched-and-retyped context: these sit in LaTeX
    # prose that gets rewrapped, and a pattern spanning the following comma
    # stops matching the moment a line break lands in it. Rewrapping the zenodo
    # bibitem broke exactly that and the bump refused to run.
    (
        "paper/paper.tex",
        r"(?<=\()v[0-9]+\.[0-9]+\.[0-9]+(?=; the DOI for all versions)",
        "v{v}",
    ),
    ("paper/paper.tex", r"(?<=version )v[0-9]+\.[0-9]+\.[0-9]+(?=,\s+Zenodo)", "v{v}"),
    ("paper/references.bib", r"version   = \{v[0-9]+\.[0-9]+\.[0-9]+\}", "version   = {{v{v}}}"),
)

# The concept DOI is excluded by matching only DOIs that are not it.
_VERSION_DOI = r"10\.5281/zenodo\.(?!22215142)\d+"

DOI_SITES = (
    (
        "paper/paper.tex",
        rf"\\href\{{https://doi\.org/{_VERSION_DOI}\}}\{{doi:{_VERSION_DOI}\}}",
        r"\href{{https://doi.org/{d}}}{{doi:{d}}}",
    ),
    ("paper/references.bib", rf"doi       = \{{{_VERSION_DOI}\}}", "doi       = {{{d}}}"),
)


def _apply(sites, value: str, field: str, root: Path = ROOT) -> int:
    edits = 0
    for relative, pattern, template in sites:
        path = root / relative
        text = path.read_text()
        replacement = template.format(v=value, d=value)

        # A function replacement, not a string: the templates are LaTeX and
        # BibTeX, and their backslashes would otherwise be read as escapes.
        def _fixed(_match: re.Match[str], value: str = replacement) -> str:
            return value

        updated, n = re.subn(pattern, _fixed, text)
        if n == 0:
            raise SystemExit(f"no {field} site matched in {relative}: {pattern}")
        if updated != text:
            path.write_text(updated)
        edits += n
        print(f"  {relative}: {n} site(s)")
    return edits


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="the release version, without a leading v")
    parser.add_argument("--doi", help="the version DOI Zenodo minted, e.g. 10.5281/zenodo.22562235")
    args = parser.parse_args(argv)

    if not args.version and not args.doi:
        parser.error("give --version, --doi, or both")

    if args.version:
        if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
            parser.error(f"expected X.Y.Z, got {args.version!r}")
        print(f"version -> {args.version}")
        _apply(VERSION_SITES, args.version, "version", root)

    if args.doi:
        if not re.fullmatch(r"10\.5281/zenodo\.\d+", args.doi):
            parser.error(f"expected 10.5281/zenodo.NNNNNNN, got {args.doi!r}")
        if args.doi == CONCEPT_DOI:
            parser.error("that is the concept DOI, which addresses all versions and never moves")
        print(f"version DOI -> {args.doi}")
        _apply(DOI_SITES, args.doi, "DOI", root)

    print("\nRebuild the PDFs and re-run `make paper-fresh`, or the committed PDF")
    print("still carries the old version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
