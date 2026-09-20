# Putting the CICLOP result into the manuscript

This is the procedure for the last step of
[ciclop-replication-lock.md](ciclop-replication-lock.md), once
`results/ciclop.json` exists. Nothing here can be done before that, and the test
suite enforces it: until the artifact exists, `tests/test_ciclop_tables.py` fails
if `paper.tex` or `supplementary.tex` mentions CICLOP at all.

## Numbers are pasted, never typed

```bash
python3 tools/ciclop_tables.py      # writes build/ciclop/ and prints it
```

That writes the three tables for the supplement (the column mapping, the
population by device, and the result beside the HDB5 rerun) and `facts.json`,
which holds every quantity the artifact carries, formatted once, as it is to be
printed.

Every CICLOP passage in either document goes between two comment lines:

```latex
% ciclop:begin
... text, tables, captions ...
% ciclop:end
```

The test suite then requires two things. Every mention of CICLOP outside the
bibliography sits inside a marked passage. And every numeral inside one is a
value in `facts.json`. A number the code did not generate fails the suite, so
there is one copy of each value and it is the artifact's.

What the scan leaves alone: names that contain digits (HDB5, IPB98(y,2), S17,
JT-60U, SHA-256), table and figure layout, and integer powers in units such as
`10^{19}`. A decimal exponent is treated as data. Small counts that are not
facts should be spelled out.

## Where things go

The supplement has 16 sections, so CICLOP is appended as S17 and no existing
pointer moves. Add `17` to `EXPECTED_POINTERS` in
`tests/test_paper_submission.py`. The abstract has a ceiling of 290 words
(`ABSTRACT_WORD_CEILING`), and the abstract changes only if the verdict is a
clean one.

`results/ciclop.json` and its figure change `results/`, so the paper's pinned
commit moves again. The order is in `paper/README.md`: commit `results/` on its
own, put that hash in `paper.tex`, rebuild both PDFs, then commit the paper.

Add `python3 analysis_ciclop.py` to `make results` only if the file can be
fetched unattended. If it stays behind a login, leave it out, and say in Data
availability that this one input cannot be fetched by `make reproduce`.

## What the text may say, by verdict

The verdict is `facts.json["verdict"]`. The wording follows it and nothing else.

| Verdict | Main text | Abstract |
| --- | --- | --- |
| `reproduces_inversion` | State it, then quantify it at once: devices, pulses, both splits, W of k, the gap and its interval. | One clause, if it fits. |
| `degradation_without_inversion` | "CICLOP reproduced the transfer degradation but not the literal ranking inversion." Say whether the forest failed to win interpolation, which the manuscript already names as the prerequisite. | Leave it out. |
| `no_important_differential_degradation` | Say so plainly, in the main text and not only the supplement. Narrow the Discussion and the Limitations. | Leave it out. |
| `contradicts` | In the main text. Give the differences between the datasets that can be shown: feature coverage, regime selection, distances, target definition, machines. Propose no mechanism. | Leave it out. |
| `cannot_be_evaluated` | No replication is claimed. One sentence in Limitations on why the candidate database did not support the comparison. | Leave it out. |

Whatever the verdict, the text keeps four distinctions. CICLOP was assembled
separately, and its devices partly overlap HDB5's. It repeated the split
comparison, the paired ranking and the distance diagnostic, and it did not repeat
the conformal intervals, the Gaussian-process experiment or the size cut. It is
small and was selected for long pulses. The ITER-size-matched cut still rests on
HDB5 alone.

The Limitations sentence that the fusion evidence rests on one ITPA file is
replaced only if CICLOP could be evaluated, whichever way it came out, and then
by a sentence that says what the second dataset is and is not.

## References, verified against Crossref on 20 September 2026

`tests/test_paper_bibliography.py` requires every entry in `references.bib` to be
cited and every `\bibitem` to be numbered in order of first citation, so these go
in with the text that cites them and not before.

```bibtex
@article{litaudon24,
  author  = {Litaudon, X. and Bosch, H.-S. and Morisaki, T. and Barbarino, M.
             and Bock, A. and Belonohy, E. and others},
  title   = {Long plasma duration operation analyses with an international
             multi-machine (tokamaks and stellarators) database},
  journal = {Nuclear Fusion},
  volume  = {64},
  number  = {1},
  pages   = {015001},
  year    = {2024},
  doi     = {10.1088/1741-4326/ad0606},
}

@article{litaudon26,
  author  = {Litaudon, X. and Lerche, E. and Grulke, O. and Holcomb, C. T.
             and Huang, J. and Jakubowski, M. and others},
  title   = {Investigating long-duration plasma operation with the international
             multi-machine {CICLOP} database},
  journal = {Nuclear Fusion},
  volume  = {66},
  number  = {9},
  pages   = {095001},
  year    = {2026},
  doi     = {10.1088/1741-4326/ae89cc},
}
```

```latex
\bibitem{litaudon24} X. Litaudon \emph{et al.}, ``Long plasma duration operation
analyses with an international multi-machine (tokamaks and stellarators)
database,'' \emph{Nucl.\ Fusion} \textbf{64}, 015001 (2024).
\href{https://doi.org/10.1088/1741-4326/ad0606}{doi:10.1088/1741-4326/ad0606}
\bibitem{litaudon26} X. Litaudon \emph{et al.}, ``Investigating long-duration
plasma operation with the international multi-machine CICLOP database,''
\emph{Nucl.\ Fusion} \textbf{66}, 095001 (2026).
\href{https://doi.org/10.1088/1741-4326/ae89cc}{doi:10.1088/1741-4326/ae89cc}
```

The 2024 paper was published online on 16 November 2023 and belongs to the
January 2024 issue, which is the year it is cited under. Crossref lists 28
authors for it, the last being the group "JET contributors", and 30 for the 2026
paper, the last being the CICLOP group itself. Both are CC BY 4.0. Neither is a
conference paper, and the FEC 2025 proceedings version of the second is not
cited.

An acknowledgement of the data providers goes in the Acknowledgements, worded so
that it does not imply the IAEA, the IEA or the CICLOP group endorse the
analysis. Take the wording of any required citation from the terms of use that
come with the file.
