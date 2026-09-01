# Repository layout and module ownership

What each file is for, which module owns which decision, and the
repository-level caveats. The limitations that bear on the *reported results*
are separate, and live in
[results/RESULTS.md](../results/RESULTS.md#limitations).

## Repository Structure

Grouped by what it is for rather than alphabetically: the real-data study first, the shared plumbing next, the neutron-yield infrastructure last.

```text
FusionFlux/
│
│   # real-data confinement study: everything the reported results come from
├── hdb5.py                          # ITPA HDB5 pipeline: download, pin check, cleaning, model zoo, CLI
├── scaling_law.py                   # from-scratch least squares; fits and audits scaling laws
├── analysis_scaling_law.py          # Results 1 to 3: rank audit, IPB98 refit, conditioning,
│                                    #   bootstrap resolution (2b) and the solver sweep (2c)
├── analysis_extrapolation.py        # Result 4: leave-one-tokamak-out study and figure
├── analysis_flexibility_sweep.py    # Result 4e: polynomial degree against ridge penalty
├── analysis_size_extrapolation.py   # Result 5: size-ordered cut at the ITER-matched jump
├── analysis_hybrid.py               # Result 6: power law plus a damped residual correction
├── analysis_conformal.py            # Result 7: split-conformal coverage under each split
├── dimensional.py                   # Connor-Taylor constraints, derived from the group definitions
├── spectral.py                      # prior-shrinkage family aimed at Result 3's weak direction
├── conformal_shift.py               # machine-level and distance-scaled interval calibration
├── replication.py                   # DB5.2.3 revision, pinned; the two STD5-disjoint arms
├── forecast.py                      # device design points and the locked prediction record
├── analysis_dimensional.py          # Results 8 and 9: physics as a constraint, then as a prior
├── analysis_conformal_shift.py      # Result 10: repairing the interval collapse, and its limit
├── analysis_replication.py          # Result 11: the reversal on rows STD5 does not contain
├── analysis_forecast.py             # Result 12: SPARC, JT-60SA and ITER, written down in advance
├── lawson.py                        # standalone Lawson criterion utility
├── results/
│   ├── RESULTS.md                   # the writeup: every claim, table and limitation
│   ├── extrapolation.png            # Result 4 figure, plus its .json/.csv companions
│   ├── flexibility_sweep.png        # Result 4e figure, plus its .json/.csv companions
│   ├── size_extrapolation.png       # Result 5 figure, plus its .json/.csv companions
│   ├── hybrid.png                   # Result 6 figure, plus its .json/.csv companions
│   ├── conformal.png                # Result 7 figure, plus its .json/.csv companions
│   ├── dimensional.png              # Results 8 and 9 figure, plus its .json/.csv companions
│   ├── conformal_shift.png          # Result 10 figure, plus its .json/.csv companions
│   ├── replication_scores.csv       # Result 11: both arms under both splits
│   ├── forecast.json                # Result 12: the locked record, with its content digest
│   ├── singular_value_spectrum.png  # Results 1 to 3 figure
│   ├── solver_conditioning.png      # Result 2c: forward error against condition number
│   ├── analysis.json                # rank audit, refit exponents, conditioning, solver sweep
│   ├── bootstrap_resolution.csv     # Result 2b: exponent intervals at all three units
│   └── ipb98_refit_exponents.csv    # refit against published, with bootstrap intervals
│
├── paper/
│   ├── paper.tex                    # nine-page writeup; build with `tectonic paper/paper.tex`
│   └── README.md                    # how to build it, and the Zenodo DOI flow
├── site/
│   ├── page.template.html           # the one-page interactive summary
│   └── build_page.py                # fills it from results/; writes site/index.html
├── docs/
│   ├── usage.md                     # installation and the CLI for every pipeline
│   ├── testing.md                   # suite layout, the CI gate, the reproducibility check
│   ├── repository.md                # this file: layout, ownership, repo-level caveats
│   └── neutron-yield-pipeline.md    # operating detail for the synthetic-data infrastructure
├── tools/
│   └── compare_results.py           # numeric diff of a regenerated results/ against the committed one
│
│   # shared plumbing, used by both pipelines
├── config.py                        # paths, column config, physics constants and tolerances
├── storage.py                       # atomic file writes and JSON/CSV helpers
├── validation.py                    # physics input validation primitives
│
│   # neutron-yield infrastructure (synthetic demo data, no scientific claim)
├── train_model.py                   # CLI entrypoint and compatibility facade over the package
├── neutron_yield/                   # the pipeline itself, packaged away from the science
│   ├── __init__.py                  # states the scope: infrastructure, not a physical claim
│   ├── fusionflux_cli.py            # argparse CLI behind the `fusionflux` console script
│   ├── features.py                  # alias mapping, validation, feature engineering, contract
│   ├── artifact_model.py            # saved-model wrapper with preprocessing + clipping guardrails
│   ├── training.py                  # training orchestration and holdout evaluation
│   ├── training_artifacts.py        # per-run path layout, staged write, atomic publish
│   ├── training_registry.py         # preprocessor and candidate model factories
│   ├── training_reports.py          # residual and feature-importance plots
│   ├── training_split.py            # random and grouped holdout / CV split selection
│   ├── inference.py                 # single/batch prediction flow, public inference API
│   ├── inference_artifacts.py       # artifact schema, metadata parsing, run-manifest writers
│   └── inference_selection.py       # artifact discovery, default selection, loading
│
├── Makefile
├── pyproject.toml
├── requirements.txt
├── constraints.txt
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml
│       ├── pages.yml
│       └── reproduce.yml            # regenerates results/ from the raw data and diffs it
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_hdb5.py                 # confinement pipeline, on small in-memory frames
│   ├── test_dataset_integrity.py    # the HDB5 content pin, including how it fails
│   ├── test_scaling_law.py          # the three hand-written solvers against a known answer
│   ├── test_solver_conditioning.py  # Result 2c: the kappa^2 vs kappa slope separation
│   ├── test_bootstrap_resolution.py # Result 2b: which exponents widen, and why
│   ├── test_extrapolation.py        # Result 4, including the tree ceiling bound
│   ├── test_flexibility_sweep.py    # Result 4e, incl. the sklearn-equivalence cross-check
│   ├── test_size_extrapolation.py   # Result 5, including that the cut is data-picked
│   ├── test_hybrid.py               # Result 6, incl. the bounded/unbounded correction contrast
│   ├── test_conformal.py            # Result 7, incl. the finite-sample conformal rank
│   ├── test_analysis_scaling_law.py # the Results 1 to 3 analysis script
│   ├── test_reported_numbers.py     # every headline number, bound to its artifact
│   ├── test_supported_python_versions.py  # the badge and CI matrix, bound to requires-python
│   ├── test_compare_results.py      # the comparator behind the reproduce gate
│   ├── test_release_metadata.py     # .zenodo.json, which nothing else consumes
│   ├── test_lawson.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_inference.py
└── data/
    ├── raw/
    │   ├── hdb5_std5.csv            # not committed; fetched via `python3 hdb5.py download`
    │   └── synthetic_nuclear_fusion_experiment.csv   # sample/reference copy only
    └── processed/
        ├── hdb5_confinement/
        │   ├── confinement_model.joblib
        │   ├── confinement_metrics.csv
        │   └── confinement_metadata.json
        ├── latest_training_run.json
        └── runs/
            └── <training_run_id>/
                ├── feature_importance.csv
                ├── fusion_dataset_processed.csv
                ├── synthetic_training_input.csv   # only for --allow-synthetic runs
                ├── metrics.csv
                ├── physics_mismatch_flags.csv
                ├── test_predictions.csv
                ├── training_metadata.json
                ├── models/
                │   └── best_model.joblib
                └── plots/
                    ├── <best_model>_residuals.png
                    └── feature_importance.png
```

## Module Ownership

Each pipeline is split into a thin orchestration module plus focused helpers, so the pieces can change independently without an import cycle.

Real-data confinement study:

- `hdb5.py` owns the entire real-data confinement-time pipeline (download, cleaning, features, model zoo, training, prediction, and its own CLI). It shares only `config.py` and `storage.py` with the neutron-yield pipeline.
- `scaling_law.py` owns the from-scratch linear algebra: the three classical least-squares solvers, design-matrix conditioning analysis, scaling-law fitting, and bootstrap confidence intervals. It deliberately does not call scikit-learn.
- `dimensional.py` owns the Connor-Taylor constraint hierarchy. It derives each rung's constraint matrix from the definitions of rho*, beta and nu* rather than hard-coding exponent vectors, and exposes the constrained fit as a scikit-learn estimator so it drops into the same zoo and the same three splits as everything else. It borrows `scaling_law.solve_constrained_lstsq` rather than adding a solver.
- `spectral.py` owns the prior-shrinkage family: shrinking a scaling law toward IPB98(y,2)'s published exponents along the singular directions the data cannot resolve. It is the control Result 8 is measured against, not a competing recommendation.
- `conformal_shift.py` owns the two repaired interval schemes of Result 10, machine-level calibration and distance-scaled nonconformity. It delegates the `split` baseline to `hdb5._conformal_arm` rather than reimplementing it, so the comparison is against Result 7's exact procedure.
- `replication.py` owns the full DB5.2.3 revision: its own SHA-256 pin, the unit conversions to STD5's units, the row match that establishes disjointness, and the ITER89-P L-mode baseline the non-H arm needs. It reuses `hdb5.map_to_canonical` for cleaning, because a replication that cleaned its data differently would not be replicating anything.
- `forecast.py` owns the three device design points, the tree-ensemble bound check, and the locked prediction record with its content digest.
- `lawson.py` owns the standalone triple-product and ignition-ratio calculation, and is the one physics utility both pipelines can borrow from.

Shared and entrypoints:

- `config.py` owns paths, column configuration, physics constants, and tolerances.
- `storage.py` owns atomic file writes and the JSON/CSV output helpers.
- `validation.py` owns the physics input validation primitives used by both pipelines and by `lawson.py`.

Neutron-yield infrastructure, training side:

- `neutron_yield/training.py` owns training orchestration, holdout evaluation, metric/metadata assembly, and artifact writing.
- `neutron_yield/training_split.py` owns holdout and cross-validation split selection, including the exact bounded subset-sum search for row-targeted grouped holdouts and its linear greedy fallback for very large group sets.
- `neutron_yield/training_registry.py` owns the preprocessing transformer and the candidate model factories that training cross-validates and selects among.
- `neutron_yield/training_artifacts.py` owns the per-run path layout plus the staged-write and atomic-publish/cleanup logic for a run directory.
- `neutron_yield/training_reports.py` owns the best-effort diagnostic plots; matplotlib and seaborn are imported lazily, and failures here degrade to "reports skipped" instead of discarding a successful run.

Neutron-yield infrastructure, inference side:

- `neutron_yield/inference.py` owns the single-case and batch prediction flow and re-exports the public inference API, so `from neutron_yield import inference` stays the one stable entry point.
- `neutron_yield/inference_artifacts.py` owns the versioned artifact schema, the strict metadata parsers/validators, and the run-manifest writers that training persists.
- `neutron_yield/inference_selection.py` owns artifact discovery, compatibility ranking under the configured selection mode, and deserialization of the first loadable candidate.

Neutron-yield infrastructure, both sides:

- `neutron_yield/features.py` owns alias mapping, temperature normalization, feature engineering, and the versioned preprocessing contract.
- `neutron_yield/artifact_model.py` owns the `FusionFluxModelArtifact` wrapper that enforces preprocessing compatibility and clips negative predictions.
- `neutron_yield/fusionflux_cli.py` owns the argparse CLI behind the installed `fusionflux` console script.
- `train_model.py` stays at the repository root as the CLI entrypoint and a compatibility facade over the package, so every documented `python3 train_model.py ...` command and every `train_model.<name>` import keeps working across the move.

## Notes / Limitations

The limitations that bear on the reported results are stated in full in [results/RESULTS.md](../results/RESULTS.md#limitations). What follows is repository-level.

**Real data and physics**

- The HDB5 dataset is third-party scientific data. It is fetched on demand from OSF and is not redistributed in this repository, so `data/raw/hdb5_std5.csv` is gitignored. Commands that need it and cannot find it raise a `FileNotFoundError` naming the OSF source and the `--dataset-path` override; run `python3 hdb5.py download` (or `train --download-if-missing`) to fetch it.
- The confinement pipeline reports against the analytic IPB98(y,2) scaling law rather than against a mean baseline alone. Treat `beats_physics_baseline` in `confinement_metadata.json` as the headline result: a model that does not beat published physics on grouped cross-validation has not learned anything useful.
- The Lawson utility uses a simplified D-T ignition threshold from `config.py` and is best treated as a compact educational or screening tool rather than a full plasma physics simulator.

**Neutron-yield infrastructure**

- Synthetic data is useful for demos and pipeline validation, but it is not a substitute for real experimental fusion data. The training CLI only uses it when you pass `--allow-synthetic`.
- Model quality depends on the dataset, feature coverage, and split behavior; holdout artifacts are for reporting, while the saved production model is refit on all prepared rows.
- The prediction CLIs expect a trained model and metadata file unless you provide custom `--model-path` and `--metadata-path` values. They validate the saved preprocessing contract against the current runtime code before scoring. Explicit artifact selection requires exact recorded runtime versions, while default artifact selection may accept limited compatible drift with warnings.
- Batch CSV prediction only streams non-grouped inputs. Grouped time-series inputs are read as a whole file so shot-level aggregation can see every row for a shot.
- The strict preprocessing contract is intentional. In this repo, silent feature drift is more dangerous than the inconvenience of regenerating artifacts, because the goal is fail-fast behavior around physics results. The contract is an explicit, versioned structural description (columns, feature schema, physics constants and tolerances); it deliberately does not fingerprint function source or bytecode, since that broke on harmless reformatting and forced spurious retrains. Bump `PREPROCESSING_CONTRACT_VERSION` in `neutron_yield/features.py` whenever you change preprocessing semantics.

**Both**

- The test suite exercises many pipeline paths, but ML changes should still be validated by rerunning training and reviewing the saved artifacts.
