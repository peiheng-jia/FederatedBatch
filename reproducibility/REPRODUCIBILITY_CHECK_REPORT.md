# Reproducibility and release-candidate audit

Audit updated: 26 October 2026. Scope: source/configuration inspection,
saved-artifact contracts, synthetic CPU training/inference, saved-metric
statistics round-trips, Figure 2 saved-input rendering, and lightweight
accounting/assembly tests. No canonical-seven training, raw-dataset evaluation,
privacy attack, or SPU computation was run. Manuscript baselines were unchanged.

## Static verification in WSL

- Environment: WSL Ubuntu 22.04; Git 2.34.1; Conda 25.9.1. The existing native environment has Python 3.10.13, PyTorch 2.1.2 and CUDA 11.8, with UMAP/Matplotlib drift from pins. The existing FedscGen environment has Python 3.10.19/PyTorch 1.13.1 and passes imports when its Conda C++ runtime is selected through `LD_LIBRARY_PATH`. No clean installation was tested. Baseline Git commit is `66ae21ce3b7af4f4a35cbc355649a9ec697cd358`, branch `main`.
- Syntax/configuration: the latest traversal parsed 139 Python, 23 JSON and 14 YAML files. Counts include generated smoke JSON and are not release-file counts.
- Imports: `federatedbatch` and `concord` imported successfully using `PYTHONPATH=src`.
- CLI surfaces: Figure 1-4 experiment drivers, Figure 2-5 renderers, the Figure 3 coordinate generator, Figure 4 exporter, Figure 5 attack/reconstruction adapters, accounting, and plot-input assembler completed `--help`. The SPU script could not import in the native environment because JAX/SPU are absent.
- README redraw inputs: Figure 2 summary, Figure 3 metrics plus the three bundled coordinate tables, Figure 4 preprocessing-fixed summaries, and Figure 5 plot inputs are present. Figure 3 renderer defaults to saved coordinate tables under `results/plot_inputs/figure3/`; no UMAP fitting path is used for redraw.
- Figure 3G: source keeps PCA's seven seed-specific evaluations and sample SD (`ddof=1`); existing size and color-saturation encodings were not altered in release preparation.
- Figure 4: final summaries identify the preprocessing-fixed analysis. The final figure directory has only automated B/C panels; A is author-managed. No Figure 4D output is included.
- Figures 1-5 saved-result contract test: PASS. It verified canonical seed sets, method/dataset coverage, test-family row counts, Figure 2 trajectories, Figure 3 coordinate schemas, Figure 4's 252 preprocessing-fixed scenarios and A/B/C input structure, and balanced Figure 5 membership/nonmembership counts.
- Figures 1–5: curated exports are present where automated panels are part of the release. Figure 1A and Figure 4A are not included because they are author-managed and no final editable source is registered.
- Licenses/provenance: FedscGen and CONCORD notices remain. `baselines/fedscgen/upstream/` is the authoritative checksummed source at commit `b266f744fcc0d97ac0db25f00316b6a080929d0a`; its local compatibility patch and BSD-3-Clause component notice are recorded. The duplicate `third_party/FedscGen/` tree is retained locally for audit history but ignored and excluded from the release inventory.

## Path, archive, and content audit

- Figure 2's legacy evaluator loader now points to the staged helper under `external_artifacts/figure1/scripts/`. The Figure 1 R scripts' color helper now points to the copied release-local file under `external_artifacts/figure1/figures/figure1/r/`.
- Remaining old project locations occur only as provenance/source-history strings or in standalone legacy Figure 1 recomputation copies. Those old copies target obsolete data/seed layouts and are not documented as current commands. They are not loaded by the result renderers.
- Figure 3 UMAP coordinates are bundled. Figure 1's fixed PCA cache, features, and source rows are bundled as documented in `artifact_manifest.tsv`.
- No `.venvs`, temporary logs, pilot/debug files, or release-tracked files larger than 100 MB were found. Local interpreter caches remain excluded by `.gitignore`; they were not added to the integration changes.
- The baseline public commit contains 218 files. Cache bytecode, the duplicate `third_party/FedScGen/` tree, legacy Figure 2 run artifacts, obsolete Figure 1 previews, and unmanifested historical outputs remain excluded by ignore rules. New integration files are intentionally uncommitted pending review.
- Figures 1-4 now have repository-local staged experiment chains. Figure 3/4 commands select only the requested figure, and finalization uses the final statistical families without obsolete renderers. Figure 5 has separate explicit-comparator preparation/training, parameterized release attacks/reconstruction, accounting, SPU inputs and assembly; fresh attacks and SPU remain runtime-unverified.
- Figure 2 renderer annotations now read Holm-adjusted p values from the selected statistics CSV instead of embedding locked manuscript p values in source.
- Figure 5 analytic accounting reproduced the bundled epsilon grid exactly in a temporary test (`sigma=2`, `delta=1e-5`: `epsilon=1.9930914044151202`). The plot-input assembler completed against bundled inputs in `/tmp`; neither test changed manuscript artifacts.

## Runtime and wiring checks

- `scripts/smoke/run_native_training_smoke.py`: PASS on synthetic CPU input
  (64 training cells, two clients, two rounds, one local epoch). PCA, Harmony,
  centralized CONCORD, NoQueue, clipped sigma=0 and private sigma=2 training
  all ran. Sixteen independent inference cells passed shape/finiteness checks;
  clipping and noise diagnostics confirmed those paths ran.
- `scripts/smoke/run_fedscgen_training_smoke.py`: PASS (64 training cells,
  16 held-out cells, two clients, one round, one local epoch). Checkpoint
  reload preserved embeddings exactly; direct and corrected inference passed.
  The synthetic input includes the upstream-required `batch` annotation.
- The FedscGen adapter now supplies the pinned upstream's package import path
  itself, so standalone and canonical-driver calls use the same local source.
  No upstream code, aggregation policy or training algorithm was changed.
- `scripts/smoke/validate_workflow_wiring.py`: PASS for figure isolation,
  explicit training gate, PCA n=7, 72 Figure 3 tests and 12 Figure 4 tests.
  Finalizers round-tripped bundled metrics in temporary directories; maximum
  raw-p differences were 9.37e-14 (Figure 3) and 6.22e-15 (Figure 4).
- `smoke --kind all` runs static/contracts only. Training requires explicit
  `--kind runtime` or the documented standalone FedscGen smoke command.
- Reports are under ignored `results/smoke/`. The final FedscGen adapter test
  is `fedscgen_training_adapter_verified/runtime_report.json`; native results
  are `native_training/runtime_report.json`; wiring is `workflow_wiring.json`.
- The CSV comparison helper passed an identity check on 90 Figure 3 summary
  rows with zero differences; its output is separate from manuscript inputs.
  A real fresh-versus-reference comparison awaits fresh model outputs.
- Static preflight reports source readiness for all five figures but BLOCKED
  data readiness for all five. Figure 5 additionally reports the SPU gap;
  these blocked statuses are retained, not converted to passes.

## Unresolved requirements

- Fresh end-to-end reproduction is **NOT VERIFIED**. Static checks and renderer `--help` do not exercise workflows or prove numerical results.
- Processed input datasets are not staged in the release. Historical copies of all three were found and their size/SHA-256 match the manifest and actual-run provenance; see `data/README.md`. Public API retrieval remains blocked by HTTP 403, with historical version/direct file IDs unverified. Fresh checkpoints and embeddings are workflow outputs rather than unpublished mandatory inputs. Clean GPU/R/FedscGen/SPU runtime specifications remain unresolved.
- Figure 2 saved-input rendering passed. Clean-checkout rendering, real-data Figure 1-4 workflows, fresh privacy attacks, R rendering and SPU execution remain unverified.
- Figure 1A and Figure 4A need author-provided panels. Figure 4 B/C use the bundled preprocessing-fixed results. WSL has Rscript, but `ggplot2`, `dplyr`, and `cowplot` are absent.
- The release is not end-to-end reproducible from a clean checkout until data retrieval and supported native, FedscGen, R, and SPU environments pass runtime smoke tests. Figure 1A and Figure 4A still require author-provided assets.
- Baseline commit `66ae21ce3b7af4f4a35cbc355649a9ec697cd358` exists; the present integration changes are uncommitted. No GitHub remote release/tag or Zenodo DOI exists.

## Status summary

| Check | Status |
|---|---|
| Python AST, JSON/YAML parsing in WSL | PASS |
| Core package imports in WSL | PASS |
| Figure 1-5 non-SPU CLI help and documented option names | PASS |
| Saved-result release contracts for Figures 1-5 | PASS |
| Figure 3 coordinate inputs and Figure 4 corrected summaries present | PASS |
| Figure rendering from a clean checkout | NOT RUN |
| Native synthetic shared training/inference | PASS; real-data figure workflows untested |
| FedscGen synthetic training/checkpoint/correction | PASS; real-data LOBO untested |
| Figure 3/4 selection and final-statistics wiring | PASS; saved metrics only |
| Historical input size/SHA-256 | PASS for all three files; not copied into the release |
| Data-preparation safety | PASS, 8 synthetic tests; live metadata request blocked by HTTP 403 |
| Real-data training/inference/evaluation | NOT RUN; release inputs not staged and clean environments unvalidated |
| End-to-end reproduction | NOT VERIFIED |
| Git release candidate | Baseline commit exists; integration changes uncommitted |
| GitHub/Zenodo publication readiness | SOURCE COMMIT CANDIDATE; ARCHIVAL REPRODUCTION NOT READY |

Static success is not a claim of end-to-end reproduction.
