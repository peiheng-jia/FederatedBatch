# Reproducibility and release-candidate audit

Audit date: 23 October 2026. Scope: source/configuration inspection and static checks only. No model training, inference, metric/statistical evaluation, or figure rendering was run.

## Static verification in WSL

- Environment: WSL Ubuntu 22.04; Git 2.34.1; Python 3.10.12; PyYAML 6.0.2. `conda` is not installed. Git was initialized in this release candidate with branch `main`; the index is empty and no commit exists.
- Syntax and configuration: Python AST parsing passed for 122 `.py` files; 12 JSON and 12 YAML files parsed.
- Imports: `federatedbatch` and `concord` imported successfully using `PYTHONPATH=src`.
- Renderer CLI: Figure 2 result-only, Figure 3 publication, Figure 4 preprocessing-fixed B/C, and Figure 5 variant renderer `--help` commands completed. Their README `--input`/`--output` arguments match the declared options. This checks argument surfaces only, not the plotting operation.
- README redraw inputs: Figure 2 summary, Figure 3 metrics plus the three bundled coordinate tables, Figure 4 preprocessing-fixed summaries, and Figure 5 plot inputs are present. Figure 3 renderer defaults to saved coordinate tables under `results/plot_inputs/figure3/`; no UMAP fitting path is used for redraw.
- Figure 3G: source keeps PCA's seven seed-specific evaluations and sample SD (`ddof=1`); existing size and color-saturation encodings were not altered in release preparation.
- Figure 4: final summaries identify the preprocessing-fixed analysis. The final figure directory has only automated B/C panels; A is author-managed. No Figure 4D output is included.
- Figures 1–5: curated exports are present where automated panels are part of the release. Figure 1A and Figure 4A are not included because they are author-managed and no final editable source is registered.
- Licenses/provenance: FedscGen and CONCORD notices remain. `baselines/fedscgen/upstream/` is the authoritative checksummed source at commit `b266f744fcc0d97ac0db25f00316b6a080929d0a`; its local compatibility patch and BSD-3-Clause component notice are recorded. The duplicate `third_party/FedscGen/` tree is retained locally for audit history but ignored and excluded from the release inventory.

## Path, archive, and content audit

- Figure 2's legacy evaluator loader now points to the staged helper under `external_artifacts/figure1/scripts/`. The Figure 1 R scripts' color helper now points to the copied release-local file under `external_artifacts/figure1/figures/figure1/r/`.
- Remaining old project locations occur only as provenance/source-history strings or in standalone legacy Figure 1 recomputation copies. Those old copies target obsolete data/seed layouts and are not documented as current commands. They are not loaded by the result renderers.
- Figure 3 UMAP coordinates are bundled. Figure 1's fixed PCA cache, features, and source rows are bundled as documented in `artifact_manifest.tsv`.
- A search found no AI conversation, prompt, pasted-text, or generated-content attribution artifacts. Author and third-party attribution was retained.
- No `.venvs`, temporary logs, pilot/debug files, or files larger than 100 MB were found. Eight `__pycache__` directories and 76 `.pyc` files remain physically present because deletion was blocked. `.gitignore` and the release manifest exclude them; verify they are not staged when initializing Git.
- Git's dry-run candidate list contains 218 files; the index remains empty. Cache bytecode, the duplicate `third_party/FedscGen/` tree, legacy Figure 2 run artifacts, obsolete Figure 1 previews, and five unmanifested historical outputs are excluded by ignore rules. The curated 12.2 MB Figure 1 deterministic baseline cache is the only bundled file over 10 MB; it is documented as a redraw/evaluation input, not a dataset or model checkpoint.
- Figure 2 YAML files are protocol metadata, not executable training configuration. Figure 2's original data/checkpoints and several intermediate results are not bundled; fresh training/evaluation is not self-contained.

## Unresolved requirements

- Fresh end-to-end reproduction is **NOT VERIFIED**. Static checks and renderer `--help` do not exercise workflows or prove numerical results.
- External data, checkpoints, complete embeddings, some Figure 5 intermediate artifacts, and portable GPU/runtime specifications are missing or not publicly linked.
- No redraw command was actually run. Clean-checkout rendering, Figure 1/2 training, Figure 3/4 fresh experiments, Figure 4 checkpoint-only re-evaluation, R scripts, and SPU execution remain unverified.
- Figure 1A and Figure 4A need author-provided panels. Figure 4 B/C use the bundled preprocessing-fixed results.
- The release is not end-to-end reproducible from a clean checkout until stable public locations for external datasets/checkpoints and supported environments are provided. Figure 1A and Figure 4A still require author-provided assets.
- No Git commit, GitHub repository/release/tag, or Zenodo DOI exists. Required dataset/checkpoint distribution locations are not established.

## Status summary

| Check | Status |
|---|---|
| Python AST, JSON/YAML parsing in WSL | PASS |
| Core package imports in WSL | PASS |
| Figure 2–5 renderer CLI help and documented option names | PASS |
| Figure 3 coordinate inputs and Figure 4 corrected summaries present | PASS |
| Figure rendering from a clean checkout | NOT RUN |
| Fresh model training/inference/evaluation | NOT RUN; dependencies unavailable in release tree |
| End-to-end reproduction | NOT VERIFIED |
| Git release candidate | INITIALIZED; 218 dry-run candidates; index empty |
| GitHub/Zenodo publication readiness | SOURCE COMMIT CANDIDATE; ARCHIVAL REPRODUCTION NOT READY |

Static success is not a claim of end-to-end reproduction.
