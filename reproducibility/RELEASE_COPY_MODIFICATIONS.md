# Release-copy modifications

This directory is a publication packaging copy. Original experiment directories were not moved, renamed, deleted, or overwritten.

- The public package name and Python package path use `FederatedBatch` / `federatedbatch`.
- Compact final summaries and plot inputs were copied from the accepted canonical-seven result directories.
- Figure 4 uses only the corrected preprocessing result set; the historical Figure 4 evidence pack and old D panel are excluded from final summaries.
- Figure 3 PCA is described as seven seed-specific evaluations wherever the release documentation reports it.
- Figure outputs were copied into `figures/final/` with stable filenames. Figure 1A and Figure 4A remain author-managed panels.
- Historical scripts and data sources that could not yet be made self-contained remain marked as external or pending in the manifests; their absolute paths are not treated as public runtime dependencies.
- Replaced the public-facing README with a release-candidate guide that separates confirmed redraw commands from unavailable training/evaluation workflows; the redraw commands were not executed.
- Replaced the public `configs/figure2.yaml` entry with a metadata-only record of the final NoQueue/Remote A/B/FederatedBatch protocol. Preserved the old configuration as `configs/figure2_legacy_freshqueue.yaml` and also retained the explicit `configs/figure2_final.yaml` record; no results or training artifacts changed.
- Selected `baselines/fedscgen/upstream/` as the authoritative FedscGen source. Compared it with the pinned upstream Git tree, retained the missing BSD-3-Clause component license, recorded blob IDs and vendored SHA-256 values, and changed the adapter to validate the checksum manifest when nested Git metadata is absent.
- Redirected Figure 3/4 FedscGen compatibility import paths to the authoritative tree. The byte-identical historical copy under `third_party/FedscGen/` remains locally for audit history but is excluded from Git.
- Added an external-dependency inventory covering omitted data/checkpoints, legacy sibling-project references, staged plotting coordinates, and unverified environment requirements.
- Copied the Figure 1 helper inputs from the existing reproducibility project into `external_artifacts/figure1/`; redirected Figure 1 data/config/partition/source references to release-local paths and Figure 5 reconstruction data loading to `data/`.
- Restricted the copied Figure 1 source table to the six deterministic PCA metric rows actually consumed by the canonical-seven driver; retained their values and definitions and made the source-cache reference repository-relative.
- Identified an unused legacy Figure 1 seed-block CSV under `results/plot_inputs/figure1/`. Its removal was blocked, so it is explicitly ignored and excluded from the release artifact manifest.
- Changed the Figure 3 publication renderer to load staged coordinate tables by default and removed its embedding/UMAP fitting path. Figure 3G summary and bubble encoding calculations were left unchanged.
- Retired three obsolete shell wrappers with explicit messages; removed their former worktree and virtual-environment paths. Made the optional FedscGen shared-library path configurable with `FEDSCGEN_LIBRARY_PATH`.
- Disabled direct execution of two Figure 1 helper modules whose legacy main routines use a superseded seed block; their functions remain available to the canonical-seven driver.
- Staged the Figure 3/4 base configuration locally and redirected Figure 3/4 training/evaluation and Figure 4 preprocessing-fix script dependencies from the sibling working project to the release root.
- Added Figure 3 coordinate tables and Figure 1 staged dependencies to the artifact manifest; updated `.gitignore` for environment caches, data/model artifacts, temporary files, curated figure assets, and the required Figure 1 PCA cache.
- Figure 5 variant renderer defaults to the release plot-input and redraw directories; Figure 4 renderer has explicit release-local defaults and `--input`/`--output` options. These path adaptations were not execution-tested.
- Redirected the Figure 2 legacy metric helper loader and the retained Figure 1 R palette imports to release-local files. Bundled the original Figure 1 R palette without changing its definitions.
- Added the obsolete standalone Figure 1 recomputation copies to `.gitignore` and marked them excluded in the artifact manifest; their old sibling-tree data/seed layouts are not current release workflows.
- Rewrote the README to correct encoding damage and clearly distinguish protocol metadata, statically checked renderer arguments, and unverified experiment/render commands.
- Repeated the WSL static audit after path/document updates: 122 Python files parsed by AST, 12 JSON and 12 YAML files parsed, core imports passed, and Figure 2–5 renderer help passed. No rendering or model workflow was run.
- Updated `.gitignore` so curated files under `figures/final/` can be archived while generated figures remain ignored; removed the stale repository URL from `CITATION.cff` because no public repository URL has been verified.
- Python bytecode cache removal was attempted but was blocked by the current execution environment; 8 `__pycache__` directories and 76 `.pyc` files remain in this copy. They are ignored and excluded from the public file manifest. No source, result, or figure files were deleted.
- During initial packaging, no numerical value, model architecture, training hyperparameter, or result table was recomputed. Later bounded integration checks are recorded below; manuscript baselines remain unchanged.

## From-scratch workflow integration (26 October 2026)

- Added guarded stage dispatch, data acquisition/checksum staging, Figure 2
  three-arm training, fresh Figure 3 UMAP generation, final Figure 4 table
  export, and separate Figure 5 comparator/attack/accounting/SPU/assembly paths.
- Corrected native config validation to accept implemented final RoundOnce
  modes. No training loss, privacy mechanism or canonical parameters changed.
- Isolated Figure 3 and Figure 4 execution through an explicit figure selector;
  retired calls from finalization to obsolete plots/reports. Figure 3 keeps
  72 tests; corrected Figure 4 uses nine-batch averaging and 12 tests.
- Made the FedscGen adapter supply its upstream import directory itself. The
  authoritative vendored tree and compatibility patch are unchanged.
- Added synthetic CPU runtime checks: native shared components and FedscGen
  training/checkpoint/direct-inference/correction passed. Smoke-only reduced
  dimensions/rounds exist only in smoke scripts, not canonical YAML files.
- Rechecked saved-table statistics in temporary directories. Added a keyed
  result comparator that writes discrepancies separately and protects bundled
  result/plot/figure directories. No existing manuscript table was overwritten.
- Updated environment switching, commands, manifests and audit limitations.
  Raw data, clean installs, real-data per-figure runs, fresh privacy attacks,
  R and SPU execution remain unverified. No full training, commit or push ran.

## Dataset-source and preparation follow-up (26 October 2026)

- Located all three historical `.h5ad` inputs and verified their size/SHA-256
  against the release manifest and saved Figure 3 run provenance. Files remain
  outside the release; no dataset was downloaded, copied, or rewritten.
- Added exact source evidence, classification, per-figure consumers, missing
  historical version/file-ID metadata, and empty-data preparation instructions.
- Staging now verifies sources and temporary copies before publication and
  protects invalid destinations without an explicit replacement flag.
  Downloading checks provider name/size/MD5 and verifies payload size/SHA-256
  before publication; metadata-only mode transfers no dataset payloads.
- Eight synthetic preparation tests passed; live metadata lookup returned
  HTTP 403 in WSL. The public acquisition route is not marked verified.
- Corrected the dataset loader's preparation-script path in its error message.
  No data processing, model, training parameter, statistical rule, or figure
  changed. No training, inference, evaluation, commit, or push was run.
