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
- Scanned filenames and documentation for AI conversation/prompt artifacts and generated-content markers; none were found. Third-party and author attributions were retained.
- Updated `.gitignore` so curated files under `figures/final/` can be archived while generated figures remain ignored; removed the stale repository URL from `CITATION.cff` because no public repository URL has been verified.
- Python bytecode cache removal was attempted but was blocked by the current execution environment; 8 `__pycache__` directories and 76 `.pyc` files remain in this copy. They are ignored and excluded from the public file manifest. No source, result, or figure files were deleted.
- No numerical value, model architecture, training hyperparameter, or result table was recomputed during packaging.
