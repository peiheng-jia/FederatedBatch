# External dependencies and reproduction limits

This inventory separates bundled redraw inputs from resources required to rerun experiments. Static checks do not establish end-to-end reproducibility.

## External data and generated artifacts

| Resource | Used for | Status |
|---|---|---|
| Pancreas, Immune, and Lung `.h5ad` files | Fresh preprocessing, training, and evaluation | Not bundled. `data/datasets.json` records filenames, dimensions, annotation keys, sizes, hashes, and the registered Figshare DOI. |
| Figure 3/4 model checkpoints and complete embeddings | Fresh training and Figure 4 checkpoint-only evaluation | Not bundled; no stable public artifact URL is recorded. |
| Figure 5 attack and secure-computation intermediate artifacts | Re-running attacks or the SPU PoC | Not all bundled. Plot summaries are bundled under `results/plot_inputs/figure5/`. |
| Figure 1A and Figure 4A panels | Full assembled manuscript figures | Author-managed and absent from this code tree. |

Figure 3's three representative-seed UMAP coordinate tables are bundled under `results/plot_inputs/figure3/`. The release renderer reads those saved coordinates; it does not recompute UMAP. Figure 4's preprocessing-fixed summary and per-batch results are bundled under `results/final_summaries/figure4/`; the automated output is limited to panels B/C.

## Runtime and source dependencies

- The native training environment has not been recreated from a clean checkout. The available WSL Ubuntu 22.04 environment has Git 2.34.1 and Python 3.10.12; Conda is unavailable. GPU/CUDA equivalence is unverified.
- FedscGen uses a separate legacy stack. `baselines/fedscgen/upstream/` is the authoritative checksummed snapshot. Its unmodified files match the pinned upstream Git blobs and its two modified Python files exactly match the recorded compatibility patch. The adapter validates the vendored SHA-256 manifest when nested Git metadata is absent. The redundant `third_party/FedscGen/` audit copy is ignored and is not part of the public inventory.
- R packages/runtime for retained historical Figure 1 scripts are not pinned by a complete lockfile. The local color configuration is bundled at `external_artifacts/figure1/figures/figure1/r/color_config.R`; historical scripts still need their stated result inputs, and are not claimed as verified redraw commands.
- SecretFlow-SPU/JAX/XLA and compatible hardware are required to rerun the local 3-party ABY3 proof of concept. The recorded local runtime is not a WAN deployment benchmark.
- A small number of historical Figure 1 metric helper copies retain source-location metadata for the original evaluator. The final release entry points import the release-local helper and do not load another project tree. Standalone legacy recomputation scripts refer to old result layouts and are not current reproduction commands.

## Repository and archive state

- This release candidate has no `.git` directory. Git availability was verified in WSL, but tracked/untracked status, a release commit, and a reproducible source revision therefore cannot be reported.
- Datasets, checkpoints, and full embeddings do not have verified public download URLs in this tree. This blocks clean-checkout reproduction of fresh experiments.
- Interpreter caches remain physically present because cleanup was blocked by the Windows execution environment: 8 `__pycache__` directories and 76 `.pyc` files at the latest inventory. `.gitignore` and `artifact_manifest.tsv` exclude them. No virtual environment, temporary log, pilot/debug file, or file larger than 100 MB was found.
- No GitHub repository/tag or Zenodo DOI has been created.

## Before claiming reproducibility

Publish stable, policy-appropriate locations for required data/checkpoints; pin supported native, FedscGen, R, and SPU environments; add the author-managed panels to the manuscript asset workflow; and run the documented redraw/smoke checks from a clean checkout. Do not describe the present static audit as experiment reproduction.
