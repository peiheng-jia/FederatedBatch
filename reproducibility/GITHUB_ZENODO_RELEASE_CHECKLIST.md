# GitHub/Zenodo release checklist

## Required before publication

- Create a clean Git repository from this release tree and commit the exact source/configuration snapshot.
- Exclude obsolete duplicate snapshots from the archive and ensure every public command resolves paths relative to the repository root. The duplicate `third_party/FedscGen/` tree is currently ignored and documented; the authoritative source is `baselines/fedscgen/upstream/`.
- Remove the remaining generated `__pycache__` directories and `.pyc` files before creating the archival commit, or verify that ignore rules exclude them (the current environment prevented physical deletion).
- Add a public archival location for the large dataset/checkpoint bundle, or document a controlled-access workflow.
- Record the resolved native, FedscGen, R, and SecretFlow/SPU environments, including CUDA and hardware.
- Run the redraw-only commands in a clean environment and inspect every output.
- Run at least one smoke test for data staging, native imports, the FedscGen adapter, and each renderer.
- Verify the staged Figure 3 coordinate-table hashes and metadata before release; the renderer uses saved coordinates and does not recompute UMAP.
- Confirm dependency-license reporting for Scanpy and SecretFlow/SPU. CONCORD and the vendored FedscGen source already retain their upstream licenses and attribution.
- Tag the release and mint a Zenodo DOI only after the clean-checkout checks pass.

## Current status

The present directory is a release candidate. No commit, tag, push, Zenodo record, training, inference, evaluation, or plotting run was performed during packaging. A final archival DOI is therefore **MISSING**.
