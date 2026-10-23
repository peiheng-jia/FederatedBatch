# Reproducibility guide

This guide distinguishes three levels of use:

1. **Redraw from saved results.** Compact tables in `results/final_summaries/` and `results/plot_inputs/` provide inputs for the documented Figure 2–5 renderers. Figure 3 representative-seed UMAP coordinates are bundled.
2. **Re-evaluation from saved checkpoints.** The Figure 4 preprocessing repair requires original FedscGen checkpoints and the external datasets; these artifacts are not bundled.
3. **Fresh training.** Figures 1–5 require datasets, fixed partitions, compatible runtime stacks, and the exact source/configuration snapshot. This candidate has not been validated from an empty checkout.

The file inventory is in `experiment_manifest.tsv`, `artifact_manifest.tsv`, and `SOURCE_FILE_ORIGINS.tsv`. `RELEASE_COPY_MODIFICATIONS.md` records adaptations made inside this release copy only.

## External resources

The data manifest points to Figshare DOI `10.6084/m9.figshare.12420968`. Full `.h5ad` files, checkpoints, and complete embeddings must be distributed through an archival or controlled artifact service. Required names, hashes, and roles are recorded in the manifests; a public URL for the final checkpoint bundle is not recorded.

The authoritative FedscGen snapshot is included under `baselines/fedscgen/upstream/` with the recorded local compatibility patch. Its unmodified files were matched against the pinned upstream Git blobs, and all included source files are covered by `baselines/fedscgen/upstream-source-manifest.json`.

## Figure-level status

| Figure | Redraw input | Fresh computation status |
|---|---|---|
| 1 | `results/final_summaries/figure1/` and `results/plot_inputs/figure1/` | Requires external Pancreas data and a compatible environment; no release redraw command is confirmed |
| 2 | `results/final_summaries/figure2/` | Final tables bundled; training/evaluation requires omitted inputs and checkpoints |
| 3 | `results/final_summaries/figure3/` and `results/plot_inputs/figure3/` | Metrics and representative coordinates bundled; fresh embeddings require datasets and checkpoints |
| 4 | `results/final_summaries/figure4/` | Preprocessing-fixed B/C summaries bundled; checkpoint-only repair requires external FedscGen checkpoints |
| 5 | `results/plot_inputs/figure5/` | Redraw inputs bundled; attack/MPC reruns require their recorded environments and intermediate artifacts |

Static checks do not imply end-to-end numerical reproduction or a successful figure render.
