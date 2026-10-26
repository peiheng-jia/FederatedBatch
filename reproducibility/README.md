# Reproducibility guide

This guide distinguishes three levels of use:

1. **Redraw from saved results.** Compact tables in `results/final_summaries/` and `results/plot_inputs/` provide inputs for the documented Figure 2–5 renderers. Figure 3 representative-seed UMAP coordinates are bundled.
2. **Historical checkpoint repair.** The retained Figure 4 repair script documents how the accepted preprocessing correction reused old FedscGen checkpoints; this is provenance, not the full reproduction route.
3. **Fresh training.** Figures 1–5 have registered source-level stages that generate checkpoints and embeddings from the staged datasets. This candidate has not been runtime-validated from an empty checkout.

The file inventory is in `experiment_manifest.tsv`, `artifact_manifest.tsv`, and `SOURCE_FILE_ORIGINS.tsv`. `RELEASE_COPY_MODIFICATIONS.md` records adaptations made inside this release copy only.

## External resources

The data manifest points to Figshare DOI `10.6084/m9.figshare.12420968`. The downloader resolves exact provider filenames and verifies byte size and SHA-256; a manual verified staging fallback is documented for WAF-blocked environments. Checkpoints and complete embeddings are generated outputs and are intentionally excluded from Git.

The authoritative FedscGen snapshot is included under `baselines/fedscgen/upstream/` with the recorded local compatibility patch. Its unmodified files were matched against the pinned upstream Git blobs, and all included source files are covered by `baselines/fedscgen/upstream-source-manifest.json`.

## Figure-level status

| Figure | Redraw input | Fresh computation status |
|---|---|---|
| 1 | `results/final_summaries/figure1/` and `results/plot_inputs/figure1/` | Native shared components smoke-tested; real-data driver and R renderer untested |
| 2 | `results/final_summaries/figure2/` | Three-arm shared components and saved-input rendering tested; real-data driver untested |
| 3 | `results/final_summaries/figure3/` and `results/plot_inputs/figure3/` | Native/FedscGen synthetic components passed; 72-test summary and figure isolation verified; real-data workflow untested |
| 4 | `results/final_summaries/figure4/` | Synthetic inference/correction passed; preprocessing-fixed 12-test summary verified; real-data LOBO untested |
| 5 | `results/plot_inputs/figure5/` | Staged source chain connected; native attacks untested and SPU environment not locked |

Synthetic and static checks do not imply end-to-end numerical reproduction.
See `REPRODUCIBILITY_CHECK_REPORT.md` for exact tested scope and
`REPRODUCTION_COMMANDS_AND_LOCAL_SOURCES.md` for environment-specific commands.
