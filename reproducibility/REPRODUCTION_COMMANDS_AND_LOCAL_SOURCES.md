# Reproduction commands and local sources

This release records commands that are present in the source tree. They are
classified as **redraw-only**, **checkpoint/data dependent**, or **not verified**;
none was executed during the packaging audit.

| Figure | Training/evaluation entry point | Summary/statistics | Redraw |
|---|---|---|---|
| 1 | `experiments/run_figure1_canonical7.py` (Pancreas data and compatible environment required; fixed helpers are bundled) | `experiments/finalize_figure1_statistics_umap.py` is imported as a helper; saved tables under `results/final_summaries/figure1/` | No verified release redraw; panel A is author-managed |
| 2 | Saved final provenance in `results/final_summaries/figure2/provenance.json`; original training inputs are external | `experiments/finalize_figure2_reviewer.py` (not clean-checkout verified) | `python scripts/render/render_figure2_result_only.py --input results/final_summaries/figure2 --output figures/redraw/figure2` |
| 3 | `experiments/run_final_figure3_4_canonical7.py` (external data/checkpoints) | bundled `results/final_summaries/figure3/` | `python scripts/render/render_figure3_publication.py --input results/final_summaries/figure3 --output figures/redraw/figure3`; renderer reads bundled coordinates from `results/plot_inputs/figure3/` and does not recompute UMAP |
| 4 | `scripts/fix_figure4_fedscgen_preprocessing.py` (external data/checkpoints; no retraining in repair) | bundled corrected summaries under `results/final_summaries/figure4/` | `python scripts/render/render_figure4_preprocessing_fixed_redesign.py --input results/final_summaries/figure4 --output figures/redraw/figure4` |
| 5 | `experiments/final_private_ab_attack_validation.py`, `experiments/final_expression_reconstruction_attack.py`, and `experiments/secure_AB_spu_poc.py` (external artifacts/environments) | bundled Figure 5 plot inputs and evidence pack | `python scripts/render/make_figure5_variants.py --input results/plot_inputs/figure5 --output figures/redraw/figure5` |

## Local source records

- `configs/figure1.yaml`, `configs/figure2.yaml`, `configs/figure2_final.yaml`,
  `configs/figure3_4_native.yaml`, `configs/figure3_4_fedscgen.yaml`, and
  `configs/figure5.yaml` record the final protocol metadata.
- `configs/figure2_legacy_freshqueue.yaml` is retained for historical provenance
  only and is not the final Figure 2 method.
- `results/final_summaries/` contains compact copied result tables; it does not
  imply that the omitted checkpoints or raw data are recoverable from this tree.
- `reproducibility/SOURCE_FILE_ORIGINS.tsv` and
  `reproducibility/EXTERNAL_DEPENDENCIES.md` identify copy/adaptation status and
  unresolved external inputs.
