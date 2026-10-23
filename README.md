# FederatedBatch

Code and compact results for **FederatedBatch: Federated Contrastive Learning for Privacy-Preserving Single-Cell Batch Correction**. This directory is a publication release candidate. It has not been verified by a clean-checkout end-to-end reproduction. Raw datasets, model checkpoints, and full embeddings are not included.

## Contents

- `src/`: FederatedBatch and vendored CONCORD implementation.
- `baselines/`: FedscGen adapter and the authoritative checksummed source snapshot; see `THIRD_PARTY_LICENSES.md`.
- `configs/`: protocol records. Figure 2 YAML files are metadata, not runnable training configurations.
- `data/`: dataset manifest and staging instructions.
- `experiments/`: experiment, analysis, and legacy helper scripts.
- `results/final_summaries/`: compact final tables and provenance.
- `results/plot_inputs/`: saved plotting inputs, including Figure 3 UMAP coordinates.
- `figures/final/`: curated final figure exports. Figure 1A and Figure 4A are author-managed and are not included.
- `reproducibility/`: artifact manifests, dependency inventory, and release audits.

## Environment

Environment specifications are provided in `environment/`, `environment.yml`, and `environment-fedscgen.yml`. They have not been validated from a clean public checkout. Native training requires a compatible Python/PyTorch stack and CUDA for the recorded environment. FedscGen, R rendering, and SecretFlow/SPU use separate dependencies; see `reproducibility/EXTERNAL_DEPENDENCIES.md`.

```bash
conda env create -f environment/federatedbatch.yml
conda activate federatedbatch
pip install -e .
```

## Data

The Pancreas, Immune, and Lung processed datasets are registered in `data/datasets.json` with the listed Figshare DOI, filenames, dimensions, annotation keys, sizes, and hashes. Download and stage them according to `data/README.md`. The `.h5ad` files are not included.

## Final protocol

The canonical training seeds are `218, 394, 542, 639, 788, 814, 880`. FederatedBatch uses 20 communication rounds, five local epochs, batch size 128, learning rate 0.001, RoundOnce remote A/B interaction in rounds 4–20, 102 unique remote cells per source and active round, one-time participation, Joint-C50 clipping (`C=1206.3391650512762`), `sigma=2`, `tau_q=0.20`, and `alpha=0.10`. The formal cell-level DP guarantee applies only to the cross-client A/B release channel, not to end-to-end federated training.

Figure 2 compares NoQueue, Remote A/B (`sigma=0`), and FederatedBatch. Figure 3 PCA statistics summarize seven seed-specific evaluations as mean ± SD. Figure 4 uses the preprocessing-fixed result set and contains automated panels B/C; panel A is author-managed. The old Figure 4D is not part of the final figure.

## Result redraws

The commands below match the renderer CLI options. WSL static checks confirmed each renderer's `--help`; no redraw command was run, so successful rendering is not claimed.

Figure 2:

```bash
python scripts/render/render_figure2_result_only.py --input results/final_summaries/figure2 --output figures/redraw/figure2
```

Figure 3 loads the bundled representative-seed coordinate tables from `results/plot_inputs/figure3/` by default and does not recompute UMAP:

```bash
python scripts/render/render_figure3_publication.py --input results/final_summaries/figure3 --output figures/redraw/figure3
```

Figure 4 panels B/C:

```bash
python scripts/render/render_figure4_preprocessing_fixed_redesign.py --input results/final_summaries/figure4 --output figures/redraw/figure4
```

Figure 5:

```bash
python scripts/render/make_figure5_variants.py --input results/plot_inputs/figure5 --output figures/redraw/figure5
```

There is no verified release redraw command for Figure 1. Figure 1 training/evaluation and Figures 2–5 fresh experiment workflows are not confirmed as self-contained in this release copy. Figure 4 checkpoint-only re-evaluation requires external data and checkpoints; it is not required for the bundled corrected summary redraw.

## Limitations

- Clean-checkout end-to-end reproduction is **NOT VERIFIED**. Static parsing, package imports, path/CLI checks, and renderer help do not execute model workflows.
- Datasets, checkpoints, full embeddings, and some attack/MPC intermediate artifacts are external. Public stable download locations for required checkpoints are not established.
- Figure 1A and Figure 4A are author-provided schematic panels and are not included because no final file path or editable source is registered in the experiment project. Figure 1 training needs the staged Pancreas dataset and a compatible runtime.
- Interpreter caches may remain in this working copy because deletion was blocked by the environment. `.gitignore` and the release manifest exclude caches, virtual environments, temporary logs, and non-final artifacts.
- A GitHub repository, release commit/tag, and Zenodo DOI have not been created.

## Citation and licenses

Use `CITATION.cff` for the manuscript citation. CONCORD and FedscGen attribution and license notices are retained. Review `THIRD_PARTY_LICENSES.md` and `reproducibility/GITHUB_ZENODO_RELEASE_CHECKLIST.md` before publication.
