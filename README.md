# FederatedBatch

FederatedBatch learns a shared single-cell representation using local
CONCORD training, FedAvg, and cross-client contrastive interaction. This
repository contains the implementation, baseline adapters, experiment scripts,
and saved tables/plotting inputs for the FederatedBatch manuscript.
The output is a latent embedding, not a corrected gene-expression matrix.

## Installation

Run commands from the repository root. The native environment specifies
Python 3.10.13, PyTorch 2.1.2 and CUDA 11.8; Python dependencies are listed in
`pyproject.toml`. The small example below runs on CPU.

```bash
conda env create -f environment/federatedbatch.yml
conda activate federatedbatch
python -m pip install -e .
```

FedscGen has a separate environment (`environment/fedscgen.yml`). Figure 1
rendering uses R with `ggplot2`, `dplyr` and `cowplot`; Figure 5 secure-computation
experiments require a separate SecretFlow-SPU/JAX environment. See the
[environment notes](docs/reproduction.md#environments-and-inputs).

The paper commands for Figures 3-5 are written for Bash (including WSL on
Windows), not PowerShell. The minimal CPU example is independent of those shell
snippets and does not require Git metadata or a GPU.

## Data download

Download the following processed scIB datasets from the
[official Figshare collection](https://doi.org/10.6084/m9.figshare.12420968).
Large datasets and checkpoints are not included in Git.

| Dataset | Required filename in `data/` | Cells / batches / cell types |
|---|---|---|
| Pancreas | `pancreas.h5ad` | 16382 / 9 / 14 |
| Immune | `Immune_ALL_human.h5ad` | 33506 / 10 / 16 |
| Lung | `Lung_atlas_public.h5ad` | 32472 / 16 / 17 |

For browser-downloaded files:

```bash
python scripts/prepare/prepare_data.py --source-dir /path/to/downloads --dataset all
python scripts/prepare/prepare_data.py --verify-only --dataset all
```

Alternatively, the datasets can be downloaded using:

```bash
python scripts/prepare/download_data.py --dataset all
```

If automatic downloading is unavailable, download the datasets manually from
the official scIB Figshare collection. Required filenames, byte sizes and
SHA-256 checksums are provided in `data/datasets.json`. Detailed instructions
are available in [Data Preparation](data/README.md).

## Preprocessing

The paper loaders restore `layers['counts']`, harmonize batch/cell-type labels,
and use 2,000 HVGs with library-size normalization (`target_sum=10000`) and
`log1p`. Normalization is applied across the full gene set before subsetting
model features. Figure 1 uses its saved feature list and fixed partition;
Figure 4 determines HVGs from training batches and applies the corrected
full-gene normalization to held-out data before selecting those HVGs.
The scripts perform these steps; do not normalize the downloaded files again
before passing them to a paper experiment.

## Quick start

```bash
python examples/minimal.py
```

This example creates a small count-like AnnData input (64 cells, 24 genes,
two batches), runs two CPU communication rounds through the existing
`train_federated` function with private RoundOnce A/B interaction, saves the
embedding/checkpoint, and checks inference after loading the checkpoint.
No dataset download or cell-type labels are required. Outputs are under
`results/runs/minimal/`: `embedding.h5ad` contains
`obsm['X_federatedbatch']`, `model.pt` holds the model, and `config.yaml`
records the example settings. Choose a new `--output` directory to run again.

To try a small custom `.h5ad`:

```bash
python examples/minimal.py --input /path/to/small_counts.h5ad --output results/runs/custom
```

Use unnormalized count-like `X` or `layers['counts']`, unique cell/gene names,
and `obs['batch']` with at least two batches and eight cells per batch. The
example uses all supplied genes and reduced model/training settings. It is a
usage demonstration, not a paper benchmark or a validation of the paper's
privacy budget. It simulates clients in one process.


## Paper results

The direct experiment, summary and plotting commands for Figures 1-5 are in
[paper reproduction commands](docs/reproduction.md).
The [experiment entry-point guide](experiments/README.md) lists the final
commands; shared helpers live in `src/federatedbatch/experiment_support/`
and plotting scripts in `scripts/render/`. No workflow dispatcher is required. Full paper runs use seeds `218, 394, 542, 639, 788, 814, 880` and
can be expensive. New outputs go under `results/reproduction/`.

For example, redraw Figure 3 from the saved metrics and UMAP coordinates:

```bash
python scripts/render/render_figure3_publication.py --input results/final_summaries/figure3 --output figures/redraw/figure3
```

Saved-result redraws do not rerun experiments. Figure 3 PCA statistics retain
seven seed-specific evaluations. Figure 4 uses the preprocessing-fixed results
and has A/B/C only; Figure 1A and Figure 4A are author-managed schematic panels.

## Reproducibility Notes

The repository provides the FederatedBatch implementation, a tested usage
example, and the configurations and scripts associated with the reported
experiments. Full benchmark reproduction requires the corresponding datasets
and computational environments. Detailed instructions are provided in the
[Reproducibility Guide](docs/reproduction.md).

Differential privacy is applied to the cross-client A/B statistics release,
while secure computation is evaluated using a local three-party ABY3 proof
of concept.

## Citation

Use [CITATION.cff](CITATION.cff) for the software authors and manuscript title.
Please also cite CONCORD and, when using its baseline, FedscGen; source links
and the FedscGen publication DOI are in [third-party notices](THIRD_PARTY_LICENSES.md).
The archived [v0.3.0 release](https://github.com/peiheng-jia/FederatedBatch/releases/tag/v0.3.0)
is available on Zenodo: [10.5281/zenodo.23694614](https://doi.org/10.5281/zenodo.23694614)
(published 27 October 2026).

## License

FederatedBatch is distributed under the [MIT License](LICENSE). CONCORD's MIT
notice and FedscGen's Apache-2.0/BSD-3-Clause notices, attribution and local
patch records are retained; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
