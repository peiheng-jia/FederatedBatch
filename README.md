# FederatedBatch

Minimal release repository for the FederatedBatch paper: federated contrastive learning for single-cell RNA-seq batch correction.

This release is intentionally small. It contains the core FederatedBatch implementation, a reproducible main experiment entry point, data download instructions, and manuscript text. Raw datasets, trained checkpoints, generated result folders, and figure outputs are excluded from Git.

## Repository Layout

```text
FederatedBatch-Paper/
├── src/                    # Python package: FederatedBatch, baselines, utilities
├── experiments/            # Minimal reproducible experiment scripts
├── scripts/                # Data and helper scripts
├── manuscript/             # Manuscript markdown files
├── data/                   # Data README only; raw data are not tracked
├── results/                # Result README only; generated outputs are not tracked
├── requirements.txt
├── environment_r_viz.yml
├── setup.py
└── LICENSE
```

## Installation

```bash
git clone https://github.com/REPLACE_WITH_OWNER/FederatedBatch-Paper.git
cd FederatedBatch-Paper

conda create -n federatedbatch python=3.10
conda activate federatedbatch
pip install -r requirements.txt
pip install -e .
```

The optional centralized CONCORD wrapper in `src/baselines/centralized.py` requires the original CONCORD package. Install it separately or place its `src` directory on `PYTHONPATH` if you need that baseline:

```bash
export PYTHONPATH="/path/to/Concord-main/src:$PYTHONPATH"
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="C:\path\to\Concord-main\src;$env:PYTHONPATH"
```

## Data

Raw `.h5ad` datasets are not tracked in Git. Download the pancreas dataset with:

```bash
python scripts/download_data.py --dataset pancreas --output data
```

The expected main file is:

```text
data/pancreas.h5ad
```

The input AnnData object must contain batch and cell type annotations. The scripts standardize common column names such as `tech`, `cell_type`, and `celltype`.

## Main Experiment

Run the minimal FederatedBatch experiment:

```bash
python experiments/run_main.py --data data/pancreas.h5ad --output results/main
```

Outputs:

```text
results/main/
├── metrics.csv
├── config.json
└── adata_with_embeddings.h5ad
```

Use `--quick` for a short smoke-test run:

```bash
python experiments/run_main.py --data data/pancreas.h5ad --output results/quick --quick
```

## Notes For Reviewers

- Raw data, generated figures, checkpoints, and intermediate result folders are intentionally excluded.
- The main release path is `experiments/run_main.py`.
- The default main experiment reports PCA, Harmony when available, and FederatedBatch.
- External CONCORD is required only for the optional centralized CONCORD wrapper; the minimal FederatedBatch implementation lives in `src/federatedbatch`.
- For exact paper numbers, use the same data preprocessing, seeds, and hardware reported in the manuscript.

## Citation

```bibtex
@article{federatedbatch,
  title   = {FederatedBatch: Federated Contrastive Learning for Single-Cell RNA-seq Batch Correction},
  author  = {FederatedBatch authors},
  year    = {2026}
}
```
