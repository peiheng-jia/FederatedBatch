# Data

Raw datasets are not tracked in this repository.

## Benchmark datasets

The paper uses the Pancreas, Human Immune, and Human Lung atlas integration datasets from the scIB dataset collection:

```text
https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets/12420968
```

Please cite:

```text
Luecken MD, Buttner M, Chaichoompu K, Danese A, Interlandi M, Mueller MF, et al.
Benchmarking atlas-level data integration in single-cell genomics.
Nature Methods 19, 41-50 (2022).
https://doi.org/10.1038/s41592-021-01336-8
```

Place downloaded `.h5ad` files in this directory and pass the selected file to:

```bash
python experiments/run_main.py --data data/YOUR_DATASET.h5ad --output results/main
```

## Smoke-test dataset

For a lightweight installation check, this repository can download the public Scanpy pancreas dataset:

```text
data/pancreas.h5ad
```

```bash
python scripts/download_data.py --dataset pancreas --output data
```

The AnnData object should contain:

- `obs["batch"]` or `obs["tech"]`
- `obs["celltype"]` or `obs["cell_type"]`

Large files such as `.h5ad`, `.h5`, `.npy`, and `.npz` are ignored by `.gitignore`.
