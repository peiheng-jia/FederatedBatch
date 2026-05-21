# Data

Raw datasets are not tracked in this repository.

Expected file for the main experiment:

```text
data/pancreas.h5ad
```

Download:

```bash
python scripts/download_data.py --dataset pancreas --output data
```

The AnnData object should contain:

- `obs["batch"]` or `obs["tech"]`
- `obs["celltype"]` or `obs["cell_type"]`

Large files such as `.h5ad`, `.h5`, `.npy`, and `.npz` are ignored by `.gitignore`.
