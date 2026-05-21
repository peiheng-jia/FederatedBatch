# Results

Generated experiment outputs are written here and are not tracked in Git.

Run:

```bash
python experiments/run_main.py --data data/pancreas.h5ad --output results/main
```

Expected outputs:

- `metrics.csv`
- `config.json`
- `adata_with_embeddings.h5ad`

Do not commit generated result directories, checkpoints, or large AnnData files.
