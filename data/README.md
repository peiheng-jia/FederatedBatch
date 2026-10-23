# Data

Raw `.h5ad` files are intentionally excluded from Git. The manuscript uses the
three processed integration-task datasets listed in `datasets.json`.

The downloaded files contain two expression representations. `X` is already
transformed; `layers["counts"]` is the provider's count-like input used by this
repository. Experiment loaders restore that layer before selecting HVGs and
performing the configured normalization and `log1p`. Pancreas contains some
non-integer values in this provider layer, so it should be described as
count-like rather than guaranteed integer raw counts.

1. Download the datasets from the scIB Figshare collection linked in the
   manifest.
2. Keep the filenames exactly as listed in the manifest.
3. Stage and verify them with:

```bash
python scripts/prepare/prepare_data.py --source-dir /path/to/downloads --dataset all
```

To verify files already in this directory:

```bash
python scripts/prepare/prepare_data.py --verify-only --dataset all
```

Do not substitute `scanpy.datasets.pancreas()`: it has a different number of
cells and batches from the Pancreas dataset used by the manuscript.
