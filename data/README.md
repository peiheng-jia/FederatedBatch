# Data preparation

The paper uses processed scIB integration-task AnnData bundles from
[Figshare, DOI 10.6084/m9.figshare.12420968](https://doi.org/10.6084/m9.figshare.12420968)
(recorded license: CC-BY-4.0). Download the files yourself; dataset payloads are
excluded from Git. These are upstream analysis inputs, not sequencing reads.

| Dataset | Required path | Cells / batches / cell types | Used by |
|---|---|---|---|
| Pancreas | `data/pancreas.h5ad` | 16382 / 9 / 14 | Figures 1-5 |
| Immune | `data/Immune_ALL_human.h5ad` | 33506 / 10 / 16 | Figure 3 |
| Lung | `data/Lung_atlas_public.h5ad` | 32472 / 16 / 17 | Figure 3 |

## Download and verify

From an otherwise empty data directory, retain this README and
[`datasets.json`](datasets.json). Open the official collection above, download
the exact named files to a directory of your choice, then run from the repository
root:

```bash
python scripts/prepare/prepare_data.py --source-dir /path/to/downloads --dataset all
python scripts/prepare/prepare_data.py --verify-only --dataset all
```

The script verifies byte sizes and SHA-256 hashes before staging files under
`data/`. It retains existing valid files and rejects mismatches. All three files
total approximately 3.40 GB; allow additional space for staging. Use
`--dataset pancreas` if only Pancreas workflows are needed.

An existing downloader can resolve filenames against actual Figshare metadata:

```bash
python scripts/prepare/download_data.py --dataset all --resolve-only
# Downloads the large files only when this command is explicitly run:
python scripts/prepare/download_data.py --dataset all
```

If automatic downloading is unavailable, download the exact files from the
official collection above and stage them with `prepare_data.py`. Expected
filenames and SHA-256 checksums are listed in `data/datasets.json`. Use files
that match the listed hashes; if matching files are unavailable, contact the
authors rather than substituting another atlas version.

The manuscript file hashes are:

```text
97e6dfd65553e4d10aa3ef5d904362970a75c677c31d70fabc9234191a09db8c  pancreas.h5ad
96709b5fa677d5eafc734793a7873a2c763453ba8abddad24c0fba39e0d7f2c8  Immune_ALL_human.h5ad
623f84435efdce9af31fc2be393ae7d2dfcf427414567b2fd994ea7422f5feeb  Lung_atlas_public.h5ad
```

For an independent check, run `sha256sum data/*.h5ad` on Linux, or
`Get-FileHash -Algorithm SHA256 data/*.h5ad` in PowerShell.

## Preprocessing performed by the paper scripts

The downloaded files contain transformed `X` and a count-like `layers['counts']`.
Paper loaders restore the counts layer; the Pancreas layer can contain
non-integer values. Do not normalize these files separately before running the
experiments.

| Dataset | Batch annotation | Cell-type annotation |
|---|---|---|
| Pancreas | `batch` | `celltype` |
| Immune | `batch` | `final_annotation` |
| Lung | `batch` | `cell_type` |

Loaders harmonize these annotations to the internal names. They select 2,000
HVGs from count-like input and apply full-gene `normalize_total(target_sum=10000)`
followed by `log1p`, before subsetting the model features.

- Figure 1 loads its bundled feature list and fixed six-client partition.
- Figure 2 generates the configured five-client seed-specific partitions and
  one-time remote-cell schedules; Figure 3 uses three-client partitions.
- Figure 4 uses training-only HVGs and nine LOBO splits. Held-out data undergo
  full-gene normalization and `log1p` before selecting training HVGs, including
  the corrected FedscGen inference path.
- Figure 5 uses Pancreas and generated Figure 2 outputs for privacy evaluation.

Partitions and feature lists are generated or loaded by the existing experiment
scripts; see the [paper commands](../docs/reproduction.md).
The bundled summaries and UMAP coordinates support quick redraws only. Another
Pancreas helper dataset, a renamed file, or a rewritten AnnData object is not a
verified replacement for the manuscript inputs.
