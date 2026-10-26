# Paper reproduction commands

Run from the repository root after [installation](../README.md#installation)
and [data preparation](../data/README.md). Commands below call the existing
scripts directly; no workflow dispatcher is required. Full runs use seeds
218, 394, 542, 639, 788, 814 and 880. They are expensive and have not been
executed from a clean installation. The lightweight CPU example was tested;
that does not validate the canonical paper runs.

Use the `federatedbatch` environment unless noted otherwise. Training and
evaluation write to `results/reproduction/`; renderers write to
`figures/reproduction/`. Bundled manuscript results remain separate.
Commands for Figures 3-5 use Bash syntax (`export`, shell variables and `$...`)
and should be run in Bash or WSL, not copied directly into PowerShell.

## Figure 1: Pancreas main experiment

Configuration: `configs/figure1.yaml`; bundled fixed six-client partition and
feature list. The driver trains/evaluates methods and summarizes results.
Rendering requires R packages `ggplot2`, `dplyr` and `cowplot` (not installed in
the checked WSL environment). Panel A is an author-provided schematic.

```bash
python experiments/run_figure1_canonical7.py --preflight --output results/reproduction/figure1
python experiments/run_figure1_canonical7.py --resume --output results/reproduction/figure1
Rscript experiments/finalize_figure1_canonical7_bcd.R --input results/reproduction/figure1 --output figures/reproduction/figure1
```

## Figure 2: remote-interaction ablation

Configuration: `configs/figure2_training.yaml`; five-client partitions.
The three arms are NoQueue, Remote A/B (sigma=0), and FederatedBatch with
RoundOnce / Joint-C50 / sigma=2. No historical FIFO experiment is required.

```bash
python experiments/run_figure2_from_scratch.py --config configs/figure2_training.yaml --stage preflight
python experiments/run_figure2_from_scratch.py --config configs/figure2_training.yaml --stage train
python experiments/run_figure2_from_scratch.py --config configs/figure2_training.yaml --stage summarize
python scripts/render/render_figure2_result_only.py --input results/reproduction/figure2 --output figures/reproduction/figure2
```

## Figure 3: Pancreas, Immune and Lung

The driver loads `configs/figure3_4_base.yaml` and
`configs/figure3_4_fedscgen.yaml`, using three-client partitions and applying
the canonical seeds/private settings in code. `configs/figure3_4_native.yaml`
is a protocol record, not a directly loaded training configuration. Run the
FedscGen stage in its separate environment, created with
`conda env create -f environment/fedscgen.yml`. The locked configuration requests
`cuda:0`, so the canonical FedscGen stages require a CUDA-capable GPU and working
CUDA driver. On the checked WSL installation it also needs
`export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`
after activation to select the compatible Conda C++ runtime.

```bash
python experiments/run_final_figure3_4_canonical7.py --figure 3 --stage preflight
python experiments/run_final_figure3_4_canonical7.py --figure 3 --stage native
conda activate fedscgen
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python experiments/run_final_figure3_4_canonical7.py --figure 3 --stage fedscgen
conda activate federatedbatch
python experiments/run_final_figure3_4_canonical7.py --figure 3 --stage finalize
python scripts/summarize/generate_figure3_umap_coordinates.py --input results/reproduction/figure3 --output results/reproduction/figure3/coordinates
python scripts/render/render_figure3_publication.py --input results/reproduction/figure3 --coordinates results/reproduction/figure3/coordinates --output figures/reproduction/figure3
```

The coordinate generator reads newly generated embeddings. The renderer can
also use bundled coordinates for quick redraws. Figure 3G retains seven
seed-specific PCA evaluations, sample SD (`ddof=1`), and the existing bubble
size/color rules. Finalization uses the paper's 72-test Holm family.

## Figure 4: corrected LOBO analysis

The shared Figure 3/4 configurations are used, but `--figure 4` selects only
LOBO. The FedscGen inference path normalizes the full gene matrix before
subsetting training HVGs. The fresh workflow trains models; it does not require
unpublished checkpoints. Each seed averages nine held-out batches before the
seven-seed paired tests (12-test Holm family).
The FedscGen stage uses the same Bash/WSL, separate Conda environment and
CUDA-capable GPU requirements described for Figure 3.

```bash
python experiments/run_final_figure3_4_canonical7.py --figure 4 --stage preflight
python experiments/run_final_figure3_4_canonical7.py --figure 4 --stage native
conda activate fedscgen
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python experiments/run_final_figure3_4_canonical7.py --figure 4 --stage fedscgen
conda activate federatedbatch
python experiments/run_final_figure3_4_canonical7.py --figure 4 --stage finalize
python scripts/summarize/export_figure4_final_tables.py --input results/reproduction/figure4/per_seed_per_heldout_metrics.csv --output results/reproduction/figure4/final_tables
python scripts/render/render_figure4_preprocessing_fixed_redesign.py --input results/reproduction/figure4/final_tables --output figures/reproduction/figure4
```

Figure 4 has A/B/C only. The renderer produces B/C; A is author-provided.
The bundled reference tables use the final preprocessing-fixed analysis.

## Figure 5: privacy attacks, accounting and secure computation

Run Figure 2 first. Configuration: `configs/figure5_privacy.yaml`.
`configs/figure2_legacy_freshqueue.yaml` is used **only** to reproduce the
explicit-embedding attack comparator, not the final FederatedBatch method.
The following are implemented CLI entry points, but the fresh attack chain
has not been runtime-validated. The SPU environment is not yet pinned/tested;
it must be prepared separately before the secure-computation command. Therefore
the saved-result redraw below is currently supported, while a complete fresh
Figure 5 reproduction is **not yet verified**.

The privacy preflight verifies source identity. A normal Git clone must retain
the matching commit and unchanged core source. A formal source archive without
`.git` is accepted only when its bundled `RELEASE_PROVENANCE.json` version and
core source hashes verify; an arbitrary copied directory is rejected.

```bash
# Native environment: explicit comparator preparation/training and attacks.
f2=results/reproduction/figure2
f5=results/reproduction/figure5
python scripts/prepare/prepare_figure5_explicit_baseline.py --config configs/figure2_legacy_freshqueue.yaml --output results/reproduction/figure2_explicit_baseline
python experiments/figure5_privacy.py --config configs/figure5_privacy.yaml --preflight-only
python experiments/figure5_privacy.py --config configs/figure5_privacy.yaml --resume
python experiments/run_figure5_explicit_full_data.py --config configs/figure2_legacy_freshqueue.yaml --output "$f5/explicit_full_data" --stage preflight
python experiments/run_figure5_explicit_full_data.py --config configs/figure2_legacy_freshqueue.yaml --output "$f5/explicit_full_data" --stage train
python experiments/final_private_ab_attack_validation.py --final-root "$f2" --partition-root "$f2" --explicit-privacy-root "$f5/explicit_baseline_privacy" --output "$f5/release_attacks"
python experiments/final_expression_reconstruction_attack.py --final-root "$f2" --partition-root "$f2" --explicit-privacy-root "$f5/explicit_baseline_privacy" --explicit-full-root "$f5/explicit_full_data" --previous-reconstruction "$f5/release_attacks/reconstruction_attack.csv" --output "$f5/expression_reconstruction"
python scripts/summarize/generate_figure5_privacy_accounting.py --config configs/figure5_privacy.yaml --figure2-provenance "$f2/provenance.json" --output "$f5/privacy_accounting.csv"

# In a separately prepared SecretFlow-SPU environment, same repository/shell paths:
python experiments/secure_AB_spu_poc.py --repetitions 3 --embedding "$f2/training/seed_218/Joint-C50-sigma2/final_embedding.npy" --partition "$f2/partitions/seed_218.json" --schedule "$f2/schedules/remote_cell_schedule_seed_218.csv" --output "$f5/spu"

# Return to the native environment for summary assembly and plotting.
conda activate federatedbatch
python scripts/summarize/assemble_figure5_plot_inputs.py --membership "$f5/release_attacks/membership_attack.csv" --reconstruction "$f5/expression_reconstruction/reconstruction_per_cell.csv" --accounting "$f5/privacy_accounting.csv" --runtime "$f5/spu/runtime_benchmark.csv" --correctness "$f5/spu/correctness.csv" --output "$f5/plot_inputs"
python scripts/render/make_figure5_variants.py --input "$f5/plot_inputs" --output figures/reproduction/figure5
```

The ABY3 experiment is a local three-party PoC, not a WAN deployment benchmark.
Formal DP protects only the remote A/B release channel, not end-to-end training.

## Quick redraw from bundled results (no training)

```bash
python scripts/render/render_figure2_result_only.py --input results/final_summaries/figure2 --output figures/redraw/figure2
python scripts/render/render_figure3_publication.py --input results/final_summaries/figure3 --output figures/redraw/figure3
python scripts/render/render_figure4_preprocessing_fixed_redesign.py --input results/final_summaries/figure4 --output figures/redraw/figure4
python scripts/render/make_figure5_variants.py --input results/plot_inputs/figure5 --output figures/redraw/figure5
```

These use saved inputs and do not validate fresh experiment results. Full
checkpoints and datasets are intentionally excluded from Git. Third-party
source/license details are in [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md);
environment limitations are in [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md).
