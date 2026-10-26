#!/usr/bin/env python3
"""Canonical-seven Figure 1 rerun for Harmony, CONCORD and private FederatedBatch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SEEDS = (218, 394, 542, 639, 788, 814, 880)
EVAL_SEEDS = (0, 1, 2, 3, 4)
REPRESENTATIVE_SEED = 218
PCA_RANDOM_STATE = 42
OUT = ROOT / "results" / "reproduction" / "figure1"
REPRO = ROOT / "external_artifacts" / "figure1"
PCA_REFERENCE = REPRO / "results" / "figure1_rebuild" / "deterministic_baseline_embeddings.npz"

METRICS = ["ARI", "NMI", "ASW_batch", "ASW_cell", "Graph_connectivity", "iLISI"]
CANONICAL = list(SEEDS)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIVATE = load_module(
    "figure1_private_impl",
    ROOT / "experiments" / "finalize_figure1_private.py",
)
HARMONY = load_module(
    "figure1_harmony_impl",
    REPRO / "scripts" / "run_figure1_harmony_five_seed.py",
)
UMAP_PLOT = load_module(
    "figure1_umap_impl",
    ROOT / "experiments" / "finalize_figure1_statistics_umap.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_inputs():
    base = PRIVATE.load_config(str(PRIVATE.FIGURE1_CONFIG))
    base["data_dir"] = str(ROOT / "data")
    base["output_dir"] = str(OUT)
    adata, data_path = PRIVATE.read_dataset("pancreas", base["data_dir"], verify_hash=False)
    features = PRIVATE.FEATURES.read_text(encoding="utf-8").splitlines()
    if len(features) != 2000:
        raise ValueError(f"expected 2000 Figure 1 features, got {len(features)}")
    PRIVATE.normalize(adata, 10000, True)
    groups, partition_payload = PRIVATE.load_partition(adata.n_obs)
    schedule, schedule_frame = PRIVATE.build_schedule(groups, np.asarray(adata.obs_names))
    return base, adata, data_path, features, groups, partition_payload, schedule, schedule_frame


def preflight() -> dict:
    existing = set(path.name for path in OUT.iterdir()) if OUT.exists() else set()
    if existing - {"preflight.json"}:
        raise RuntimeError(f"new output directory is not empty: {OUT}")
    base, adata, data_path, features, groups, partition_payload, schedule, schedule_frame = load_inputs()
    if tuple(SEEDS) != (218, 394, 542, 639, 788, 814, 880):
        raise RuntimeError("canonical seed list mismatch")
    if len(groups) != 6 or len(features) != 2000:
        raise RuntimeError("Figure 1 client/feature protocol mismatch")
    if schedule_frame["cell_id"].nunique() != len(schedule_frame):
        raise RuntimeError("Figure 1 remote schedule is not one-use")
    cfg = PRIVATE.configure(base, REPRESENTATIVE_SEED, schedule)
    queue = cfg["federated"]["queue"]
    checks = {
        "harmony_entrypoint": callable(HARMONY.harmony_embedding),
        "pca_entrypoint": callable(__import__("federatedbatch.baselines", fromlist=["run_pca"]).run_pca),
        "centralized_entrypoint": callable(__import__("federatedbatch.baselines", fromlist=["run_centralized"]).run_centralized),
        "private_federated_entrypoint": callable(PRIVATE.train_federated),
        "canonical_seeds": list(SEEDS),
        "n_clients": cfg["federated"]["n_clients"] == 6,
        "private_queue_mode": queue["mode"] == "round_once_joint_cellclip_noise",
        "private_C": math.isclose(float(queue["joint_clipping_threshold"]), PRIVATE.C, rel_tol=0, abs_tol=1e-9),
        "private_sigma": float(queue["noise_multiplier"]) == PRIVATE.SIGMA,
        "active_rounds": queue["active_rounds"] == list(range(4, 21)),
        "evaluation_leiden_seeds": list(EVAL_SEEDS),
        "umap_random_state": 42,
        "new_output_directory": not bool(existing - {"preflight.json"}),
        "partition_seed": int(partition_payload.get("partition_seed", 20260918)) == 20260918,
    }
    if not all(value is True for key, value in checks.items() if key not in {"canonical_seeds", "evaluation_leiden_seeds", "umap_random_state"}):
        raise RuntimeError(f"preflight failed: {checks}")
    return {
        "status": "PASS",
        "output": str(OUT),
        "dataset": str(data_path),
        "dataset_sha256": sha256(data_path),
        "canonical_training_seeds": list(SEEDS),
        "evaluation_leiden_seeds": list(EVAL_SEEDS),
        "umap_random_state": 42,
        "umap_representative_training_seed": REPRESENTATIVE_SEED,
        "pca_random_state": PCA_RANDOM_STATE,
        "n_clients": 6,
        "partition_seed": 20260918,
        "remote_schedule_rows": int(len(schedule_frame)),
        "max_remote_participation": int(schedule_frame.groupby("cell_id").size().max()),
        "private_setting": {
            "mode": "RoundOnce joint cell-level clipping + one joint Gaussian release",
            "C": PRIVATE.C,
            "sigma": PRIVATE.SIGMA,
            "tau_q": 0.20,
            "alpha": 0.10,
            "active_rounds": list(range(4, 21)),
            "remote_cells_per_source_per_active_round": 102,
        },
        "checks": checks,
    }


def evaluate_row(legacy, adata, cfg, embedding: np.ndarray, method: str, seed: int, path: Path, elapsed: float) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("analysis", {})
    cfg["analysis"].setdefault("leiden_random_states", list(EVAL_SEEDS))
    cfg["analysis"].setdefault("purity_k", 15)
    metric = PRIVATE.evaluate(legacy, adata, cfg, embedding, f"fig1_canonical7_{method}_{seed}")
    return {
        "method": method,
        "training_seed": int(seed),
        **{name: float(metric[name]) for name in METRICS},
        "ARI_leiden_sd": float(metric["ARI_sd"]),
        "NMI_leiden_sd": float(metric["NMI_sd"]),
        "training_seconds": float(elapsed),
        "training_stable": bool(np.isfinite(embedding).all()),
        "embedding_path": str(path),
        "embedding_sha256": sha256(path),
        "evaluation_leiden_seeds": ";".join(map(str, EVAL_SEEDS)),
        "evaluation_replicates_are_not_training_replicates": True,
        "leiden_repeats": metric["leiden_repeats"],
    }


def load_previous_rows() -> pd.DataFrame:
    path = OUT / "per_seed_metrics_partial.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def run_methods():
    OUT.mkdir(parents=True, exist_ok=True)
    base, adata, data_path, features, groups, partition_payload, schedule, schedule_frame = load_inputs()
    write_csv(schedule_frame, OUT / "remote_cell_schedule.csv")
    write_json(OUT / "preflight.json", preflight_payload)
    legacy = PRIVATE.load_legacy()
    rows = load_previous_rows()
    completed = {(str(r.method), int(r.training_seed)) for r in rows.itertuples()} if not rows.empty else set()

    from federatedbatch.baselines import run_centralized, run_pca

    # PCA is generated from the same normalized matrix and fixed feature list.
    # The bundled historical array is an optional benchmark, never a required input.
    pca_path = OUT / "pca" / "final_embedding.npy"
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    pca_config = copy.deepcopy(base)
    pca_config["seed"] = PCA_RANDOM_STATE
    if pca_path.exists():
        pca_embedding = np.load(pca_path).astype(np.float32, copy=False)
        pca_elapsed = 0.0
    else:
        pca_embedding, pca_elapsed = run_pca(adata, features, pca_config)
        pca_embedding = np.asarray(pca_embedding, dtype=np.float32)
        np.save(pca_path, pca_embedding)
    if pca_embedding.shape != (adata.n_obs, base["federated"]["latent_dim"]):
        raise RuntimeError(f"invalid generated PCA shape: {pca_embedding.shape}")
    pca_row = evaluate_row(
        legacy, adata, pca_config, pca_embedding, "PCA", PCA_RANDOM_STATE, pca_path, pca_elapsed
    )
    pca_row["training_seed"] = "deterministic"
    write_csv(pd.DataFrame([pca_row]), OUT / "pca_metrics.csv")

    pca_qc = {
        "generated_path": str(pca_path),
        "generated_sha256": sha256(pca_path),
        "random_state": PCA_RANDOM_STATE,
        "reference_present": PCA_REFERENCE.is_file(),
    }
    if PCA_REFERENCE.is_file():
        with np.load(PCA_REFERENCE) as cache:
            reference = np.asarray(cache["PCA"], dtype=np.float32)
        pca_qc.update(
            {
                "reference_path": str(PCA_REFERENCE),
                "same_shape": reference.shape == pca_embedding.shape,
                "max_abs_difference": (
                    float(np.max(np.abs(reference - pca_embedding)))
                    if reference.shape == pca_embedding.shape
                    else math.nan
                ),
                "allclose_rtol_1e-5_atol_1e-6": bool(
                    reference.shape == pca_embedding.shape
                    and np.allclose(reference, pca_embedding, rtol=1e-5, atol=1e-6)
                ),
            }
        )
    write_json(OUT / "pca_reproduction_qc.json", pca_qc)

    for seed in SEEDS:
        # Harmony: fixed PCA input, original max_iter and evaluation seeds.
        key = ("Harmony", seed)
        path = OUT / "harmony" / f"seed_{seed}" / "final_embedding.npy"
        if key not in completed or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            embedding = HARMONY.harmony_embedding(pca_embedding, adata.obs, seed)
            np.save(path, embedding)
            elapsed = time.perf_counter() - started
            row = evaluate_row(legacy, adata, {**base, "seed": seed}, embedding, "Harmony", seed, path, elapsed)
            rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
            write_csv(rows, OUT / "per_seed_metrics_partial.csv")
            print(f"completed Harmony seed {seed}", flush=True)

        # Centralized CONCORD: same normalized data/features and 100 epochs as Figure 1.
        key = ("centralized_CONCORD", seed)
        path = OUT / "centralized" / f"seed_{seed}" / "final_embedding.npy"
        if key not in completed or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            cfg = copy.deepcopy(base)
            cfg["seed"] = int(seed)
            started = time.perf_counter()
            embedding, elapsed_train = run_centralized(adata, features, cfg, path.parent / "concord")
            elapsed = time.perf_counter() - started
            np.save(path, embedding.astype(np.float32))
            row = evaluate_row(legacy, adata, cfg, np.asarray(embedding), "centralized_CONCORD", seed, path, elapsed)
            row["concord_training_seconds"] = float(elapsed_train)
            rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
            write_csv(rows, OUT / "per_seed_metrics_partial.csv")
            print(f"completed centralized CONCORD seed {seed}", flush=True)

        # Final private FederatedBatch: exact locked setting, fixed six-client partition.
        key = ("FederatedBatch_final_private", seed)
        path = OUT / "federatedbatch" / "training" / f"seed_{seed}" / "final_embedding.npy"
        if key not in completed or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            cfg = PRIVATE.configure(base, seed, schedule)
            started = time.perf_counter()
            result = PRIVATE.train_federated(
                adata,
                features,
                None,
                cfg,
                path.parent,
                domain_key="batch",
                client_indices=groups,
            )
            elapsed = time.perf_counter() - started
            embedding = result.embeddings.astype(np.float32)
            np.save(path, embedding)
            pd.DataFrame(result.joint_clipping_diagnostics).to_csv(path.parent / "clipping_diagnostics.csv", index=False)
            pd.DataFrame(result.joint_noise_diagnostics).to_csv(path.parent / "noise_diagnostics.csv", index=False)
            row = evaluate_row(legacy, adata, cfg, embedding, "FederatedBatch_final_private", seed, path, elapsed)
            clip = pd.DataFrame(result.joint_clipping_diagnostics)
            targets = pd.DataFrame(result.joint_noise_diagnostics)
            targets = targets[targets.record_type == "target_postprocess"] if len(targets) else targets
            invalid = int(targets.invalid_a_before_clamp_count.sum()) if len(targets) else 0
            acount = int(targets.a_count.sum()) if len(targets) else 0
            row.update({
                "clipping_fraction": float(clip.clipped.astype(bool).mean()) if len(clip) else math.nan,
                "invalid_a_frequency": float(invalid / acount) if acount else math.nan,
                "invalid_a_count": invalid,
                "a_count": acount,
                "max_remote_participation_count": int(schedule_frame.groupby("cell_id").size().max()),
            })
            rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
            write_csv(rows, OUT / "per_seed_metrics_partial.csv")
            print(f"completed final private FederatedBatch seed {seed}", flush=True)

    rows = rows.sort_values(["method", "training_seed"]).reset_index(drop=True)
    if set(rows.method) != {"Harmony", "centralized_CONCORD", "FederatedBatch_final_private"}:
        raise RuntimeError("one or more methods did not produce rows")
    for method in rows.method.unique():
        observed = set(rows.loc[rows.method == method, "training_seed"].astype(int))
        if observed != set(SEEDS):
            raise RuntimeError(f"{method} completed seeds {sorted(observed)}, expected {list(SEEDS)}")
    write_csv(rows, OUT / "per_seed_metrics.csv")
    summarize(rows)
    make_source_and_umap(rows, adata, pca_row, pca_embedding)
    write_final_report(rows)


def summarize(rows: pd.DataFrame) -> None:
    summary = []
    for method in ["Harmony", "centralized_CONCORD", "FederatedBatch_final_private"]:
        subset = rows[rows.method == method]
        for metric in METRICS:
            values = subset[metric].to_numpy(float)
            sd = float(np.std(values, ddof=1))
            margin = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
            summary.append({"method": method, "metric": metric, "n": len(values), "mean": float(values.mean()), "sd": sd, "ci95_low": float(values.mean() - margin), "ci95_high": float(values.mean() + margin), "training_seeds": ";".join(map(str, SEEDS))})
    write_csv(pd.DataFrame(summary), OUT / "metric_summary.csv")

    tests = []
    fb = rows[rows.method == "FederatedBatch_final_private"].set_index("training_seed")
    for comparator in ["centralized_CONCORD", "Harmony"]:
        comp = rows[rows.method == comparator].set_index("training_seed")
        for metric in METRICS:
            difference = fb.loc[list(SEEDS), metric].to_numpy(float) - comp.loc[list(SEEDS), metric].to_numpy(float)
            t_stat, t_p = stats.ttest_rel(fb.loc[list(SEEDS), metric], comp.loc[list(SEEDS), metric])
            try:
                wil = stats.wilcoxon(difference, alternative="two-sided", method="exact")
                wil_p = float(wil.pvalue)
                wil_stat = float(wil.statistic)
            except ValueError:
                wil_p = math.nan
                wil_stat = math.nan
            mean = float(difference.mean())
            sd = float(difference.std(ddof=1))
            margin = float(stats.t.ppf(0.975, 6) * sd / math.sqrt(7))
            tests.append({"reference":"FederatedBatch_final_private", "comparator":comparator, "metric":metric, "n_pairs":7, "seed_identifiers":";".join(map(str, SEEDS)), "paired_differences":";".join(f"{x:.17g}" for x in difference), "mean_paired_difference":mean, "sd_paired_difference":sd, "ci95_low":mean-margin, "ci95_high":mean+margin, "t_statistic":float(t_stat), "df":6, "paired_t_raw_p":float(t_p), "wilcoxon_statistic":wil_stat, "wilcoxon_raw_p":wil_p, "difference_direction":"FederatedBatch_minus_comparator", "metric_note":"Raw ASW_batch; larger is not automatically better." if metric == "ASW_batch" else ""})
    tests_df = pd.DataFrame(tests)
    tests_df["paired_t_holm_p"] = holm(tests_df["paired_t_raw_p"].to_numpy(float))
    tests_df["wilcoxon_holm_p"] = holm(tests_df["wilcoxon_raw_p"].to_numpy(float))
    tests_df["multiplicity_family"] = "12 tests per test family: 6 metrics x 2 comparators"
    write_csv(tests_df, OUT / "paired_tests.csv")


def holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    order = np.argsort(values[finite])
    indices = np.flatnonzero(finite)[order]
    running = 0.0
    for rank, index in enumerate(indices):
        running = max(running, min(1.0, (len(indices) - rank) * values[index]))
        output[index] = running
    return output


def make_source_and_umap(
    rows: pd.DataFrame, adata, pca_row: dict, pca_embedding: np.ndarray
) -> None:
    replacements = []
    for metric in METRICS:
        replacements.append(
            {
                "method": "PCA",
                "training_seed": "deterministic",
                "metric": metric,
                "value": float(pca_row[metric]),
                "source": str(OUT / "pca_metrics.csv"),
                "evaluation_definition": "Figure 1 legacy evaluation computed from repository-generated deterministic PCA",
                "value_unit": "one deterministic embedding",
            }
        )
    for method, output_method in [("Harmony", "Harmony"), ("centralized_CONCORD", "CONCORD"), ("FederatedBatch_final_private", "FederatedBatch")]:
        for record in rows[rows.method == method].itertuples():
            for metric in METRICS:
                replacements.append({"method": output_method, "training_seed": int(record.training_seed), "metric": metric, "value": float(getattr(record, metric)), "source": str(OUT / "per_seed_metrics.csv"), "evaluation_definition": "Figure 1 legacy metrics; five fixed Leiden states 0-4 averaged for ARI/NMI; raw ASW, legacy graph connectivity and full-pairwise legacy iLISI", "value_unit": "method-level value"})
    source = pd.DataFrame(replacements)
    write_csv(source, OUT / "figure1B_source_values_canonical7.csv")

    embeddings = {
        "PCA": np.asarray(pca_embedding, dtype=np.float32),
        "Harmony": np.load(OUT / "harmony" / f"seed_{REPRESENTATIVE_SEED}" / "final_embedding.npy").astype(np.float32),
        "CONCORD": np.load(OUT / "centralized" / f"seed_{REPRESENTATIVE_SEED}" / "final_embedding.npy").astype(np.float32),
        "FederatedBatch": np.load(OUT / "federatedbatch" / "training" / f"seed_{REPRESENTATIVE_SEED}" / "final_embedding.npy").astype(np.float32),
    }
    for key, value in embeddings.items():
        if value.shape != (adata.n_obs, 100) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid representative embedding {key}: {value.shape}")
    batch_coords = UMAP_PLOT.plot_umap(adata, embeddings, "batch", UMAP_PLOT.BATCH_COLORS, OUT / "figure1C_final.png", "Batch")
    cell_coords = UMAP_PLOT.plot_umap(adata, embeddings, "celltype", UMAP_PLOT.CELLTYPE_COLORS, OUT / "figure1D_final.png", "Cell Type")
    if not all(np.array_equal(batch_coords[key], cell_coords[key]) for key in embeddings):
        raise RuntimeError("Figure 1C/D UMAP coordinates differ")
    coord = pd.DataFrame({"cell_id": adata.obs_names.astype(str), "batch": adata.obs["batch"].astype(str).to_numpy(), "celltype": adata.obs["celltype"].astype(str).to_numpy()})
    for method, values in batch_coords.items():
        coord[f"UMAP1_{method}"] = values[:, 0]
        coord[f"UMAP2_{method}"] = values[:, 1]
    write_csv(coord, OUT / "figure1_umap_coordinates_final.csv")


def write_final_report(rows: pd.DataFrame) -> None:
    tests = pd.read_csv(OUT / "paired_tests.csv")
    summary = pd.read_csv(OUT / "metric_summary.csv")
    lines = [
        "# Figure 1 canonical-seven rerun",
        "",
        "Status: COMPLETE when all three methods contain the seven canonical training seeds.",
        "",
        "## Locked protocol",
        "",
        "- Training seeds: 218, 394, 542, 639, 788, 814, 880.",
        "- Dataset: Pancreas, 16,382 cells, six clients, fixed partition seed 20260918.",
        "- FederatedBatch: final private RoundOnce joint cell-level clipping, C=1206.3391650512762, sigma=2, tau_q=0.20, alpha=0.10, 102 remote cells/source/active round, active rounds 4-20.",
        "- Leiden/evaluation seeds: 0,1,2,3,4; these are evaluation repeats, not independent training replicates.",
        "- UMAP: n_neighbors=15, min_dist=0.1, Euclidean, random_state=42; representative training seed 218 because 3205 is not in canonical seven.",
        "- ASW, graph connectivity and iLISI use the legacy Figure 1 definitions.",
        "",
        "## Method completion",
        "",
        "| Method | Completed seeds | Stable |",
        "|---|---|---|",
    ]
    for method in ["Harmony", "centralized_CONCORD", "FederatedBatch_final_private"]:
        subset = rows[rows.method == method]
        lines.append(f"| {method} | {','.join(map(str, sorted(subset.training_seed.astype(int))))} | {bool(subset.training_stable.astype(bool).all())} |")
    lines += ["", "## Mean +/- SD", "", "| Method | Metric | n | Mean | SD |", "|---|---|---:|---:|---:|"]
    for record in summary.itertuples():
        lines.append(f"| {record.method} | {record.metric} | {int(record.n)} | {record.mean:.8g} | {record.sd:.8g} |")
    lines += ["", "## Paired tests", "", "Final FederatedBatch minus comparator; two-sided paired t-tests and exact Wilcoxon signed-rank sensitivity analyses are reported. Holm correction is applied separately within each 12-test family.", "", "| Comparator | Metric | Mean delta | 95% CI | t(6) | t raw p | t Holm p | Wilcoxon raw p | Wilcoxon Holm p |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for record in tests.itertuples():
        lines.append(f"| {record.comparator} | {record.metric} | {record.mean_paired_difference:.8g} | [{record.ci95_low:.8g}, {record.ci95_high:.8g}] | {record.t_statistic:.8g} | {record.paired_t_raw_p:.8g} | {record.paired_t_holm_p:.8g} | {record.wilcoxon_raw_p:.8g} | {record.wilcoxon_holm_p:.8g} |")
    lines += ["", "## Figure readiness", "", "The rerun is ready for Figure 1 layout work only if this report has status COMPLETE, all three method blocks contain all seven seeds, all training_stable values are true, and the representative UMAP files exist. Historical Figure 1 outputs remain untouched.", "", "Outputs: `per_seed_metrics.csv`, `metric_summary.csv`, `paired_tests.csv`, `figure1B_source_values_canonical7.csv`, `figure1_umap_coordinates_final.csv`, `figure1C_final.png`, `figure1D_final.png`.", ""]
    (OUT / "FIGURE1_CANONICAL7_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUT, preflight_payload
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if args.preflight:
        payload = preflight()
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "preflight.json", payload)
        print(json.dumps(payload, indent=2), flush=True)
        return
    if not args.resume and OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"output exists; use --resume only for this same run: {OUT}")
    preflight_payload = preflight() if not OUT.exists() else json.loads((OUT / "preflight.json").read_text(encoding="utf-8"))
    run_methods()
    print(f"Figure 1 canonical-seven rerun completed: {OUT}", flush=True)


if __name__ == "__main__":
    main()
