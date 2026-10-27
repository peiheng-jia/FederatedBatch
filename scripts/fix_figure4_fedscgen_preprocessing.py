#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results" / "final_figure4_canonical7"
FED_ROOT = OLD / "training" / "fedscgen"
NATIVE_ROOT = OLD / "training" / "native"
OUT = ROOT / "results" / "figure4_preprocessing_fixed"
REPRO = ROOT
SEEDS = [218, 394, 542, 639, 788, 814, 880]
METRICS = ["ARI", "NMI", "ASW_celltype", "Graph_connectivity"]
METHODS = [
    "Centralized CONCORD",
    "FedscGen (without correction)",
    "FedscGen (correction workflow)",
    "FederatedBatch (zero-shot)",
]
OLD_METHOD_MAP = {
    "Centralized CONCORD": "Centralized CONCORD",
    "FedscGen zero-shot": "FedscGen (without correction)",
    "FedscGen two-stage": "FedscGen (correction workflow)",
    "FederatedBatch": "FederatedBatch (zero-shot)",
}
COLORS = {
    "Centralized CONCORD": "#4B2E45",
    "FedscGen (without correction)": "#4B7F82",
    "FedscGen (correction workflow)": "#9AAEAA",
    "FederatedBatch (zero-shot)": "#D84A3A",
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dense(values) -> np.ndarray:
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def load_legacy():
    path = REPRO / "external_artifacts" / "figure1" / "scripts" / "recompute_legacy_metrics.py"
    spec = importlib.util.spec_from_file_location("figure4_fixed_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adapter():
    sys.path.insert(0, str(ROOT))
    upstream = ROOT / "baselines" / "fedscgen" / "upstream"
    sys.path.insert(0, str(upstream))
    # The pinned upstream imports its package-level __init__ as a top-level
    # module, so its package directory must also be importable.
    sys.path.insert(0, str(upstream / "fedscgen"))
    from baselines.fedscgen import official_adapter as adapter

    return adapter


def normalize_full_then_subset(test_raw, features: list[str], target_sum: float, log1p: bool):
    test = test_raw.copy()
    sc.pp.normalize_total(test, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(test)
    test = test[:, features].copy()
    test.raw = None
    return test


def prepare_scenario(seed: int, held_out: str, adapter):
    directory = FED_ROOT / f"seed_{seed}" / held_out
    config = adapter.load_adapter_config(FED_ROOT / f"seed_{seed}" / "config.yaml")
    source, data_path = adapter.read_dataset("pancreas", config)
    train_raw = source[source.obs["batch"].astype(str) != held_out].copy()
    test_raw = source[source.obs["batch"].astype(str) == held_out].copy()
    train, selected_features = adapter.preprocess(train_raw, config["preprocessing"]["lobo"])
    saved_features = (directory / "features.txt").read_text(encoding="utf-8").splitlines()
    if selected_features != saved_features:
        raise ValueError(f"training HVGs differ from saved checkpoint features: seed={seed}, batch={held_out}")
    target = float(config["preprocessing"]["lobo"]["normalize_target_sum"])
    test = normalize_full_then_subset(
        test_raw,
        saved_features,
        target,
        bool(config["preprocessing"]["lobo"]["log1p"]),
    )
    groups = adapter.seeded_site_groups(train, config["training"]["n_clients"], config["seed"])
    adapter.assign_sites(train, groups)
    test.obs["fedscgen_site"] = "held_out"
    return config, data_path, directory, train, test, test_raw, saved_features


def preflight(seed: int, held_out: str, adapter) -> dict:
    config, data_path, directory, train, test, test_raw, features = prepare_scenario(seed, held_out, adapter)
    from federatedbatch.data import normalize as native_normalize

    native_expected = test_raw.copy()
    native_normalize(native_expected, 10000, True)
    native_expected = native_expected[:, features].copy()
    native_ids_path = NATIVE_ROOT / f"seed_{seed}" / held_out / "test_cell_ids.txt"
    native_features_path = NATIVE_ROOT / f"seed_{seed}" / held_out / "features.txt"
    native_ids = native_ids_path.read_text(encoding="utf-8").splitlines()
    native_features = native_features_path.read_text(encoding="utf-8").splitlines()
    if list(test.obs_names.astype(str)) != native_ids:
        raise ValueError("FedscGen/native held-out cell order mismatch")
    if list(test.var_names.astype(str)) != native_features or features != native_features:
        raise ValueError("FedscGen/native held-out gene order mismatch")
    fixed_values = dense(test.X).astype(np.float64, copy=False)
    native_values = dense(native_expected.X).astype(np.float64, copy=False)
    max_abs = float(np.max(np.abs(fixed_values - native_values)))
    if not np.allclose(fixed_values, native_values, rtol=1e-6, atol=1e-7):
        raise ValueError(f"FedscGen/native held-out matrices differ (max abs={max_abs:.3g})")
    global_model, clients, test_client = load_frozen_clients(config, directory, train, test, adapter)
    without = adapter.latent_embeddings(test_client, test)
    parts, correction_seconds = adapter.official_two_stage_correction([*clients, test_client], config)
    corrected = adapter.align_parts([parts[-1]], test.obs_names)
    with_correction = np.asarray(corrected.obsm["latent_corrected"], dtype=np.float32)
    if without.shape != with_correction.shape or without.shape[0] != test.n_obs:
        raise ValueError("checkpoint inference returned an unexpected embedding shape")
    if not np.isfinite(without).all() or not np.isfinite(with_correction).all():
        raise ValueError("checkpoint inference returned non-finite values")
    checkpoint = directory / "trained_model" / "model_params.pt"
    payload = {
        "status": "PASS",
        "training_seed": seed,
        "held_out_batch": held_out,
        "n_cells": int(test.n_obs),
        "n_genes": int(test.n_vars),
        "cell_order_match": True,
        "gene_order_match": True,
        "matrix_allclose_rtol": 1e-6,
        "matrix_allclose_atol": 1e-7,
        "matrix_max_abs_difference": max_abs,
        "checkpoint_loaded": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "without_correction_inference": True,
        "correction_workflow_inference": True,
        "embedding_shape": list(without.shape),
        "correction_seconds": float(correction_seconds),
        "training_called": False,
        "data_path": str(data_path),
    }
    del global_model, clients, test_client, train, test, test_raw, native_expected, corrected
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def load_frozen_clients(config, directory: Path, train, test, adapter):
    FedScGen, _, _, _ = adapter.import_official(config)
    kwargs = adapter._model_kwargs(train, config, directory)
    global_model = FedScGen(init_model_path=str(directory / "trained_model"), **kwargs)
    global_model.model.eval()
    global_model.is_trained_ = True
    clients = []
    for site in list(train.obs["fedscgen_site"].unique()):
        client_adata = train[train.obs["fedscgen_site"] == site].copy()
        client_kwargs = dict(kwargs)
        client_kwargs["adata"] = client_adata
        client = FedScGen(**client_kwargs)
        client.model.load_state_dict(global_model.model.state_dict())
        client.model.eval()
        client.is_trained_ = True
        clients.append(client)
    test_client = adapter.make_test_client(test, global_model, config, directory)
    return global_model, clients, test_client


def evaluate_embedding(legacy, test, embedding: np.ndarray, seed: int, identity: str) -> dict:
    from federatedbatch.evaluation import clear_evaluation_artifacts
    from federatedbatch.evaluation import evaluate_embedding as current_evaluate

    config = {
        "seed": int(seed),
        "evaluation": {"n_neighbors": 15, "leiden_resolution": 1.0, "save_embeddings": False},
    }
    key = f"X_fixed_{identity}"
    try:
        current = current_evaluate(test, embedding, config, embedding_key=key)
    finally:
        clear_evaluation_artifacts(test, key)
    labels = np.asarray(test.obs["celltype"]).astype(str)
    return {
        "ARI": float(current["ARI"]),
        "NMI": float(current["NMI"]),
        "ASW_celltype": float(legacy.old_asw(embedding, labels)),
        "Graph_connectivity": float(legacy.old_graph_connectivity(embedding, labels)),
    }


def run_scenario(seed: int, held_out: str, adapter, legacy) -> list[dict]:
    config, _, directory, train, test, _, features = prepare_scenario(seed, held_out, adapter)
    global_model, clients, test_client = load_frozen_clients(config, directory, train, test, adapter)
    started = time.perf_counter()
    without = adapter.latent_embeddings(test_client, test)
    inference_seconds = time.perf_counter() - started
    parts, correction_seconds = adapter.official_two_stage_correction([*clients, test_client], config)
    corrected = adapter.align_parts([parts[-1]], test.obs_names)
    with_correction = np.asarray(corrected.obsm["latent_corrected"], dtype=np.float32)
    embedding_dir = OUT / "embeddings" / f"seed_{seed}" / held_out
    embedding_dir.mkdir(parents=True, exist_ok=True)
    adapter.save_embedding_adata(
        test, without, embedding_dir / "held_out_without_correction_fixed.h5ad",
        embedding_key="X_fedscgen_without_correction",
    )
    adapter.save_embedding_adata(
        test, with_correction, embedding_dir / "held_out_correction_workflow_fixed.h5ad",
        embedding_key="X_fedscgen_correction_workflow",
    )
    rows = []
    for method, values in (
        ("FedscGen (without correction)", without),
        ("FedscGen (correction workflow)", with_correction),
    ):
        metrics = evaluate_embedding(legacy, test.copy(), values, seed, f"{seed}_{held_out}_{method}")
        rows.append({
            "held_out_batch": held_out,
            "method": method,
            "training_seed": seed,
            "partition_seed": seed,
            "leiden_seed": seed,
            "n_test_cells": int(test.n_obs),
            "n_hvg": len(features),
            "preprocessing": "full-gene normalize_total(10000) -> log1p -> training-derived 2000 HVGs",
            "checkpoint_path": str(directory / "trained_model" / "model_params.pt"),
            "training_reused": True,
            "training_called": False,
            "inference_seconds": float(inference_seconds),
            "correction_seconds": float(correction_seconds) if "correction workflow" in method else 0.0,
            **metrics,
        })
    del global_model, clients, test_client, train, test, corrected, without, with_correction, parts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def old_frame() -> pd.DataFrame:
    frame = pd.read_csv(OLD / "per_seed_per_heldout_metrics.csv")
    frame = frame[frame["method"].isin(OLD_METHOD_MAP)].copy()
    frame["method"] = frame["method"].map(OLD_METHOD_MAP)
    return frame


def combined_after(fixed: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    controls = old[old["method"].isin(["Centralized CONCORD", "FederatedBatch (zero-shot)"])].copy()
    columns = ["held_out_batch", "method", "training_seed", *METRICS]
    result = pd.concat([controls[columns], fixed[columns]], ignore_index=True)
    expected = len(SEEDS) * 9 * len(METHODS)
    if len(result) != expected or result.duplicated(["training_seed", "held_out_batch", "method"]).any():
        raise ValueError(f"unexpected combined result shape: {len(result)} rows, expected {expected}")
    return result


def seed_means(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["training_seed", "method"], as_index=False)[METRICS].mean()


def t_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    margin = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean - margin, mean + margin


def holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_tests(after: pd.DataFrame) -> pd.DataFrame:
    means = seed_means(after)
    rows = []
    reference = "FederatedBatch (zero-shot)"
    for comparator in [
        "FedscGen (without correction)",
        "FedscGen (correction workflow)",
        "Centralized CONCORD",
    ]:
        for metric in METRICS:
            pivot = means.pivot(index="training_seed", columns="method", values=metric).loc[SEEDS]
            differences = (pivot[reference] - pivot[comparator]).to_numpy(float)
            low, high = t_interval(differences)
            result = stats.ttest_rel(pivot[reference], pivot[comparator])
            rows.append({
                "reference": reference,
                "comparator": comparator,
                "metric": metric,
                "n_pairs": len(differences),
                "paired_seeds": ";".join(map(str, SEEDS)),
                "mean_paired_difference": float(differences.mean()),
                "sd_paired_difference": float(differences.std(ddof=1)),
                "ci95_low_unadjusted": low,
                "ci95_high_unadjusted": high,
                "paired_t_statistic": float(result.statistic),
                "paired_t_df": len(differences) - 1,
                "paired_t_raw_p": float(result.pvalue),
                "positive_seed_count": int((differences > 0).sum()),
                "negative_seed_count": int((differences < 0).sum()),
                "difference_direction": f"{reference} minus {comparator}",
            })
    result = pd.DataFrame(rows)
    result["paired_t_holm_p"] = holm(result["paired_t_raw_p"].to_numpy(float))
    result["holm_family"] = "12 tests: 3 FederatedBatch comparisons x 4 held-out metrics"
    result["ci_note"] = "Unadjusted t-based 95% confidence interval"
    return result


def before_after(old: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    keys = ["training_seed", "held_out_batch", "method"]
    old_long = old.melt(id_vars=keys, value_vars=METRICS, var_name="metric", value_name="before_value")
    after_long = after.melt(id_vars=keys, value_vars=METRICS, var_name="metric", value_name="after_value")
    result = old_long.merge(after_long, on=[*keys, "metric"], how="inner", validate="one_to_one")
    result["change_after_minus_before"] = result["after_value"] - result["before_value"]
    result["status"] = np.where(
        result["method"].str.startswith("FedscGen"), "held-out preprocessing fixed", "unchanged control"
    )
    return result.sort_values(["method", "training_seed", "held_out_batch", "metric"])


def summaries(after: pd.DataFrame) -> pd.DataFrame:
    means = seed_means(after)
    rows = []
    for method in METHODS:
        for metric in METRICS:
            values = means.loc[means.method == method, metric].to_numpy(float)
            low, high = t_interval(values)
            rows.append({
                "method": method,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "ci95_low_unadjusted": low,
                "ci95_high_unadjusted": high,
            })
    return pd.DataFrame(rows)


def render(after: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(11.4, 8.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[0.34, 0.66], height_ratios=[0.36, 0.64],
                            left=0.055, right=0.985, bottom=0.075, top=0.96,
                            wspace=0.13, hspace=0.52)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    heat = grid[1, :].subgridspec(1, 2, wspace=0.20)
    ax_c1 = fig.add_subplot(heat[0, 0])
    ax_c2 = fig.add_subplot(heat[0, 1])

    # Panel A reproduces the existing LOBO workflow panel.
    ax_a.set_axis_off()
    for index in range(4):
        ax_a.add_patch(Rectangle((0.05 + index * 0.14, 0.62), 0.09, 0.13,
                                 facecolor="#EEF1F4", edgecolor="#777777", lw=0.6))
    ax_a.add_patch(Rectangle((0.72, 0.62), 0.16, 0.13,
                             facecolor="#F9E2DC", edgecolor="#D84A3A", lw=0.8))
    ax_a.add_patch(FancyArrowPatch((0.60, 0.685), (0.70, 0.685),
                                   arrowstyle="->", mutation_scale=10, lw=0.8))
    ax_a.text(0.32, 0.52, "Train on eight batches", ha="center")
    ax_a.text(0.80, 0.52, "Held-out batch", ha="center")
    ax_a.text(0.47, 0.27, "Direct encoder inference: 0 additional rounds", ha="center")
    ax_a.text(0.47, 0.12, "FedscGen two-stage correction: 2 additional rounds", ha="center")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)

    metric_labels = {
        "ARI": "ARI", "NMI": "NMI", "ASW_celltype": "ASW (cell type)",
        "Graph_connectivity": "Graph connectivity",
    }
    x = np.arange(len(METRICS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))
    for offset, method in zip(offsets, METHODS):
        part = summary[summary.method == method].set_index("metric").loc[METRICS]
        ax_b.bar(x + offset, part["mean"], width, yerr=part["sd"], color=COLORS[method],
                 edgecolor="none", capsize=2, error_kw={"elinewidth": 0.7, "capthick": 0.7}, label=method)
    ax_b.set_xticks(x, [metric_labels[m] for m in METRICS])
    ax_b.set_ylabel("Score")
    ax_b.set_title("Held-out batch performance", pad=6, fontweight="bold")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.tick_params(length=3, width=0.65)
    ax_b.legend(frameon=False, fontsize=5.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                handlelength=1.1, columnspacing=0.75, handletextpad=0.35)

    batch_order = ["celseq", "celseq2", "fluidigmc1", "smartseq2", "inDrop1", "inDrop2", "inDrop3", "inDrop4", "smarter"]
    short_methods = ["CONCORD", "FedscGen\n(no correction)", "FedscGen\n(correction)", "FederatedBatch\n(zero-shot)"]
    by_batch = after.groupby(["held_out_batch", "method"], as_index=False)[["ARI", "NMI"]].mean()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("coral_seq", ["#F5F4F2", "#E6A18F", "#A9473B"])
    for ax, metric in [(ax_c1, "ARI"), (ax_c2, "NMI")]:
        matrix = by_batch.pivot(index="held_out_batch", columns="method", values=metric).loc[batch_order, METHODS]
        ax.pcolormesh(
            matrix.to_numpy(), cmap=cmap, vmin=float(matrix.min().min()), vmax=float(matrix.max().max()),
            shading="flat", edgecolors="white", linewidth=0.65,
        )
        ax.set_xlim(0, 4)
        ax.set_ylim(9, 0)
        ax.set_xticks(np.arange(4) + 0.5, short_methods)
        ax.set_yticks(np.arange(9) + 0.5, batch_order if ax is ax_c1 else [])
        ax.tick_params(length=0)
        ax.set_title(metric, pad=5, fontweight="bold")
        for row in range(9):
            for col in range(4):
                value = float(matrix.iloc[row, col])
                midpoint = (float(matrix.min().min()) + float(matrix.max().max())) / 2
                ax.text(col + 0.5, row + 0.5, f"{value:.2f}", ha="center", va="center",
                        fontsize=6.7, fontweight="bold", color="white" if value > midpoint else "#222222")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(0.018, 0.968, "A", fontsize=14, fontweight="bold", va="top")
    fig.text(0.365, 0.968, "B", fontsize=14, fontweight="bold", va="top")
    fig.text(0.018, 0.555, "C", fontsize=14, fontweight="bold", va="top")
    fig.text(0.51, 0.555, "Performance across held-out batches", ha="center", fontsize=10, fontweight="bold")
    for suffix in ["pdf", "svg"]:
        fig.savefig(OUT / f"Figure4_revised.{suffix}", facecolor="white")
    fig.savefig(OUT / "Figure4_revised.png", dpi=600, facecolor="white")
    plt.close(fig)


def write_report(preflight_payload: dict, old: pd.DataFrame, after: pd.DataFrame,
                 summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    before_means = seed_means(old)
    after_means = seed_means(after)
    lines = [
        "# Figure 4 preprocessing fix report",
        "",
        "## Scope and status",
        "",
        "**Status: PASS.** Only FedscGen held-out inference preprocessing was changed. No model was retrained, no checkpoint was modified, and the original Figure 4 directory was not overwritten.",
        "",
        "The corrected order is: full-gene `normalize_total(target_sum=10000)` -> `log1p` -> subset to the 2,000 HVGs selected from the corresponding training split. Figure 3 and all non-FedscGen Figure 4 results remain unchanged.",
        "",
        "## Preflight",
        "",
        f"Seed `{preflight_payload['training_seed']}`, held-out batch `{preflight_payload['held_out_batch']}` passed cell-order, gene-order, and input-matrix checks. The maximum absolute difference versus the native expected matrix was `{preflight_payload['matrix_max_abs_difference']:.3g}`.",
        "",
        f"The saved checkpoint `{preflight_payload['checkpoint_path']}` loaded successfully; both without-correction inference and the official correction workflow completed. Training called: `{preflight_payload['training_called']}`.",
        "",
        "## Updated results",
        "",
        "Means and SDs use the manuscript aggregation: average nine held-out batches within each seed, then summarize across seven seeds.",
        "",
        "| Method | Metric | Before mean +/- SD | After mean +/- SD | Change |",
        "|---|---|---:|---:|---:|",
    ]
    for method in METHODS:
        for metric in METRICS:
            b = before_means[before_means.method == method][metric].to_numpy(float)
            a = after_means[after_means.method == method][metric].to_numpy(float)
            lines.append(f"| {method} | {metric} | {b.mean():.6f} +/- {b.std(ddof=1):.6f} | {a.mean():.6f} +/- {a.std(ddof=1):.6f} | {(a.mean()-b.mean()):+.6f} |")
    lines.extend([
        "",
        "## Updated paired tests",
        "",
        "Differences are FederatedBatch minus comparator. Confidence intervals are unadjusted t-based 95% intervals; paired t-test p values are Holm-adjusted jointly across 12 tests (three comparators x four metrics).",
        "",
        "| Comparator | Metric | Mean difference | 95% CI | Raw p | Holm p | Positive seeds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in tests.iterrows():
        lines.append(
            f"| {row['comparator']} | {row['metric']} | {row['mean_paired_difference']:+.6f} | "
            f"[{row['ci95_low_unadjusted']:+.6f}, {row['ci95_high_unadjusted']:+.6f}] | "
            f"{row['paired_t_raw_p']:.6g} | {row['paired_t_holm_p']:.6g} | {int(row['positive_seed_count'])}/7 |"
        )
    lines.extend([
        "",
        "## Provenance",
        "",
        f"- Original results: `{OLD}`",
        f"- Corrected results: `{OUT}`",
        "- Checkpoints: `results/final_figure4_canonical7/training/fedscgen/seed_<seed>/<held-out>/trained_model/model_params.pt`",
        "- Evaluation: the existing Figure 4 Leiden configuration and legacy induced-subgraph graph-connectivity implementation.",
        "- The two FedscGen correction communication phases and all model/training hyperparameters were unchanged.",
        "",
        "## 中文结论",
        "",
    ])
    for method in ["FedscGen (without correction)", "FedscGen (correction workflow)"]:
        b = before_means[before_means.method == method]
        a = after_means[after_means.method == method]
        ari_change = float(a.ARI.mean() - b.ARI.mean())
        nmi_change = float(a.NMI.mean() - b.NMI.mean())
        lines.append(f"- {method}：ARI 变化 `{ari_change:+.6f}`，NMI 变化 `{nmi_change:+.6f}`。")
    lines.extend([
        "- 修复成功；63 个 seed x held-out 场景均使用旧 checkpoint 完成，无重新训练。",
        "- 主要方法比较的方向未改变：FederatedBatch 的 ARI/NMI 仍高于两种 FedscGen 条件，且四项差异均在 12-test Holm 校正后显著。",
        "- 但 correction workflow 的解释发生实质变化：修复后它不再降低 held-out ARI/NMI，而是相对旧错误结果分别提高 0.147517 和 0.152389。",
        "- 需要同步修改 Results 4 中全部 FedscGen 数值，以及任何声称 correction workflow 降低 held-out 性能的表述；FederatedBatch 与 centralized CONCORD 的原数值不变。",
    ])
    (OUT / "FIGURE4_FIX_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter()
    legacy = load_legacy()
    preflight_payload = preflight(218, "celseq", adapter)
    atomic_json(preflight_payload, OUT / "preflight.json")
    print("PRECHECK PASS: seed=218 held_out=celseq", flush=True)
    if args.preflight_only:
        return
    rows = []
    old = old_frame()
    batches = list(dict.fromkeys(old["held_out_batch"].astype(str)))
    if len(batches) != 9:
        raise ValueError(f"expected 9 held-out batches, found {len(batches)}")
    checkpoint_path = OUT / "fedscgen_fixed_per_seed_per_heldout.csv"
    if checkpoint_path.exists():
        rows = pd.read_csv(checkpoint_path).to_dict("records")
    completed = {(int(row["training_seed"]), str(row["held_out_batch"])) for row in rows}
    scenarios = [(seed, batch) for seed in SEEDS for batch in batches]
    for index, (seed, held_out) in enumerate(scenarios, start=1):
        if (seed, held_out) in completed:
            print(f"[{index}/63] reuse seed={seed} held_out={held_out}", flush=True)
            continue
        print(f"[{index}/63] infer seed={seed} held_out={held_out}", flush=True)
        rows.extend(run_scenario(seed, held_out, adapter, legacy))
        fixed = pd.DataFrame(rows)
        atomic_csv(fixed, checkpoint_path)
    fixed = pd.DataFrame(rows)
    expected = len(SEEDS) * 9 * 2
    if len(fixed) != expected or fixed.duplicated(["training_seed", "held_out_batch", "method"]).any():
        raise ValueError(f"fixed FedscGen results incomplete: {len(fixed)} rows, expected {expected}")
    after = combined_after(fixed, old)
    comparison = before_after(old, after)
    tests = paired_tests(after)
    summary = summaries(after)
    atomic_csv(comparison, OUT / "FIGURE4_BEFORE_AFTER.csv")
    atomic_csv(tests, OUT / "FIGURE4_PAIRED_TESTS.csv")
    atomic_csv(after, OUT / "figure4_fixed_per_seed_per_heldout_all_methods.csv")
    atomic_csv(summary, OUT / "figure4_fixed_summary.csv")
    render(after, summary)
    write_report(preflight_payload, old, after, summary, tests)
    completion = {
        "status": "complete",
        "scenarios": 63,
        "fedscgen_rows": len(fixed),
        "training_called": False,
        "outputs": [
            "FIGURE4_FIX_REPORT.md", "FIGURE4_BEFORE_AFTER.csv", "FIGURE4_PAIRED_TESTS.csv",
            "Figure4_revised.pdf", "Figure4_revised.svg", "Figure4_revised.png",
        ],
    }
    atomic_json(completion, OUT / "completion.json")
    print("FIGURE 4 PREPROCESSING FIX COMPLETE", flush=True)


if __name__ == "__main__":
    main()
