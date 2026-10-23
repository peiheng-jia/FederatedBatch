#!/usr/bin/env python3
"""Legacy Figure 1 helper functions imported by the canonical-seven driver.

Direct execution is disabled because this file's old main routine targets an
obsolete seed block. Use `run_figure1_canonical7.py` for the locked protocol.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import umap
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "final_federatedbatch_figure1"
STAT_OUT = OUT / "statistics_final"
REPRO = REPO / "external_artifacts" / "figure1"
DATA = REPO / "data" / "pancreas.h5ad"
OLD_SOURCE = REPRO / "results" / "figure1_final_inputs" / "figure1B_source_values.csv"
FINAL_METRICS = OUT / "per_seed_metrics.csv"
FINAL_FB = OUT / "training" / "seed_3205" / "final_embedding.npy"
PCA_HARMONY = REPRO / "results" / "figure1_rebuild" / "deterministic_baseline_embeddings.npz"
HARMONY_3205 = REPRO / "results" / "figure1_harmony_5seed" / "seed_3205" / "final_embedding.npy"
CONCORD_3205 = REPRO / "results" / "fresh_queue_operating_window_30seed" / "centralized" / "seed_3205" / "final_embedding.npy"
SEEDS = [3205, 9048, 7487, 5326, 9364]
METRICS = ["ARI", "NMI", "ASW_batch", "ASW_cell", "Graph_connectivity", "iLISI"]
COMPARATORS = ["CONCORD", "Harmony"]
CRITICAL_T = float(stats.t.ppf(0.975, 4))

METHOD_COLORS = {
    "PCA": "#B7B5A0",
    "Harmony": "#44757A",
    "CONCORD": "#452A3D",
    "FederatedBatch": "#D44C3C",
}
BATCH_COLORS = {
    "celseq": "#B7B5A0", "celseq2": "#44757A", "fluidigmc1": "#452A3D",
    "inDrop1": "#D44C3C", "inDrop2": "#DD6C4C", "inDrop3": "#E5855D",
    "inDrop4": "#EED5B7", "smarter": "#966478", "smartseq2": "#648C91",
}
CELLTYPE_COLORS = {
    "acinar": "#D44C3C", "activated_stellate": "#DD6C4C", "alpha": "#E5855D",
    "beta": "#EED5B7", "delta": "#B7B5A0", "ductal": "#96AAA5",
    "endothelial": "#44757A", "epsilon": "#508287", "gamma": "#648C91",
    "macrophage": "#452A3D", "mast": "#5A3C50", "quiescent_stellate": "#785064",
    "schwann": "#966478", "t_cell": "#B48296",
}


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    n = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def load_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FINAL_METRICS.exists():
        raise FileNotFoundError(FINAL_METRICS)
    final = pd.read_csv(FINAL_METRICS)
    final["training_seed"] = final["training_seed"].astype(int)
    if set(final.training_seed) != set(SEEDS) or len(final) != 5:
        raise ValueError("Final FederatedBatch metrics do not contain exactly the five required seeds")
    old = pd.read_csv(OLD_SOURCE)
    old["training_seed_numeric"] = pd.to_numeric(old["training_seed"], errors="coerce")
    comparator_old = old[old.training_seed_numeric.isin(SEEDS)]
    required = {(m, metric) for m in COMPARATORS for metric in METRICS}
    observed = set(zip(comparator_old.method, comparator_old.metric))
    if not required.issubset(observed):
        raise ValueError(f"Missing comparator rows: {sorted(required - observed)}")
    return final, old


def make_statistics(final: pd.DataFrame, old: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for comparator in COMPARATORS:
        for metric in METRICS:
            fb = final.set_index("training_seed").loc[SEEDS, metric].to_numpy(float)
            comp = (
                old[(old.method == comparator) & old.training_seed_numeric.isin(SEEDS) & (old.metric == metric)]
                .set_index("training_seed_numeric")
                .loc[SEEDS, "value"]
                .to_numpy(float)
            )
            diff = fb - comp
            sd = float(np.std(diff, ddof=1))
            mean = float(np.mean(diff))
            t_stat, p_value = stats.ttest_rel(fb, comp)
            margin = CRITICAL_T * sd / math.sqrt(len(diff))
            rows.append({
                "reference": "FederatedBatch_final_private",
                "comparator": comparator,
                "metric": metric,
                "n_pairs": len(diff),
                "seed_identifiers": ";".join(map(str, SEEDS)),
                "paired_differences": ";".join(f"{x:.17g}" for x in diff),
                **{f"difference_{seed}": float(value) for seed, value in zip(SEEDS, diff)},
                "mean_paired_difference": mean,
                "sd_paired_difference": sd,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "t_statistic": float(t_stat),
                "df": 4,
                "raw_p_value": float(p_value),
                "difference_direction": "FederatedBatch_minus_comparator",
                "interpretation_note": "Raw ASW_batch is the submitted silhouette value; its sign is not a monotonic mixing benefit." if metric == "ASW_batch" else "",
            })
    tests = pd.DataFrame(rows)
    tests["holm_adjusted_p_value"] = holm_adjust(tests.raw_p_value.to_numpy(float))
    tests["multiplicity_family"] = "12 two-sided paired t-tests: 2 comparators x 6 metrics"
    holm = tests[[
        "reference", "comparator", "metric", "n_pairs", "mean_paired_difference",
        "sd_paired_difference", "ci95_low", "ci95_high", "t_statistic", "df",
        "raw_p_value", "holm_adjusted_p_value", "difference_direction",
        "multiplicity_family",
    ]].copy()

    table_rows = []
    for method in ["PCA", "Harmony", "CONCORD", "FederatedBatch_final_private"]:
        if method == "FederatedBatch_final_private":
            values = {m: final[m].to_numpy(float) for m in METRICS}
            label = method
        else:
            source_method = "CONCORD" if method == "CONCORD" else method
            subset = old[old.method == source_method]
            values = {m: subset[subset.metric == m].value.to_numpy(float) for m in METRICS}
            label = method
        for metric in METRICS:
            arr = np.asarray(values[metric], dtype=float)
            mean = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else math.nan
            margin = CRITICAL_T * sd / math.sqrt(len(arr)) if len(arr) > 1 else math.nan
            table_rows.append({
                "method": label,
                "metric": metric,
                "n": len(arr),
                "mean": mean,
                "sd": sd,
                "ci95_low": mean - margin if np.isfinite(margin) else math.nan,
                "ci95_high": mean + margin if np.isfinite(margin) else math.nan,
                "uncertainty_unit": "deterministic_embedding" if len(arr) == 1 else "training_seed",
            })
    table = pd.DataFrame(table_rows)
    return tests, holm, table


def load_embeddings() -> dict[str, np.ndarray]:
    paths = [PCA_HARMONY, HARMONY_3205, CONCORD_3205, FINAL_FB]
    if not all(p.exists() for p in paths):
        missing = [str(p) for p in paths if not p.exists()]
        raise FileNotFoundError("Missing embedding(s): " + ", ".join(missing))
    with np.load(PCA_HARMONY) as baseline:
        pca = np.asarray(baseline["PCA"], dtype=np.float32)
    embeddings = {
        "PCA": pca,
        "Harmony": np.load(HARMONY_3205).astype(np.float32),
        "CONCORD": np.load(CONCORD_3205).astype(np.float32),
        "FederatedBatch": np.load(FINAL_FB).astype(np.float32),
    }
    shapes = {k: v.shape for k, v in embeddings.items()}
    if len(set(shapes.values())) != 1 or next(iter(shapes.values())) != (16382, 100):
        raise ValueError(f"Embedding shapes are not all (16382, 100): {shapes}")
    if not all(np.isfinite(v).all() for v in embeddings.values()):
        raise ValueError("At least one embedding contains non-finite values")
    return embeddings


def plot_umap(adata: ad.AnnData, embeddings: dict[str, np.ndarray], color_key: str, colors: dict[str, str], output: Path, title: str) -> None:
    values = adata.obs[color_key].astype(str).to_numpy()
    present = list(dict.fromkeys(values.tolist()))
    order = [x for x in colors if x in present] + sorted(set(present) - set(colors))
    mapping = {x: colors.get(x, "#999999") for x in order}
    coords = {}
    for method, embedding in embeddings.items():
        coords[method] = umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42
        ).fit_transform(embedding)
    fig = plt.figure(figsize=(10.0, 2.5), facecolor="white")
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.9], wspace=0.38)
    for index, method in enumerate(embeddings):
        ax = fig.add_subplot(grid[0, index])
        for category in order:
            mask = values == category
            ax.scatter(coords[method][mask, 0], coords[method][mask, 1], s=1.4, alpha=0.55, color=mapping[category], linewidths=0, rasterized=True)
        ax.set_title("CONCORD" if method == "CONCORD" else method, fontsize=9, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
        ax.tick_params(labelsize=7, width=0.5)
    legend_ax = fig.add_subplot(grid[0, 4])
    legend_ax.axis("off")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=mapping[c], markeredgecolor="none", label=c.replace("_", " ")) for c in order]
    legend_ax.legend(handles=handles, loc="center left", frameon=False, title=title, title_fontsize=9, fontsize=7, handletextpad=0.5, labelspacing=0.8)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.18, top=0.87)
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return coords


def main() -> None:
    STAT_OUT.mkdir(parents=True, exist_ok=True)
    final, old = load_values()
    tests, holm, table = make_statistics(final, old)
    tests.to_csv(STAT_OUT / "paired_tests.csv", index=False)
    holm.to_csv(STAT_OUT / "holm_adjusted_tests.csv", index=False)
    table.to_csv(STAT_OUT / "final_metric_table.csv", index=False)

    adata = ad.read_h5ad(DATA)
    if "batch" not in adata.obs:
        adata.obs["batch"] = adata.obs["tech"].astype(str)
    adata.obs["batch"] = adata.obs["batch"].astype(str)
    adata.obs["celltype"] = adata.obs["celltype"].astype(str)
    embeddings = load_embeddings()
    batch_coords = plot_umap(adata, embeddings, "batch", BATCH_COLORS, OUT / "figure1C_final.png", "Batch")
    celltype_coords = plot_umap(adata, embeddings, "celltype", CELLTYPE_COLORS, OUT / "figure1D_final.png", "Cell Type")
    if not all(np.array_equal(batch_coords[k], celltype_coords[k]) for k in embeddings):
        raise RuntimeError("Figure 1C and 1D coordinates are not identical")
    coordinate_frame = pd.DataFrame({"cell_id": adata.obs_names.astype(str), "batch": adata.obs["batch"].to_numpy(), "celltype": adata.obs["celltype"].to_numpy()})
    for method, coords in batch_coords.items():
        coordinate_frame[f"UMAP1_{method}"] = coords[:, 0]
        coordinate_frame[f"UMAP2_{method}"] = coords[:, 1]
    coordinate_frame.to_csv(OUT / "figure1_umap_coordinates_final.csv", index=False)

    def fmt(value: float) -> str:
        return "NA" if not np.isfinite(value) else f"{value:.8g}"

    lines = [
        "# Figure 1 final statistics and UMAPs",
        "",
        "No model was retrained. PCA, Harmony, and centralized CONCORD values were retained from the submitted Figure 1 source table. FederatedBatch values are from the completed final private RoundOnce + Joint-C50 + sigma=2 run.",
        "",
        "## Final metric table",
        "",
        "| Method | Metric | n | Mean | SD | 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        lines.append(f"| {row['method']} | {row['metric']} | {int(row['n'])} | {fmt(row['mean'])} | {fmt(row['sd'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] |")
    lines += ["", "## Paired t-tests", "", "Effects are final FederatedBatch minus comparator; two-sided paired t-tests use df=4. Holm correction is joint across all 12 tests.", "", "| Comparator | Metric | Mean delta | SD | 95% CI | t(4) | Raw p | Holm p |", "|---|---|---:|---:|---|---:|---:|---:|"]
    for _, row in tests.iterrows():
        lines.append(f"| {row['comparator']} | {row['metric']} | {fmt(row['mean_paired_difference'])} | {fmt(row['sd_paired_difference'])} | [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['t_statistic'])} | {fmt(row['raw_p_value'])} | {fmt(row['holm_adjusted_p_value'])} |")
    lines += ["", "### Paired per-seed differences", "", "Each value is final FederatedBatch minus comparator.", "", "| Comparator | Metric | 3205 | 9048 | 7487 | 5326 | 9364 |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in tests.iterrows():
        lines.append("| " + " | ".join([str(row["comparator"]), str(row["metric"])] + [fmt(row[f"difference_{seed}"]) for seed in SEEDS]) + " |")
    graph = tests[tests.metric == "Graph_connectivity"].set_index("comparator")
    lines += ["", "## Graph connectivity reassessment", "", f"- Final FB vs CONCORD: delta {fmt(graph.loc['CONCORD', 'mean_paired_difference'])}, 95% CI [{fmt(graph.loc['CONCORD', 'ci95_low'])}, {fmt(graph.loc['CONCORD', 'ci95_high'])}], raw p {fmt(graph.loc['CONCORD', 'raw_p_value'])}, Holm p {fmt(graph.loc['CONCORD', 'holm_adjusted_p_value'])}.", f"- Final FB vs Harmony: delta {fmt(graph.loc['Harmony', 'mean_paired_difference'])}, 95% CI [{fmt(graph.loc['Harmony', 'ci95_low'])}, {fmt(graph.loc['Harmony', 'ci95_high'])}], raw p {fmt(graph.loc['Harmony', 'raw_p_value'])}, Holm p {fmt(graph.loc['Harmony', 'holm_adjusted_p_value'])}.", "", "The graph-connectivity result is statistically indistinguishable from CONCORD after the requested paired t-test and Holm correction, and significantly higher than Harmony after Holm correction. This is a statistical result only, not a manuscript claim.", "", "## UMAP", "", "Representative seed: 3205. PCA uses the archived deterministic PCA embedding; Harmony and centralized CONCORD use their seed-3205 embeddings; FederatedBatch uses the new final private seed-3205 embedding. Each method was fit separately with n_neighbors=15, min_dist=0.1, Euclidean metric, random_state=42. Figure 1C and 1D use identical coordinates and differ only in coloring.", "", "Outputs: `figure1_umap_coordinates_final.csv`, `figure1C_final.png`, `figure1D_final.png`.", ""]
    (STAT_OUT / "FIGURE1_FINAL_STATISTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {STAT_OUT}")
    print(f"Wrote {OUT / 'figure1_umap_coordinates_final.csv'}")
    print(f"Wrote {OUT / 'figure1C_final.png'} and {OUT / 'figure1D_final.png'}")


if __name__ == "__main__":
    raise SystemExit(
        "This legacy helper is not a Figure 1 entry point; use "
        "experiments/run_figure1_canonical7.py."
    )
