#!/usr/bin/env python3
"""Render Figure 3 from saved metrics and staged UMAP coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SEEDS = (218, 394, 542, 639, 788, 814, 880)
REPRESENTATIVE_SEED = 218
DATASETS = ("pancreas", "immune", "lung")
DATASET_LABELS = {"pancreas": "Pancreas", "immune": "Immune", "lung": "Lung"}
UMAP_METHODS = ("Centralized CONCORD", "FederatedBatch", "FedscGen", "PCA")
ALL_METHODS = ("PCA", "Harmony", "Centralized CONCORD", "FedscGen", "FederatedBatch")
METHOD_LABELS = {
    "PCA": "PCA",
    "Harmony": "Harmony",
    "Centralized CONCORD": "CONCORD",
    "FedscGen": "FedscGen",
    "FederatedBatch": "FederatedBatch",
}
METHOD_COLORS = {
    "PCA": "#B5B2A0",
    "Harmony": "#4F7F83",
    "Centralized CONCORD": "#4B3042",
    "FedscGen": "#8B78A5",
    "FederatedBatch": "#D84C3E",
}
METRICS = ("ARI", "NMI", "ASW_batch", "ASW_celltype", "Graph_connectivity", "iLISI")
METRIC_LABELS = {
    "ARI": "ARI",
    "NMI": "NMI",
    "ASW_batch": "Raw ASW (batch)",
    "ASW_celltype": "ASW (cell type)",
    "Graph_connectivity": "Graph connectivity",
    "iLISI": "iLISI",
}
T_CRITICAL_DF6 = 2.4469118511449692

PANCREAS_COLORS = {
    "acinar": "#DF5B4D",
    "activated_stellate": "#E98463",
    "alpha": "#F19A75",
    "beta": "#F2D8B4",
    "delta": "#B8B6A0",
    "ductal": "#98AAA4",
    "endothelial": "#4F7F83",
    "epsilon": "#689095",
    "gamma": "#86A4A5",
    "macrophage": "#4A3540",
    "mast": "#634A55",
    "quiescent_stellate": "#7A606A",
    "schwann": "#9A7682",
    "t_cell": "#B58D97",
}
QUALITATIVE_COLORS = (
    "#D84C3E", "#E77757", "#F0A06F", "#F1C99A", "#B7B5A2", "#8DA5A0",
    "#4F7F83", "#5F9297", "#79A5A7", "#3F3340", "#5B4755", "#735C6A",
    "#8E7080", "#A98A99", "#7A6A9D", "#9A83B6", "#C095A8", "#6E89A8",
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def palette_for(values: pd.Series, dataset: str) -> dict[str, str]:
    labels = sorted(values.astype(str).unique())
    if dataset == "pancreas" and set(labels).issubset(PANCREAS_COLORS):
        return {label: PANCREAS_COLORS[label] for label in labels}
    return {label: QUALITATIVE_COLORS[index % len(QUALITATIVE_COLORS)] for index, label in enumerate(labels)}


def load_coordinates(coordinate_dir: Path) -> dict[str, pd.DataFrame]:
    """Load staged Figure 3 coordinates; this renderer never fits UMAP."""
    result = {}
    for dataset in DATASETS:
        path = coordinate_dir / f"{dataset}_seed218_umap_coordinates.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        expected = {f"UMAP1_{method}" for method in UMAP_METHODS} | {f"UMAP2_{method}" for method in UMAP_METHODS}
        if not expected.issubset(frame.columns):
            raise ValueError(f"incomplete coordinate table: {path}")
        result[dataset] = frame
    return result


def style_umap_axis(axis: plt.Axes, *, xlabel: bool, ylabel: bool) -> None:
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#4B4B4B")
    axis.set_box_aspect(1)
    axis.tick_params(width=0.5, length=2.0, pad=1.5, labelsize=5.4)
    if xlabel:
        axis.set_xlabel("UMAP 1", labelpad=2)
    if ylabel:
        axis.set_ylabel("UMAP 2", labelpad=2)


def draw_umap_rows(fig: plt.Figure, outer_spec, coordinates: dict[str, pd.DataFrame]) -> list[plt.Axes]:
    """Draw the reference layout: two aligned rows per dataset and four method columns."""
    grid = outer_spec.subgridspec(
        6, 5, width_ratios=(1, 1, 1, 1, 0.78), wspace=0.25, hspace=0.34
    )
    axes: list[plt.Axes] = []
    panel_letters = ("A", "B", "C", "D", "E", "F")
    for dataset_index, dataset in enumerate(DATASETS):
        frame = coordinates[dataset]
        for within_index, color_key in enumerate(("celltype", "batch")):
            row_index = dataset_index * 2 + within_index
            palette = palette_for(frame[color_key], dataset)
            order = np.random.default_rng(7300 + row_index).permutation(len(frame))
            for column_index, method in enumerate(UMAP_METHODS):
                axis = fig.add_subplot(grid[row_index, column_index])
                axes.append(axis)
                colors = frame[color_key].astype(str).map(palette).to_numpy()[order]
                axis.scatter(
                    frame[f"UMAP1_{method}"].to_numpy()[order],
                    frame[f"UMAP2_{method}"].to_numpy()[order],
                    c=colors,
                    s=1.05 if dataset == "pancreas" else 0.62,
                    alpha=0.76,
                    linewidths=0,
                    rasterized=True,
                )
                style_umap_axis(axis, xlabel=row_index == 5, ylabel=column_index == 0)
                axis.set_title(METHOD_LABELS[method], fontsize=7.5, fontweight="bold", pad=3)
                if column_index == 0 and within_index == 0:
                    axis.text(
                        -0.28, 1.14, DATASET_LABELS[dataset], transform=axis.transAxes,
                        ha="left", va="bottom", fontsize=7.8, fontweight="bold",
                    )
                if column_index == 0:
                    axis.text(
                        -0.31, 1.02, panel_letters[row_index], transform=axis.transAxes,
                        ha="left", va="bottom", fontsize=11, fontweight="bold",
                    )
            legend_axis = fig.add_subplot(grid[row_index, 4])
            legend_axis.axis("off")
            labels = list(palette)
            handles = [
                Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette[label],
                       markeredgecolor="none", markersize=3.5)
                for label in labels
            ]
            legend_axis.legend(
                handles,
                [label.replace("_", " ") for label in labels],
                title="Cell type" if color_key == "celltype" else "Batch",
                title_fontsize=6.2,
                frameon=False,
                loc="center left",
                ncol=1,
                handletextpad=0.35,
                labelspacing=0.08,
                borderaxespad=0,
                fontsize=4.4 if len(labels) > 12 else 4.8,
            )
    return axes


def draw_metrics_panel(fig: plt.Figure, outer_spec, metrics: pd.DataFrame, panel: str) -> list[plt.Axes]:
    """One compact matrix with restrained within-row relative bubble areas."""
    axis = fig.add_subplot(outer_spec)
    method_count = len(ALL_METHODS)
    group_gap = 0.75
    x_positions: dict[tuple[str, str], float] = {}
    for dataset_index, dataset in enumerate(DATASETS):
        base = dataset_index * (method_count + group_gap)
        for method_index, method in enumerate(ALL_METHODS):
            x_positions[(dataset, method)] = base + method_index

    y_positions = {metric: len(METRICS) - 1 - index for index, metric in enumerate(METRICS)}
    for metric in METRICS:
        y = y_positions[metric]
        for dataset in DATASETS:
            method_values: dict[str, np.ndarray] = {}
            means: dict[str, float] = {}
            for method in ALL_METHODS:
                values = (
                    metrics.loc[(metrics.dataset == dataset) & (metrics.method == method)]
                    .set_index("training_seed").loc[list(SEEDS), metric].to_numpy(float)
                )
                method_values[method] = values
                means[method] = float(values.mean())
            if metric == "ASW_batch":
                desirability = {method: -abs(value) for method, value in means.items()}
            else:
                desirability = means
            low = min(desirability.values())
            high = max(desirability.values())
            for method in ALL_METHODS:
                values = method_values[method]
                mean = means[method]
                sd = float(values.std(ddof=1))
                x = x_positions[(dataset, method)]
                normalized = 0.5 if high == low else (desirability[method] - low) / (high - low)
                # Area and saturation use the same within-comparison ranking; hue remains method-specific.
                bubble_area = 150.0 + 240.0 * normalized
                base_rgb = np.asarray(mpl.colors.to_rgb(METHOD_COLORS[method]))
                white_mix = 0.42 * (1.0 - normalized)
                fill_rgb = base_rgb * (1.0 - white_mix) + np.ones(3) * white_mix
                axis.scatter(x, y, s=bubble_area, color=fill_rgb, alpha=0.97,
                             edgecolor=METHOD_COLORS[method], linewidth=0.65, zorder=2)
                digits = 2 if metric in {"ARI", "NMI", "iLISI"} else 3
                luminance = 0.2126 * fill_rgb[0] + 0.7152 * fill_rgb[1] + 0.0722 * fill_rgb[2]
                text_color = "#222222" if luminance > 0.60 else "white"
                axis.text(x, y, f"{mean:.{digits}f}", ha="center", va="center",
                          fontsize=5.2, fontweight="bold", color=text_color, zorder=3)
                secondary = f"±{sd:.3f}"
                axis.text(x, y - 0.36, secondary, ha="center", va="center",
                          fontsize=4.0, color="#555555", zorder=3)

    ticks = [x_positions[(dataset, method)] for dataset in DATASETS for method in ALL_METHODS]
    display_method = {**METHOD_LABELS, "FederatedBatch": "Federated\nBatch"}
    labels = [display_method[method] for dataset in DATASETS for method in ALL_METHODS]
    axis.set_xticks(ticks, labels, rotation=0, ha="center")
    row_labels = {
        "ARI": "ARI ↑",
        "NMI": "NMI ↑",
        "ASW_batch": "Raw ASW (batch)",
        "ASW_celltype": "ASW (cell type) ↑",
        "Graph_connectivity": "Graph connectivity ↑",
        "iLISI": "iLISI ↑",
    }
    axis.set_yticks([y_positions[m] for m in METRICS], [row_labels[m] for m in METRICS])
    axis.tick_params(axis="x", length=0, pad=5, labelsize=4.9)
    axis.tick_params(axis="y", length=0, pad=6, labelsize=6.4)
    axis.set_xlim(-0.75, max(ticks) + 0.75)
    axis.set_ylim(-0.65, len(METRICS) - 0.15)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(False)

    top = len(METRICS) - 0.03
    for dataset in DATASETS:
        center = np.mean([x_positions[(dataset, method)] for method in ALL_METHODS])
        left = x_positions[(dataset, ALL_METHODS[0])] - 0.42
        right = x_positions[(dataset, ALL_METHODS[-1])] + 0.42
        axis.plot([left, right], [top, top], color="#2B2B2B", linewidth=0.8, clip_on=False)
        axis.text(center, top + 0.12, DATASET_LABELS[dataset], ha="center", va="bottom",
                  fontsize=7.5, fontweight="bold")
    axis.text(-0.09, 1.07, panel, transform=axis.transAxes, fontsize=11,
              fontweight="bold", ha="left", va="bottom")
    return [axis]


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".svg"), dpi=600, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.035)


def render_separate(output: Path, coordinates: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 10.6))
    spec = fig.add_gridspec(1, 1, left=0.095, right=0.985, top=0.985, bottom=0.055)[0, 0]
    draw_umap_rows(fig, spec, coordinates)
    save_figure(fig, output / "figure3A_F_aligned_umaps")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.0))
    spec = fig.add_gridspec(1, 1, left=0.14, right=0.985, top=0.88, bottom=0.19)[0, 0]
    draw_metrics_panel(fig, spec, metrics, "G")
    save_figure(fig, output / "figure3G_metric_matrix")
    plt.close(fig)


def render_combined(output: Path, coordinates: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 14.7))
    grid = fig.add_gridspec(
        2, 1,
        height_ratios=(2.65, 1.0),
        left=0.095,
        right=0.985,
        top=0.985,
        bottom=0.055,
        hspace=0.12,
    )
    draw_umap_rows(fig, grid[0, 0], coordinates)
    draw_metrics_panel(fig, grid[1, 0], metrics, "G")
    save_figure(fig, output / "figure3_publication_redesign_v8_pca_fixed")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coordinates",
        type=Path,
        help="Coordinate directory (default: results/plot_inputs/figure3)",
    )
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    configure_style()

    metrics = pd.read_csv(source / "per_seed_per_dataset_metrics.csv")
    expected = {(dataset, method, seed) for dataset in DATASETS for method in ALL_METHODS for seed in SEEDS}
    observed = set(zip(metrics.dataset, metrics.method, metrics.training_seed.astype(int)))
    if observed != expected:
        raise ValueError(f"canonical metric table mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}")
    coordinate_dir = args.coordinates.resolve() if args.coordinates else root / "results" / "plot_inputs" / "figure3"
    coordinates = load_coordinates(coordinate_dir)
    render_separate(output, coordinates, metrics)
    render_combined(output, coordinates, metrics)

    provenance = {
        "source": str(source),
        "coordinate_source": str(coordinate_dir),
        "canonical_training_seeds": list(SEEDS),
        "representative_training_seed": REPRESENTATIVE_SEED,
        "umap": {"n_neighbors": 15, "min_dist": 0.5, "metric": "euclidean", "random_state": 0},
        "umap_methods": list(UMAP_METHODS),
        "quantitative_methods": list(ALL_METHODS),
        "metrics_recomputed": False,
        "models_retrained": False,
        "same_coordinates_for_celltype_and_batch": True,
        "dense_scatter_layers_rasterized": True,
        "text_axes_titles_legends_vector": True,
    }
    (output / "figure3_visualization_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Publication Figure 3 written to {output}", flush=True)


if __name__ == "__main__":
    main()
