#!/usr/bin/env python3
"""Render the final result-only Figure 2 from locked canonical-seven outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SEEDS = (218, 394, 542, 639, 788, 814, 880)
METHODS = ("NoQueue", "Remote A/B (sigma=0)", "FederatedBatch")
TRAJECTORY_METHODS = METHODS[:2]
COLORS = {
    "NoQueue": "#8B8B8B",
    "Remote A/B (sigma=0)": "#E49A67",
    "FederatedBatch": "#D64F43",
}
MARKERS = {
    "NoQueue": "o",
    "Remote A/B (sigma=0)": "s",
    "FederatedBatch": "D",
}
LONG_LABELS = {
    "NoQueue": "NoQueue",
    "Remote A/B (sigma=0)": "Remote A/B\n($\\sigma=0$)",
    "FederatedBatch": "FederatedBatch",
}
SHORT_LABELS = {
    "NoQueue": "NoQueue",
    "Remote A/B (sigma=0)": "A/B\n($\\sigma=0$)",
    "FederatedBatch": "FB",
}
METRIC_LABELS = {
    "ARI": "ARI",
    "NMI": "NMI",
    "ASW_batch": "Raw ASW (batch)",
    "ASW_cell": "ASW (cell type)",
    "graph_connectivity": "Graph connectivity",
    "iLISI": "iLISI",
}
HOLM_FB_VS_NOQUEUE = {"ARI": 0.28995531785508455, "NMI": 0.014835526950737258}
T_CRITICAL_DF6 = 2.4469118511449692


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.3,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(True)
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.spines["right"].set_linewidth(0.65)
    axis.tick_params(direction="out", width=0.6, length=2.7, pad=2.0)
    axis.grid(False)


def facet_strip(axis: plt.Axes, label: str) -> None:
    strip = Rectangle(
        (0, 1.0),
        1,
        0.12,
        transform=axis.transAxes,
        facecolor="#E7E7E7",
        edgecolor="#4B4B4B",
        linewidth=0.65,
        clip_on=False,
        zorder=8,
    )
    axis.add_patch(strip)
    axis.text(
        0.5,
        1.06,
        label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#252525",
        zorder=9,
    )


def validate_inputs(per_seed: pd.DataFrame, trajectory: pd.DataFrame, summary: pd.DataFrame) -> None:
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    observed = set(zip(per_seed["seed"].astype(int), per_seed["method"].astype(str)))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"final metric rows mismatch; missing={missing}, extra={extra}")
    for method in TRAJECTORY_METHODS:
        subset = trajectory.loc[trajectory["method"] == method]
        if set(subset["seed"].astype(int)) != set(SEEDS):
            raise ValueError(f"trajectory seed mismatch for {method}")
        for seed, frame in subset.groupby("seed"):
            if tuple(sorted(frame["round"].astype(int))) != tuple(range(1, 21)):
                raise ValueError(f"trajectory rounds mismatch for {method}, seed {seed}")
    required_summary = {
        (method, round_number, metric)
        for method in TRAJECTORY_METHODS
        for round_number in range(1, 21)
        for metric in ("ARI", "NMI")
    }
    observed_summary = set(
        zip(summary["method"], summary["round"].astype(int), summary["metric"])
    )
    if not required_summary.issubset(observed_summary):
        raise ValueError("trajectory summary is incomplete")


def paired_limits(values: np.ndarray, metric: str, annotation: bool) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = max(high - low, 1e-6)
    lower_pad = 0.10 * span
    upper_pad = (0.26 if annotation else 0.14) * span
    if metric == "NMI":
        lower_pad = max(lower_pad, 0.003)
        upper_pad = max(upper_pad, 0.007 if annotation else 0.003)
    return low - lower_pad, high + upper_pad


def draw_significance_bracket(axis: plt.Axes, x1: float, x2: float, text: str) -> None:
    ymin, ymax = axis.get_ylim()
    height = ymax - ymin
    y = ymax - 0.075 * height
    tick = 0.025 * height
    axis.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="#3F3F3F", linewidth=0.65, clip_on=False)
    axis.text((x1 + x2) / 2, y + 0.012 * height, text, ha="center", va="bottom", fontsize=5.6, color="#333333")


def draw_paired(
    axis: plt.Axes,
    per_seed: pd.DataFrame,
    metric: str,
    *,
    compact: bool,
    annotate: bool,
    connect_pairs: bool = True,
) -> None:
    indexed = per_seed.pivot(index="seed", columns="method", values=metric).loc[list(SEEDS)]
    x = np.arange(len(METHODS), dtype=float)
    jitter = np.linspace(-0.095, 0.095, len(SEEDS)) if not connect_pairs else np.zeros(len(SEEDS))
    for seed_index, (_, row) in enumerate(indexed.iterrows()):
        values = row[list(METHODS)].to_numpy(float)
        if connect_pairs:
            axis.plot(x, values, color="#B8B8B8", linewidth=0.55, alpha=0.68, zorder=1)
        for position, method, value in zip(x, METHODS, values):
            axis.scatter(
                position + jitter[seed_index],
                value,
                s=16 if compact else 19,
                marker=MARKERS[method],
                facecolor=COLORS[method],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
    for position, method in zip(x, METHODS):
        values = indexed[method].to_numpy(float)
        mean = float(values.mean())
        ci = T_CRITICAL_DF6 * float(values.std(ddof=1)) / np.sqrt(len(values))
        axis.errorbar(
            position,
            mean,
            yerr=ci,
            fmt="D",
            markersize=4.6 if compact else 5.1,
            color="#202020",
            markerfacecolor=COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.4,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.8,
            zorder=5,
        )
    axis.set_xticks(x)
    labels = SHORT_LABELS if compact else LONG_LABELS
    if compact:
        axis.set_xticklabels([labels[method] for method in METHODS], rotation=0, ha="center", fontsize=5.9)
    else:
        axis.set_xticklabels([labels[method] for method in METHODS], rotation=22, ha="right", fontsize=5.7)
    axis.set_ylabel(METRIC_LABELS[metric])
    values = indexed[list(METHODS)].to_numpy(float)
    axis.set_ylim(*paired_limits(values, metric, annotate))
    clean_axis(axis)
    facet_strip(axis, METRIC_LABELS[metric])
    if annotate:
        p_value = HOLM_FB_VS_NOQUEUE[metric]
        p_text = f"Holm p={p_value:.3f}"
        draw_significance_bracket(axis, 0, 2, p_text)


def draw_trajectory(axis: plt.Axes, raw: pd.DataFrame, summary: pd.DataFrame, metric: str) -> None:
    for method in TRAJECTORY_METHODS:
        subset = raw.loc[raw["method"] == method].sort_values(["seed", "round"])
        for _, seed_frame in subset.groupby("seed", sort=True):
            axis.plot(
                seed_frame["round"],
                seed_frame[metric],
                color=COLORS[method],
                linewidth=0.48,
                alpha=0.16,
                zorder=1,
            )
        curve = summary.loc[
            (summary["method"] == method) & (summary["metric"] == metric)
        ].sort_values("round")
        rounds = curve["round"].to_numpy(float)
        means = curve["mean"].to_numpy(float)
        low = curve["ci95_low"].to_numpy(float)
        high = curve["ci95_high"].to_numpy(float)
        axis.fill_between(rounds, low, high, color=COLORS[method], alpha=0.10, linewidth=0, zorder=0)
        axis.plot(
            rounds,
            means,
            color=COLORS[method],
            linewidth=1.45,
            label=LONG_LABELS[method].replace("\n", " "),
            zorder=3,
        )
    axis.axvline(4, color="#8A8A8A", linestyle=(0, (3, 2)), linewidth=0.6, zorder=2)
    axis.set_xlim(1, 20)
    axis.set_xticks([1, 4, 8, 12, 16, 20])
    axis.set_xlabel("Communication round")
    axis.set_ylabel(METRIC_LABELS[metric])
    clean_axis(axis)
    facet_strip(axis, METRIC_LABELS[metric])


def add_group_title(fig: plt.Figure, axes: list[plt.Axes], label: str, panel: str) -> None:
    boxes = [axis.get_position() for axis in axes]
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    top = max(box.y1 for box in boxes)
    fig.text((left + right) / 2, top + 0.055, label, ha="center", va="bottom", fontsize=8.7, fontweight="bold")
    fig.text(left - 0.042, top + 0.067, panel, ha="left", va="bottom", fontsize=11.2, fontweight="bold")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)


def render_combined(output: Path, per_seed: pd.DataFrame, raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.20, 5.45), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=(1.0, 0.94),
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.105,
        wspace=0.58,
        hspace=0.72,
    )
    axes_a = [fig.add_subplot(grid[0, index]) for index in range(2)]
    axes_b = [fig.add_subplot(grid[0, index]) for index in range(2, 4)]
    axes_c = [fig.add_subplot(grid[1, index]) for index in range(4)]

    for axis, metric in zip(axes_a, ("ARI", "NMI")):
        draw_paired(axis, per_seed, metric, compact=False, annotate=True, connect_pairs=True)
    for axis, metric in zip(axes_b, ("ARI", "NMI")):
        draw_trajectory(axis, raw, summary, metric)
    for axis, metric in zip(axes_c, ("ASW_batch", "ASW_cell", "graph_connectivity", "iLISI")):
        draw_paired(axis, per_seed, metric, compact=True, annotate=False, connect_pairs=False)

    axes_b[1].legend(
        frameon=False,
        loc="lower right",
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.25,
    )
    add_group_title(fig, axes_a, "Final-round performance", "A")
    add_group_title(fig, axes_b, "Performance over communication rounds", "B")
    add_group_title(fig, axes_c, "Other integration metrics", "C")
    save_figure(fig, output / "figure2_result_only_final")
    plt.close(fig)


def render_panel_a(output: Path, per_seed: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.85))
    for axis, metric in zip(axes, ("ARI", "NMI")):
        draw_paired(axis, per_seed, metric, compact=False, annotate=True, connect_pairs=True)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.76, bottom=0.19, wspace=0.34)
    add_group_title(fig, list(axes), "Final-round performance", "A")
    save_figure(fig, output / "figure2A_final_performance")
    plt.close(fig)


def render_panel_b(output: Path, raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.85))
    for axis, metric in zip(axes, ("ARI", "NMI")):
        draw_trajectory(axis, raw, summary, metric)
    axes[1].legend(frameon=False, loc="lower right", handlelength=1.6, labelspacing=0.25)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.76, bottom=0.19, wspace=0.34)
    add_group_title(fig, list(axes), "Performance over communication rounds", "B")
    save_figure(fig, output / "figure2B_round_trajectories")
    plt.close(fig)


def render_panel_c(output: Path, per_seed: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.65))
    for axis, metric in zip(axes, ("ASW_batch", "ASW_cell", "graph_connectivity", "iLISI")):
        draw_paired(axis, per_seed, metric, compact=True, annotate=False, connect_pairs=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.73, bottom=0.21, wspace=0.58)
    add_group_title(fig, list(axes), "Other integration metrics", "C")
    save_figure(fig, output / "figure2C_other_metrics")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()

    per_seed = pd.read_csv(source / "per_seed_metrics_final.csv")
    trajectory = pd.read_csv(source / "round_trajectory.csv")
    summary = pd.read_csv(source / "round_trajectory_summary.csv")
    validate_inputs(per_seed, trajectory, summary)

    render_combined(output, per_seed, trajectory, summary)
    render_panel_a(output, per_seed)
    render_panel_b(output, trajectory, summary)
    render_panel_c(output, per_seed)

    print(f"Result-only Figure 2 written to {output}")


if __name__ == "__main__":
    main()
