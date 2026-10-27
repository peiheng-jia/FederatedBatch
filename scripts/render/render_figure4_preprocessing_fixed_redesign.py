#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "final_summaries" / "figure4"
OUT = ROOT / "figures" / "redraw" / "figure4"

METHODS = [
    "Centralized CONCORD",
    "FedscGen (without correction)",
    "FedscGen (correction workflow)",
    "FederatedBatch (zero-shot)",
]
COLORS = {
    "Centralized CONCORD": "#452A3D",
    "FedscGen (without correction)": "#44757A",
    "FedscGen (correction workflow)": "#96AAA5",
    "FederatedBatch (zero-shot)": "#D44C3C",
}
LEGEND_LABELS = {
    "Centralized CONCORD": "Centralized CONCORD",
    "FedscGen (without correction)": "FedscGen without correction",
    "FedscGen (correction workflow)": "FedscGen correction workflow",
    "FederatedBatch (zero-shot)": "FederatedBatch zero-shot",
}
HEAT_LABELS = {
    "Centralized CONCORD": "CONCORD",
    "FedscGen (without correction)": "FedscGen\n(no correction)",
    "FedscGen (correction workflow)": "FedscGen\n(correction)",
    "FederatedBatch (zero-shot)": "FederatedBatch\n(zero-shot)",
}


def style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    )


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.04)


def panel_b():
    summary = pd.read_csv(SRC / "figure4_fixed_summary.csv")
    metrics = [
        ("ARI", "ARI"),
        ("NMI", "NMI"),
        ("ASW_celltype", "ASW (cell type)"),
        ("Graph_connectivity", "Graph connectivity"),
    ]
    means = summary.pivot(index="method", columns="metric", values="mean").loc[METHODS]
    sds = summary.pivot(index="method", columns="metric", values="sd").loc[METHODS]

    fig, ax = plt.subplots(figsize=(2.72, 4.78))
    y = np.arange(len(metrics))
    height = 0.145
    for i, method in enumerate(METHODS):
        pos = y + (i - 1.5) * height
        vals = means.loc[method, [key for key, _ in metrics]].to_numpy(float)
        errs = sds.loc[method, [key for key, _ in metrics]].to_numpy(float)
        ax.barh(
            pos,
            vals,
            height=height,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            label=LEGEND_LABELS[method],
            zorder=2,
        )
        ax.errorbar(
            vals,
            pos,
            xerr=errs,
            fmt="none",
            ecolor="#303030",
            elinewidth=0.65,
            capsize=1.7,
            capthick=0.65,
            zorder=3,
        )
    ax.set_yticks(y, [label for _, label in metrics], fontsize=6.1)
    ax.invert_yaxis()
    ax.set_xlabel("Score", fontsize=6.3, labelpad=2)
    ax.set_xlim(0, 0.9)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.5, length=2, labelsize=5.7, pad=2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1,
        frameon=False,
        fontsize=4.8,
        columnspacing=0.5,
        handlelength=0.9,
        labelspacing=0.35,
        handletextpad=0.3,
    )
    fig.text(0.025, 0.982, "B", fontsize=10, fontweight="bold", va="top")
    fig.text(0.54, 0.982, "Held-out performance", fontsize=7.2, fontweight="bold", ha="center", va="top")
    fig.subplots_adjust(left=0.39, right=0.98, top=0.88, bottom=0.25)
    save(fig, "figure4B_heldout_performance")
    plt.close(fig)


def panel_c():
    raw = pd.read_csv(SRC / "figure4_fixed_per_seed_per_heldout_all_methods.csv")
    batches = list(dict.fromkeys(raw["held_out_batch"].astype(str)))
    rows = []
    for metric in ("ARI", "NMI"):
        grouped = raw.groupby(["held_out_batch", "method"], sort=False)[metric].mean().reset_index()
        grouped["metric"] = metric
        rows.append(grouped)
    data = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.72), gridspec_kw={"wspace": 0.22})
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "quiet", ["#F5F5F2", "#EADDD8", "#E3B3A7", "#D78371", "#C45D4D", "#8F3F38"]
    )
    for ax, metric, title in zip(axes, ("ARI", "NMI"), ("ARI", "NMI")):
        p = (
            data[data.metric == metric]
            .pivot(index="held_out_batch", columns="method", values=metric)
            .reindex(index=batches, columns=METHODS)
        )
        vals = p.to_numpy(float)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        ax.pcolormesh(
            np.arange(5),
            np.arange(len(batches) + 1),
            vals,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            shading="flat",
            edgecolors="white",
            linewidth=0.35,
        )
        ax.set_xlim(0, 4)
        ax.set_ylim(len(batches), 0)
        for r in range(len(batches)):
            for c in range(len(METHODS)):
                value = vals[r, c]
                text_color = "white" if (value - lo) / (hi - lo) > 0.62 else "#222222"
                ax.text(c + 0.5, r + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=5.7, fontweight="bold", color=text_color)
        ax.set_title(title, fontweight="bold", fontsize=8, pad=5)
        ax.set_xticks(np.arange(4) + 0.5, [HEAT_LABELS[m] for m in METHODS], fontsize=5.2)
        ax.set_yticks(np.arange(len(batches)) + 0.5, batches if ax is axes[0] else [], fontsize=5.8)
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.015, 0.975, "C", fontsize=12, fontweight="bold", va="top")
    fig.text(0.5, 0.975, "Performance across held-out batches", fontsize=9, fontweight="bold", ha="center", va="top")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.85, bottom=0.17)
    save(fig, "figure4C_heldout_heatmaps")
    plt.close(fig)


def preview():
    images = [Image.open(OUT / "figure4B_heldout_performance.png").convert("RGB"), Image.open(OUT / "figure4C_heldout_heatmaps.png").convert("RGB")]
    width = max(im.width for im in images)
    scaled = [im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS) for im in images]
    canvas = Image.new("RGB", (width, sum(im.height for im in scaled)), "white")
    y = 0
    for im in scaled:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(OUT / "figure4_BC_layout_preview.png", dpi=(300, 300))


def main():
    global SRC, OUT
    parser = argparse.ArgumentParser(description='Render the Figure 4 B/C panels in the approved historical style.')
    parser.add_argument("--input", type=Path, default=SRC)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    SRC = args.input.resolve()
    OUT = args.output.resolve()
    style()
    panel_b()
    panel_c()
    preview()
    (OUT / "FIGURE4_FIXED_REDESIGN_REPORT.md").write_text(
        "# Figure 4 corrected-data redesign\n\n"
        "Panel A remains author-managed. Panels B and C reproduce the previously approved Figure 4 visual style: grouped horizontal bars for held-out performance and annotated ARI/NMI heatmaps across held-out batches.\n\n"
        "The plotting inputs are the checkpoint-only preprocessing-fixed results in `results/figure4_preprocessing_fixed/`. No model was retrained and no metric was recomputed by this renderer.\n\n"
        "Outputs are vector PDF/SVG plus 600-dpi PNG previews.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
