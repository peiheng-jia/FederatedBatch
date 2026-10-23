#!/usr/bin/env python3
"""Create Illustrator-friendly hybrid Figure 1C/D outputs.

Only the dense UMAP point collections are rasterized. Text, axes, borders,
panel labels, titles, and legends remain vector/editable in PDF and SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results" / "final_figure1_canonical7" / "reference_style_cd"
SOURCE = SOURCE_DIR / "figure1_umap_coordinates_reference_style.csv"
OUT = SOURCE_DIR / "hybrid_editable"

METHODS = ("PCA", "Harmony", "CONCORD", "FederatedBatch")

BATCH_COLORS = {
    "celseq": "#B7B5A0",
    "celseq2": "#44757A",
    "fluidigmc1": "#452A3D",
    "inDrop1": "#D44C3C",
    "inDrop2": "#DD6C4C",
    "inDrop3": "#E5855D",
    "inDrop4": "#EED5B7",
    "smarter": "#966478",
    "smartseq2": "#648C91",
}

CELLTYPE_COLORS = {
    "acinar": "#D44C3C",
    "activated_stellate": "#DD6C4C",
    "alpha": "#E5855D",
    "beta": "#EED5B7",
    "delta": "#B7B5A0",
    "ductal": "#96AAA5",
    "endothelial": "#44757A",
    "epsilon": "#508287",
    "gamma": "#648C91",
    "macrophage": "#452A3D",
    "mast": "#5A3C50",
    "quiescent_stellate": "#785064",
    "schwann": "#966478",
    "t_cell": "#B48296",
}


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def add_umap_axis(ax, frame: pd.DataFrame, method: str, field: str, palette: dict[str, str]) -> None:
    labels = frame[field].astype(str).to_numpy()
    missing = sorted(set(labels) - set(palette))
    if missing:
        raise RuntimeError(f"Missing colors for {field}: {missing}")
    colors = np.asarray([palette[label] for label in labels], dtype=object)

    # One rasterized collection replaces 16,382 editable circle objects.
    ax.scatter(
        frame[f"UMAP1_{method}"].to_numpy(),
        frame[f"UMAP2_{method}"].to_numpy(),
        s=1.35,
        c=colors,
        alpha=0.55,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_title(method, fontsize=10.5, fontweight="bold", pad=5)
    ax.set_xlabel("UMAP 1", fontsize=8.5, labelpad=2)
    ax.set_ylabel("UMAP 2", fontsize=8.5, labelpad=2)
    ax.tick_params(axis="both", labelsize=7.5, length=3)
    # Match the approved R figure: every plotting box is the same square size.
    # Do not use equal data-unit scaling, which makes box sizes differ when
    # methods have different x/y coordinate ranges.
    ax.set_box_aspect(1)
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.55)


def add_legend(ax, field: str, palette: dict[str, str], title: str) -> None:
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.8,
            markerfacecolor=color,
            markeredgecolor="none",
            alpha=0.85,
            label=label,
        )
        for label, color in palette.items()
    ]
    label_spacing = 0.42 if field == "batch" else 0.26
    ax.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        title=title,
        title_fontsize=9.5,
        fontsize=7.5,
        handletextpad=0.45,
        labelspacing=label_spacing,
        borderaxespad=0,
    )


def make_panel(
    frame: pd.DataFrame,
    field: str,
    palette: dict[str, str],
    legend_title: str,
    panel_label: str,
):
    fig = plt.figure(figsize=(10, 2.5), facecolor="white")
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1, 1, 1, 1, 0.82],
        wspace=0.34,
    )
    for index, method in enumerate(METHODS):
        add_umap_axis(fig.add_subplot(grid[0, index]), frame, method, field, palette)
    add_legend(fig.add_subplot(grid[0, 4]), field, palette, legend_title)
    fig.text(0.002, 0.985, panel_label, fontsize=13, fontweight="bold", va="top", ha="left")
    fig.subplots_adjust(left=0.047, right=0.995, bottom=0.19, top=0.86)
    return fig


def export(fig, stem: str) -> None:
    # dpi controls only rasterized artists in PDF/SVG; vector elements are unaffected.
    fig.savefig(OUT / f"{stem}.pdf", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    plt.close(fig)


def record_panel_geometry(fig, panel: str) -> list[dict[str, float | str]]:
    fig.canvas.draw()
    figure_width, figure_height = fig.get_size_inches()
    rows = []
    for method, ax in zip(METHODS, fig.axes[:4]):
        box = ax.get_position()
        physical_width = box.width * figure_width
        physical_height = box.height * figure_height
        rows.append(
            {
                "panel": panel,
                "method": method,
                "width_inches": physical_width,
                "height_inches": physical_height,
                "height_minus_width_inches": physical_height - physical_width,
            }
        )
    return rows


def make_combined(frame: pd.DataFrame):
    fig = plt.figure(figsize=(10, 5), facecolor="white")
    outer = fig.add_gridspec(2, 1, hspace=0.38)
    settings = (
        ("batch", BATCH_COLORS, "Batch", "C"),
        ("celltype", CELLTYPE_COLORS, "Cell Type", "D"),
    )
    for row, (field, palette, title, label) in enumerate(settings):
        grid = outer[row].subgridspec(
            1,
            5,
            width_ratios=[1, 1, 1, 1, 0.82],
            wspace=0.34,
        )
        for index, method in enumerate(METHODS):
            add_umap_axis(fig.add_subplot(grid[0, index]), frame, method, field, palette)
        add_legend(fig.add_subplot(grid[0, 4]), field, palette, title)
        fig.text(
            0.002,
            0.992 if row == 0 else 0.493,
            label,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )
    fig.subplots_adjust(left=0.047, right=0.995, bottom=0.095, top=0.93)
    return fig


def main() -> None:
    configure_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    if len(frame) != 16382:
        raise RuntimeError(f"Expected 16,382 cells, found {len(frame)}")

    figure_c = make_panel(frame, "batch", BATCH_COLORS, "Batch", "C")
    figure_d = make_panel(frame, "celltype", CELLTYPE_COLORS, "Cell Type", "D")
    geometry = record_panel_geometry(figure_c, "C") + record_panel_geometry(figure_d, "D")
    pd.DataFrame(geometry).to_csv(OUT / "HYBRID_PANEL_GEOMETRY_QC.csv", index=False)
    export(figure_c, "Figure1C_hybrid_600dpi")
    export(figure_d, "Figure1D_hybrid_600dpi")
    export(make_combined(frame), "Figure1_CD_hybrid_600dpi")

    (OUT / "HYBRID_OUTPUT_README.md").write_text(
        "\n".join(
            [
                "# Figure 1C-D hybrid editable outputs",
                "",
                "- Dense UMAP point collections: rasterized at 600 dpi.",
                "- Titles, panel labels, axes, ticks, borders, and legends: vector/editable.",
                "- Coordinates and colors: identical to the locked reference-style Figure 1C/D source.",
                "- Full-vector master files remain in the parent directory.",
                "- Use PDF as the preferred Adobe Illustrator editing source.",
                "- PNG is preview-only.",
                "- All four UMAP plotting boxes are equal-size squares; see HYBRID_PANEL_GEOMETRY_QC.csv.",
                "- Only the point layers are rasterized; all labels and axes remain editable.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Hybrid editable Figure 1C-D written to {OUT}")


if __name__ == "__main__":
    main()
