from __future__ import annotations

from pathlib import Path

import anndata as ad

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np

import umap

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
