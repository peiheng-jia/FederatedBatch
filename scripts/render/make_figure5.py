import argparse
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "plot_inputs" / "figure5"
OUTPUT = ROOT / "figures" / "redraw" / "figure5"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

PURPLE = "#8B78A5"
TEAL = "#4F7F83"
RED = "#D44C3C"
INK = "#303030"
GREY = "#707070"
LIGHT = "#D9D9D9"
METHODS = ["Explicit", "Remote A/B", "FederatedBatch"]
COLORS = [PURPLE, TEAL, RED]


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def panel_label(ax, letter):
    ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left", color=INK)


def read_data():
    attack = pd.read_csv(INPUT / "membership_attack.csv")
    attack = attack[attack["round"].astype(str).eq("all")].copy()
    interface_map = {
        "Vanilla FreshQueue: individual embedding exposure": METHODS[0],
        "RoundOnce Joint-C50 aggregate A/B, sigma=0": METHODS[1],
        "Final private RoundOnce Joint-C50 aggregate A/B, sigma=2": METHODS[2],
    }
    attack["display"] = attack["interface"].map(interface_map)

    recon = pd.read_csv(INPUT / "reconstruction_per_cell.csv")
    recon = recon[recon["decoder"].eq("MLP")].copy()
    condition_map = {
        "Vanilla FreshQueue direct embedding": METHODS[0],
        "Aggregate A/B sigma=0": METHODS[1],
        "Final Joint-C50 sigma=2": METHODS[2],
    }
    recon["display"] = recon["condition"].map(condition_map)

    dp = pd.read_csv(INPUT / "privacy_accounting.csv")
    dp = dp[np.isclose(dp["delta"], 1e-5)].sort_values("sigma")

    runtime = pd.read_csv(INPUT / "runtime_benchmark.csv")
    runtime = runtime[(runtime["case"].eq("full_round4_128x102")) &
                      (runtime["phase"].eq("measured"))].copy()
    correctness = pd.read_csv(INPUT / "correctness.csv")
    correctness = correctness[correctness["case"].eq("full_round4_128x102")].copy()
    return attack, recon, dp, runtime, correctness


def draw_a(ax, attack, label=True):
    rng = np.random.default_rng(42)
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        vals = attack.loc[attack["display"].eq(method), "auroc"].to_numpy()
        jitter = np.linspace(-0.07, 0.07, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=20, facecolor="white",
                   edgecolor=color, linewidth=0.9, zorder=3)
        mean = vals.mean()
        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
        ax.errorbar(i, mean, yerr=ci, fmt="D", ms=5, color=color,
                    mfc=color, mec="white", mew=0.5, capsize=3, lw=1.3, zorder=4)
        ax.text(i, mean + (0.018 if mean < 0.98 else -0.025), f"{mean:.3f}",
                ha="center", va="bottom" if mean < 0.98 else "top", fontsize=7.5, color=INK)
    ax.axhline(0.5, color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(2.42, 0.51, "Random", ha="right", va="bottom", color=GREY, fontsize=7)
    ax.set_xticks(range(3), METHODS)
    ax.set_ylabel("Attack AUROC")
    ax.set_ylim(0.46, 1.04)
    ax.set_title("Membership inference attack", loc="left", pad=8, fontweight="bold")
    clean_axis(ax)
    if label: panel_label(ax, "A")


def box_with_mean(ax, recon, metric, ylabel, better_note):
    arrays = [recon.loc[recon["display"].eq(m), metric].dropna().to_numpy() for m in METHODS]
    bp = ax.boxplot(arrays, positions=np.arange(3), widths=0.52, patch_artist=True,
                    showfliers=False, medianprops={"color": "white", "linewidth": 1.2},
                    whiskerprops={"color": GREY, "linewidth": 0.8},
                    capprops={"color": GREY, "linewidth": 0.8})
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    for i, (arr, color) in enumerate(zip(arrays, COLORS)):
        ax.scatter(i, arr.mean(), marker="D", s=22, color="white", edgecolor=INK,
                   linewidth=0.8, zorder=5)
        ax.text(i, arr.mean(), f" {arr.mean():.3f}", ha="left", va="center", fontsize=7.2)
    ax.set_xticks(range(3), METHODS)
    ax.tick_params(axis="x", labelsize=6.2, pad=3)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    ax.text(0.99, 1.025, better_note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.0, color=GREY, clip_on=False)
    clean_axis(ax)


def draw_b(container, recon, label=True):
    container.axis("off")
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=container.get_subplotspec(), wspace=0.30)
    ax1 = container.figure.add_subplot(inner[0, 0])
    ax2 = container.figure.add_subplot(inner[0, 1])
    box_with_mean(ax1, recon, "pearson", "Pearson correlation", "Lower = weaker attack")
    box_with_mean(ax2, recon, "rmse", "Reconstruction RMSE", "Higher = weaker attack")
    ax1.set_ylim(-0.05, 1.02)
    ax2.set_ylim(0.08, 0.70)
    ax1.text(0, 1.14, "Gene-expression reconstruction", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", ha="left", va="bottom")
    if label: panel_label(ax1, "B")
    return ax1, ax2


def draw_c(ax, dp, label=True):
    x, y = dp["sigma"].to_numpy(), dp["epsilon"].to_numpy()
    ax.plot(x, y, color=GREY, lw=1.25, marker="o", ms=4.3, mfc="white", mec=GREY)
    for xx, yy in zip(x, y):
        ax.text(xx, yy + 0.42, f"{yy:.3f}", ha="center", va="bottom", fontsize=7.5)
    final = dp[np.isclose(dp["sigma"], 2.0)].iloc[0]
    ax.scatter([final.sigma], [final.epsilon], s=48, color=RED, edgecolor="white",
               linewidth=0.8, zorder=5)
    ax.annotate("Selected setting", xy=(final.sigma, final.epsilon), xytext=(2.60, 3.3),
                arrowprops={"arrowstyle": "-", "lw": 0.8, "color": RED},
                color=RED, fontsize=7.5, ha="center")
    ax.set_xlabel("Gaussian noise multiplier, $\\sigma$")
    ax.set_ylabel("Privacy budget, $\\epsilon$ ($\\delta=10^{-5}$)")
    ax.set_xticks(x)
    ax.set_ylim(0, 11.2)
    ax.set_title("Differential privacy accounting", loc="left", pad=8, fontweight="bold")
    clean_axis(ax)
    if label: panel_label(ax, "C")


def draw_d(ax, runtime, correctness, label=True):
    backend_order = ["plaintext", "secretflow_spu_aby3"]
    labels = ["Plaintext", "SPU / ABY3"]
    colors = [TEAL, RED]
    for i, (backend, color) in enumerate(zip(backend_order, colors)):
        vals = runtime.loc[runtime["backend"].eq(backend), "wall_clock_seconds"].to_numpy()
        jitter = np.linspace(-0.06, 0.06, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=23, facecolor="white",
                   edgecolor=color, linewidth=0.9, zorder=3)
        ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="D", ms=6,
                    color=color, mfc=color, mec="white", mew=0.5, capsize=4, lw=1.3)
        ax.text(i, vals.mean() + 0.14, f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f} s",
                ha="center", va="bottom", fontsize=7.5)
    s = runtime.loc[runtime["backend"].eq("secretflow_spu_aby3"), "wall_clock_seconds"].mean()
    slowdown = runtime.loc[runtime["backend"].eq("secretflow_spu_aby3"), "slowdown_vs_plaintext"].mean()
    ae = correctness["a_relative_l2_error"].mean() * 100
    be = correctness["b_relative_l2_error"].mean() * 100
    ax.text(0.04, 0.94, f"Slowdown  {slowdown:.1f}×\nA error  {ae:.4f}%\nB error  {be:.4f}%\nSuccess  3/3",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, linespacing=1.45,
            color=INK)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_ylim(0, 3.15)
    ax.set_title("Secure computation feasibility", loc="left", pad=8, fontweight="bold")
    clean_axis(ax)
    if label: panel_label(ax, "D")


def save_all(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def export_plot_data(attack, recon, dp, runtime, correctness):
    rows = []
    for _, r in attack.iterrows():
        rows.append(["A", "membership_auroc", r["display"].replace("\n", " "),
                     "seed", int(r.seed), "auroc", r.auroc, "membership_attack.csv"])
    for _, r in recon.iterrows():
        for metric in ["pearson", "rmse"]:
            rows.append(["B", "expression_reconstruction", r["display"].replace("\n", " "),
                         "target_cell", r.cell_id, metric, r[metric], "reconstruction_per_cell.csv"])
    for _, r in dp.iterrows():
        rows.append(["C", "analytic_gaussian_accounting", "Joint cell-level Gaussian release",
                     "sigma", r.sigma, "epsilon_delta_1e-5", r.epsilon, "privacy_accounting.csv"])
    for _, r in runtime.iterrows():
        rows.append(["D", "secure_runtime", r.backend, "repetition", int(r.repetition),
                     "wall_clock_seconds", r.wall_clock_seconds, "runtime_benchmark.csv"])
    for metric in ["a_relative_l2_error", "b_relative_l2_error", "b_cosine_similarity",
                   "reconstructed_remote_gradient_cosine"]:
        for _, r in correctness.iterrows():
            rows.append(["D", "secure_correctness", "secretflow_spu_aby3", "repetition",
                         int(r.repetition), metric, r[metric], "correctness.csv"])
    with open(OUTPUT / "figure5_plot_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "analysis", "method", "unit_type", "unit_id", "metric", "value", "source_file"])
        w.writerows(rows)


def main():
    global INPUT, OUTPUT
    parser = argparse.ArgumentParser(description="Redraw Figure 5 from bundled plot inputs.")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    INPUT = args.input.resolve()
    OUTPUT = args.output.resolve()
    attack, recon, dp, runtime, correctness = read_data()
    export_plot_data(attack, recon, dp, runtime, correctness)

    fig = plt.figure(figsize=(10.0, 7.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.34,
                  left=0.08, right=0.98, top=0.95, bottom=0.08)
    draw_a(fig.add_subplot(gs[0, 0]), attack)
    draw_b(fig.add_subplot(gs[0, 1]), recon)
    draw_c(fig.add_subplot(gs[1, 0]), dp)
    draw_d(fig.add_subplot(gs[1, 1]), runtime, correctness)
    save_all(fig, "Figure5")

    fig, ax = plt.subplots(figsize=(5.2, 4.0)); draw_a(ax, attack); save_all(fig, "Figure5A_membership_attack")
    fig = plt.figure(figsize=(6.7, 3.9)); gs = GridSpec(1, 1, figure=fig, left=0.10, right=0.98, top=0.84, bottom=0.16)
    draw_b(fig.add_subplot(gs[0, 0]), recon); save_all(fig, "Figure5B_expression_reconstruction")
    fig, ax = plt.subplots(figsize=(5.2, 4.0)); draw_c(ax, dp); save_all(fig, "Figure5C_DP_accounting")
    fig, ax = plt.subplots(figsize=(5.2, 4.0)); draw_d(ax, runtime, correctness); save_all(fig, "Figure5D_secure_computation")


if __name__ == "__main__":
    main()
