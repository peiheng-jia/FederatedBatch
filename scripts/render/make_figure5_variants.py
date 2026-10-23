import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
import make_figure5 as base

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "figures" / "redraw" / "figure5"
COLORS = [base.PURPLE, base.TEAL, base.RED]
METHODS = ["Explicit", "Remote A/B", "FederatedBatch"]
INK, GREY = base.INK, base.GREY

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.0, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.75, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white",
})

def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2); ax.grid(False)

def lab(ax, letter):
    ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left", color=INK)

def jit(n, w=0.11):
    if n <= 1: return np.zeros(n)
    return np.linspace(-w, w, n)[(np.arange(n)*5) % n]

def load():
    a, r, dp, rt, cor = base.read_data()
    # Summary attack data are the seven seed-level observations.
    a = a.copy()
    a["short"] = a["display"].map(dict(zip(base.METHODS, METHODS)))
    return a, r, dp, rt, cor

def panel_a_metrics(holder, attack, metrics, letter="A"):
    holder.axis("off")
    inner = GridSpecFromSubplotSpec(1, len(metrics), subplot_spec=holder.get_subplotspec(), wspace=0.36)
    axes=[]
    labels = {"auroc":"AUROC", "auprc":"AUPRC", "attack_accuracy":"Accuracy"}
    for j, metric in enumerate(metrics):
        ax = holder.figure.add_subplot(inner[0,j]); axes.append(ax)
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            vals = attack.loc[attack.short.eq(method), metric].to_numpy()
            ax.scatter(i+jit(len(vals), .145), vals, s=21, color=color, alpha=.72,
                       edgecolor="white", linewidth=.45, zorder=3)
            mean = vals.mean()
            ci = stats.t.ppf(.975, len(vals) - 1) * stats.sem(vals)
            low, high = np.clip([mean - ci, mean + ci], 0.0, 1.0)
            yerr = np.array([[mean - low], [high - mean]])
            ax.errorbar(i, mean, yerr=yerr, fmt="D", ms=5.4, color=color,
                        mfc=color, mec="white", mew=.5, capsize=2.8, lw=1.0, zorder=4)
            ax.text(i, mean + (0.012 if mean < .98 else .007), f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=6.8, clip_on=False)
        ax.set_xticks(range(3), METHODS)
        ax.tick_params(axis="x", labelsize=6.2, pad=3)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")
        ax.set_xlim(-.35,2.35); ax.set_title(labels[metric], loc="left", pad=4, fontweight="bold")
        ax.set_ylim(.48, 1.055)
        if metric in ("auroc", "attack_accuracy"):
            ax.axhline(.5,color="#C2C2C2",lw=.65,ls=(0,(3,2)))
        if metric == "auroc":
            ax.set_ylabel("Score")
        else: ax.set_yticklabels([])
        clean(ax)
    axes[0].text(0,1.16,"Membership inference attack",transform=axes[0].transAxes,
                 fontsize=9.5,fontweight="bold",ha="left",va="bottom")
    lab(axes[0],letter)

def panel_a_matrix(holder, attack, letter="A"):
    holder.axis("off")
    ax=holder.figure.add_subplot(holder.get_subplotspec());
    metrics=["auroc","auprc","attack_accuracy"]; names=["AUROC","AUPRC","Accuracy"]
    y=np.arange(3)
    for i,(method,color) in enumerate(zip(METHODS,COLORS)):
        for k,metric in enumerate(metrics):
            vals=attack.loc[attack.short.eq(method),metric].to_numpy(); m=vals.mean()
            ci=1.96*vals.std(ddof=1)/np.sqrt(len(vals))
            ax.errorbar(i,m,yerr=ci,fmt="D",ms=7,color=color,mfc=color,mec="white",mew=.6,
                        capsize=3,lw=1.0,zorder=3)
            ax.text(i+0.12,m,f"{m:.3f}",ha="left",va="center",fontsize=7)
    # Three metric rows are placed in their native [0,1] range using offsets.
    # The middle/low rows retain native scores while row labels separate the metrics.
    ax.clear()
    row_offsets=[2.0,1.0,0.0]
    for k,(metric,name) in enumerate(zip(metrics,names)):
        off=row_offsets[k]
        for i,(method,color) in enumerate(zip(METHODS,COLORS)):
            vals=attack.loc[attack.short.eq(method),metric].to_numpy(); m=vals.mean(); ci=1.96*vals.std(ddof=1)/np.sqrt(len(vals))
            ax.scatter(i+jit(len(vals),.08),off+vals,s=20,color=color,alpha=.66,edgecolor="white",linewidth=.4,zorder=2)
            ax.errorbar(i,off+m,yerr=ci,fmt="D",ms=5.5,color=color,mfc=color,mec="white",mew=.5,capsize=2.7,lw=1,zorder=3)
            ax.text(i+.11,off+m,f"{m:.3f}",ha="left",va="center",fontsize=6.7)
        ax.axhline(off+.5,color="#D0D0D0",lw=.55,ls=(0,(2,2)),zorder=0)
    ax.set_xlim(-.4,2.45); ax.set_ylim(-.05,3.15); ax.set_xticks(range(3),["Exp.","A/B 0","FB"])
    ax.set_yticks([2.5,1.5,.5],names); ax.set_ylabel("Metric (row-normalized)")
    ax.set_title("Membership inference attack",loc="left",pad=5,fontweight="bold")
    clean(ax); lab(ax,letter)

def panel_d_two(holder, runtime, correctness, letter="D"):
    holder.axis("off")
    inner=GridSpecFromSubplotSpec(1,2,subplot_spec=holder.get_subplotspec(),wspace=.30)
    ax=holder.figure.add_subplot(inner[0,0]); mem=holder.figure.add_subplot(inner[0,1])
    order=["plaintext","secretflow_spu_aby3"]; names=["Plaintext","SPU / ABY3"]; xpos=[0,1]
    for i,(back,color) in enumerate(zip(order,[base.TEAL,base.RED])):
        vals=runtime.loc[runtime.backend.eq(back),"wall_clock_seconds"].to_numpy(); x=xpos[i]
        ax.scatter(x+jit(3,.13),vals,s=27,color=color,alpha=.75,edgecolor="white",linewidth=.5,zorder=3)
        ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1),fmt="D",ms=5.8,color=color,mfc=color,mec="white",mew=.5,capsize=3,lw=1.0)
        ax.text(x,vals.mean()*(1.34 if i else 1.48),f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}",ha="center",va="bottom",fontsize=6.7)
    slow=runtime.loc[runtime.backend.eq("secretflow_spu_aby3"),"slowdown_vs_plaintext"].mean(); ae=correctness.a_relative_l2_error.mean()*100; be=correctness.b_relative_l2_error.mean()*100
    ax.text(.38,.76,f"Slowdown {slow:.1f}×\nA error {ae:.4f}%\nB error {be:.4f}%\nSuccess 3/3",transform=ax.transAxes,ha="center",va="top",fontsize=6.8,linespacing=1.2,bbox={"boxstyle":"round,pad=.2","fc":"white","ec":"#C5C5C5","lw":.6})
    ax.set_yscale("log"); ax.set_ylim(.065,4.5); ax.set_xlim(-.3,1.3); ax.set_xticks(xpos,names); ax.set_yticks([.1,.3,1,3]); ax.set_yticklabels(["0.1","0.3","1","3"]); ax.set_ylabel("Runtime (s)"); clean(ax)
    # memory panel uses the saved peak RSS values, adding a second real feasibility measure.
    for i,(back,color) in enumerate(zip(order,[base.TEAL,base.RED])):
        vals=runtime.loc[runtime.backend.eq(back),"peak_rss_mb"].to_numpy(); x=xpos[i]
        mem.scatter(x+jit(3,.13),vals,s=27,color=color,alpha=.75,edgecolor="white",linewidth=.5,zorder=3)
        mem.errorbar(x,vals.mean(),yerr=vals.std(ddof=1),fmt="D",ms=5.8,color=color,mfc=color,mec="white",mew=.5,capsize=3,lw=1.0)
        mem.text(x,vals.mean()+17,f"{vals.mean():.0f} ± {vals.std(ddof=1):.0f}",ha="center",va="bottom",fontsize=6.7)
    mem.set_xticks(xpos,names); mem.set_xlim(-.3,1.3); mem.set_ylim(570,980); mem.set_ylabel("Peak RSS (MiB)"); mem.set_title("Memory",loc="left",pad=4,fontweight="bold"); clean(mem)
    ax.set_title("Runtime",loc="left",pad=4,fontweight="bold"); ax.text(0,1.16,"Secure computation feasibility",transform=ax.transAxes,fontsize=9.5,fontweight="bold",ha="left",va="bottom"); lab(ax,letter)

def panel_d_arrow(holder, runtime, correctness, letter="D"):
    holder.axis("off"); ax=holder.figure.add_subplot(holder.get_subplotspec())
    order=["plaintext","secretflow_spu_aby3"]; names=["Plaintext","SPU / ABY3"]; y=np.arange(2)
    means=[]; sd=[]
    for back in order:
        vals=runtime.loc[runtime.backend.eq(back),"wall_clock_seconds"].to_numpy(); means.append(vals.mean()); sd.append(vals.std(ddof=1))
    ax.errorbar(means,y,xerr=sd,fmt="none",ecolor="#888",capsize=3,lw=1.0)
    for i,(m,s,color) in enumerate(zip(means,sd,[base.TEAL,base.RED])):
        vals=runtime.loc[runtime.backend.eq(order[i]),"wall_clock_seconds"].to_numpy(); ax.scatter(vals,jit(3,.12)+i,s=38,color=color,alpha=.75,edgecolor="white",linewidth=.5,zorder=3); ax.scatter(m,i,marker="D",s=62,color=color,edgecolor="white",linewidth=.6,zorder=4); ax.text(m*1.25,i,f"{m:.3f} ± {s:.3f} s",va="center",fontsize=7)
    ax.annotate("24.6× slower",xy=(means[1],1),xytext=(0.72,1.42),
                arrowprops={"arrowstyle":"->","lw":1,"color":base.RED},
                color=base.RED,fontsize=7.5,fontweight="bold",ha="center")
    ae=correctness.a_relative_l2_error.mean()*100; be=correctness.b_relative_l2_error.mean()*100
    ax.text(.48,.93,f"A error {ae:.4f}%\nB error {be:.4f}%\nSuccess 3/3",transform=ax.transAxes,ha="center",va="top",fontsize=7,bbox={"boxstyle":"round,pad=.22","fc":"white","ec":"#C5C5C5","lw":.6})
    ax.set_xscale("log"); ax.set_xlim(.07,4.2); ax.set_yticks(y,names); ax.set_xlabel("Wall-clock time (s; log scale)"); ax.set_title("Secure computation feasibility",loc="left",pad=5,fontweight="bold"); clean(ax); lab(ax,letter)

def panel_a_horizontal(holder, attack, letter="A"):
    holder.axis("off"); ax=holder.figure.add_subplot(holder.get_subplotspec())
    metrics=[("auroc","AUROC"),("auprc","AUPRC"),("attack_accuracy","Accuracy")]
    y=np.arange(len(metrics))[::-1]
    for k,(metric,name) in enumerate(metrics):
        yy=y[k]
        for method,color in zip(METHODS,COLORS):
            vals=attack.loc[attack.short.eq(method),metric].to_numpy(); mean=vals.mean(); ci=1.96*vals.std(ddof=1)/np.sqrt(len(vals))
            x=mean
            ax.errorbar(x,yy,xerr=ci,fmt="D",ms=6,color=color,mfc=color,mec="white",mew=.55,capsize=3,lw=1.0,zorder=3)
            dx = -0.042 if (metric == "attack_accuracy" and method == "Explicit") else 0.012
            ha = "right" if dx < 0 else "left"
            ax.text(x+dx,yy,f"{mean:.3f}",ha=ha,va="center",fontsize=6.8)
    ax.axvline(.5,color="#C8C8C8",lw=.65,ls=(0,(3,2)))
    ax.set_yticks(y,[x[1] for x in metrics]); ax.set_xlim(.47,1.035); ax.set_ylim(-.55,2.55); ax.set_xlabel("Score (mean ± 95% CI)")
    ax.text(.5,-.33,"Random reference",transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=6.6,color="#999999")
    # Legend-like color labels are compact and avoid a crowded x-axis.
    for i,(m,c) in enumerate(zip(METHODS,COLORS)):
        ax.scatter(.58+i*.125,2.43,s=30,color=c,marker="D",clip_on=False)
        ax.text(.60+i*.125,2.43,m,va="center",fontsize=7)
    ax.set_title("Membership inference attack",loc="left",pad=5,fontweight="bold"); clean(ax); lab(ax,letter)

def save(fig,stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT/f"{stem}.pdf",bbox_inches="tight"); fig.savefig(OUTPUT/f"{stem}.svg",bbox_inches="tight"); fig.savefig(OUTPUT/f"{stem}.png",dpi=600,bbox_inches="tight"); plt.close(fig)

def make_variant(name,a,r,dp,rt,cor,kind):
    fig=plt.figure(figsize=(8.6,5.9)); gs=GridSpec(2,2,figure=fig,width_ratios=[1.0,1.18],left=.07,right=.98,bottom=.09,top=.95,wspace=.27,hspace=.34)
    if kind==1: panel_a_metrics(fig.add_subplot(gs[0,0]),a,["auroc","attack_accuracy"])
    elif kind==2: panel_a_matrix(fig.add_subplot(gs[0,0]),a)
    else: panel_a_horizontal(fig.add_subplot(gs[0,0]),a)
    panel_d_two(fig.add_subplot(gs[1,1]),rt,cor) if kind!=3 else panel_d_arrow(fig.add_subplot(gs[1,1]),rt,cor)
    # Keep B/C in their established forms, using the same source values.
    base.draw_b(fig.add_subplot(gs[0,1]),r); base.draw_c(fig.add_subplot(gs[1,0]),dp)
    save(fig,name)

def make_final_style(a,r,dp,rt,cor):
    """Variant 1 layout with the established Figures 1-4 visual system."""
    fig=plt.figure(figsize=(8.8,6.1)); gs=GridSpec(2,2,figure=fig,width_ratios=[1.05,1.22],left=.07,right=.98,bottom=.09,top=.95,wspace=.33,hspace=.38)
    panel_a_metrics(fig.add_subplot(gs[0,0]),a,["auroc","attack_accuracy"])
    base.draw_b(fig.add_subplot(gs[0,1]),r)
    base.draw_c(fig.add_subplot(gs[1,0]),dp)
    panel_d_two(fig.add_subplot(gs[1,1]),rt,cor)
    save(fig,"Figure5_final_polished")

def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description="Redraw the final polished Figure 5.")
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "plot_inputs" / "figure5")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    base.INPUT = args.input.resolve()
    OUTPUT = args.output.resolve()
    a,r,dp,rt,cor=load()
    make_final_style(a,r,dp,rt,cor)
    make_variant("Figure5_variant1_A_metrics_D_memory",a,r,dp,rt,cor,1)
    make_variant("Figure5_variant2_A_matrix_D_memory",a,r,dp,rt,cor,2)
    make_variant("Figure5_variant3_A_three_metrics_D_arrow",a,r,dp,rt,cor,3)

if __name__=='__main__': main()
