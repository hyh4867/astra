"""Validation PR curves and recall at shared precision floors, no test data."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT=Path(__file__).resolve().parents[1]/"outputs/direction_control"
COLORS={"forward":"#186286","max":"#b5461e"}
LABELS={"forward":"Forward","max":"Temporal maximum"}


def main():
    with (OUT/"validation_pr.csv").open() as f:
        rows=list(csv.DictReader(f))
    with (OUT/"validation_mean.csv").open() as f:
        target=next(float(r["precision"]) for r in csv.DictReader(f) if r["mode"]=="forward" and r["pct"]=="50")
    fig,axs=plt.subplots(2,3,figsize=(12.5,7.2),sharex="row",sharey="row",layout="constrained")
    floors=[n/100 for n in range(1,100)]+[target]
    matched=[]
    for j,seed in enumerate(("42","1042","2042")):
        axs[0,j].set_title(f"Seed {seed}",loc="left",fontweight="bold")
        available={}
        for mode in COLORS:
            group=[(float(r["precision"]),float(r["recall"])) for r in rows if r["seed"]==seed and r["mode"]==mode]
            available[mode]=group
            axs[0,j].plot([r for p,r in group],[p for p,r in group],".-",color=COLORS[mode],label=LABELS[mode])
        for mode,group in available.items():
            x,y=[],[]
            for floor in sorted(floors):
                matches={m:[r for p,r in g if p>=floor] for m,g in available.items()}
                if not all(matches.values()) or floor<max(min(p for p,r in g) for g in available.values()):
                    continue
                x.append(floor); y.append(max(matches[mode]))
                matched.append(dict(seed=seed,mode=mode,precision_floor=floor,recall=y[-1]))
            axs[1,j].step(x,y,where="pre",color=COLORS[mode])
        axs[1,j].axvline(target,color="0.35",ls=":",lw=1.2)
        axs[1,j].set_xlabel("Required precision")
        axs[0,j].set_xlabel("Recall")
        for ax in axs[:,j]:
            ax.grid(alpha=.25);ax.spines[["right","top"]].set_visible(False)
            ax.xaxis.set_major_formatter(PercentFormatter(1,0))
            ax.yaxis.set_major_formatter(PercentFormatter(1,0))
    axs[0,0].set_ylabel("Precision")
    axs[1,0].set_ylabel("Best eligible recall")
    axs[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle("FOTBCD validation · 984 pairs\nThresholds 0.10–0.90 in steps of 0.05; dotted line = mean forward precision at threshold 0.50",fontsize=13)
    fig.savefig(OUT/"validation_pr.png",dpi=160)
    plt.close(fig)
    with (OUT/"matched_precision_curve.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,matched[0],lineterminator="\n");w.writeheader();w.writerows(matched)


if __name__=="__main__":
    main()
