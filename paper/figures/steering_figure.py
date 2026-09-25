#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from style import C, COL, MDE, load, save, en

AILE = [("olmo", "OLMo-2-13B"), ("tulu3", "Tulu-3-8B"), ("zephyr", "Zephyr-7B")]


def inset(ax):
    ax.set_xlim(-1.15, 1.5); ax.set_ylim(-0.75, 1.35); ax.axis("off")
    ax.annotate("", xy=(1.12, 0), xytext=(-0.95, 0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C["ink"], mutation_scale=6))
    ax.text(1.16, 0, "D axis", fontsize=5.4, va="center", color=C["ink"])
    ax.plot(-0.75, 0, "o", ms=3.2, color=C["null"])
    ax.text(-0.78, -0.26, "base", fontsize=5.2, ha="center", color=C["null"])
    ax.add_patch(FancyArrowPatch((-0.75, 0), (0.55, 0.92), arrowstyle="-|>",
                                 mutation_scale=6, lw=1.0, color=C["accent"],
                                 connectionstyle="arc3,rad=-0.42"))
    ax.text(0.62, 0.98, "injected\nshift", fontsize=5.2, color=C["accent"], va="center")
    ax.plot([0.55, 0.55], [0, 0.92], ls=(0, (1.4, 1.4)), lw=0.6, color=C["null"])
    ax.text(0.60, 0.42, "orthogonal", fontsize=5.0, color=C["null"], rotation=90, va="center")


def main():
    d, prov = load("results/steering_scorecard_2026-08-23.json")
    V = {}
    for k, ad in AILE:
        A = d["AYRINTI"][k]["ayrinti"]
        V[ad] = {e: {c: (A[e]["yon_null"][c]["sd_kati"], bool(A[e][f"{c}_ayrik"]))
                     for c in ("duz", "dik")} for e in ("D", "A")}
    print(f"{len(V)}")

    fig = plt.figure(figsize=(COL * 2.05, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(-MDE, MDE, color=C["band"], lw=0, zorder=0)
    ax.axhline(0, color=C["null"], lw=0.6, zorder=1)
    x = np.arange(len(AILE)); w = 0.19
    for oi, (eks, etk, renk) in enumerate([("D", "dominance", C["pos"]),
                                           ("A", "arousal  (negative control)", C["null"])]):
        for ci, (kol, koyu) in enumerate([("duz", False), ("dik", True)]):
            xs = x + (oi - 0.5) * 0.46 + (ci - 0.5) * w
            for i, (_, ad) in enumerate(AILE):
                z, ayrik = V[ad][eks][kol]
                fc = renk if ayrik else "white"
                ax.bar(xs[i], z, width=w, color=fc, edgecolor=renk, lw=0.7,
                       alpha=1.0 if koyu else 0.42, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([a for _, a in AILE], fontsize=6.2)
    ax.set_ylabel("travel  (direction-null sd)", fontsize=6.6)
    ax.set_title("Injecting the dominance direction moves the model mostly "
                 "ORTHOGONAL to it", fontsize=6.8, pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, fc=C["pos"], ec=C["pos"], alpha=0.42,
                       label="along the D axis"),
         plt.Rectangle((0, 0), 1, 1, fc=C["pos"], ec=C["pos"], label="orthogonal to it"),
         plt.Rectangle((0, 0), 1, 1, fc="white", ec=C["null"], label="arousal control (hollow = not disjoint)")]
    ax.legend(handles=h, loc="upper center", ncol=3, fontsize=5.2, handlelength=1.2,
              bbox_to_anchor=(0.5, -0.13), columnspacing=1.0)
    inset(fig.add_subplot(gs[0, 1]))
    save(fig, "F3_steering", [prov],
         "Injecting the sealed dominance direction moves all three aligned models far "
         "further ORTHOGONAL to that axis than along it, while the arousal negative "
         "control stays inside the null band — the intervention detours around the axis "
         "it was built from.")


if __name__ == "__main__":
    main()
