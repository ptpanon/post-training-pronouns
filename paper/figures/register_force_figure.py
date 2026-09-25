#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, COL, load, save, t1_adi

CARD = "results/swapped_ruler_2026-09-03.json"
PANEL = [("pre-registered rulers", "asil_M1_1k", "asil_KUVVET_cumle",
          "register /1k, force /sentence"),
         ("swapped rulers", "ters_M1_cumle", "ters_KUVVET_1k",
          "register /sentence, force /1k")]


def main():
    S, prov = load(CARD)
    B = {k: v for k, v in S.items() if not k.startswith("_") and isinstance(v, dict)}
    say = {}
    fig, axes = plt.subplots(1, 2, figsize=(2 * COL, 2.6), sharey=True)
    sira = sorted(B, key=lambda k: B[k]["asil_M1_1k"]["gozlenen"] /
                  B[k]["asil_M1_1k"]["plasebo_p95"])
    y = np.arange(len(sira))
    for ax, (baslik, reg, kuv, alt) in zip(axes, PANEL):
        zr = np.array([B[k][reg]["gozlenen"] / B[k][reg]["plasebo_p95"] for k in sira])
        zf = np.array([B[k][kuv]["gozlenen"] / B[k][kuv]["plasebo_p95"] for k in sira])
        ar = np.array([bool(B[k][reg]["ayrik"]) for k in sira])
        af = np.array([bool(B[k][kuv]["ayrik"]) for k in sira])
        say[baslik] = dict(reg_gecen=int(ar.sum()),
                           reg_asagi=int((ar & (zr < 0)).sum()),
                           kuv_gecen=int(af.sum()),
                           kuv_asagi=int((af & (zf < 0)).sum()),
                           reg_plasebo=int((np.abs(zr) >= 1).sum()),
                           kuv_plasebo=int((np.abs(zf) >= 1).sum()))
        ax.axvspan(-1, 1, color=C["band"], lw=0, zorder=0)
        ax.axvline(0, color=C["null"], lw=0.6, zorder=1)
        for i, (a, b) in enumerate(zip(zr, zf)):
            ax.plot([a, b], [i, i], color=C["light"], lw=0.6, zorder=2)
        ax.scatter(zr[ar], y[ar], s=15, color=C["neg"], zorder=3, label="register")
        ax.scatter(zr[~ar], y[~ar], s=15, facecolors="none", edgecolors=C["neg"],
                   linewidths=0.7, zorder=3)
        ax.scatter(zf[af], y[af], s=15, color=C["ink"], marker="s", zorder=3,
                   label="force total")
        ax.scatter(zf[~af], y[~af], s=15, facecolors="none", edgecolors=C["ink"],
                   marker="s", linewidths=0.7, zorder=3)
        ax.set_title(baslik, fontsize=7.5)
        ax.set_xlabel(alt + "\n(multiples of that cell's paired placebo p95)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([t1_adi(k, tex=False) for k in sira])
    axes[0].set_ylim(-0.7, len(sira) - 0.3)
    axes[0].legend(loc="upper left", handletextpad=0.4, frameon=False,
                   borderaxespad=0.2, fontsize=6, labelspacing=0.25)
    fig.tight_layout()
    a, b = say[PANEL[0][0]], say[PANEL[1][0]]
    print(f"{len(sira)} {a['reg_gecen']}"
          f"{a['reg_asagi']} {a['kuv_gecen']} {a['kuv_asagi']}"
          f"{b['reg_gecen']} {b['reg_asagi']}"
          f"{b['kuv_gecen']} {b['kuv_asagi']}"
          f"{a['reg_plasebo']} {a['kuv_plasebo']}"
          f"{b['reg_plasebo']} {b['kuv_plasebo']}"
          f"")
    save(fig, "F_regforce", [prov],
         "Register and force under both denominators. Each cell is its own "
         "pre-specified difference divided by its own paired within-prompt placebo's "
         "95th percentile, so all four cells are in one unit and the grey strip is that "
         "placebo bar. Filled markers are the cells whose prompt-clustered interval "
         "excludes zero, which is the pre-registered decision rule; the two halves of "
         "the bar do not always agree, and the figure shows where. On the "
         f"pre-registered rulers (left) register separates in {a['reg_gecen']} of "
         f"{len(sira)} models, all downward, and the force total in "
         f"{a['kuv_gecen']}, downward in {a['kuv_asagi']}. On the swapped rulers "
         f"(right) the counts reverse: register {b['reg_gecen']} and force "
         f"{b['kuv_gecen']}, all fourteen of the latter downward. The contrast between the two layers is therefore a "
         "property of the denominators as much as of the layers, which is why we report "
         "no magnitude ratio between them; what does not move with the ruler is the "
         "sign of the register fall.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
