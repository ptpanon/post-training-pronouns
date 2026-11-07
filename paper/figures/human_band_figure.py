#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import C, COL, load, save

INSAN = "results/human_band_2026-09-10_reservoir_fixed.json"
YAZILI = "results/written_anchor_2026-09-01.json"
PANEL = "results/PREREG9_A_OLCUM_2026-08-31.json"
ASISTAN = "results/v27_human_assistant_2026-09-10_reservoir_fixed.json"
TEMPLATE = "results/template_filtered_2026-09-07.json"
OLCU = [("M1", "2nd-person forms\nper 1000 tokens"),
        ("M3", "impersonal share\nof corrections"),
        ("KUVVET", "force total\n(acts per sentence)")]


def main():
    H, p1 = load(INSAN); S, p2 = load(PANEL); Y, p3 = load(YAZILI)
    A, p4 = load(ASISTAN)
    B = {k: v for k, v in S.items() if isinstance(v, dict) and v.get("sinif") == "BIRINCIL"}
    taban, hizali = {}, {}
    for o, _ in OLCU:
        b, i = [], []
        for v in B.values():
            k0, k1 = v["kontrast"].split("→")
            b.append(v["basamak_tablosu"][k0][o]); i.append(v["basamak_tablosu"][k1][o])
        taban[o] = (min(b), max(b)); hizali[o] = (min(i), max(i))

    NOKTA = [("justices", H["SCOTUS"]["J"], "spoken"),
             ("debate", H["CMV"]["hepsi"], "spoken"),
             ("advocates", H["SCOTUS"]["A"], "spoken"),
             ("how-to", Y["WIKIHOW"], "written"),
             ("Q&A", Y["SE"], "written"),
             ("textbook", Y["OPENSTAX"], "written")]
    if A.get("assistant", {}).get("hal") == "ÖLCÜLDÜ":
        NOKTA.append(("human assistants", A["assistant"], "assistant"))
    kon = sorted([x for x in NOKTA if x[2] == "spoken"], key=lambda x: -x[1]["M1"])
    yaz = sorted([x for x in NOKTA if x[2] == "written"], key=lambda x: -x[1]["M1"])
    asi = sorted([x for x in NOKTA if x[2] == "assistant"], key=lambda x: -x[1]["M1"])
    SIRA = kon + yaz + asi
    T, p5 = load(TEMPLATE)
    _h = [r["sbl"]["M1_hizali"] for r in T["aileler"] if r["hal"] == "ÖLCÜLDÜ"]
    sbl_bant = {"M1": (min(_h), max(_h))}
    BANT = [("base models", taban, C["ink"]),
            ("aligned assistants", hizali, C["neg"]),
            ("aligned, chat template", sbl_bant, C["accent"])]
    n = len(SIRA) + len(BANT)
    yv = {ad: n - 1 - i for i, (ad, *_) in enumerate(SIRA)}
    for j, (ad, *_) in enumerate(BANT):
        yv[ad] = n - 1 - (len(SIRA) + j)
    REN = {"spoken": C["accent"], "written": C["pos"], "assistant": C["neg"]}
    print(f"{len(OLCU)} {n}"
          f"{len(kon)} {len(yaz)} {len(asi)}"
          f"{len(BANT)}"
          f"")

    fig, axes = plt.subplots(1, 3, figsize=(COL * 2.05, 1.55), sharey=True)
    fig.subplots_adjust(wspace=0.12, left=0.20)
    for ax, (o, etiket) in zip(axes, OLCU):
        ax.axhspan(yv["textbook"] - 0.45, yv["textbook"] + 0.45,
                   color=C["band"], alpha=0.55, lw=0, zorder=0)
        for ad, d, grup in SIRA:
            if d.get(o) is None:
                continue
            ax.scatter([d[o]], [yv[ad]], s=20, color=REN[grup],
                       marker="o" if grup == "spoken" else "s",
                       facecolors=REN[grup] if grup == "spoken" else "none",
                       edgecolors=REN[grup], linewidths=1.0, zorder=4)
        for ad, kay, col in BANT:
            if o not in kay:
                continue
            lo, hi = kay[o]; y = yv[ad]
            ax.add_patch(Rectangle((lo, y - 0.17), hi - lo, 0.34, facecolor=col,
                                   alpha=0.35, edgecolor=col, lw=0.8, zorder=3))
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.28, y + 0.28], color=col, lw=0.9, zorder=4)
        ax.set_xlabel(etiket); ax.set_ylim(-0.7, n - 0.3)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([yv[a] for a in list(yv)])
    axes[0].set_yticklabels(list(yv), fontsize=6)
    for t_, a in zip(axes[0].get_yticklabels(), list(yv)):
        t_.set_color(dict(SIRA and [(x[0], REN[x[2]]) for x in SIRA]).get(
            a, C["ink"] if a == "base models"
            else C["accent"] if a == "aligned, chat template" else C["neg"]))
    axes[0].tick_params(axis="y", length=0)
    print(f"{len(_h)}"
          f"{min(_h):.2f} {max(_h):.2f}"
          f"{A['assistant']['M1_ci'][0]:.2f} {A['assistant']['M1_ci'][1]:.2f}"
          f"")
    save(fig, "F_humanband", [p1, p2, p3, p4, p5],
         "Seven human references and the three model bands on one row layout, read "
         "with the same counters. Row order is fixed by second-person density "
         "and held across all three panels. Filled circles are spoken and "
         "interactive registers, open squares written expository ones; the "
         "shaded row is the textbook. The aligned band sits below every "
         "conversational group and below written how-to and Q&A prose, and "
         "lands on the textbook row on both counters that measure person. "
         "The third band reads the same aligned checkpoints through their own "
         "chat templates; it is measured on the first ruler only, so it is drawn "
         "in the first panel alone. It spans the human-assistant row rather than "
         "sitting below it, and no aligned checkpoint lands inside that row's "
         "interval under that protocol.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
