#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from style import C, COL, load, save, en

KAN = [("c_KE", "interpersonal\ncommand"), ("c_YE", "procedural\ncommand"),
       ("c_DB", "dominant\nassertion")]
AILE = None
N_ILK = 6
SUB = [("SCOTUS (oral argument)", "SCOTUS (OKUMA-B)"), ("Reddit CMV", "cga-cmv")]


def sema(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.35); ax.axis("off")
    def kutu(x, y, w, h, t, fc, ec, fs=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=C["ink"])
    kutu(0.15, 1.30, 2.5, 0.72, "real context\n(class $c$)", "#e8f0f7", C["pos"])
    kutu(0.15, 0.28, 2.5, 0.72, "matched context\n(pseudo-neutral)", "#f7ece8", C["neg"])
    kutu(3.55, 0.79, 2.5, 0.72, "IDENTICAL\nscored text", "#f2f2f2", C["ink"], 6.4)
    kutu(7.0, 0.79, 2.85, 0.72,
         r"$\Delta = \mathrm{NLL}_{c} - \mathrm{NLL}_{\mathrm{pseudo}}$" + "\naligned − base",
         "white", C["ink"], 6.2)
    for y0 in (1.66, 0.64):
        ax.add_patch(FancyArrowPatch((2.72, y0), (3.50, 1.15), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.7, color=C["null"],
                                     connectionstyle="arc3,rad=%.2f" % (0.12 if y0 < 1 else -0.12)))
    ax.add_patch(FancyArrowPatch((6.12, 1.15), (6.95, 1.15), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.7, color=C["null"]))
    ax.text(4.80, 0.28, "only the context differs", ha="center", fontsize=5.6,
            color=C["null"], style="italic")


def main():
    F, prov = load("results/shadow_reading_16_scorecard_2026-08-25.json")
    kayit = [prov]
    global AILE
    AILE = list(F["substrat"]["SCOTUS (OKUMA-B)"]["hucre"])
    for sub, S in F["substrat"].items():
        ks = S.get("KOSULMADI", [])
        print(f"  [PAYDA] f5_{sub[:6]}: n_panel={len(S['hucre']) + len(ks)} · "
              f"hal_puanlanan={len(S['hucre'])} · atl_kosulmadi={len(ks)}"
              + (f"  ({', '.join(ks)})" if ks else ""))
    fig = plt.figure(figsize=(COL * 2.05, 5.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 2.10], hspace=0.24, wspace=0.10)
    sema(fig.add_subplot(gs[0, :]))

    for j, (baslik, anahtar) in enumerate(SUB):
        ax = fig.add_subplot(gs[1, j])
        S = F["substrat"][anahtar]
        for xi, (kan, _) in enumerate(KAN):
            for yi, a in enumerate(AILE):
                if S["hucre"][a].get("ad") == "KOSULMADI":
                    ax.text(xi, yi, "–", ha="center", va="center", fontsize=5.4,
                            color=C["light"], zorder=4)
                    continue
                v = S["hucre"][a]["kanal"][kan]
                s = S["hucre"][a]["sayilar"][kan]
                z = (s.get("sd_kati") or 0)
                if v == "POZ":
                    ax.scatter([xi], [yi], s=210, c=C["pos"], zorder=3, linewidths=0)
                    ax.text(xi, yi, f"{z:+.1f}", ha="center", va="center",
                            fontsize=4.6, color="white", zorder=4)
                elif v == "NEG":
                    ax.scatter([xi], [yi], s=210, c=C["neg"], zorder=3, linewidths=0)
                    ax.text(xi, yi, f"{z:+.1f}", ha="center", va="center",
                            fontsize=4.6, color="white", zorder=4)
                elif v == "KAPI":
                    ax.plot(xi, yi, "x", ms=5, color=C["null"], mew=1.0, zorder=3)
                else:
                    ax.scatter([xi], [yi], s=210, facecolors="none", edgecolors=C["light"],
                               linewidths=0.7, zorder=3)
                    ax.text(xi, yi, f"{z:+.1f}", ha="center", va="center",
                            fontsize=4.4, color=C["null"], zorder=4)
        ax.set_xlim(-0.55, 2.55); ax.set_ylim(-0.6, len(AILE) - 0.4)
        ax.axhline(N_ILK - 0.5, color=C["light"], lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(range(3)); ax.set_xticklabels([t for _, t in KAN], fontsize=5.8)
        ax.set_yticks(range(len(AILE)))
        ax.set_yticklabels(AILE if j == 0 else [], fontsize=5.2)
        ax.invert_yaxis()
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{baslik}\n{en(S['panel_adi_ESKI_BAR'])}  ·  {en(S['ilan_panel'])}",
                     fontsize=6.2, pad=5)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=5, color=C["pos"],
                    label="disjoint, positive (context makes reading harder)"),
         plt.Line2D([], [], marker="o", ls="none", ms=5, color=C["neg"],
                    label="disjoint, negative (context makes reading easier)"),
         plt.Line2D([], [], marker="o", ls="none", ms=5, mfc="none", mec=C["light"],
                    label="not disjoint  (cell value = null-sd multiple)")]
    fig.legend(handles=h, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.075),
               handletextpad=0.5)
    save(fig, "F5_shadow", kayit,
         "Identical text is scored under a real and a matched pseudo-neutral context; "
         "under the two-sided empirical bar the interpersonal-command effect replicates "
         "across both substrates (3/6 and 3/6), while the procedural command separates "
         "only on Reddit CMV and the dominant assertion is silent on SCOTUS but "
         "reading-facilitating on CMV. The panel is now sixteen models; the dashed "
         "rule separates the six on which the effect was originally established from the ten "
         "added afterwards, so the reader can see the replication rather than take it "
         "on trust.")


if __name__ == "__main__":
    main()
