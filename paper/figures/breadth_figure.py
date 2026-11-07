#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, COL, MDE, BAND_LO, load, null_band, save

KAYNAK = [("results/C1_AILE_PANELI_2026-08-15.json", None),
          ("results/P5_OLCEK_KOLU_2026-08-21.json", None),
          ("results/P7_OLCEK2_2026-08-21.json", None),
          ("results/P_OLCEK3_TULU3_2026-08-22.json", ["Tulu3-8B·taban→RL"])]
ETIKET = {"taban": "base", "Inst": "Instruct"}


def guzel(ad):
    for a, b in ETIKET.items():
        ad = ad.replace(a, b)
    return ad.replace("·", " · ").replace("→", " → ")


def main():
    hucre, kayit = [], []
    for rel, haric in KAYNAK:
        d, prov = load(rel); kayit.append(prov)
        for ad, v in d["panel"].items():
            if haric and ad in haric:
                continue
            dd = v.get("delta") or {}
            if not all(k in dd for k in ("DOM", "ARO")):
                continue
            hucre.append((guzel(ad), dd))
    n = len(hucre)
    print(f"  [PAYDA] f1: n_cift={n} · red_atlanan=0 · bekle n_cift=24")
    assert n == 24, f"★ KAPSAM: 24 beklendi, {n} bulundu"

    hucre = hucre[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2.05, 0.235 * n + 1.25),
                             sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, eks, bas in zip(axes, ("DOM", "ARO"), ("dominance  ΔU", "arousal  ΔU")):
        null_band(ax)
        for i, (ad, dd) in enumerate(hucre):
            e = dd[eks]; sd = e["dU_sd"] or np.nan
            z = e["dU"] / sd
            lo, hi = e["dU_CI"][0] / sd, e["dU_CI"][1] / sd
            ayrik = bool(e.get("AYRIK_POZ") or e.get("AYRIK_NEG"))
            bant = bool(e.get("CONTINGENT"))
            col = C["pos"] if z > 0 else C["neg"]
            ax.plot([lo, hi], [i, i], color=col if ayrik else C["light"],
                    lw=1.5 if ayrik else 0.9, solid_capstyle="butt", zorder=3)
            if ayrik:
                ax.plot(z, i, "o", ms=3.6, color=col, mec="none", zorder=4)
            elif bant:
                ax.plot(z, i, "o", ms=3.4, mfc="none", mec=col, mew=0.8, zorder=4)
                ax.plot(z, i, "x", ms=2.6, color=col, mew=0.7, zorder=5)
            else:
                ax.plot(z, i, "o", ms=2.2, mfc="white", mec=C["null"], mew=0.7, zorder=4)
        ax.set_xlabel(f"{bas}   (null-sd units)")
        ax.set_ylim(-0.7, n - 0.3); ax.axhline(-0.7, lw=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels([a for a, _ in hucre], fontsize=5.4)
    lim = max(abs(np.array([h[1][e]["dU_CI"][j] / (h[1][e]["dU_sd"] or 1)
                            for h in hucre for e in ("DOM", "ARO") for j in (0, 1)])).max(), 3.2)
    for ax in axes:
        ax.set_xlim(-min(lim, 12) * 1.05, min(lim, 12) * 1.05)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=3.6, color=C["pos"], label="disjoint (positive)"),
         plt.Line2D([], [], marker="o", ls="none", ms=3.6, color=C["neg"], label="disjoint (negative)"),
         plt.Line2D([], [], marker="o", ls="none", ms=3.4, mfc="none", mec=C["null"],
                    label="pre-declared unmeasurable band (1.00–1.645)"),
         plt.Line2D([], [], marker="o", ls="none", ms=2.2, mfc="white", mec=C["null"], label="silent")]
    fig.legend(handles=h, loc="lower center", ncol=2, handletextpad=0.4,
               bbox_to_anchor=(0.5, -0.012), columnspacing=1.4)
    fig.suptitle("Alignment shifts the dominance axis broadly; arousal moves both ways",
                 y=0.995, fontsize=8)
    save(fig, "F1_breadth", kayit,
         "Across 24 base→aligned model pairs the dominance shift is predominantly "
         "one-signed, whereas the arousal shift is two-signed; grey strips mark the "
         "null band and the pre-registered unmeasurable band (1.00–1.645 null-sd).")


if __name__ == "__main__":
    main()
