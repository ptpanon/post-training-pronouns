#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import numpy as np
import matplotlib.pyplot as plt
from style import C, COL, load, null_band, save, en

MERDIVEN = [
    ("OLMo-2-13B", "results/P7_OLCEK2_2026-08-21.json",
     ["OLMo2-13B·taban→SFT", "OLMo2-13B·SFT→DPO", "OLMo2-13B·DPO→Inst"],
     ["base→SFT", "SFT→DPO", "DPO→Instruct"]),
    ("OLMo-2-32B", "results/P7_OLCEK2_2026-08-21.json",
     ["OLMo2-32B·taban→SFT", "OLMo2-32B·SFT→DPO", "OLMo2-32B·DPO→Inst"],
     ["base→SFT", "SFT→DPO", "DPO→Instruct"]),
    ("Tulu-3-8B", "results/P_OLCEK3_TULU3_2026-08-22.json",
     ["Tulu3-8B·taban→SFT", "Tulu3-8B·SFT→DPO", "Tulu3-8B·DPO→RL"],
     ["base→SFT", "SFT→DPO", "DPO→RLVR"]),
    ("OLMo-2-7B  RLVR trajectory", "results/rlvr_olmo2_7b_2026-08-23.json",
     ["OLMo2-7B·DPO→STEP_60", "OLMo2-7B·STEP_60→STEP_120", "OLMo2-7B·STEP_120→STEP_180",
      "OLMo2-7B·STEP_180→STEP_240", "OLMo2-7B·STEP_240→STEP_300", "OLMo2-7B·STEP_300→STEP_360"],
     ["60", "120", "180", "240", "300", "360"]),
]


def main():
    kayit, veri = [], []
    onbellek = {}
    for ad, rel, uye, etik in MERDIVEN:
        if rel not in onbellek:
            d, prov = load(rel); onbellek[rel] = d; kayit.append(prov)
        P = onbellek[rel]["panel"]
        z, lo, hi, ayrik = [], [], [], []
        for u in uye:
            e = P[u]["delta"]["DOM"]; sd = e["dU_sd"]
            z.append(e["dU"] / sd); lo.append(e["dU_CI"][0] / sd); hi.append(e["dU_CI"][1] / sd)
            ayrik.append(bool(e.get("AYRIK_POZ") or e.get("AYRIK_NEG")))
        veri.append((ad, etik, np.array(z), np.array(lo), np.array(hi), ayrik))
    print(f"  [PAYDA] f2: n_merdiven={len(veri)} · n_basamak={sum(len(v[1]) for v in veri)} · red_eksik=0")

    w = [len(v[1]) for v in veri]
    fig, axes = plt.subplots(1, 4, figsize=(COL * 2.05, 2.15),
                             gridspec_kw=dict(width_ratios=w, wspace=0.14))
    ylim = max(max(abs(v[4]).max(), abs(v[3]).max()) for v in veri) * 1.12
    for ax, (ad, etik, z, lo, hi, ayrik) in zip(axes, veri):
        null_band(ax, horizontal=False)
        x = np.arange(len(z))
        for i in x:
            col = C["pos"] if z[i] > 0 else C["neg"]
            ax.plot([i, i], [lo[i], hi[i]], color=col if ayrik[i] else C["light"],
                    lw=1.6 if ayrik[i] else 0.9, solid_capstyle="butt", zorder=3)
            if ayrik[i]:
                ax.plot(i, z[i], "o", ms=4.2, color=col, mec="none", zorder=4)
            else:
                ax.plot(i, z[i], "o", ms=2.4, mfc="white", mec=C["null"], mew=0.7, zorder=4)
        ax.plot(x, z, color=C["null"], lw=0.55, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(etik, rotation=42, ha="right", fontsize=5.4)
        ax.set_xlim(-0.6, len(z) - 0.4); ax.set_ylim(-ylim, ylim)
        ax.set_title(ad, fontsize=6.3, pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("dominance ΔU  (null-sd units)", fontsize=6.6)
    axes[-1].set_xlabel("RLVR step", fontsize=6.0)
    fig.suptitle("Where the dominance shift is built differs by model — "
                 "and RLVR adds nothing", y=1.045, fontsize=7.6)
    save(fig, "F2_ladder", kayit,
         "Per-rung dominance shift: in OLMo-2-13B and Tulu-3-8B the shift is built at the "
         "SFT→DPO rung, whereas OLMo-2-32B moves the OPPOSITE way at base→SFT (−2.6 null-sd) "
         "and never recovers it — the model-specific signature — and six successive RLVR "
         "steps leave the axis flat.")


if __name__ == "__main__":
    main()
