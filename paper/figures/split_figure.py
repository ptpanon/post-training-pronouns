#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys, io, json
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator
from style import C, OUT, load, save, sha16, ROOT

IKINCI = "results/template_filtered_2026-09-07.json"
BIRINCI = "results/template_filtered_first_person_2026-09-15.json"
BANT_META = f"{OUT}/BANT.meta.json"
EN = {"Tulu3-8B": "T\u00fclu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}
YUK = 0.40
GEN = 5.5


def ad(a):
    return EN.get(a, a)


def sinif(d, lo_hi_key="ci"):
    lo, hi = d[lo_hi_key]
    return "up" if lo > 0 else "down" if hi < 0 else "null"


def main():
    K2, p2 = load(IKINCI)
    K1, p1 = load(BIRINCI)
    R = [r for r in K2["aileler"] if r["hal"] == "ÖLCÜLDÜ"]
    T1M = json.load(io.open(f"{OUT}/T1_families.meta.json", encoding="utf-8"))
    SIRA_T1 = T1M["sira"]
    assert sorted(SIRA_T1) == sorted(r["aile"] for r in R), (SIRA_T1, [r["aile"] for r in R])
    R.sort(key=lambda r: SIRA_T1.index(r["aile"]))
    S1 = {a["aile"]: a for a in K1["aileler"] if a["hal"] == "ÖLCÜLDÜ"}
    red_esdeger = sum(1 for r in R if not (
        S1[r["aile"]]["sbl"]["esdeger_card"] and S1[r["aile"]]["cip"]["esdeger_card"]
        and abs(S1[r["aile"]]["sbl"]["d2"] - r["sbl"]["dM1"]) < 1e-9
        and abs(S1[r["aile"]]["cip"]["d2"] - r["cip"]["dM1"]) < 1e-9
        and abs(S1[r["aile"]]["tutulan_oran"] - r["tutulan_oran"]) < 1e-9))
    PANEL = [
        ("(a) second person, raw continuation",
         {r["aile"]: (r["cip"]["M1_taban"], r["cip"]["M1_hizali"], sinif(r["cip"])) for r in R}),
        ("(b) second person, chat template",
         {r["aile"]: (r["sbl"]["M1_taban"], r["sbl"]["M1_hizali"], sinif(r["sbl"])) for r in R}),
        ("(c) first person, chat template",
         {r["aile"]: (S1[r["aile"]]["sbl"]["S1_taban"], S1[r["aile"]]["sbl"]["S1_hizali"],
                      sinif(S1[r["aile"]]["sbl"], "ci1")) for r in R}),
    ]
    def say(p):
        return {k: sorted(a for a, v in p.items() if v[2] == k) for k in ("up", "down", "null")}
    beklenen = [
        {"up": K2["sayim"]["ciplak"]["pozitif"], "down": K2["sayim"]["ciplak"]["negatif"],
         "null": K2["sayim"]["ciplak"]["null"]},
        {"up": K2["sayim"]["sablonlu"]["pozitif"], "down": K2["sayim"]["sablonlu"]["negatif"],
         "null": K2["sayim"]["sablonlu"]["null"]},
        {"up": K1["sayim"]["sbl_1"]["yukari"], "down": K1["sayim"]["sbl_1"]["asagi"],
         "null": K1["sayim"]["sbl_1"]["null"]},
    ]
    red_sayim = sum(1 for (e, p), b in zip(PANEL, beklenen)
                    for k in ("up", "down", "null") if say(p)[k] != sorted(b[k]))
    B = json.load(io.open(BANT_META, encoding="utf-8"))
    bant_kaynak_taze = any(s["yol"].endswith("template_filtered_2026-09-07.json")
                           and s["sha256_16"] == p2["sha256_16"] for s in B["sources"])
    tb = [v[0] for v in PANEL[1][1].values()]; hz = [v[1] for v in PANEL[1][1].values()]
    red_oran = int(abs(max(tb) / min(tb) - B["oran_taban"]) > 1e-9
                   or abs(max(hz) / min(hz) - B["oran_hizali"]) > 1e-9)
    tum = [x for _, p in PANEL for v in p.values() for x in v[:2]]
    print(f"{len(R)} {len(PANEL)}"
          f"{ {k: len(v) for k, v in say(PANEL[0][1]).items()} }"
          f"{ {k: len(v) for k, v in say(PANEL[1][1]).items()} }"
          f"{ {k: len(v) for k, v in say(PANEL[2][1]).items()} }"
          f"{red_sayim} {red_esdeger} {red_oran}"
          f"{bant_kaynak_taze} {min(tum):.2f}"
          f"")
    if red_sayim or red_esdeger or red_oran or not bant_kaynak_taze or min(tum) <= 0 or len(R) != 16:
        return 3

    SIRA = [r["aile"] for r in R]
    n = len(SIRA)
    BOS = 0.8
    yv = {a: (n - 1 - i) + 2 + BOS for i, a in enumerate(SIRA)}
    Y_TABAN, Y_HIZALI = 1.0, 0.0
    REN = {"up": C["pos"], "down": C["neg"], "null": C["null"]}
    XMIN, XMAX = 0.7, 60.0

    def ciz(ax, baslik, P, kip):
        for a in SIRA:
            t, h, s = P[a]; y = yv[a]
            if kip == "duzey":
                ax.scatter([h], [y], s=9, color=C["ink"], edgecolors=C["ink"], linewidths=0.7, zorder=5)
                continue
            ax.plot([t, h], [y, y], color=REN[s], lw=0.8, zorder=2,
                    ls=":" if s == "null" else "-")
            if s != "null":
                ax.annotate("", xy=(h, y), xytext=(t, y), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=REN[s], lw=0.8,
                                            mutation_scale=5, shrinkA=0, shrinkB=2.2))
            ax.scatter([t], [y], s=9, facecolors="white", edgecolors=C["ink"],
                       linewidths=0.7, zorder=4)
            ax.scatter([h], [y], s=9, color=REN[s], edgecolors=REN[s],
                       linewidths=0.7, zorder=5)
        tb_ = [P[a][0] for a in SIRA]; hz_ = [P[a][1] for a in SIRA]
        _bantlar = [((min(tb_), max(tb_)), Y_TABAN, False), ((min(hz_), max(hz_)), Y_HIZALI, True)]
        if kip == "duzey":
            _bantlar = _bantlar[1:]
        for (lo, hi), y, dolu in _bantlar:
            ax.add_patch(Rectangle((lo, y - 0.2), hi - lo, 0.4,
                                   facecolor=C["light"] if dolu else "white",
                                   edgecolor=C["ink"], lw=0.6, zorder=3))
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.32, y + 0.32], color=C["ink"], lw=0.7, zorder=4)
        if baslik.startswith("(b)") or "base to aligned" in baslik:
            for oran, y, hi in (((B["oran_hizali"], Y_HIZALI, max(hz_)),) if kip == "duzey" else
                                ((B["oran_taban"], Y_TABAN, max(tb_)), (B["oran_hizali"], Y_HIZALI, max(hz_)))):
                ax.text(hi * 1.12, y, r"$%.1f\times$" % oran, va="center",
                        ha="left", fontsize=5.5, color=C["ink"])
        gruplar = [(k, [a for a in SIRA if P[a][2] == k]) for k in ("up", "null", "down")]
        for k, uy in (gruplar if kip == "kayma" else []):
            if not uy:
                continue
            y0, y1 = min(yv[a] for a in uy), max(yv[a] for a in uy)
            xb = 1.02
            tr = ax.get_yaxis_transform()
            ax.plot([xb, xb + 0.025, xb + 0.025, xb], [y1 + 0.35, y1 + 0.35, y0 - 0.35, y0 - 0.35],
                    transform=tr, color=C["ink"], lw=0.5, clip_on=False)
            ax.text(xb + 0.05, (y0 + y1) / 2, "%d\n%s" % (len(uy), k), transform=tr,
                    va="center", ha="left", fontsize=5.2, linespacing=0.9,
                    color=C["ink"], clip_on=False)
        ax.set_title(baslik, fontsize=6.5, pad=2.5)
        ax.set_xscale("log"); ax.set_xlim(XMIN, XMAX)
        ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["1", "3", "10", "30"])
        ax.tick_params(axis="x", labelsize=5.5, length=2, pad=1.5)
        ax.set_ylim(-0.6, max(yv.values()) + 0.6)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="y", length=0)
    fig, axes = plt.subplots(1, 3, figsize=(GEN, GEN * YUK), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.155, right=0.955, top=0.875, bottom=0.135, wspace=0.30)
    KIP = [("(a) second person\nraw continuation", "kayma"),
           ("(b) second person, aligned\nown chat template", "duzey"),
           ("(c) first person\nchat template", "kayma")]
    for ax, (_, P), (baslik, kip) in zip(axes, PANEL, KIP):
        ciz(ax, baslik, P, kip)
    axes[1].set_xlabel("forms per 1,000 tokens (log scale)", fontsize=6, labelpad=1.5)
    ylab = [ad(a) for a in SIRA] + ["base range", "aligned range"]
    axes[0].set_yticks([yv[a] for a in SIRA] + [Y_TABAN, Y_HIZALI])
    axes[0].set_yticklabels(ylab, fontsize=5.5)
    save(fig, "F_split", [p2, p1, dict(yol="paper/figures/BANT.meta.json",
                                       sha256_16=sha16(BANT_META))],
         "Second-person and first-person density for the sixteen models, base "
         "(open) and aligned (filled), raw continuation and through each aligned checkpoint's "
         "own template, on the same filtered cells; one row order in all three "
         "panels, taken from Table 1 (T1_families.meta.json «sira»).")
    SIRA = [r["aile"] for r in sorted(R, key=lambda r: -r["sbl"]["dM1"])]
    yv = {a: (n - 1 - i) + 2 + BOS for i, a in enumerate(SIRA)}
    ylab = [ad(a) for a in SIRA] + ["base range", "aligned range"]
    fig, ax = plt.subplots(1, 1, figsize=(GEN * 0.46, GEN * YUK))
    fig.subplots_adjust(left=0.30, right=0.84, top=0.915, bottom=0.135)
    ciz(ax, "second person, chat template: base to aligned", PANEL[1][1], "kayma")
    ax.set_xlabel("forms per 1,000 tokens (log scale)", fontsize=6, labelpad=1.5)
    ax.set_yticks([yv[a] for a in SIRA] + [Y_TABAN, Y_HIZALI])
    ax.set_yticklabels(ylab, fontsize=5.5)
    save(fig, "F_split_delta", [p2, dict(yol="paper/figures/BANT.meta.json",
                                         sha256_16=sha16(BANT_META))],
         "Chat-template second-person density, base (open) to aligned (filled), sixteen models, "
         "same filtered cells as the split figure (fig:split), rows sorted by the shift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
