#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys, io, json
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import matplotlib.pyplot as plt
from style import C, OUT, load
import style, urial_verdict_loader

HAVUZ = "results/four_seed_level_2026-09-16.json"
TEMPLATE = "results/template_filtered_2026-09-07.json"
EN = {"Tulu3-8B": "Tülu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}
YUK = 0.40
GEN = 5.5


def ad(a):
    return EN.get(a, a)


def sinif(ci):
    lo, hi = ci
    return "up" if lo > 0 else "down" if hi < 0 else "null"


def main():
    H, ph = load(HAVUZ)
    S, ps = load(TEMPLATE)
    U, pu = urial_verdict_loader.yukle(urial_verdict_loader.CARD_2048)
    SIRA = json.load(io.open(f"{OUT}/T1_families.meta.json", encoding="utf-8"))["sira"]
    Hd = {k: v for k, v in H.items() if not k.startswith("_") and v.get("hal") == "m=4"}
    Sd = {r["aile"]: r for r in S["aileler"] if r["hal"] == "ÖLCÜLDÜ"}
    Ua, Us, Uk = urial_verdict_loader.kiyas(U, "i")
    Ud = {a: x for a, x in Ua.items() if x.get("hal") == "ÖLCÜLDÜ"}
    assert sorted(SIRA) == sorted(Hd) == sorted(Sd) == sorted(Ud), ""
    PANEL = [
        ("(a) raw continuation", {a: (Hd[a]["havuz_dM1"], Hd[a]["ci"]) for a in SIRA}),
        ("(b) own chat template", {a: (Sd[a]["sbl"]["dM1"], Sd[a]["sbl"]["ci"]) for a in SIRA}),
        ("(c) in-context assistant prompt", {a: (Ud[a]["dM1"], Ud[a]["ci"]) for a in SIRA}),
    ]
    say = [{k: sorted(a for a, (_, ci) in P.items() if sinif(ci) == k) for k in ("down", "up", "null")} for _, P in PANEL]
    beklenen = [
        dict(down=sorted(a for a in SIRA if Hd[a]["ayrik"] and Hd[a]["havuz_dM1"] < 0),
             up=sorted(a for a in SIRA if Hd[a]["ayrik"] and Hd[a]["havuz_dM1"] > 0),
             null=sorted(a for a in SIRA if not Hd[a]["ayrik"])),
        dict(down=sorted(S["sayim"]["sablonlu"]["negatif"]), up=sorted(S["sayim"]["sablonlu"]["pozitif"]),
             null=sorted(S["sayim"]["sablonlu"]["null"])),
        None,
    ]
    red_c = int(len(say[2]["down"]) != Us["asagi"] or len(say[2]["up"]) != Us["ters"] or Us["bicim"] != 0
                or Us["olculemez_kirpma"] != 0 or len(say[2]["null"]) != 0)
    red_sayim = sum(1 for i in (0, 1) for k in ("down", "up", "null") if say[i][k] != beklenen[i][k]) + red_c
    print(f"  [PAYDA] f_effects: n_aile={len(SIRA)} · n_panel={len(PANEL)} · "
          + " · ".join(f"{b.split(')')[0]})={ {k: len(v) for k, v in s.items()} }" for (b, _), s in zip(PANEL, say))
          + f"{red_sayim}")
    if red_sayim or len(SIRA) != 16:
        return 3

    n = len(SIRA)
    yv = {a: n - 1 - i for i, a in enumerate(SIRA)}
    REN = {"up": C["pos"], "down": C["neg"], "null": C["null"]}
    uc = [x for _, P in PANEL for _, (lo, hi) in P.values() for x in (lo, hi)]
    XLIM = (min(uc) - 1.5, max(uc) + 1.5)
    print(f"  [PAYDA] f_effects eksen: uc {min(uc):.2f} … {max(uc):.2f} ⇒ ortak xlim {XLIM[0]:.1f} … {XLIM[1]:.1f}")
    fig, axes = plt.subplots(1, 3, figsize=(GEN, GEN * YUK), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.875, bottom=0.135, wspace=0.12)
    for ax, (baslik, P), s in zip(axes, PANEL, say):
        for a in SIRA:
            d, (lo, hi) = P[a]; k = sinif((lo, hi)); y = yv[a]
            ax.plot([lo, hi], [y, y], color=REN[k], lw=0.9, ls=":" if k == "null" else "-", zorder=2)
            ax.scatter([d], [y], s=9, zorder=3, linewidths=0.7, edgecolors=REN[k],
                       facecolors="white" if k == "null" else REN[k])
        ax.axvline(0, color=C["ink"], lw=0.5, zorder=1)
        etiket = ", ".join(f"{len(s[k])} {k}" for k in ("down", "up", "null") if s[k])
        ax.set_title(f"{baslik}\n{etiket}", fontsize=6.5, pad=2.5)
        ax.set_xlim(XLIM)
        ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
        ax.tick_params(axis="x", labelsize=5.5, length=2, pad=1.5)
        ax.set_ylim(-0.7, n - 0.3)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlabel("aligned minus base, second-person forms per 1,000 tokens", fontsize=6, labelpad=1.5)
    axes[0].set_yticks([yv[a] for a in SIRA])
    axes[0].set_yticklabels([ad(a) for a in SIRA], fontsize=5.5)
    style.save(fig, "F_effects", [ph, ps, pu],
               "Second-person change, aligned minus base, with prompt-clustered 95% intervals, sixteen models, under "
               "three prompt formats: (a) raw continuation, pooled seeds, all cells (Table 1); (b) own chat template, "
               "filtered cells, first seed (Figure A1, Table A5); (c) in-context assistant prompt vs own template, "
               "2,048-token window (Table A5, task). One shared x-axis. Rows follow Table 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
