#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import COL, C, load, save, sha16, ROOT, null_band, en
sys.path.insert(0, f"{ROOT}/scripts")

ALTI = ["Qwen2.5-7B", "Mistral-7B-v0.3", "OLMo-3-7B", "Llama-3.1-8B",
        "Gemma-3-4B", "Gemma-3-12B"]
EK10 = ["Qwen2.5-1.5B", "Zephyr-7B", "Falcon-H1-7B", "Tulu-3-8B", "Granite-4.1-8B",
        "OLMo-2-13B", "Qwen2.5-14B", "Mistral-Small-24B", "OLMo-2-32B", "Gemma-3-27B"]
A16 = ALTI + EK10
KANAL = [("c_KE", "interpersonal\ncommand"), ("c_YE", "procedural\ncommand"),
         ("c_DB", "dominant\nassertion")]


def f1_reading_map():
    d, kay = load("results/shadow_reading_16_scorecard_2026-08-25.json")
    SUB = [("SCOTUS (OKUMA-B)", "SCOTUS oral arguments"), ("cga-cmv", "Reddit CMV")]
    M = np.full((len(A16), len(SUB) * 3), np.nan)
    for si, (s, _) in enumerate(SUB):
        H = d["substrat"][s]["hucre"]
        for ai, a in enumerate(A16):
            for ki, (k, _) in enumerate(KANAL):
                v = H.get(a, {}).get("sayilar", {}).get(k, {})
                if "sd_kati" in v:
                    M[ai, si * 3 + ki] = v["sd_kati"]
    fig, ax = plt.subplots(figsize=(COL * 1.75, 3.5))
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-3, vmax=3, aspect="auto")
    for j in range(M.shape[1]):
        for i in range(M.shape[0]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=4.6,
                        color=("white" if abs(M[i, j]) > 1.9 else C["ink"]))
    ax.axvline(2.5, color=C["ink"], lw=1.1)
    ax.axhline(len(ALTI) - 0.5, color=C["ink"], lw=1.1, ls="--")
    ax.set_xticks(range(6))
    ax.set_xticklabels([k[1] for k in KANAL] * 2, fontsize=5.4)
    ax.set_yticks(range(len(A16))); ax.set_yticklabels(A16, fontsize=5.8)
    ax.set_title(f"{SUB[0][1]}                    {SUB[1][1]}", fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=.55, pad=.02); cb.set_label("null-sd", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.text(5.62, len(ALTI) - 0.62, "first six", fontsize=5, ha="right",
            va="bottom", color=C["null"])
    ax.text(5.62, len(ALTI) - 0.38, "later ten", fontsize=5, ha="right",
            va="top", color=C["null"])
    print(f"  [PAYDA] f1: n_aile={len(A16)} · n_substrat={len(SUB)} · n_kanal={len(KANAL)} "
          f"· hal_dolu={int(np.isfinite(M).sum())}/{M.size}")
    save(fig, "F1_reading_map", [kay],
         "Context-shadow effect sizes in null-sd for sixteen base->aligned pairs on two "
         "substrates and three context classes; the dashed rule separates the six models "
         "on which the effect was first established from the ten added afterwards, and the "
         "solid rule separates the substrates.")


def f2_mood_ruler():
    kay = []
    NLL = {}
    for rel in ("results/plainness_2026-08-27.json",
                "results/exploratory_mood16_2026-08-28.json"):
        d, k = load(rel); kay.append(k)
        for a, v in d.get("aile", {}).items():
            if isinstance(v.get("ort"), dict):
                NLL[a] = (v["ort"]["kisa"], v["ort"]["uzun"])
    import mood_swap_ruler_figure as F4
    esit, ters, duz, yok, n_kaps = F4.cetvel_sayimi()
    kay.append(dict(yol=__DNH_DATA__ + "/c1_kip_takas/kip_takas.jsonl",
                    sha256_16="(veri: repo disi, cite-only)",
                    rol="98 cross-tier pairs — mood_swap_ruler_figure.cetvel_sayimi()"))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL * 2.1, 2.3),
                                   gridspec_kw=dict(wspace=0.42, width_ratios=[1.45, 1]))
    a_sirali = sorted(NLL, key=lambda a: NLL[a][0])
    x = np.array([NLL[a][0] for a in a_sirali]); y = np.array([NLL[a][1] for a in a_sirali])
    axL.axhline(0, color=C["null"], lw=.6); axL.axvline(0, color=C["null"], lw=.6)
    for ad, kum, ren, mrk in (("first six", [a for a in a_sirali if a in ALTI], C["pos"], "o"),
                              ("later ten", [a for a in a_sirali if a not in ALTI], C["neg"], "^")):
        ix = [a_sirali.index(a) for a in kum]
        axL.scatter(x[ix], y[ix], s=28, c=ren, marker=mrk,
                    edgecolor="k", linewidth=.4, zorder=3, label=ad)
    for a, xi, yi in zip(a_sirali, x, y):
        axL.annotate(a, (xi, yi), fontsize=4.4, xytext=(3, 2), textcoords="offset points")
    rho = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    axL.set_xlabel("mood effect, short frame  (ΔNLL nats/token)")
    axL.set_ylabel("mood effect, long frame")
    axL.set_title(f"mood swap, {len(NLL)} models   (Spearman ρ = {rho:+.2f})", fontsize=7)
    axL.legend(loc="upper left", fontsize=5.4)
    pay = np.array([esit, ters, duz]) / n_kaps
    axR.bar(range(3), pay, color=[C["null"], C["neg"], C["pos"]], width=.62,
            edgecolor="k", linewidth=.4)
    for i, p in enumerate(pay):
        axR.text(i, p + .02, f"{p:.3f}", ha="center", fontsize=6)
    axR.set_xticks(range(3))
    axR.set_xticklabels(["silent\n(no lexicon\ndifference)", "REVERSED\n(request scores\n"
                         "higher)", "correct\ndirection"], fontsize=5.4)
    axR.set_ylabel("share of covered pairs"); axR.set_ylim(0, .78)
    axR.set_title(f"lexicon ruler vs. grammatical ground truth\n"
                  f"{n_kaps} of 98 cross-tier pairs ({yok} had no lexicon word)", fontsize=6.4)
    print(f"  [PAYDA] f2: n_aile={len(NLL)} · rho_kisa_uzun={rho:.3f} · "
          f"hal_esit={esit} · hal_ters={ters} · hal_duz={duz} · n_kapsanan={n_kaps}")
    save(fig, "F2_mood_ruler", kay,
         f"Left: the grammatical mood swap moves teacher-forced NLL in the same direction "
         f"under a short and a long frame across {len(NLL)} models (Spearman rho = {rho:+.2f}), "
         f"so the effect is not an artefact of frame length. Right: on the same swap, where "
         f"the ground truth is grammatical rather than judged, the Warriner dominance lexicon "
         f"is silent on {esit}/{n_kaps} cross-tier pairs and, on {ters}/{ters+duz} of the "
         f"pairs where it does move, ranks the polite request as the MORE dominant text --- "
         f"the cheap ruler does not merely lack power here, it points the wrong way.")


def f3_rungs():
    from matplotlib.patches import Patch
    fs, k1 = load("results/form_count_2026-08-28.json")
    ms, k2 = load("results/ladder_template_2026-08-27.json")
    MER = [("OLMo2-13B", "OLMo-2-13B"), ("OLMo2-32B", "OLMo-2-32B"), ("Tulu3-8B", "Tülu-3-8B")]
    SAY = [("a", "bare\nimperative"), ("c", "hedge"), ("d", "interrog.\ndirective"),
           ("e", "hortative")]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL * 2.15, 2.55),
                                   gridspec_kw=dict(wspace=0.30, width_ratios=[1.75, 1]))
    w, ADIM, n_bag = 0.34, len(SAY) + 1.7, 0
    xs, xlab, bagimli = [], [], []
    for gi, (g, ad) in enumerate(MER):
        h = fs["merdiven"][g]
        for si, (sc, et) in enumerate(SAY):
            x0 = gi * ADIM + si
            xs.append(x0); xlab.append(et)
            for pi, pro in enumerate(("ciplak", "template")):
                T = h[pro][sc]["TAM"]
                axL.bar(x0 + (pi - .5) * w, T["delta"], width=w * .9,
                        color=(C["pos"] if pi == 0 else C["accent"]),
                        edgecolor="k", linewidth=.35,
                        alpha=(1.0 if T["frac2"] <= .05 else .32), zorder=3)
            if h["_cift"][sc] == "PROTOKOLE-BAGIMLI":
                bagimli.append(x0); n_bag += 1
    axL.axhline(0, color=C["ink"], lw=.6)
    lo, hi = axL.get_ylim(); pay = (hi - lo) * 0.10
    axL.set_ylim(lo - pay * 0.4, hi + pay * 2.6)
    for x0 in bagimli:
        axL.text(x0, hi + pay * 0.35, "▲", ha="center", va="bottom", fontsize=7,
                 color=C["neg"])
    for gi, (_, ad) in enumerate(MER):
        axL.text(gi * ADIM + (len(SAY) - 1) / 2, hi + pay * 1.75, ad, ha="center",
                 va="bottom", fontsize=6.4, color=C["ink"])
        if gi:
            axL.axvline(gi * ADIM - 1.35, color=C["light"], lw=.5, ls=":")
    axL.set_xticks(xs)
    axL.set_xticklabels([t.replace(chr(10), " ") for t in xlab], rotation=40,
                        ha="right", fontsize=5.0)
    axL.tick_params(axis="x", pad=1)
    axL.set_ylabel("Δ form rate at SFT→DPO   (counts / sentence)", fontsize=6.6)
    axL.legend(handles=[Patch(fc=C["pos"], ec="k", lw=.35, label="bare protocol"),
                        Patch(fc=C["accent"], ec="k", lw=.35, label="chat template"),
                        Patch(fc="w", ec="w", label="▲ protocol-dependent")],
               loc="lower right", fontsize=5.2, ncol=1)
    axL.set_title("form counters, both protocols   (faded = inside the bar)", fontsize=6.8)
    B = {k: v for k, v in ms["basamak"].items() if "→DPO" in k}
    ad = list(B); yy = np.arange(len(ad))
    TR = dict(MER)
    axR.barh(yy - .19, [B[k]["dU_ciplak"] for k in ad], height=.36, color=C["pos"],
             edgecolor="k", linewidth=.35)
    axR.barh(yy + .19, [B[k]["dU_sablonlu"] for k in ad], height=.36, color=C["accent"],
             edgecolor="k", linewidth=.35)
    axR.axvline(0, color=C["ink"], lw=.6)
    axR.set_yticks(yy)
    axR.set_yticklabels([TR.get(k.split("·")[0], k.split("·")[0]) for k in ad], fontsize=5.8)
    axR.set_xlabel("ΔU  (dominance, embedding estimator)", fontsize=6.4)
    axR.set_title("same rungs, pre-specified estimator", fontsize=6.8)
    axR.tick_params(axis="x", labelsize=5.6)
    print(f"  [PAYDA] f3: n_merdiven={len(MER)} · n_sayac={len(SAY)} · "
          f"hal_protokole_bagimli={n_bag} · n_dU_basamak={len(ad)}")
    save(fig, "F3_rungs", [k1, k2],
         f"The SFT→DPO rung read twice --- under the bare prompt and under each model's own "
         f"chat template. Solid bars clear the two-sided empirical bar, faded bars do not, and "
         f"a triangle marks the {n_bag} counters whose verdict changes name between protocols. "
         f"Right: the sealed embedding estimator on the same rungs. The chat template is not a "
         f"nuisance parameter.")


def f4_dpo_delta():
    ETI = {"zephyr": ("Zephyr-7B (dSFT→dDPO)", C["accent"]),
           "tulu3": ("Tülu-3-8B (SFT→DPO)", C["pos"]),
           "olmo": ("OLMo-2-13B (SFT→DPO)", C["neg"]),
           "olmo32": ("OLMo-2-32B (SFT→DPO)", "#5a5a5a")}
    import glob
    fig, ax = plt.subplots(figsize=(COL * 1.55, 2.35))
    kay, n = [], 0
    for y in sorted(glob.glob(f"{ROOT}/results/D_DELTA_*.json")):
        rel = os.path.relpath(y, ROOT); d, k = load(rel)
        h = d.get("hucre")
        if h not in ETI:
            continue
        kay.append(k); n += 1
        a = np.array([float(x) for x in d["T"]]); t = np.array(list(d["T"].values()), float)
        i = np.argsort(a); a, t = a[i], t[i]
        t0 = t[list(a).index(0.0)]
        ad, col = ETI[h]
        ax.plot(a, t - t0, "o-", color=col, lw=1.2, ms=3.4,
                label=f"{ad} — {en(d['AD'])}")
        ax.axhline(d["T_taban"] - t0, color=col, lw=.7, ls="--", alpha=.55, zorder=2)
        ax.scatter([-0.72], [d["T_taban"] - t0], marker="*", s=46, color=col,
                   edgecolor="k", linewidth=.4, zorder=4, clip_on=False)
    ax.axhline(0, color=C["null"], lw=.6); ax.axvline(0, color=C["null"], lw=.6, ls=":")
    ax.axvline(1, color=C["null"], lw=.6, ls=":")
    ax.set_xlabel("α   (0 = SFT, 1 = DPO)")
    ax.set_ylabel("T(α) − T(SFT),   T = NLL(order) − NLL(request)", fontsize=6.4)
    ax.set_title("undoing the preference delta\n(dashed line + ★ = pre-trained base)",
                 fontsize=6.8)
    ax.set_xlim(-0.78, 1.62)
    ax.legend(fontsize=4.9, loc="lower left")
    print(f"  [PAYDA] f4d: n_merdiven={n} · bekle=4")
    save(fig, "F4_dpo_delta", kay,
         "Interpolating the preference delta θ(α)=θ_SFT+α(θ_DPO−θ_SFT) moves the mood "
         "statistic monotonically in all four ladders (ρ = −1.000 each), yet only OLMo-2-13B "
         "returns toward its pre-trained base (star) when the delta is undone; at 32B the "
         "same lab, recipe and grid give a curve that moves away from base --- the direction "
         "is shared, the return is not, and it is scale that breaks it.")


def f5_map():
    import map_carpistir as HC, family_profile_map as HP
    R = HP.topla()
    M, A, E = HC.matris(R)
    B, Ab, Eb, at_a, at_e = HC.tam_blok(M, A, E)
    Z = HC.z(B)
    from sklearn.decomposition import PCA
    P = PCA(n_components=2).fit(Z); Y = P.transform(Z)
    RENK = {"RLVR": C["neg"], "DPO": C["pos"], "RLHF": "#2a9d8f",
            "SFT/tercih": "#e9c46a", "tarif-yok": "#999999"}
    EN = {"SFT/tercih": "SFT/preference", "tarif-yok": "recipe not recorded"}
    rc = R["recete"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL * 2.15, 2.6),
                                   gridspec_kw=dict(wspace=.5, width_ratios=[1.25, 1]))
    Zn = np.full_like(M, np.nan)
    for j in range(M.shape[1]):
        v = M[:, j]; ok = ~np.isnan(v)
        if ok.sum() >= 2 and v[ok].std() > 0:
            Zn[ok, j] = (v[ok] - v[ok].mean()) / v[ok].std()
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#dddddd")
    axL.imshow(np.ma.masked_invalid(Zn), cmap=cmap, vmin=-2.2, vmax=2.2, aspect="auto")
    axL.set_xticks(range(len(E))); axL.set_xticklabels(E, rotation=68, ha="right", fontsize=3.9)
    axL.set_yticks(range(len(A)))
    axL.set_yticklabels(A, fontsize=4.8)
    for i, a in enumerate(A):
        axL.get_yticklabels()[i].set_color(RENK.get(rc[a], "#333"))
    axL.set_title("profile matrix (within-axis z; grey = not measured)", fontsize=6.4)
    for i, a in enumerate(Ab):
        axR.scatter(Y[i, 0], Y[i, 1], s=42, color=RENK.get(rc[a], "#999"),
                    edgecolor="k", linewidth=.4, zorder=3)
        axR.annotate(a, (Y[i, 0], Y[i, 1]), fontsize=4.6, xytext=(3, 2),
                     textcoords="offset points")
    axR.axhline(0, color=C["light"], lw=.6); axR.axvline(0, color=C["light"], lw=.6)
    axR.set_xlabel(f"PC1 ({P.explained_variance_ratio_[0]*100:.0f}%)")
    axR.set_ylabel(f"PC2 ({P.explained_variance_ratio_[1]*100:.0f}%)")
    axR.set_title(f"complete block {len(Ab)}×{len(Eb)}, colour = recipe", fontsize=6.4)
    for k, c in RENK.items():
        if any(rc[a] == k for a in Ab):
            axR.scatter([], [], color=c, edgecolor="k", linewidth=.4,
                        label=EN.get(k, k))
    axR.legend(fontsize=4.8, loc="best")
    kay = [dict(yol=y, sha256_16=v) for y, v in R["kaynak_sha"].items()]
    print(f"  [PAYDA] f5: n_aile={len(A)} · n_eksen={len(E)} · "
          f"n_blok={len(Ab)}x{len(Eb)} · red_elenen_aile={len(at_a)}")
    save(fig, "F5_map", kay,
         f"EXPLORATORY (no bar, no verdict name): sixteen models against twenty measured "
         f"axes, and the {len(Ab)}x{len(Eb)} complete block projected onto its first two "
         f"principal components. Empty cells are left empty --- nothing is imputed. Colour "
         f"is the training recipe recorded in our own panel, and Tülu-3-8B, whose weights "
         f"descend from Llama-3.1-8B, sits with the other AI2 models rather than with Llama.")


if __name__ == "__main__":
    for fn in (f1_reading_map, f2_mood_ruler, f3_rungs, f4_dpo_delta, f5_map):
        print(f"── {fn.__name__}")
        fn()
