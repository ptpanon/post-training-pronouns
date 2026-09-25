#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load, t1_adi

SEY = "results/dilution_2026-09-03.json"
KUV = "results/v22_force_dilution_2026-09-06.json"
KUM = "results/v22_force_dilution_ladder_2026-09-06.json"


def _f(x, n=3):
    return f"${x:+.{n}f}$" if x is not None else "---"


def _s(nokta, ci, ayrik):
    if not ayrik or nokta is None or ci is None:
        return "---", "---"
    return f"${nokta:+.2f}$", f"$[{ci[0]:+.2f},{ci[1]:+.2f}]$"


def main():
    S, ps = load(SEY); K, pk = load(KUV); M, pm = load(KUM)
    if not K["_kunye"].get("esdegerlik_gecti"):
        print("  ★★ ESDEGERLIK SINAVI GECMEMIS ⇒ tablo basilmaz (§8)"); return 3
    sat = []
    for a, v in sorted(S["_aile"].items()):
        k = K["aile"].get(a)
        sp = v["seyrelme_payi"]
        m1_ayrik = v["ci_M1_1k"][0] * v["ci_M1_1k"][1] > 0
        sat.append(dict(ad=a, blok=0, m1=v["dlog_M1_1k"], m1c=v["dlog_M1_cumle"],
                        uz=v["dlog_len"], sM1=sp.get("nokta"), sM1ci=sp.get("ci"),
                        m1_ayrik=m1_ayrik,
                        kv=(k or {}).get("dlog", {}).get("kuv_1k"),
                        kvc=(k or {}).get("dlog", {}).get("kuv_c"),
                        sK=(k or {}).get("s_kuvvet"),
                        sKci=((k or {}).get("ci") or {}).get("s_kuv"),
                        k_ayrik=bool((k or {}).get("kuv_1k_ayrik"))))
    for a, adimlar in S["_merdiven"].items():
        for ad, v in adimlar.items():
            k = (M["merdiven"].get(a) or {}).get(ad)
            sp = v["seyrelme_payi"]
            _ad = a + " " + ad.replace("→", " $\\to$ ")
            sat.append(dict(ad=_ad, blok=1,
                            m1=v["dlog_M1_1k"], m1c=v["dlog_M1_cumle"],
                            uz=v["dlog_len"], sM1=sp.get("nokta"), sM1ci=sp.get("ci"),
                            m1_ayrik=v["ci_M1_1k"][0] * v["ci_M1_1k"][1] > 0,
                            kv=(k or {}).get("dlog", {}).get("kuv_1k"),
                            kvc=(k or {}).get("dlog", {}).get("kuv_c"),
                            sK=(k or {}).get("s_kuvvet"),
                            sKci=((k or {}).get("ci") or {}).get("s_kuv"),
                            k_ayrik=bool((k or {}).get("ayrik"))))
    nA = sum(1 for r in sat if r["blok"] == 0)
    kA = sum(1 for r in sat if r["blok"] == 0 and r["k_ayrik"])
    kR = sum(1 for r in sat if r["blok"] == 1 and r["k_ayrik"])
    yuk = sum(1 for r in sat if r["blok"] == 0 and r["k_ayrik"] and r["kvc"] > 0)
    print(f"{nA} {len(sat)-nA}"
          f"{kA} {kR} {yuk}"
          f"")
    if nA != 16:
        print(""); return 3
    L = [r"\begin{table}[tb]\centering\scriptsize\setlength{\tabcolsep}{2pt}",
         r"\caption{\textbf{Dilution share, for both layers.} Columns are the log "
         r"change per thousand tokens, the log change per sentence, the log change in "
         r"tokens per sentence, and the dilution share "
         r"$s=\Delta\log(\text{tokens per sentence})/\Delta\log(X/\mathrm{1k})$ with "
         r"its prompt-clustered $95\%$ interval, first for second-person density and "
         r"then for the force total. The same identity governs both, so the "
         r"per-sentence change is $(1+s)$ times the per-thousand change in each half. "
         r"$\Delta\log\mathrm{len}$ is shared. Negative $s$ means the per-sentence "
         r"fall is smaller than the per-thousand fall. Where the per-thousand change's "
         r"interval includes zero the ratio has no stable sign and no number is "
         f"printed. Rows below the rule are the three preference checkpoint sequences, stage by stage. "
         f"\\textbf{{Force dilutes harder than register}}: the force total separates per "
         f"thousand tokens in {kA} of {nA} models, but in {yuk} of those the "
         f"per-sentence count moves \\emph{{up}} while the per-thousand count falls, "
         f"which is the sentence growing rather than the acts thinning. On the stages "
         f"only {kR} of {len(sat)-nA} stages separate on force at all."
         r"}",
         r"\label{tab:dilution}",
         r"\begin{tabular}{l" + "rrr" + "rl" + "c" + "rr" + "rl}", r"\toprule",
         r" & \multicolumn{5}{c}{second person ($M_1$)} & &"
         r" \multicolumn{4}{c}{force total} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){8-11}",
         r"model / stage & $/\mathrm{1k}$ & $/\mathrm{sent}$ & $\Delta\log\mathrm{len}$"
         r" & $s$ & $95\%$ CI & & $/\mathrm{1k}$ & $/\mathrm{sent}$ & $s$ & $95\%$ CI \\",
         r"\midrule"]
    for i, r in enumerate(sat):
        if r["blok"] == 1 and sat[i - 1]["blok"] == 0:
            L.append(r"\midrule")
        a, b = _s(r["sM1"], r["sM1ci"], r["m1_ayrik"])
        c, d = _s(r["sK"], r["sKci"], r["k_ayrik"])
        L.append(f"{t1_adi(r['ad'])} & {_f(r['m1'])} & {_f(r['m1c'])} & "
                 f"{_f(r['uz'])} & {a} & {b} & & {_f(r['kv'])} & {_f(r['kvc'])} & "
                 f"{c} & {d} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T9_dilution.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    import json
    json.dump(dict(table="T9_dilution", sources=[ps, pk, pm],
                   payda=dict(n_aile=nA, n_basamak=len(sat) - nA,
                              kuvvet_ayrik_aile=kA, kuvvet_ayrik_basamak=kR,
                              cumle_cetvelinde_yukari=yuk),
                   note="renders measured quantities; computes no new statistic"),
              open(f"{OUT}/T9_dilution.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ✓ T9_dilution.tex ({len(L)} satir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
