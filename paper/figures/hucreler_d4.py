#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

R = __DNH_ROOT__ + ""
B1K, V91K = "results/siradan_instruction_reading_2026-09-17.json", "results/v91_konusan_sayimi_2026-09-17.json"
Q4K, K3K = "results/v96_q4_urial_ozdes_2026-09-19.json", "results/v96_k3_impersonal_ozdes_2026-09-19.json"
R2K = "results/v97_d4_kod_citi_2026-09-19.json"
R1K = "results/v98_cpu_readings_2026-09-21.json"
AD = {"Tulu3-8B": r"T\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}
S3 = {"asagi": "d", "yukari": "u", "null": "n", "asagi": "d", "yukari": "u"}


def ad(a): return AD.get(a, a)
def sayim(xs): return {c: sum(1 for x in xs if x == c) for c in "dun"}


def hucre(v, s, nd=2):
    t = f"{v:+.{nd}f}"
    return rf"$\mathbf{{{t}}}$" if s != "n" else f"${t}$"


def main():
    for k in (B1K, V91K, Q4K, K3K):
        if not os.path.exists(f"{R}/{k}"):
            print(f"{k}"); return 3
    B, pb = load(B1K); V, pv = load(V91K); Q, pq = load(Q4K); K, pk = load(K3K)
    ok = (B["n_okunan"] == 16 and not B["bacak_eksik"] and V["esdeger"]["hal"] == "ESDEGER" and Q["alet"]["hal"] == "ESDEGER"
          and not Q["bacak_eksik"]["K1"] and not Q["bacak_eksik"]["K2"] and K["alet"]["hal"] == "ESDEGER" and not K["bacak_eksik"])
    aile = sorted(B["aile"])
    if not ok or len(aile) != 16 or any(a not in Q["aile"] or a not in K["aile"] for a in aile):
        print(""); return 3
    sat = {}
    for a in aile:
        b, v, q, k = B["aile"][a]["M1"], V["aile"][a]["IS1"], Q["aile"][a], K["aile"][a]
        sat[a] = dict(
            B1=(b["gozlenen"], S3[B["aile"][a]["M1"]["yon"]] if b["ayrik"] else "n", b.get("plasebo_ustu"),
                v["gozlenen"], S3[V["aile"][a]["IS1_sinif3"]], v.get("plasebo_ustu")),
            K1=(q["K1"]["M1"]["gozlenen"], S3[q["K1"]["M1_sinif3"]], q["K1"]["M1"].get("plasebo_ustu"),
                q["K1"]["IS1"]["gozlenen"], S3[q["K1"]["IS1_sinif3"]], q["K1"]["IS1"].get("plasebo_ustu")),
            K2=(q["K2"]["M1"]["gozlenen"], S3[q["K2"]["M1_sinif3"]], q["K2"]["M1"].get("plasebo_ustu"),
                q["K2"]["IS1"]["gozlenen"], S3[q["K2"]["IS1_sinif3"]], q["K2"]["IS1"].get("plasebo_ustu")),
            K3=(k["M1_1k"]["gozlenen"], S3[k["M1_1k_sinif3"]], k["M1_1k"].get("plasebo_ustu"),
                k["IS1_1k"]["gozlenen"], S3[k["IS1_1k_sinif3"]], k["IS1_1k"].get("plasebo_ustu")),
            D=[(k[m]["gozlenen"], S3[k[f"{m}_sinif3"]]) for m in ("duz_M1_1k", "duz_IS1_1k", "duz_M1_cumle", "duz_IS1_cumle")],
            at=(k["atilan_satir_payi"]["taban"], k["atilan_satir_payi"]["hizali"]))
    say = {h: (sayim([sat[a][h][1] for a in aile]), sayim([sat[a][h][4] for a in aile]),
               sum(1 for a in aile if sat[a][h][1] == "d" and sat[a][h][2]), sum(1 for a in aile if sat[a][h][4] == "d" and sat[a][h][5]))
           for h in ("B1", "K1", "K2", "K3")}
    sayD = [sayim([sat[a]["D"][j][1] for a in aile]) for j in range(4)]
    at_t, at_h = float(np.median([sat[a]["at"][0] for a in aile])), float(np.median([sat[a]["at"][1] for a in aile]))
    oran = B["bant"]["hizali"]["oran"]
    b1m, b1i, k2m, k2i = say["B1"][0]["d"], say["B1"][1]["d"], say["K2"][0]["d"], say["K2"][1]["d"]
    if (k2m, k2i) != (b1m, b1i):
        print(f"{k2m} {k2i} {b1m} {b1i}"); return 3
    hep = lambda n: "all 16" if n == 16 else f"{n} of 16"
    cumle = (rf"On $300$ ordinary instructions, each checkpoint in its own format, ``you'' falls in {hep(b1m)} and ``I'' in ${b1i}$, and the "
             rf"aligned range of ``you'' is ${oran:.1f}$-fold. With the base on a person-stripped in-context prompt, ``you'' again falls in {hep(k2m)} and ``I'' in ${k2i}$. On an identical "
             rf"string the aligned model reads a format it was not trained on and the sign of ``you'' splits (${say['K1'][0]['d']}$ down on "
             rf"the released in-context prompt; ${say['K3'][0]['d']}$ down, ${say['K3'][0]['u']}$ up on the person-stripped one) while ``I'' falls in "
             rf"${say['K1'][1]['d']}$ and ${say['K3'][1]['d']}$ (Appendix~\ref{{app:D4}})")
    io.open(f"{OUT}/HUCRELER_CUMLE.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/hucreler_d4.py)\n" + cumle)
    r2 = ""; pr2 = None
    if os.path.exists(f"{R}/{R2K}"):
        C, pr2 = load(R2K)
        if C["alet"]["hal"] == "ESDEGER" and len(C["aile"]) == 16:
            sb = C["sayim"]["b"]; kes = float(np.median([v["b"]["hizali_kesilen_pay"] for v in C["aile"].values()]))
            kmax = max(C["aile"].items(), key=lambda kv: kv[1]["b"]["hizali_kesilen_pay"])
            r2 = (rf" \textbf{{Cutting the aligned outputs at their first code fence as well}}, as the base outputs are cut, changes at most "
                  rf"${100*kmax[1]['b']['hizali_kesilen_pay']:.1f}\%$ of aligned rows ({ad(kmax[0])}), and ``you'' still falls in "
                  rf"${sb['M1']['asagi']}$ of $16$ models and ``I'' in ${sb['IS1']['asagi']}$.")
    kd = lambda h, j: f"{say[h][j]['d']}/{say[h][j]['u']}/{say[h][j]['n']}"
    DZ, dz_say = {}, {}
    if os.path.exists(f"{R}/{R1K}"):
        C1, pr1 = load(R1K)
        O = C1["okumalar"].get("R1_Q1_duzyazi_B1_K2", {})
        if O.get("alet", {}).get("hal") == "ESDEGER":
            for a in aile:
                v = O["aile"].get(a, {})
                DZ[a] = {h: [(v[h][m]["gozlenen"], S3[v[h][f"{m}_sinif3"]])
                             for m in ("duz_M1_1k", "duz_IS1_1k", "duz_M1_cumle", "duz_IS1_cumle")]
                         for h in ("B1", "K2") if v.get(h, {}).get("hal") == "OKUNDU"}
            dz_say = {h: {m: O["sayim"][h][m] for m in ("duz_M1_1k", "duz_IS1_1k", "duz_M1_cumle", "duz_IS1_cumle")}
                      for h in ("B1", "K2")}
    yuk3 = [a for a in sorted(aile, key=lambda x: sat[x]["K3"][0], reverse=True) if sat[a]["K3"][1] == "u"]
    liste = lambda xs: ad(xs[0]) if len(xs) == 1 else ", ".join(ad(x) for x in xs[:-1]) + " and " + ad(xs[-1])
    L = [r"\paragraph*{Four cells on the ordinary instructions (\S\ref{sec:deperson}).}",
         r"\vekalet{The reading above pairs each aligned model in its own chat template with its base in the URIAL frame, so the two "
         r"checkpoints differ in prompt as well as in weights. Three further cells hold the string fixed, strip it of persons, or both "
         r"(Table~\ref{tab:cells}). "
         r"In \textbf{B1} (the reading above) the checkpoints are in different formats. In \textbf{K1} both checkpoints read the identical URIAL "
         r"string, the released \texttt{inst\_1k} prompt at $K=3$, so the aligned model reads a format it was not trained on. In "
         r"\textbf{K2} the aligned outputs are unchanged and the base reads a version of that frame stripped of first- and second-person "
         r"forms. In \textbf{K3} both checkpoints read that person-stripped string. Every cell uses the same $300$ instructions, four "
         r"sampling seeds, degeneracy shelf, instruction-clustered intervals ($1{,}000$ draws) and within-instruction paired placebo. "
         r"Every URIAL-format output is cut at its first code fence or new query line, and the first-person count is direct, with a "
         r"capitalised \emph{US} not counted. The reading rule for K1 and K2 and the registration of K3 were committed before any of "
         r"their generations was counted, and all four cells are descriptive, with no bar. "
         rf"\textbf{{In its own format the aligned model says ``you'' less in {hep(b1m)} models, and against a base on the "
         rf"person-stripped frame in {hep(k2m)}.}} On an identical string the sign splits: ${kd('K1', 0)}$ down, up and unclear on the "
         rf"standard frame and ${kd('K3', 0)}$ on the person-stripped one, where {liste(yuk3)} rise. "
         rf"``I'' holds better: it falls in ${say['K1'][1]['d']}$ models on the standard string and ${say['K3'][1]['d']}$ on the "
         r"person-stripped one. "
         + (rf"\textbf{{On prose lines alone the instruction reading is unchanged}}: in B1 ``you'' falls in "
            rf"{dz_say['B1']['duz_M1_1k']['asagi']} of 16 models per token and {dz_say['B1']['duz_M1_cumle']['asagi']} per sentence, "
            rf"and in K2 in {dz_say['K2']['duz_M1_1k']['asagi']} and {dz_say['K2']['duz_M1_cumle']['asagi']}; ``I'' falls in "
            rf"{dz_say['B1']['duz_IS1_1k']['asagi']} and {dz_say['B1']['duz_IS1_cumle']['asagi']} in B1 and in "
            rf"{dz_say['K2']['duz_IS1_1k']['asagi']} and {dz_say['K2']['duz_IS1_cumle']['asagi']} in K2. " if dz_say else "")
         + rf"On prose lines alone, with list, numbered and heading lines dropped (a median ${100*at_h:.0f}\%$ of aligned lines against "
         rf"${100*at_t:.0f}\%$ of base lines in K3), ``you'' falls in ${sayD[0]['d']}$ and rises in ${sayD[0]['u']}$ per token, and "
         rf"falls in ${sayD[2]['d']}$ and rises in ${sayD[2]['u']}$ per sentence, while ``I'' falls in ${sayD[1]['d']}$ per token and "
         rf"${sayD[3]['d']}$ per sentence and rises in at most ${max(sayD[1]['u'], sayD[3]['u'])}$." + r2 +
         r" The aligned model is therefore read in its own format for the claim about ``you'' on instructions, and the identical-string "
         r"cells are reported as a limit of that claim (Limitations).}",
         r"\begin{table}[tb]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Ordinary instructions in four cells.} Aligned minus base, per 1,000 tokens, with the first person counted "
         r"directly. B1: aligned in its own template, base in the URIAL frame. K1: both checkpoints on the identical URIAL string. K2: aligned in "
         r"its own template, base on the person-stripped frame. K3: both checkpoints on the identical person-stripped string. \textbf{Bold}: "
         r"instruction-clustered $95\%$ interval excludes zero. The lower block reads K3 on prose lines only, per token and per sentence, "
         r"with the share of lines dropped as list, numbered or heading lines.}\label{tab:cells}",
         r"\begin{tabular}{lrrrrrrrr}\toprule",
         r" & \multicolumn{2}{c}{B1} & \multicolumn{2}{c}{K1} & \multicolumn{2}{c}{K2} & \multicolumn{2}{c}{K3} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"model & $\Delta M_1$ & $\Delta$1st & $\Delta M_1$ & $\Delta$1st & $\Delta M_1$ & $\Delta$1st & $\Delta M_1$ & $\Delta$1st \\ \midrule"]
    for a in aile:
        L.append(ad(a) + " & " + " & ".join(f"{hucre(sat[a][h][0], sat[a][h][1])} & {hucre(sat[a][h][3], sat[a][h][4])}"
                                             for h in ("B1", "K1", "K2", "K3")) + r" \\")
    L += [r"\midrule down / up / unclear & " + " & ".join(f"{kd(h, 0)} & {kd(h, 1)}" for h in ("B1", "K1", "K2", "K3")) + r" \\",
          r"down, above placebo & " + " & ".join(f"${say[h][2]}$ & ${say[h][3]}$" for h in ("B1", "K1", "K2", "K3")) + r" \\",
          r"\bottomrule\end{tabular}", r"\\[6pt]",
          r"\begin{tabular}{l" + "rr" * (5 if DZ else 3) + r"}\toprule",
          " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in
                             ((["B1 prose/token", "K2 prose/token"] if DZ else []) +
                              ["K3 prose/token", "K3 prose/sentence", "lines dropped"])) + r" \\",
          "".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(5 if DZ else 3)),
          "model & " + " & ".join([r"$\Delta M_1$ & $\Delta$1st"] * (4 if DZ else 2)) + r" & base & aligned \\ \midrule"]
    for a in aile:
        d = sat[a]["D"]
        on = "".join(f"{hucre(*DZ[a][h][0])} & {hucre(*DZ[a][h][1])} & " for h in ("B1", "K2")) if DZ and a in DZ else ""
        L.append(ad(a) + " & " + on + f"{hucre(*d[0])} & {hucre(*d[1])} & {hucre(d[2][0], d[2][1], 3)} & {hucre(d[3][0], d[3][1], 3)} & "
                 f"${100*sat[a]['at'][0]:.0f}\\%$ & ${100*sat[a]['at'][1]:.0f}\\%$ \\\\")
    ds = lambda h, m: f"{dz_say[h][m]['asagi']}/{dz_say[h][m]['yukari']}/{dz_say[h][m]['null']}"
    on_say = ("".join(f"{ds(h, 'duz_M1_1k')} & {ds(h, 'duz_IS1_1k')} & " for h in ("B1", "K2")) if dz_say else "")
    L += [r"\midrule down / up / unclear & " + on_say + " & ".join(f"{s['d']}/{s['u']}/{s['n']}" for s in sayD) +
          rf" & ${100*at_t:.0f}\%$ & ${100*at_h:.0f}\%$ \\", r"\bottomrule\end{tabular}", r"\end{table}"]
    io.open(f"{OUT}/HUCRELER_D4.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/hucreler_d4.py)\n" + "\n".join(L) + "\n")
    src = [pb, pv, pq, pk] + ([pr2] if pr2 else [])
    json.dump(dict(table="HUCRELER_D4", ciktilar=["HUCRELER_D4.tex", "HUCRELER_CUMLE.tex"], sources=src,
                   payda=dict(n_model=16, sayim={h: dict(M1=say[h][0], IS1=say[h][1], M1_plasebo=say[h][2], IS1_plasebo=say[h][3]) for h in say},
                              K3_duzyazi=sayD, atilan_ortanca=dict(taban=round(at_t, 4), hizali=round(at_h, 4)), bant_hizali=oran,
                              r2_basildi=bool(r2)),
                   note="renders four descriptive cell cards; no statistic computed here"),
              io.open(f"{OUT}/HUCRELER_D4.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  [PAYDA] hucreler_d4: n_model=16 · B1 {say['B1'][:2]} · K1 {say['K1'][:2]} · K2 {say['K2'][:2]} · K3 {say['K3'][:2]} · "
          f"K3 düzyazi {sayD} · r2={'VAR' if r2 else ''} ⇒ esik: dört card ESDEGER ⇒ EYLEM: degilse YAZILMAZ")
    print("✓ HUCRELER_D4.tex · HUCRELER_CUMLE.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
