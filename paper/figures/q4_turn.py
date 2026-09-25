#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

R = __DNH_ROOT__ + ""
CARD = "results/v97_q4_turn_2026-09-21.json"
AD = {"Tulu3-8B": r"T\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}
S3 = {"asagi": "d", "yukari": "u", "null": "n"}
TUR = ["TUR_BLOG", "TUR_FORUM", "TUR_NASIL", "TUR_NOTICE"]
BAS = {"TUR_BLOG": "blog", "TUR_FORUM": "forum", "TUR_NASIL": "how-to", "TUR_NOTICE": "news"}


def ad(a): return AD.get(a, a)
def sayim(xs): return {c: sum(1 for x in xs if x == c) for c in "dun"}


def hucre(v, s, nd=2):
    t = f"{v:+.{nd}f}"
    return rf"$\mathbf{{{t}}}$" if s != "n" else f"${t}$"


def main():
    if not os.path.exists(f"{R}/{CARD}"):
        print(f"{CARD}"); return 3
    K, pk = load(CARD)
    aile = [a for a in K["aile"] if K["aile"][a].get("hal") == "OKUNDU"]
    if K["alet"]["hal"] != "ESDEGER" or len(aile) != 16 or K["bacak_eksik"]:
        print(f"{K['alet']['hal']} {len(aile)}"
              f"{K['bacak_eksik']}"); return 3
    E = ["TUM"] + TUR
    sat = {a: {e: tuple(list(K["aile"][a][e][m]["gozlenen"] for m in ("M1_1k", "IS1_1k"))
                        + [S3[K["aile"][a][e][f"{m}_sinif3"]] for m in ("M1_1k", "IS1_1k")]) for e in E}
           for a in aile}
    duz = {a: [(K["aile"][a]["TUM"][m]["gozlenen"], S3[K["aile"][a]["TUM"][f"{m}_sinif3"]])
               for m in ("duz_M1_1k", "duz_IS1_1k", "duz_M1_cumle", "duz_IS1_cumle")] for a in aile}
    say = {e: {m: K["sayim"][e][m] for m in ("M1_1k", "IS1_1k", "duz_M1_1k", "duz_IS1_1k", "duz_M1_cumle", "duz_IS1_cumle")}
           for e in E}
    pla = K["asagi_ve_plasebo_ustu"]
    d = lambda e, m: say[e][m]["asagi"]
    u = lambda e, m: say[e][m]["yukari"]
    nn = lambda e, m: say[e][m]["null"]
    kd = lambda e, m: f"{d(e, m)}/{u(e, m)}/{nn(e, m)}"
    hep = lambda n: "all 16" if n == 16 else f"{n} of 16"
    MT = {b: {t: {v: sum(K["aile"][a]["uyum"][b]["matris"][t][v] for a in aile) for v in TUR} for t in TUR}
          for b in ("taban", "hizali")}
    pay = {b: {t: MT[b][t][t] / max(sum(MT[b][t].values()), 1) for t in TUR} for b in MT}
    n_satir = K["tasarim"]["n_satir_bacak"]
    topl = {b: {t: sum(MT[b][t].values()) for t in TUR} for b in MT}
    kay = {}
    for v in TUR:
        ic = MT["hizali"][v][v] / max(topl["hizali"][v], 1)
        dis = float(np.mean([MT["hizali"][t][v] / max(topl["hizali"][t], 1) for t in TUR if t != v]))
        kay[v] = (ic, 100 * (ic - dis), dis)
    en_kay = max(TUR, key=lambda t: kay[t][1])
    hav = {b: {v: sum(MT[b][t][v] for t in TUR) / max(sum(topl[b].values()), 1) for v in TUR} for b in MT}
    cumle = (rf"With the genre fixed by a one-line header, ``I'' falls in {d('TUM', 'IS1_1k')} of 16 models across four genres "
             rf"and ``you'' in {d('TUM', 'M1_1k')} (Appendix~\ref{{app:D2}})")
    io.open(f"{OUT}/TUR_CUMLE.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/q4_turn.py)\n" + cumle)
    en_iyi = max(TUR, key=lambda t: pay["hizali"][t])
    en_kotu = min(TUR, key=lambda t: pay["hizali"][t])
    L = [r"\paragraph*{Raw continuation with the genre fixed by a header.}",
         r"\vekalet{The debate stubs leave the genre of the continuation open, so a model that answers in a less personal genre and a "
         r"model that uses fewer persons within a genre are not separated. To separate them we re-ran raw continuation with a "
         r"one-line bracketed header in front of the same 34 stubs: \texttt{[Personal blog post]}, "
         r"\texttt{[Reply in a discussion forum]}, \texttt{[How-to guide]} and \texttt{[News report]}. None of the four contains a "
         r"first- or second-person form. Both checkpoints read the identical plain string, with no chat template, at the panel's own "
         rf"decoding settings and 12 samples per prompt, so each checkpoint has {n_satir:{','}} generations. The design, the single reading "
         r"path and the genre rule below were committed before any of these generations existed. The reading is descriptive: there "
         r"is no bar and no outcome name. "
         rf"\textbf{{With the genre fixed, both forms still fall}}: ``you'' in {hep(d('TUM', 'M1_1k'))} models across the four "
         rf"genres together and ``I'' in {hep(d('TUM', 'IS1_1k'))}, with {pla['TUM']['M1_1k']} and {pla['TUM']['IS1_1k']} of those "
         r"above their paired placebo (Table~\ref{tab:tur}). Genre by genre the counts are "
         + ", ".join(rf"{kd(t, 'M1_1k')} for ``you'' and {kd(t, 'IS1_1k')} for ``I'' on {BAS[t]}" for t in TUR)
         + r" (down / up / unclear); each genre carries a quarter of the data, so an interval covering zero there is a loss of "
         r"power and not a result. On prose lines alone, across the four genres together, ``you'' falls in "
         rf"{d('TUM', 'duz_M1_1k')} models per token and {d('TUM', 'duz_M1_cumle')} per sentence, and ``I'' in "
         rf"{d('TUM', 'duz_IS1_1k')} and {d('TUM', 'duz_IS1_cumle')}, with no model rising in either. "
         r"\textbf{The header fixes the request, not the realised genre.} Each generation is classified by a rule fixed before "
         r"counting: forum cues, then how-to cues, then news cues, none of which contains a person, with blog as the residual "
         r"class. A cue class is entered only when its cue fires, so those three shares are lower bounds on the genre a reader "
         r"would name, and blog's is an upper bound. Asking for a genre moves the class a generation lands in by at most "
         rf"{kay[en_kay][1]:.0f} percentage points (aligned {BAS[en_kay]}: ${100*kay[en_kay][0]:.0f}\%$ of {BAS[en_kay]} requests "
         rf"against ${100*kay[en_kay][2]:.0f}\%$ of the others), and by three points or fewer in the other three, where the "
         r"cue rule fires rarely. The two checkpoints also still differ from each other: pooled over the four requests the base "
         rf"generations are classified blog in ${100*hav['taban']['TUR_BLOG']:.0f}\%$ and how-to in "
         rf"${100*hav['taban']['TUR_NASIL']:.0f}\%$, the aligned ones in ${100*hav['hizali']['TUR_BLOG']:.0f}\%$ and "
         rf"${100*hav['hizali']['TUR_NASIL']:.0f}\%$. The header narrows the genre difference between the checkpoints without closing "
         r"it, so what this prompt set adds is that the fall survives when the same genre is requested of both checkpoints, not that genre "
         r"is held constant.}",
         r"\begin{table}[tb]\centering\scriptsize\setlength{\tabcolsep}{2.8pt}",
         r"\caption{\textbf{Raw continuation with the genre fixed by a one-line header.} Aligned minus base, per 1,000 tokens, "
         r"first person counted directly. \emph{all} pools the four genres. \textbf{Bold}: prompt-clustered $95\%$ interval "
         r"excludes zero. The middle block reads the pooled panel on prose lines only, per token and per sentence. The lower block "
         r"is the genre a generation was classified into, summed over the 16 models, against the genre its header requested; blog "
         r"is the residual class, so its diagonal is an upper bound.}\label{tab:tur}",
         r"\begin{tabular}{l" + "rr" * 5 + r"}\toprule",
         r" & \multicolumn{2}{c}{all} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{BAS[t]}}}" for t in TUR) + r" \\",
         "".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(5)),
         "model & " + " & ".join([r"$\Delta M_1$ & $\Delta$1st"] * 5) + r" \\ \midrule"]
    for a in aile:
        L.append(ad(a) + " & " + " & ".join(f"{hucre(sat[a][e][0], sat[a][e][2])} & {hucre(sat[a][e][1], sat[a][e][3])}" for e in E) + r" \\")
    L += [r"\midrule down / up / unclear & " + " & ".join(f"{kd(e, 'M1_1k')} & {kd(e, 'IS1_1k')}" for e in E) + r" \\",
          r"down, above placebo & " + " & ".join(f"${pla[e]['M1_1k']}$ & ${pla[e]['IS1_1k']}$" for e in E) + r" \\",
          r"\bottomrule\end{tabular}", r"\\[6pt]",
          r"\begin{tabular}{lrrrr}\toprule",
          r" & \multicolumn{2}{c}{prose, per token} & \multicolumn{2}{c}{prose, per sentence} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
          r"model & $\Delta M_1$ & $\Delta$1st & $\Delta M_1$ & $\Delta$1st \\ \midrule"]
    for a in aile:
        q = duz[a]
        L.append(ad(a) + f" & {hucre(*q[0])} & {hucre(*q[1])} & {hucre(q[2][0], q[2][1], 3)} & {hucre(q[3][0], q[3][1], 3)}" + r" \\")
    L += [r"\midrule down / up / unclear & " + " & ".join(kd("TUM", m) for m in ("duz_M1_1k", "duz_IS1_1k", "duz_M1_cumle", "duz_IS1_cumle")) + r" \\",
          r"\bottomrule\end{tabular}", r"\\[6pt]",
          r"\begin{tabular}{l" + "r" * 8 + r"}\toprule",
          r" & \multicolumn{4}{c}{base outputs, classified as} & \multicolumn{4}{c}{aligned outputs, classified as} \\",
          r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
          "header asks for & " + " & ".join([BAS[t] for t in TUR] * 2) + r" \\ \midrule"]
    for t in TUR:
        L.append(BAS[t] + " & " + " & ".join(f"${MT[b][t][v]:{','}}$".replace(",", "{,}") for b in ("taban", "hizali") for v in TUR) + r" \\")
    L += [r"\midrule matches the request & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{" +
          ", ".join(f"{BAS[t]} ${100*pay[b][t]:.0f}\\%$" for t in TUR) + r"}" for b in ("taban", "hizali")) + r" \\",
          r"\bottomrule\end{tabular}", r"\end{table}"]
    io.open(f"{OUT}/TUR_D2.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/q4_turn.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="TUR_D2", ciktilar=["TUR_D2.tex", "TUR_CUMLE.tex"], sources=[pk],
                   payda=dict(n_model=16, n_satir_bacak=n_satir, sayim=say, plasebo=pla,
                              uyum_pay={b: {t: round(pay[b][t], 4) for t in TUR} for b in pay},
                              uyum_kayma={t: [round(kay[t][0], 4), round(kay[t][1], 2), round(kay[t][2], 4)] for t in TUR},
                              havuz={b: {t: round(hav[b][t], 4) for t in TUR} for b in hav},
                              prereg="preregistration/prereg_v97_q4_turn_2026-09-20.md"),
                   note="renders the descriptive genre-fixed card; no statistic computed here"),
              io.open(f"{OUT}/TUR_D2.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  [PAYDA] q4_tur: n_model={len(aile)} · kapi {K['alet']['kapi_fark_max']:.1e} · TÜM M1 {kd('TUM','M1_1k')} · "
          f"IS1 {kd('TUM','IS1_1k')} · plasebo {pla['TUM']['M1_1k']}/{pla['TUM']['IS1_1k']} · "
          "uyum(hizali) " + " ".join("%s %.2f" % (BAS[t], pay["hizali"][t]) for t in TUR) +
          " ⇒ esik: ESDEGER + 16 model ⇒ EYLEM: degilse YAZILMAZ")
    print("✓ TUR_D2.tex · TUR_CUMLE.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
