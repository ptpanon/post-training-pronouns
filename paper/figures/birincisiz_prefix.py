#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/v96_q1_birincisiz_prefix_2026-09-18.json"
AD = {"Tulu3-8B": r"T\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def ad(a): return AD.get(a, a)


def h(v, s):
    t = f"{v:+.2f}"
    return rf"$\mathbf{{{t}}}$" if s != "null" else f"${t}$"


def main():
    K, pk = load(CARD)
    A = K["aile"]
    if K["alet"]["hal"] != "ESDEGER" or len(A) != 16:
        print(""); return 3
    n4t = [a for a, v in A.items() if v["K4"]["sinif_d1st_turetilmis"] != "asagi"]
    n4d = [a for a, v in A.items() if v["K4"]["sinif_d1st_dogrudan"] != "asagi"]
    if sorted(n4t) != sorted(n4d):
        print(f"  [PAYDA] birincisiz_onek: K4 türetilmis {n4t} ≠ dogrudan {n4d} ⇒ EYLEM: blok YAZILMAZ"); return 3
    s = K["sayim"]; k4 = s["K4_sinif_d1st_dogrudan"]; k2m = s["K2_sinif_dM1"]; k2d = s["K2_sinif_d1st_dogrudan"]
    istisna = " and ".join(ad(a) for a in sorted(n4d))
    cumle = (rf"On the four prefix conditions that carry no first person the fall holds in ${k4['asagi']}$ of $16$"
             + (rf" ({istisna}'s interval covers zero)" if len(n4d) == 1 and A[n4d[0]]["K4"]["sinif_d1st_dogrudan"] == "null" else
                (rf" ({istisna} excepted)" if n4d else "")))
    io.open(f"{OUT}/BIRINCISIZ_CUMLE.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/birincisiz_prefix.py)\n" + cumle)
    L = [r"\paragraph*{The first-person fall on the prefix conditions that carry no first person.}",
         r"\vekalet{Twelve of the $16$ prefix conditions carry first-person forms of their own, so a base checkpoint could copy them. "
         r"Four carry none (the arousal conditions, one- and four-utterance, in both directions), and two of these carry no second "
         r"person either. We re-read the first-person change on those four alone (Table~\ref{tab:birincisiz}), on the original sampling seed, with the "
         r"prompt-clustered interval of the full panel ($1{,}000$ draws). The same loop reproduces the full panel's first-person change "
         r"in all $16$ models, to within $10^{-9}$, before any subset is read. "
         rf"\textbf{{On the four conditions the first person falls in ${k4['asagi']}$ of $16$ models}}, counted directly or as derived, and "
         rf"{istisna} is the exception ({'interval covering zero' if A[n4d[0]]['K4']['sinif_d1st_dogrudan'] == 'null' else 'a rise'}). "
         rf"On the two conditions that carry neither person the first person falls in "
         rf"${k2d['asagi']}$ and rises in ${k2d['yukari']}$, and the second person falls in ${k2m['asagi']}$, rises in "
         rf"${k2m['yukari']}$ and is unclear in ${k2m['null']}$. The four conditions hold a quarter of the data and the two an eighth, "
         r"so intervals are wider and an unclear reading is a loss of power, not a finding of no change. The reading is descriptive: "
         r"its rule was committed before the count, with no bar.}",
         r"\begin{table}[tb]\centering\scriptsize\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{The first-person fall without first-person prefixes.} Aligned minus base per 1,000 tokens, raw continuation, "
         r"original seed. \emph{All 16}: every prefix condition. \emph{Four}: the conditions with no first person. \emph{Two}: the "
         r"conditions with neither person. $\Delta$1st is counted directly, with a capitalised \emph{US} not counted. \textbf{Bold}: "
         r"prompt-clustered $95\%$ interval excludes zero.}\label{tab:birincisiz}",
         r"\begin{tabular}{lrrrrr}\toprule",
         r" & all 16 prefixes & \multicolumn{2}{c}{four} & \multicolumn{2}{c}{two} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"model & $\Delta$1st & $\Delta$1st & $95\%$ CI & $\Delta M_1$ & $\Delta$1st \\ \midrule"]
    for a in sorted(A, key=lambda x: A[x]["K4"]["d1st_dogrudan"]):
        v = A[a]; t, k, k2 = v["TUM"], v["K4"], v["K2"]
        L.append(f"{ad(a)} & {h(t['d1st_dogrudan'], t['sinif_d1st_dogrudan'])} & {h(k['d1st_dogrudan'], k['sinif_d1st_dogrudan'])} & "
                 f"$[{k['ci_d1st_dogrudan'][0]:+.2f},{k['ci_d1st_dogrudan'][1]:+.2f}]$ & {h(k2['dM1'], k2['sinif_dM1'])} & "
                 f"{h(k2['d1st_dogrudan'], k2['sinif_d1st_dogrudan'])} \\\\")
    say = lambda d: f"{d['asagi']}/{d['yukari']}/{d['null']}"
    L += [rf"\midrule down / up / unclear & {say(s['TUM_sinif_d1st_dogrudan'])} & {say(k4)} & & {say(k2m)} & {say(k2d)} \\",
          r"\bottomrule\end{tabular}\end{table}"]
    io.open(f"{OUT}/BIRINCISIZ_D2.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/birincisiz_prefix.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="BIRINCISIZ_D2", ciktilar=["BIRINCISIZ_D2.tex", "BIRINCISIZ_CUMLE.tex"], sources=[pk],
                   payda=dict(n_model=16, K4_dogrudan=k4, K2_dM1=k2m, K2_dogrudan=k2d, istisna=n4d),
                   note="renders the descriptive C(1) card; no statistic computed here"),
              io.open(f"{OUT}/BIRINCISIZ_D2.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  [PAYDA] birincisiz_onek: n_model=16 · K4 {k4} · K2 ΔM1 {k2m} · K2 Δ1st {k2d} · istisna {n4d} ⇒ esik: ESDEGER ⇒ EYLEM: degilse YAZILMAZ")
    print("✓ BIRINCISIZ_D2.tex · BIRINCISIZ_CUMLE.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
