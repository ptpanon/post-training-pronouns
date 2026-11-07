#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/v90_tez_kume_2026-09-17.json"
AD = {"Tulu3-8B": r"T\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def ad(a):
    return AD.get(a, a).replace("--", "-")


def main():
    if not os.path.exists(f"{__DNH_ROOT__}/{CARD}"):
        print(f""); return 3
    K, pk = load(CARD)
    hal = K["alet"]["hal"]
    print(f"{hal} {len(K.get('aile', {}))}")
    if hal != "ESDEGER" or len(K["aile"]) != 16:
        return 3
    A = K["aile"]; say = K["sayim"]
    nM, n1 = say["dM1_tez"]["asagi"], say["d1st_tez"]["asagi"]
    nM_i, n1_i = say["dM1_istem"]["asagi"], say["d1st_istem"]["asagi"]
    gen = float(np.median([v["genislik_orani"] for v in A.values()]))
    gen1 = float(np.median([v["genislik_orani_1st"] for v in A.values()]))
    kaybolan = sorted(a for a, v in A.items() if v["sinif_istem"] == "asagi" and v["sinif_tez"] != "asagi")
    kaybolan1 = sorted(a for a, v in A.items() if v["sinif1_istem"] == "asagi" and v["sinif1_tez"] != "asagi")
    def liste(xs):
        xs = [ad(x) for x in xs]
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    c = (rf"Resampled by thesis, $\Delta M_1$ stays negative with an interval excluding zero in ${nM}$ of $16$ models "
         rf"(${nM_i}$ of $16$ by prompt) and $\Delta$1st in ${n1}$ of $16$ (${n1_i}$ by prompt); the intervals widen by a median "
         rf"factor of ${gen:.2f}$ and ${gen1:.2f}$.")
    if kaybolan:
        c += rf" The second-person interval reaches zero under thesis clustering in {liste(kaybolan)}."
    if kaybolan1:
        c += rf" The first-person interval reaches zero in {liste(kaybolan1)}."
    L = [r"\paragraph*{Intervals clustered on the thesis.}",
         r"\vekalet{The $34$ prompts are $17$ theses, each in two stances, so prompt-clustered intervals treat the two stances of a thesis as independent. "
         r"We resampled theses instead ($1000$ draws on four pooled sampling seeds; with prompts as the unit the loop reproduces the second-person intervals of Table~\ref{tab:families} exactly; the first-person column here is the derived measure, which that table replaces with the direct count on the same seeds). "
         + c +
         r" With $17$ clusters a percentile interval is coarse, and the thesis absorbs the dependence between two stances, not the scaffold and prefix conditions that every prompt shares.}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrrr}\toprule",
         r" & \multicolumn{3}{c}{$\Delta M_1$ (four seeds pooled)} & \multicolumn{3}{c}{$\Delta$1st, derived (four seeds pooled)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"model & shift & by prompt & by thesis & shift & by prompt & by thesis \\ \midrule"]
    ci = lambda x: f"$[{x[0]:+.2f},{x[1]:+.2f}]$"
    for a, v in sorted(A.items(), key=lambda kv: kv[1]["dM1"]):
        L.append(f"{ad(a)} & ${v['dM1']:+.2f}$ & {ci(v['ci_istem'])} & {ci(v['ci_tez'])} & "
                 f"${v['d1st']:+.2f}$ & {ci(v['ci1_istem'])} & {ci(v['ci1_tez'])} \\\\")
    L += [r"\bottomrule\end{tabular}\end{center}"]
    io.open(f"{OUT}/TEZ_KUME.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/tez_kume.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="TEZ_KUME", ciktilar=["TEZ_KUME.tex"], sources=[pk],
                   payda=dict(n_aile=16, dM1_tez_asagi=nM, d1st_tez_asagi=n1, dM1_istem_asagi=nM_i, d1st_istem_asagi=n1_i),
                   note="renders the thesis-clustered card; computes no new statistic beyond medians of stored width ratios"),
              io.open(f"{OUT}/TEZ_KUME.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ TEZ_KUME.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
