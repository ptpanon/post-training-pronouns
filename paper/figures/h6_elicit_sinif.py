#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/reviewer6_d_elicit_sinif_2026-09-17.json"
SIRA = [("YO", "false premise"), ("KP", "bad plan"), ("AO", "reasoning to ratify"), ("SI", "borderline request"), ("TK", "contested opinion")]
COGUL = {"YO": "false premises", "KP": "bad plans", "AO": "requests to ratify reasoning", "SI": "borderline requests", "TK": "contested opinions"}


def ad(a): return a.replace("Tulu3-8B", r"T\"ulu-3-8B").replace("OLMo2-", "OLMo-2-")


def main():
    D, pk = load(CARD)
    print(f"  [PAYDA] h6_elicit_sinif: alet={D['alet']['hal']} · n_aile={len(D['aile'])} ⇒ esik ESDEGER ⇒ EYLEM: degilse YAZMA")
    if D["alet"]["hal"] != "ESDEGER":
        return 3
    A, S = D["aile"], D["sayim"]
    n = len(A)
    ist = sorted({A[a][c]["n_istem"] for a in A for c, _ in SIRA})
    zayif = [c for c, _ in SIRA if S[c]["M1"]["asagi"] == min(S[x]["M1"]["asagi"] for x, _ in SIRA)
             and S[c]["1st"]["asagi"] == min(S[x]["1st"]["asagi"] for x, _ in SIRA)]
    ad_c = dict(SIRA)
    cumle_zayif = (rf" The {ad_c[zayif[0]]} class is the weakest on both counters." if len(zayif) == 1 else "")
    L = [r"\paragraph*{The five stimulus classes of ELICIT-99.}",
         r"\vekalet{Each class of ELICIT-99 was read on its own with ELICIT-99's pairing, prompt clustering and paired "
         rf"placebo (${ist[0]}$--${ist[-1]}$ prompts per class, so intervals are wide and a null means not resolved at that size). "
         r"$\Delta M_1$ excludes zero downward in "
         + rf"${S[SIRA[0][0]]['M1']['asagi']}$ of ${n}$ models on {COGUL[SIRA[0][0]]}, " + ", ".join(rf"${S[c]['M1']['asagi']}$ on {COGUL[c]}" for c, _ in SIRA[1:-1])
         + rf" and ${S[SIRA[-1][0]]['M1']['asagi']}$ on {COGUL[SIRA[-1][0]]}; $\Delta$1st does so in "
         + ", ".join(rf"${S[c]['1st']['asagi']}$" for c, _ in SIRA[:-1]) + rf" and ${S[SIRA[-1][0]]['1st']['asagi']}$ in the same order."
         + cumle_zayif + r"}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrr}\toprule",
         r"$\Delta M_1$ & " + " & ".join(x for _, x in SIRA) + r" \\ \midrule"]
    for a in sorted(A):
        hucre = []
        for c, _ in SIRA:
            v = A[a][c]
            s = f"${v['dM1']:+.2f}$"
            hucre.append(rf"\textbf{{{s}}}" if v["sinif_M1"] != "null" else s)
        L.append(f"{ad(a)} & " + " & ".join(hucre) + r" \\")
    L += [r"\midrule",
          r"down / up / null, $\Delta M_1$ & " + " & ".join(f"{S[c]['M1']['asagi']}/{S[c]['M1']['yukari']}/{S[c]['M1']['null']}" for c, _ in SIRA) + r" \\",
          r"down / up / null, $\Delta$1st & " + " & ".join(f"{S[c]['1st']['asagi']}/{S[c]['1st']['yukari']}/{S[c]['1st']['null']}" for c, _ in SIRA) + r" \\",
          r"\bottomrule\end{tabular}\end{center}",
          r"\vekalet{Bold: the prompt-clustered $95\%$ interval excludes zero. " + " and ".join(D["raf"]) + (" is" if len(D["raf"]) == 1 else " are") + r" on the degeneracy shelf of ELICIT-99 and not read.}"]
    io.open(f"{OUT}/H6_ELICIT_SINIF.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/h6_elicit_sinif.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="H6_ELICIT_SINIF", ciktilar=["H6_ELICIT_SINIF.tex"], sources=[pk],
                   note="renders the rule-first per-class reading of ELICIT-99; computes no new statistic"),
              io.open(f"{OUT}/H6_ELICIT_SINIF.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ H6_ELICIT_SINIF.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
