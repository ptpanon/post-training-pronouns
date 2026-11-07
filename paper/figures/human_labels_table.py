#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys, json, os, math
ROOT = __DNH_ROOT__ + ""
KAYNAK = f"{ROOT}/results/human_labels_result_2026-09-02.json"
OUT = os.path.dirname(os.path.abspath(__file__))
AD = {"KP-ac9d": "aligned assistant", "CMV": "online debate",
      "SCOTUS": "courtroom argument"}


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def hucre(k, n):
    if not n:
        return r"\textsc{n/a}"
    lo, hi = wilson(k, n)
    return f"{k}/{n} & {k/n:.2f} & [{lo:.2f}, {hi:.2f}]"


def main():
    S = json.load(open(KAYNAK, encoding="utf-8"))
    T = S["substrat"]; g = S["_sayim"]
    L = [r"\begin{table}[tb]\centering\small",
         r"\caption{\textbf{The correction counter against a human labeller.} "
         r"Forty counter-flagged sentences, drawn balanced across three "
         r"substrates and across the counter's own two decisions, labelled by "
         r"one person under a single three-way question: does this sentence "
         r"correct a claim, and is the correction aimed at the reader or at the "
         r"claim. \textsc{precision}: the share the labeller agreed were "
         r"corrections at all. \textsc{impersonal}: among those, the share aimed "
         r"at the claim. Two items were set to \textsc{no-context} because the package "
         r"showed one sentence and the deictic reached back into the previous one "
         r"(one returned unlabelled, one labelled with that note). They "
         r"leave the denominator by name. Intervals are Wilson $95\%$. "
         r"Recall is \emph{not} measurable here: every item was already flagged "
         r"by the counter.}\label{tab:humanlabels}",
         r"\begin{tabular}{lrrrcrrc}", r"\toprule",
         r"substrate & $n$ & \multicolumn{3}{c}{precision} & "
         r"\multicolumn{3}{c}{impersonal share} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r" & & count & rate & 95\% CI & count & rate & 95\% CI \\", r"\midrule"]
    for sub in ("KP-ac9d", "CMV", "SCOTUS"):
        d = T[sub]; kab = d["A"] + d["B"]; gec = d["n_gecerli"]
        L.append(f"{AD[sub]} & {d['n']} & {hucre(kab, gec)} & {hucre(d['B'], kab)} \\\\")
    L.append(r"\midrule")
    L.append(f"pooled & {g['n_madde']} & {hucre(g['A'] + g['B'], g['n_gecerli'])} & "
             f"{hucre(g['B'], g['A'] + g['B'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T6_humanlabels.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    a = wilson(T["KP-ac9d"]["A"] + T["KP-ac9d"]["B"], T["KP-ac9d"]["n_gecerli"])
    s = wilson(T["SCOTUS"]["A"] + T["SCOTUS"]["B"], T["SCOTUS"]["n_gecerli"])
    ayrik = a[0] > s[1]
    print(f"  [PAYDA] t6_humanlabels: n_substrat=3 · n_madde={g['n_madde']} · "
          f"red_baglam_yok={g['X']} · asistan↔SCOTUS kesinlik CI-ayrik={int(ayrik)}")
    print(f"  ✓ T6_humanlabels.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
