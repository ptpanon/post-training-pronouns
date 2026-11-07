#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/reviewer6_template_mode_2026-09-17.json"
GEMMA = ("Gemma-3-4B", "Gemma-3-12B", "Gemma-3-27B")


def ad(a): return a.replace("Tulu3-8B", r"T\"ulu-3-8B").replace("OLMo2-", "OLMo-2-")


def main():
    C, pk = load(CARD)
    print(f"  [PAYDA] h6_template_kip: alet={C['alet']['hal']} · n_aile={len(C['aile'])} ⇒ esik ESDEGER ⇒ EYLEM: degilse YAZMA")
    if C["alet"]["hal"] != "ESDEGER":
        return 3
    A, B, E = C["aile"], C["kip_ici_bant"], C["bant_esik_satir"]
    g = [100 * A[a]["hizali"]["kip"]["ASISTAN"]["pay"] for a in GEMMA]
    tul_h, tul_t = 100 * A["Tulu3-8B"]["hizali"]["kip"]["DEVAM"]["pay"], 100 * A["Tulu3-8B"]["taban"]["kip"]["DEVAM"]["pay"]
    d, s, m = B["hizali"]["DEVAM"], B["hizali"]["ASISTAN"], B["hizali"]["META"]
    L = [r"\paragraph*{Which opening the aligned outputs take under the template.}",
         r"\vekalet{The opening rule used on the raw-continuation prompt (\S\ref{sec:deperson}: refusal, assistant opener, meta or summary, "
         r"otherwise continuation, first match on the opening) was applied unchanged to both checkpoints of the chat-template panel, on the "
         r"cells behind the template band. Under the template the three aligned Gemma-3 models open as an assistant in "
         rf"${min(g):.0f}$--${max(g):.0f}\%$ of generations, while T\"ulu-3-8B's aligned outputs continue in ${tul_h:.0f}\%$ and its "
         rf"base in ${tul_t:.0f}\%$. The table gives each model's rate within an opening where at least ${E}$ generations fall "
         r"there. \textbf{We read no range across models from those columns.} The rule reads only the opening, and \emph{continue} "
         r"means only that no listed opener was found, so a rate read inside that class carries the rule's own error; the spread "
         r"under the template is reported in \S\ref{sec:deperson} over all generations, not within an opening.}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrrr}\toprule",
         r" & \multicolumn{3}{c}{aligned outputs, share of generations (\%)} & \multicolumn{2}{c}{aligned $M_1$ within} & base \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r"model & assistant & meta & continue & continue & assistant & continue (\%) \\ \midrule"]
    for a in sorted(A):
        h, t = A[a]["hizali"]["kip"], A[a]["taban"]["kip"]
        f = lambda k: (f"${h[k]['M1']:.2f}$" if h[k]["n"] >= E and h[k]["M1"] is not None else "--")
        L.append(f"{ad(a)} & ${100*h['ASISTAN']['pay']:.0f}$ & ${100*h['META']['pay']:.0f}$ & ${100*h['DEVAM']['pay']:.0f}$ & "
                 f"{f('DEVAM')} & {f('ASISTAN')} & ${100*t['DEVAM']['pay']:.0f}$ \\\\")
    L += [r"\bottomrule\end{tabular}\end{center}"]
    io.open(f"{OUT}/H6_TEMPLATE_KIP.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/h6_template_mode.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="H6_TEMPLATE_KIP", ciktilar=["H6_TEMPLATE_KIP.tex"], sources=[pk],
                   note="renders the rule-first mode reading of the chat-template panel; computes no new statistic"),
              io.open(f"{OUT}/H6_TEMPLATE_KIP.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ H6_TEMPLATE_KIP.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
