#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/template_kadran_betim_2026-09-17.json"
ETIKET = {"KISI_ASAGI": "person-down", "KISI_YUKARI": "person-up", "RASTGELE": "unselected"}
SIRA = ["KISI_ASAGI", "RASTGELE", "KISI_YUKARI"]


def main():
    if not os.path.exists(f"{__DNH_ROOT__}/{CARD}"):
        print(""); return 3
    K, pk = load(CARD)
    kol = K["kol"]
    n_tam = sum(1 for v in K["yuva_kapisi"].values() for x in v.values() if x == "TAM")
    raf = [k for k in SIRA if kol[k]["raf"]]
    print(f"{len(kol)} {K['dize_tam']} {n_tam} {raf or 'YOK'}"
          f"")
    if not (len(kol) == 3 and K["dize_tam"] and n_tam == 12 and all(k in kol for k in SIRA)):
        return 3
    ci = lambda x: f"$[{x[0]:+.2f},{x[1]:+.2f}]$"
    A = kol["KISI_ASAGI"]; Y = kol["KISI_YUKARI"]; R = kol["RASTGELE"]
    sapma = (r"Ran, and the registration's own degeneracy gate fired: one run falls below the \mbox{distinct-4} shelf "
             rf"(${Y['distinct4']:.2f}$ against a floor of $0.60$). The slot collision that had produced that run's generations "
             r"without the template was repaired and the run regenerated in an empty slot, four seeds, all twelve generation sets complete.")
    io.open(f"{OUT}/TEMPLATE_KADRAN_SAPMA.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/template_kadran.py)\n" + sapma)
    L = [r"\paragraph*{The targeted preference runs under the chat template.}",
         r"\vekalet{The targeted preference runs in Table~\ref{tab:dial_dial} are read on the raw-continuation panel. We also read them through the "
         r"T\"ulu-3-8B-SFT chat template, against that template's own SFT outputs, on the same debate prompts and four generation seeds, "
         r"with the run's own within-prompt paired placebo. "
         rf"The shift survives the template and is larger than on the raw-continuation prompt: the person-down run moves "
         rf"${A['dA']:+.2f}$ {ci(A['ciA'])} per thousand against the chat-template SFT outputs (raw continuation: ${A['ciplak_dA']:+.2f}$) and "
         rf"${A['dB']:+.2f}$ {ci(A['ciB'])} against the unselected run, against a placebo p95 of ${A['plasebo_p95']:.2f}$. "
         rf"\textbf{{No outcome name is given here}}: the person-up run's generations fall below the \mbox{{distinct-4}} shelf "
         rf"(${Y['distinct4']:.2f}$), which the registration set as a degeneracy floor before the run, so its level "
         rf"(${Y['dA']:+.2f}$ {ci(Y['ciA'])}) is printed and not read as a magnitude (Appendix~\ref{{app:sapma}}). "
         rf"The unselected run, which the raw-continuation reading leaves at ${R['ciplak_dA']:+.2f}$, moves ${R['dA']:+.2f}$ {ci(R['ciA'])} "
         r"under the template: the template itself carries a little of the axis, which is why the second column of the table "
         r"references every run to it. "
         r"All four runs passed the identical-string gate, so the three trained runs and the SFT checkpoint answer the same prompts.}",
         r"\begin{center}\small\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{lrrrrr}\toprule",
         r" & \multicolumn{2}{c}{$\Delta M_1$ per thousand} & & & raw-continuation \\",
         r"\cmidrule(lr){2-3}",
         r"run & vs chat-template SFT & vs unselected run & placebo p95 & distinct-4 & vs SFT \\ \midrule"]
    for k in SIRA:
        v = kol[k]
        b = r"$^{\dagger}$" if v["raf"] else ""
        L.append(f"{ETIKET[k]}{b} & ${v['dA']:+.2f}$ {ci(v['ciA'])} & "
                 + (f"${v['dB']:+.2f}$ {ci(v['ciB'])}" if k != "RASTGELE" else "---")
                 + f" & ${v['plasebo_p95']:.2f}$ & ${v['distinct4']:.3f}$ & ${v['ciplak_dA']:+.2f}$ \\\\")
    L += [r"\bottomrule\end{tabular}\end{center}",
          rf"\vekalet{{\footnotesize $^{{\dagger}}$ below the \mbox{{distinct-4}} floor of $0.60$; level printed, magnitude not read. "
          rf"The chat-template SFT outputs sit at ${K['taban_M1']:.2f}$ per thousand.}}"]
    io.open(f"{OUT}/TEMPLATE_KADRAN.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/template_kadran.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="TEMPLATE_KADRAN", ciktilar=["TEMPLATE_KADRAN.tex", "TEMPLATE_KADRAN_SAPMA.tex"], sources=[pk],
                   payda=dict(n_kol=len(kol), yuva_tam=n_tam, raf=raf, taban_M1=round(K["taban_M1"], 4)),
                   note="renders the descriptive chat-template-dial card; no statistic computed here"),
              io.open(f"{OUT}/TEMPLATE_KADRAN.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ TEMPLATE_KADRAN.tex · TEMPLATE_KADRAN_SAPMA.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
