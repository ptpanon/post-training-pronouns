#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/siradan_instruction_reading_2026-09-17.json"
SECIM = "results/siradan_instruction_secim_2026-09-17.json"
IS1 = "results/v91_konusan_sayimi_2026-09-17.json"
AD = {"Tulu3-8B": r"T\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def ad(a):
    return AD.get(a, a)


def main():
    R = __DNH_ROOT__ + ""
    if not all(os.path.exists(f"{R}/{x}") for x in (CARD, SECIM, IS1)):
        print(""); return 3
    K, pk = load(CARD)
    S, ps = load(SECIM)
    I, pi = load(IS1)
    A = K["aile"]; say = K["sayim"]; bant = K["bant"]
    n_asagi = len(say["M1"]["asagi"])
    n_ayrik = sum(1 for v in A.values() if v["M1"]["ayrik"] and v["M1"]["yon"] == "asagi")
    n_plasebo = sum(1 for v in A.values() if v["M1"].get("plasebo_ustu"))
    n1_yukari = len(say["SAHIS1"]["yukari"])
    AI = I["aile"]; sayI = I["sayim"]["IS1"]
    n_is1_asagi, n_is1_yukari, n_is1_null = len(sayI["asagi"]), len(sayI["yukari"]), len(sayI["null"])
    n_is1_p95 = I["n_plasebo_ustu"]
    yukari_ad = null_ad = ""
    n_istem = len(S["istemler"]) if isinstance(S["istemler"], list) else int(S["istemler"])
    print(f"{K['n_aile']} {K['n_okunan']} {len(K['bacak_eksik'])}"
          f"{n_istem} {n_asagi} {n_ayrik} {n_plasebo}"
          f"{n_is1_asagi} {n_is1_yukari} {n_is1_null} {n_is1_p95}"
          f"{I['esdeger']['hal']} {n1_yukari}"
          f"")
    if not (K["n_aile"] == K["n_okunan"] == 16 and not K["bacak_eksik"] and n_istem == 300 and n_asagi == n_ayrik == 16
            and I["esdeger"]["hal"] == "ESDEGER" and len(AI) == 16):
        return 3
    hiz, tab = bant["hizali"], bant["taban"]
    liste = lambda xs: (ad(xs[0]) if len(xs) == 1 else " and ".join([", ".join(ad(x) for x in xs[:-1]), ad(xs[-1])]))
    yukari_ad, null_ad = liste(sayI["yukari"]), liste(sayI["null"])
    kes = float(np.median([v["taban_kesilen_pay"] for v in A.values()]))
    devam = float(np.median([v["kip"]["hizali"]["DEVAM"]["pay"] for v in A.values()]))
    devam_t = float(np.median([v["kip"]["taban"]["DEVAM"]["pay"] for v in A.values()]))
    asis_t = float(np.median([v["kip"]["taban"]["ASISTAN"]["pay"] for v in A.values()]))
    ki = bant["kip_ici"]; sn = bant["sinif_hizali"]
    cumle = (rf"On ${n_istem}$ ordinary instructions, with the task held fixed, ``you'' falls in ${n_ayrik}$ of $16$ "
             rf"models and the aligned range is ${hiz['oran']:.1f}$-fold")
    io.open(f"{OUT}/SIRADAN_TALIMAT_CUMLE.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/siradan_instruction.py)\n" + cumle)
    _uy = "rises" if len(sayI["yukari"]) == 1 else "rise"
    _nu = "shows" if len(sayI["null"]) == 1 else "show"
    birinci = (rf"On ${n_istem}$ ordinary instructions they fall in ${n_is1_asagi}$ of $16$, while {yukari_ad} {_uy}"
               + (rf" and {null_ad} {_nu} no clear change" if sayI["null"] else ""))
    io.open(f"{OUT}/SIRADAN_TALIMAT_BIRINCI.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/siradan_instruction.py)\n" + birinci)
    sapma = (rf"Instrument fixed and committed before the run; no acceptance bound and no outcome name. The aligned outputs are each "
             rf"model's own chat template and the base outputs the same model in the URIAL frame, so the two checkpoints differ in prompt as "
             rf"well as in weights, and the base text is truncated at its first code fence (median ${100*kes:.0f}\%$ of rows). "
             rf"The first-person column counts first-person forms directly, excluding the capitalised \emph{{US}}; the derived "
             rf"measure the first reading printed (first person minus sentence-opening vocatives) is not reported.")
    io.open(f"{OUT}/SIRADAN_TALIMAT_SAPMA.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/siradan_instruction.py)\n" + sapma)
    L = [r"\paragraph*{Ordinary instructions, and the band they give.}",
         r"\vekalet{The continuation prompt is not an instruction, so we also read the battery on plain instructions. "
         rf"The stimulus is ${n_istem}$ instructions from Alpaca \citep{{taori2023alpaca}}, $75$ in each of four classes (summarize, classify, "
         r"how-to, explain), taken in hash order after dropping every instruction that carries a first- or second-person form of its "
         r"own, names an addressee genre, contains a code block or runs past $150$ words; the selection rule and the list are in the "
         r"released repository. The aligned outputs answer through the model's own chat template with no system line; the base checkpoint "
         r"receives the same instruction inside the URIAL frame \citep{lin2024urial}. Each checkpoint is read at four sampling seeds "
         rf"$\times$ ${n_istem}$ instructions ($1{{,}}200$ rows), cells are paired on instruction and seed, a cell enters only if both "
         r"checkpoints clear the degeneracy shelf, and the base text is cut at its first code fence or new query line "
         rf"(median ${100*kes:.0f}\%$ of base rows). Intervals are instruction-clustered ($1000$ draws) and each model carries its own "
         r"within-instruction paired placebo. "
         rf"\textbf{{Second-person density falls in ${n_ayrik}$ of $16$ models with intervals excluding zero, all ${n_plasebo}$ above "
         rf"their placebo}}, from a base range of ${tab['min']:.2f}$--${tab['max']:.2f}$ per thousand (${tab['oran']:.2f}\times$) to an "
         rf"aligned ${hiz['min']:.2f}$--${hiz['max']:.2f}$ (${hiz['oran']:.2f}\times$). "
         r"\textbf{The band does not open the way the raw-continuation panel's does}: with the task fixed and each pipeline given its own template, "
         r"the aligned spread stays inside a factor of three. "
         rf"\textbf{{The first person falls here too, but not everywhere}}: counted directly, first-person forms fall in "
         rf"${n_is1_asagi}$ of $16$ models with intervals excluding zero, rise in ${n_is1_yukari}$ ({yukari_ad}) and are null in "
         rf"${n_is1_null}$ ({null_ad}); ${n_is1_p95}$ of $16$ clear their own paired placebo. "
         r"(An earlier reading printed a derived measure, first-person forms minus sentence-opening vocatives, which rises "
         r"almost everywhere; we report the direct count instead.) "
         rf"The mode shares move with the template: continuations are ${100*devam_t:.0f}\%$ of base rows and ${100*devam:.0f}\%$ of "
         rf"aligned rows (median over models), and the assistant opening drops from ${100*asis_t:.0f}\%$ to the shelf. "
         rf"Read within mode, the aligned band is ${ki['hizali']['DEVAM']['oran']:.2f}\times$ among continuations "
         rf"(${ki['taban']['DEVAM']['oran']:.2f}\times$ at base, $16$ models) and ${ki['hizali']['ASISTAN']['oran']:.2f}\times$ among "
         rf"assistant openings (${ki['taban']['ASISTAN']['oran']:.2f}\times$, ${ki['hizali']['ASISTAN']['n']}$ models with at least "
         rf"${K['bant_esik_satir']}$ rows); refusals and meta rows are too few for a band. "
         r"\textbf{Instruction class matters more than model}: the aligned band within a class is "
         rf"${sn['NASIL']['oran']:.2f}\times$ for how-to (${sn['NASIL']['min']:.2f}$--${sn['NASIL']['max']:.2f}$ per thousand) and "
         rf"${sn['ACIKLAMA']['oran']:.2f}\times$ for explain (${sn['ACIKLAMA']['min']:.2f}$--${sn['ACIKLAMA']['max']:.2f}$), but "
         rf"${sn['SINIFLAMA']['oran']:.2f}\times$ for classify (${sn['SINIFLAMA']['min']:.2f}$--${sn['SINIFLAMA']['max']:.2f}$) and "
         rf"${sn['OZET']['oran']:.2f}\times$ for summarize (${sn['OZET']['min']:.2f}$--${sn['OZET']['max']:.2f}$): on those two "
         r"classes several models answer with almost no second person at all, so the ratio is a ratio of small numbers and is not "
         r"read as a spread. "
         r"This reading is descriptive: the instrument was committed before the run, and no acceptance bound was written "
         r"(Appendix~\ref{app:sapma}).}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrr}\toprule",
         r"model & base $M_1$ & aligned $M_1$ & $\Delta M_1$ $[$CI$]$ & $\Delta$ first person $[$CI$]$ & continuations \\ \midrule"]
    for a, v in sorted(A.items(), key=lambda kv: kv[1]["M1"]["gozlenen"]):
        m, s = v["M1"], AI[a]["IS1"]
        L.append(f"{ad(a)} & ${m['taban']:.2f}$ & ${m['taban']+m['gozlenen']:.2f}$ & "
                 f"${m['gozlenen']:+.2f}$ $[{m['ci'][0]:+.2f},{m['ci'][1]:+.2f}]$ & "
                 f"${s['gozlenen']:+.2f}$ $[{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}]$ & "
                 f"${100*v['kip']['hizali']['DEVAM']['pay']:.0f}\\%$ \\\\")
    L += [r"\bottomrule\end{tabular}\end{center}"]
    io.open(f"{OUT}/SIRADAN_TALIMAT.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/siradan_instruction.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="SIRADAN_TALIMAT", ciktilar=["SIRADAN_TALIMAT.tex", "SIRADAN_TALIMAT_CUMLE.tex", "SIRADAN_TALIMAT_BIRINCI.tex", "SIRADAN_TALIMAT_SAPMA.tex"],
                   sources=[pk, ps, pi],
                   payda=dict(n_aile=K["n_aile"], n_istem=n_istem, M1_ayrik=n_ayrik, plasebo_ustu=n_plasebo,
                              IS1_asagi=n_is1_asagi, IS1_yukari=n_is1_yukari, IS1_null=n_is1_null, IS1_plasebo_ustu=n_is1_p95,
                              bilesik_d1st_yukari=n1_yukari, bant_taban=tab["oran"], bant_hizali=hiz["oran"],
                              kesilen_ortanca=round(kes, 4)),
                   note="renders the descriptive ordinary-instruction card; no statistic computed here"),
              io.open(f"{OUT}/SIRADAN_TALIMAT.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ SIRADAN_TALIMAT.tex · SIRADAN_TALIMAT_CUMLE.tex · SIRADAN_TALIMAT_SAPMA.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
