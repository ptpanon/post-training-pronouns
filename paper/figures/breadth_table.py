#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import load, ROOT
import breadth_figure as F1
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

ISARET = {"AYRIK_POZ": "$+$", "AYRIK_NEG": "$-$", "CONTINGENT": "(c)", "~": "$\\sim$"}


def _ad(c):
    for k in ("AYRIK_POZ", "AYRIK_NEG", "CONTINGENT"):
        if c.get(k):
            return k
    return "~"


def _tex(s):
    return s.replace(" · ", " ").replace(" → ", "$\\rightarrow$")


def main():
    hucre, kayit = [], []
    for rel, haric in F1.KAYNAK:
        d, prov = load(rel); kayit.append(prov)
        for ad, v in d["panel"].items():
            if haric and ad in haric:
                continue
            dd = v.get("delta") or {}
            if not all(k in dd for k in ("DOM", "ARO")):
                continue
            hucre.append((_tex(F1.guzel(ad)), ad, dd))
    KOL, KAYNAK_SATIR = {}, {}
    for _rel in ("results/template_robustness_2026-08-26.json",
                 "results/template_robustness_24_2026-08-27.json",
                 "results/template_holes_ladder_rungs_2026-08-29.json"):
        _d, prov = load(_rel); kayit.append(prov)
        for _k, _v in _d["kol"].items():
            if _v.get("HAL", "").startswith("KAPI"):
                continue
            if _k in KOL:
                raise SystemExit(f"★ CAKISMA: {_k} iki kaynakta — hangisi?")
            KOL[_k] = _v; KAYNAK_SATIR[_k] = os.path.basename(_rel)
    BOS_AD = {"Qwen2.5-72B": "\\textsc{vram}", "Llama-3.1-70B": "\\textsc{vram}"}
    for _r in ("OLMo2-13B·SFT→DPO", "OLMo2-13B·DPO→Inst", "OLMo2-32B·SFT→DPO",
               "OLMo2-32B·DPO→Inst", "Tulu3-8B·SFT→DPO", "Tulu3-8B·DPO→RL"):
        BOS_AD[_r] = "\\textsc{protocol}"
    n_template = n_degisti = 0
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{\\textbf{Breadth, and where the second protocol reaches.} "
         "\\textbf{Carried from a separate instrument study; not a contribution here. $\\Delta U$ is defined in Appendix~\\ref{app:D7}.} "
         "Dominance shift for every base$\\rightarrow$aligned pair we ran, in "
         "null-sd, under the raw-continuation prompt. The right-hand block repeats the read "
         "under each aligned model's own chat template. Outcomes: $+$/$-$ "
         "CI-disjoint, (c) contingent, $\\sim$ null. \\textbf{The 24 rows are not "
         "one kind of object}: 15 are end-to-end pairs and 9 are checkpoint sequence stages. "
         "The template run now covers 16 of the 24, and the eight that stay empty are "
         "\\emph{named, not blank}. \\textsc{vram}: the 70B/72B pairs were produced "
         "across two cards (measured peak 67.2 and 70.8\\,GiB) and one card was "
         "available. \\textsc{protocol}: for the six intermediate stages the "
         "\\emph{base} checkpoint is itself an instruction-tuned checkpoint that carries a "
         "chat template, so a raw-continuation-base vs.\\ chat-template-aligned contrast would not "
         "isolate the template, so those cells need a chat-template-on-both-checkpoints protocol we "
         "have not pre-registered. \\textbf{Caveat on the template column.} The "
         "template also suppresses degenerate generation on the aligned outputs only "
         "(17--21\\% falling to 1.4--4.6\\% on the three stages measured here, while "
         "the raw-continuation base outputs stay at 19--28\\%), so $\\Delta U$ under template mixes "
         "a register change with a degeneracy change, and the design does not separate "
         "them. The three end-to-end composites and the nine stages are compared "
         "across protocols in Table~\\ref{tab:template}, whose panel outcome is "
         "\\textsc{template-dependent} (same 3 / changed 6).}",
         "\\label{tab:breadth}",
         "\\begin{tabular}{lrcrcc}", "\\toprule",
         "& \\multicolumn{2}{c}{raw-continuation protocol} & \\multicolumn{2}{c}{chat template} & \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Pair & $\\Delta U_{\\mathrm{DOM}}$ & outcome & $\\Delta U$ & outcome & name \\\\",
         "\\midrule"]
    for guzel_ad, ham_ad, dd in hucre:
        c = dd["DOM"]
        sat = f"{guzel_ad} & {c['dU']:+.3f} & {ISARET[_ad(c)]} "
        k = KOL.get(ham_ad)
        if k:
            n_template += 1
            deg = k.get("HAL") == "AD-DEGISTI"
            n_degisti += int(deg)
            ad_et = "\\textbf{changed}" if deg else "same"
            sat += (f"& {k['dU_sablonlu']:+.3f} "
                    f"& {ISARET.get(k['ad_sablonlu'], k['ad_sablonlu'])} & {ad_et} ")
        else:
            _et = BOS_AD.get(ham_ad, "---")
            sat += f"& --- & --- & {_et} "
        L.append(sat + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    bekl = sum(1 for a, _, _ in [(h[1], 0, 0) for h in hucre]
               if a in KOL and KOL[a].get("HAL") == "AD-DEGISTI")
    payda("a1_appendix", n_cift=len(hucre), n_template_kapsanan=n_template,
          hal_ad_degisti=n_degisti, red_sablonsuz=len(hucre) - n_template,
          bekle={"n_cift": 24})
    print(f"  [PAYDA] a1: n_cift={len(hucre)} · n_template={n_template} · "
          f"hal_degisti={n_degisti} · red_sablonsuz={len(hucre)-n_template} "
          f"· karne_capraz={bekl}")
    assert n_degisti == bekl, ""
    out = __DNH_ROOT__ + "/paper/figures/A1_breadth.tex"
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(table="A1_breadth", n_cift=len(hucre), n_template_kapsanan=n_template,
                   hal_ad_degisti=n_degisti, sources=kayit,
                   satir_kaynagi={h[1]: KAYNAK_SATIR[h[1]] for h in hucre
                                  if h[1] in KAYNAK_SATIR},
                   bos_hucre_adi={k: v.replace("\\textsc{", "").rstrip("}")
                                  for k, v in BOS_AD.items()},
                   note="renders pre-specified measurements; computes no new statistic",
                   KARAR="ASSISTANT 28.08 · F1_breadth setten cikti, appendix tablosu"),
              open(out.replace(".tex", ".meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ✓ {out}")


if __name__ == "__main__":
    main()
