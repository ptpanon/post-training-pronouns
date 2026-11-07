#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import json, re
from style import OUT, load

EN = {"AYRIK_POZ": r"$+$", "AYRIK_NEG": r"$-$", "CONTINGENT": r"(c)", "~": r"$\sim$"}
HAL = {"AYNI-AD": r"same", "AD-DEGISTI": r"\textbf{changed}", "KAPI": r"gate"}


def _satir(ad, v, basamak=False):
    return dict(ad=ad, hal=v["HAL"], ac=v["ad_ciplak"], as_=v["ad_sablonlu"],
                uc=v["dU_ciplak"], us=v["dU_sablonlu"],
                dondu=v.get("ISARET_DONDU", False),
                dej_c=v.get("DEJ_ciplak"), dej_s=v.get("DEJ_sablonlu"),
                karisik=v.get("SERH_BICIM_ICERIK_KARISIK", False) if basamak else None)


def main():
    A, pA = load("results/template_robustness_2026-08-26.json")
    B, pB = load("results/ladder_template_2026-08-27.json")
    cift = [_satir(k, v) for k, v in A["kol"].items()]
    bas = [_satir(k, v, True) for k, v in B["basamak"].items()]
    def say(S, et):
        d = sum(1 for s in S if s["hal"] == "AD-DEGISTI")
        dn = sum(1 for s in S if s["dondu"])
        print(f"{et} {len(S)} {d}"
              f"{sum(1 for s in S if s['hal']=='AYNI-AD')}"
              f"{dn} {sum(1 for s in S if s['hal']=='KAPI')}")
        return d, dn
    dc, nc = say(cift, "cift"); db, nb = say(bas, "basamak")
    assert (dc, len(cift)) == (A["sayac"]["degisti"], A["sayac"]["ayni"] + A["sayac"]["degisti"]
                               + A["sayac"]["kapi"]), "★ SAYI-GÖCÜ: cift blogu karneyle tutmuyor"
    assert db == B["sayac"]["degisti"], "★ SAYI-GÖCÜ: basamak blogu karneyle tutmuyor"
    print(f""
          f"")

    tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{\textbf{The chat template is not a nuisance parameter.} "
        r"\textbf{Carried from a separate instrument study; not a contribution here. $\\Delta U$ is defined in Appendix~\\ref{app:D7}.} "
        r"$\Delta U_{\mathrm{DOM}}$ \emph{outcome} under a raw-continuation prefix and under "
        r"the aligned model's own chat template; the base checkpoint is untouched in every "
        r"row. Outcomes: $+$/$-$ CI-disjoint positive/negative, (c) contingent, "
        r"$\sim$ null. \emph{Top:} nine end-to-end pairs, same 3 / changed 6, so the "
        r"panel outcome is \textsc{template-dependent}. \emph{Bottom:} nine checkpoint sequence "
        r"steps, each checkpoint in its own template; the \textsc{sft}$\rightarrow$"
        r"\textsc{dpo} name survives on 2 of 3 checkpoint sequences but not on T\"u3, the checkpoint sequence "
        r"on which it was first observed. $\dagger$: the base side is itself a "
        r"checkpoint with a template, so format and content are not separable there.}",
        r"\label{tab:template}",
        r"\begin{tabular}{lccrrc}", r"\toprule",
        r"& \multicolumn{2}{c}{outcome} & \multicolumn{2}{c}{$\Delta U_{\mathrm{DOM}}$} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Pair / step & raw-continuation & template & raw-continuation & template & name \\", r"\midrule",
        r"\multicolumn{6}{l}{\emph{(a) end-to-end pairs: base side is a true base model}}\\",
    ]
    for s in cift:
        tex.append("\\quad " + s["ad"].replace("·uctan-uca", "") + " & " + EN[s["ac"]] +
                   " & " + EN[s["as_"]] + f" & {s['uc']:+.3f} & {s['us']:+.3f} & " +
                   HAL[s["hal"]] + r" \\")
    tex += [r"\midrule",
            r"\multicolumn{6}{l}{\emph{(b) checkpoint sequence steps: each checkpoint in its own template}}\\"]
    for s in bas:
        AD_ING = {"taban": "base", "basamak": "stage", "kol": "run"}
        ad = s["ad"].replace("·", " ")
        for _tr, _en in AD_ING.items():
            ad = re.sub(rf"(?<![\w]){_tr}(?![\w])", _en, ad)
        ad = ad.replace("→", r"$\to$")
        dag = "$^\\dagger$" if s["karisik"] else ""
        tex.append("\\quad " + ad + dag + " & " + EN[s["ac"]] + " & " + EN[s["as_"]] +
                   f" & {s['uc']:+.3f} & {s['us']:+.3f} & " + HAL[s["hal"]] + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T2_template.tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")
    json.dump(dict(table="T2_template", n_cift=len(cift), n_basamak=len(bas),
                   cift=cift, basamak=bas, sources=[pA, pB],
                   panel_cift=A["PANEL"], panel_basamak=B["panel"],
                   sayac_cift=A["sayac"], sayac_basamak=B["sayac"],
                   SERH=("No number is computed here; both blocks are read from their cards and "
                         "each is cross-checked against its card's own counter (K-2f).")),
              open(f"{OUT}/T2_template.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"★ T2 → {OUT}/T2_template.tex  ({len(cift)} cift + {len(bas)} basamak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
