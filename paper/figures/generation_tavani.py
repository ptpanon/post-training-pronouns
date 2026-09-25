#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/v97_generation_tavani_2026-09-19.json"
AD = {"Tulu3-8B": r"T\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}
KOKLER = [("ham", "raw"), ("template", "template"), ("urial", "in-context"), ("siradan_B1", "instructions"), ("elicit", "ELICIT-99")]


def ad(a): return AD.get(a, a)
def hiz(x): return x.get("instruct") or x.get("rl")
def tab(x): return x.get("base") or x.get("taban")
def yuz(v): return f"{100*v:.0f}"


def main():
    K, pk = load(CARD)
    S = K["kok"]
    aile = sorted(S["ham"])
    ok = len(aile) == 16 and all(a in S[k] and hiz(S[k][a]) and tab(S[k][a]) for k, _ in KOKLER for a in aile)
    uy = K.get("sayac_kayit_uyumu", {})
    if not ok or uy.get("n_bacak") != 32:
        print(f"{ok} {uy.get('n_bacak')}"); return 3
    P = {k: ([tab(S[k][a])["pay_tavan"] for a in aile], [hiz(S[k][a])["pay_tavan"] for a in aile]) for k, _ in KOKLER}
    E = S["elicit"]
    ek_h = sorted(((E[a]["instruct"]["pay_tavan"], a) for a in aile), reverse=True)
    buyuk = [(v, a) for v, a in ek_h if v >= 0.10]
    gem = [(v, a) for v, a in buyuk if a.startswith("Gemma")]
    diger = [(v, a) for v, a in buyuk if not a.startswith("Gemma")]
    kalan_max = max(v for v, a in ek_h if (v, a) not in buyuk)
    gem_sira = sorted(gem, key=lambda x: float(x[1].rsplit("-", 1)[-1].rstrip("B")))
    gem_tab = [E[a]["base"]["pay_tavan"] for _, a in gem_sira]
    liste = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else xs[0]
    cumle = (rf"In {liste([ad(a) if i == 0 else '-' + a.rsplit('-', 1)[-1] for i, (_, a) in enumerate(gem_sira)])} the aligned ELICIT-99 outputs reach the $1{{,}}024$-token "
             rf"cap in {liste([yuz(v) for v, _ in gem_sira])}\% of draws (the base outputs in {liste([yuz(v) for v in gem_tab])}\%), "
             r"so part of their density there is read on truncated text"
             + (rf"; {liste([ad(a) + chr(39) + 's (' + yuz(v) + chr(92) + '%)' for v, a in diger])} is the only other aligned checkpoint above a tenth" if len(diger) == 1 else
                (rf"; {liste([ad(a) + ' (' + yuz(v) + chr(92) + '%)' for v, a in diger])} are the other aligned checkpoints above a tenth" if diger else ""))
             + rf", and every other aligned checkpoint stays under ${100*kalan_max:.0f}\%$")
    if kalan_max < 0.01:
        cumle = cumle.replace(rf"stays under ${100*kalan_max:.0f}\%$", r"stays under $1\%$")
    io.open(f"{OUT}/ELICIT_TAVAN_CUMLE.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/generation_tavani.py)\n" + cumle)
    md = lambda xs: float(np.median(xs))
    rng = lambda xs: f"${100*min(xs):.0f}$--${100*max(xs):.0f}\\%$"
    L = [r"\paragraph*{Generation caps, and how often a generation reaches one.}",
         r"\vekalet{The debate prompts, in all three prompt formats, and the ordinary instructions are generated with a cap of $160$ new "
         r"tokens, and ELICIT-99 with $1{,}024$. Table~\ref{tab:tavan} gives, for each model and checkpoint, the share of generations that reach "
         r"the cap, counted by re-tokenizing each output with its own checkpoint's tokenizer and counting a generation that ends within "
         r"three tokens of the cap. On ELICIT-99, where the generator also recorded its own token count, the two agree in at least "
         rf"${100*uy['en_dusuk_2']:.0f}\%$ of generations on {uy['n_bacak'] - 1} of the {uy['n_bacak']} generation sets; the exception is one base checkpoint "
         r"whose recorded count is itself unusable. "
         rf"\textbf{{Most debate-prompt generations run to the cap on both checkpoints}}: on raw continuation a median ${yuz(md(P['ham'][0]))}\%$ of "
         rf"base and ${yuz(md(P['ham'][1]))}\%$ of aligned generations do ({rng(P['ham'][0])} and {rng(P['ham'][1])}), so rates on this "
         r"prompt set are rates over the first $160$ tokens of a continuation that would have run on. "
         rf"On the ordinary instructions the aligned answers reach the cap in {rng(P['siradan_B1'][1])} of cases, and the base "
         rf"continuations in {rng(P['siradan_B1'][0])} before they are cut at their first code fence or new query. " + cumle + ".}",
         r"\begin{table}[tb]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}",
         r"\caption{\textbf{Share of generations that reach the token cap} (\%), base and aligned outputs, counted by re-tokenizing each output "
         r"with its own checkpoint's tokenizer. Cap $160$ new tokens except ELICIT-99 ($1{,}024$). Raw, template and in-context are the "
         r"three formats of the debate prompts; the in-context aligned checkpoint reads the identical string. Instruction base outputs are "
         r"counted before they are cut.}\label{tab:tavan}",
         r"\begin{tabular}{l" + "rr" * len(KOKLER) + r"}\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{b}}}" for _, b in KOKLER) + r" \\",
         "".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(KOKLER))),
         "model & " + " & ".join(["base & al."] * len(KOKLER)) + r" \\ \midrule"]
    for i, a in enumerate(aile):
        L.append(ad(a) + " & " + " & ".join(f"{yuz(P[k][0][i])} & {yuz(P[k][1][i])}" for k, _ in KOKLER) + r" \\")
    L += [r"\midrule median & " + " & ".join(f"{yuz(md(P[k][0]))} & {yuz(md(P[k][1]))}" for k, _ in KOKLER) + r" \\",
          r"\bottomrule\end{tabular}\end{table}"]
    io.open(f"{OUT}/TAVAN_D1.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/generation_tavani.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="TAVAN_D1", ciktilar=["TAVAN_D1.tex", "ELICIT_TAVAN_CUMLE.tex"], sources=[pk],
                   payda=dict(n_model=16, kok=[k for k, _ in KOKLER], medyan={k: [round(md(P[k][0]), 4), round(md(P[k][1]), 4)] for k, _ in KOKLER},
                              elicit_hizali_buyuk={a: v for v, a in buyuk}, sayac_kayit_uyumu=uy),
                   note="renders the R5 cap card; no statistic computed here"),
              io.open(f"{OUT}/TAVAN_D1.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(KOKLER)} {md(P['ham'][0]):.2f} {md(P['ham'][1]):.2f}"
          f"{[(a, v) for v, a in buyuk]} {uy}")
    print("✓ TAVAN_D1.tex · ELICIT_TAVAN_CUMLE.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
