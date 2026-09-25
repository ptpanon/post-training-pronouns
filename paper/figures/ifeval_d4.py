#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json

KOK = __DNH_ROOT__ + ""
CARD = f"{KOK}/results/ifeval_kayit_2026-09-24.json"
FIG = f"{KOK}/paper/figures"
BAS = "% * URETILDI - elle duzenlenmez (figures/ifeval_d4.py)\n"
KOSMAYAN = ("Llama-3.1-70B", "Qwen2.5-72B")
AD = {"kadran-KISI_ASAGI": "targeted run, lower-rate chosen", "kadran-KISI_YUKARI": "targeted run, higher-rate chosen",
      "kadran-RASTGELE": "targeted run, random pairs", "Tulu3-8B-SFT": "T\\\"ulu-3-8B SFT checkpoint",
      "Tulu3-8B": "T\\\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def yaz(ad, g):
    io.open(f"{FIG}/{ad}", "w", encoding="utf-8").write(BAS + g.strip() + "\n")
    print(f"✓ {ad} ({len(g.split())} kelime)")


def main():
    K = json.load(io.open(CARD, encoding="utf-8"))
    M, P = K["model"], K["panel"]
    hiz = sorted([m for m in M if not m.startswith("kadran-") and m != "Tulu3-8B-SFT"],
                 key=lambda x: M[x]["ps"]["delta"])
    n = P["ps"]["n"]
    dus = sum(1 for m in hiz if M[m]["ps"]["delta"] < 0)
    ayr = P["ps"]["ayrik_negatif"]; poz = P["ps"]["ayrik_pozitif"]
    med = P["ps"]["medyan"]
    assert len(hiz) == n, (len(hiz), n)
    assert dus == n, f"sahip «{n} of {n}» dedi, kartta düsen {dus} ⇒ YAZILMAZ"
    assert poz == 0, f"sahip «none rising» dedi, kartta ayrik-pozitif {poz} ⇒ YAZILMAZ"
    print(f"  [PAYDA] ifeval_d4: n_model={n} · hal_dusen={dus} · hal_ayrik={ayr} · "
          f"red_ayrik_pozitif={poz} · medyan={med} ⇒ esik: düsen≠n ya da pozitif>0 ise YAZILMAZ")

    import re as _re
    PANEL = json.load(open(f"{KOK}/results/v103_diskten_2026-09-23.json", encoding="utf-8"))["e"]["_kaynak"]["panel16"]
    boy = lambda a: float(_re.search(r"(\d+(?:\.\d+)?)B", a).group(1))
    yok = sorted(set(PANEL) - set(hiz), key=boy)
    assert len(PANEL) == 16 and set(hiz) <= set(PANEL) and len(yok) == 16 - n, (len(PANEL), yok)
    assert sorted(yok, key=boy) == sorted(PANEL, key=boy)[-len(yok):], f"{yok}"
    print(f"  [PAYDA] ifeval_d4 payda: panel {len(PANEL)} · okunan {n} · kosulmayan {yok} (en büyük {len(yok)}) ⇒ esik: degilse YAZILMAZ")
    yaz("IFEVAL_CUMLE.tex",
        f"The sentence is not free. On the Instruction-Following Evaluation benchmark (IFEval, \\citealp{{zhou2023ifeval}}) it lowers "
        f"prompt-level strict accuracy in "
        f"{dus} of the {len(PANEL)} models (the {['', 'largest', 'two largest', 'three largest'][len(yok)]} not run; median ${med:.2f}$, {ayr} with intervals excluding zero, "
        f"none rising; Appendix~\\ref{{app:D4}})")

    yaz("IFEVAL_LIMIT.tex",
        f"The IFEval cost of the restoring sentence (\\S\\ref{{sec:deperson}}) is to verifiable instruction following, not to "
        f"helpfulness, which we did not measure. The sentence also asks for a continuation where "
        f"IFEval poses a task, a mismatch we did not separate from the cost.")

    L = [r"\paragraph*{What the restoring sentence costs on IFEval.}",
         r"\vekalet{The pronoun-free register sentence of \S\ref{sec:deperson} was prepended to each "
         r"IFEval prompt inside the model's own chat template, greedy, one sample, and both conditions were "
         r"scored with the official strict and loose evaluators over its $541$ prompts. Intervals are "
         r"prompt-clustered ($2{,}000$ resamples, prompts resampled and the difference taken within "
         r"draw). Table~\ref{tab:ifeval} gives the per-model reading: every condition falls, and none "
         r"rises with an interval excluding zero.}",
         r"\begin{table}[t]\centering\small",
         r"\caption{\textbf{IFEval prompt-level strict accuracy with and without the restoring "
         r"sentence.} Plain is the IFEval prompt alone; \emph{+sentence} prepends it. "
         r"$\Delta$ is aligned-with-sentence minus plain, with a $95\%$ prompt-clustered interval. "
         r"Bold marks an interval excluding zero. The three targeted runs and the SFT checkpoint are read "
         r"through the T\"ulu-3-8B template.}",
         r"\label{tab:ifeval}",
         r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"model & plain & +sentence & $\Delta$ (95\% CI) \\", r"\midrule"]
    for m in hiz:
        r = M[m]["ps"]
        d = f"${r['delta']:+.3f}$ [${r['ci'][0]:+.3f}$, ${r['ci'][1]:+.3f}$]"
        L.append(f"{AD.get(m, m)} & ${r['duz']:.3f}$ & ${r['kayit']:.3f}$ & "
                 + (f"\\textbf{{{d}}}" if r["ayrik"] else d) + r" \\")
    L.append(r"\midrule")
    for m in ("kadran-KISI_ASAGI", "kadran-KISI_YUKARI", "kadran-RASTGELE", "Tulu3-8B-SFT"):
        if m not in M:
            continue
        r = M[m]["ps"]
        d = f"${r['delta']:+.3f}$ [${r['ci'][0]:+.3f}$, ${r['ci'][1]:+.3f}$]"
        L.append(f"{AD[m]} & ${r['duz']:.3f}$ & ${r['kayit']:.3f}$ & "
                 + (f"\\textbf{{{d}}}" if r["ayrik"] else d) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    L.append(r"\vekalet{\textbf{Two limits of this reading.} "
             + f"First, {KOSMAYAN[0]} and {KOSMAYAN[1]} were not run, for time, so the panel here is "
             + f"{n} of the sixteen models; the other counts in this paper are unaffected. "
             + r"Second, the official evaluator fills a constraint parameter at random whenever the "
             r"benchmark leaves it unspecified, so an IFEval accuracy depends on the order in which "
             r"cells are scored unless the seed is fixed: unseeded, two runs of this same panel moved "
             r"the median by about $0.002$, one prompt in $541$. We fix the seed, and two full runs "
             r"then agree exactly.}")
    yaz("IFEVAL_D4.tex", "\n".join(L))

    json.dump(dict(table="IFEVAL_D4",
                   sources=[{"yol": "results/ifeval_kayit_2026-09-24.json"}],
                   ciktilar=["IFEVAL_CUMLE.tex", "IFEVAL_LIMIT.tex", "IFEVAL_D4.tex"],
                   n_model=n, medyan_ps=med, ayrik=ayr),
              io.open(f"{FIG}/IFEVAL_D4.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
