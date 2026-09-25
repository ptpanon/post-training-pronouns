#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load, t1_adi

A_CARD = "results/PREREG9_A_OLCUM_2026-08-31.json"
OLCU = (("M2", "$\\Delta M_2$"), ("M4", "$\\Delta M_4$"), ("M5", "$\\Delta M_5$"))


def ayrik(f):
    return f["ci"][0] * f["ci"][1] > 0


def _satir(k, v):
    f = v["fark"]
    return dict(ad=k, hucre={m: (f[m]["gozlenen"], ayrik(f[m])) for m, _ in OLCU},
                ci={m: f[m]["ci"] for m, _ in OLCU},
                sirala=f["M4"]["gozlenen"])


def main():
    A, pa = load(A_CARD)
    blok = {s: sorted((_satir(k, v) for k, v in A.items()
                       if isinstance(v, dict) and v.get("sinif") == s
                       and "fark" in v), key=lambda r: r["ad"])
            for s in ("BIRINCIL", "GENISLETME")}
    ana = blok["BIRINCIL"]
    say = {}
    for m, _ in OLCU:
        dus = sum(1 for r in ana if r["hucre"][m][1] and r["hucre"][m][0] < 0)
        yuk = sum(1 for r in ana if r["hucre"][m][1] and r["hucre"][m][0] > 0)
        say[m] = (dus, yuk)
    n = len(ana)
    print(f"  [PAYDA] t7: BIRINCIL={n} · GENISLETME={len(blok['GENISLETME'])} · "
          + " · ".join(f"{m}: {d}↓/{y}↑" for m, (d, y) in say.items())
          + f" ⇒ esik: BIRINCIL≠16 ise EYLEM = panel tanimi degismis, kapi düser")
    if n != 16:
        print(""); return 3

    ozet = (f"Counts are over the {n} panel models above the rule. "
            + "; ".join(f"$M_{m[-1]}$ falls in {d} of {n} and rises in {y}"
                        for m, (d, y) in say.items())
            + (", so $M_5$ carries no direction."
               if say["M5"][0] == say["M5"][1] else "."))
    L = [r"\begin{table}[tb]\centering\scriptsize\setlength{\tabcolsep}{3.2pt}",
         r"\caption{\textbf{The other three register counters, model by model.} "
         r"Per-model base$\rightarrow$aligned change on the debate prompts with "
         r"prompt-clustered $95\%$ intervals printed beside each value; \textbf{bold} "
         r"marks an interval that excludes zero. $M_2$ is addressee-directed acts, $M_4$ the first-person "
         r"volitional share, $M_5$ a distancing index; definitions are in "
         r"Appendix~\ref{app:D1}. " + ozet +
         r" The four models below the rule are not part of the "
         r"sixteen-model panel, so they are shown but not counted.}",
         r"\label{tab:otherM}",
         r"\begin{tabular}{l" + "rl" * len(OLCU) + "}", r"\toprule",
         "model & " + " & ".join(f"{b} & $95\\%$ CI" for _, b in OLCU) + r" \\",
         r"\midrule"]
    for i, s in enumerate(("BIRINCIL", "GENISLETME")):
        if i:
            L.append(r"\midrule")
        for r in blok[s]:
            hu = []
            for m, _ in OLCU:
                g, a = r["hucre"][m]
                c = r.get("ci", {}).get(m)
                hu.append(f"\\textbf{{{g:+.4f}}}" if a else f"{g:+.4f}")
                hu.append(f"$[{c[0]:+.3f},{c[1]:+.3f}]$" if c else "---")
            L.append(f"{t1_adi(r['ad'])} & " + " & ".join(hu) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T7_otherM.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    import json
    json.dump(dict(table="T7_otherM", sources=[pa],
                   payda=dict(n_birincil=n, n_genisletme=len(blok["GENISLETME"]),
                              **{f"{m}_dus_yuk": list(v) for m, v in say.items()}),
                   note="renders sealed per-model differences; computes no new statistic"),
              open(f"{OUT}/T7_otherM.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ✓ T7_otherM.tex ({len(L)} satir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
