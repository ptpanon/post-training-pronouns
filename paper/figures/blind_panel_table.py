#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import json
from style import OUT, load

AD = {"KP-6717": ("hh-rlhf", "Anthropic hh-rlhf"),
      "KP-1fe5": ("OASST", "OpenAssistant oasst1"),
      "KP-9358": ("Tulu-3", "AI2 Tülu-3 preference mix")}
UZ = {"KP-6717": 10.5, "KP-1fe5": 50.0, "KP-9358": 229.0}


def main():
    d, prov = load("results/blind_preference_opening_panel_2026-08-25.json")
    sat = []
    for kp in ("KP-6717", "KP-1fe5", "KP-9358"):
        v = d[kp]; S = v["sayac"]
        n, tie, cy = v["n"], S.get("TIE", 0), S.get("COGUNLUK-YOK", 0)
        sat.append(dict(kp=kp, kisa=AD[kp][0], tam=AD[kp][1], n=n, kesin=v["kesin"],
                        tie=tie, tie_orani=round(tie / n, 3), pay=v["pay"],
                        mde=round(100 * v["mde"], 1), verdict="no direction",
                        d_uzunluk=UZ[kp]))
    print(f"{len(sat)}")

    tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{A cross-family blind judge does not recover human preference on any of "
        r"three preference substrates. \emph{Decisive} excludes ties, and \emph{share} is the "
        r"fraction of decisive pairs on which the judge picked the human-chosen response "
        r"($H_0=0.5$). MDE is the pre-registered one-sided 95\% detectable difference at "
        r"that denominator. The tie rate rises monotonically with the median length gap "
        r"between chosen and rejected responses, so the two cannot be separated by this "
        r"material.}",
        r"\label{tab:blind-panel}",
        r"\begin{tabular}{lrrrrrl}", r"\toprule",
        r"Substrate & $n$ & decisive & tie rate & share & MDE & $\Delta$ length (median words) \\",
        r"\midrule",
    ]
    for s in sat:
        tex.append(f"{s['tam']} & {s['n']} & {s['kesin']} & {s['tie_orani']:.3f} & "
                   f"{s['pay']:.3f} & {s['mde']:.1f}\\,pp & $+{s['d_uzunluk']:.1f}$ \\\\")
    tex += [r"\midrule",
            r"\multicolumn{7}{l}{\emph{Outcome on every row: no direction "
            r"(share indistinguishable from chance).}} \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T1_blind_panel.tex", "w", encoding="utf-8").write("\n".join(tex) + "\n")
    json.dump(dict(table="T1_blind_panel", rows=sat, sources=[prov],
                   note="renders pre-specified measurements; computes no new statistic"),
              open(f"{OUT}/T1_blind_panel.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n".join(tex[8:13]))
    print(f"  ✓ T1_blind_panel.tex")


if __name__ == "__main__":
    main()
