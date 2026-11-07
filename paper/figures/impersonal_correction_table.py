#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load, t1_adi

NOTR = "results/v20_panel_2026-09-06.json"
ELIC = "results/v97_elicited_impersonal_correction_b15_2026-09-19.json"


def main():
    N, pn = load(NOTR); E, pe = load(ELIC)
    A, B = N["aile"], E["aile"]
    sat = []
    for k, v in A.items():
        z0, z1 = v["kontrast"].split("→")
        ham = v[z0]["duzeltme"] + v[z1]["duzeltme"]
        e = B.get(k)
        sat.append(dict(ad=k, d=v["dM3"], ci=v["dM3_ci"], ham=ham,
                        nb=v[z0]["M3"], ni=v[z1]["M3"],
                        eb=(e or {}).get("M3_base"), ei=(e or {}).get("M3_instruct"),
                        ed=(e or {}).get("dM3"), eci=(e or {}).get("dM3_ci"),
                        eham=((e or {}).get("duzeltme_base", 0) + (e or {}).get("duzeltme_instruct", 0)) if e else None))
    sat.sort(key=lambda r: -r["d"])
    ay = lambda c: c is not None and c[0] * c[1] > 0
    nA = sum(1 for r in sat if ay(r["ci"])); nAu = sum(1 for r in sat if ay(r["ci"]) and r["d"] > 0)
    nB = sum(1 for r in sat if ay(r["eci"])); nBu = sum(1 for r in sat if ay(r["eci"]) and r["ed"] > 0)
    nBp = sum(1 for r in sat if r["ed"] is not None)
    tb = [r["nb"] for r in sat]; ti = [r["ni"] for r in sat]
    print(f"  [PAYDA] t8: aile={len(sat)} · nötr ayrik={nA} (yukari {nAu}) · "
          f"elicited ölcülen={nBp} ayrik={nB} (yukari {nBu}) · ham düzeltme "
          f"toplam={sum(r['ham'] for r in sat):,.0f} · taban M3 "
          f"{min(tb):.3f}–{max(tb):.3f} · hizali M3 {min(ti):.3f}–{max(ti):.3f} "
          f"⇒ esik: taban bandi 0,85'in altina inerse EYLEM = «taban da kisisiz "
          f"düzeltir» cümlesi daralir")
    L = [r"\begin{table}[tb]\centering\scriptsize\setlength{\tabcolsep}{2.8pt}",
         r"\caption{\textbf{Where corrections lose their addressee, model by model.} "
         r"$\Delta M_3$ is the base$\rightarrow$aligned change in the share of "
         r"corrections that name no person, with prompt-clustered $95\%$ intervals; "
         r"\textbf{bold} marks an interval excluding zero. \textsc{corr.} is the "
         r"number of correction sentences the counter flagged across both checkpoints---the "
         r"denominator the share is taken over, without which the share cannot be "
         r"read. \textbf{The base column is the reference the change is read against}: "
         r"impersonal correction is already the base checkpoints' default "
         f"({min(tb):.2f}--{max(tb):.2f} across the sixteen), so what alignment moves "
         f"is a share that starts near its ceiling. "
         f"On the debate prompts the share rises in {nAu} models and falls in "
         f"{nA - nAu}, and does not separate in {len(sat) - nA}. On ELICIT-99, which "
         f"scores {nBp} models, {nB} separate; the one not measured there falls "
         r"to the degeneracy shelf on its aligned outputs. "
         r"The two prompt sets are not the same object: they differ in stimulus and, for "
         r"most models, in prompt protocol (Table~\ref{tab:families}).}",
         r"\label{tab:m3}",
         r"\begin{tabular}{lrrrlr@{\hspace{8pt}}rrrlr}", r"\toprule",
         r" & \multicolumn{5}{c}{debate prompts} & \multicolumn{5}{c}{ELICIT-99} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"model & base & aligned & $\Delta M_3$ & $95\%$ CI & \textsc{corr.} & "
         r"base & aligned & $\Delta M_3$ & $95\%$ CI & \textsc{corr.} \\",
         r"\midrule"]
    for r in sat:
        a = f"\\textbf{{{r['d']:+.3f}}}" if ay(r["ci"]) else f"{r['d']:+.3f}"
        ac = f"$[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]$"
        if r["ed"] is None:
            b = r"\multicolumn{5}{c}{\textsc{not measured}}"
        else:
            bb = f"\\textbf{{{r['ed']:+.3f}}}" if ay(r["eci"]) else f"{r['ed']:+.3f}"
            b = (f"{r['eb']:.3f} & {r['ei']:.3f} & {bb} & "
                 f"$[{r['eci'][0]:+.3f},{r['eci'][1]:+.3f}]$ & {r['eham']:,.0f}")
        L.append(f"{t1_adi(r['ad'])} & {r['nb']:.3f} & {r['ni']:.3f} & "
                 f"{a} & {ac} & {r['ham']:,.0f} & {b} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T8_m3.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    import json
    json.dump(dict(table="T8_m3", sources=[pn, pe],
                   payda=dict(n_aile=len(sat), notr_ayrik=nA, notr_yukari=nAu,
                              elicit_olculen=nBp, elicit_ayrik=nB, elicit_yukari=nBu),
                   note="renders measured differences; computes no new statistic"),
              open(f"{OUT}/T8_m3.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ T8_m3.tex ({len(L)} satir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
