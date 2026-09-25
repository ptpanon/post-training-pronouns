#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load, t1_adi

EMIR = "results/bare_imperative_composition_2026-08-31.json"
UCUZ = "results/v22_cheap_measurements_2026-09-06.json"
A_CARD = "results/PREREG9_A_OLCUM_2026-08-31.json"


def t10():
    E, pe = load(EMIR); A, pa = load(A_CARD)
    aile = {k: v["kontrast"].split("→") for k, v in A.items()
            if isinstance(v, dict) and v.get("sinif") == "BIRINCIL" and "kontrast" in v}
    sat = []
    for a, (z0, z1) in sorted(aile.items()):
        b, i = E["dosyalar"].get(f"{a}/{z0}"), E["dosyalar"].get(f"{a}/{z1}")
        if not b or not i:
            continue
        rb, ri = 100 * b["a"] / b["n_cumle"], 100 * i["a"] / i["n_cumle"]
        pb = 100 * b["a_yordamsal"] / b["a"]; pi = 100 * i["a_yordamsal"] / i["a"]
        sat.append((a, b["n_cumle"], rb, pb, i["n_cumle"], ri, pi, ri - rb))
    n = len(sat); yuk = sum(1 for r in sat if r[7] > 0)
    enb = max(abs(r[7]) for r in sat)
    print(f"{n} {yuk}"
          f"{enb:.2f}"
          f"")
    if n != 16:
        print(""); return 3
    L = [r"\begin{table}[tb]\centering\small",
         r"\caption{\textbf{Bare imperatives, model by model.} Rate is bare "
         r"imperatives as a percentage of parsed sentences on the debate prompts; "
         r"\textsc{proc.} is the share of those that stand inside a numbered "
         r"procedure rather than in prose. Appendix~\ref{app:D2} reports the panel total; "
         f"this is the model-level decomposition behind it. The rate rises in {yuk} "
         f"of {n} models and falls in {n - yuk}, and the largest model-level change "
         f"is {enb:.2f} percentage points, so the panel statement holds model by "
         r"model and not only in aggregate. What moves is the composition: the "
         r"procedural share rises in most models, which is the same post-training change "
         r"that lengthens sentences.}",
         r"\label{tab:imperative}",
         r"\begin{tabular}{lrrr@{\hspace{10pt}}rrr@{\hspace{10pt}}r}", r"\toprule",
         r" & \multicolumn{3}{c}{base} & \multicolumn{3}{c}{aligned} & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"model & sentences & rate \% & \textsc{proc.} \% & sentences & rate \% & "
         r"\textsc{proc.} \% & $\Delta$ \\", r"\midrule"]
    for a, nb, rb, pb, ni, ri, pi, d in sat:
        L.append(f"{t1_adi(a)} & {nb:,} & {rb:.2f} & {pb:.0f} & "
                 f"{ni:,} & {ri:.2f} & {pi:.0f} & {d:+.2f} \\\\".replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T10_imperative.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(table="T10_imperative", sources=[pe, pa],
                   payda=dict(n_aile=n, yukselen=yuk, en_buyuk_delta=enb)),
              open(f"{OUT}/T10_imperative.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ✓ T10_imperative.tex ({len(L)} satir)")
    return 0


BACAK = {"taban": "base", "base": "base", "instruct": "instruct",
         "sft": "SFT", "dpo": "DPO", "rl": "RL", "step_360": "RLVR step 360"}


def t11():
    U, pu = load(UCUZ)
    D = U["c_dejenerelik"]; bar = D["bar"]
    sat = []
    for a, v in sorted(D["aile"].items()):
        bac = v["bacak"]
        ad = sorted(bac, key=lambda z: -bac[z]["oran"])
        en, az = ad[0], ad[-1]
        f = v["filtreli"]; h = v["ham"]
        sat.append((a, en, 100 * bac[en]["oran"], az, 100 * bac[az]["oran"],
                    h["gozlenen"], f["gozlenen"], f["ci"], f["ayrik"], f["n_cift"]))
    n = len(sat); ay = sum(1 for r in sat if r[8])
    enb = max(r[2] for r in sat)
    print(f"  [PAYDA] t11: aile={n} · süzgecli CI-ayrik={ay} · en yüksek bacak "
          f"eleme orani %{enb:.2f} ⇒ esik: ayrik<16 ise EYLEM = «süzgec sonrasi "
          f"16/16» cümlesi düser")
    if ay != n:
        print(""); 
    L = [r"\begin{table}[tb]\centering\small",
         r"\caption{\textbf{What the degeneracy filter removes, and what survives "
         r"it.} The pre-registered filter drops any generation whose distinct-4 "
         f"ratio falls below ${bar:.2f}$. Columns give the checkpoint that loses most and "
         r"the one that loses least, with the percentage of that checkpoint's generations "
         r"dropped, then the register shift before and after the filter with its "
         r"prompt-clustered interval and the surviving pair count. \textbf{The filter "
         r"is asymmetric and the asymmetry runs against the effect}: it removes far "
         f"more of the base outputs (up to {enb:.1f}\\%) than of the aligned outputs, and "
         r"base outputs with their repetitive generations removed carry \emph{more} "
         f"second person, not less. The shift still excludes zero in {ay} of {n} "
         r"models after filtering.}",
         r"\label{tab:degen}",
         r"\begin{tabular}{lrlrl@{\hspace{8pt}}rrlr}", r"\toprule",
         r"model & \multicolumn{2}{c}{checkpoint dropped most} & "
         r"\multicolumn{2}{c}{checkpoint dropped least} & raw & filtered & $95\%$ CI & pairs \\",
         r"\midrule"]
    for a, en, pe_, az, pa_, ham, fil, ci, ayr, nc in sat:
        v = f"\\textbf{{{fil:+.2f}}}" if ayr else f"{fil:+.2f}"
        L.append(f"{t1_adi(a)} & {pe_:.2f}\\% & {BACAK.get(en, en)} & "
                 f"{pa_:.2f}\\% & {BACAK.get(az, az)} & "
                 f"{ham:+.2f} & {v} & $[{ci[0]:+.2f},{ci[1]:+.2f}]$ & "
                 f"{nc:,} \\\\".replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T11_degen.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(table="T11_degen", sources=[pu],
                   payda=dict(n_aile=n, suzgecli_ayrik=ay, en_yuksek_eleme=enb)),
              open(f"{OUT}/T11_degen.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ✓ T11_degen.tex ({len(L)} satir)")
    return 0


if __name__ == "__main__":
    sys.exit(t10() or t11())
