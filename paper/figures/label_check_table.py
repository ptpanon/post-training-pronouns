#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys, json, os
ROOT = __DNH_ROOT__ + ""
KAYNAK = f"{ROOT}/results/correction_counters_label_validation_2026-09-01.json"
OUT = os.path.dirname(os.path.abspath(__file__))
AD = {"claude": "Primary judge", "qwen": "Cross-check judge",
      "insan": "Authors (12 items)"}


def main():
    S = json.load(open(KAYNAK, encoding="utf-8"))
    L = [r"\begin{table}[tb]\centering\small\setlength{\tabcolsep}{2.5pt}",
         r"\caption{\textbf{Register counters against act-level labels.} The "
         r"correction counter and the impersonality share, checked against the "
         r"$580$ act-level labels on responses to ELICIT-99 and to neutral control prompts. \textsc{det.}: does a "
         r"correction occur (counter $\ge 1$ correction sentence vs.\ the "
         r"rubric's yes/no). \textsc{imp.}: among responses where both detect a "
         r"correction, is it impersonal. Degenerate margins are flagged: where "
         r"one cell of the table is empty, $\kappa$ has no base and the raw "
         r"agreement is the readable number.}\label{tab:labelcheck}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"contrast & labeller & both yes & rubric only & counter only & both no"
         r" & agree & $\kappa$ \\", r"\midrule"]
    dejenere = []
    for kip, alan, etiket in (("det.", "A_duzeltme_tespiti", "correction detected"),
                              ("imp.", "B_kisisizlik", "impersonal")):
        for kol in ("claude", "qwen", "insan"):
            m = S[kol][alan]
            if not isinstance(m, dict):
                L.append(f"{kip} & {AD[kol]} & \\multicolumn{{6}}{{l}}{{"
                         r"\textsc{not measurable}: the impersonality question "
                         r"was not put to this labeller} \\")
                continue
            bay = ""
            if m["her_ikisi_HAYIR"] == 0 or m["her_ikisi_EVET"] == 0:
                bay = r"$^{\dagger}$"; dejenere.append(f"{kip}/{kol}")
            k = "--" if m["kappa"] is None else f"{m['kappa']:+.2f}{bay}"
            L.append(f"{kip} & {AD[kol]} & {m['her_ikisi_EVET']} & "
                     f"{m['rubrik_EVET_sayac_HAYIR']} & {m['rubrik_HAYIR_sayac_EVET']} & "
                     f"{m['her_ikisi_HAYIR']} & {m['uyum']:.2f} & {k} \\\\")
        if kip == "det.":
            L.append(r"\midrule")
    _ti = S["insan"]; _tb = _ti["tabaka"]
    assert sum(_tb.values()) == _ti["A_duzeltme_tespiti"]["n"], (_tb, _ti["A_duzeltme_tespiti"]["n"])
    L += [r"\midrule",
          r"\multicolumn{8}{p{.95\linewidth}}{\footnotesize $^{\dagger}$Degenerate "
          r"margin: one cell is empty, so $\kappa$ is not interpretable and the "
          r"agreement column is the readable number. The authors' row covers the "
          rf"${_ti['A_duzeltme_tespiti']['n']}$ of the authors' $50$ items on which the correction question was "
          rf"put (${_tb['ANLASMAZLIK']}$ where the two judges disagreed, ${_tb['ANLASMA-EVET']}$ where both said yes and "
          rf"${_tb['ANLASMA-HAYIR']}$ where both said no). The $50$ are disagreement-enriched ($40$ were "
          r"sampled where the two judges disagreed), so that row is agreement "
          r"\emph{within the disagreement stratum}, not field accuracy. The "
          r"impersonality question was never put to the authors.}\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T5_labelcheck.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"  [PAYDA] t5_labelcheck: n_kip=2 · n_kol=3 · hal_dejenere={len(dejenere)} "
          f"({', '.join(dejenere)}) · red_eksik=0")
    print(f"  ✓ T5_labelcheck.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
