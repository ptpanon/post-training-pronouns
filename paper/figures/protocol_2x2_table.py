#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys, json, io, os
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load
import urial_verdict_loader

KAY = __DNH_ROOT__ + "/results/template_filtered_2026-09-07.json"
V23_CARD = "results/v23_template_count_2026-09-06.json"
KIRILGAN = {"Qwen2.5-1.5B"}
AYNI_TABAN = {"Tulu3-8B": "Llama-3.1-8B", "Llama-3.1-8B": "Tulu3-8B"}
EN = {"Tulu3-8B": "T\\\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def ci(v):
    return r"$%+.2f$ {\scriptsize[$%+.2f$,$%+.2f$]}" % (v["dM1"], v["ci"][0], v["ci"][1])


def uci(v):
    return r"$%+.2f$ {\scriptsize[$%+.2f$,$%+.2f$]}" % (v["dM1"], v["ci"][0], v["ci"][1])


def nk(v):
    lo, hi = v["ci"][0], v["ci"][1]
    ayr = bool(v.get("ayrik", lo * hi > 0))
    return (r"$\mathbf{%+.2f}$" if ayr else r"$%+.2f$") % v["dM1"]


def main():
    d = json.load(io.open(KAY, encoding="utf-8"))
    UX, pux = urial_verdict_loader.yukle(urial_verdict_loader.CARD_2048)
    Xail, Xsay, Xkap = urial_verdict_loader.kiyas(UX, "i")
    Uail, Usay, Ukap = Xail, Xsay, Xkap
    R = [r for r in d["aileler"] if r["hal"] == "ÖLCÜLDÜ"]
    R.sort(key=lambda r: -r["sbl"]["dM1"])
    _suz_poz = sum(1 for r in R if r["sbl"]["dM1_suzgecsiz"] > 0)
    _suz_neg = len(R) - _suz_poz
    _V23, p23 = load(V23_CARD)
    _B = _V23["B_sablonlu_delta"]
    _cu = sum(1 for v in _B.values() if v["ci"][0] > 0)
    _cd = sum(1 for v in _B.values() if v["ci"][1] < 0)
    _cn = len(_B) - _cu - _cd
    print(f"{len(_B)} {_cu}"
          f"{_cd} {_cn}"
          f"")
    _donen = [EN.get(r["aile"], r["aile"]) for r in R
              if (r["sbl"]["dM1"] > 0) != (r["sbl"]["dM1_suzgecsiz"] > 0)]
    _tb = {r["aile"]: r["sbl"]["M1_taban"] for r in R}
    _fark = abs(_tb.get("Tulu3-8B", 0) - _tb.get("Llama-3.1-8B", 0))
    print(f"{len(R)} {_suz_poz}"
          f"{_suz_neg} {len(_donen)} {_donen}"
          f"{_fark:.2f}"
          f"")
    _t8 = next(r for r in R if r["aile"] == "Tulu3-8B")["sbl"]
    _l8 = next(r for r in R if r["aile"] == "Llama-3.1-8B")["sbl"]
    _snote = (r"$^{\S}$T\"ulu-3-8B and Llama-3.1-8B descend from the "
              r"\emph{same} base checkpoint, and \textbf{the two pipelines answer at "
              "$%.2f$ and $%.2f$ second persons per thousand in their own formats} "
              "--- the sign here is set by the pipeline, not by the base weights, and "
              "that statement needs no base outputs. Read against the base outputs instead, "
              "the same weights differ by $%.2f$ points between the two chat "
              "formats, but that checkpoint is prompted through a template it never saw "
              "and invents turn markers of its own, so we do not rest on it. "
              % (_t8["M1_hizali"], _l8["M1_hizali"], _fark)
              + r"\emph{dgn.} is the base outputs' degeneracy share under the "
                r"template. Read \emph{without} the four-quadrant filter the "
                "chat-template column is %d up and %d down \\emph{by sign}, or "
                "%d up, %d down and %d null once each interval is read; the only "
                "model whose sign changes is %s --- the row already marked "
                "$\\dagger$ --- so the filter does not create the split. "
                "\\textbf{Sign counts and interval counts are different numbers} and "
                "both are given here, because the filtered column above (7/7/2) is "
                "classified by interval. "
              % (_suz_poz, _suz_neg, _cu, _cd, _cn,
                 ", ".join(_donen) if _donen else "none"))
    _kirpad = sorted(a for a, v in Xail.items() if v.get("AD") == "ÖLCÜLEMEZ-KIRPMA")
    _kol_lafzi = ("Read on the six single-strength prefix conditions, where the prompt fits the "
                  "generator's window," if UX.get("kol_kipi") == "x1" else
                  "Read on all sixteen prefix conditions at the $%s$-token window, where no prompt "
                  "is truncated," % "{:,}".format(UX["kirpma"]["max_length"]).replace(",", "{,}"))
    _uout = (r"It carries the same distinct-4 floor per checkpoint and no checkpoint was dropped "
             r"by it. " + _kol_lafzi + r" its outcome is %d down, %d up and %d "
             r"\textsc{format-only} across %d of %d models%s."
             r"}\label{tab:twobytwo}"
             % (Xsay["asagi"], Xsay["ters"], Xsay["bicim"],
                Xkap["n_olculen"] - len(_kirpad), Xkap["n_aile"],
                (r"; %s exceeds the window on its own tokenizer and is "
                 r"\textsc{unmeasurable-truncated}"
                 % ", ".join(a.replace("-", "--") for a in _kirpad)) if _kirpad else ""))
    _M = "Read against the base outputs instead,"
    _s_bulgu, _s_serh = _snote.split(_M, 1)
    _s_serh = _M + " " + _s_serh
    _u_govde = _uout.replace(r"}\label{tab:twobytwo}", "")

    F = [r"\begin{table}[t]\centering\scriptsize"
         r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{The same contrast under three prompt protocols.} "
         r"Every model with its "
         r"prompt-clustered $95\%$ CI, plus the two template levels and the "
         r"retention columns; Figure~\ref{fig:split} draws the raw-continuation and "
         r"chat-template contrasts as levels, in the row order of Table~\ref{tab:families}. $\Delta M_1$ is instruct $-$ base in second-person "
         r"forms per thousand tokens. A cell (prefix condition $\times$ prompt $\times$ draw) "
         r"enters the first two columns only if it is non-degenerate in all four "
         r"quadrants, so those two are read on the \emph{same} cells; "
         r"\emph{kept} is the surviving share and \emph{dgn.} the base outputs' "
         r"degeneracy share under the template. The daggered rows, the task "
         r"prompt set and the unfiltered reading are below."
         r"}\label{tab:twobytwo-full}",
         r"\begin{tabular}{l r r r r r r r}",
         r"\toprule",
         r" & \multicolumn{2}{c}{$M_1$ under template} & "
         r"\multicolumn{3}{c}{$\Delta M_1$ (instruct $-$ base)} & & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
         r"model & base & inst. & raw-continuation & chat-template & task$^{\ast}$ & "
         r"dgn. & kept \\",
         r"\midrule"]
    SAT = []
    for r in R:
        ad = EN.get(r["aile"], r["aile"])
        if r["aile"] in KIRILGAN:
            ad += r"$^\dagger$"
        u = Uail[r["aile"]]
        _im = r"$^{\S}$" if r["aile"] in AYNI_TABAN else ""
        F.append(r"%s%s & %.1f & %.1f & %s & %s & %s & %.2f & %.0f\%% \\" % (
            ad, _im, r["sbl"]["M1_taban"], r["sbl"]["M1_hizali"],
            ci(r["cip"]), ci(r["sbl"]), uci(u),
            r["hucre"]["sbl_taban"]["dej"], 100 * r["tutulan_oran"]))
        SAT.append((ad + _im, nk(r["cip"]), nk(r["sbl"]), nk(u)))
    ns = sum(1 for r in R if r["sbl"]["ayrik"] and r["sbl"]["dM1"] > 0)
    nn = sum(1 for r in R if r["sbl"]["ayrik"] and r["sbl"]["dM1"] < 0)
    nu = sum(1 for r in R if not r["sbl"]["ayrik"])
    nc = sum(1 for r in R if r["cip"]["ayrik"] and r["cip"]["dM1"] < 0)
    _OZET = (r"raw-continuation: %d/%d disjoint and down. chat template: %d up, %d down, %d null. "
             r"task: %d down, %d up, %d format-only." % (
                 nc, len(R), ns, nn, nu, Usay["asagi"], Usay["ters"], Usay["bicim"]))
    F += [r"\midrule",
          r"\multicolumn{8}{l}{\footnotesize %s} \\" % _OZET,
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    io.open(f"{OUT}/T13_full.tex", "w", encoding="utf-8").write("\n".join(F) + "\n")
    print("  ✓ T13_full.tex (Ek E)")

    _yari = (len(SAT) + 1) // 2
    L = [r"\begin{table}[t]\centering\scriptsize"
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{\textbf{The same contrast under three prompt protocols.} "
         r"$\Delta M_1$ is instruct $-$ base in second-person forms per thousand "
         r"tokens; \textbf{bold} marks a prompt-clustered $95\%$ interval "
         r"excluding zero. \emph{raw-continuation} and \emph{chat-template} are read on the same "
         r"non-degenerate cells; \emph{task}$^{\ast}$ holds the task fixed and "
         r"prompts each checkpoint in its own format. Intervals, the two template "
         r"levels, the retention columns, the daggered rows and the unfiltered "
         r"reading are in Table~\ref{tab:twobytwo-full}."
         r"}\label{tab:twobytwo}",
         r"\begin{tabular}{l rrr @{\hspace{1.3em}} l rrr}",
         r"\toprule",
         r"model & raw-continuation & templ. & task$^{\ast}$ & model & raw-continuation & templ. & "
         r"task$^{\ast}$ \\",
         r"\midrule"]
    for k in range(_yari):
        sol = SAT[k]
        sag = SAT[k + _yari] if k + _yari < len(SAT) else ("", "", "", "")
        L.append(r"%s & %s & %s & %s & %s & %s & %s & %s \\" % (sol + sag))
    L += [r"\midrule",
          r"\multicolumn{8}{l}{\footnotesize %s} \\" % _OZET,
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _SOZ = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen".split()
    io.open(f"{OUT}/T13_body.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/protocol_2x2_table.py)\n"
        + ("Under the chat-template prompt the same contrast splits (Figure~\\ref{fig:effects}(b)). On the filtered "
           "set the second-person rate rises in %d models, falls in %d and shows no clear change in %d, while "
           % (ns, nn, nu))
        + (("raw continuation on the same set stays falling in all %d. " % nc) if nc == len(R)
           else ("raw continuation on the same set stays at %d of %d falling. " % (nc, len(R))))
        + ("T\\\"ulu-3-8B and Llama-3.1-8B descend from the same base checkpoint, and their two pipelines say "
           "``you'' $%.1f$ and $%.1f$ times per 1,000 tokens through their own templates%s. Since the base weights are "
           "shared, the pipeline sets the sign of the change here." % (_t8["M1_hizali"], _l8["M1_hizali"],
           ", one falling and one rising against its base" if (_t8["dM1"] < 0) != (_l8["dM1"] < 0) else "")) + "\n")
    print("  ✓ T13_body.tex")
    io.open(f"{OUT}/T13_note.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/protocol_2x2_table.py)\n"
        "\\paragraph*{Table~\\ref{tab:twobytwo-full}: symbols, the in-context assistant prompt and the "
        "unfiltered reading.}\n"
        "\\vekalet{" + _s_bulgu.replace("$^{\\S}$", "") + " "
        "$\\dagger$ marks the one positive whose CI lower bound is $+0.05$ "
        "at the lowest retention. $^{\\ast}$\\emph{task} is a separate measurement on a "
        "separate prompt set and is \\emph{not} read on the filtered cells of the first two "
        "columns: the base outputs are generated with the assistant task as three in-context examples "
        "\\citep{lin2024urial} and the aligned outputs are read through the model's own template, so "
        "each checkpoint is prompted in the format it was trained for while the task is held "
        "fixed. " + _s_serh + _u_govde + "}\n")
    print("  ✓ T13_note.tex")
    io.open(f"{OUT}/T13_2x2.tex", "w", encoding="utf-8").write(
        "% * v72: moved out of the main text - p11.tex does not \\input this file "
        "(see fig/F_split.pdf; full version T13_full.tex)\n" + "\n".join(L) + "\n")
    import style as _ST
    _kay = [pux, dict(yol="results/" + os.path.basename(KAY),
                          sha256_16=_ST.sha16(KAY))]
    json.dump(dict(table="T13_2x2", sources=_kay,
                   payda=dict(n_aile=len(R), poz=ns, neg=nn, null=nu, ciplak_neg=nc,
                              U_asagi=Usay["asagi"], U_ters=Usay["ters"],
                              U_bicim=Usay["bicim"]),
                   note="renders FILTERED_2x2_2026-09-07; bar fixed before the run"),
              io.open(f"{OUT}/T13_2x2.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("  ✓ T13_2x2.meta.json")
    eksik = sorted({r["aile"] for r in R} - set(Uail))
    if eksik:
        print(f"  ★ URIAL BACAGI EKSIK: {eksik} ⇒ HATA (§8)"); return 3
    print(f"{len(R)} {ns} {nn}"
          f"{nu} {nc} {Usay['asagi']}"
          f"{Usay['ters']} {Usay['bicim']}"
          f"{Ukap['bacak_eksik']}"
          f"{len(R)} {len(R)}"
          f"")
    print(f"✓ {OUT}/T13_2x2.tex")


if __name__ == "__main__":
    main()
