#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, os, sys, time
ROOT = __DNH_ROOT__ + ""
SAY = f"{ROOT}/results/continuation_mode_exit_count_2026-09-15.json"
DUY = f"{ROOT}/results/continuation_mode_exit_sensitivity_2026-09-15.json"
TAV = f"{ROOT}/results/continuation_mode_exit_ceiling_2026-09-15.json"
OUT = os.path.dirname(os.path.abspath(__file__))
AD = {"Tulu3-8B": "T\\\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def sha16(y):
    return hashlib.sha256(open(y, "rb").read()).hexdigest()[:16]


def nf(x):
    return "{:,}".format(x).replace(",", "{,}")


def main():
    S = json.load(io.open(SAY, encoding="utf-8")); D = json.load(io.open(DUY, encoding="utf-8"))
    TV = json.load(io.open(TAV, encoding="utf-8"))
    assert TV["red_esdeger_M1"] == 0 and TV["n_aile"] == 16
    A, B = S["aile"], D["aile"]
    assert set(A) == set(B) and len(A) == S["n_aile"] == D["n_aile"] == 16
    if D.get("red_esdeger_M1", 1) != 0:
        print(""); return 3
    cik = lambda r: r["RET"] + r["ASISTAN"] + r["META"]
    satir = []
    T = {"h": [0, 0, 0, 0, 0], "t": [0, 0, 0, 0, 0]}
    for a in A:
        h, t = A[a]["hizali"], A[a]["taban"]
        assert cik(h) == B[a]["cikis_hizali"] and cik(t) == B[a]["cikis_taban"], a
        for k, r in (("h", h), ("t", t)):
            T[k][0] += cik(r); T[k][1] += r["n"]; T[k][2] += r["RET"]; T[k][3] += r["ASISTAN"]; T[k][4] += r["META"]
        satir.append(dict(aile=a, ph=100 * cik(h) / h["n"], pt=100 * cik(t) / t["n"], tum=B[a]["dM1_tum"],
                          dev=B[a]["dM1_devam"], ci=B[a]["ci_devam"], ayrik=B[a]["ayrik_asagi"],
                          pay=100 * B[a]["ikinci_sahis_payi_cikis_hizali"]))
    satir.sort(key=lambda r: r["tum"])
    n_ayrik = sum(r["ayrik"] for r in satir)
    assert n_ayrik == D["n_ayrik_asagi_devam"]
    en = max(satir, key=lambda r: abs(r["dev"] - r["tum"]))
    ph = [r["ph"] for r in satir]; pt = [r["pt"] for r in satir]; py = [r["pay"] for r in satir]
    nm = lambda a: AD.get(a, a)
    TT = TV["toplam"]
    DAR = sorted(a for a, o in TV["aile"].items() if abs(o["dM1_tavan"]) < abs(o["dM1_tum"]))
    assert len(DAR) == 2, DAR
    para = (
        r"\paragraph*{Whether the aligned outputs leave the continuation on the raw-continuation prompt.}" "\n"
        r"\vekalet{\textbf{A raw continuation prompt can still be answered as an assistant, and we counted how often it is.} "
        r"A rule written before the count reads only the opening of each generation --- its first $160$ characters, "
        r"after leading whitespace, quotation marks and markdown are stripped --- and assigns the first class that matches: "
        r"a refusal (\emph{I'm sorry}, \emph{I cannot}, \emph{As an AI}), an assistant opening (\emph{Sure}, "
        r"\emph{Certainly}, \emph{Here is}, \emph{Let me}) or a meta-comment on the text (\emph{This argument}, "
        r"\emph{It seems}, \emph{In summary}, \emph{You've provided}); everything else is read as a continuation, and "
        r"the same rule is applied to the base outputs as its structural reference. "
        r"Of $%s$ aligned generations, $%s$ ($%.1f\%%$) are flagged --- $%d$ refusals, $%d$ assistant openings and "
        r"$%s$ meta-comments --- against $%s$ ($%.1f\%%$) in the base outputs; refusals and assistant openings alone are "
        r"$%.1f\%%$ ($%d$ of $%s$), and the meta-comments are the rest. Per model the aligned share runs from "
        r"$%.1f\%%$ to $%.1f\%%$ and the base share from $%.1f\%%$ to $%.1f\%%$. "
        r"\textbf{With the flagged generations removed from both checkpoints, $\Delta M_1$ stays negative and "
        r"interval-disjoint in $%d$ of $%d$ models} (Table~\ref{tab:q2kip}); the largest change is $%.2f$ per "
        r"thousand (%s, $%.2f\to%.2f$), and the full-panel $\Delta M_1$ recomputed by the same counter reproduces "
        r"the registered single-seed reading in all sixteen. Between $%.1f\%%$ and $%.1f\%%$ of an aligned checkpoint's second persons "
        r"sit in its flagged generations. "
        r"\textbf{The reading has three limits.} The rule reads only the opening, so a generation that changes mode "
        r"mid-text is not seen; a deliberately over-cutting reading, its rule committed before its count, that flags a refusal "
        r"or assistant marker at the start of any line or sentence and any turn marker or markdown structure anywhere in "
        r"the generation removes $%.1f\%%$ of aligned and $%.1f\%%$ of base generations and leaves $%d$ of $%d$ negative "
        r"and interval-disjoint, the fall moving by at most $%.2f$ per thousand (%s, $%.2f\to%.2f$). "
        r"The $%.1f\%%$ it removes from the base outputs, which have no assistant mode to leave, is that rule's false-positive "
        r"floor; the fall widens in all but two models and narrows slightly in %s. "
        r"Its class precision was not measured. And on a seeded sample of flagged generations "
        r"from three models, the meta-comment class also collects legitimate debate continuations (\emph{The "
        r"argument is based on\dots}) and the refusal class a continuation such as \emph{I can't prove it}, so among "
        r"openings the removed set errs on the large side, and $%d$ of $%d$ holds with that larger set removed. "
        r"The flagged share is therefore neither a lower nor an upper bound on how often the aligned outputs leave the "
        r"continuation. The reading is descriptive: the rule was written before the count, but no acceptance bound "
        r"was (Appendix~\ref{app:sapma}).}" "\n"
        % (nf(T["h"][1]), nf(T["h"][0]), 100 * T["h"][0] / T["h"][1], T["h"][2], T["h"][3], nf(T["h"][4]),
           nf(T["t"][0]), 100 * T["t"][0] / T["t"][1],
           100 * (T["h"][2] + T["h"][3]) / T["h"][1], T["h"][2] + T["h"][3], nf(T["h"][1]),
           min(ph), max(ph), min(pt), max(pt),
           n_ayrik, len(satir), abs(en["dev"] - en["tum"]), nm(en["aile"]), en["tum"], en["dev"],
           min(py), max(py),
           100 * TT["hizali"]["atilan"] / TT["hizali"]["n"], 100 * TT["taban"]["atilan"] / TT["taban"]["n"],
           TV["n_ayrik_asagi_tavan"], TV["n_aile"], TV["en_buyuk_kayma"]["kayma"], nm(TV["en_buyuk_kayma"]["aile"]),
           TV["aile"][TV["en_buyuk_kayma"]["aile"]]["dM1_tum"], TV["aile"][TV["en_buyuk_kayma"]["aile"]]["dM1_tavan"],
           100 * TT["taban"]["atilan"] / TT["taban"]["n"], " and ".join(nm(a) for a in DAR),
           n_ayrik, len(satir)))
    tab = [r"\par\medskip\noindent\begin{minipage}{\linewidth}\sabittablo\small",
           r"\caption{\textbf{The raw-continuation-prompt contrast with the generations that leave the continuation removed.} "
           r"Flagged share is the percentage of a checkpoint's $6{,}528$ generations whose opening a rule written before the "
           r"count reads as a refusal, an assistant opening or a meta-comment. $\Delta M_1$ is recomputed on both checkpoints "
           r"without the flagged generations, with its prompt-clustered $95\%$ CI; the last column is the share of the "
           r"aligned outputs' second persons that sit in its flagged generations. Rows follow Table~\ref{tab:families}.}",
           r"\label{tab:q2kip}",
           r"\centering\begin{tabular}{lrrrrr}", r"\toprule",
           r" & \multicolumn{2}{c}{flagged share} & \multicolumn{2}{c}{$\Delta M_1$} & aligned 2nd persons \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"model & aligned & base & all & without flagged [$95\%$ CI] & in flagged \\",
           r"\midrule"]
    for r in satir:
        v = (r"\textbf{%.2f}" if r["ayrik"] else "%.2f") % r["dev"]
        tab.append(r"%s & %.2f\%% & %.2f\%% & %.2f & %s [%.2f,\,%.2f] & %.1f\%% \\"
                   % (nm(r["aile"]), r["ph"], r["pt"], r["tum"], v, r["ci"][0], r["ci"][1], r["pay"]))
    tab += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}\par\medskip"]
    io.open(f"{OUT}/Q2_KIP.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/continuation_mode_exit_block.py)\n" + para + "\n".join(tab) + "\n")
    json.dump(dict(table="Q2_KIP", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sinif="descriptive · unsealed · rule written before the count, no acceptance bound",
                   alet="scripts/continuation_mode_exit.py",
                   sources=[dict(yol=os.path.relpath(SAY, ROOT), sha256_16=sha16(SAY)),
                            dict(yol=os.path.relpath(DUY, ROOT), sha256_16=sha16(DUY)),
                            dict(yol=os.path.relpath(TAV, ROOT), sha256_16=sha16(TAV))],
                   tavan=dict(toplam=TT, n_ayrik=TV["n_ayrik_asagi_tavan"], en_buyuk_kayma=TV["en_buyuk_kayma"]),
                   toplam=dict(hizali=T["h"], taban=T["t"]), n_ayrik=n_ayrik, en_buyuk_kayma=en),
              io.open(f"{OUT}/Q2_KIP.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(satir)} {T['h'][0]} {T['h'][1]} {T['t'][0]} {T['t'][1]}"
          f"{n_ayrik} {len(satir)} {abs(en['dev']-en['tum']):.4f} {en['aile']}"
          f"")
    print("✓ Q2_KIP.tex + Q2_KIP.meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
