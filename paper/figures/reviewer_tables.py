#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time, hashlib
ROOT = __DNH_ROOT__ + ""
OUT = os.path.dirname(os.path.abspath(__file__))
G = f"{ROOT}/results"
K = dict(q2="reviewer_q2_prose_2026-09-15.json", is1="reviewer_task1_prose_first_person_2026-09-15.json",
         q4="reviewer_q4_protocol_sign_2026-09-15.json", q5="reviewer_q5_band_three_substrates_2026-09-15.json",
         cumle="reviewer_v87_sentence_prose_2026-09-16.json",
         cumle_on="reviewer_v87_sentence_prose_ten_families_2026-09-16.json")
EN = {"Tulu3-8B": "T\\\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def sha16(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()[:16]


def ad(a): return EN.get(a, a)


def v(d, ci, kalin=True):
    s = f"{d:+.2f}".replace("-", "$-$").replace("+", "$+$")
    return f"\\textbf{{{s}}}" if kalin and (ci[0] > 0 or ci[1] < 0) else s


def main():
    C = {k: json.load(io.open(f"{G}/{f}", encoding="utf-8")) for k, f in K.items()}
    for k in ("q2", "is1", "q5", "cumle", "cumle_on"):
        assert C[k]["alet"]["hal"] == "ESDEGER", (k, C[k]["alet"])
    Q2, I1, Q4 = C["q2"]["aile"], C["is1"]["aile"], C["q4"]["tablo"]
    CU = {**C["cumle"]["aile"], **C["cumle_on"]["aile"]}
    assert len(CU) == 16 and set(CU) == set(Q2), sorted(CU)
    _hicbiri = sorted(a for a in Q2 if Q2[a]["duzyazi"]["sinif"] != "asagi" and CU[a]["duzyazi_cumle"]["sinif"] != "asagi")
    aileler = sorted(Q2)
    assert set(aileler) == set(I1) == set(Q4) and len(aileler) == 16
    L = [r"\paragraph*{Prose lines alone, for both persons (\S\ref{sec:deperson}).}",
         r"\vekalet{A rule committed before each count drops every line that opens as a heading, a bulleted or numbered item, "
         r"or consists of bold text alone, and reads what remains with the canonical counters; the first person is counted directly, "
         r"with a capitalised \emph{US} not counted. Bold marks a prompt-clustered $95\%$ interval that excludes zero. "
         r"The drop is line-level, so a sentence that runs from a prose line into a list line is not seen by this rule. "
         r"The sentence column classifies whole sentences instead, each generation parsed once: a sentence touching any list "
         r"line is dropped whole. By sentence the fall holds in $" + str(sum(1 for a in CU if CU[a]["duzyazi_cumle"]["sinif"] == "asagi"))
         + r"$ of $16$ models; it holds by neither unit in " + ", ".join(ad(x) for x in _hicbiri[:-1])
         + r" and " + ad(_hicbiri[-1]) + r".}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrrrr}\toprule",
         r"model & lines dropped, base & aligned & $\Delta M_1$, all lines & $\Delta M_1$, prose lines & prose sentences & $\Delta$1st, prose \\ \midrule"]
    for a in aileler:
        q, i = Q2[a], I1[a]
        L.append(f"{ad(a)} & {q['atilan_taban']['satir_payi']:.3f} & {q['atilan_hizali']['satir_payi']:.3f} & "
                 f"{q['tam']['d']:+.2f} & {v(q['duzyazi']['d'], q['duzyazi']['ci'])} & "
                 f"{v(CU[a]['duzyazi_cumle']['d'], CU[a]['duzyazi_cumle']['ci']) if a in CU else '{---}'} & "
                 f"{v(i['duzyazi_d1st']['d'], i['duzyazi_d1st']['ci'])} \\\\".replace("& +", "& $+$").replace("& -", "& $-$"))
    n2 = sum(1 for a in aileler if Q2[a]["duzyazi"]["sinif"] == "asagi")
    n1 = sum(1 for a in aileler if I1[a]["duzyazi_d1st"]["sinif"] == "asagi")
    nc = sum(1 for a in CU if CU[a]["duzyazi_cumle"]["sinif"] == "asagi")
    L += [r"\midrule", f"negative, interval excluding zero & & & $16$ & ${n2}$ & ${nc}$ & ${n1}$ \\\\", r"\bottomrule\end{tabular}\end{center}"]
    io.open(f"{OUT}/HAKEM_DUZYAZI.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/reviewer_tables.py)\n" + "\n".join(L) + "\n")
    L = [r"\paragraph*{The second person by model and protocol (\S\ref{sec:deperson}).}",
         r"\vekalet{$\Delta M_1$ per model on the raw-continuation prompt, through each aligned model's own template on the four-quadrant-filtered cells, "
         r"and in the URIAL frame read two ways: the base outputs in the frame against the aligned model's own template, and both checkpoints on the identical string. "
         r"Bold marks a prompt-clustered $95\%$ interval that excludes zero. No new count: the values are those of the three published readings.}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrr}\toprule",
         r"model & raw-continuation & template & URIAL, own template & URIAL, identical string \\ \midrule"]
    for a in aileler:
        t = Q4[a]
        L.append(f"{ad(a)} & " + " & ".join(v(t[z]['d'], t[z]['ci']) for z in ("ciplak", "template", "urial_i", "urial_ii")) + r" \\")
    s = C["q4"]["sayim"]
    L += [r"\midrule"]
    for c, e in (("asagi", "down"), ("yukari", "up"), ("null", "null")):
        L.append(f"{e} & " + " & ".join(f"${s[z][c]}$" for z in ("ciplak", "template", "urial_i", "urial_ii")) + r" \\")
    L += [r"\bottomrule\end{tabular}\end{center}"]
    io.open(f"{OUT}/HAKEM_ISARET.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/reviewer_tables.py)\n" + "\n".join(L) + "\n")
    B = C["q5"]["bant"]
    sat = (("raw-continuation prompt", "ciplak"), ("own template, filtered", "template_filtered"), ("URIAL frame, base; own template, aligned", "urial_i"),
           ("own template, filtered, echo and turn-marked lines dropped", "template_filtered_yankisiz_tursuz"))
    L = [r"\paragraph*{The band on three prompt formats (\S\ref{sec:deperson}).}",
         r"\vekalet{Largest over smallest model-level $M_1$, per thousand tokens, for the sixteen base and the sixteen aligned checkpoints. "
         r"The URIAL row reads the base outputs inside the frame, with the task in context, and the aligned outputs through its own template.}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrrr}\toprule",
         r"prompt format & base min & base max & base ratio & aligned min & aligned max & aligned ratio \\ \midrule"]
    for e, z in sat:
        b = B[z]
        L.append(f"{e} & ${b['taban']['min']:.2f}$ & ${b['taban']['max']:.2f}$ & ${b['taban']['oran']:.2f}\\times$ & "
                 f"${b['hizali']['min']:.2f}$ & ${b['hizali']['max']:.2f}$ & ${b['hizali']['oran']:.2f}\\times$ \\\\")
    L += [r"\bottomrule\end{tabular}\end{center}"]
    io.open(f"{OUT}/HAKEM_BANT.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/reviewer_tables.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="HAKEM_TABLOLARI", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=f"results/{f}", sha256_16=sha16(f"{G}/{f}")) for f in K.values()],
                   payda=dict(n_aile=16, q2_asagi=n2, is1_asagi=n1), note="renders four cards; computes no new statistic"),
              io.open(f"{OUT}/HAKEM_TABLOLARI.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{n2} {n1}")
    print("✓ HAKEM_DUZYAZI.tex · HAKEM_ISARET.tex · HAKEM_BANT.tex")


if __name__ == "__main__":
    sys.exit(main())
