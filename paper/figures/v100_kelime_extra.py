#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, statistics as st, sys
ROOT = __DNH_ROOT__ + ""
OUT = f"{ROOT}/paper/figures"
CARD = "results/v100_kelime_payda_2026-09-22.json"
T1META = f"{OUT}/T1_families.meta.json"
AD = {"Tulu3-8B": "T\\\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def ad(a):
    return AD.get(a, a)


def ve(xs):
    xs = [ad(a) for a in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def yaz(dosya, s, not_):
    io.open(f"{OUT}/{dosya}.tex", "w", encoding="utf-8").write(s.rstrip() + "\n")
    json.dump(dict(table=dosya, ciktilar=[f"{dosya}.tex"], note=not_,
                   sources=[dict(yol=CARD, sha256_16=hashlib.sha256(open(f"{ROOT}/{CARD}", "rb").read()).hexdigest()[:16])]),
              io.open(f"{OUT}/{dosya}.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ {dosya}.tex ({len(s)} kar)")


def main():
    K = json.load(io.open(f"{ROOT}/{CARD}", encoding="utf-8"))
    R = K["karar"]; h = R["verdict"]; sw, sj = R["sayim"]["kelime"], R["sayim"]["jeton"]
    assert h in ("GÖVDEYE", "LIMITATIONS", "OKUNAMADI")
    assert sj["asagi"] == 16 and sj["asagi_plasebo_ustu"] == 16 or h == "OKUNAMADI"
    A = R["aile"]
    sira = json.load(io.open(T1META, encoding="utf-8"))["sira"]
    assert sorted(sira) == sorted(A)
    med_w = st.median(v["kelime"][0] for v in A.values()); med_t = st.median(v["jeton"][0] for v in A.values())
    oran = R["seviye_orani_kelime_jeton"]
    or_t = [v["taban"] for v in oran.values()]; or_h = [v["hizali"] for v in oran.values()]
    print(f"{h} {sw} {sj} {[x['okuma'] for x in K['alet_kaydi']]}"
          f"")
    tanim = ("per 1,000 words, where a word is a token carrying at least one letter, so that punctuation, list markers, "
             "markdown and numerals leave the denominator")
    govde = lim = "% v100 A: bu dal yazilmadi (karar " + h + ")"
    if h == "GÖVDEYE":
        assert sw["asagi"] == 16 and sw["asagi_plasebo_ustu"] == 16
        govde = (f"Per 1,000 words instead of tokens, the fall holds in all 16 models and clears the paired placebo in all 16 "
                 f"(Appendix~\\ref{{app:D2}}).")
    elif h == "LIMITATIONS":
        lim = (f"\\vekalet{{\\textbf{{Denominator.}} Counted {tanim}, the fall in ``you'' on raw continuation separates and "
               f"clears the paired placebo in {sw['asagi_plasebo_ustu']} of 16 models rather than 16, at a median of "
               f"${med_w:.2f}$ per 1{{,}}000 words against ${med_t:.2f}$ per 1{{,}}000 tokens (Appendix~\\ref{{app:D2}}).}}")
    else:
        lim = ("\\vekalet{\\textbf{Denominator.} A re-read per 1,000 words could not be read: its equivalence gate against "
               "Table~\\ref{tab:families} did not pass (Appendix~\\ref{app:D2}).}")
    yaz("KELIME_CUMLE", govde, "v100 A: §3 sentence (GÖVDEYE branch only)")
    yaz("KELIME_LIM", lim, "v100 A: Limitations sentence (LIMITATIONS / OKUNAMADI branch only)")
    O = K["okuma"]
    n_kan = 16 - len(K["B_Q2"]["duzyazi_cumle_null"]["adlar"]); n_es = O["ham_duz_p"]["duz_cumle"]["jeton"]["asagi"]
    fark = {1: "one model more", 2: "two models more", -1: "one model fewer", -2: "two models fewer"}.get(n_es - n_kan, f"{n_es - n_kan:+d} models")
    DUZC = ("``you'' on prose selected by sentence, with this paired reading's own intervals" +
            ("" if n_es == n_kan else f", which separate {fark} than the {n_kan} of \\S\\ref{{sec:deperson}}"))
    ETK = [("template", "sbl_dM1", "``you'' under the chat template (filtered set, one seed)", False),
           ("b1", "M1", "``you'' on ordinary instructions (in-context assistant prompt for the base, own template for the aligned model)", True),
           ("ham_1st", "d1st", "``I'' counted directly on raw continuation (original seed)", False),
           ("template_1st", "sbl_1st", "derived first person under the chat template", False),
           ("b1_1st", "IS1", "``I'' counted directly on ordinary instructions", True),
           ("ham_duz", "duz_1k", "``you'' on prose lines alone", False),
           ("ham_duz_p", "duz_cumle", DUZC, True)]
    parca = []
    for okuma, alt, etiket, pl in ETK:
        if okuma not in O:
            parca.append(f"{etiket}: not read (equivalence gate)"); continue
        j, w = O[okuma][alt]["jeton"], O[okuma][alt]["kelime"]
        def f(c):
            t = f"{c['asagi']} down, {c['yukari']} up, {c['null']} covering zero"
            return t + (f", {c['asagi_plasebo_ustu']} falling above the placebo" if pl and "asagi_plasebo_ustu" in c else "")
        parca.append(f"{etiket}: {f(j)} per 1{{,}}000 tokens and {f(w)} per 1{{,}}000 words")
    S = ["\\begin{center}\\scriptsize\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{lrrrrrr}\\toprule",
         " & \\multicolumn{2}{c}{per 1,000 tokens} & \\multicolumn{3}{c}{per 1,000 words} & words per token \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
         "model & $\\Delta$ & $\\pm$ & $\\Delta$ & $\\pm$ & placebo p95 & base / aligned \\\\ \\midrule"]
    for a in sira:
        (dj, cj, _, _), (dw, cw, pw, p95w) = A[a]["jeton"], A[a]["kelime"]
        b = lambda d, c: f"$\\mathbf{{{d:+.2f}}}$" if (c[1] < 0 or c[0] > 0) else f"${d:+.2f}$"
        S.append(f"{ad(a)} & {b(dj, cj)} & {(cj[1]-cj[0])/2:.2f} & {b(dw, cw)} & {(cw[1]-cw[0])/2:.2f} & {p95w:.2f} & "
                 f"{1/oran[a]['taban']:.2f} / {1/oran[a]['hizali']:.2f} \\\\")
    S += ["\\bottomrule\\end{tabular}\\end{center}"]
    d2 = ("\\paragraph*{The main contrast per 1,000 words.}\n"
          f"\\vekalet{{Table~\\ref{{tab:families}} reads the second-person rate per 1,000 tokens, a denominator that counts "
          f"punctuation, list markers and markdown. Here the same outputs, the same pooled seeds, the same prompt-clustered "
          f"intervals and the same paired placebo are read {tanim}; only the denominator changes, and the numerator is "
          f"unchanged. Read with tokens, the code reproduces Table~\\ref{{tab:families}} exactly ({R['tablo1_esdeger']['n_fark']} "
          f"differences over sixteen models). With words the fall holds in {sw['asagi']} of 16 models and clears the placebo in "
          f"{sw['asagi_plasebo_ustu']}; a rule fixed before the count sent the result to the main text at 12 or more and to "
          f"the Limitations below 12. The base outputs carry ${min(1/x for x in or_t):.2f}$ to ${max(1/x for x in or_t):.2f}$ "
          f"words per token and the aligned outputs ${min(1/x for x in or_h):.2f}$ to ${max(1/x for x in or_h):.2f}$. "
          f"Bold marks an interval excluding zero.}}\n" + "\n".join(S) + "\n"
          "\\vekalet{The other readings under both denominators, each on its own set as in the main text: "
          + "; ".join(parca) + ". Measures per sentence divide by the number of sentences and do not change with the "
          "denominator; the code checks that they are identical under both. The rule and the equivalence gate were committed "
          "before any count per word was read; the re-read is descriptive.}")
    yaz("KELIME_D2", d2, "v100 A: Appendix D.2 paragraph, table + other readings under both denominators")
    B = K["B_Q2"]

    def kol(k, adlar_basligi):
        x = B[k]
        if x["okuma"] == "ALET-KAYDI":
            return f"{adlar_basligi}: not read (equivalence gate)"
        ic = [a for a, v in x["okuma"].items() if v["jeton"]["ici"]]
        dis = [a for a, v in x["okuma"].items() if not v["jeton"]["ici"]]
        dis_a = [a for a in dis if x["okuma"][a]["jeton"]["sinif"] == "asagi"]
        dis_n = [a for a in dis if a not in dis_a]
        SOZ = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        t = f"{adlar_basligi.replace('the models', 'the ' + SOZ.get(len(x['adlar']), str(len(x['adlar']))) + ' models')}, "
        t += (f"{ve(ic)} sit{'s' if len(ic) == 1 else ''} inside the band" if ic else "none sits inside the band")
        for grup, ek in ((dis_n, "with an interval that covers zero"), (dis_a, "with an interval excluding zero in this paired reading")):
            if grup:
                t += f"; {ve(grup)} lie{'s' if len(grup) == 1 else ''} beyond it, {ek}"
        return t
    q2 = ("\\paragraph*{Where the controlled readings fail: inside the placebo band, or a wider interval?}\n"
          "\\vekalet{For each model whose fall does not separate on prose lines or per sentence, we compare the fall with the "
          "95th percentile of its paired within-prompt placebo, built on the same pairs and the same readings. "
          + kol("duzyazi_satir_null", "Of the models that do not separate on prose lines")
          + ". " + kol("cumle_basina_null", "Of the models that do not separate per sentence")
          + ". " + kol("duzyazi_cumle_null", "Of the models that do not separate on prose selected by sentence")
          + ". Inside the band means the reading cannot tell the fall from a split of the model's own outputs; beyond it with "
          "an interval covering zero means the interval, not the effect, is what the reading loses.}")
    yaz("Q2_PLASEBO_D2", q2, "v100 B (Assistant-v99 Q2): failing models vs paired placebo p95")
    return 0


if __name__ == "__main__":
    sys.exit(main())
