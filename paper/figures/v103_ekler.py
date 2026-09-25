#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os

KOK = __DNH_ROOT__ + ""
CARD = f"{KOK}/results/v103_diskten_2026-09-23.json"
FIG = f"{KOK}/paper/figures"
BAS = "% * URETILDI - elle duzenlenmez (figures/v103_ekler.py)\n"


def yaz(ad, govde):
    y = f"{FIG}/{ad}"
    io.open(y, "w", encoding="utf-8").write(BAS + govde.strip() + "\n")
    print(f"✓ {ad} ({len(govde.split())} kelime)")


def n(x, k=2):
    return f"${x:.{k}f}$"


def yuzde(x, k=2):
    return f"${100*x:.{k}f}\\%$"


def tam(x):
    return "$" + f"{int(x):,}".replace(",", "{,}") + "$"


def parca_a(K):
    a = K["a"]; t, l = a["Tulu3-8B"], a["Llama-3.1-8B"]
    yaz("V103_A_TALIMAT.tex", rf"""
\paragraph*{{The two pipelines on one base, read here (\S\ref{{sec:deperson}}).}}
\vekalet{{The pair that says ``you'' $2.0$ and $20.0$ times per 1,000 tokens through their own chat
templates reads {n(t['M1_hizali'])} (T\"ulu-3-8B) and {n(l['M1_hizali'])} (Llama-3.1-8B-Instruct) on these
instructions, each in its own format: a {n(a['kat'])}-fold gap against the $20.5$-fold range of the
template condition. Their shared base reads {n(t['M1_taban'])} and {n(l['M1_taban'])} on the cells each
pair keeps ({tam(t['n_tutulan'])} and {tam(l['n_tutulan'])} of $1{{,}}200$), which is the same checkpoint read twice.
\textbf{{The divergence between these two pipelines is therefore a property of the chat-template
condition and does not carry to the instruction set}}, where the whole aligned panel spans
{n(a['bant_hizali']['oran'])}-fold. The level under a template is what a user meets; the level here is what
the same weights produce when the task is fixed, and the two are not the same claim.}}
""")


def parca_c(K):
    c = K["c"]; s, r = c["KISI_ASAGI"], c["RASTGELE"]; p = c["munazara_paneli"]
    assert p["n_kod_citi"] == 0 and p["n_kod_ipucu"] == 0, ""
    yaz("V103_C_KOD.tex", rf"""
\paragraph*{{How far apart the code blocks are, and where the runs are read.}}
\vekalet{{Selecting pairs on pronoun use also selects on code, and the size of that difference is
measurable. Counting the tokens inside fenced code blocks, the chosen side of the
{tam(s['n_cift'])} selected pairs carries {yuzde(s['chosen']['kod_jeton_pay'])} of its tokens in code against
{yuzde(s['rejected']['kod_jeton_pay'])} on the rejected side, a gap of
${100*s['fark']['kod_jeton_pay']:+.2f}$ points, or ${s['fark']['kod_jeton_ort']:+.1f}$ code tokens per
response; in an unselected sample of the same size the same gap is
${100*r['fark']['kod_jeton_pay']:+.2f}$ points. \textbf{{The two readings of ``fewer code blocks'' point in
opposite directions}}: the share of chosen responses that contain any code is
${-100*s['fark']['kod_iceren_pay']:.2f}$ points \emph{{lower}}, while the chosen side's code
\emph{{volume}} is higher. For a density the volume is the reading that bears: code carries almost no
grammatical person, so a chosen side richer in code tokens lowers the rate in the same direction as the
selection, and the control run matched on refusals and length does not remove it. \textbf{{The runs are read
on the debate prompts, which contain no code prompts}}: across their {tam(p['n_istem_metni'])} distinct prompt
strings ({p['n_tez_durus']} thesis-by-stance prompts and {tam(p['n_onek'])} prefix strings) there is no fenced
block and no match for any of the {len(p['ipucu_listesi'])} programming cues we fixed before counting. The
confound is therefore in the training pairs and not in the evaluation, and only a run matched on code
blocks separates them, which we have not run.}}
""")


def parca_e(K):
    e = K["e"]; sy = e["_sayim"]
    mx = e["Mixtral-8x7B"]
    assert sy["n_uretilip_disarida"] == 0, ""
    n16 = len([k for k, v in e.items() if not k.startswith("_") and v["sinif"] == "PANEL-16"])
    nek = len([k for k, v in e.items() if not k.startswith("_") and v["sinif"] == "EK-H-4"])
    niz = len([k for k, v in e.items() if not k.startswith("_") and v["sinif"] == "KÂGITTA IZ VAR"])
    nbiz = len([k for k, v in e.items() if not k.startswith("_") and v["sinif"] == "BIZIM-EGITIM"])
    yaz("V103_E_EKH.tex", rf"""
\vekalet{{\textbf{{No other model was generated and left out}}, and we checked rather than asserted it.
The panel's generation store has {sy['n_dizin']} model entries: {n16} are the panel, {nek} are the
four above, {niz} are the checkpoint sequences of \S\ref{{sec:origin}} and the released pairs they are read
against, and {nbiz} holds the checkpoints this paper trained. One entry is left,
\texttt{{Mixtral-8x7B}}: it holds {sum(mx['yarim_kol_dosyasi'].values())} partial prefix-condition files for
a base checkpoint, no assembled output, and no aligned checkpoint. That run was abandoned before it
produced a generation, so there is nothing from it to report and nothing was dropped after being read.
Every name in this store is either in this paper or is that one entry.}}
""")


def parca_b(K):
    b = K["b"]; sy = b["_sayim"]
    ad = [a for a in sy["hala_yukselen"]]
    dus = [a for a in sy["dusen"]]
    olc = [a for a in b if not a.startswith("_") and b[a].get("hal") == "ÖLCÜLDÜ"]
    import statistics as st
    m_tam = st.median([b[a]["tam"]["dM1"] for a in olc])
    m_at = st.median([b[a]["ilk_cumle_atildi"]["dM1"] for a in olc])
    pay = st.median([b[a]["pay_jeton_atilan_hizali"] for a in olc])
    ad = [x.replace("Tulu3", 'T\\"ulu-3').replace("OLMo2", "OLMo-2") for x in dus]
    liste = (" and ".join([", ".join(ad[:-1]), ad[-1]]) if len(ad) > 1 else (ad[0] if ad else "none"))
    frj = " (Qwen2.5-1.5B is the fragile positive this appendix already flags)" if "Qwen2.5-1.5B" in dus else ""
    yaz("V103_B_ILKCUMLE.tex", rf"""
\paragraph*{{Is the rise the opening sentence?}}
\vekalet{{Under a chat template an aligned model often opens with a fixed conversational frame, so we
dropped the first sentence of every response, in \emph{{both}} checkpoints and on the same cells, and read
the contrast again. Across the ${sy['n_olculen']}$ models whose rate rises, that removes a median
{yuzde(pay, 1)} of the aligned tokens, and the median shift moves from ${m_tam:+.2f}$ to ${m_at:+.2f}$ per
1,000 tokens. The rise survives with an interval excluding zero in
${sy['n_hala_yukselen_ayrik']}$ of ${sy['n_olculen']}${'' if not dus else f", and stops in {liste}{frj}"}.
\textbf{{The opening frame is therefore part of the rise and not the whole of it}}, which is what a
density spread over a whole response should show; it does not explain why the sign differs between
pipelines, and we still have no mechanism for that.}}
""")


def parca_d(K):
    d = K["d"]; m = d["_medyan"]
    k = {a: m[a] for a in ("kayit_sahissiz", "kayit", "acik")}
    yaz("V103_D_UCSURUM.tex", rf"""
\paragraph*{{What else the one-sentence instruction moves.}}
\vekalet{{The sentence is scored on the counters, so we also read three surface properties of the same
outputs, each against the same model's aligned output without the sentence and summarised as a median
over the ${k['kayit_sahissiz']['n']}$ models. \textbf{{Length and refusals barely move; the structure of
the answer does}}. The version with no pronoun moves length by ${k['kayit_sahissiz']['jeton_ort']:+.1f}$
tokens and the refusal rate by ${k['kayit_sahissiz']['ret_orani']:+.4f}$, the version with one ``you'' by
${k['kayit']['jeton_ort']:+.1f}$ and ${k['kayit']['ret_orani']:+.4f}$, and the version that names the
forms by ${k['acik']['jeton_ort']:+.1f}$ and ${k['acik']['ret_orani']:+.4f}$. The share of list, numbered
and heading lines falls by ${-k['kayit_sahissiz']['liste_payi']:.3f}$, ${-k['kayit']['liste_payi']:.3f}$ and
${-k['acik']['liste_payi']:.3f}$ respectively, so a quarter to a third of the structured lines become
prose. \textbf{{These are surface properties, not quality}}: we have no measure of whether the answers
stay as helpful, so what the instruction costs is untested. It also means the restoration is not a pure
register move: asked to be personal, the aligned model drops much of its list format as well, and a model
card offered this sentence should be told both.}}
""")


def main():
    K = json.load(io.open(CARD, encoding="utf-8"))
    for p, fn in (("a", parca_a), ("b", parca_b), ("c", parca_c), ("d", parca_d), ("e", parca_e)):
        if p in K:
            fn(K)
        else:
            print(f"{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
