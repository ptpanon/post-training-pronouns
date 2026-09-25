#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, hashlib, time
ROOT = __DNH_ROOT__ + ""
OUT = os.path.dirname(os.path.abspath(__file__))
CARD = "results/model_card_audit_2026-09-16.json"
SOZ = {"EVET": "yes", "HAYIR": "no", "KISI": "person", "YAKIN": "tone", "CARD-YOK": "not on disk"}


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def tex(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("Tülu", r"T\"ulu")


def main():
    K = json.load(io.open(f"{ROOT}/{CARD}", encoding="utf-8"))
    S = K["satir"]; card = [s for s in S if s["tur"] == "card"]; amac = [s for s in S if s["tur"] == "amac"]
    gor, ks = set(), []
    for s in card:
        anahtar = s["depo"]
        if anahtar in gor:
            continue
        gor.add(anahtar); ks.append(s)
    L = [r"\paragraph*{Who names the axis: sixteen models' cards and six objective texts.}",
         r"\vekalet{Search strings and reading labels were committed before any document was read. "
         r"A match counts as \emph{named} unless its context is unrelated (\emph{register an account}, "
         r"\emph{personal information}); \emph{metric} requires a reported number on pronoun or grammatical-person use; "
         r"the objective column reads \emph{person} only where a training objective, reward criterion, preference-collection "
         r"criterion or rater guideline names grammatical person, pronouns or addressing the user as a target, and "
         r"\emph{tone} where it names tone, register or politeness instead. Cards are the files in our own snapshots; "
         r"\emph{not on disk} is not \emph{no}. Absence here is absence in these documents.}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llp{4.1cm}l}\toprule",
         r"document & register, tone, person or pronoun named & person metric reported & objective names the axis \\ \midrule"]
    for s in ks:
        L.append(f"\\texttt{{{tex(s['depo'])}}} & {SOZ[s['adi_geciyor']]} & {SOZ[s['metrik']]} & {SOZ[s['amac']]} \\\\")
    import re
    H = json.load(io.open(f"{ROOT}/{K['ham']}", encoding="utf-8"))
    tv = next(d for d in H["belge"] if d.get("ad") == "Touvron 2023 (Llama 2)")
    ikinci = sorted({m.group(1) for h in tv["isabet"]
                     for m in [re.search(r"2nd\s*\(you[^)]*\)\s*([\d.]+)%", h["baglam"])] if m})
    assert len(ikinci) == 1, ikinci
    NOT = {"Touvron 2023 (Llama 2)": "yes: reports the pretraining corpus's person shares (second person "
                                     + f"${float(ikinci[0]):.1f}\\%$), not the model's output register"}
    L.append(r"\midrule")
    for s in amac:
        L.append(f"{tex(s['ad'])} & {SOZ[s['adi_geciyor']]} & {NOT.get(s['ad'], SOZ[s['metrik']])} & {SOZ[s['amac']]} \\\\")
    say = K["sayim"]
    L += [r"\bottomrule\end{tabular}\end{center}",
          r"\vekalet{Of " + f"${say['card_yuva']}$ card slots, ${say['card_okunan']}$ are on disk and ${say['card_yok']}$ are not "
          r"(their weights are, their cards were never downloaded). No card reports a person metric and no objective text names "
          r"grammatical person as a target. Two sources name \emph{tone}: the Llama~3.1 cards state that safety responses "
          r"were modified to follow tone guidelines, and Llama~2's context-distillation preprompt asks for an empathetic tone. "
          r"One reports a person metric outside the cards: Llama~2 gives the share of pretraining documents containing each "
          r"grammatical person (second person $" + ikinci[0] + r"\%$), for demographic representation, not for what post-training does "
          r"to it.}"]
    io.open(f"{OUT}/CARD_DENETIMI.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/model_card_audit.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="CARD_DENETIMI", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=CARD, sha256_16=sha16(f"{ROOT}/{CARD}")),
                            dict(yol=K["ham"], sha256_16=sha16(f"{ROOT}/{K['ham']}"))],
                   note="renders the committed card audit; no statistic computed here"),
              io.open(f"{OUT}/CARD_DENETIMI.meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(ks)} {len(amac)}"
          f"{say['card_yok']}")
    print(f"  ✓ {OUT}/CARD_DENETIMI.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
