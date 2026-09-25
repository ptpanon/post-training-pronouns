#!/usr/bin/env python3
from __future__ import annotations
import glob, os, re, sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P11 = os.path.join(KOK, "paper")

MUAF = {"figures/ORNEK_CIFT.tex": "model outputs printed verbatim (data, not prose)"}

SOZLUK = [
    ("moved-from",   r"\(moved from\b",                       "parcanin kâgit icindeki göcü"),
    ("line-budget",  r"\bline budget\b",                       "kesim gerekcesi okura ait degil"),
    ("ledger",       r"\b(our|the) ledger\b",                  "ic kütüge gönderme"),
    ("next-prereg",  r"\bthe next pre-?registration\b",        "gelecek turun vaadi"),
    ("has-now-run",  r"\bhas now (run|been|fetched)\b",        "yazimin zaman kipi"),
    ("in-full-colon", r"\bin full:",                           "«gövdeden tasindi» ifsasi"),
    ("now-fetched",  r"\bwe have now fetched\b",               "arastirmanin degil yazimin ani"),
    ("used-to-leave", r"\bused to leave\b|\bthan it used to\b", "paragrafin eski hâli"),
    ("earlier-ver",  r"\ban earlier version of (that|this) (figure|section|draft)\b",
                     "taslagin eski sürümü (ÖLCÜMÜN eski sürümü DEGIL)"),
    ("first-reading", r"\bour first reading\b",                "yazarin okuma gecmisi"),
    ("superseded",   r"\bis superseded here\b",                "kâgit ici sürüm iliskisi"),
    ("when-written", r"\bwhen this passage was written\b",     "yazim ani"),
    ("once-did",     r"\bthat once did\b",                     "tablonun eski hâli"),
]

KAYIT_ORNEK = [
    "an earlier version of the person-restoration comparison paired each item",
    "a sampler concentrated in recent terms read advocates at 10.3 instead",
    "two legs of the first reading hit a device-mapping fault and were re-read",
    "bullfighting has now become a thing of the past",
]
TASLAK_ORNEK = [
    "the clause that left \\S\\ref{sec:dial} for the line budget",
    "the coordinate is recorded as open in our ledger",
    "\\paragraph*{The control, in full (moved from \\S\\ref{sec:x}).}",
    "The control has now run on 16 of 16 families.",
    "the next pre-registration adds an echo flag",
]


def _yorumsuz(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def tara(kok: str = P11):
    hedef = ["p11.tex"] + sorted(
        os.path.relpath(f, kok) for f in glob.glob(os.path.join(kok, "fig", "*.tex")))
    vur, n_satir, n_muaf = [], 0, 0
    for rel in hedef:
        if rel in MUAF:
            n_muaf += 1
            continue
        p = os.path.join(kok, rel)
        if not os.path.exists(p):
            continue
        for i, ham in enumerate(open(p, encoding="utf-8", errors="ignore"), 1):
            n_satir += 1
            s = _yorumsuz(ham)
            for ad, d, _ in SOZLUK:
                for m in re.finditer(d, s, re.I):
                    vur.append((ad, rel, i, s[max(0, m.start() - 40):m.start() + 45].strip()))
    return vur, n_satir, len(hedef), n_muaf


def prova():
    def ates(s):
        return [ad for ad, d, _ in SOZLUK if re.search(d, s, re.I)]
    poz = {s: ates(s) for s in TASLAK_ORNEK}
    neg = {s: ates(s) for s in KAYIT_ORNEK}
    tamam_poz = all(v for v in poz.values())
    tamam_neg = all(not v for v in neg.values())
    print(f"   ★ KAPI-23 PROVA · pozitif {sum(1 for v in poz.values() if v)}/{len(poz)} atesledi · "
          f"negatif {sum(1 for v in neg.values() if not v)}/{len(neg)} sessiz kaldi")
    for s, v in poz.items():
        if not v:
            print(f"     ✗ ATESLEMEDI (taslak): {s[:70]}")
    for s, v in neg.items():
        if v:
            print(f"     ✗ YANLIS-POZITIF (arastirma kaydi): {v} ⇐ {s[:70]}")
    return tamam_poz and tamam_neg


def main():
    if "--prova" in sys.argv:
        return 0 if prova() else 23
    vur, n_satir, n_dosya, n_muaf = tara()
    print(f"   ★ KAPI-23 · taslak-ifade sözlügü · taranan {n_dosya - n_muaf} kaynak dosya "
          f"({n_satir} satir, yorumlar haric) · muaf {n_muaf} "
          f"({', '.join(f'{k} — {v}' for k, v in MUAF.items())}) · "
          f"sözlük {len(SOZLUK)} sinif · IHLAL = {len(vur)} ⇒ esik 0 ⇒ "
          f"EYLEM: >0 ise DERLEME DÜSER")
    for ad, rel, i, c in vur[:15]:
        print(f"     ✗ [{ad}] {rel}:{i}: …{c}…")
    if not prova():
        print("")
        return 23
    return 0 if not vur else 23


if __name__ == "__main__":
    sys.exit(main())
