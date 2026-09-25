#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, re, sys
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import form_count as FS

MADDE = re.compile(r"^\s*(?:\d+\s*[.)\]]|[-*•·+]|[a-zA-Z]\s*[.)])\s+")
ADIM  = re.compile(r"^\s*(?:step|adim)\s*\d*\s*[:.\)]?\s", re.I)
BASLIK = re.compile(r"^\s*#{1,6}\s|^\s*\*\*[^*]+\*\*\s*:?\s*$")
KOD   = re.compile(r"^\s*(?:```|~~~|    \S|\t)")
SAHIS2 = re.compile(r"\b(?:you|your|yours|yourself)\b", re.I)
ALT = ("a_yordamsal", "a_duzyazi")
TESHIS = ("a_duzyazi_2sahis",)


def _satir_bayrak(metin):
    bayrak, kod_ici, ofs = [], False, 0
    for satir in metin.split("\n"):
        if satir.strip().startswith(("```", "~~~")):
            kod_ici = not kod_ici
            b = True
        else:
            b = bool(MADDE.match(satir) or ADIM.match(satir) or BASLIK.match(satir)
                     or KOD.match(satir)) or kod_ici
        bayrak.append((ofs, ofs + len(satir), b))
        ofs += len(satir) + 1
    return bayrak


def _ofs_bayrak(bayrak, i):
    for a, b, f in bayrak:
        if a <= i <= b:
            return f
    return False


def belge_say(doc, metin):
    bayrak = _satir_bayrak(metin)
    S = []
    for s in doc.sents:
        if not any((not x.is_punct and not x.is_space) for x in s):
            continue
        f = FS.sayaclar(s)
        S.append(dict(a=f["a"], liste=_ofs_bayrak(bayrak, s.start_char),
                      metin=" ".join(s.text.split())))
    for i, x in enumerate(S):
        komsu = ((i > 0 and S[i - 1]["a"]) or (i + 1 < len(S) and S[i + 1]["a"]))
        x["dizi"] = bool(x["a"] and komsu)
        x["a_yordamsal"] = int(bool(x["a"] and (x["liste"] or x["dizi"])))
        x["a_duzyazi"] = int(bool(x["a"] and not x["a_yordamsal"]))
        cev = " ".join(S[j]["metin"] for j in range(max(0, i - 1), min(len(S), i + 2)))
        x["a_duzyazi_2sahis"] = int(bool(x["a_duzyazi"] and SAHIS2.search(cev)))
    return S


def say(nlp, metinler, n_process=1, batch=200):
    A = {k: np.zeros(len(metinler)) for k in ("a",) + ALT + TESHIS}
    n_c = np.zeros(len(metinler), dtype=int)
    for i, doc in enumerate(nlp.pipe(metinler, n_process=n_process, batch_size=batch)):
        S = belge_say(doc, metinler[i])
        n_c[i] = len(S)
        for x in S:
            for k in A:
                A[k][i] += x[k]
    return A, n_c


TEST = [
    ("1. Preheat the oven to 350 degrees.", "a_yordamsal"),
    ("2) Add the flour slowly.", "a_yordamsal"),
    ("- Check your network settings.", "a_yordamsal"),
    ("* Restart the service.", "a_yordamsal"),
    ("Step 3: Tighten the bolt.", "a_yordamsal"),
    ("Preheat the oven. Add the flour. Stir well.", "a_yordamsal"),
    ("Talk to your landlord before you withhold rent.", "a_duzyazi"),
    ("Consider seeing a doctor about that pain.", "a_duzyazi"),
    ("Reconsider this plan.", "a_duzyazi"),
    ("Take a breath and read that again.", "a_duzyazi"),
    ("Please stop talking.", None),
    ("You must leave now.", None),
    ("Let's go home.", None),
    ("Would you close the door?", None),
    ("I would drink from your skull.", None),
    ("Maybe just wait a little.", None),
    ("The oven should be preheated.", None),
    ("Why don't we start over?", None),
    ("Can you hear me?", None),
    ("Stop.", "a_duzyazi"),
]


def prova():
    nlp = FS._boru()
    ok, rapor = 0, []
    for t, bek in TEST:
        doc = nlp(t)
        S = belge_say(doc, t)
        top = {k: sum(x[k] for x in S) for k in ("a",) + ALT}
        if bek is None:
            gec = top["a"] == 0
            bul = "a=0" if top["a"] == 0 else "a=" + str(top["a"])
        else:
            gec = top[bek] >= 1
            bul = "+".join(f"{k}={top[k]}" for k in ALT if top[k]) or "a=0"
        ok += int(gec)
        rapor.append(dict(metin=t, beklenen=bek or "a=0", bulunan=bul, gecti=bool(gec)))
        print(f"  {'✓' if gec else '★ DÜSTÜ'} {t[:46]:46s} bekle {str(bek or 'a=0'):12s} → {bul}")
    payda("a_cerceve_prova", n_vaka=len(TEST), hal_gecen=ok, red_dusen=len(TEST) - ok)
    print(f"{ok} {len(TEST)}"
          f"")
    return ok == len(TEST), rapor


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--prova", action="store_true")
    a = ap.parse_args()
    if a.prova:
        ok, _ = prova(); sys.exit(0 if ok else 1)
    print("kullanim: --prova  (tarama `a_bilesim.py`'de)")
