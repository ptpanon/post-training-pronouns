#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, glob, hashlib, json, os, re, sys
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import form_count as FS
import bare_imperative_cerceve as AC

VERI = __DNH_DATA__ + "/c1_panel"
SAHIS2 = re.compile(r"\b(?:you|your|yours|yourself|yourselves)\b", re.I)
HITAP  = re.compile(r"^\s*(?:hey|hi|hello|look|listen|friend|dear reader)\b[,!]", re.I)
DEONTIK = FS.B_CEKIRDEK
DEGERLENDIRME = re.compile(
    r"\byou(?:'re| are)\s+(?:wrong|mistaken|incorrect|assuming|confusing|missing|conflating)\b"
    r"|\byour\s+(?:reasoning|argument|logic|assumption|premise|conclusion|claim)\b"
    r"|\byou\s+(?:assume|seem to assume|appear to|misunderstand|overlook)\b", re.I)
DUZELTME = re.compile(
    r"\bis\s+a\s+(?:common\s+)?(?:myth|misconception|misunderstanding|fallacy)\b"
    r"|\b(?:this|that|the\s+(?:claim|idea|notion|belief|assertion|concept|statement))\s+"
    r"(?:that\s+.{0,60}?\s+)?is\s+(?:not|incorrect|false|inaccurate|unsupported)\b"
    r"|\bnot\s+(?:accurate|true|correct|supported by)\b"
    r"|\bcontrary\s+to\s+(?:popular\s+)?(?:belief|opinion)\b"
    r"|\bactually,\s|\bin\s+fact,\s|\bis\s+incorrect\b|\bis\s+false\b", re.I)
VOLITIF = re.compile(r"\b(?:i|we)\s+(?:would|could|might)\b", re.I)
SAHIS1 = re.compile(r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.I)
SAHIS3 = re.compile(r"\b(?:he|she|it|they|him|her|them|his|hers|its|their|theirs)\b", re.I)
TANIMLIK = {"the", "a", "an"}


def cumle_olc(sent, a_duz):
    t = " ".join(sent.text.split())
    f = FS.sayaclar(sent)
    s2 = bool(SAHIS2.search(t)) or bool(HITAP.search(t))
    deo = bool(DEONTIK.search(t))
    deg = bool(DEGERLENDIRME.search(t))
    duz = bool(DUZELTME.search(t))
    jeton = [x for x in sent if not x.is_space]
    art = sum(1 for x in jeton if x.lower_ in TANIMLIK and x.pos_ == "DET")
    edat = sum(1 for x in jeton if x.pos_ == "ADP")
    p3 = len(SAHIS3.findall(t)); p1 = len(SAHIS1.findall(t)); p2 = len(SAHIS2.findall(t))
    modal = sum(1 for x in sent if x.tag_ == "MD")
    return dict(
        n_jeton=len(jeton),
        m1_sahis2=p2 + (1 if HITAP.search(t) else 0),
        m2=int(bool(a_duz) or (s2 and (deo or deg))),
        duzeltme=int(duz), duzeltme_kisisiz=int(duz and not s2),
        m4_volitif=int(bool(VOLITIF.search(t))), n_modal=modal,
        m5_uzak=art + edat + p3, m5_yakin=p1 + p2,
        kuvvet=int(bool(f["a"]) or deo or deg))


def belge_olc(doc, metin):
    S = AC.belge_say(doc, metin)
    out, i = [], 0
    for s in doc.sents:
        if not any((not x.is_punct and not x.is_space) for x in s):
            continue
        a_duz = S[i]["a_duzyazi"] if i < len(S) else 0
        out.append(cumle_olc(s, a_duz)); i += 1
    return out


def topla(nlp, metinler, n_process=6):
    ALAN = ("n_jeton", "m1_sahis2", "m2", "duzeltme", "duzeltme_kisisiz",
            "m4_volitif", "n_modal", "m5_uzak", "m5_yakin", "kuvvet")
    A = {k: np.zeros(len(metinler)) for k in ALAN}
    nc = np.zeros(len(metinler), dtype=int)
    for i, doc in enumerate(nlp.pipe(metinler, n_process=n_process, batch_size=200)):
        R = belge_olc(doc, metinler[i]); nc[i] = len(R)
        for r in R:
            for k in ALAN:
                A[k][i] += r[k]
    return A, nc


def olculer(A, nc):
    j = A["n_jeton"].sum(); c = nc.sum()
    du = A["duzeltme"].sum(); md = A["n_modal"].sum()
    return dict(
        M1=1000 * A["m1_sahis2"].sum() / max(j, 1),
        M2=A["m2"].sum() / max(c, 1),
        M3=(A["duzeltme_kisisiz"].sum() / du) if du > 0 else None,
        M4=(A["m4_volitif"].sum() / md) if md > 0 else None,
        M5=(A["m5_uzak"].sum() - A["m5_yakin"].sum()) / max(j, 1),
        KUVVET=A["kuvvet"].sum() / max(c, 1),
        _payda=dict(n_jeton=int(j), n_cumle=int(c), n_duzeltme=int(du), n_modal=int(md)))


TEST = [
 ("You should reconsider this plan.", dict(m2=1, kuvvet=1)),
 ("You're assuming the mayor caused it.", dict(m2=1, kuvvet=1)),
 ("Preheat the oven to 350 degrees.", dict(m2=1)),
 ("1. Preheat the oven to 350 degrees.", dict(m2=0)),
 ("Preheat the oven. Add the flour. Stir well.", dict(m2=0)),
 ("Talk to your landlord first.", dict(m2=1)),
 ("This is a common misconception.", dict(duzeltme=1, duzeltme_kisisiz=1)),
 ("Your claim that the wall is visible is not accurate.", dict(duzeltme=1, duzeltme_kisisiz=0)),
 ("I would suggest a different approach.", dict(m4_volitif=1)),
 ("The evidence is mixed.", dict(m2=0, duzeltme=0)),
 ("Contrary to popular belief, bats are not blind.", dict(duzeltme=1, duzeltme_kisisiz=1)),
 ("Hey, listen carefully.", dict(m2=1)),
]


def prova():
    nlp = FS._boru(); ok = 0
    print("★ PROVA (K-2e) — istendigi kipte atesliyor mu")
    for t, bek in TEST:
        doc = nlp(t); R = belge_olc(doc, t)
        if not R:
            print(f"  ★ DÜSTÜ (cümle yok) {t}"); continue
        r = R[0]; g = all(r[k] == v for k, v in bek.items()); ok += g
        print(f"  {'✓' if g else '★ DÜSTÜ'} {t[:46]:46s} "
              + " ".join(f"{k}={r[k]}(bek {v})" for k, v in bek.items()))
    payda("muhatap_olcu_prova", n_vaka=len(TEST), hal_gecen=ok, red_dusen=len(TEST) - ok)
    print(f"{ok} {len(TEST)}")
    return ok == len(TEST)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prova", action="store_true")
    a = ap.parse_args()
    if a.prova:
        sys.exit(0 if prova() else 1)
    print("kullanim: --prova  (tarama addressee_run.py'de)")
