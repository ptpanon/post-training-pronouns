#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, glob, json, os, re, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import dejenerelik_olcer as DJ

VERI = __DNH_DATA__ + ""
CIK = f"{VERI}/form_sayim"
DOC = f"{ROOT}/results"
TESTSETI = f"{DOC}/FORM_TESTSETI_2026-08-28.json"

SAYAC = ("a", "b", "c", "d", "e")
SAYAC_AD = {"a": "ciplak-emir", "b": "yükümlülük", "c": "hafifletici",
            "d": "soru-formlu yönlendirici", "e": "öneri-formu"}
HAFIF = re.compile(r"\b(would|could|please|mind|perhaps|maybe|just)\b", re.I)
B_CEKIRDEK = re.compile(r"\byou\s+(?:must|should|need\s+to|have\s+to|ought\s+to|shall)\b", re.I)
B_EK = re.compile(r"\b(?:i|we)\s+(?:need|want)\s+you\s+to\b"
                  r"|\byou\s*(?:'re|\s+are)\s+required\s+to\b", re.I)
E_ONERI = re.compile(r"\blet'?s\b|\blet\s+us\b|\bshall\s+we\b|\bwhy\s+don'?t\s+we\b"
                     r"|\bwe\s+(?:could|should|might|can)\b", re.I)
OZNE_DEP = ("nsubj", "nsubjpass", "expl", "csubj", "csubjpass")


def sayaclar(sent) -> dict:
    t = " ".join(sent.text.split())
    soru = t.endswith("?")
    c = bool(HAFIF.search(t))
    kok = sent.root
    a = (kok.tag_ == "VB" and kok.lower_ != "let" and kok.lemma_.lower() != "let"
         and not any(ch.dep_ in OZNE_DEP for ch in kok.children)
         and not soru and not c)
    b_c = bool(B_CEKIRDEK.search(t))
    b_g = b_c or bool(B_EK.search(t))
    d = False
    if soru:
        md = [x.i for x in sent if x.tag_ == "MD"]
        yu = [x.i for x in sent if x.lower_ == "you"]
        vb = [x.i for x in sent if x.tag_ in ("VB", "VBG")]
        if md and yu and vb:
            d = any(m < y < v for m in md for y in yu for v in vb)
    e = bool(E_ONERI.search(t))
    return dict(a=int(a), b=int(b_g), b_cekirdek=int(b_c), c=int(c),
                d=int(d), e=int(e))


def _boru(n_process=1):
    import spacy
    return spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])


def uretim_say(nlp, metinler, n_process=1, batch=200):
    N = len(metinler)
    A = {k: np.zeros(N, dtype=np.int32) for k in SAYAC + ("b_cekirdek",)}
    n_c = np.zeros(N, dtype=np.int32); n_j = np.zeros(N, dtype=np.int32)
    it = nlp.pipe(metinler, batch_size=batch, n_process=n_process)
    for i, doc in enumerate(it):
        n_j[i] = sum(1 for x in doc if not x.is_space)
        for s in doc.sents:
            if not any((not x.is_punct and not x.is_space) for x in s):
                continue
            n_c[i] += 1
            r = sayaclar(s)
            for k in A:
                A[k][i] += r[k]
    return A, n_c, n_j


BAR_DDOG = 0.90
ASGARI_POZ, ASGARI_KOL = 6, 3


def sinav(cikti=None):
    import spacy
    nlp = _boru()
    M = json.load(open(TESTSETI, encoding="utf-8"))
    payda("form_sinav_yukle", n_madde=len(M), bekle={"n_madde": 70})
    G = {k: np.array([m["altin"][k] for m in M]) for k in SAYAC}
    P = {k: np.zeros(len(M), dtype=int) for k in SAYAC + ("b_cekirdek",)}
    for i, doc in enumerate(nlp.pipe([m["cumle"] for m in M], batch_size=32)):
        ss = [s for s in doc.sents if any((not x.is_punct and not x.is_space) for x in s)]
        for s in ss:
            r = sayaclar(s)
            for k in P:
                P[k][i] |= r[k]
    bol = sum(1 for m in M if len(list(nlp(m["cumle"]).sents)) > 1)
    R, hata = {}, {}
    for k in list(SAYAC) + ["b_cekirdek"]:
        g = G["b" if k == "b_cekirdek" else k]; p = P[k]
        tp = int(((g == 1) & (p == 1)).sum()); fn = int(((g == 1) & (p == 0)).sum())
        tn = int(((g == 0) & (p == 0)).sum()); fp = int(((g == 0) & (p == 1)).sum())
        duy = tp / (tp + fn) if tp + fn else float("nan")
        ozg = tn / (tn + fp) if tn + fp else float("nan")
        dd = (duy + ozg) / 2 if tp + fn else float("nan")
        n_poz = int(g.sum()); n_kol = len({m["kol"] for m, x in zip(M, g) if x})
        if n_poz < ASGARI_POZ or n_kol < ASGARI_KOL:
            ad = "DOGRULUK-ÖLCÜLEMEZ"
        else:
            ad = "GECER" if dd >= BAR_DDOG else "ÖLCÜLEMEZ"
        R[k] = dict(tp=tp, fp=fp, tn=tn, fn=fn, duyarlilik=duy, ozgulluk=ozg,
                    dengeli_dogruluk=dd, duz_dogruluk=float((g == p).mean()),
                    hep_negatif_dogruluk=float((g == 0).mean()),
                    n_pozitif=n_poz, n_pozitif_kol=n_kol, AD=ad)
        hata[k] = [M[i]["kod"] for i in range(len(M)) if g[i] != p[i]]
    payda("form_sinav", n_sayac=len(R), n_madde=len(M), red_cok_cumleli=bol)
    O = dict(surum="form_sayim v1", bar=BAR_DDOG, asgari_pozitif=ASGARI_POZ,
             asgari_kol=ASGARI_KOL, n_madde=len(M), red_cok_cumleli=bol,
             sayac=R, hatali_maddeler=hata)
    if cikti:
        json.dump(O, open(cikti, "w"), ensure_ascii=False, indent=1)
    for k in list(SAYAC) + ["b_cekirdek"]:
        r = R[k]
        print(f"  ({k}) {SAYAC_AD.get(k,'(b) cekirdek liste'):26s} "
              f"dd={r['dengeli_dogruluk']:.3f} (duy {r['duyarlilik']:.3f} / özg {r['ozgulluk']:.3f}) "
              f"· düz={r['duz_dogruluk']:.3f} · hep-negatif tabani={r['hep_negatif_dogruluk']:.3f} "
              f"· poz={r['n_pozitif']}/{r['n_pozitif_kol']} kol ⇒ ★ {r['AD']}"
              + (f"  hata: {','.join(hata[k])}" if hata[k] else ""))
    return O


def kollar():
    Y = []
    for p in sorted(glob.glob(f"{VERI}/c1_panel/*/*/uretim.jsonl")):
        Y.append(("ciplak", os.path.relpath(p, VERI).replace("/uretim.jsonl", ""), p, "metin"))
    for p in sorted(glob.glob(f"{VERI}/c1_panel_template/*/*/uretim.jsonl")):
        Y.append(("template", os.path.relpath(p, VERI).replace("/uretim.jsonl", ""), p, "metin"))
    for p in sorted(glob.glob(f"{VERI}/restorasyon2/*/*.jsonl")):
        Y.append(("r2", os.path.relpath(p, VERI)[:-6], p, "uretim"))
    return Y


def havuz(n_process=24, yalniz=None):
    nlp = _boru()
    os.makedirs(CIK, exist_ok=True)
    K = kollar()
    if yalniz:
        K = [k for k in K if yalniz in k[1]]
    t0 = time.time(); n_yaz = n_atl = 0; n_ur = 0
    for gi, (grup, rel, yol, alan) in enumerate(K):
        hed = f"{CIK}/{rel.replace('/', '__')}.npz"
        if os.path.exists(hed):
            n_atl += 1; continue
        R = [json.loads(l) for l in open(yol, encoding="utf-8")]
        T = [r.get(alan, "") or "" for r in R]
        A, n_c, n_j = uretim_say(nlp, T, n_process=n_process)
        dej = np.array([DJ.bayrakla(t)["DEJENERE"] for t in T], dtype=bool)
        ek = {}
        if grup in ("ciplak", "template"):
            ek = dict(kol_ad=np.array([r["kol"] for r in R]),
                      istem=np.array([r["istem_i"] for r in R], dtype=np.int32),
                      cekim=np.array([r["cekim"] for r in R], dtype=np.int32))
        else:
            ek = dict(istem=np.array([r["istem_i"] for r in R], dtype=np.int32))
        np.savez_compressed(hed, n_cumle=n_c, n_jeton=n_j, dejenere=dej,
                            **{f"s_{k}": v for k, v in A.items()}, **ek)
        n_yaz += 1; n_ur += len(R)
        gec = time.time() - t0
        print(f"  [{gi+1}/{len(K)}] {rel} · n={len(R)} · cümle={int(n_c.sum())} "
              f"· dejenere={int(dej.sum())} · {gec/60:.1f} dk", flush=True)
    payda("form_havuz", n_kol=len(K), n_yazilan=n_yaz, atl_zaten=n_atl, n_uretim=max(n_ur, 1))
    print(f"[PAYDA] form_havuz: n_kol={len(K)} · n_yazilan={n_yaz} · atl_zaten={n_atl} "
          f"· süre={(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", required=True, choices=["sinav", "havuz"])
    ap.add_argument("--cikti"); ap.add_argument("--surec", type=int, default=24)
    ap.add_argument("--yalniz")
    A = ap.parse_args()
    if A.asama == "sinav":
        sinav(A.cikti)
    else:
        havuz(A.surec, A.yalniz)
