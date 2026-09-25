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

RICA = re.compile(r"\b(?:would|could)\s+you\b"
                  r"|\bwould\s+you\s+mind\b|\bdo\s+you\s+mind\b"
                  r"|\bif\s+you\s+(?:would|could)\b", re.I)
KOR_CAN = re.compile(r"\b(?:can|will)\s+you\b", re.I)
ISTEK = re.compile(r"\b(?:i|we)\s+(?:would|could|might)\b", re.I)
LUTFEN = re.compile(r"\bplease\b", re.I)
ALT = ("c_rica", "c_istek", "c_lutfen", "c_artik")
TESHIS = ("c_KOR_can_will",)


def alt_siniflar(sent):
    t = " ".join(sent.text.split())
    c = int(bool(FS.HAFIF.search(t)))
    kor = int(bool(KOR_CAN.search(t)))
    if not c:
        return dict(c=0, c_rica=0, c_istek=0, c_lutfen=0, c_artik=0, c_KOR_can_will=kor)
    rica = int(bool(RICA.search(t)))
    lutfen = int(bool(LUTFEN.search(t)))
    istek = 0
    if ISTEK.search(t):
        for x in sent:
            if x.tag_ == "MD" and x.lower_ in ("would", "could", "might"):
                if any(ch.dep_ == "nsubj" and ch.lower_ in ("i", "we")
                       for ch in x.children) or \
                   any(ch.dep_ == "nsubj" and ch.lower_ in ("i", "we")
                       for ch in (x.head.children if x.head is not x else [])):
                    istek = 1
                    break
    artik = int(not (rica or istek or lutfen))
    return dict(c=c, c_rica=rica, c_istek=istek, c_lutfen=lutfen, c_artik=artik,
                c_KOR_can_will=kor)


def say(nlp, metinler, n_process=1, batch=200):
    A = {k: np.zeros(len(metinler)) for k in ("c",) + ALT + TESHIS}
    n_c = np.zeros(len(metinler), dtype=int)
    for i, doc in enumerate(nlp.pipe(metinler, n_process=n_process, batch_size=batch)):
        for s in doc.sents:
            if not any((not x.is_punct and not x.is_space) for x in s):
                continue
            f = alt_siniflar(s)
            for k in A:
                A[k][i] += f[k]
            n_c[i] += 1
    return A, n_c


TEST = [
    ("I would drink from your skull.", "c_istek"),
    ("Would you mind closing the door?", "c_rica"),
    ("Could you please send the file?", "c_rica"),
    ("Please stop talking.", "c_lutfen"),
    ("Maybe we should just wait.", "c_artik"),
    ("I could help with that.", "c_istek"),
    ("Can you hear me?", None),
    ("Will you close the door?", None),
    ("Perhaps it is fine.", "c_artik"),
    ("You must leave now.", None),
]


def prova():
    nlp = FS._boru()
    ok, rapor = 0, []
    for t, bek in TEST:
        d = nlp(t); s = list(d.sents)[0]
        f = alt_siniflar(s)
        if bek is None:
            gec = f["c"] == 0
            bulunan = "c=0" if f["c"] == 0 else "c=1"
        else:
            gec = f[bek] == 1
            bulunan = "+".join(k for k in ALT if f[k]) or "—"
        ok += int(gec)
        rapor.append(dict(cumle=t, beklenen=bek or "c=0", bulunan=bulunan, gecti=bool(gec)))
        print(f"  {'✓' if gec else '★ DÜSTÜ'} {t:38s} bekle {str(bek or 'c=0'):9s} → {bulunan}")
    payda("c_cerceve_prova", n_vaka=len(TEST), hal_gecen=ok, red_dusen=len(TEST) - ok)
    if ok < len(TEST):
        raise SystemExit(f"{ok} {len(TEST)}")
    return rapor


KOL_YUVA = {"EMIR_ARTI": 236, "EMIR_EKSI": 664, "FORM_YOK": 710, "RASTGELE": 207}
PANEL = __DNH_DATA__ + "/c1_panel"
K0 = __DNH_DATA__ + "/k0_gurultu"


def _bir(nlp, yol, npr):
    R = [json.loads(l) for l in open(f"{yol}/uretim.jsonl", encoding="utf-8")]
    A, n_c = say(nlp, [r["metin"] for r in R], n_process=npr, batch=128)
    top_c = max(float(n_c.sum()), 1.0)
    return {k: float(v.sum()) / top_c for k, v in A.items()}, len(R), int(n_c.sum())


def kos(npr=24):
    import time
    t0 = time.time(); nlp = FS._boru(npr); O = {}
    for kol, kk in KOL_YUVA.items():
        V = []
        for i in range(4):
            y = f"{PANEL}/miniDPO/ckpt-{kk + i*1000:03d}"
            if not os.path.exists(f"{y}/uretim.jsonl"):
                print(f"{kol} {i+1}"); continue
            v, n, nc = _bir(nlp, y, npr); V.append(v)
            print(f"  {kol:10s} m{i+1} · " + " · ".join(f"{k} {v[k]:.5f}"
                  for k in ("c",) + ALT + TESHIS), flush=True)
        O[kol] = {k: [float(np.mean([v[k] for v in V])),
                      float(np.std([v[k] for v in V], ddof=1))] for k in V[0]} if V else None
    T = []
    for t in range(6):
        y = f"{K0}/seed{t}"
        if os.path.exists(f"{y}/uretim.jsonl"):
            v, n, nc = _bir(nlp, y, npr); T.append(v)
    O["TABAN_m6"] = {k: [float(np.mean([v[k] for v in T])),
                         float(np.std([v[k] for v in T], ddof=1))] for k in T[0]} if T else None
    import hashlib
    r = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="BETIM — bar YOK · ad YOK · prediction YOK (DENETIM-GÜNÜ §5)",
             sha_rule=hashlib.sha256((RICA.pattern + ISTEK.pattern
                                       + LUTFEN.pattern + KOR_CAN.pattern).encode()).hexdigest()[:16],
             prova=prova(), cumle_basina=O, saniye=round(time.time() - t0, 1))
    payda("c_cerceve_kos", n_kol=len(KOL_YUVA), n_okunan=sum(1 for v in O.values() if v),
          n_alt=len(ALT))
    y = f"{ROOT}/results/C_CERCEVE_2026-08-31.json"
    json.dump(r, open(y, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {y}  ({r['saniye']/60:.1f} dk)")
    return r


if __name__ == "__main__":
    import hashlib
    if "--kos" in sys.argv:
        raise SystemExit(0 if kos() else 1)
    print(json.dumps(dict(prova=prova(),
                          sha_rule=hashlib.sha256(
                              (RICA.pattern + ISTEK.pattern + LUTFEN.pattern).encode()
                          ).hexdigest()[:16]), ensure_ascii=False, indent=1)[-260:])
