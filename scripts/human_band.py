#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, random, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
from reading_style_object import payda

EV = "<home>/user"
KAYNAK = {
    "SCOTUS": f"{EV}/.convokit/saved-corpora/supreme-corpus/utterances.jsonl",
    "CMV":    f"{EV}/.convokit/subreddit-changemyview/utterances.jsonl",
}
N_HEDEF = 6528
ASGARI_KELIME = 40
CIK = os.environ.get("INSAN_BANDI_CIK",
                     f"{ROOT}/results/INSAN_BANDI_2026-08-31.json")


def rezervuar_al(R, kay, i, n, rng):
    if len(R) < n:
        R.append(kay); return True
    j = rng.randrange(i)
    if j < n:
        R[j] = kay; return True
    return False


def rezervuar(yol, n, grup_f, seed=20260831):
    rng = random.Random(seed); R = {}; SAY = {}
    gor = 0; kisa = 0; ayrist_hata = 0
    with open(yol, encoding="utf-8", errors="replace") as f:
        for ham in f:
            if len(ham) < ASGARI_KELIME * 4:
                kisa += 1; continue
            gor += 1
            try:
                d = json.loads(ham)
            except Exception:
                ayrist_hata += 1; continue
            t = (d.get("text") or "").strip()
            if len(t.split()) < ASGARI_KELIME:
                kisa += 1; continue
            g = grup_f(d)
            if g is None:
                continue
            R.setdefault(g, []); SAY[g] = SAY.get(g, 0) + 1
            rezervuar_al(R[g], d, SAY[g], n, rng)
    return R, dict(n_satir_gorulen=gor, red_kisa=kisa, red_ayristirma=ayrist_hata,
                   **{f"n_uygun_{g}": v for g, v in sorted(SAY.items())})


def olc(nlp, kayitlar, kume_f):
    M = [(k.get("text") or "").strip() for k in kayitlar]
    V = MK.satir_bilesenleri(nlp, [{"metin": m} for m in M])
    tab = MK.olc_toplam(V)
    ku = np.array([str(kume_f(k)) for k in kayitlar])
    ad = np.unique(ku); rng = np.random.default_rng(20260831); B = []
    _ALAN = ("M1", "M2", "M3", "M4", "M5", "KUVVET")
    for _ in range(500):
        sel = rng.choice(ad, len(ad), replace=True)
        ix = np.concatenate([np.where(ku == u)[0] for u in sel])
        _t = MK.olc_toplam(V[ix])
        B.append([_t[a] for a in _ALAN])
    Bm = np.array(B, dtype=float)
    _ci = {}
    for j, a in enumerate(_ALAN):
        c = Bm[:, j]; c = c[~np.isnan(c)]
        _ci[f"{a}_ci"] = ([float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))]
                          if len(c) else None)
    payda("insan_bandi_ci", n_olcu=len(_ALAN), n_cekim=500, hal_kume=int(len(ad)),
          red_ci_kurulamadi=sum(1 for v in _ci.values() if v is None))
    return tab | dict(n=len(M), n_kume=int(len(ad)), **_ci,
                      jeton_ort=float(V[:, 0].mean()))


def main():
    nlp = FS._boru(); t0 = time.time(); S = {}
    R, p = rezervuar(KAYNAK["SCOTUS"], N_HEDEF,
                     lambda d: (d.get("meta") or {}).get("speaker_type"))
    print(f"  [{time.time()-t0:.0f}s] SCOTUS örneklem: "
          f"{ {k: len(v) for k, v in R.items()} } · payda {p}", flush=True)
    S["SCOTUS"] = {"_payda": p}
    for g, kay in R.items():
        if g not in ("J", "A") or len(kay) < 200:
            continue
        S["SCOTUS"][g] = olc(nlp, kay, lambda d: (d.get("meta") or {}).get("case_id"))
        print(f"  [{time.time()-t0:.0f}s] SCOTUS/{g}: n={len(kay)} "
              f"M1={S['SCOTUS'][g]['M1']:.3f}", flush=True)
    R2, p2 = rezervuar(KAYNAK["CMV"], N_HEDEF, lambda d: "hepsi")
    print(f"  [{time.time()-t0:.0f}s] CMV örneklem: "
          f"{ {k: len(v) for k, v in R2.items()} } · payda {p2}", flush=True)
    S["CMV"] = {"_payda": p2}
    for g, kay in R2.items():
        S["CMV"][g] = olc(nlp, kay, lambda d: d.get("root") or d.get("conversation_id"))
        print(f"  [{time.time()-t0:.0f}s] CMV/{g}: n={len(kay)} "
              f"M1={S['CMV'][g]['M1']:.3f}", flush=True)
    S["_kunye"] = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       SINIF="KESIF/BETIM — bar YOK · ad YOK · prediction YOK (§10)",
                       n_hedef=N_HEDEF, asgari_kelime=ASGARI_KELIME,
                       motor="addressee_run.py b114a7b46a3369cb (edit yok)",
                       not_="")
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("insan_bandi", n_kaynak=len(KAYNAK),
          n_grup_olculen=len([1 for k in ("SCOTUS", "CMV") for g in S[k] if not g.startswith("_")]))
    print(f"\n✓ {CIK}  ({time.time()-t0:.0f} sn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
