#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
from reading_style_object import payda

MERD = {"Tulu3-8B":  [("taban", "sft"), ("sft", "dpo"), ("dpo", "rl")],
        "OLMo2-13B": [("taban", "sft"), ("sft", "dpo"), ("dpo", "instruct")],
        "OLMo2-32B": [("taban", "sft"), ("sft", "dpo"), ("dpo", "instruct")]}
NB = 1000
SEED = 20260906


def _prova():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {"i_yuzdelik_sirali": bool(lo < hi),
            "ii_sifir_iceriyor": bool(lo < 0 < hi),
            "iii_kayik_dislar": bool(np.percentile(x + 10, 2.5) > 0)}


def kume_boot(Va, ia, Vb, ib, alan="KUVVET"):
    ua = np.unique(ia); ka = {u: np.where(ia == u)[0] for u in ua}
    kb = {u: np.where(ib == u)[0] for u in np.unique(ib)}
    ortak = [u for u in ua if u in kb]
    rng = np.random.default_rng(SEED); out = []
    for _ in range(NB):
        sec = rng.choice(ortak, size=len(ortak), replace=True)
        ai = np.concatenate([ka[u] for u in sec]); bi = np.concatenate([kb[u] for u in sec])
        out.append(MK.olc_toplam(Va[ai])[alan] - MK.olc_toplam(Vb[bi])[alan])
    o = np.array(out); o = o[~np.isnan(o)]
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)), len(ortak)


def main():
    P = _prova(); print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    if not all(P.values()):
        return 4
    nlp = FS._boru(); t0 = time.time(); S = {}
    onbellek = {}

    def yukle(a, z):
        if (a, z) not in onbellek:
            R = MK.oku(a, z)
            if R is None:
                onbellek[(a, z)] = None
            else:
                V = MK.satir_bilesenleri(nlp, R)
                onbellek[(a, z)] = (V, np.array([r.get("istem_i") for r in R]))
        return onbellek[(a, z)]

    for aile, adimlar in MERD.items():
        S[aile] = {}
        for (z0, z1) in adimlar:
            A, B = yukle(aile, z1), yukle(aile, z0)
            if A is None or B is None:
                print(f"{aile} {z0} {z1}"); continue
            d = MK.olc_toplam(A[0])["KUVVET"] - MK.olc_toplam(B[0])["KUVVET"]
            lo, hi, nk = kume_boot(A[0], A[1], B[0], B[1])
            ayrik = (lo > 0) or (hi < 0)
            S[aile][f"{z0}→{z1}"] = dict(dKUVVET=round(float(d), 6),
                                         ci=[round(lo, 6), round(hi, 6)],
                                         ayrik=bool(ayrik), n_istem=nk)
            print(f"  {aile:11s} {z0:6s}→{z1:9s} Δforce {d:+.6f} "
                  f"CI [{lo:+.6f}, {hi:+.6f}] · {'AYRIK' if ayrik else 'sifiri iceriyor'} "
                  f"· küme {nk} istem", flush=True)
    CIK = f"{ROOT}/results/BASAMAK_KUVVET_CI_2026-09-06.json"
    json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   SINIF="ÖLCÜM — istem-kümeli bootstrap, kuvvet toplami",
                   nb=NB, seed=SEED, kume="istem", adimlar=S),
              open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n = sum(len(v) for v in S.values())
    ay = sum(1 for v in S.values() for x in v.values() if x["ayrik"])
    print(f"\n★ {n} basamak · CI-ayrik {ay} · sifiri iceren {n-ay}")
    print(f"✓ {CIK} · {(time.time()-t0)/60:.1f} dk")
    payda("basamak_kuvvet_ci", n_basamak=n, hal_ayrik=ay, red_ayrik_degil=n-ay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
