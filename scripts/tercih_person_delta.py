#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import filtre_verim as FV
import addressee_olcu as MO
import form_count as FS
from reading_style_object import payda
CIK = f"{ROOT}/results/TERCIH_KISI_DELTA_2026-09-01.json"


def yogunluk(A, nc):
    j = max(A["n_jeton"].sum(), 1)
    p2 = A["m1_sahis2"].sum()
    p1 = A["m5_yakin"].sum() - p2
    return dict(M1=1000 * p2 / j, sahis1=1000 * p1 / j,
                m2=A["m2"].sum() / max(nc.sum(), 1), n_jeton=int(j))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    nlp = FS._boru(); t0 = time.time(); rng = np.random.default_rng(20260829)
    S = {}; toplam_cift = 0
    setler = getattr(FV, "SETLER", None)
    if not setler:
        print("★ FV.SETLER yok ⇒ yükleyici arayüzü degismis, DUR"); return 4
    for cfg in setler:
        ad = cfg["ad"]
        try:
            cikti = FV.yukle(cfg, a.n, rng)
            C, R = (cikti[0], cikti[1]) if len(cikti) >= 2 else (None, None)
            n_top = cikti[2] if len(cikti) > 2 else len(C)
        except Exception as e:
            print(f"  ✗ {ad}: {type(e).__name__}: {e}"); S[ad] = "KOSULMADI"; continue
        if C is None:
            S[ad] = "KOSULMADI"; continue
        Ac, nc_c = MO.topla(nlp, C, n_process=6)
        Ar, nc_r = MO.topla(nlp, R, n_process=6)
        yc, yr = yogunluk(Ac, nc_c), yogunluk(Ar, nc_r)
        jc = np.maximum(Ac["n_jeton"], 1); jr = np.maximum(Ar["n_jeton"], 1)
        d_m1 = 1000 * (Ac["m1_sahis2"] / jc - Ar["m1_sahis2"] / jr)
        d_s1 = 1000 * ((Ac["m5_yakin"] - Ac["m1_sahis2"]) / jc
                       - (Ar["m5_yakin"] - Ar["m1_sahis2"]) / jr)
        S[ad] = dict(n_cift=len(C), n_toplam=n_top,
                     chosen=yc, rejected=yr,
                     havuz_dM1=yc["M1"] - yr["M1"], havuz_dSahis1=yc["sahis1"] - yr["sahis1"],
                     medyan_dM1=float(np.median(d_m1)), medyan_dSahis1=float(np.median(d_s1)),
                     ort_dM1=float(np.mean(d_m1)), sem_dM1=float(np.std(d_m1) / np.sqrt(len(d_m1))),
                     jeton_ort=dict(chosen=float(jc.mean()), rejected=float(jr.mean())))
        toplam_cift += len(C)
        print(f"  [{time.time()-t0:.0f}s] {ad:28s} n={len(C):6d} · "
              f"ΔM1 havuz {S[ad]['havuz_dM1']:+.3f} · medyan {S[ad]['medyan_dM1']:+.4f} · "
              f"Δ1.kisi havuz {S[ad]['havuz_dSahis1']:+.3f}", flush=True)
    S["_kunye"] = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       SINIF="ÖN-KAYITLI (ASSISTANT HAFTA-1 §2) — iki sonuc da §5'e paragraf",
                       yukleyici="",
                       sayac="addressee_olcu.topla sha 777430dfc659dbca",
                       n_cift_toplam=toplam_cift)
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("tercih_kisi_delta", n_set=len(setler), n_olculen=len([k for k in S if not k.startswith("_")]),
          n_cift=toplam_cift)
    print(f"\n✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
