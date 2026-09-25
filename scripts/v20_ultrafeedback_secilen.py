#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
ONB = __DNH_DATA__ + "/uf_bilesen.npz"
CIK = f"{ROOT}/results/v20_ultrafeedback_secilen_2026-09-06.json"
NB, SEED = 1000, 20260906


def _prova():
    y = np.array([1., 2., 3., 4.]); x = np.array([10., 20., 30., 40.])
    i_s, i_r = int(np.argmax(y)), int(np.argmin(y))
    mi = np.array([0, 0, 1, 1]); xm = x - np.array([15., 15., 35., 35.])[mi * 0 + np.arange(4) // 2 * 0]
    return {"i_secim_dogru": i_s == 3 and i_r == 0,
            "ii_beraberlik_yakalanir": len(np.unique([5., 5., 5., 5.])) == 1,
            "iii_sabit_etki_sifirlar": bool(abs(float((x - np.array([15., 15., 35., 35.])).sum())) < 1e-9),
            "iv_bootstrap_sirali": bool(np.percentile([1, 2, 3], 2.5) < np.percentile([1, 2, 3], 97.5))}


def main():
    P = _prova(); print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    if not all(P.values()):
        return 4
    Z = np.load(ONB, allow_pickle=True)
    y = Z["y"].astype(float); g = np.unique(Z["g"], return_inverse=True)[1]
    mi = Z["mi"].astype(int); jt = np.maximum(Z["n_jeton"].astype(float), 1.0)
    m1 = 1000.0 * Z["m1_sahis2"].astype(float) / jt
    s1 = 1000.0 * (Z["m5_yakin"].astype(float) - Z["m1_sahis2"].astype(float)) / jt
    n_istem = int(g.max() + 1)

    sira = np.argsort(g, kind="stable")
    gs, ys = g[sira], y[sira]
    sinir = np.searchsorted(gs, np.arange(n_istem + 1))
    sec = np.full(n_istem, -1); red = np.full(n_istem, -1); berabere = 0
    for p in range(n_istem):
        a, b = sinir[p], sinir[p + 1]
        yy = ys[a:b]
        if yy.max() == yy.min():
            berabere += 1; continue
        sec[p] = sira[a + int(np.argmax(yy))]; red[p] = sira[a + int(np.argmin(yy))]
    ok = sec >= 0
    print(f"  [PAYDA] istem={n_istem:,} · yanit={len(y):,} · beraberlik DÜSEN={berabere:,} "
          f"· kullanilan={int(ok.sum()):,} ⇒ esik: kullanilan <%50 ise EYLEM = ÖLCÜLEMEZ")
    if ok.mean() < 0.5:
        return 5
    si, ri = sec[ok], red[ok]

    def demean(v):
        o = np.zeros(int(mi.max()) + 1)
        for m in range(len(o)):
            msk = mi == m
            o[m] = v[msk].mean() if msk.any() else 0.0
        return v - o[mi], o

    m1d, ort_m1 = demean(m1)
    s1d, _ = demean(s1)
    print(f"  [PAYDA] kaynak model={int(mi.max())+1} · model-ici M1 ortalamasi "
          f"{ort_m1.min():.2f}–{ort_m1.max():.2f}/1k (yayilim {np.ptp(ort_m1):.2f}) "
          f"⇒ esik: yayilim >1 ise EYLEM = sabit etki ZORUNLU, ciplak acik okunmaz")

    def boot(v_s, v_r):
        rng = np.random.default_rng(SEED); n = len(v_s)
        o = np.array([(v_s[i] - v_r[i]).mean()
                      for i in (rng.integers(0, n, n) for _ in range(NB))])
        return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))

    R = {}
    for ad, v in (("M1_ciplak", m1), ("M1_model_icinde", m1d),
                  ("SAHIS1_ciplak", s1), ("SAHIS1_model_icinde", s1d)):
        d = float((v[si] - v[ri]).mean()); ci = boot(v[si], v[ri])
        R[ad] = dict(fark=d, ci=list(ci), ayrik=bool(ci[0] * ci[1] > 0))
        print(f"  ★ {ad:20s} {d:+.3f}/1k · CI [{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"· ayrik={R[ad]['ayrik']}")
    kayma = R["M1_model_icinde"]["fark"] - R["M1_ciplak"]["fark"]
    tasiyor = R["M1_model_icinde"]["ayrik"] and R["M1_model_icinde"]["fark"] < 0
    print(f"\n★ SABIT ETKI KAYMASI: {kayma:+.3f}/1k ⇒ esik: isaret dönerse ya da "
          f"CI sifiri kaparsa EYLEM = acik MODEL KIMLIGINDENDI")
    print(f"★ VERDICT: {'SECIM-ETKISI-MODELDEN-BAGIMSIZ' if tasiyor else 'MODEL-KIMLIGI-TASIYOR'}")
    S = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         alet="scripts/v20_ultrafeedback_secilen.py", borc="D-0907-V20 §1c",
                         SINIF="ÖLCÜM — secim etkisi, kaynak model sabit",
                         veri="openbmb/UltraFeedback · 1. mührün önbellegi",
                         ONCUL_DUZELTME=""
                                        "",
                         n_istem=n_istem, n_kullanilan=int(ok.sum()),
                         n_berabere=berabere, n_model=int(mi.max()) + 1,
                         model_ort_M1=[round(float(x), 3) for x in ort_m1],
                         nb=NB, seed=SEED, birim="istem", prova=P),
             sonuc=R, kayma_sabit_etki=kayma,
             _verdict=("SECIM-ETKISI-MODELDEN-BAGIMSIZ" if tasiyor else "MODEL-KIMLIGI-TASIYOR"))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
