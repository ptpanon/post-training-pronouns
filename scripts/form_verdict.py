#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")

BAR_FRAC2 = 0.05
K_NULL = 2000
SEED = 20260828

ADLAR_HUCRE = ("DÜSER", "DÜSMEZ", "ÖLCÜLEMEZ")
ADLAR_CIFT = ("DÜSER", "DÜSMEZ", "PROTOKOLE-BAGIMLI", "ÖLCÜLEMEZ")
ADLAR_R2 = ("IGNE-FORMDA-GÖRÜNÜR", "IGNE-FORMDA-GÖRÜNMEZ", "PLASEBO-KIRLENDI", "ÖLCÜLEMEZ")


def kume_null(Sb, Cb, Sa, Ca, K=K_NULL, seed=SEED):
    rng = np.random.default_rng(seed)
    F = rng.integers(0, 2, size=(K, len(Sb))).astype(bool)
    sb = np.where(F, Sa, Sb).sum(1); cb = np.where(F, Ca, Cb).sum(1)
    sa = np.where(F, Sb, Sa).sum(1); ca = np.where(F, Cb, Ca).sum(1)
    return sa / np.maximum(ca, 1) - sb / np.maximum(cb, 1)


def kontrast(Sb, Cb, Sa, Ca, K=K_NULL, seed=SEED):
    ob = Sb.sum() / max(Cb.sum(), 1); oa = Sa.sum() / max(Ca.sum(), 1)
    delta = float(oa - ob)
    null = kume_null(Sb, Cb, Sa, Ca, K, seed)
    m = float(null.mean()); sd = float(null.std(ddof=1))
    frac2 = float((np.abs(null - m) >= abs(delta - m)).mean())
    return dict(oran_once=float(ob), oran_sonra=float(oa), delta=delta,
                null_ort=m, null_sd=sd, frac2=frac2, K=int(K),
                mde=float(1.645 * sd), n_kume=int(len(Sb)),
                n_cumle_once=int(Cb.sum()), n_cumle_sonra=int(Ca.sum()))


def verdict_hucre(delta, frac2, sayac_ad="GECER", kol_eksik=False):
    if sayac_ad != "GECER" or kol_eksik:
        return ADLAR_HUCRE[2]
    if frac2 <= BAR_FRAC2 and delta < 0:
        return ADLAR_HUCRE[0]
    return ADLAR_HUCRE[1]


def verdict_cift(ad_ciplak, ad_template):
    o = ADLAR_HUCRE[2]
    if ad_ciplak == o and ad_template == o:
        return ADLAR_CIFT[3]
    if ad_ciplak == o:
        return ad_template
    if ad_template == o:
        return ad_ciplak
    return ad_ciplak if ad_ciplak == ad_template else ADLAR_CIFT[2]


def verdict_r2(frac2_ad, frac2_ab, sayac_ad="GECER", kol_eksik=False):
    if sayac_ad != "GECER" or kol_eksik:
        return ADLAR_R2[3]
    if frac2_ab <= BAR_FRAC2:
        return ADLAR_R2[2]
    return ADLAR_R2[0] if frac2_ad <= BAR_FRAC2 else ADLAR_R2[1]


def tarama():
    import full_tarama as TT
    SAY = ["GECER", "DOGRULUK-ÖLCÜLEMEZ"]
    _yok = lambda kw: False
    R = []
    R.append(TT.tam_tarama(
        verdict_hucre, {"delta": TT.izgara(-0.2, 0.2, 9),
                      "frac2": TT.izgara(0.0, 1.0, 9), "sayac_ad": SAY,
                      "kol_eksik": [False, True]},
        set(ADLAR_HUCRE), etiket="form_hucre"))
    R.append(TT.tam_tarama(
        verdict_cift, {"ad_ciplak": list(ADLAR_HUCRE), "ad_template": list(ADLAR_HUCRE)},
        set(ADLAR_CIFT), etiket="form_cift"))
    R.append(TT.tam_tarama(
        verdict_r2, {"frac2_ad": TT.izgara(0.0, 1.0, 9),
                   "frac2_ab": TT.izgara(0.0, 1.0, 9), "sayac_ad": SAY,
                   "kol_eksik": [False, True]},
        set(ADLAR_R2), etiket="form_r2"))
    V = []
    V.append(TT.tam_tarama(
        verdict_hucre, {"delta": TT.izgara(-0.2, 0.2, 9),
                      "frac2": TT.izgara(0.0, 1.0, 9), "sayac_ad": SAY},
        set(ADLAR_HUCRE), etiket="form_hucre_VERI", turetilmis={"kol_eksik": _yok}))
    V.append(TT.tam_tarama(
        verdict_cift, {"ad_ciplak": list(ADLAR_HUCRE), "ad_template": list(ADLAR_HUCRE)},
        set(ADLAR_CIFT), etiket="form_cift_VERI_merdiven"))
    V.append(TT.tam_tarama(
        verdict_r2, {"frac2_ad": TT.izgara(0.0, 1.0, 9),
                   "frac2_ab": TT.izgara(0.0, 1.0, 9), "sayac_ad": SAY},
        set(ADLAR_R2), etiket="form_r2_VERI", turetilmis={"kol_eksik": _yok}))
    for r in R + V:
        TT.bas(r)
    TT.veri_yolu_karsilastir(R[0], V[0], etiket="form_hucre")
    TT.veri_yolu_karsilastir(R[1], V[1], etiket="form_cift_merdiven")
    TT.veri_yolu_karsilastir(R[2], V[2], etiket="form_r2")
    return TT.kapi(*(R + V), etiket="form_kaskad_kapisi")


if __name__ == "__main__":
    tarama()
