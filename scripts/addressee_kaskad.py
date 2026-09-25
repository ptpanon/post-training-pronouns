#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

BAR_N_ORTAK = 24
BAR_REJIM = 1.645
BAR_PLASEBO = 0.05
BAR_PANEL_HUCRE = 8
BAR_PANEL_ORAN = 2 / 3
BAR_PANEL_AILE = 3

ADLAR_HUCRE = ["KAPI/ÖLCÜLEMEZ", "NULL-AYIRT-EDEMEDI", "PLASEBO-AYIRT-EDEMEDI",
               "MUHATAP-BÜYÜTÜR", "MUHATAP-KÜCÜLTÜR", "MUHATAP-FARKI-YOK"]
ADLAR_PANEL = ["ÖLCÜLEMEZ-PANEL", "MUHATAP-TASIYICI", "MUHATAP-AILEYE-BAGLI",
               "MUHATAP-KARMA", "MUHATAP-TASIMAZ"]


def hucre(n_ortak_var, n_ortak_yok, null_ort, null_sd, D_ci_alt, D_ci_ust,
          frac_plasebo):
    if min(n_ortak_var, n_ortak_yok) < BAR_N_ORTAK:
        return "KAPI/ÖLCÜLEMEZ"
    if abs(null_ort) / (null_sd or 1e-12) > BAR_REJIM:
        return "NULL-AYIRT-EDEMEDI"
    yonlu = D_ci_alt > 0 or D_ci_ust < 0
    if yonlu and frac_plasebo > BAR_PLASEBO:
        return "PLASEBO-AYIRT-EDEMEDI"
    if D_ci_alt > 0:
        return "MUHATAP-BÜYÜTÜR"
    if D_ci_ust < 0:
        return "MUHATAP-KÜCÜLTÜR"
    return "MUHATAP-FARKI-YOK"


def panel(n_olculebilir, n_buyutur, n_kucultur, n_aile_buyutur, n_aile_kucultur):
    if n_olculebilir < BAR_PANEL_HUCRE:
        return "ÖLCÜLEMEZ-PANEL"
    yonlu = n_buyutur + n_kucultur
    if yonlu == 0:
        return "MUHATAP-TASIMAZ"
    for n, na in ((n_buyutur, n_aile_buyutur), (n_kucultur, n_aile_kucultur)):
        if n >= BAR_PANEL_ORAN * n_olculebilir and na >= BAR_PANEL_AILE:
            return "MUHATAP-TASIYICI"
    if n_buyutur > 0 and n_kucultur > 0:
        return "MUHATAP-AILEYE-BAGLI"
    return "MUHATAP-KARMA"


if __name__ == "__main__":
    from full_tarama import full_tarama, izgara, kapi, bas
    Rh = tam_tarama(
        hucre,
        dict(n_ortak_var=[0, 12, 23, 24, 30, 34], n_ortak_yok=[0, 23, 24, 34],
             null_ort=[-0.02, 0.0, 0.02], null_sd=[0.005, 0.01, 0.05],
             D_ci_alt=[-0.20, -0.05, 0.0, 0.03, 0.10],
             D_ci_ust=[-0.03, 0.0, 0.05, 0.30],
             frac_plasebo=[0.0, 0.04, 0.05, 0.20]),
        ADLAR_HUCRE, etiket="muhatap_hucre")
    bas(Rh, "MUHATAP · HÜCRE KASKADI")
    Rp = tam_tarama(
        panel,
        dict(n_olculebilir=[0, 7, 8, 12, 22], n_buyutur=[0, 2, 6, 8, 16],
             n_kucultur=[0, 2, 6, 8, 16], n_aile_buyutur=[0, 2, 3, 5],
             n_aile_kucultur=[0, 2, 3, 5]),
        ADLAR_PANEL, etiket="muhatap_panel")
    bas(Rp, "MUHATAP · PANEL KASKADI")
    kapi(Rh, Rp, etiket="muhatap_tam_tarama")
