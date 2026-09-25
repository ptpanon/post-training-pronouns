#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import load

CARD_1024 = "results/urial_verdict_2026-09-09.json"
CARD_X1 = "results/urial_verdict_x1_2026-09-10.json"

CARD_2048 = "results/URIAL_VERDICT_2048_16_K2048_2026-09-11.json"
CARD_2048_X1 = "results/urial_verdict_2048_x1_2026-09-11.json"
CARD = CARD_2048


def yukle(card=CARD):
    return load(card)


def kiyas(D, kk):
    k = D["kiyas"][kk]
    return k["aile"], k["sayim"], dict(n_aile=k["n_aile"], n_olculen=k["n_olculen"],
                                       bacak_eksik=k["n_bacak_eksik"],
                                       kapsam=k["KAPSAM_HUKMU"])


def bant(aile, alan, disla=("ÖLCÜLEMEZ-KIRPMA",)):
    kul = {a: x for a, x in aile.items()
           if x.get("hal") == "ÖLCÜLDÜ" and x.get("AD") not in disla}
    dis = sorted(set(aile) - set(kul))
    v = [x[alan] for x in kul.values()]
    return min(v), max(v), len(v), dis
