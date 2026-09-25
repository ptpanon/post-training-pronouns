#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, json

KOK = __DNH_ROOT__ + ""
CARD = "results/P11_2X2_ELICIT_2026-09-03.json"


def _sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def yukle(card: str = CARD):
    p = card if card.startswith("/") else f"{KOK}/{card}"
    D = json.load(open(p, encoding="utf-8"))
    B = {}
    for ad, v in D["_aile"].items():
        m = v["M1_1k"]
        B[ad] = dict(
            basamak_tablosu={"base": {"M1": m["taban"]},
                             "instruct": {"M1": m["taban"] + m["gozlenen"]}},
            kontrast="base→instruct",
            n_cift=v.get("n_cift"), n_istem=v.get("n_kume"),
            fark={"M1": {"gozlenen": m["gozlenen"], "ci": m["ci"],
                         "plasebo_ustu": m["plasebo_ustu"],
                         "plasebo_p95": m["plasebo_p95"]}},
            _kesik=v.get("_kesik"))
    k = D["_kunye"]
    B["_kunye"] = dict(damga_utc=k["damga_utc"], kip="9_B_KOLU · 16/16 kapsam",
                       dejenerasyon_esigi=k["dejenerasyon_esigi"],
                       n_uygun=len(D["_aile"]), n_raf=len(D["_raf"]),
                       n_kosulmadi=len(D["_kosulmadi"]),
                       raf=D["_raf"], kosulmadi=list(D["_kosulmadi"]),
                       TAVAN=k["TAVAN"])
    return B, dict(yol=card, sha256_16=_sha16(p))


def kesik_doygun(B, esik: float = 1.0):
    return sorted(a for a, v in B.items()
                  if not a.startswith("_") and v.get("_kesik")
                  and (v["_kesik"].get("instruct") or 0) >= esik)
