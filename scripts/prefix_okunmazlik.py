#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import re
import sys
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from prefix_korpus_pilot import tekrar_orani

BAR_ORAN = 0.25
P_CAPA = 95

_SOZ = re.compile(r"^[A-Za-z][A-Za-z'’\-]*[.,;:!?)\]\"'”’]*$")
_MD_KIRPMA = str.maketrans({c: " " for c in "—–*_`#>|"})
_MD_BAS = re.compile(r"^(?:[-•+]|\d+[.)])$")


def okunmazlik(metin):
    p = [t for t in str(metin or "").translate(_MD_KIRPMA).split()
         if not _MD_BAS.match(t)]
    if len(p) < 2:
        return 1.0
    return 1.0 - sum(1 for t in p if _SOZ.match(t)) / len(p)


def esik_capa(degerler_capa, p=P_CAPA):
    v = np.asarray([x for x in degerler_capa if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, p))


def cift_kanal(satirlar, kol_al, metin_al, capa_kolu="l0", bar=BAR_ORAN):
    T, O, K = [], [], []
    for r in satirlar:
        m = metin_al(r)
        T.append(float(tekrar_orani(m)))
        O.append(okunmazlik(m))
        K.append(kol_al(r))
    T, O, K = np.array(T), np.array(O), np.array(K, dtype=object)
    capa = K == capa_kolu
    if not capa.any():
        raise SystemExit(f"{capa_kolu}")
    eT, eO = esik_capa(T[capa]), esik_capa(O[capa])
    hucre = {}
    for k in sorted(set(K.tolist())):
        m = K == k
        fT = float(np.mean(T[m] > eT))
        fO = float(np.mean(O[m] > eO))
        hucre[k] = dict(n=int(m.sum()), frac_tekrar=round(fT, 4), frac_okunmazlik=round(fO, 4),
                        kirli_tekrar=bool(fT >= bar), kirli_okunmazlik=bool(fO >= bar),
                        kirli=bool(fT >= bar or fO >= bar))
    return dict(esik_tekrar=round(eT, 4), esik_okunmazlik=round(eO, 4), bar=bar,
                capa_kolu=capa_kolu, hucre=hucre,
                payda=dict(n_satir=len(satirlar), n_capa=int(capa.sum()),
                           n_kol=len(hucre),
                           red_kirli=sum(1 for v in hucre.values() if v["kirli"])))


def _test():
    olcum = []
    olcum.append(("temiz", okunmazlik("The court ruled that the policy was unlawful, and "
                                      "the ministry withdrew it within a week.")))
    olcum.append(("tekrarli-ama-okunur", okunmazlik("no no no no no no no no no no")))
    olcum.append(("yikilmis", okunmazlik("### ]]}} \\x0a >>>> ~~~ 0x1f ||| ¤¤ →→ ###")))
    olcum.append(("bos", okunmazlik("")))
    for ad, v in olcum:
        print(f"  {ad:24s} okunmazlik={v:.3f}")
    t = dict(olcum)
    assert t["temiz"] < 0.20, "temiz metin yüksek okundu"
    assert t["tekrarli-ama-okunur"] < 0.20, "★ AYRIM BOZUK: tekrar kanalini taklit ediyor"
    assert t["yikilmis"] > 0.80, "yikilmis metin düsük okundu"
    assert t["bos"] == 1.0
    print("")


if __name__ == "__main__":
    _test()
