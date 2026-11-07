#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, json, os, sys
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
CIK = f"{ROOT}/results/DEJENERASYON_TABANI_2026-08-31.json"


def farkli4(t):
    w = t.split()
    if len(w) < 8:
        return None
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(g)) / len(g)


def kume(satirlar):
    v = [x for x in (farkli4(r.get("metin") or "") for r in satirlar) if x is not None]
    if not v:
        return None
    a = np.array(v)
    return dict(n=len(a), ort=float(a.mean()), p05=float(np.percentile(a, 5)),
                p01=float(np.percentile(a, 1)), min=float(a.min()),
                pay_alt_030=float((a < 0.30).mean()), pay_alt_050=float((a < 0.50).mean()))


def main():
    S = {"panel": {}, "elicit_j1024": {}}
    for y in sorted(glob.glob(__DNH_DATA__ + "/c1_panel/*/*/uretim.jsonl")):
        a, b = y.split("/")[-3], y.split("/")[-2]
        if a.startswith(("miniDPO", "_")):
            continue
        S["panel"][f"{a}/{b}"] = kume([json.loads(l) for l in open(y, encoding="utf-8")])
    for y in sorted(glob.glob(__DNH_DATA__ + "/elicit/uretim_j1024_*.jsonl")):
        S["elicit_j1024"][os.path.basename(y)[13:-6]] = \
            kume([json.loads(l) for l in open(y, encoding="utf-8")])
    hepsi = [v for g in S.values() for v in g.values() if v]
    enk_ort = min(v["ort"] for v in hepsi)
    enb_pay30 = max(v["pay_alt_030"] for v in hepsi)
    S["_TABAN"] = dict(
        n_bilinen_iyi_anahtar=len(hepsi),
        en_kucuk_ort_farkli4=enk_ort,
        en_buyuk_pay_alt_030=enb_pay30,
        onerilen_esik_ort=round(enk_ort - 0.05, 3),
        onerilen_esik_pay30=round(min(enb_pay30 * 3 + 0.05, 0.5), 3))
    S["_kunye"] = dict(
        SINIF="ESIK TABANI — §4 md-7; B koluna BAKILMADAN ölcüldü",
        kaynak="c1_panel (A kolu) + elicit_j1024 (PREREG-7, ateslenmis)",
        disarida="")
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"★ bilinen-iyi anahtar: {len(hepsi)}")
    print(f"★ en KÜCÜK ort(farkli-4gram): {enk_ort:.4f}")
    print(f"★ en BÜYÜK 'farkli4 < 0,30' payi: {enb_pay30:.4f}")
    kot = sorted(((v['ort'], k) for g in S.values() if isinstance(g, dict)
                  for k, v in g.items() if isinstance(v, dict)))[:5]
    print("★ en tekrarli bes bilinen-iyi anahtar:")
    for o, k in kot:
        print(f"    {k:34s} ort={o:.4f}")
    payda("dejenerasyon_tabani", n_anahtar=len(hepsi), n_kume=2, red_b_kolu=1)
    print(f"\n✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
