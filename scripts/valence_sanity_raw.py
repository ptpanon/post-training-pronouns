#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import csv, json, os, re, sys, collections
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

PANEL = __DNH_DATA__ + "/c1_panel"
SOZLUK = f"{ROOT}/unreleased/Ratings_Warriner_et_al.csv"
CIK = __DNH_DATA__ + "/aksam/hasat/h4_v_saglik"
JETON = re.compile(r"[a-z']+")


def sozluk_yukle():
    V, A, D = {}, {}, {}
    with open(SOZLUK, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            w = (r.get("Word") or "").strip().lower()
            if not w:
                continue
            try:
                V[w] = float(r["V.Mean.Sum"]); A[w] = float(r["A.Mean.Sum"]); D[w] = float(r["D.Mean.Sum"])
            except (KeyError, ValueError, TypeError):
                continue
    return V, A, D


def profil(metin, L):
    ks = JETON.findall((metin or "").lower())
    v = [L[w] for w in ks if w in L]
    return (float(np.mean(v)) if v else None), len(v), len(ks)


def main():
    os.makedirs(CIK, exist_ok=True)
    V, A, D = sozluk_yukle()
    print(f"  [SÖZLÜK] Warriner {len(V)} kelime (V/A/D)")
    satir, bulunan, toplam, n_bos = [], 0, 0, 0
    for aile in sorted(os.listdir(PANEL)):
        for zemin in ("base", "instruct"):
            p = f"{PANEL}/{aile}/{zemin}/uretim.jsonl"
            if not os.path.exists(p):
                continue
            grup = collections.defaultdict(lambda: collections.defaultdict(list))
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                for ad, L in (("V", V), ("A", A), ("D", D)):
                    m, b, t = profil(r.get("metin"), L)
                    if ad == "V":
                        bulunan += b; toplam += t
                        if m is None:
                            n_bos += 1
                    if m is not None:
                        grup[(r.get("kol"), r.get("tur"))][ad].append(m)
            for (kol, tur), dd in sorted(grup.items(), key=lambda x: str(x[0])):
                satir.append(dict(aile=aile, zemin=zemin, kol=kol, tur=tur,
                                  n=len(dd["V"]),
                                  **{f"{k}_ort": round(float(np.mean(x)), 4) for k, x in dd.items()},
                                  **{f"{k}_sd": round(float(np.std(x, ddof=1)), 4)
                                     for k, x in dd.items() if len(x) > 1}))
            print(f"  ✓ {aile}/{zemin} · {len(grup)} kol×tur", flush=True)
    yol = f"{CIK}/V_SAGLIK_HAM.jsonl"
    with open(yol, "w", encoding="utf-8") as f:
        for r in satir:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kapsam = bulunan / max(1, toplam)
    payda("h4_v_saglik", n_satir=len(satir), n_aile=len({r["aile"] for r in satir}),
          n_sozluk=len(V), hal_kapsam_orani=round(kapsam, 4),
          hal_bos_uretim=n_bos, hal_jeton_toplam=toplam)
    json.dump(dict(SINIF="★ KESIF HAMI — verdict YOK, bar YOK, karsilastirma YOK",
                   cetvel="Warriner ve ark. (13 915 kelime, V/A/D)",
                   SINIR="sözlük-tabanli: kelime torbasi · baglam körü · olumsuzlama körü; GÖMÜ DEGILDIR",
                   n_satir=len(satir), sozluk_kapsam_orani=round(kapsam, 4),
                   hal_bos_uretim=n_bos, dosya=yol),
              open(f"{CIK}/KUNYE.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"★ H4 ham: {len(satir)} satir · sözlük kapsami {kapsam:.3f} → {yol}")


if __name__ == "__main__":
    main()
