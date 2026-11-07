#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import template_filtered as SS
import addressee_run as MK
import form_count as FS
import v23_template_count as V23
import refusal_counter as RS
from judge_signature4 import liste_kod_payi
from reading_style_object import payda

CARD = f"{ROOT}/results/template_filtered_2026-09-07.json"
CIK = f"{ROOT}/results/spread_calibration_2026-09-14.json"


def aile_olc(arg):
    a, zt, zh = arg
    nlp = FS._boru()
    B = {("sbl", "taban"): SS.oku(SS.SAB, a, zt), ("sbl", "hizali"): SS.oku(SS.SAB, a, zh),
         ("cip", "taban"): SS.oku(SS.CIP, a, zt), ("cip", "hizali"): SS.oku(SS.CIP, a, zh)}
    if any(R is None or len(R) != SS.N_BEK for R in B.values()):
        return dict(aile=a, hal="EKSIK")
    F = {k: SS.bayraklar(R) for k, R in B.items()}
    K = {k: [SS.anahtar(r) for r in R] for k, R in B.items()}
    kirli = set()
    for k in B:
        kirli |= {K[k][i] for i in np.where(F[k][0])[0]}
    tut = {k: np.array([x not in kirli for x in K[k]]) for k in B}
    oran = float(tut[("sbl", "taban")].mean())
    if oran < SS.TUT_ESIK:
        return dict(aile=a, hal="ÖLCÜLEMEZ-SÜZGEC", tutulan_oran=oran)
    out = dict(aile=a, hal="ÖLCÜLDÜ", tutulan_oran=round(oran, 4))
    ij = MK.ALAN.index("n_jeton")
    for z in ("taban", "hizali"):
        R, m = B[("sbl", z)], tut[("sbl", z)]
        V = MK.satir_bilesenleri(nlp, R)
        metin = [r["metin"] for r, t in zip(R, m) if t]
        out[z] = dict(M1=float(MK.olc_toplam(V[m])["M1"]),
                      uzunluk_jeton=float(V[m, ij].mean()),
                      red_orani=float(np.mean([bool(RS.red_mi(x)) for x in metin])),
                      listekod_payi=float(np.mean([liste_kod_payi(x) for x in metin])),
                      n_satir=int(m.sum()))
    return out


def yayilim(deg):
    lo, hi = min(deg), max(deg)
    return dict(min=lo, max=hi, max_min=(hi / lo) if lo > 0 else None)


def main():
    t0 = time.time()
    CIFT, _ = V23.cift_kur()
    S = []
    for i, c in enumerate(CIFT, 1):
        S.append(aile_olc(c)); print(f"  [{i}/{len(CIFT)}] {c[0]} · {(time.time()-t0)/60:.1f} dk", flush=True)
    card = {r["aile"]: r for r in json.load(io.open(CARD, encoding="utf-8"))["aileler"]}
    olc = [r for r in S if r["hal"] == "ÖLCÜLDÜ"]
    sapma = max(max(abs(r[z]["M1"] - card[r["aile"]]["sbl"][f"M1_{z}"]) for z in ("taban", "hizali")) for r in olc)
    print(f"  [PROVA] card M1 ↔ bu kosu · n_aile={len(olc)} · en büyük sapma {sapma:.2e} ⇒ esik 1e-3 ⇒ EYLEM: asarsa CIKIS 4")
    if len(olc) != 16 or sapma > 1e-3:
        print("  ★★ PROVA DÜSTÜ ⇒ kiyas YAZILMAZ"); return 4
    NIC = ("M1", "uzunluk_jeton", "red_orani", "listekod_payi")
    Y = {z: {n: yayilim([r[z][n] for r in olc]) for n in NIC} for z in ("taban", "hizali")}
    for z in ("hizali", "taban"):
        print(f"\n★ {z.upper()} bacagi (sablonlu, süzülmüs, 16 aile) · max/min")
        for n in NIC:
            y = Y[z][n]
            mm = "tanimsiz (min=0)" if y["max_min"] is None else f"{y['max_min']:.2f}×"
            print(f"   {n:15s} min {y['min']:.4f} · max {y['max']:.4f} · {mm}")
    payda("yayilim_kalibrasyon", n_aile=len(CIFT), hal_olculdu=len(olc), hal_prova_sapma=float(sapma),
          red_olculemez=len(S) - len(olc))
    json.dump(dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               alet="scripts/spread_calibration.py", sinif="BETIM · verdict yok · prereg yok",
                               kaynak_card="results/template_filtered_2026-09-07.json",
                               suzgec="template_filtered dört hücre kurali (ithal)", olcu="16 aile arasi max/min",
                               prova_M1_sapma=float(sapma), gecen_dk=round((time.time() - t0) / 60, 1)),
                   yayilim=Y, aileler=S),
              io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
