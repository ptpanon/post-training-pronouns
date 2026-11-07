#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, sys, time
import numpy as np
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import judge_signature as Y1
import judge_signature2 as Y2
import refusal_counter as RS
from reading_style_object import payda

ONB = __DNH_DATA__ + "/uf_bilesen.npz"
HED = __DNH_DATA__ + "/uf_hedge.npy"
RED = __DNH_DATA__ + "/uf_red.npy"
CIK = f"{ROOT}/results/judge_signature3_2026-09-05.json"
TABAN, MARJ = 0.05, 1.5
ADLAR = ("M1_1k", "SAHIS1_1k", "modal_1k", "hedge_1k", "RED")


def red_vektoru(n_bekle):
    if os.path.exists(RED):
        v = np.load(RED)
        if len(v) == n_bekle:
            print(f"  ★ red vektörü ÖNBELLEKTEN ({len(v):,})", flush=True)
            return v
    from datasets import load_dataset
    d = load_dataset("openbmb/UltraFeedback")["train"]
    out = []
    for i in range(len(d)):
        for c in d[i]["completions"]:
            out.append(RS.red_mi(c.get("response") or ""))
    v = np.array(out, dtype=np.float64)
    np.save(RED, v)
    return v


def _prova():
    def hk(b_m1, ci, b_red):
        bar = abs(b_m1) >= TABAN and ci[0] * ci[1] > 0 and b_m1 < 0
        m = abs(b_m1) / max(abs(b_red), 1e-12)
        if not bar:
            return "IMZA-YOK", m
        return ("ÖZGÜL-IMZA-RED-GECTI" if m >= MARJ else "RED-CEZASI"), m
    return {
        "i_ozgul_dogar": hk(-0.19, (-0.20, -0.18), 0.05)[0] == "ÖZGÜL-IMZA-RED-GECTI",
        "ii_red_cezasi_dogar": hk(-0.19, (-0.20, -0.18), 0.18)[0] == "RED-CEZASI",
        "iii_imza_yok_dogar": hk(-0.01, (-0.02, 0.01), 0.05)[0] == "IMZA-YOK",
        "iv_isaret_kapisi": hk(+0.19, (0.18, 0.20), 0.05)[0] == "IMZA-YOK",
        "v_red_sayaci_provasi": all(RS._prova().values()),
    }, hk


def main():
    P, hk = _prova()
    print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    print("   ⇒", "GECTI" if all(P.values()) else "DÜSTÜ", flush=True)
    if not all(P.values()):
        return 4
    Z = np.load(ONB, allow_pickle=True)
    H = np.load(HED)
    y = Z["y"]; g = np.unique(Z["g"], return_inverse=True)[1]; mi = Z["mi"]
    n_model = int(Z["n_model"]); nc = Z["nc"]; jt = Z["n_jeton"].astype(float)
    R_ = red_vektoru(len(y))
    assert len(R_) == len(y), "HIZALAMA BOZUK ⇒ cikis 6"
    print(f"  [PAYDA] red: {int(R_.sum()):,}/{len(R_):,} yanit = "
          f"%{100*R_.mean():.3f} ⇒ esik: %0 ya da %100 ise sütun DEJENERE "
          f"⇒ EYLEM: verdict ÖLCÜLEMEZ", flush=True)
    if R_.sum() < 100 or R_.sum() > len(R_) - 100:
        print("  ★★ RED SÜTUNU DEJENERE ⇒ cikis 7"); return 7
    k1000 = lambda x: 1000.0 * x / np.maximum(jt, 1)
    sut = [Y1._z(k1000(Z["m1_sahis2"])), Y1._z(k1000(Z["m5_yakin"] - Z["m1_sahis2"])),
           Y1._z(k1000(Z["n_modal"])), Y1._z(k1000(H)), Y1._z(R_),
           Y1._z(np.log(np.maximum(jt, 1.0))), Y1._z(np.log(np.maximum(nc, 1.0)))]
    D = np.zeros((len(y), n_model - 1))
    for k in range(1, n_model):
        D[:, k - 1] = (mi == k)
    t0 = time.time()
    R = Y2.kos(y, sut, g, mi, D, ADLAR)
    print(f"{time.time()-t0:.0f}", flush=True)
    b1 = R["M1_1k"]["beta"]; ci = R["M1_1k"]["ci"]; br = R["RED"]["beta"]
    ad, marj = hk(b1, ci, br)
    ad2, marj2 = hk(b1, R["M1_1k"]["ci_iki_yonlu"], br)
    for k in ADLAR:
        print(f"  ★ β_{k:10s} = {R[k]['beta']:+.4f} · CI {R[k]['ci']}", flush=True)
    print(f"\n★ VERDICT: {ad} · marj_red = {marj:.2f} (bar {MARJ}) · "
          f"iki-yönlü okuma: {ad2}", flush=True)
    S = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         SINIF="ÖLCÜM — BANKA ÖNCESI KONTROL",
                         veri="openbmb/UltraFeedback · 1. mührün önbellegi",
                         n_yanit=int(len(y)), n_istem=int(len(np.unique(g))),
                         red_orani=round(float(R_.mean()), 5),
                         n_red=int(R_.sum()), bar=TABAN, marj_bar=MARJ,
                         red_leksikonu="scripts/refusal_counter.py",
                         K3_SERHI="red PASIF sayildi; hicbir mekanizma zayiflatilmadi"),
             _ortak=R, _verdict=dict(birincil=ad, iki_yonlu=ad2, marj_red=float(marj),
                                   marj_red_iki_yonlu=float(marj2),
                                   kume_bagimli=bool(ad != ad2)))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    payda("yargic_imzasi3", n_yanit=int(len(y)), n_istem=int(len(np.unique(g))),
          hal_red=int(R_.sum()), n_sutun=len(ADLAR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
