#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, re, sys, time
import numpy as np
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import judge_signature as Y1
import judge_signature2 as Y2
import judge_signature3 as Y3

ONB = __DNH_DATA__ + "/uf_bilesen.npz"
HED = __DNH_DATA__ + "/uf_hedge.npy"
LKD = __DNH_DATA__ + "/uf_listekod.npy"
CIK = f"{ROOT}/results/judge_signature4_2026-09-06.json"
TABAN, MARJ = 0.05, 1.5
ADLAR = ("M1_1k", "SAHIS1_1k", "modal_1k", "hedge_1k", "RED", "LISTEKOD",
         "LOG_JETON", "LOG_CUMLE")
YUZEY = ("modal_1k", "hedge_1k", "RED", "LISTEKOD", "LOG_JETON", "LOG_CUMLE")
ONCEKI_M1 = -0.192

MADDE = re.compile(r"^\s*(?:[-*+•]\s+|\d+[.)]\s+|#{1,6}\s+)")
GIRINTI = re.compile(r"^(?: {4,}|\t)")
CIT = re.compile(r"^\s*(?:```|~~~)")


def liste_kod_payi(t: str) -> float:
    sat = (t or "").split("\n")
    n = k = 0
    icinde = False
    for s in sat:
        if CIT.match(s):
            icinde = not icinde
            n += 1; k += 1
            continue
        if not s.strip():
            continue
        n += 1
        if icinde or MADDE.match(s) or GIRINTI.match(s):
            k += 1
    return k / n if n else 0.0


def _prova():
    duz = "This is a plain paragraph.\nAnd a second one."
    lst = "- alpha\n- beta\n1. gamma"
    kod = "text\n```\nx = 1\ny = 2\n```"

    def hk(b1, ci, byuzey):
        bar = abs(b1) >= TABAN and ci[0] * ci[1] > 0 and b1 < 0
        m = abs(b1) / max(max(abs(v) for v in byuzey), 1e-12)
        if not bar:
            return "IMZA-YOK", m
        return ("ÖZGÜL-IMZA-YÜZEY-GECTI" if m >= MARJ else "YÜZEY-CEZASI"), m

    return {
        "i_duz_metin_sifir": liste_kod_payi(duz) == 0.0,
        "ii_tam_liste_bir": liste_kod_payi(lst) == 1.0,
        "iii_kod_bloku_sayilir": liste_kod_payi(kod) > 0.5,
        "iv_bos_metin_sifir": liste_kod_payi("") == 0.0,
        "v_ozgul_dogar": hk(-0.19, (-0.2, -0.18), [0.05, 0.01])[0] == "ÖZGÜL-IMZA-YÜZEY-GECTI",
        "vi_yuzey_cezasi_dogar": hk(-0.19, (-0.2, -0.18), [0.05, 0.18])[0] == "YÜZEY-CEZASI",
        "vii_imza_yok_dogar": hk(-0.01, (-0.02, 0.01), [0.05])[0] == "IMZA-YOK",
        "viii_isaret_kapisi": hk(+0.19, (0.18, 0.2), [0.05])[0] == "IMZA-YOK",
    }, hk


def listekod_vektoru(n_bekle):
    if os.path.exists(LKD):
        v = np.load(LKD)
        if len(v) == n_bekle:
            print(f"  ★ liste/kod vektörü ÖNBELLEKTEN ({len(v):,})", flush=True)
            return v
    from datasets import load_dataset
    d = load_dataset("openbmb/UltraFeedback")["train"]
    out = []
    for i in range(len(d)):
        for c in d[i]["completions"]:
            out.append(liste_kod_payi(c.get("response") or ""))
        if i % 10000 == 0:
            print(f"    … {i:,}/{len(d):,}", flush=True)
    v = np.array(out, dtype=np.float64)
    np.save(LKD, v)
    return v


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
    R_ = Y3.red_vektoru(len(y))
    L_ = listekod_vektoru(len(y))
    assert len(R_) == len(y) and len(L_) == len(y), "HIZALAMA BOZUK ⇒ cikis 6"
    print(f"{100*L_.mean():.2f}"
          f"{int((L_ == 0).sum()):,} {len(L_):,}"
          f"{int((L_ == 1).sum()):,}"
          f"", flush=True)
    if L_.std() < 1e-9:
        print("  ★★ LISTE/KOD SÜTUNU DEJENERE ⇒ cikis 7"); return 7

    k1000 = lambda x: 1000.0 * x / np.maximum(jt, 1)
    sut = [Y1._z(k1000(Z["m1_sahis2"])), Y1._z(k1000(Z["m5_yakin"] - Z["m1_sahis2"])),
           Y1._z(k1000(Z["n_modal"])), Y1._z(k1000(H)), Y1._z(R_), Y1._z(L_),
           Y1._z(np.log(np.maximum(jt, 1.0))), Y1._z(np.log(np.maximum(nc, 1.0)))]
    D = np.zeros((len(y), n_model - 1))
    for k in range(1, n_model):
        D[:, k - 1] = (mi == k)
    t0 = time.time()
    R = Y2.kos(y, sut, g, mi, D, ADLAR)
    print(f"  [{time.time()-t0:.0f}s] ortak denklem (8 sütun) bitti", flush=True)
    b1 = R["M1_1k"]["beta"]; ci = R["M1_1k"]["ci"]
    byuzey = [R[k]["beta"] for k in YUZEY]
    ad, marj = hk(b1, ci, byuzey)
    ad2, marj2 = hk(b1, R["M1_1k"]["ci_iki_yonlu"], byuzey)
    en_guclu = max(YUZEY, key=lambda k: abs(R[k]["beta"]))
    for k in ADLAR:
        print(f"  ★ β_{k:10s} = {R[k]['beta']:+.4f} · CI {R[k]['ci']}", flush=True)
    kayma = abs(b1) - abs(ONCEKI_M1)
    print(f"\n★ VERDICT: {ad} · marj_yüzey = {marj:.2f} (bar {MARJ}) · "
          f"en güclü yüzey kontrolü = {en_guclu} · iki-yönlü okuma: {ad2}")
    print(f"★ ÖNCEKI OKUMAYA GÖRE: |β_M1| {abs(ONCEKI_M1):.3f} → {abs(b1):.3f} "
          f"({kayma:+.3f}) ⇒ esik: |kayma| ≥ 0,02 ise EYLEM = metinde beyan",
          flush=True)
    S = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         SINIF="ÖLCÜM — YÜZEY KONTROLLERI",
                         alet="scripts/judge_signature4.py", borc="D-0907-V19-B §2a",
                         veri="openbmb/UltraFeedback · 1. mührün önbellegi",
                         n_yanit=int(len(y)), n_istem=int(len(np.unique(g))),
                         listekod_ort=round(float(L_.mean()), 5),
                         bar=TABAN, marj_bar=MARJ, yuzey=list(YUZEY),
                         onceki_beta_M1=ONCEKI_M1, prova=P),
             _ortak=R, _verdict=dict(birincil=ad, iki_yonlu=ad2,
                                   marj_yuzey=float(marj),
                                   marj_yuzey_iki_yonlu=float(marj2),
                                   en_guclu_yuzey=en_guclu,
                                   beta_M1=float(b1), kayma_onceki=float(kayma)))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
