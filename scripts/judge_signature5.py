#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, re, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import judge_signature as Y1
import judge_signature2 as Y2
import judge_signature3 as Y3
import judge_signature4 as Y4
import v29_question_ve_seviye as SQ

ONB = __DNH_DATA__ + "/uf_bilesen.npz"
HED = __DNH_DATA__ + "/uf_hedge.npy"
S3 = __DNH_DATA__ + "/uf_sahis3.npy"
AQ = __DNH_DATA__ + "/uf_aciklama_sorusu.npy"
CIK = f"{ROOT}/results/judge_signature5_2026-09-10.json"
TABAN, MARJ = 0.05, 1.5
ADLAR = ("M1_1k", "SAHIS1_1k", "SAHIS3_1k", "ACIKLAMA_SORUSU", "modal_1k",
         "hedge_1k", "RED", "LISTEKOD", "LOG_JETON", "LOG_CUMLE")
YUZEY = ("modal_1k", "hedge_1k", "RED", "LISTEKOD", "LOG_JETON", "LOG_CUMLE",
         "SAHIS3_1k", "ACIKLAMA_SORUSU")
ONCEKI_M1 = -0.193
SAHIS3 = re.compile(r"\b(he|him|his|she|her|hers|it|its|they|them|their|theirs|"
                    r"himself|herself|itself|themselves)\b", re.I)


def _prova():
    a = "She told him that their report was his."
    b = "I will summarise the material now."
    q = "Would you like me to explain it further?"
    r = "Here is the explanation."
    return {"i_sahis3_sayiyor": len(SAHIS3.findall(a)) == 4,
            "ii_sahis3_sifir": len(SAHIS3.findall(b)) == 0,
            "iii_aq_yakalar": bool(_aq(q)),
            "iv_aq_bos": not bool(_aq(r))}


def _aq(t):
    for c in SQ.cumleler(t or ""):
        if c.endswith("?") and SQ.SAHIS2.search(c) and SQ.TEKLIF.search(c):
            return 1.0
    return 0.0


def _vektorler(n_bekle):
    if os.path.exists(S3) and os.path.exists(AQ):
        v3, va = np.load(S3), np.load(AQ)
        if len(v3) == n_bekle == len(va):
            print(f"  ★ iki vektör de ÖNBELLEKTEN ({len(v3):,})", flush=True)
            return v3, va
    from datasets import load_dataset
    d = load_dataset("openbmb/UltraFeedback")["train"]
    o3, oa = [], []
    for i in range(len(d)):
        for c in d[i]["completions"]:
            t = c.get("response") or ""
            o3.append(float(len(SAHIS3.findall(t))))
            oa.append(_aq(t))
        if i % 10000 == 0:
            print(f"    … {i:,}/{len(d):,}", flush=True)
    v3 = np.array(o3, dtype=np.float64); va = np.array(oa, dtype=np.float64)
    np.save(S3, v3); np.save(AQ, va)
    return v3, va


def main():
    P = _prova()
    print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    if not all(P.values()):
        print("   ⇒ DÜSTÜ"); return 4
    Z = np.load(ONB, allow_pickle=True); H = np.load(HED)
    y = Z["y"]; g = np.unique(Z["g"], return_inverse=True)[1]; mi = Z["mi"]
    n_model = int(Z["n_model"]); nc = Z["nc"]; jt = Z["n_jeton"].astype(float)
    R_ = Y3.red_vektoru(len(y)); L_ = Y4.listekod_vektoru(len(y))
    V3, VA = _vektorler(len(y))
    assert len(V3) == len(VA) == len(y), "HIZALAMA BOZUK ⇒ cikis 6"
    k1000 = lambda x: 1000.0 * x / np.maximum(jt, 1)
    s3 = k1000(V3)
    print(f"  [PAYDA] yeni_sutunlar: sahis3 ort={s3.mean():.2f}/1k sd={s3.std():.2f} · "
          f"aciklama_sorusu pay=%{100*VA.mean():.2f} tam-sifir={int((VA==0).sum()):,}"
          f"/{len(VA):,} ⇒ esik: sd=0 ise sütun DEJENERE ⇒ verdict ÖLCÜLEMEZ", flush=True)
    if s3.std() < 1e-9 or VA.std() < 1e-9:
        print("  ★★ SÜTUN DEJENERE ⇒ cikis 7"); return 7
    sut = [Y1._z(k1000(Z["m1_sahis2"])),
           Y1._z(k1000(Z["m5_yakin"] - Z["m1_sahis2"])),
           Y1._z(s3), Y1._z(VA),
           Y1._z(k1000(Z["n_modal"])), Y1._z(k1000(H)),
           Y1._z(R_), Y1._z(L_),
           Y1._z(np.log(np.maximum(jt, 1.0))), Y1._z(np.log(np.maximum(nc, 1.0)))]
    D = np.zeros((len(y), n_model - 1))
    for k in range(1, n_model):
        D[:, k - 1] = (mi == k)
    t0 = time.time()
    R = Y2.kos(y, sut, g, mi, D, ADLAR)
    print(f"  [{time.time()-t0:.0f}s] ortak denklem (10 sütun) bitti", flush=True)
    b1 = R["M1_1k"]["beta"]; ci = R["M1_1k"]["ci"]
    byuzey = [R[k]["beta"] for k in YUZEY]
    bar = abs(b1) >= TABAN and ci[0] * ci[1] > 0 and b1 < 0
    marj = abs(b1) / max(max(abs(v) for v in byuzey), 1e-12)
    ad = ("IMZA-YOK" if not bar else
          ("ÖZGÜL-IMZA-YÜZEY-GECTI" if marj >= MARJ else "YÜZEY-CEZASI"))
    en_guclu = max(YUZEY, key=lambda k: abs(R[k]["beta"]))
    for k in ADLAR:
        print(f"  ★ β_{k:16s} = {R[k]['beta']:+.4f} · CI {R[k]['ci']}", flush=True)
    kayma = abs(b1) - abs(ONCEKI_M1)
    print(f"\n★ VERDICT: {ad} · marj_yüzey = {marj:.2f} (bar {MARJ}) · "
          f"en güclü yüzey = {en_guclu}")
    print(f"★ 4. MÜHÜRE GÖRE: |β_M1| {abs(ONCEKI_M1):.3f} → {abs(b1):.3f} "
          f"({kayma:+.3f}) ⇒ esik |kayma| ≥ 0,02 ⇒ EYLEM: metinde beyan")
    S = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         SINIF="ÖLCÜM — KISI EKSENI KONTROLLERI",
                         alet="scripts/judge_signature5.py", borc="D-0910-V29 §3b",
                         veri="openbmb/UltraFeedback · 1. mührün önbellegi",
                         n_yanit=int(len(y)), n_istem=int(len(np.unique(g))),
                         sahis3_ort_1k=round(float(s3.mean()), 4),
                         aciklama_sorusu_pay=round(float(VA.mean()), 5),
                         bar=TABAN, marj_bar=MARJ, yuzey=list(YUZEY),
                         onceki_beta_M1=ONCEKI_M1, prova=P),
             _ortak=R, _verdict=dict(ad=ad, marj_yuzey=float(marj),
                                   en_guclu_yuzey=en_guclu, beta_M1=float(b1),
                                   kayma_onceki=float(kayma)))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
