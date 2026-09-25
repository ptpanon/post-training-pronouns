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
from reading_style_object import payda

ONB = __DNH_DATA__ + "/uf_bilesen.npz"
HED = __DNH_DATA__ + "/uf_hedge.npy"
CIK = f"{ROOT}/results/judge_signature2_2026-09-03.json"
NB, KPERM, SEED, TABAN, MARJ = 1000, 200, 20260903, 0.05, 1.5
ADLAR = ("M1_1k", "SAHIS1_1k", "modal_1k", "hedge_1k")


def iki_yonlu_se(X, e, g, m):
    XtX = np.linalg.pinv(X.T @ X)
    def V(idx):
        U = np.zeros((len(np.unique(idx)), X.shape[1]))
        for k, u in enumerate(np.unique(idx)):
            s = idx == u
            U[k] = (X[s] * e[s, None]).sum(axis=0)
        return XtX @ (U.T @ U) @ XtX
    gm = g.astype(np.int64) * (m.max() + 1) + m
    Vt = V(g) + V(m) - V(gm)
    d = np.diag(Vt)
    return np.sqrt(np.maximum(d, 0.0))


def kos(y, sut, g, m, D, ad_sut):
    X = np.column_stack(list(sut) + [D])
    X = Y1._sogur(X.astype(float), g); yy = Y1._sogur(y.reshape(-1, 1).copy(), g)[:, 0]
    b = Y1._fit(X, yy); e = yy - X @ b
    se2 = iki_yonlu_se(X, e, g, m)
    ad = np.unique(g); rng = np.random.default_rng(SEED)
    ix_of = {u: np.where(g == u)[0] for u in ad}
    B = []
    for _ in range(NB):
        sel = rng.choice(ad, len(ad), replace=True)
        ix = np.concatenate([ix_of[u] for u in sel])
        B.append(Y1._fit(X[ix], yy[ix])[:len(ad_sut)])
    B = np.array(B)
    rng2 = np.random.default_rng(SEED + 1); N = []
    for _ in range(KPERM):
        yp = yy.copy()
        for u in ad:
            i = ix_of[u]; yp[i] = yy[rng2.permutation(i)]
        N.append(Y1._fit(X, yp)[:len(ad_sut)])
    N = np.array(N)
    out = {}
    for j, a in enumerate(ad_sut):
        ci = [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))]
        ci2 = [float(b[j] - 1.96 * se2[j]), float(b[j] + 1.96 * se2[j])]
        out[a] = dict(beta=float(b[j]), ci=ci, ayrik=bool(ci[1] < 0 or ci[0] > 0),
                      ci_iki_yonlu=ci2, ayrik_iki_yonlu=bool(ci2[1] < 0 or ci2[0] > 0),
                      se_iki_yonlu=float(se2[j]),
                      frac_null=float((np.abs(N[:, j]) >= abs(b[j])).mean()))
    return out


def verdict(R, anahtar="ci"):
    m1 = R["M1_1k"]
    ayrik = m1["ayrik"] if anahtar == "ci" else m1["ayrik_iki_yonlu"]
    bar = abs(m1["beta"]) >= TABAN and ayrik and m1["beta"] < 0
    en_buyuk_kontrol = max(abs(R["modal_1k"]["beta"]), abs(R["hedge_1k"]["beta"]))
    marj = abs(m1["beta"]) / max(en_buyuk_kontrol, 1e-12)
    if not bar:
        return "IMZA-YOK", marj
    return ("ÖZGÜL-IMZA" if marj >= MARJ else "GENEL-ÜSLUP-CEZASI"), marj


def main():
    Z = np.load(ONB, allow_pickle=True)
    H = np.load(HED)
    y = Z["y"]; g = np.unique(Z["g"], return_inverse=True)[1]; mi = Z["mi"]
    n_model = int(Z["n_model"]); nc = Z["nc"]; jt = Z["n_jeton"].astype(float)
    assert len(H) == len(y), "HIZALAMA BOZUK ⇒ cikis 6"
    k1000 = lambda x: 1000.0 * x / np.maximum(jt, 1)
    sut = [Y1._z(k1000(Z["m1_sahis2"])), Y1._z(k1000(Z["m5_yakin"] - Z["m1_sahis2"])),
           Y1._z(k1000(Z["n_modal"])), Y1._z(k1000(H)),
           Y1._z(np.log(np.maximum(jt, 1.0))), Y1._z(np.log(np.maximum(nc, 1.0)))]
    D = np.zeros((len(y), n_model - 1))
    for k in range(1, n_model):
        D[:, k - 1] = (mi == k)
    t0 = time.time()
    R = kos(y, sut, g, mi, D, ADLAR)
    print(f"  [{time.time()-t0:.0f}s] ortak denklem bitti", flush=True)
    h1, marj = verdict(R, "ci"); h2, _ = verdict(R, "ci_iki_yonlu")
    from datasets import load_dataset
    d = load_dataset("openbmb/UltraFeedback")["train"]
    YB = {a: [] for a in ("helpfulness", "honesty", "instruction_following", "truthfulness")}
    for i in range(len(d)):
        for c in d[i]["completions"]:
            an = c.get("annotations") or {}
            for a in YB:
                v = (an.get(a) or {}).get("Rating")
                try:
                    YB[a].append(float(v))
                except (TypeError, ValueError):
                    YB[a].append(np.nan)
    BOY = {}
    for a, v in YB.items():
        v = np.array(v, float); ok = ~np.isnan(v)
        if ok.sum() < 1000:
            BOY[a] = dict(hal="ÖLCÜLEMEZ — yetersiz"); continue
        rr = kos(v[ok], [s[ok] for s in sut], np.unique(g[ok], return_inverse=True)[1],
                 mi[ok], D[ok], ADLAR)
        BOY[a] = {k: dict(beta=rr[k]["beta"], ci=rr[k]["ci"]) for k in ADLAR}
        BOY[a]["n"] = int(ok.sum())
        print(f"  [{time.time()-t0:.0f}s] Rating/{a}: β_M1={rr['M1_1k']['beta']:+.4f}", flush=True)
    S = dict(_ortak=R, _verdict=dict(birincil=h1, iki_yonlu=h2, marj=float(marj),
                                   kume_bagimli=bool(h1 != h2)),
             _rating_betimsel=dict(SINIF="", **BOY),
             _kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         veri="openbmb/UltraFeedback · 1. mührün önbellegi",
                         n_yanit=int(len(y)), n_istem=int(len(np.unique(g))),
                         kapasite=dict(M1_1k=12.311, modal_1k=14.051, hedge_1k=2.320,
                                       modal_orani=1.14, hedge_orani=0.19,
                                       SERH="hedge ESLI DEGIL, sönümlü ⇒ yük tasiyan modal"),
                         taban=TABAN, marj_bari=MARJ, n_bootstrap=NB, k_perm=KPERM))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    payda("p11_yargic_imzasi2", n_yanit=len(y), n_istem=len(np.unique(g)),
          n_rating_kol=len([1 for v in BOY.values() if "hal" not in v]),
          hal_kume_bagimli=int(h1 != h2))
    print(f"\n{'sütun':<12}{'β_std':>9}{'CI95 (istem)':>24}{'CI95 (iki yönlü)':>26}{'frac':>7}")
    for a in ADLAR:
        v = R[a]
        print(f"{a:<12}{v['beta']:>+9.4f}  [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]"
              f"   [{v['ci_iki_yonlu'][0]:+.4f},{v['ci_iki_yonlu'][1]:+.4f}]{v['frac_null']:>7.3f}")
    print(f"\n★ MARJ = |β_M1| / max(|β_modal|,|β_hedge|) = {marj:.2f} ⇒ bar {MARJ}")
    print(f"★★★ VERDICT: {h1}" + (f"  ★ KÜME-BAGIMLI (iki yönlü: {h2})" if h1 != h2 else
                                f"  · iki yönlü okuma HEMFIKIR"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
