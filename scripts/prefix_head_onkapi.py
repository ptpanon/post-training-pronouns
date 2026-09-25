#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, collections
import numpy as np
ROOT = __DNH_ROOT__ + ""
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from pergel_cekirdek import foldlar
from prefix_muhurlu_kosu import auc_np
import prefix_ladder as M
import prefix_head_avi as H

ESLEME = __DNH_DATA__ + "/onek_korpus/merdiven_esleme/esleme_merdiven_2026-08-07.json"
YARGI = __DNH_DATA__ + "/onek_korpus/merdiven_yargi"
CIKTI = f"{ROOT}/unreleased/HEAD_ONKAPI_2026-08-08.json"
B, SEED, BAR, K_NULL = 200, 20260808, 0.70, 5000
ADLAR = ("AYRIK", "YAPISIK", "ÖLCÜLEMEZ")


def birim(A):
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)


def yon_demeti(X, y, kume, rng):
    from sklearn.linear_model import LogisticRegression
    import anchor_kodlama as CK
    ks = sorted(set(kume.tolist())); ix = {k: np.flatnonzero(kume == k) for k in ks}
    W = []
    for _ in range(B):
        s = rng.choice(len(ks), size=len(ks), replace=True)
        j = np.concatenate([ix[ks[i]] for i in s])
        if len(set(y[j].tolist())) < 2:
            continue
        W.append(LogisticRegression(**CK.KESTIRICI).fit(X[j], y[j]).coef_[0])
    return birim(np.asarray(W, dtype=np.float64))


def kararlilik(W, rng):
    n = len(W); p = rng.permutation(n)
    A, Bh = W[p[:n // 2]], W[p[n // 2:]]
    ua = np.linalg.svd(A, full_matrices=False)[2][:3]
    ub = np.linalg.svd(Bh, full_matrices=False)[2][:3]
    C = np.clip(W @ W.T, -1, 1); iu = np.triu_indices(n, 1)
    return dict(merkezsiz_PC1=round(float(abs(np.dot(ua[0], ub[0]))), 4),
                PC2=round(float(abs(np.dot(ua[1], ub[1]))), 4),
                aci_medyan=round(float(np.median(np.degrees(np.arccos(np.abs(C[iu]))))), 2),
                ort_yon_kosinus=round(float(np.median(np.abs(W @ birim(W.mean(0))))), 4),
                n_yon=int(n))


if __name__ == "__main__":
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    okur, kol = H.BIRINCIL
    X, (y, kume, tarz) = H.veri(okur, kol)
    L, NH, HD = 24, 16, 64
    tut = np.isin(kume, H.TUTULAN)
    sec = ~tut
    ix_s = np.flatnonzero(sec)
    R = dict(damga=damga, prereg="bd19821", hucre=[okur, kol], B=B, adlar=list(ADLAR),
             not_="")
    print(f"HEAD ÖN-KAPISI · damga {damga} · {okur} × {kol} · secim n={len(ix_s)}")

    d_auc = np.zeros((L, NH))
    for l in range(L):
        Xl = np.concatenate([np.asarray(x[:, l, :], dtype=np.float32) for x in X])
        for h in range(NH):
            d_auc[l, h] = H.oof_auc(Xl[ix_s, h*HD:(h+1)*HD], y[ix_s], kume[ix_s], "cuda:0")
        del Xl
    kes_d = np.abs(d_auc - 0.5)
    tum = [(l, h) for l in range(L) for h in range(NH)]
    sirali_d = sorted(tum, key=lambda t: -kes_d[t])
    H5, H2 = sirali_d[:20], sirali_d[:8]
    en_l = int(np.argmax(kes_d.max(1)))
    print(f"  durus: en iyi head AUC {d_auc.max():.4f} · en iyi katman ℓ{en_l} · "
          f"H-5% katmanlari {sorted(l for l,_ in H5)}")

    def birlestir(hs, idx):
        return np.concatenate([H.dilim(X, idx, l, h, HD) for l, h in hs], axis=1)
    Xh = birlestir(H5, ix_s)
    Xk = np.concatenate([np.asarray(x[:, en_l, :], dtype=np.float32) for x in X])[ix_s]
    R["a_kararlilik"] = {}
    for ad, XX in (("H-5% (20 head, d=1280)", Xh), (f"KATMAN ℓ{en_l} (d=1024)", Xk)):
        W = yon_demeti(XX, y[ix_s], kume[ix_s], np.random.default_rng(SEED))
        v = kararlilik(W, np.random.default_rng(SEED + 1))
        v["gecti_bar070"] = bool(v["merkezsiz_PC1"] >= BAR)
        R["a_kararlilik"][ad] = v
        print(f"  (a) {ad:26s} merkezsiz PC1 {v['merkezsiz_PC1']:.3f} · aci medyan "
              f"{v['aci_medyan']:.1f}° · ort-yön kos {v['ort_yon_kosinus']:.3f} "
              f"⇒ bar 0,70 {'GECTI' if v['gecti_bar070'] else 'DÜSTÜ'}", flush=True)
    rng = np.random.default_rng(SEED + 2)
    ks = sorted(set(kume[ix_s].tolist())); ixk = {k: np.flatnonzero(kume[ix_s] == k) for k in ks}
    jac = []
    for _ in range(20):
        s = rng.choice(len(ks), size=len(ks), replace=True)
        j = ix_s[np.concatenate([ixk[ks[i]] for i in s])]
        a2 = np.zeros((L, NH))
        for l in range(L):
            Xl = np.concatenate([np.asarray(x[:, l, :], dtype=np.float32) for x in X])
            for h in range(NH):
                a2[l, h] = auc_np(y[j], Xl[j, h*HD:(h+1)*HD] @ birim(
                    Xl[j][y[j] == 1].mean(0)[h*HD:(h+1)*HD] -
                    Xl[j][y[j] == 0].mean(0)[h*HD:(h+1)*HD]))
            del Xl
        s2 = set(sorted(tum, key=lambda t: -abs(a2[t]-0.5))[:20])
        jac.append(len(s2 & set(H5)) / len(s2 | set(H5)))
    R["a_secim_jaccard"] = dict(medyan=round(float(np.median(jac)), 3),
                                p10=round(float(np.percentile(jac, 10)), 3), K=len(jac))
    print(f"  (a') head SECIMI Jaccard medyan {R['a_secim_jaccard']['medyan']:.3f}")

    E = json.load(open(ESLEME, encoding="utf-8"))
    Y = {}
    for f in sorted(os.listdir(YARGI)):
        for r in json.load(open(f"{YARGI}/{f}", encoding="utf-8")):
            Y[r["id"]] = r["tone"]
    anah = {}
    off = {b: i * 816 for i, b in enumerate(M.SIRA)}
    tez_ix = {}
    for i, b in enumerate(M.SIRA):
        _, S = H.G.satirlar(kol, b)
        for j, r in enumerate(S):
            tez_ix[(b, r["tez"], r["durus"], r["cekim"])] = off[b] + j
    for i, m in E.items():
        if m["sinif"] != "merdiven" or m["merdiven_kol"] != kol or i not in Y:
            continue
        if Y[i] not in ("HARSH", "CALM"):
            continue
        k = (m["basamak"], m["tez"], m["durus"], m["cekim"])
        if k in tez_ix:
            anah[tez_ix[k]] = 1 if Y[i] == "HARSH" else 0
    ti = np.array(sorted(anah)); tl = np.array([anah[i] for i in ti], dtype=np.int8)
    dag = collections.Counter(tarz[ti][tl == 1].tolist())
    print(f"  (b) ton etiketli satir {len(ti)} · HARSH {int(tl.sum())} "
          f"(basamak dagilimi {dict(dag)})")
    t_auc = np.zeros((L, NH))
    for l in range(L):
        Xl = np.concatenate([np.asarray(x[:, l, :], dtype=np.float32) for x in X])
        for h in range(NH):
            t_auc[l, h] = H.oof_auc(Xl[ti, h*HD:(h+1)*HD], tl, kume[ti], "cuda:0")
        del Xl
    T20 = set(sorted(tum, key=lambda t: -abs(t_auc[t]-0.5))[:20])
    kesisim = len(T20 & set(H5))
    rng = np.random.default_rng(SEED + 3)
    nul = np.array([len(set(rng.choice(len(tum), 20, replace=False)) &
                        set(rng.choice(len(tum), 20, replace=False)))
                    for _ in range(K_NULL)])
    p50, p95 = float(np.percentile(nul, 50)), float(np.percentile(nul, 95))
    frac = float((nul >= kesisim).mean())
    verdict = "ÖLCÜLEMEZ" if len(ti) < 30 else ("AYRIK" if kesisim <= p50 else "YAPISIK")
    R["b_ton"] = dict(n_etiketli=int(len(ti)), n_harsh=int(tl.sum()),
                      harsh_basamak=dict(dag), ton_en_iyi_AUC=round(float(t_auc.max()), 4),
                      kesisim=int(kesisim), null_p50=p50, null_p95=p95,
                      null_ort=round(float(nul.mean()), 3), frac_null_gecti=round(frac, 4),
                      verdict=verdict, K_null=K_NULL,
                      ton_top20=[[int(l), int(h)] for l, h in sorted(T20)])
    print(f"      ton en iyi AUC {t_auc.max():.4f} · **KESISIM {kesisim}** · "
          f"null ort {nul.mean():.2f} p50 {p50:.0f} p95 {p95:.0f} · "
          f"frac(null≥) {frac:.3f} ⇒ **{verdict}**")

    R["c_merdiven"] = {}
    for t in M.SIRA:
        s2 = (~tut) & (tarz != t); v1 = tut & (tarz != t)
        idx = np.flatnonzero(s2 | v1); tr, va = s2[idx], v1[idx]
        satir = {}
        for ad, hs in (("H-2% (8)", H2), ("H-5% (20)", H5)):
            satir[ad] = round(H.egit_degerlendir(birlestir(hs, idx), y[idx], tr, va,
                                                 "cuda:0")[0], 4)
        R["c_merdiven"][t] = satir
        print(f"  (c) [{t}] " + " · ".join(f"{k} {v:.3f}" for k, v in satir.items()))
    payda("head_onkapi", n_head=L*NH, n_etiketli=len(ti), B=B, n_null=K_NULL,
          bekle={"n_head": 384, "n_etiketli": 30, "B": 200})
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    print(f"\n★ (b) KUPON: **{verdict}**  ·  → {CIKTI}")
