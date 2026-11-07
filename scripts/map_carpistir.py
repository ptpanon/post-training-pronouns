#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, sys
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import family_profile_map as HP
DOC = f"{ROOT}/results"

N_ASGARI_CIFT = 6
OKUMA_ONEK = "okuma_"
URETIM = ("form_", "kip_", "dU_", "EK_dU", "dpo_", "cos_", "lm_head", "sonda_")


def matris(R):
    A, E = R["aile"], R["eksen"]
    return np.array([[R["X"][a][e] for e in E] for a in A], float), A, E


def spearman_ciftli(M):
    from scipy.stats import spearmanr
    p = M.shape[1]
    C = np.full((p, p), np.nan); N = np.zeros((p, p), int)
    for i in range(p):
        for j in range(p):
            ok = ~np.isnan(M[:, i]) & ~np.isnan(M[:, j])
            N[i, j] = ok.sum()
            if ok.sum() >= N_ASGARI_CIFT and i != j:
                v = spearmanr(M[ok, i], M[ok, j]).statistic
                C[i, j] = float(v) if np.isfinite(v) else np.nan
            elif i == j:
                C[i, j] = 1.0
    return C, N


def tam_blok(M, A, E, asgari_eksen_n=13):
    sut = [j for j in range(len(E)) if (~np.isnan(M[:, j])).sum() >= asgari_eksen_n]
    Ms = M[:, sut]
    sat = [i for i in range(len(A)) if not np.isnan(Ms[i]).any()]
    return Ms[sat], [A[i] for i in sat], [E[j] for j in sut], \
        [A[i] for i in range(len(A)) if i not in sat], \
        [E[j] for j in range(len(E)) if j not in sut]


def z(M):
    s = M.std(0, ddof=0); s[s == 0] = 1.0
    return (M - M.mean(0)) / s


def cift_null(M, K=400, seed=20260828):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    n, p = M.shape
    maxlar, hepsi = [], []
    per_n = {}
    for _ in range(K):
        S = np.full_like(M, np.nan)
        for j in range(p):
            ok = np.where(~np.isnan(M[:, j]))[0]
            S[ok, j] = M[rng.permutation(ok), j]
        en = 0.0
        for i in range(p):
            for j in range(i + 1, p):
                o = ~np.isnan(S[:, i]) & ~np.isnan(S[:, j])
                if o.sum() >= N_ASGARI_CIFT:
                    v = spearmanr(S[o, i], S[o, j]).statistic
                    if np.isfinite(v):
                        hepsi.append(abs(float(v))); en = max(en, abs(float(v)))
                        per_n.setdefault(int(o.sum()), []).append(abs(float(v)))
        maxlar.append(en)
    return dict(K=K, max_ort=float(np.mean(maxlar)), max_p95=float(np.percentile(maxlar, 95)),
                cift_p95=float(np.percentile(hepsi, 95)), cift_p99=float(np.percentile(hepsi, 99)),
                n_bazli={int(k): dict(p95=float(np.percentile(v, 95)),
                                      p99=float(np.percentile(v, 99)), n_cekim=len(v))
                         for k, v in sorted(per_n.items())})


def kos():
    R = HP.topla()
    M, A, E = matris(R)
    C, N = spearman_ciftli(M)
    NUL = cift_null(M)
    ciftler = []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            if np.isfinite(C[i, j]):
                ciftler.append((abs(C[i, j]), C[i, j], E[i], E[j], int(N[i, j])))
    ciftler.sort(reverse=True)
    B, Ab, Eb, atilan_aile, atilan_eksen = tam_blok(M, A, E)
    Z = z(B)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    L = linkage(pdist(Z, metric="euclidean"), method="ward")
    kume = {k: {} for k in (2, 3, 4)}
    for k in kume:
        et = fcluster(L, k, criterion="maxclust")
        for a, c in zip(Ab, et):
            kume[k].setdefault(int(c), []).append(a)
    from sklearn.decomposition import PCA
    n_b = min(3, Z.shape[0], Z.shape[1])
    P = PCA(n_components=n_b).fit(Z)
    Y = P.transform(Z)
    yuk = {f"PC{i+1}": {e: float(P.components_[i][j]) for j, e in enumerate(Eb)}
           for i in range(n_b)}
    tek_satir = {}
    for i in range(n_b):
        v = P.components_[i]
        ok = np.array([e.startswith(OKUMA_ONEK) for e in Eb])
        ur = np.array([any(e.startswith(u) for u in URETIM) for e in Eb])
        tek_satir[f"PC{i+1}"] = dict(
            aciklanan=float(P.explained_variance_ratio_[i]),
            tarafsiz_okuma_payi=float(ok.sum() / len(Eb)),
            tarafsiz_uretim_payi=float(ur.sum() / len(Eb)),
            okuma_ort_mutlak_yuk=float(np.abs(v[ok]).mean()) if ok.any() else np.nan,
            uretim_ort_mutlak_yuk=float(np.abs(v[ur]).mean()) if ur.any() else np.nan,
            okuma_toplam_kare=float((v[ok] ** 2).sum()),
            uretim_toplam_kare=float((v[ur] ** 2).sum()))
    fazla = []
    for m, r, a, b, nn in ciftler:
        ref = NUL["n_bazli"].get(nn, {}).get("p95", np.nan)
        fazla.append((m - ref if np.isfinite(ref) else -np.inf, m, r, a, b, nn, ref))
    fazla.sort(reverse=True)
    payda("harita_carpistir", n_eksen=len(E), n_cift_basilan=len(ciftler),
          n_blok_aile=len(Ab), n_blok_eksen=len(Eb),
          red_atilan_aile=len(atilan_aile), red_atilan_eksen=len(atilan_eksen))
    return dict(R=R, M=M, A=A, E=E, C=C, N=N, ciftler=ciftler, fazla=fazla, null=NUL, B=B, Z=Z, Ab=Ab, Eb=Eb,
                atilan_aile=atilan_aile, atilan_eksen=atilan_eksen, L=L, kume=kume,
                PCA=P, Y=Y, yuk=yuk, tek_satir=tek_satir)


if __name__ == "__main__":
    S = kos()
    print(f"\n★ TAM BLOK: {len(S['Ab'])} aile × {len(S['Eb'])} eksen")
    print(f"  elenen aile : {S['atilan_aile']}")
    print(f"  elenen eksen: {S['atilan_eksen']}")
    n = S["null"]
    print(f"\n★ REFERANS (karistirma, K={n['K']}): max|ρ| ort={n['max_ort']:.3f} "
          f"p95={n['max_p95']:.3f} · tek-cift |ρ| p95={n['cift_p95']:.3f} p99={n['cift_p99']:.3f}")
    print("  n-bazli |ρ| referansi: " + " · ".join(
        f"n={k}: p95={v['p95']:.3f}/p99={v['p99']:.3f}" for k, v in sorted(n["n_bazli"].items())))
    print("★ EN GÜCLÜ 10 CIFT (|ρ|):")
    for m, r, a, b, nn in S["ciftler"][:10]:
        ref = S["null"]["n_bazli"].get(nn, {})
        bay = "★ASIYOR" if ref and m > ref["p99"] else ("~p95" if ref and m > ref["p95"] else "")
        print(f"  ρ={r:+.3f} (n={nn:2d})  {a:26s} ↔ {b:26s}"
              f"  [karistirma p95={ref.get('p95', float('nan')):.3f} p99={ref.get('p99', float('nan')):.3f}] {bay}")
    print("")
    for f, m, r, a, b, nn, ref in S["fazla"][:5]:
        print(f"  +{f:.3f}   ρ={r:+.3f} (n={nn:2d}, p95={ref:.3f})  {a}  ↔  {b}")
    print("\n★ KÜMELER (ward, z-skorlu tam blok):")
    for k, v in S["kume"].items():
        print(f"  k={k}: " + " | ".join(f"K{c}: {', '.join(u)}" for c, u in sorted(v.items())))
    print("\n★ PCA:")
    for pc, d in S["tek_satir"].items():
        print(f"  {pc} aciklanan={d['aciklanan']:.3f} · okuma Σyük²={d['okuma_toplam_kare']:.3f}"
              f" (tarafsiz {d['tarafsiz_okuma_payi']:.3f})"
              f" · üretim Σyük²={d['uretim_toplam_kare']:.3f}"
              f" (tarafsiz {d['tarafsiz_uretim_payi']:.3f})")
