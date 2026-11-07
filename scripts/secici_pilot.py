import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from kt_kodlama import load_split, BOLMELER, KapsamHatasi

MNT = __DNH_DATA__ + ""
MAN = __DNH_ROOT__ + "/unreleased/secici_pilot_2026-07-30.json"
NFOLD, NBOOT, NPERM, NSTAB = 5, 1000, 200, 10
CREG, HAVUZ = 0.05, "sontoken"
ONELEME_KAT = 8
FISTA_IT, PATH_N = 150, 10
SEED = 20260731

KESKILER = {
    "K1": dict(dizin=f"{MNT}/kt_layers_v2", onek="", model="Qwen3-Embedding-4B", L=37, D=2560),
    "K2": dict(dizin=f"{MNT}/kt_qwen25", onek="", model="Qwen2.5-7B base", L=29, D=3584),
    "K3": dict(dizin=f"{MNT}/kt_batarya", onek="a1__", model="Qwen2.5-7B-Instruct", L=29, D=3584),
    "K4": dict(dizin=f"{MNT}/kt_batarya", onek="a2__", model="Mistral-7B-v0.3 base", L=33, D=4096),
    "K5": dict(dizin=f"{MNT}/kt_batarya", onek="a3__", model="Mistral-7B-Instr-v0.3", L=33, D=4096),
}


def l2(A):
    return A / np.clip(np.linalg.norm(A, axis=-1, keepdims=True), 1e-9, None)


def veri():
    y, g, bol = [], [], []
    for s in BOLMELER:
        r = load_split(s)
        y += [x["etiket"] for x in r]; g += [x["forum_id"] for x in r]; bol += [s] * len(r)
    return np.array(y), np.array(g), np.array(bol)


def yukle(K, n_bek):
    p_, c_ = [], []
    for taraf, akm in (("ebeveyn", p_), ("cevap", c_)):
        for s in BOLMELER:
            f = f"{K['dizin']}/{K['onek']}{s}__{taraf}__{HAVUZ}.fp16.npy"
            a = np.load(f, mmap_mode="r")
            if a.shape[0] != n_bek[s]:
                raise KapsamHatasi(f"{f}: satir {a.shape[0]} ≠ {n_bek[s]}")
            akm.append(np.asarray(a, np.float32))
    P, C = l2(np.concatenate(p_, 0)), l2(np.concatenate(c_, 0))
    nf = int((~np.isfinite(P)).sum() + (~np.isfinite(C)).sum())
    if nf:
        raise KapsamHatasi(f"SONLU DEGIL: {nf} deger — bu gecis degil HATA")
    return P, C


def rol_kur(P, C, rng):
    flip = rng.random(len(P)) < 0.5
    A, B = P.copy(), C.copy()
    A[flip], B[flip] = C[flip], P[flip]
    return A, B, flip.astype(int)


def ozn(A, B, koord):
    kat = {}
    for l, j in koord:
        kat.setdefault(l, []).append(j)
    bloklar = []
    for l in sorted(kat):
        js = np.array(kat[l])
        a, b = A[:, l, :][:, js], B[:, l, :][:, js]
        bloklar.append(np.stack([a, b, np.abs(a - b)], -1).reshape(len(A), -1))
    return np.concatenate(bloklar, 1)


def koord_sirali(kat):
    return [(l, j) for l in sorted(kat) for j in kat[l]]


def oneleme_skoru(A, B, y, dev):
    yd = y[dev].astype(np.float64); yd = (yd - yd.mean()) / (yd.std() + 1e-12)
    L, D = A.shape[1], A.shape[2]
    S = np.zeros((L, D), np.float32)
    for l in range(L):
        a, b = A[dev, l, :], B[dev, l, :]
        for M in (a, b, np.abs(a - b)):
            M = M - M.mean(0)
            r = np.abs((M * yd[:, None]).sum(0) / (np.linalg.norm(M, axis=0) * np.sqrt(len(yd)) + 1e-12))
            S[l] = np.maximum(S[l], r)
    return S


def _lip(X, it=8):
    rng = np.random.default_rng(0)
    v = rng.normal(size=X.shape[1]).astype(X.dtype); v /= np.linalg.norm(v)
    for _ in range(it):
        v = X.T @ (X @ v); nv = np.linalg.norm(v)
        if nv < 1e-30: return 1.0
        v /= nv
    return float(nv) / (4 * len(X))


def group_lasso(X, y01, lam, G, it=FISTA_IT, w0=None, b0=0.0, lip=None):
    dt = X.dtype
    y01 = y01.astype(dt, copy=False)
    n, p = X.shape
    Lf = (_lip(X) if lip is None else lip) + 1e-12
    t = dt.type(1.0 / Lf)
    w = np.zeros(p, dt) if w0 is None else w0.astype(dt, copy=True)
    z, b, bz, tk = w.copy(), dt.type(b0), dt.type(b0), 1.0
    for _ in range(it):
        s = 1.0 / (1.0 + np.exp(-np.clip(X @ z + bz, -30, 30), dtype=dt))
        r = (s - y01) / n
        gw, gb = X.T @ r, r.sum()
        u = z - t * gw
        Uw = u.reshape(-1, G)
        nrm = np.linalg.norm(Uw, axis=1)
        sc = np.maximum(dt.type(0), dt.type(1) - t * dt.type(lam)
                        / np.maximum(nrm, dt.type(1e-30))).astype(dt, copy=False)
        wn = (Uw * sc[:, None]).ravel().astype(dt, copy=False)
        bn = dt.type(bz - t * gb)
        tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        mo = dt.type((tk - 1) / tk1)
        z = (wn + mo * (wn - w)).astype(dt, copy=False)
        bz = dt.type(bn + mo * (bn - b))
        w, b, tk = wn.astype(dt, copy=False), dt.type(bn), tk1
    return w, b


def destek(w, G):
    return np.linalg.norm(w.reshape(-1, G), axis=1)


def lam_yolu(X, y01, G, butce, n_lam=PATH_N):
    n = len(X)
    r0 = (y01.mean() - y01) / n
    gm = np.linalg.norm((X.T @ (-r0)).reshape(-1, G), axis=1)
    lmax = float(gm.max()) * 1.001
    if lmax < 1e-6 * max(1.0, float(np.abs(X).mean())):
        raise KapsamHatasi(""
                           "")
    lams = lmax * np.geomspace(1.0, 0.02, n_lam)
    w, b, sec = None, 0.0, None
    lip = _lip(X)
    for lam in lams:
        w, b = group_lasso(X, y01, lam, G, w0=w, b0=b, lip=lip)
        nz = int((destek(w, G) > 1e-8).sum())
        if nz <= butce:
            sec = (float(lam), nz, w.copy(), b)
        else:
            break
    if sec is None:
        sec = (float(lams[0]), int((destek(w, G) > 1e-8).sum()), w, b)
    return sec, [float(x) for x in lams]


def top_destek(w, G, butce):
    d = destek(w, G)
    k = min(butce, int((d > 1e-8).sum()))
    if k == 0:
        return np.array([], int)
    return np.argsort(-d)[:k]


def duz_probe(X, y, tr, te):
    m = LogisticRegression(C=CREG, max_iter=300).fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]


def hist_derinlik(koord, L, nb=10):
    d = np.array([l / (L - 1) for l, _ in koord])
    h, _ = np.histogram(d, bins=nb, range=(0, 1))
    return (h / max(h.sum(), 1)).astype(float)


def w1(h1, h2):
    return float(np.abs(np.cumsum(h1) - np.cumsum(h2)).sum() / len(h1))


def kapi():
    rng = np.random.default_rng(0)
    n, Gn, G = 800, 40, 3
    y01 = (rng.random(n) < 0.4).astype(np.float32)
    print("[kapi] group-lasso dejenere girdi sinamasi — hicbir sey yazilmaz")
    yarim = n // 2
    for ad, X in (("sabit", np.ones((n, Gn * G), np.float32)),
                  ("saf gürültü", rng.normal(size=(n, Gn * G)).astype(np.float32))):
        try:
            (lam, nz, w, b), _ = lam_yolu(X, y01, G, butce=5)
        except KapsamHatasi as e:
            print(f"  {ad:16s} → KapsamHatasi: {e}  (DOGRU davranis: dejenere girdi DURDURUR)")
            continue
        (l2_, _, w2, b2), _ = lam_yolu(X[:yarim], y01[:yarim], G, butce=5)
        p2 = 1 / (1 + np.exp(-(X[yarim:] @ w2 + b2)))
        auc = roc_auc_score(y01[yarim:], p2) if len(set(p2.round(9))) > 1 else 0.5
        print(f"  {ad:16s} secilen grup {nz:3d}/{Gn} (≤5 olmali) · ÖRNEKLEM-DISI AUC {auc:.3f} "
              f"(0,5 civari olmali)")
    X = rng.normal(size=(n, Gn * G)).astype(np.float32)
    X[:, 3 * 7:3 * 7 + 3] += y01[:, None] * 2.0
    (lam, nz, w, b), _ = lam_yolu(X, y01, G, butce=5)
    d = destek(w, G); sec = np.argsort(-d)[:nz]
    print(f"{'enjekte grup=7':16s} {nz} {sec[:3].tolist()}"
          f"{d[7]:.4f} {np.median(d):.6f}")
    if 7 not in sec.tolist():
        raise SystemExit("")
    print("[kapi] GECTI.", flush=True)


def kos(keskiler):
    t0 = time.time()
    y_d, g, bol = veri()
    tr_all = np.where(bol == "train")[0]; dev = np.where(bol == "dev")[0]
    n_bek = {s: int((bol == s).sum()) for s in BOLMELER}
    print(f"[kapsam] train {len(tr_all)} ({len(np.unique(g[tr_all]))} forum) · "
          f"dev {len(dev)} ({len(np.unique(g[dev]))} forum) · dispute %{100*y_d.mean():.1f} · "
          f"TEST KAPALI · havuz {HAVUZ}", flush=True)
    R = {"_statu": "KESIF — verdict YOK. Esik yok, karar kurali yok. 'Devre bulundu' YASAK.",
         "_protokol": "PROTOKOL_secici_pilot_2026-07-30.md @ 11e78dd",
         "_beklentiler": "BEKLENTILER_secici_pilot_2026-07-30.md @ 76526b3 "
                         "sha256 e84a99d53594dbdc641c9a0ef0771daf775c61731a34dbd176f5c1f5f9c64951",
         "_kapsam": dict(train=len(tr_all), dev=len(dev), havuz=HAVUZ, test="KAPALI",
                         nboot=NBOOT, nperm=NPERM, nstab=NSTAB, oneleme_kat=ONELEME_KAT),
         "keskiler": {}}
    rng = np.random.default_rng(SEED)
    katlar = list(GroupKFold(NFOLD).split(np.zeros(len(tr_all)), y_d[tr_all], g[tr_all]))
    uf = np.unique(g[tr_all]); fidx = {f: np.where(g[tr_all] == f)[0] for f in uf}
    BS = [np.concatenate([fidx[f] for f in rng.choice(uf, len(uf), True)]) for _ in range(NBOOT)]
    STAB = [np.concatenate([fidx[f] for f in rng.choice(uf, len(uf), True)]) for _ in range(NSTAB)]

    for kk in keskiler:
        K = KESKILER[kk]; L, D = K["L"], K["D"]
        P, C = yukle(K, n_bek)
        R["keskiler"][kk] = dict(model=K["model"], L=L, D=D, butce_grup=D, gorevler={})
        for gorev in ("dispute", "rol"):
            tg = time.time()
            if gorev == "dispute":
                A, B, yy = P, C, y_d
            else:
                A, B, yy = rol_kur(P, C, np.random.default_rng(SEED + 7))
            S = oneleme_skoru(A, B, yy, dev)
            kat_skor = np.sort(S, 1)[:, -D:].mean(1)
            l_i = int(np.argmax(kat_skor))
            k_i = koord_sirali({l_i: list(range(D))})
            l_iii = np.argsort(-kat_skor)[:3].tolist()
            hav = [(l, j) for l in l_iii for j in range(D)]
            sk = np.array([S[l, j] for l, j in hav])
            k_ii = [hav[t] for t in np.argsort(-sk)[:D]]
            duz = S.ravel(); ad_n = min(ONELEME_KAT * D, duz.size)
            sel = np.argpartition(-duz, ad_n - 1)[:ad_n]
            k_aday = koord_sirali({l: sorted(sel[sel // D == l] % D) for l in np.unique(sel // D)})
            elenen = 1 - ad_n / duz.size
            Xa = ozn(A, B, k_aday)
            mu, sd = Xa[tr_all].mean(0), Xa[tr_all].std(0) + 1e-9
            Xa = ((Xa - mu) / sd).astype(np.float32)
            (lam, nz, w_dev, b_dev), lams = lam_yolu(Xa[dev], yy[dev].astype(np.float32), 3, D)
            Xtr = Xa[tr_all]; lip_tr = _lip(Xtr)
            w_tr, _ = group_lasso(Xtr, yy[tr_all].astype(np.float32), lam, 3, lip=lip_tr)
            idx_iii = top_destek(w_tr, 3, D)
            k_iii = [k_aday[t] for t in idx_iii]
            print(f"[{kk}/{gorev}] (i) ℓ={l_i} · (ii) ℓ={sorted(l_iii)} · (iii) aday {ad_n} grup "
                  f"(elenen %{100*elenen:.1f}) · λ={lam:.5f} · dev-nz {nz} · train-secilen "
                  f"{len(k_iii)}/{D}", flush=True)
            kol = {}
            oof3, oof3s = np.full(len(tr_all), np.nan), np.full(len(tr_all), np.nan)
            kat_destek = []
            for tr, te in katlar:
                wf, _ = group_lasso(Xtr[tr], yy[tr_all][tr].astype(np.float32), lam, 3)
                idf = top_destek(wf, 3, D); kat_destek.append(idf)
                cf = (3 * idf[:, None] + np.arange(3)).ravel()
                oof3[te] = duz_probe(Xtr[:, cf], yy[tr_all], tr, te)
                cs = (3 * idx_iii[:, None] + np.arange(3)).ravel()
                oof3s[te] = duz_probe(Xtr[:, cs], yy[tr_all], tr, te)
            for ad, koord in (("i", k_i), ("ii", k_ii), ("iii", None), ("iii_sizintili", None)):
                if ad == "iii":
                    oof, koord = oof3, k_iii
                elif ad == "iii_sizintili":
                    oof, koord = oof3s, k_iii
                else:
                    if not koord:
                        kol[ad] = None; continue
                    X = ozn(A, B, koord)
                    m2, s2 = X[tr_all].mean(0), X[tr_all].std(0) + 1e-9
                    X = (X - m2) / s2
                    oof = np.full(len(tr_all), np.nan)
                    for tr, te in katlar:
                        oof[te] = duz_probe(X[tr_all], yy[tr_all], tr, te)
                    del X
                a = float(roc_auc_score(yy[tr_all], oof))
                bo = [roc_auc_score(yy[tr_all][ii], oof[ii]) for ii in BS
                      if len(set(yy[tr_all][ii].tolist())) == 2]
                kol[ad] = dict(auc=round(a, 4), n_grup=len(koord),
                               ci=[round(float(np.percentile(bo, 2.5)), 4),
                                   round(float(np.percentile(bo, 97.5)), 4)],
                               oof_boot_n=len(bo))
            rk = [(int(l), int(j)) for l, j in zip(rng.integers(0, L, D), rng.integers(0, D, D))]
            Xr = ozn(A, B, koord_sirali({l: [j for ll, j in rk if ll == l]
                                         for l in sorted({l for l, _ in rk})}))
            Xr = (Xr - Xr[tr_all].mean(0)) / (Xr[tr_all].std(0) + 1e-9)
            oofr = np.full(len(tr_all), np.nan)
            for tr, te in katlar:
                oofr[te] = duz_probe(Xr[tr_all], yy[tr_all], tr, te)
            kol["rastgele_D"] = dict(auc=round(float(roc_auc_score(yy[tr_all], oofr)), 4),
                                     n_grup=len(rk))
            del Xr
            h_on = hist_derinlik(k_aday, L); h_la = hist_derinlik(k_iii, L)
            kat_h = [hist_derinlik([k_aday[t] for t in idf], L) for idf in kat_destek]
            w1_kat = [w1(kat_h[i], kat_h[j]) for i in range(len(kat_h))
                      for j in range(i + 1, len(kat_h))]
            w1_st = None
            if gorev == "dispute":
                st_h = []
                for ii in STAB:
                    ws, _ = group_lasso(Xtr[ii], yy[tr_all][ii].astype(np.float32), lam, 3)
                    st_h.append(hist_derinlik([k_aday[t] for t in top_destek(ws, 3, D)], L))
                w1_st = [w1(st_h[i], st_h[j]) for i in range(len(st_h))
                         for j in range(i + 1, len(st_h))]
            R["keskiler"][kk]["gorevler"][gorev] = dict(
                kollar=kol, kol_i_katman=l_i, kol_ii_katmanlar=sorted(l_iii),
                lam=round(lam, 6), dev_nz=nz, aday_grup=ad_n, elenen_oran=round(float(elenen), 4),
                secilen=[[int(l), int(j)] for l, j in k_iii],
                hist_oneleme=[round(x, 4) for x in h_on],
                hist_lasso=[round(x, 4) for x in h_la],
                hist_w1_oneleme_lasso=round(w1(h_on, h_la), 4),
                w1_katlar=dict(ort=round(float(np.mean(w1_kat)), 4),
                               min=round(float(np.min(w1_kat)), 4),
                               maks=round(float(np.max(w1_kat)), 4), n=len(w1_kat)),
                w1_stabilite=(dict(ort=round(float(np.mean(w1_st)), 4),
                                   min=round(float(np.min(w1_st)), 4),
                                   maks=round(float(np.max(w1_st)), 4), n=len(w1_st),
                                   _not="B=10 SPREAD, CI DEGIL (protokol §5.3)")
                              if w1_st else None),
                sure_dk=round((time.time() - tg) / 60, 2))
            z = R["keskiler"][kk]["gorevler"][gorev]
            print(f"   AUC (i) {kol['i']['auc']:.4f}{kol['i']['ci']} · (ii) {kol['ii']['auc']:.4f}"
                  f"{kol['ii']['ci']} · (iii) {kol['iii']['auc']:.4f}{kol['iii']['ci']} · "
                  f"rastgele-D {kol['rastgele_D']['auc']:.4f} · W1(ön-eleme,lasso) "
                  f"{z['hist_w1_oneleme_lasso']:.4f} · W1 katlar {z['w1_katlar']['ort']:.4f} · "
                  f"[SIZINTILI (iii) {kol['iii_sizintili']['auc']:.4f}] · "
                  f"{z['sure_dk']:.1f} dk", flush=True)
            del Xa
        gd = R["keskiler"][kk]["gorevler"]
        Aset = {tuple(x) for x in gd["dispute"]["secilen"]}
        Bset = {tuple(x) for x in gd["rol"]["secilen"]}
        kes = len(Aset & Bset); mn = min(len(Aset), len(Bset))
        jac = kes / max(len(Aset | Bset), 1)
        B_kat = {}
        for l, j in Bset:
            B_kat.setdefault(l, 0); B_kat[l] += 1
        nulls = []
        for _ in range(NPERM):
            Bn = set()
            for l, k in B_kat.items():
                Bn |= {(l, int(j)) for j in rng.choice(D, k, replace=False)}
            nulls.append(len(Aset & Bn) / max(mn, 1))
        nu = np.array(nulls); obs = kes / max(mn, 1)
        R["keskiler"][kk]["ortusme"] = dict(
            n_dispute=len(Aset), n_rol=len(Bset), kesisim=kes,
            oran_min=round(obs, 4), jaccard=round(jac, 4),
            null_ort=round(float(nu.mean()), 4), null_p95=round(float(np.percentile(nu, 95)), 4),
            null_ustu=bool(obs > np.percentile(nu, 95)), yuzde50_alti=bool(obs < 0.50),
            _null="")
        o = R["keskiler"][kk]["ortusme"]
        print(f"[{kk}/örtüsme] |A|={len(Aset)} |B|={len(Bset)} ∩={kes} · oran/min "
              f"{obs:.4f} · null ort {nu.mean():.4f} p95 {np.percentile(nu,95):.4f} · "
              f"null-üstü {o['null_ustu']} · <%50 {o['yuzde50_alti']}", flush=True)
        del P, C
        json.dump(R, open(MAN, "w"), ensure_ascii=False, indent=1)
    json.dump(R, open(MAN, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {MAN} · toplam {(time.time()-t0)/60:.1f} dk", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--kos", action="store_true")
    ap.add_argument("--keski", default="K1,K2,K3,K4,K5")
    a = ap.parse_args()
    if a.gate:
        kapi()
    elif a.kos:
        kos([k for k in a.keski.split(",") if k])
    else:
        ap.error("--gate ya da --kos")
