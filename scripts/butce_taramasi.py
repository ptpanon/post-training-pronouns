import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from kt_kodlama import KapsamHatasi
import secici_pilot as SP

MAN = __DNH_ROOT__ + "/unreleased/butce_taramasi_2026-07-30.json"
D_IZGARA = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
HIST_D = [8, 64, 512]
HAVUZ_N = 8192
LAM_N = 22
NFOLD, NBOOT, NPERM = 5, 1000, 200
SEED = 20260731


def r_sema(D):
    return 10 if D <= 128 else 5


def auc_sutun(y, S):
    y = y.astype(np.float64)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[1], np.nan)
    o = np.argsort(S, axis=0, kind="stable")
    r = np.empty(o.shape, np.float64)
    np.put_along_axis(r, o, np.broadcast_to(
        np.arange(1, len(y) + 1, dtype=np.float64)[:, None], S.shape), axis=0)
    return ((r * y[:, None]).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def yol_destekler(X, y01, hedefler, G=3):
    n = len(X)
    r0 = (y01.mean() - y01) / n
    gm = np.linalg.norm((X.T @ (-r0)).reshape(-1, G), axis=1)
    lmax = float(gm.max()) * 1.001
    if lmax < 1e-6 * max(1.0, float(np.abs(X).mean())):
        raise KapsamHatasi("")
    lams = lmax * np.geomspace(1.0, 0.01, LAM_N)
    lip = SP._lip(X)
    w, b, bulunan, maxnz = None, 0.0, {}, 0
    kalan = sorted(hedefler)
    for lam in lams:
        w, b = SP.group_lasso(X, y01, lam, G, w0=w, b0=b, lip=lip)
        nz = int((SP.destek(w, G) > 1e-8).sum()); maxnz = max(maxnz, nz)
        for D in list(kalan):
            if nz >= D:
                bulunan[D] = (SP.top_destek(w, G, D), float(lam), nz); kalan.remove(D)
        if not kalan:
            break
    return bulunan, maxnz, float(lmax)


def _sut(idx):
    return (3 * np.asarray(idx, int)[:, None] + np.arange(3)).ravel()


def _std(X):
    return ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)


def dilim(P, C, rows, flip=None):
    A, B = P[rows].copy(), C[rows].copy()
    if flip is not None:
        f = flip[rows]
        A[f], B[f] = C[rows][f], P[rows][f]
    return A, B


def kapi(kk="K2"):
    print("═" * 88)
    print("")
    SP.kapi()

    print("\nKAPI YÜZ-1b — auc_sutun ↔ roc_auc_score özdesligi (kümeli bootstrap örneginde)")
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.3).astype(int); s = rng.random(600) + 0.4 * y
    ii = rng.integers(0, 600, 900)
    a1 = float(auc_sutun(y[ii], s[ii][:, None])[0]); a2 = roc_auc_score(y[ii], s[ii])
    print(f"  auc_sutun {a1:.10f} vs roc_auc_score {a2:.10f} · |Δ| {abs(a1-a2):.2e}")
    if abs(a1 - a2) > 1e-9:
        raise SystemExit("")

    print(f"\nKAPI YÜZ-2 — ISLEYIS NOKTASINDA MALIYET (DEBTS B11(d)); keski {kk}")
    y_d, g, bol = SP.veri()
    tr_all = np.where(bol == "train")[0]; dev = np.where(bol == "dev")[0]
    n_bek = {s_: int((bol == s_).sum()) for s_ in SP.BOLMELER}
    K = SP.KESKILER[kk]; L, D = K["L"], K["D"]
    t = time.time(); P, C = SP.yukle(K, n_bek); t_yukle = time.time() - t
    t = time.time(); S = SP.oneleme_skoru(P, C, y_d, dev); t_on = time.time() - t
    duz = S.ravel(); sel = np.argpartition(-duz, HAVUZ_N - 1)[:HAVUZ_N]
    k_aday = SP.koord_sirali({l: sorted(sel[sel // D == l] % D) for l in np.unique(sel // D)})
    t = time.time(); Xa = SP.ozn(P[tr_all], C[tr_all], k_aday); t_ozn = time.time() - t
    Xa = ((Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)).astype(np.float32)
    katlar = list(GroupKFold(NFOLD).split(np.zeros(len(tr_all)), y_d[tr_all], g[tr_all]))
    tr, te = katlar[0]
    t = time.time(); SP.group_lasso(Xa[tr], y_d[tr_all][tr].astype(np.float32), 0.01, 3)
    t_fista = time.time() - t
    t = time.time(); SP.duz_probe(Xa[:, _sut(np.arange(1024))], y_d[tr_all], tr, te)
    t_log1024 = time.time() - t
    t = time.time(); XL = SP.ozn(P[tr_all], C[tr_all], SP.koord_sirali({0: list(range(D))}))
    t_ozn_kat = time.time() - t
    t = time.time(); SP.duz_probe(XL, y_d[tr_all], tr, te); t_log_kat = time.time() - t
    del XL
    n_cekim = sum(r_sema(d) for d in D_IZGARA)
    hucre = (t_ozn + NFOLD * (LAM_N * t_fista)
             + NFOLD * len(D_IZGARA) * t_log1024
             + NFOLD * n_cekim * t_log1024
             + NFOLD * n_cekim * (t_log1024 + t_ozn * 1024 / HAVUZ_N)
             + L * (t_ozn_kat + NFOLD * t_log_kat)
             + 3 * t_ozn_kat + NFOLD * t_log_kat)
    print(f"  yükle {t_yukle:.1f}s · ön-eleme {t_on:.1f}s · ozn(havuz 8192) {t_ozn:.1f}s · "
          f"FISTA(1 λ) {t_fista:.2f}s · lojistik(D=1024) {t_log1024:.2f}s · "
          f"ozn(1 katman) {t_ozn_kat:.1f}s · lojistik(1 katman) {t_log_kat:.2f}s")
    print(f"  → hücre basina ≈ {hucre/60:.1f} dk ⇒ 5 keski × 2 görev ≈ "
          f"{(2*hucre + t_yukle + t_on)*5/3600:.2f} SAAT (projeksiyon; fiili süre rapora yazilir)")

    print("\nKAPI YÜZ-3 — ISLEYIS NOKTASINDA AYIRT EDILEBILIRLIK (I17)")
    uf = np.unique(g[tr_all]); fidx = {f: np.where(g[tr_all] == f)[0] for f in uf}
    rr = np.random.default_rng(1)
    BS = [np.concatenate([fidx[f] for f in rr.choice(uf, len(uf), True)]) for _ in range(200)]
    for Dt in (16, 1024):
        bul, maxnz, _ = yol_destekler(Xa[tr], y_d[tr_all][tr].astype(np.float32), [Dt])
        if Dt not in bul:
            print(f"{Dt} {maxnz}")
            continue
        idl = bul[Dt][0]
        idr = rr.choice(HAVUZ_N, Dt, replace=False)
        ortak = len(set(idl.tolist()) & set(idr.tolist()))
        ool, oor = np.full(len(tr_all), np.nan), np.full(len(tr_all), np.nan)
        for tr2, te2 in katlar:
            bl, _, _ = yol_destekler(Xa[tr2], y_d[tr_all][tr2].astype(np.float32), [Dt])
            if Dt not in bl:
                break
            ool[te2] = SP.duz_probe(Xa[:, _sut(bl[Dt][0])], y_d[tr_all], tr2, te2)
            oor[te2] = SP.duz_probe(Xa[:, _sut(idr)], y_d[tr_all], tr2, te2)
        SCm = np.column_stack([ool, oor])
        ds = [np.diff(auc_sutun(y_d[tr_all][ii], SCm[ii]))[0] * -1 for ii in BS]
        print(f"  D={Dt}: lasso∩rastgele {ortak}/{Dt} grup (özdes DEGIL) · "
              f"Δ {np.mean(ds):+.4f} · esli bootstrap sd {np.std(ds):.4f} "
              f"({'>0 ⇒ ayirt edilebilir' if np.std(ds) > 1e-6 else 'sd=0 ⇒ MEKANIK, DUR'})")
        if np.std(ds) <= 1e-6:
            raise SystemExit("[kapi] KALDI: esli sd = 0 — tasarim bu veride ayirt edemiyor")
    print("\n[kapi] ÜC YÜZ GECTI.", flush=True)


def kos(keskiler):
    t0 = time.time()
    y_d, g, bol = SP.veri()
    tr_all = np.where(bol == "train")[0]; dev = np.where(bol == "dev")[0]
    n_bek = {s_: int((bol == s_).sum()) for s_ in SP.BOLMELER}
    yy_tr_kayit = {}
    print(f"[kapsam] train {len(tr_all)} ({len(np.unique(g[tr_all]))} forum) · dev {len(dev)} "
          f"({len(np.unique(g[dev]))} forum) · TEST KAPALI · havuz {SP.HAVUZ} · "
          f"D-izgara {D_IZGARA} · aday havuzu {HAVUZ_N}", flush=True)
    R = {"_statu": "KESIF — verdict YOK. Esik yok, karar kurali yok. 'Devre bulundu' YASAK.",
         "_protokol": "PROTOKOL_butce_taramasi_2026-07-30.md @ 30e62f7",
         "_beklentiler": ""
                         ""
                         "",
         "_kapsam": dict(train=len(tr_all), dev=len(dev), havuz=SP.HAVUZ, test="KAPALI",
                         D_izgara=D_IZGARA, hist_D=HIST_D, aday_havuz=HAVUZ_N,
                         nboot=NBOOT, nperm=NPERM, lam_n=LAM_N,
                         R_sema={str(d): r_sema(d) for d in D_IZGARA}),
         "keskiler": {}}
    rng = np.random.default_rng(SEED)
    katlar = list(GroupKFold(NFOLD).split(np.zeros(len(tr_all)), y_d[tr_all], g[tr_all]))
    uf = np.unique(g[tr_all]); fidx = {f: np.where(g[tr_all] == f)[0] for f in uf}
    BS = [np.concatenate([fidx[f] for f in rng.choice(uf, len(uf), True)]) for _ in range(NBOOT)]

    for kk in keskiler:
        K = SP.KESKILER[kk]; L, D = K["L"], K["D"]
        P, C = SP.yukle(K, n_bek)
        Rk = dict(model=K["model"], L=L, D=D, gorevler={})
        R["keskiler"][kk] = Rk
        elit = {}
        flip = np.random.default_rng(SEED + 7).random(len(P)) < 0.5
        for gorev in ("dispute", "rol"):
            tg = time.time()
            fl = None if gorev == "dispute" else flip
            yy = y_d if gorev == "dispute" else flip.astype(int)
            ytr = yy[tr_all]; yy_tr_kayit[gorev] = ytr
            Adev, Bdev = dilim(P, C, dev, fl)
            S = SP.oneleme_skoru(Adev, Bdev, yy[dev], np.arange(len(dev)))
            del Adev, Bdev
            duz = S.ravel(); sel = np.argpartition(-duz, HAVUZ_N - 1)[:HAVUZ_N]
            k_aday = SP.koord_sirali({l: sorted(sel[sel // D == l] % D)
                                      for l in np.unique(sel // D)})
            elenen = 1 - HAVUZ_N / duz.size
            Atr, Btr = dilim(P, C, tr_all, fl)
            Xa = _std(SP.ozn(Atr, Btr, k_aday))

            SUT, ETI = [], []
            lasso = {}; kat_destek = {}
            ulasilamayan = []
            maxnz_kat = []
            for d in D_IZGARA:
                lasso[d] = np.full(len(tr_all), np.nan); kat_destek[d] = []
            for tr, te in katlar:
                bul, maxnz, _ = yol_destekler(Xa[tr], ytr[tr].astype(np.float32), D_IZGARA)
                maxnz_kat.append(maxnz)
                for d in D_IZGARA:
                    if d in bul:
                        idx = bul[d][0]; kat_destek[d].append(idx)
                        lasso[d][te] = SP.duz_probe(Xa[:, _sut(idx)], ytr, tr, te)
            for d in D_IZGARA:
                if len(kat_destek[d]) < NFOLD:
                    ulasilamayan.append(d)
            bul_t, maxnz_t, lmax_t = yol_destekler(Xa, ytr.astype(np.float32), D_IZGARA)
            elit[gorev] = {d: [k_aday[t] for t in bul_t[d][0]] for d in bul_t}

            rast = {"rastgele_havuz": {}, "rastgele_tam": {}}
            for d in D_IZGARA:
                if d in ulasilamayan:
                    continue
                Rn = r_sema(d)
                for kaynak in ("rastgele_havuz", "rastgele_tam"):
                    oofs = []
                    for _ in range(Rn):
                        if kaynak == "rastgele_havuz":
                            Xr = Xa[:, _sut(rng.choice(HAVUZ_N, d, replace=False))]
                        else:
                            rk = {}
                            for l, j in zip(rng.integers(0, L, d), rng.integers(0, D, d)):
                                rk.setdefault(int(l), set()).add(int(j))
                            kd = SP.koord_sirali({l: sorted(v) for l, v in rk.items()})
                            Xr = _std(SP.ozn(Atr, Btr, kd))
                        oof = np.full(len(tr_all), np.nan)
                        for tr, te in katlar:
                            oof[te] = SP.duz_probe(Xr, ytr, tr, te)
                        oofs.append(oof); del Xr
                    rast[kaynak][d] = oofs
                SUT.append(lasso[d]); ETI.append(("lasso", d, 0))
                for kaynak in ("rastgele_havuz", "rastgele_tam"):
                    for r_, oo in enumerate(rast[kaynak][d]):
                        SUT.append(oo); ETI.append((kaynak, d, r_))
                print(f"[{kk}/{gorev}] D={d:5d} bitti ({(time.time()-tg)/60:.1f} dk)", flush=True)

            kat_oof = []
            for l in range(L):
                Xl = _std(SP.ozn(Atr, Btr, SP.koord_sirali({l: list(range(D))})))
                oo = np.full(len(tr_all), np.nan)
                for tr, te in katlar:
                    oo[te] = SP.duz_probe(Xl, ytr, tr, te)
                kat_oof.append(oo); del Xl
            kat_auc = np.array([roc_auc_score(ytr, o) for o in kat_oof])
            uc = np.argsort(-kat_auc)[:3].tolist()
            Xu = _std(SP.ozn(Atr, Btr, SP.koord_sirali({l: list(range(D)) for l in uc})))
            oof_uc = np.full(len(tr_all), np.nan)
            for tr, te in katlar:
                oof_uc[te] = SP.duz_probe(Xu, ytr, tr, te)
            del Xu
            print(f"[{kk}/{gorev}] referans: post-hoc en iyi katman ℓ={int(kat_auc.argmax())} "
                  f"AUC {kat_auc.max():.4f} (bütce {D}) · post-hoc üclü {sorted(uc)} "
                  f"AUC {roc_auc_score(ytr, oof_uc):.4f} (bütce {3*D})", flush=True)

            SCk = np.column_stack(kat_oof)
            SCa = np.column_stack(SUT + [oof_uc])
            boot = {e: [] for e in ETI}; boot_uc = []; boot_maxkat = []; boot_argkat = []
            for ii in BS:
                yb = ytr[ii]
                if len(set(yb.tolist())) < 2:
                    continue
                aa = auc_sutun(yb, SCa[ii])
                for t_, e in enumerate(ETI):
                    boot[e].append(aa[t_])
                boot_uc.append(aa[-1])
                ak = auc_sutun(yb, SCk[ii])
                boot_maxkat.append(np.nanmax(ak)); boot_argkat.append(int(np.nanargmax(ak)))
            nb = len(boot_uc)

            def ozet(vals):
                v = np.asarray(vals, float)
                return [round(float(np.percentile(v, 2.5)), 4),
                        round(float(np.percentile(v, 97.5)), 4)]

            kollar = {"lasso": [], "rastgele_havuz": [], "rastgele_tam": []}
            delta = []; delta_tam = []
            for d in D_IZGARA:
                if d in ulasilamayan:
                    for kn in kollar:
                        kollar[kn].append(dict(D=d, auc=None, ci=[None, None],
                                               not_="yol ULASMADI"))
                    delta.append(dict(D=d, fark=None, ci=[None, None], sd=None))
                    delta_tam.append(dict(D=d, fark=None, ci=[None, None], sd=None))
                    continue
                bl = np.array(boot[("lasso", d, 0)], float)
                kollar["lasso"].append(dict(D=d, auc=round(float(roc_auc_score(ytr, lasso[d])), 4),
                                            ci=ozet(bl), R=1))
                bh = {}
                for kaynak in ("rastgele_havuz", "rastgele_tam"):
                    Rn = r_sema(d)
                    M = np.array([boot[(kaynak, d, r_)] for r_ in range(Rn)], float)
                    bh[kaynak] = M.mean(0)
                    aucs = [roc_auc_score(ytr, o) for o in rast[kaynak][d]]
                    kollar[kaynak].append(dict(
                        D=d, auc=round(float(np.mean(aucs)), 4), ci=ozet(bh[kaynak]), R=Rn,
                        cekim_min=round(float(np.min(aucs)), 4),
                        cekim_maks=round(float(np.max(aucs)), 4)))
                for hedef, akm in (("rastgele_havuz", delta), ("rastgele_tam", delta_tam)):
                    dd = bl - bh[hedef]
                    akm.append(dict(D=d, fark=round(float(np.mean(dd)), 4), ci=ozet(dd),
                                    sd=round(float(np.std(dd)), 5)))
            ref = dict(
                tek_katman_posthoc=dict(
                    auc=round(float(kat_auc.max()), 4), katman=int(kat_auc.argmax()),
                    goreli=round(float(kat_auc.argmax() / (L - 1)), 3), n_grup=D,
                    ci_secim_icsel=ozet(boot_maxkat),
                    arg_dagilim_n=len(set(boot_argkat)),
                    _not=""),
                uc_katman_posthoc=dict(
                    auc=round(float(roc_auc_score(ytr, oof_uc)), 4), katmanlar=sorted(uc),
                    n_grup=3 * D, ci=ozet(boot_uc),
                    _not="post-hoc MARJINAL üclü — C(L,3) taranmadi"),
                katman_auc=[round(float(x), 4) for x in kat_auc])
            hist = {}
            for dh in HIST_D:
                if dh not in bul_t:
                    hist[str(dh)] = None; continue
                h_on = SP.hist_derinlik(k_aday, L)
                h_la = SP.hist_derinlik(elit[gorev][dh], L)
                kh = [SP.hist_derinlik([k_aday[t] for t in idf], L) for idf in kat_destek[dh]]
                w1k = [SP.w1(kh[i], kh[j]) for i in range(len(kh)) for j in range(i + 1, len(kh))]
                hist[str(dh)] = dict(oneleme=[round(x, 4) for x in h_on],
                                     lasso=[round(x, 4) for x in h_la],
                                     w1=round(SP.w1(h_on, h_la), 4),
                                     w1_katlar_ort=round(float(np.mean(w1k)), 4) if w1k else None,
                                     w1_katlar_maks=round(float(np.max(w1k)), 4) if w1k else None)
            Rk["gorevler"][gorev] = dict(
                D_izgara=D_IZGARA, kollar=kollar, delta=delta, delta_tam=delta_tam,
                referans=ref, hist=hist, elenen_oran=round(float(elenen), 4),
                aday_grup=HAVUZ_N, ulasilamayan_D=ulasilamayan,
                yol_maxnz_katlar=maxnz_kat, yol_maxnz_train=maxnz_t,
                boot_gecerli=nb, sure_dk=round((time.time() - tg) / 60, 2),
                _kapsam=f"kol {len(ETI)} sütun · {NFOLD} kat · {nb}/{NBOOT} gecerli bootstrap · "
                        f"ulasilamayan D {len(ulasilamayan)}")
            if nb == 0:
                raise KapsamHatasi("0 gecerli bootstrap cekimi — payda SIFIR, gecis degil HATA")
            print(f"[{kk}/{gorev}] bitti {(time.time()-tg)/60:.1f} dk · gecerli bootstrap "
                  f"{nb}/{NBOOT} · ulasilamayan D {ulasilamayan}", flush=True)
            del Xa, SUT, SCa, SCk, kat_oof, Atr, Btr, rast, lasso
        oz = dict(D_izgara=[], oran_min=[], jaccard=[], null_ort=[], null_p05=[], null_p95=[],
                  n_A=[], n_B=[])
        for d in D_IZGARA:
            if d not in elit.get("dispute", {}) or d not in elit.get("rol", {}):
                for k_ in oz:
                    oz[k_].append(None)
                continue
            Aset = {tuple(x) for x in elit["dispute"][d]}
            Bset = {tuple(x) for x in elit["rol"][d]}
            kes = len(Aset & Bset); mn = max(min(len(Aset), len(Bset)), 1)
            Bkat = {}
            for l, j in Bset:
                Bkat[l] = Bkat.get(l, 0) + 1
            nu = []
            for _ in range(NPERM):
                Bn = set()
                for l, kn_ in Bkat.items():
                    Bn |= {(l, int(j)) for j in rng.choice(D, kn_, replace=False)}
                nu.append(len(Aset & Bn) / mn)
            nu = np.array(nu)
            oz["D_izgara"].append(d); oz["oran_min"].append(round(kes / mn, 4))
            oz["jaccard"].append(round(kes / max(len(Aset | Bset), 1), 4))
            oz["null_ort"].append(round(float(nu.mean()), 4))
            oz["null_p05"].append(round(float(np.percentile(nu, 5)), 4))
            oz["null_p95"].append(round(float(np.percentile(nu, 95)), 4))
            oz["n_A"].append(len(Aset)); oz["n_B"].append(len(Bset))
        oz["_null"] = (""
                       "")
        Rk["ortusme"] = oz
        print(f"[{kk}/örtüsme] oran/min {oz['oran_min']} · null ort {oz['null_ort']}", flush=True)
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
