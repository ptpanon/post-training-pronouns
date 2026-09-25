#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import glob
import argparse
import collections

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from prefix_muhurlu_kosu import auc_np
import prefix_okunmazlik as OK
import prefix_ucuncu_eksen as UE

KOK = __DNH_DATA__ + "/onek_korpus"
IS = f"{KOK}/vekil"
VERI = f"{IS}/veri.json"
GOMU = f"{IS}/e5_l21.npy"
CIKTI = f"{ROOT}/unreleased/HARSH_VEKIL_2026-08-09.json"

BAR_AUC = 0.850
BAR_ECE = 0.10
BAR_DILIM = 0.20
N_FOLD, SEED = 5, 20260809
ADLAR = ("VEKIL-GECTI", "VEKIL-RAFA", "ÖLCÜLEMEZ")

KOSULLAR = [
    ("merdiven",   f"{KOK}/merdiven_kor/*.json",      f"{KOK}/merdiven_yargi",
     f"{KOK}/merdiven_esleme/esleme_merdiven_2026-08-07.json", None, None),
    ("esli",       f"{KOK}/esli_kor/*.json",          f"{KOK}/esli_yargi",
     f"{KOK}/esli_esleme/esleme_esli_2026-08-08.json", None, None),
    ("dumen",      f"{KOK}/dumen_kor/*.json",         f"{KOK}/dumen_yargi",
     f"{KOK}/dumen_esleme/esleme_dumen_2026-08-08.json", "mistral", "it"),
    ("tamir",      f"{KOK}/tamir_kor/dilim/*.json",   f"{KOK}/tamir_yargi",
     f"{KOK}/tamir_esleme/esleme.json", "mistral", "it"),
    ("taraf_ton",  f"{KOK}/taraf_ton_kor/dilim/*.json", f"{KOK}/taraf_ton_yargi",
     f"{KOK}/taraf_ton_esleme/esleme.json", "mistral", "it"),
    ("taraf2_ton", f"{KOK}/taraf2_ton_kor/dilim/*.json", f"{KOK}/taraf2_ton_yargi",
     f"{KOK}/taraf2_ton_esleme/esleme.json", "mistral", "it"),
    ("t1_rol",     f"{KOK}/t1_kor/dilim/*.json",      f"{KOK}/t1_yargi",
     f"{KOK}/t1_esleme/esleme.json", "mistral", "it"),
    ("ayirici",    f"{KOK}/ayirici_kor/dilim/*.json", f"{KOK}/ayirici_yargi",
     f"{KOK}/ayirici_esleme/esleme.json", "mistral", "it"),
    ("ton",        f"{KOK}/ton_kor/*.json",           f"{KOK}/ton_yargi",
     f"{KOK}/ton_esleme/esleme_ton_2026-08-08.json", "mistral", "it"),
    ("st",         f"{KOK}/st_kor/*.json",            f"{KOK}/st_yargi",
     f"{KOK}/st_esleme/esleme_st_2026-08-08.json", "mistral", "it"),
    ("aile2",      f"{KOK}/aile2_kor/dilim/*.json",   f"{KOK}/aile2_yargi",
     f"{KOK}/aile2_esleme/esleme.json", "qwen", "it"),
    ("tavan",      f"{KOK}/tavan_kor/dilim/*.json",   f"{KOK}/tavan_yargi",
     f"{KOK}/tavan_esleme/esleme.json", "qwen", "it"),
]


def verdict_vekil(auc_oof, ece, en_kotu_dilim, auc_lofo_min, n, n_pozitif):
    if (not all(np.isfinite(v) for v in (auc_oof, ece, en_kotu_dilim, auc_lofo_min))
            or n < 200 or n_pozitif < 50):
        return "ÖLCÜLEMEZ"
    if (auc_oof >= BAR_AUC and ece <= BAR_ECE and en_kotu_dilim <= BAR_DILIM
            and auc_lofo_min >= BAR_AUC):
        return "VEKIL-GECTI"
    return "VEKIL-RAFA"


def _prova():
    V = [("NaN", (float("nan"), .05, .1, .9, 2000, 500), "ÖLCÜLEMEZ"),
         ("n kücük", (0.95, .05, .1, .9, 100, 50), "ÖLCÜLEMEZ"),
         ("pozitif az", (0.95, .05, .1, .9, 2000, 10), "ÖLCÜLEMEZ"),
         ("AUC düsük", (0.80, .05, .1, .9, 2000, 500), "VEKIL-RAFA"),
         ("kalibrasyon kötü", (0.95, .20, .1, .9, 2000, 500), "VEKIL-RAFA"),
         ("dilim kötü", (0.95, .05, .40, .9, 2000, 500), "VEKIL-RAFA"),
         ("", (0.95, .05, .1, 0.62, 2000, 500), "VEKIL-RAFA"),
         ("hepsi gecti", (0.90, .05, .1, .88, 2000, 500), "VEKIL-GECTI")]
    ok = 0
    for ad, arg, bek in V:
        h = verdict_vekil(*arg)
        ok += h == bek
        print(f"  prova {ad:24s} → {h:14s} (bekle {bek})")
    assert ok == len(V), "prova DÜSTÜ"
    assert set(h for _, a, h in V) == set(ADLAR), "D44: ad kümesi eslesmiyor"
    print(f"  ✓ prova {ok}/{len(V)} · D44 ad kümesi {len(ADLAR)}={len(ADLAR)} esit")


def _aile_terbiye(meta, aile, terbiye):
    kol = meta.get("merdiven_kol")
    if kol:
        a = kol.split("_")[0]
        t = "base" if "base" in kol else "it"
        return a, t
    u = meta.get("uretici")
    if u and aile is None:
        return str(u), terbiye or "?"
    return aile or "?", terbiye or "?"


def hasat():
    os.makedirs(IS, exist_ok=True)
    satir, sayac = [], collections.Counter()
    for ad, kor_g, ydiz, eyol, aile, terbiye in KOSULLAR:
        metin = {}
        for f in sorted(glob.glob(kor_g)):
            D = json.load(open(f, encoding="utf-8"))
            for r in (D if isinstance(D, list) else [D]):
                if isinstance(r, dict) and "id" in r and "text" in r:
                    metin[r["id"]] = r["text"]
        etiket, guven = {}, {}
        for f in sorted(glob.glob(f"{ydiz}/*.json")):
            for y in json.load(open(f, encoding="utf-8")):
                if "tone" in y:
                    etiket[y["id"]] = y["tone"]
                    guven[y["id"]] = float(y.get("confidence") or 0.0)
        E = json.load(open(eyol, encoding="utf-8"))
        n_e, n_m, n_y = 0, 0, 0
        for pid, et in etiket.items():
            n_y += 1
            if pid not in metin:
                sayac[f"red_metinsiz_{ad}"] += 1
                continue
            n_m += 1
            meta = E.get(pid, {})
            if not meta:
                sayac[f"red_eslemesiz_{ad}"] += 1
                continue
            n_e += 1
            kume = str(meta.get("debate") or meta.get("tez") or "?").strip().lower()
            a, t = _aile_terbiye(meta, aile, terbiye)
            satir.append(dict(pid=pid, kosul=ad, metin=metin[pid], ton=et,
                              guven=guven.get(pid, 0.0), kume=kume, aile=a, terbiye=t,
                              kol=str(meta.get("kol") or meta.get("basamak") or "?")))
            sayac[f"ton_{et}"] += 1
        payda(f"hasat_{ad}", n_yargi=n_y, n_metin_esli=n_m, n_esleme_esli=n_e,
              red_metinsiz=sayac[f"red_metinsiz_{ad}"],
              red_eslemesiz=sayac[f"red_eslemesiz_{ad}"], bekle={"n_yargi": 1})
    json.dump(satir, open(VERI, "w", encoding="utf-8"), ensure_ascii=False)
    payda("hasat_toplam", n_satir=len(satir), n_kosul=len(KOSULLAR),
          n_kume=len(set(s["kume"] for s in satir)),
          n_aile=len(set(s["aile"] for s in satir)),
          bekle={"n_satir": 2000, "n_kosul": len(KOSULLAR)})
    print("  ton dagilimi:", {k[4:]: v for k, v in sorted(sayac.items()) if k.startswith("ton_")})
    print("  aile:", dict(collections.Counter(s["aile"] for s in satir)))
    print("  terbiye:", dict(collections.Counter(s["terbiye"] for s in satir)))
    print(f"→ {VERI}")


def gomu(dev):
    S = json.load(open(VERI, encoding="utf-8"))
    if os.path.exists(GOMU):
        H = np.load(GOMU)
        if len(H) == len(S):
            print(f"  ↷ ATLA {GOMU} ({H.shape}) — veriyle uyumlu")
            return
        print(f"{len(H)} {len(S)}")
    from gpu_lock import kilitle
    kilitle(dev, tam=True, etiket="harsh_vekil_e5")
    import torch
    import anchor_kodlama as CK
    from transformers import AutoTokenizer, AutoModel
    CK.DEV = dev
    K = CK.KOLLAR["c1a"]
    tok = AutoTokenizer.from_pretrained(K["model"])
    model = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(dev).eval()
    eski = (K["kes"], K["maxlen"])
    K["kes"], K["maxlen"] = 4000, 512
    sayac = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0, uzunluklar=[])
    try:
        H = CK.kodla(tok, model, [s["metin"] for s in S], "c1a", bs=32, sayac=sayac)
    finally:
        K["kes"], K["maxlen"] = eski
    A = np.asarray(H[:, 21, :], dtype=np.float32)
    np.save(GOMU + ".tmp.npy", A)
    os.replace(GOMU + ".tmp.npy", GOMU)
    payda("gomu_e5_l21", n_satir=int(sayac["satir"]), n_boyut=int(A.shape[1]),
          red_bos_havuz=int(sayac["bos_havuz"]), bekle={"n_satir": len(S)})
    print(f"→ {GOMU} {A.shape}")


OZNITELIK = ("hedge", "dom_ort", "dom_p90", "log_n", "ttr", "okunmazlik",
             "tekrar", "e5_l21_izdusum")


def yuzey(metinler):
    soz, _frek, hyol = UE.hedge_sozlugu()
    D, dyol = UE.dominance()
    hed, dom, isabet = UE.olcum(metinler, soz, D)
    dp90, logn, ttr, oku, tek = [], [], [], [], []
    for t in metinler:
        j = UE.KELIME.findall((t or "").lower())
        d = [D[w] for w in j if w in D]
        dp90.append(float(np.percentile(d, 90)) if len(d) >= 3 else np.nan)
        logn.append(float(np.log1p(len(j))))
        ttr.append(len(set(j)) / max(len(j), 1))
        oku.append(float(OK.okunmazlik(t)))
        tek.append(float(OK.tekrar_orani(t)))
    payda("yuzey", n_metin=len(metinler), n_hedge_isabet=int(isabet),
          n_sozluk=len(soz), n_warriner=len(D), bekle={"n_metin": len(metinler)})
    return dict(hedge=np.array(hed), dom_ort=np.array(dom), dom_p90=np.array(dp90),
                log_n=np.array(logn), ttr=np.array(ttr), okunmazlik=np.array(oku),
                tekrar=np.array(tek)), dict(hedge=hyol, warriner=dyol)


def _izdusum_fold(H, y, egit):
    m = H[egit].mean(0)
    Hc = H - m
    a, b = Hc[egit][y[egit] == 1], Hc[egit][y[egit] == 0]
    if len(a) < 3 or len(b) < 3:
        return np.zeros(len(H))
    d = a.mean(0) - b.mean(0)
    n = np.linalg.norm(d)
    return (Hc @ (d / n)) if n > 0 else np.zeros(len(H))


def _foldlar(kume, k, rng):
    u = sorted(set(kume.tolist()))
    rng.shuffle(u)
    at = {c: i % k for i, c in enumerate(u)}
    return np.array([at[c] for c in kume])


def _ece(y, p, n_dilim=10):
    ix = np.argsort(p)
    parca = np.array_split(ix, n_dilim)
    e, kotu, egri = 0.0, 0.0, []
    for g in parca:
        if len(g) == 0:
            continue
        pm, ym = float(p[g].mean()), float(y[g].mean())
        e += len(g) / len(y) * abs(pm - ym)
        kotu = max(kotu, abs(pm - ym))
        egri.append(dict(n=len(g), p_ort=round(pm, 4), gozlenen=round(ym, 4)))
    return e, kotu, egri


def _lojistik(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    M = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    return M.predict_proba(sc.transform(Xte))[:, 1], M


_SOZ_ONBELLEK = {}


def _sozlukler():
    if not _SOZ_ONBELLEK:
        soz, _f, hyol = UE.hedge_sozlugu()
        D, dyol = UE.dominance()
        _SOZ_ONBELLEK.update(soz=soz, D=D, hyol=hyol, dyol=dyol)
    return _SOZ_ONBELLEK


def yuzey_hizli(metinler):
    C = _sozlukler()
    hed, dom, isabet = UE.olcum(metinler, C["soz"], C["D"])
    D = C["D"]
    dp90, logn, ttr, oku, tek = [], [], [], [], []
    for t in metinler:
        j = UE.KELIME.findall((t or "").lower())
        d = [D[w] for w in j if w in D]
        dp90.append(float(np.percentile(d, 90)) if len(d) >= 3 else np.nan)
        logn.append(float(np.log1p(len(j))))
        ttr.append(len(set(j)) / max(len(j), 1))
        oku.append(float(OK.okunmazlik(t)))
        tek.append(float(OK.tekrar_orani(t)))
    return dict(hedge=np.array(hed), dom_ort=np.array(dom), dom_p90=np.array(dp90),
                log_n=np.array(logn), ttr=np.array(ttr), okunmazlik=np.array(oku),
                tekrar=np.array(tek)), int(isabet)


def _duz(F, medyan=None):
    X = np.column_stack([F[k] for k in OZNITELIK[:-1]])
    med = medyan if medyan is not None else np.nanmedian(X, axis=0)
    ix = np.where(np.isnan(X))
    X[ix] = np.take(med, ix[1])
    return X, med


def vekil_kur(dislanan_aile=None):
    S = json.load(open(VERI, encoding="utf-8"))
    H = np.load(GOMU)
    ton = np.array([s["ton"] for s in S], dtype=object)
    tut = np.isin(ton, ["HARSH", "CALM"])
    if dislanan_aile is not None:
        tut &= np.array([s["aile"] != dislanan_aile for s in S])
    y = (ton[tut] == "HARSH").astype(int)
    Ht = H[tut]
    F, _isabet = yuzey_hizli([s["metin"] for s, m in zip(S, tut) if m])
    X0, med = _duz(F)
    mu = Ht.mean(0)
    a, b = (Ht - mu)[y == 1].mean(0), (Ht - mu)[y == 0].mean(0)
    d = a - b
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    X = np.column_stack([X0, (Ht - mu) @ d])
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    M = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)

    def skor(metinler, Hy):
        Fy, _ = yuzey_hizli(metinler)
        Xy0, _ = _duz(Fy, med)
        Xy = np.column_stack([Xy0, (np.asarray(Hy, float) - mu) @ d])
        return M.predict_proba(sc.transform(Xy))[:, 1]

    karne = dict(dislanan_aile=dislanan_aile, n_egitim=int(len(y)),
                 n_pozitif=int(y.sum()), taban_orani=round(float(y.mean()), 4))
    return skor, karne


def coz():
    S = json.load(open(VERI, encoding="utf-8"))
    H = np.load(GOMU)
    assert len(H) == len(S), f"gömü {len(H)} ≠ veri {len(S)}"
    ton = np.array([s["ton"] for s in S], dtype=object)
    tut = np.isin(ton, ["HARSH", "CALM"])
    y = (ton[tut] == "HARSH").astype(int)
    S2 = [s for s, m in zip(S, tut) if m]
    H2 = H[tut]
    kume = np.array([s["kume"] for s in S2], dtype=object)
    aile = np.array([s["aile"] for s in S2], dtype=object)
    guven = np.array([s["guven"] for s in S2], float)
    F, kaynak = yuzey([s["metin"] for s in S2])
    payda("coz_giris", n_satir=len(y), n_pozitif=int(y.sum()), n_kume=len(set(kume.tolist())),
          n_aile=len(set(aile.tolist())), red_dislanan=int((~tut).sum()),
          bekle={"n_satir": 1000, "n_pozitif": 100, "n_kume": 5})

    rng = np.random.default_rng(SEED)
    fold = _foldlar(kume, N_FOLD, rng)
    Xy = np.column_stack([F[k] for k in OZNITELIK[:-1]])
    Xy = np.nan_to_num(Xy, nan=float(np.nanmedian(Xy)))
    p_oof = np.zeros(len(y))
    for f in range(N_FOLD):
        te = fold == f
        pr = _izdusum_fold(H2, y, ~te)
        X = np.column_stack([Xy, pr])
        p_oof[te], _ = _lojistik(X[~te], y[~te], X[te])
    A = float(auc_np(y, p_oof))
    ece, kotu, egri = _ece(y, p_oof)

    lofo = {}
    for a in sorted(set(aile.tolist())):
        te = aile == a
        if te.sum() < 30 or y[te].sum() < 5 or (1 - y[te]).sum() < 5:
            lofo[a] = dict(n=int(te.sum()), n_pozitif=int(y[te].sum()), auc=None,
                           not_="ölcülemez — hedefte tek sinif ya da n<30")
            continue
        pr = _izdusum_fold(H2, y, ~te)
        X = np.column_stack([Xy, pr])
        pp, _ = _lojistik(X[~te], y[~te], X[te])
        lofo[a] = dict(n=int(te.sum()), n_pozitif=int(y[te].sum()),
                       auc=round(float(auc_np(y[te], pp)), 4))
    olculen = [v["auc"] for v in lofo.values() if v.get("auc") is not None]
    lofo_min = min(olculen) if olculen else float("nan")

    tau = float(np.quantile(p_oof, 1 - y.mean()))
    tek_yanli = {}
    for a in sorted(set(aile.tolist())):
        m = (aile == a) & (y == 0)
        if m.sum() < 20:
            continue
        pr = _izdusum_fold(H2, y, ~(aile == a))
        X = np.column_stack([Xy, pr])
        pp, _ = _lojistik(X[aile != a], y[aile != a], X[m])
        tek_yanli[a] = dict(n_calm=int(m.sum()), p_ort=round(float(pp.mean()), 4),
                            p_p95=round(float(np.quantile(pp, 0.95)), 4),
                            yanlis_alarm=round(float((pp >= tau).mean()), 4))

    em = guven >= 0.90
    tavan = float(auc_np(y[em], p_oof[em])) if em.sum() > 50 and 0 < y[em].sum() < em.sum() \
        else float("nan")

    tek = {}
    for k in OZNITELIK[:-1]:
        v = np.nan_to_num(F[k], nan=float(np.nanmedian(F[k])))
        tek[k] = round(float(auc_np(y, v)), 4)
    pr_tam = _izdusum_fold(H2, y, np.ones(len(y), bool))
    tek["e5_l21_izdusum_SIZINTILI"] = round(float(auc_np(y, pr_tam)), 4)

    h = verdict_vekil(A, ece, kotu, lofo_min, len(y), int(y.sum()))
    R = dict(damga_utc=__import__("datetime").datetime.now(
                 __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             sinif="VEKIL — banka isine giremez",
             n=len(y), n_pozitif=int(y.sum()), n_kume=len(set(kume.tolist())),
             n_dislanan=int((~tut).sum()),
             dislanan_dagilim=dict(collections.Counter(
                 t for t, m in zip(ton, tut) if not m)),
             oznitelikler=list(OZNITELIK), model="LogisticRegression(C=1) + StandardScaler",
             cv=f"küme-ayrik GroupKFold k={N_FOLD} (küme = tez/debate)",
             auc_oof=round(A, 4), ece=round(ece, 4), en_kotu_dilim=round(kotu, 4),
             kalibrasyon_egrisi=egri, lofo=lofo, lofo_min=lofo_min,
             calisma_noktasi_tau=round(tau, 4), tek_yanli_transfer=tek_yanli,
             tek_yanli_not=(""
                            ""),
             tavan_vekili_guven90=None if not np.isfinite(tavan) else round(tavan, 4),
             n_guven90=int(em.sum()), tek_oznitelik_auc=tek,
             barlar=dict(auc=BAR_AUC, ece=BAR_ECE, dilim=BAR_DILIM),
             kaynak_sozlukler=kaynak, verdict=h)
    json.dump(R, open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  n={len(y)} (poz {int(y.sum())}) · küme {R['n_kume']} · dislanan {R['n_dislanan']}")
    print(f"  OOF AUC {A:.4f} (bar {BAR_AUC}) · ECE {ece:.4f} (bar {BAR_ECE}) · "
          f"en kötü dilim {kotu:.4f} (bar {BAR_DILIM})")
    for a, v in lofo.items():
        print(f"    LOFO {a:14s} n={v['n']:5d} poz={v['n_pozitif']:4d} auc={v['auc']}")
    print(f"  calisma noktasi τ={tau:.4f} (taban-orani esli)")
    for a, v in tek_yanli.items():
        print(f"    TEK-YANLI {a:10s} n_calm={v['n_calm']:5d} p_ort={v['p_ort']:.4f} "
              f"p95={v['p_p95']:.4f} yanlis-alarm={v['yanlis_alarm']:.4f}")
    print(f"  tavan-vekili (güven≥0,90, n={int(em.sum())}): {R['tavan_vekili_guven90']}")
    print("  tek-öznitelik AUC:", tek)
    print(f"\n  ★★ VERDICT: **{h}**\n→ {CIKTI}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasat", action="store_true")
    ap.add_argument("--gomu", action="store_true")
    ap.add_argument("--coz", action="store_true")
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    _prova()
    if a.hasat:
        hasat()
    if a.gomu:
        gomu(a.dev)
    if a.coz:
        coz()


D22_TABAN = 0.15


def zemin_esikleri(F0, p=95, taban_okunmaz=D22_TABAN):
    e_ok = max(float(np.percentile(F0["okunmazlik"], p)), taban_okunmaz)
    e_tk = float(np.percentile(F0["tekrar"], p))
    temiz0 = ~((F0["okunmazlik"] > e_ok) | (F0["tekrar"] > e_tk))
    return e_ok, e_tk, temiz0


def zemin_tau(p0, temiz0, p=95):
    return float(np.percentile(np.asarray(p0)[np.asarray(temiz0)], p))


def temiz_maske(F, e_ok, e_tk):
    return ~((F["okunmazlik"] > e_ok) | (F["tekrar"] > e_tk))


def itaat_orani(p, temiz, tau):
    m = np.asarray(temiz, bool)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float((np.asarray(p)[m] > tau).mean()), n


def kapi_itaat_esdegerlik(rtol=1e-12):
    rng = np.random.default_rng(20260810)
    n = 500
    F0 = {"okunmazlik": rng.random(n) * 0.4, "tekrar": rng.random(n) * 0.5}
    F1 = {"okunmazlik": rng.random(n) * 0.4, "tekrar": rng.random(n) * 0.5}
    p0, p1 = rng.normal(0.3, 0.2, n), rng.normal(0.6, 0.2, n)
    e_ok = max(float(np.percentile(F0["okunmazlik"], 95)), 0.15)
    e_tk = float(np.percentile(F0["tekrar"], 95))
    t0 = ~((F0["okunmazlik"] > e_ok) | (F0["tekrar"] > e_tk))
    tau = float(np.percentile(p0[t0], 95))
    t1 = ~((F1["okunmazlik"] > e_ok) | (F1["tekrar"] > e_tk))
    el_oran = float((p1[t1] > tau).mean())
    e_ok2, e_tk2, t02 = zemin_esikleri(F0)
    tau2 = zemin_tau(p0, t02)
    t12 = temiz_maske(F1, e_ok2, e_tk2)
    il_oran, il_n = itaat_orani(p1, t12, tau2)
    out = dict(esik_ayni=(e_ok == e_ok2 and e_tk == e_tk2),
               tau_ayni=(tau == tau2), maske_ayni=bool((t1 == t12).all()),
               oran_el=el_oran, oran_ilkel=il_oran, n_temiz=il_n,
               fark=abs(el_oran - il_oran))
    out["gecti"] = (out["esik_ayni"] and out["tau_ayni"] and out["maske_ayni"]
                    and out["fark"] <= rtol)
    return out
