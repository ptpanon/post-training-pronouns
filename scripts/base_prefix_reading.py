#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, collections
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from base_prefix import (OUT, N_EBEVEYN, K_ONEK, SEED)
from adil_mac import ebeveyn_havuzu

MNT = __DNH_DATA__ + ""
P4 = f"{MNT}/rk_p4/gomu"
CIK = f"{ROOT}/unreleased/line"
GOM = f"{OUT}/gomu"
B_BOOT, MDE_K = 2000, 1.645
AILELER = ("qwen", "mistral")
KARAR_KOL = ("base_onek", "base_oneksiz", "instruct_onek", "instruct_oneksiz")
KILL_KOL = ("base_plasebo", "base_takas", "base_karistir",
            "instruct_plasebo", "instruct_takas", "instruct_karistir")
BETIM_KOL = ("instruct_sohbet",)
TUM_KOL = KARAR_KOL + KILL_KOL + BETIM_KOL
os.makedirs(GOM, exist_ok=True)


def gomucu(ad, metinler, dev):
    import p4_yelpaze as P
    P.DEV = dev
    return P.gom(ad, metinler)


def gomu_yukle(f, beklenen_n):
    if not os.path.exists(f):
        return None
    try:
        V = np.load(f)
    except Exception as e:
        print(f"    ⚠ BOZUK ÖNBELLEK silindi ({os.path.basename(f)}): {e}", flush=True)
        os.remove(f); return None
    if beklenen_n is not None and V.shape[0] != beklenen_n:
        print(f"    ⚠ EKSIK ÖNBELLEK silindi ({os.path.basename(f)}): "
              f"{V.shape[0]} ≠ beklenen {beklenen_n}", flush=True)
        os.remove(f); return None
    return V


def gomu_yaz(f, V):
    t = f"{f}.{os.getpid()}.tmp.npy"
    np.save(t, V); os.replace(t, f)


def kol_oku(aile, kol):
    yol = f"{OUT}/uret__{aile}__{kol}.jsonl"
    if not os.path.exists(yol):
        return None
    sat = []
    for line in open(yol, encoding="utf-8"):
        try:
            sat.append(json.loads(line))
        except Exception:
            pass
    return sat


def gom_kol(aile, kol, enc, dev):
    f = f"{GOM}/{aile}__{kol}__{enc}.npy"
    ix = f"{GOM}/{aile}__{kol}__indeks.json"
    sat = kol_oku(aile, kol)
    if sat is None:
        return None, None
    if not os.path.exists(ix):
        json.dump([dict(ebeveyn=s["ebeveyn"], korpus=s["korpus"], cekim=s["cekim"],
                        kesik=s.get("kesik", 0), kopya=s.get("kopya_orani", 0.0),
                        n_kelime=len(s["metin"].split()),
                        ttr=len(set(s["metin"].lower().split())) / max(len(s["metin"].split()), 1))
                   for s in sat], open(ix, "w"))
    V = gomu_yukle(f, len(sat))
    if V is not None:
        return V, json.load(open(ix))
    t0 = time.time()
    V = gomucu(enc, [s["metin"] for s in sat], dev)
    gomu_yaz(f, V)
    print(f"    gömüldü {aile}/{kol}/{enc}: {V.shape} [{time.time()-t0:.0f}s]", flush=True)
    return V, json.load(open(ix))


def kontrol_sekiz(aile, paket):
    onek_metinleri = set(paket["X"]["onek"][:K_ONEK]) | set(paket["Y"]["onek"][:K_ONEK])
    red = collections.Counter()
    doluluk, imza = {}, {}
    for kol in TUM_KOL:
        sat = kol_oku(aile, kol)
        if sat is None:
            red["kol_yok"] += 1; continue
        met = [s["metin"] for s in sat]
        doluluk[kol] = len(met)
        red["b_kaynakla_ozdes"] += sum(1 for t in met if t.strip() in onek_metinleri)
        imza[kol] = {hash(t) for t in met}
        red["a_kol_ici_mukerrer"] += len(met) - len(imza[kol])
    kollar = sorted(imza)
    for i in range(len(kollar)):
        for j in range(i + 1, len(kollar)):
            red["d_kollar_arasi_cakisma"] += len(imza[kollar[i]] & imza[kollar[j]])
    payda(f"sekiz_kontrol_{aile}", n_kol=len(doluluk), n_satir=sum(doluluk.values()),
          **{f"red_{k}": v for k, v in red.items()})
    print(f"{aile} {len(doluluk)} {sum(doluluk.values())}"
          f"{red['a_kol_ici_mukerrer']} {red['b_kaynakla_ozdes']}"
          f"{red['d_kollar_arasi_cakisma']}", flush=True)
    if red["kol_yok"]:
        raise RuntimeError(f"§8 KONTROL DÜSTÜ — kol eksik: {dict(red)}")
    if red["b_kaynakla_ozdes"]:
        print(f"{red['b_kaynakla_ozdes']}"
              f"", flush=True)
    return doluluk, dict(red)


def _kayma(V, HX, HY):
    return V @ HX.mean(0) - V @ HY.mean(0)


def _grupla(ix, deger):
    g = collections.defaultdict(list)
    for s, d in zip(ix, deger):
        g[s["ebeveyn"]].append(float(d))
    return g


def _yelpaze(V, ix):
    g = collections.defaultdict(list)
    for k, s in enumerate(ix):
        g[s["ebeveyn"]].append(k)
    out, dus = {}, 0
    for e, idx in g.items():
        if len(idx) < 2:
            dus += 1; continue
        W = V[idx]
        C = W @ W.T
        n = len(idx)
        out[e] = float((n * n - C.sum()) / (n * (n - 1)))
    return out, dus


def kume_bootstrap(deger_per_ebeveyn, ebe_korpus, B=B_BOOT, seed=SEED):
    kume = collections.defaultdict(list)
    for e, v in deger_per_ebeveyn.items():
        kume[ebe_korpus[e]].append(v)
    ks = sorted(kume)
    if not ks:
        return None
    rng = np.random.default_rng(seed)
    ort = [np.mean(kume[k]) for k in ks]
    gercek = float(np.mean(np.concatenate([kume[k] for k in ks])))
    boot = np.empty(B)
    for b in range(B):
        sec = rng.integers(0, len(ks), len(ks))
        boot[b] = float(np.mean([ort[i] for i in sec]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(deger=gercek, lo=float(lo), hi=float(hi), sd=float(boot.std(ddof=1)),
                n_kume=len(ks), n_ebeveyn=sum(len(v) for v in kume.values()))


def fark_bootstrap(A, B_, ebe_korpus, B=B_BOOT, seed=SEED):
    ortak = sorted(set(A) & set(B_))
    d = {e: A[e] - B_[e] for e in ortak}
    r = kume_bootstrap(d, ebe_korpus, B, seed)
    if r:
        r["n_eslesen"] = len(ortak)
    return r, d


def tavan_kayma(paket, enc, dev, HX, HY, n=200):
    from base_prefix import korpus_oku, X_KORPUS, Y_KORPUS
    yasak = set(paket["X"]["onek"]) | set(paket["Y"]["onek"]) | \
        set(paket["X"]["hedef"]) | set(paket["Y"]["hedef"])
    rng = np.random.default_rng(SEED)
    out = {}
    for rol, c in (("X", X_KORPUS), ("Y", Y_KORPUS)):
        U = korpus_oku(c)
        met = [t for (t, u, r) in U.values() if t not in yasak]
        sec = [met[i] for i in rng.permutation(len(met))[:n]]
        V = gomucu(enc, sec, dev)
        out[rol] = _kayma(V, HX, HY)
    return dict(insan_X=float(np.mean(out["X"])), insan_Y=float(np.mean(out["Y"])),
                ayrim=float(np.mean(out["X"]) - np.mean(out["Y"])),
                n=n, sd_X=float(np.std(out["X"])), sd_Y=float(np.std(out["Y"])))


def tavan_kardes(ebe, enc):
    hedef = {e["id"]: e["korpus"] for e in ebe}
    per_korpus = collections.defaultdict(list)
    for i, k in hedef.items():
        per_korpus[k].append(i)
    kardes, havuz, red = {}, collections.defaultdict(list), collections.Counter()
    for c, idler in per_korpus.items():
        f = f"{P4}/{c}__{enc}.npy"
        ixf = f"{P4}/{c}__indeks.json"
        if not (os.path.exists(f) and os.path.exists(ixf)):
            red["korpus_gomu_yok"] += len(idler); continue
        V = np.load(f, mmap_mode="r")
        D = json.load(open(ixf))
        poz = {t: k for k, t in enumerate(D["idler"])}
        sat = {s["ebeveyn"]: s["kardesler"] for s in D["sat"]}
        for e in idler:
            ks = [poz[k] for k in sat.get(e, []) if k in poz]
            if len(ks) < 2:
                red["kardes_2_alti"] += 1; continue
            W = np.asarray(V[ks], dtype=np.float32)
            W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
            n = len(ks)
            kardes[e] = float((n * n - (W @ W.T).sum()) / (n * (n - 1)))
            havuz[c].append((n, W))
    rng = np.random.default_rng(SEED)
    taban = {}
    for c, lst in havuz.items():
        if len(lst) < 2:
            continue
        tum = np.concatenate([w for (_n, w) in lst])
        for (n, _w) in lst:
            idx = rng.permutation(len(tum))[:n]
            W = tum[idx]
            taban.setdefault(c, []).append(float((n * n - (W @ W.T).sum()) / (n * (n - 1))))
    payda(f"tavan_kardes_{enc}", n_ebeveyn=len(kardes), n_korpus=len(havuz),
          red_kardes_2_alti=red["kardes_2_alti"], red_korpus_gomu_yok=red["korpus_gomu_yok"])
    tb = float(np.mean([v for l_ in taban.values() for v in l_])) if taban else float("nan")
    return kardes, tb


def oku_aile(aile, enc, dev, paket, ebe):
    ebe_korpus = {e["id"]: e["korpus"] for e in ebe}
    HX = gomucu(enc, paket["X"]["hedef"], dev)
    HY = gomucu(enc, paket["Y"]["hedef"], dev)
    K, D, IX, YUZEY = {}, {}, {}, {}
    for kol in TUM_KOL:
        V, ix = gom_kol(aile, kol, enc, dev)
        if V is None:
            print(f"{aile} {kol}", flush=True); continue
        IX[kol] = ix
        km = _kayma(V, HX, HY)
        K[kol] = {e: float(np.mean(v)) for e, v in _grupla(ix, km).items()}
        y, dus = _yelpaze(V, ix)
        D[kol] = y
        YUZEY[kol] = dict(
            n_kelime={e: float(np.mean(v)) for e, v in
                      _grupla(ix, [s["n_kelime"] for s in ix]).items()},
            ttr={e: float(np.mean(v)) for e, v in _grupla(ix, [s["ttr"] for s in ix]).items()},
            kesik={e: float(np.mean(v)) for e, v in _grupla(ix, [s["kesik"] for s in ix]).items()},
            kopya={e: float(np.mean(v)) for e, v in _grupla(ix, [s["kopya"] for s in ix]).items()},
            yelpaze_dusen=dus)
    return dict(kayma=K, daralma=D, yuzey=YUZEY, ebe_korpus=ebe_korpus, HX=HX, HY=HY)


def _spearman_kume(x, y, ebe_korpus, seed=SEED, B=1000):
    from scipy.stats import spearmanr
    ortak = sorted(set(x) & set(y))
    if len(ortak) < 10:
        return None
    kume = collections.defaultdict(list)
    for e in ortak:
        kume[ebe_korpus[e]].append((x[e], y[e]))
    ks = sorted(kume)
    xs = np.array([x[e] for e in ortak]); ys = np.array([y[e] for e in ortak])
    r0 = float(spearmanr(xs, ys).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        sec = rng.integers(0, len(ks), len(ks))
        pr = [p for i in sec for p in kume[ks[i]]]
        if len(pr) < 10:
            continue
        a = np.array([p[0] for p in pr]); b = np.array([p[1] for p in pr])
        if a.std() == 0 or b.std() == 0:
            continue
        boot.append(spearmanr(a, b).statistic)
    if len(boot) < 100:
        return dict(rho=r0, lo=float("nan"), hi=float("nan"), n=len(ortak), ayrik=False)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(rho=r0, lo=float(lo), hi=float(hi), n=len(ortak),
                ayrik=bool(lo > 0 or hi < 0))


VERDICT_ADLARI = frozenset({
    "KANAL", "IKISI", "KANAL-ASIMETRIK", "AYRISMADI (güc)", "TERBIYE", "ÖLCÜLEMEZ", "YOK",
})


def verdict(sonuc):
    b, i, x = sonuc["base"], sonuc["instruct"], sonuc["etkilesim"]
    mde, mde_x = sonuc["mde"], sonuc["mde_etkilesim"]
    sd = sonuc["null_sd"]
    poz = lambda r: r["lo"] > 0
    zayif = lambda r: sd > 0 and 1.0 <= abs(r["deger"]) / sd < 1.65
    if zayif(b) or zayif(i):
        return "ÖLCÜLEMEZ", "ana etki 1–1,65 null-sd bandinda (CLAUDE.md §6: contingent)"
    ana = min(abs(b["deger"]), abs(i["deger"]))
    if poz(b) and poz(i):
        if x["lo"] <= 0 <= x["hi"]:
            if mde_x <= ana:
                return "KANAL", "iki modelde de KG-ayrik pozitif · etkilesim 0'i iceriyor · " \
                                f"MDE_etkilesim {mde_x:.4f} ≤ ana-etki {ana:.4f}"
            return "AYRISMADI (güc)", f"{mde_x:.4f} {ana:.4f}" \
                                      ""
        return "IKISI", "iki modelde de pozitif · etkilesim 0'dan AYRIK"
    if poz(i) and not poz(b):
        if not (x["lo"] <= 0 <= x["hi"]):
            return "TERBIYE", "yalniz instruct'ta KG-ayrik pozitif · etkilesim 0'dan ayrik"
        return "AYRISMADI (güc)", ""
    if poz(b) and not poz(i):
        if not (x["lo"] <= 0 <= x["hi"]):
            return "KANAL-ASIMETRIK", (""
                                       f"{x['deger']:+.4f}"
                                       f"{x['lo']:+.4f} {x['hi']:+.4f}")
        return "AYRISMADI (güc)", (""
                                   f"{x['deger']:+.4f} {x['lo']:+.4f} {x['hi']:+.4f}"
                                   "")
    if sd > 0 and max(abs(b["deger"]), abs(i["deger"])) / sd < 1.0:
        return "YOK", f"iki KG de 0'i iceriyor · |emilme| < 1 null-sd (MDE {mde:.4f})"
    return "ÖLCÜLEMEZ", f"{mde:.4f}"


def kos(aile, enc, dev, paket, ebe, sessiz=False):
    R = oku_aile(aile, enc, dev, paket, ebe)
    K, D, Y, ek = R["kayma"], R["daralma"], R["yuzey"], R["ebe_korpus"]
    out = {"aile": aile, "enc": enc}
    gerek = ("base_onek", "base_oneksiz", "instruct_onek", "instruct_oneksiz")
    if any(k not in K for k in gerek):
        raise RuntimeError(f"{[k for k in gerek if k not in K]}")

    def emilme(a, b_):
        r, d = fark_bootstrap(K[a], K[b_], ek)
        return r, d
    b_r, b_d = emilme("base_onek", "base_oneksiz")
    i_r, i_d = emilme("instruct_onek", "instruct_oneksiz")
    pl_b = pl_i = None
    if "base_plasebo" in K:
        pl_b, pl_bd = emilme("base_plasebo", "base_oneksiz")
    if "instruct_plasebo" in K:
        pl_i, pl_id = emilme("instruct_plasebo", "instruct_oneksiz")
    if pl_b is None or pl_i is None:
        raise RuntimeError("")
    null_sd = float(max(pl_b["sd"], pl_i["sd"]))
    mde = MDE_K * null_sd
    ort = sorted(set(b_d) & set(i_d))
    x_d = {e: i_d[e] - b_d[e] for e in ort}
    x_r = kume_bootstrap(x_d, ek)
    ortp = sorted(set(pl_bd) & set(pl_id))
    xp = {e: pl_id[e] - pl_bd[e] for e in ortp}
    xp_r = kume_bootstrap(xp, ek)
    mde_x = MDE_K * xp_r["sd"]
    ka_b = emilme("base_karistir", "base_oneksiz")[0] if "base_karistir" in K else None
    ka_i = emilme("instruct_karistir", "instruct_oneksiz")[0] if "instruct_karistir" in K else None
    tk_b = emilme("base_takas", "base_oneksiz")[0] if "base_takas" in K else None
    tk_i = emilme("instruct_takas", "instruct_oneksiz")[0] if "instruct_takas" in K else None
    dar_b = fark_bootstrap(D["base_onek"], D["base_oneksiz"], ek)[0]
    dar_i = fark_bootstrap(D["instruct_onek"], D["instruct_oneksiz"], ek)[0]
    kardes, kardes_taban = tavan_kardes(ebe, enc)
    tv_kardes = kume_bootstrap(kardes, ek)
    tv_kayma = tavan_kayma(paket, enc, dev, R["HX"], R["HY"])
    d_kesik = {e: Y["base_onek"]["kesik"][e] - Y["base_oneksiz"]["kesik"][e]
               for e in set(Y["base_onek"]["kesik"]) & set(Y["base_oneksiz"]["kesik"])}
    d_uzun = {e: Y["base_onek"]["n_kelime"][e] - Y["base_oneksiz"]["n_kelime"][e]
              for e in set(Y["base_onek"]["n_kelime"]) & set(Y["base_oneksiz"]["n_kelime"])}
    d_ttr = {e: Y["base_onek"]["ttr"][e] - Y["base_oneksiz"]["ttr"][e]
             for e in set(Y["base_onek"]["ttr"]) & set(Y["base_oneksiz"]["ttr"])}
    di_kesik = {e: Y["instruct_onek"]["kesik"][e] - Y["instruct_oneksiz"]["kesik"][e]
                for e in set(Y["instruct_onek"]["kesik"]) & set(Y["instruct_oneksiz"]["kesik"])}
    di_uzun = {e: Y["instruct_onek"]["n_kelime"][e] - Y["instruct_oneksiz"]["n_kelime"][e]
               for e in set(Y["instruct_onek"]["n_kelime"])
               & set(Y["instruct_oneksiz"]["n_kelime"])}
    serh = dict(
        kesik_base=_spearman_kume(b_d, d_kesik, ek),
        kesik_instruct=_spearman_kume(i_d, di_kesik, ek),
        uzunluk_base=_spearman_kume(b_d, d_uzun, ek),
        uzunluk_instruct=_spearman_kume(i_d, di_uzun, ek),
        ttr_base=_spearman_kume(b_d, d_ttr, ek),
        kopya_base=_spearman_kume(b_d, Y["base_onek"]["kopya"], ek),
        kopya_instruct=_spearman_kume(i_d, Y["instruct_onek"]["kopya"], ek))
    out.update(dict(base=b_r, instruct=i_r, etkilesim=x_r, plasebo_base=pl_b,
                    plasebo_instruct=pl_i, plasebo_etkilesim=xp_r, null_sd=null_sd, mde=mde,
                    mde_etkilesim=mde_x, karistir_base=ka_b, karistir_instruct=ka_i,
                    takas_base=tk_b, takas_instruct=tk_i, daralma_base=dar_b,
                    daralma_instruct=dar_i, tavan_kardes=tv_kardes,
                    tavan_kardes_taban=kardes_taban, tavan_kayma=tv_kayma, serh=serh,
                    kol_kayma={k: float(np.mean(list(v.values()))) for k, v in K.items()},
                    kol_daralma={k: float(np.mean(list(v.values()))) for k, v in D.items()}))
    out["verdict"], out["verdict_gerekce"] = verdict(out)
    payda(f"okuma_{aile}_{enc}", n_kol=len(K), n_ebeveyn_base=b_r["n_eslesen"],
          n_ebeveyn_instruct=i_r["n_eslesen"], n_kume=b_r["n_kume"],
          red_yelpaze_dusen=sum(Y[k]["yelpaze_dusen"] for k in Y))
    if sessiz:
        print(f"{aile} {enc}",
              flush=True)
        return out
    print(f"  ★ {aile}/{enc}: emilme base {b_r['deger']:+.4f} [{b_r['lo']:+.4f},{b_r['hi']:+.4f}]"
          f" · instruct {i_r['deger']:+.4f} [{i_r['lo']:+.4f},{i_r['hi']:+.4f}] · "
          f"etkilesim {x_r['deger']:+.4f} [{x_r['lo']:+.4f},{x_r['hi']:+.4f}] · "
          f"null-sd {null_sd:.4f} MDE {mde:.4f} ⇒ **{out['verdict']}**", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aile", default="qwen,mistral"); ap.add_argument("--enc", default="e5,bge")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--sessiz", action="store_true",
                    help="karar niceliklerini DISKE yaz, EKRANA BASMA (D44 §3)")
    ap.add_argument("--cikti", default=None, help="cikti JSON adi (varsayilan base_onek_okuma)")
    a = ap.parse_args()
    print("═" * 96); print("KARAR-42 · BASE-ÖNEK OKUMA (prereg §8'in TEK yolu)")
    import random as _r
    from base_prefix import kapi_taklit, kapi_cihaz
    paket = kapi_taklit(_r.Random(SEED))
    dev = a.dev or kapi_cihaz(6)
    ebe = ebeveyn_havuzu(N_EBEVEYN)
    print(f"  havuz {len(ebe)} ebeveyn · {len({e['korpus'] for e in ebe})} korpus", flush=True)
    T = {}
    for aile in a.aile.split(","):
        kontrol_sekiz(aile, paket)
        for enc in a.enc.split(","):
            T[f"{aile}__{enc}"] = kos(aile, enc, dev, paket, ebe, a.sessiz)
    yol = f"{CIK}/{a.cikti or 'base_onek_okuma'}.json"
    json.dump(T, open(yol, "w"), indent=1)
    if a.sessiz:
        print(f"{len(T)} {yol}"
              "")
    else:
        print("\n★ HÜKÜMLER: " + " · ".join(f"{k}={v['verdict']}" for k, v in T.items()))
