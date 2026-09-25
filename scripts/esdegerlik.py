import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, hashlib, subprocess

os.environ.setdefault("HF_HOME", __DNH_ROOT__ + "/.hf")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "32")

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA

import kt_kodlama as KT
import muhafiz_sabit_konum as MSK
import gomme_guard as GG

MNT = __DNH_DATA__ + ""
QW = f"{MNT}/kt_qwen25"
ES = f"{MNT}/esdeger"
MAN = f"{ROOT}/unreleased/manifests"
ADAPTER = f"{MNT}/unreleased/ear2_adapter"
ST5 = "cartgr/embeddings-for-preferences-st5-xl"
DEV = "cuda:1"

HAVUZLAR = ("sontoken", "ortalama", "ortalama_sinksiz")
TARAFLAR = ("ebeveyn", "cevap")
BANT = 0.02
B_BOOT = 2000
Z90 = 1.6449

os.makedirs(ES, exist_ok=True); os.makedirs(MAN, exist_ok=True)


class KapsamHatasi(AssertionError):
    pass


def l2(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)


def auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        raise KapsamHatasi(f"AUC: tek sinif (n={len(y)}) — bu gecis DEGIL hata")
    if not np.all(np.isfinite(s)):
        raise KapsamHatasi("AUC: skorlar SONLU degil")
    a = roc_auc_score(y, s)
    if not np.isfinite(a):
        raise KapsamHatasi("AUC: sonuc SONLU degil")
    return float(a)


def yaz(*a):
    print(*a, flush=True)


def disapere(split, prereg=None):
    r = KT.load_split(split, prereg=prereg)
    return ([x["ebeveyn"] for x in r], [x["cevap"] for x in r],
            np.array([x["etiket"] for x in r]), np.array([x["forum_id"] for x in r]))


def kialo():
    import bs95_ear2 as B95
    tr, ev = B95.load_kialo()
    def ay(rows):
        return ([x[0] for x in rows], [x[1] for x in rows],
                np.array([1 - x[2] for x in rows]),
                np.array([x[3] for x in rows]))
    return ay(tr), ay(ev)


def _kaydet(yol, A):
    gec = yol[:-4] + ".tmp.npy"
    np.save(gec, A); os.replace(gec, yol)


def qwen_kodla(metinler, onek, bs=32):
    import kt_qwen25 as KQ
    hedef = {h: f"{ES}/{onek}__{h}.fp16.npy" for h in HAVUZLAR}
    if all(os.path.exists(p) for p in hedef.values()):
        yaz(f"    [atla] {onek} — üc havuz da diskte"); return None
    tok, model = KQ.yukle_model()
    s = dict.fromkeys(("gecerli_token", "pos0_elenen", "sink_elenen", "yedege_dusen",
                       "ucuncu_basamak", "bos_havuz"), 0)
    ST, OR, uzun, DZ = KQ.kodla(tok, model, metinler, bs=bs, sayac=s, dev=DEV, duz_ort=True)
    del tok, model
    import torch; torch.cuda.empty_cache()
    for h, A in (("sontoken", ST), ("ortalama", DZ), ("ortalama_sinksiz", OR)):
        if not np.isfinite(A).all():
            raise KapsamHatasi(f"{onek} {h}")
        _kaydet(hedef[h], A)
    uz = np.array(uzun)
    yaz(f"    {onek}: n={len(metinler)} · token ort {uz.mean():.1f} medyan {int(np.median(uz))} "
        f"maks {uz.max()} · max_length'e carpan {int((uz >= KQ.MAXLEN).sum())}/{len(uz)}")
    yaz(f"      kapsam: gecerli-token {s['gecerli_token']} · pos0 {s['pos0_elenen']} · "
        f"sink {s['sink_elenen']} · yedek {s['yedege_dusen']} · 3.basamak {s['ucuncu_basamak']} "
        f"· BOS HAVUZ {s['bos_havuz']}")
    if s["bos_havuz"]:
        raise KapsamHatasi(f"{onek}: BOS HAVUZ {s['bos_havuz']} — yer-tutucu üretilirdi")
    return dict(n=len(metinler), tok_ort=float(uz.mean()), tok_maks=int(uz.max()),
                carpan=int((uz >= KQ.MAXLEN).sum()), **{k: int(v) for k, v in s.items()})


def ear_kodla(metinler, yol, adapter=None):
    if os.path.exists(yol):
        yaz(f"    [atla] {os.path.basename(yol)}"); return
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ST5, device=DEV)
    if adapter:
        try:
            m.load_adapter(adapter)
        except Exception:
            from peft import PeftModel
            m[0].auto_model = PeftModel.from_pretrained(m[0].auto_model, adapter)
    m = m.to(torch.bfloat16)
    E = np.asarray(m.encode(metinler, batch_size=128, normalize_embeddings=True,
                            convert_to_numpy=True, device=DEV), np.float32)
    del m; torch.cuda.empty_cache()
    if not np.isfinite(E).all():
        raise KapsamHatasi(f"{yol}: SONLU DEGIL")
    _kaydet(yol, l2(E))


def hazirla(prereg=None):
    t0 = time.time()
    yaz("══ HAZIRLIK · kodlama ══")
    olcum = {}

    yaz("")
    for sp in ("train", "dev"):
        P, C, y, f = disapere(sp)
        for taraf, T in (("ebeveyn", P), ("cevap", C)):
            ear_kodla(T, f"{ES}/dis__{sp}__{taraf}__stock.fp32.npy")
            ear_kodla(T, f"{ES}/dis__{sp}__{taraf}__ear20.fp32.npy", ADAPTER)

    if prereg:
        yaz(f"  · DISAPERE test — MÜHÜRLÜ DOKUNUS (prereg {prereg})")
        P, C, y, f = disapere("test", prereg=prereg)
        yaz(f"    test: {len(y)} cift · dispute {int(y.sum())} · forum {len(set(f.tolist()))}")
        for taraf, T in (("ebeveyn", P), ("cevap", C)):
            olcum[f"qwen_dis_test_{taraf}"] = qwen_kodla(T, f"dis__test__{taraf}")
            ear_kodla(T, f"{ES}/dis__test__{taraf}__stock.fp32.npy")
            ear_kodla(T, f"{ES}/dis__test__{taraf}__ear20.fp32.npy", ADAPTER)

    yaz("")
    (Ptr, Ctr, ytr, dtr), (Pev, Cev, yev, dev_) = kialo()
    yaz(f"    train {len(ytr)} cift / {len(set(dtr.tolist()))} tartisma · "
        f"heldout {len(yev)} cift / {len(set(dev_.tolist()))} tartisma")
    for bol, (P, C) in (("train", (Ptr, Ctr)), ("heldout", (Pev, Cev))):
        for taraf, T in (("ebeveyn", P), ("cevap", C)):
            olcum[f"qwen_kia_{bol}_{taraf}"] = qwen_kodla(T, f"kia__{bol}__{taraf}")
            ear_kodla(T, f"{ES}/kia__{bol}__{taraf}__stock.fp32.npy")
            ear_kodla(T, f"{ES}/kia__{bol}__{taraf}__ear20.fp32.npy", ADAPTER)

    json.dump({k: v for k, v in olcum.items() if v}, open(f"{ES}/kodlama_olcum.json", "w"),
              ensure_ascii=False, indent=1)
    yaz(f"══ hazirlik bitti ({time.time()-t0:.0f}s)")


def muhafiz(ad, V, kaynak=None):
    n = len(V)
    if n == 0:
        raise KapsamHatasi(f"{ad}: PAYDA SIFIR")
    W = V.reshape(n, -1).astype(np.float32)
    if not np.isfinite(W).all():
        raise KapsamHatasi(f"{ad}: SONLU DEGIL")
    sifir = int((np.linalg.norm(W, axis=1) < 1e-8).sum())
    hh = {hashlib.sha256(np.ascontiguousarray(W[i]).tobytes()).digest() for i in range(n)}
    yin = n - len(hh)
    yaz(f"    [muhafiz {ad}] payda {n} satir × {W.shape[1]} boyut · sifir-norm {sifir} · "
        f"birebir-yineleme {yin}")
    if sifir:
        raise KapsamHatasi(f"{ad}: {sifir} SIFIR VEKTÖR — yer-tutucu")
    return dict(payda=n, sifir=sifir, yineleme=yin)


def ozellik(P, C):
    P, C = l2(np.asarray(P, np.float32)), l2(np.asarray(C, np.float32))
    return np.column_stack([P, C, np.abs(P - C)])


def probe(Xtr, ytr, Xte, dengeli, iters):
    m = LogisticRegression(max_iter=iters, C=0.5,
                           class_weight=("balanced" if dengeli else None)).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


ZEMIN = {
    "disapere": dict(dengeli=True, iters=3000, kume="forum_id"),
    "kialo":    dict(dengeli=False, iters=2000, kume="debate"),
}


def qw_kat(yolonek, L, h):
    A = np.load(f"{ES if 'test' in yolonek or 'kia' in yolonek else QW}/{yolonek}__{h}.fp16.npy",
                mmap_mode="r")
    return np.asarray(A[:, L], np.float32)


def kume_boot(y, skorlar, kume, B=B_BOOT, seed=0):
    uk = np.unique(kume); io = {c: np.where(kume == c)[0] for c in uk}
    rng = np.random.default_rng(seed)
    adlar = list(skorlar)
    A = {a: [] for a in adlar}
    for _ in range(B):
        tk = np.concatenate([io[c] for c in rng.choice(uk, len(uk), True)])
        if len(set(y[tk].tolist())) < 2:
            continue
        for a in adlar:
            A[a].append(auc(y[tk], skorlar[a][tk]))
    return {a: np.array(v) for a, v in A.items()}, len(uk)


def tost(dboot, d_nokta, bant=BANT, kollar=("ilk", "ikinci")):
    lo, hi = float(np.percentile(dboot, 5)), float(np.percentile(dboot, 95))
    sd = float(dboot.std(ddof=1))
    if -bant < lo and hi < bant:
        h = "ESDEGER (bant-ici)"
    elif lo > bant or hi < -bant:
        one, ger = (kollar[0], kollar[1]) if lo > bant else (kollar[1], kollar[0])
        h = f"ESDEGER DEGIL (bant-disi, {one} ÖNDE / {ger} GERIDE)"
    else:
        h = f"KARARSIZ (bu tasarim ±{str(band).replace('.', ',')}'yi cözemiyor)"
    return dict(delta=float(d_nokta), ci90=[lo, hi], sd=sd, cozunurluk=Z90 * sd,
                erisilebilir=bool(Z90 * sd < band), verdict=h)


def sec_disapere(nL):
    _, _, ytr, _ = disapere("train"); _, _, yde, _ = disapere("dev")
    P = ZEMIN["disapere"]; izgara = []
    for h in HAVUZLAR:
        for L in range(nL):
            Xtr = ozellik(qw_kat("train__ebeveyn", L, h), qw_kat("train__cevap", L, h))
            Xde = ozellik(qw_kat("dev__ebeveyn", L, h), qw_kat("dev__cevap", L, h))
            a = auc(yde, probe(Xtr, ytr, Xde, P["dengeli"], P["iters"]))
            izgara.append(dict(katman=L, havuz=h, auc=float(a)))
    return izgara


def sec_kialo(nL):
    (_, _, ytr, dtr), _ = kialo()
    P = ZEMIN["kialo"]; gkf = list(GroupKFold(5).split(np.zeros(len(ytr)), ytr, dtr))
    izgara = []
    for h in HAVUZLAR:
        for L in range(nL):
            X = ozellik(qw_kat("kia__train__ebeveyn", L, h), qw_kat("kia__train__cevap", L, h))
            oof = np.zeros(len(ytr))
            for tri, tei in gkf:
                oof[tei] = probe(X[tri], ytr[tri], X[tei], P["dengeli"], P["iters"])
            izgara.append(dict(katman=L, havuz=h, auc=float(auc(ytr, oof))))
    return izgara


def en_iyi(izgara):
    hs = {h: i for i, h in enumerate(HAVUZLAR)}
    return sorted(izgara, key=lambda r: (-r["auc"], r["katman"], hs[r["havuz"]]))[0]


def _qw_onek(zemin, bol, taraf):
    if zemin == "kialo":
        return f"kia__{bol}__{taraf}"
    return f"dis__test__{taraf}" if bol == "test" else f"{bol}__{taraf}"


def _ear_yol(zemin, bol, taraf, kol):
    kok = "kia" if zemin == "kialo" else "dis"
    return f"{ES}/{kok}__{bol}__{taraf}__{kol}.fp32.npy"


def qw_yigin(zemin, bolumler, L, h):
    P = np.concatenate([qw_kat(_qw_onek(zemin, b, "ebeveyn"), L, h) for b in bolumler])
    C = np.concatenate([qw_kat(_qw_onek(zemin, b, "cevap"), L, h) for b in bolumler])
    return P, C


def ear_yigin(zemin, bolumler, kol):
    P = np.concatenate([np.load(_ear_yol(zemin, b, "ebeveyn", kol)) for b in bolumler])
    C = np.concatenate([np.load(_ear_yol(zemin, b, "cevap", kol)) for b in bolumler])
    return P, C


def tfidf_skor(met_tr, ytr, met_te, dengeli, iters):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(min_df=3, max_features=30000, sublinear_tf=True, ngram_range=(1, 2))
    X = vec.fit_transform(met_tr + met_te)
    Z = TruncatedSVD(200, random_state=0).fit_transform(X)
    n = len(met_tr)
    return probe(Z[:n], ytr, Z[n:], dengeli, iters)


def kollar_hesapla(zemin, egit, olc, LH, ytr, yte, kapasite=True, ek_LH=None, metinler=None):
    P = ZEMIN[zemin]; S = {}
    Ptr, Ctr = qw_yigin(zemin, egit, LH["katman"], LH["havuz"])
    Pte, Cte = qw_yigin(zemin, olc, LH["katman"], LH["havuz"])
    muhafiz(f"{zemin}/prob-ölcüm", Pte)
    S["prob"] = probe(ozellik(Ptr, Ctr), ytr, ozellik(Pte, Cte), P["dengeli"], P["iters"])
    if kapasite:
        pca_p = PCA(768, random_state=0).fit(l2(Ptr)); pca_c = PCA(768, random_state=0).fit(l2(Ctr))
        S["prob768"] = probe(ozellik(pca_p.transform(l2(Ptr)), pca_c.transform(l2(Ctr))), ytr,
                             ozellik(pca_p.transform(l2(Pte)), pca_c.transform(l2(Cte))),
                             P["dengeli"], P["iters"])
    if ek_LH:
        A, Bc = qw_yigin(zemin, egit, ek_LH["katman"], ek_LH["havuz"])
        Ce, De = qw_yigin(zemin, olc, ek_LH["katman"], ek_LH["havuz"])
        S["prob_capraz"] = probe(ozellik(A, Bc), ytr, ozellik(Ce, De), P["dengeli"], P["iters"])
    for kol in ("stock", "ear20"):
        Ktr = ear_yigin(zemin, egit, kol); Kte = ear_yigin(zemin, olc, kol)
        muhafiz(f"{zemin}/{kol}-ölcüm", Kte[0])
        S[kol] = probe(ozellik(*Ktr), ytr, ozellik(*Kte), P["dengeli"], P["iters"])
    if metinler:
        S["tfidf"] = tfidf_skor(metinler[0], ytr, metinler[1], P["dengeli"], P["iters"])
    return S


def rapor_kollar(y, S, kume, baslik):
    yaz(f"\n  ── {baslik} ── (n={len(y)}, küme={len(np.unique(kume))})")
    for a in S:
        yaz(f"      {a:12s} AUC = {auc(y, S[a]):.4f}")
    boot, nk = kume_boot(y, S, kume)
    return boot, nk


def tost_ciftleri(y, S, boot, ciftler):
    out = {}
    for ad, (a, b) in ciftler.items():
        if a not in S or b not in S:
            continue
        d = auc(y, S[a]) - auc(y, S[b])
        out[ad] = tost(boot[a] - boot[b], d, kollar=(a, b))
        r = out[ad]
        yaz(f"      TOST {ad:22s} Δ={r['delta']:+.4f}  %90CI[{r['ci90'][0]:+.4f},"
            f"{r['ci90'][1]:+.4f}]  sd={r['sd']:.4f}  cöz={r['cozunurluk']:.4f}  → {r['verdict']}")
    return out


CIFTLER = {
    "prob−ear20 (PRIMER)": ("prob", "ear20"),
    "prob−stock": ("prob", "stock"),
    "prob768−ear20 (kapasite)": ("prob768", "ear20"),
    "prob_capraz−ear20 (capraz)": ("prob_capraz", "ear20"),
    "KILL+ stock−ear20": ("stock", "ear20"),
    "KILL− tfidf−stock": ("tfidf", "stock"),
}


def sentetik_kapi():
    yaz("\n══ KAPI-A · SENTETIK (kestirici ve TOST dejenere girdide) ══")
    rng = np.random.default_rng(0); n = 400
    y = (rng.random(n) < 0.3).astype(int); k = rng.integers(0, 20, n)
    sabit = np.ones((n, 8))
    a = auc(y, probe(sabit, y, sabit, True, 200))
    yaz(f"  [1] SABIT öznitelik → AUC = {a:.4f}  (beklenen 0,5; ≠0,5 ise boru sizdiriyor)")
    tek = np.zeros(n, int); r_ham = None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r_ham = float(roc_auc_score(tek, rng.random(n)))
        except Exception as e:
            r_ham = f"{type(e).__name__}"
    try:
        auc(tek, rng.random(n)); ok2 = False
    except KapsamHatasi:
        ok2 = True
    yaz(f"  [2] TEK SINIF → HAM roc_auc_score = {r_ham}  ·  muhafizli auc() = "
        f"{'KapsamHatasi (dogru)' if ok2 else '⚠ SESSIZ GECTI'}")
    yaz(""
        "")
    s = rng.random(n) + y * 5
    yaz(f"  [3] MÜKEMMEL ayirici → AUC = {auc(y, s):.4f} (beklenen 1,0)")
    boot, _ = kume_boot(y, {"x": s, "x2": s}, k, B=200)
    t0 = tost(boot["x"] - boot["x2"], 0.0)
    yaz(f"  [4] ÖZDES kollar → Δ={t0['delta']:+.4f} sd={t0['sd']:.6f} → {t0['verdict']}")
    yaz(""
        "")
    z = rng.random(n) + y * 0.4
    boot2, _ = kume_boot(y, {"x": s, "z": z}, k, B=200)
    t1 = tost(boot2["x"] - boot2["z"], auc(y, s) - auc(y, z))
    yaz(f"{t1['verdict']}")
    return dict(sabit_auc=float(a), tek_sinif_hata=ok2, ozdes=t0, farkli=t1)


def kapi(prereg=None):
    t0 = time.time(); R = {"_zaman": time.strftime("%Y-%m-%d %H:%M:%S"), "_bant": BANT}
    R["sentetik"] = sentetik_kapi()

    nL = np.load(f"{QW}/dev__ebeveyn__sontoken.fp16.npy", mmap_mode="r").shape[1]
    yaz(f"\n══ KAPI-B · ISLETIM NOKTASI (nL={nL} durum) ══")

    yaz("\n▸ DISAPERE — secim `dev`de, ölcüm de burada (isletim-noktasi vekili)")
    ts = time.time(); izg = sec_disapere(nL); LHd = en_iyi(izg)
    yaz(f"  secim: ℓ*={LHd['katman']} h*={LHd['havuz']} dev-AUC={LHd['auc']:.4f} "
        f"({time.time()-ts:.0f}s, {len(izg)} hücre)")
    ust = sorted(izg, key=lambda r: -r["auc"])[:5]
    yaz("  ilk bes: " + " · ".join(f"ℓ{r['katman']}/{r['havuz'][:4]}={r['auc']:.4f}" for r in ust))
    Ptr, Ctr, ytr, _ = disapere("train"); Pde, Cde, yde, fde = disapere("dev")
    S = kollar_hesapla("disapere", ["train"], ["dev"], LHd, ytr, yde,
                       metinler=([a + " " + b for a, b in zip(Ptr, Ctr)],
                                 [a + " " + b for a, b in zip(Pde, Cde)]))
    boot, nk = rapor_kollar(yde, S, fde, "DISAPERE dev (isletim noktasi)")
    R["disapere_kapi"] = dict(secim=LHd, izgara_ilk5=ust, n=len(yde), kume=int(nk),
                              auc={a: float(auc(yde, S[a])) for a in S},
                              tost=tost_ciftleri(yde, S, boot, CIFTLER))

    yaz("")
    ts = time.time(); izgk = sec_kialo(nL); LHk = en_iyi(izgk)
    yaz(f"  secim: ℓ*={LHk['katman']} h*={LHk['havuz']} OOF-AUC={LHk['auc']:.4f} "
        f"({time.time()-ts:.0f}s, {len(izgk)} hücre)")
    ustk = sorted(izgk, key=lambda r: -r["auc"])[:5]
    yaz("  ilk bes: " + " · ".join(f"ℓ{r['katman']}/{r['havuz'][:4]}={r['auc']:.4f}" for r in ustk))
    (Pk, Ck, yk, dk), _ = kialo()
    gkf = list(GroupKFold(5).split(np.zeros(len(yk)), yk, dk)); P = ZEMIN["kialo"]
    SK = {}
    Xq = ozellik(*qw_yigin("kialo", ["train"], LHk["katman"], LHk["havuz"]))
    Xs = ozellik(*ear_yigin("kialo", ["train"], "stock"))
    Xe = ozellik(*ear_yigin("kialo", ["train"], "ear20"))
    for ad, X in (("prob", Xq), ("stock", Xs), ("ear20", Xe)):
        o = np.zeros(len(yk))
        for tri, tei in gkf:
            o[tei] = probe(X[tri], yk[tri], X[tei], P["dengeli"], P["iters"])
        SK[ad] = o
    boot2, nk2 = rapor_kollar(yk, SK, dk, "Kialo train-OOF (isletim noktasi)")
    R["kialo_kapi"] = dict(secim=LHk, izgara_ilk5=ustk, n=len(yk), kume=int(nk2),
                           auc={a: float(auc(yk, SK[a])) for a in SK},
                           tost=tost_ciftleri(yk, SK, boot2, CIFTLER))

    yaz("\n══ KAPI-C · MDE / ERISILEBILIRLIK ══")
    for z, key in (("DISAPERE", "disapere_kapi"), ("KIALO", "kialo_kapi")):
        t = R[key]["tost"].get("prob−ear20 (PRIMER)")
        if t:
            yaz(f"  {z}: sd(Δ)={t['sd']:.4f} ⇒ cözünürlük 1,645·sd = {t['cozunurluk']:.4f} "
                f"{'<' if t['erisilebilir'] else '≥'} bant {BANT} ⇒ "
                f"{'±0,02 ERISILEBILIR' if t['erisilebilir'] else ''}")
    json.dump(R, open(f"{MAN}/esdegerlik_kapi_2026-07-31.json", "w"), ensure_ascii=False, indent=1)
    yaz(f"\n══ kapi bitti ({time.time()-t0:.0f}s) → manifests/esdegerlik_kapi_2026-07-31.json")
    return R


def kos(prereg):
    if not prereg:
        raise SystemExit("--kos icin --prereg <commit-hash> ZORUNLU (R2: prereg = commit).")
    K = json.load(open(f"{MAN}/esdegerlik_kapi_2026-07-31.json"))
    LHd = K["disapere_kapi"]["secim"]; LHk = K["kialo_kapi"]["secim"]
    t0 = time.time()
    R = {"_prereg": prereg, "_zaman": time.strftime("%Y-%m-%d %H:%M:%S"), "_bant": BANT,
         "_secim": {"disapere": LHd, "kialo": LHk}}
    yaz(f"══ MÜHÜRLÜ KOSU · prereg {prereg} ══")
    yaz(f"  secim MÜHÜRDEN okundu: DISAPERE ℓ{LHd['katman']}/{LHd['havuz']} · "
        f"Kialo ℓ{LHk['katman']}/{LHk['havuz']}")

    Ptr, Ctr, ytr, _ = disapere("train"); Pd, Cd, yd, _ = disapere("dev")
    Pte, Cte, yte, fte = disapere("test", prereg=prereg)
    ytd = np.concatenate([ytr, yd])
    yaz(f"\n▸ DISAPERE test: {len(yte)} cift · dispute {int(yte.sum())} · "
        f"forum {len(np.unique(fte))}")
    S = kollar_hesapla("disapere", ["train", "dev"], ["test"], LHd, ytd, yte, ek_LH=LHk,
                       metinler=([a + " " + b for a, b in zip(Ptr + Pd, Ctr + Cd)],
                                 [a + " " + b for a, b in zip(Pte, Cte)]))
    boot, nk = rapor_kollar(yte, S, fte, "DISAPERE test — MÜHÜRLÜ")
    R["disapere"] = dict(n=len(yte), kume=int(nk),
                         auc={a: float(auc(yte, S[a])) for a in S},
                         tost=tost_ciftleri(yte, S, boot, CIFTLER))

    (Pk, Ck, yk, dkk), (Pe, Ce, ye, de) = kialo()
    yaz(f"\n▸ Kialo HELDOUT: {len(ye)} cift · {len(np.unique(de))} tartisma")
    SK = kollar_hesapla("kialo", ["train"], ["heldout"], LHk, yk, ye, ek_LH=LHd,
                        metinler=([a + " " + b for a, b in zip(Pk, Ck)],
                                  [a + " " + b for a, b in zip(Pe, Ce)]))
    boot2, nk2 = rapor_kollar(ye, SK, de, "Kialo HELDOUT — MÜHÜRLÜ")
    R["kialo"] = dict(n=len(ye), kume=int(nk2),
                      auc={a: float(auc(ye, SK[a])) for a in SK},
                      tost=tost_ciftleri(ye, SK, boot2, CIFTLER))

    yaz("")
    bek = {("disapere", "stock"): 0.807, ("disapere", "ear20"): 0.808,
           ("kialo", "stock"): 0.842, ("kialo", "ear20"): 0.882}
    sk = {}
    for (z, kol), b in bek.items():
        g = R[z]["auc"][kol]; sk[f"{z}/{kol}"] = dict(kayitli=b, olculen=g, fark=g - b)
        yaz(f"  {z:9s}/{kol:6s} kayitli {b:.3f} · ölcülen {g:.4f} · fark {g-b:+.4f}"
            f"{'   ⚠ >0,01' if abs(g-b) > 0.01 else ''}")
    R["sureklilik"] = sk
    json.dump(R, open(f"{MAN}/esdegerlik_2026-07-31.json", "w"), ensure_ascii=False, indent=1)
    yaz(f"\n══ kosu bitti ({time.time()-t0:.0f}s) → manifests/esdegerlik_2026-07-31.json")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazirla", action="store_true")
    ap.add_argument("--kapi", action="store_true")
    ap.add_argument("--kos", action="store_true")
    ap.add_argument("--prereg", default=None)
    A = ap.parse_args()
    if A.hazirla:
        hazirla(A.prereg)
    if A.kapi:
        kapi(A.prereg)
    if A.kos:
        kos(A.prereg)
    if not (A.hazirla or A.kapi or A.kos):
        ap.print_help()
