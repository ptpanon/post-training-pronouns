#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, glob, argparse, subprocess, hashlib, inspect, collections
from datetime import datetime, timedelta
import numpy as np

ROOT = __DNH_ROOT__ + ""
os.environ.setdefault("HF_HOME", __DNH_DATA__ + "/.hf_local")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import weight_dosyalari as AGD
from pergel_cekirdek import Cekirdek, Permutator, foldlar
from prefix_muhurlu_kosu import auc_np
from prefix_korpus_pilot import KELIME

KAYNAK = __DNH_DATA__ + "/onek_korpus/onarim"
OUT = __DNH_DATA__ + "/onek_korpus/katman"
CAPA_JSON = f"{KAYNAK}/okuma/okuma_ham.json"
PENCERE = "metin_160"
SEED, NFOLD = 20260806, 5
K_NULL, K_ACI, K_INT = 400, 400, 400
EPOK, LR, WD, GIZLI = 400, 3e-3, 1e-2, 64
BEKLE_SATIR = 17 * 2 * 2 * 24
DURAK_SAAT = 4
BAYRAK = f"{OUT}/DUR"

OKURLAR = {
    "mistral": dict(aile="mistral", repo="mistralai/Mistral-7B-v0.3",
                    yol=__DNH_DATA__ + "/.hf_local/hub/models--mistralai--Mistral-7B-v0.3"),
    "qwen":    dict(aile="qwen", repo="Qwen/Qwen2.5-7B",
                    yol=__DNH_DATA__ + "/.hf_local/hub/models--Qwen--Qwen2.5-7B"),
    "gemma":   dict(aile="gemma", repo="google/gemma-4-12b",
                    yol=__DNH_ROOT__ + "/.hf/hub/models--google--gemma-4-12b"),
}
YAZAR_AILE = {"mistral": "mistral", "gemma": "gemma"}
HUCRELER = [("mistral", "gemma"), ("qwen", "mistral"), ("qwen", "gemma"),
            ("gemma", "mistral")]
BEKLEYEN = []
YASAK_AD = ("32B", "32b", "Instruct", "instruct", "-it")


def durak():
    n = datetime.now()
    h = n.replace(hour=DURAK_SAAT, minute=0, second=0, microsecond=0)
    return h + timedelta(days=1) if n.hour >= DURAK_SAAT else h


DURAK = durak()


def dur_mu(nerede):
    if os.path.exists(BAYRAK):
        print(f"  ★ DURAK: dur-bayragi görüldü ({BAYRAK}) — {nerede} sonrasi yeni parca acilmaz.")
        return True
    if datetime.now() >= DURAK:
        print(f"  ★ DURAK: yerel {DURAK_SAAT:02d}:00 gecildi — {nerede} sonrasi yeni parca acilmaz.")
        return True
    return False


def kapi_iki_kumas(okur, yazar):
    if yazar not in YAZAR_AILE:
        raise RuntimeError(f"{yazar}")
    a = {"okuyan": OKURLAR[okur]["aile"] if okur in OKURLAR else okur, "yazan": YAZAR_AILE[yazar]}
    if len(set(a.values())) != 2:
        raise RuntimeError(f"{a}")
    payda(f"iki_kumas_{okur}_{yazar}", n_rol=len(a), n_aile=len(set(a.values())),
          bekle={"n_aile": 2})
    return a


def kapi_model(okur):
    m = OKURLAR[okur]
    for y in YASAK_AD:
        if y in m["repo"]:
            raise RuntimeError(f"YASAK MODEL ADI: {m['repo']} ({y}) — bu zincirde rol alamaz.")
    snaplar = sorted(glob.glob(f"{m['yol']}/snapshots/*"))
    agirlik = AGD.dosyalar(snaplar[-1]) if snaplar else []
    if not agirlik:
        raise RuntimeError(f"{m['yol']}"
                           f"{os.path.realpath(m['yol'])}")
    payda(f"model_{okur}", n_snapshot=len(snaplar), n_agirlik=len(agirlik),
          bekle={"n_agirlik": 1})
    return dict(repo=m["repo"], yol=m["yol"], realpath=os.path.realpath(m["yol"]),
                snapshot=os.path.basename(snaplar[-1]), n_agirlik=len(agirlik))


def on_ucus_cevresi(dev, gereken_vram_gb=25):
    import torch
    i = int(dev.split(":")[1])
    p = torch.cuda.get_device_properties(i)
    bos, top = torch.cuda.mem_get_info(i)
    df = subprocess.run(["df", "-h", "<storage>"], capture_output=True, text=True).stdout
    C = dict(
        hf_home=os.environ["HF_HOME"], hf_offline=os.environ.get("HF_HUB_OFFLINE"),
        dev=dev, gpu_adi=p.name, gpu_uuid=str(getattr(p, "uuid", "(yok)")),
        gpu_host_index=i, vram_bos_gb=round(bos / 2**30, 2), vram_top_gb=round(top / 2**30, 2),
        kapsayici="(yok — ana makine .venv)", python=sys.version.split()[0],
        torch=torch.__version__, numpy=np.__version__,
        disk=df.strip().splitlines()[-1], cikis_koku=OUT,
        damga_utc=subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                 capture_output=True, text=True).stdout.strip(),
        durak_yerel=DURAK.strftime("%Y-%m-%d %H:%M:%S"),
        payda_hash=hashlib.sha1(inspect.getsource(payda).encode()).hexdigest()[:12],
    )
    for k in ("hf_home", "gpu_uuid", "cikis_koku"):
        if not C[k] or C[k] == "(yok)":
            raise RuntimeError(f"{k}")
    C["gereken_vram_gb"] = gereken_vram_gb
    if C["vram_bos_gb"] < gereken_vram_gb:
        raise RuntimeError(f"{C['vram_bos_gb']}"
                           f"{gereken_vram_gb}"
                           f"")
    print("  [CEVRE] " + " · ".join(f"{k}={v}" for k, v in C.items() if k != "disk"))
    print(f"  [CEVRE] disk: {C['disk']}")
    return C


def satirlar(yazar):
    S = [json.loads(l) for l in open(f"{KAYNAK}/ana__{yazar}.jsonl", encoding="utf-8")]
    y_dur = np.array([1 if r["durus"] == "arti" else 0 for r in S], dtype=np.int8)
    y_cer = np.array([1 if r["isi"] == "sert" else 0 for r in S], dtype=np.int8)
    kume = np.array([r["debate"] for r in S])
    tab_dur = np.array([f'{r["debate"]}|{r["isi"]}' for r in S])
    tab_cer = np.array([f'{r["debate"]}|{r["durus"]}' for r in S])
    met = [r[PENCERE] for r in S]
    payda(f"satir_{yazar}", n_satir=len(S), n_kume=len(set(kume)),
          n_tabaka_dur=len(set(tab_dur)), n_tabaka_cer=len(set(tab_cer)),
          bekle={"n_satir": BEKLE_SATIR, "n_kume": 17, "n_tabaka_dur": 34, "n_tabaka_cer": 34})
    for ad, t in (("durus", tab_dur), ("cerceve", tab_cer)):
        d = collections.Counter(t.tolist())
        if len(set(d.values())) != 1:
            raise RuntimeError(f"TABAKA DENGESIZ ({ad}): {sorted(set(d.values()))}")
    return met, y_dur, y_cer, kume, tab_dur, tab_cer


def kodla_hucre(okur, yazar, met, dev):
    npy = f"{OUT}/H_{okur}__{yazar}.fp16.npy"
    mjs = f"{OUT}/H_{okur}__{yazar}.meta.json"
    if os.path.exists(npy) and os.path.exists(mjs):
        M = json.load(open(mjs))
        if M.get("n_satir") == len(met):
            print(f"  ↷ ATLA (tam): {npy} · {M['sekil']}")
            return npy, M
        print(f"  ⚠ YARIM checkpoint ({M.get('n_satir')}≠{len(met)}) ⇒ bastan.")
    import torch, kt_qwen25 as KT
    KT.MODEL, KT.DEV = OKURLAR[okur]["repo"], dev
    t0 = time.time()
    tok, model = KT.yukle_model()
    sayac = collections.Counter()
    ST, _OR, uz = KT.kodla(tok, model, met, bs=16, sayac=sayac, maxlen=512, dev=dev)
    L, D = int(ST.shape[1]), int(ST.shape[2])
    np.save(npy, ST)
    M = dict(okur=okur, yazar=yazar, n_satir=len(met), sekil=list(ST.shape), L=L, D=D,
             jeton_medyan=int(np.median(uz)), jeton_maks=int(np.max(uz)),
             maxlen_carpan=int(sum(1 for u in uz if u >= 512)),
             gecerli_token=int(sayac["gecerli_token"]), bos_havuz=int(sayac["bos_havuz"]),
             ucuncu_basamak=int(sayac["ucuncu_basamak"]), saniye=round(time.time() - t0, 1),
             havuz="sontoken", model=OKURLAR[okur]["repo"])
    json.dump(M, open(mjs, "w"), ensure_ascii=False, indent=1)
    payda(f"kodla_{okur}_{yazar}", n_satir=len(met), n_katman=L, n_boyut=D,
          n_gecerli_token=int(sayac["gecerli_token"]), red_bos_havuz=int(sayac["bos_havuz"]),
          bekle={"n_satir": len(met), "n_katman": 2})
    print(f"  {okur}←{yazar}: {ST.shape} · jeton med {M['jeton_medyan']} maks {M['jeton_maks']} "
          f"· 512'ye carpan {M['maxlen_carpan']} · {M['saniye']}s", flush=True)
    del model, ST, _OR
    torch.cuda.empty_cache()
    return npy, M


def auc_ve_null(X, y, kume, tabaka, dev, K):
    fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
    C = Cekirdek(X, fold, dev=dev, std=True)
    A = C.oku_toplu(y[None, :])
    auc = float(C.auc_toplu(y[None, :], A)[0])
    skor = A.cpu().numpy()[0]
    Yp = Permutator(y, tabaka.tolist(), seed=SEED).cek(K)
    nulls = C.auc_toplu(Yp, C.oku_toplu(Yp))
    C.bosalt()
    nm, ns = float(nulls.mean()), float(nulls.std(ddof=1))
    return dict(auc=auc, null_merkez=nm, null_sd=ns, z=(auc - nm) / ns,
                mde=1.645 * ns, K=int(K)), skor


def _yon(Z, y):
    u = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    return u / max(np.linalg.norm(u), 1e-12)


def aci_profili(X, y_dur, y_cer, tab_dur, K):
    Z = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
    u_c = _yon(Z, y_cer)
    ger = abs(float(_yon(Z, y_dur) @ u_c))
    Yp = Permutator(y_dur, tab_dur.tolist(), seed=SEED).cek(K)
    null = np.array([abs(float(_yon(Z, Yp[k]) @ u_c)) for k in range(K)])
    return dict(cos_abs=ger, null_merkez=float(null.mean()), null_sd=float(null.std(ddof=1)),
                z=float((ger - null.mean()) / max(null.std(ddof=1), 1e-12)), K=int(K))


def paralelkenar(X, y_dur, y_cer, kume, K, rng):
    Z = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
    kum = sorted(set(kume.tolist()))

    def uclu(yd, yc):
        I, F, S = [], [], []
        for d in kum:
            m = kume == d
            def ort(s, f):
                ix = m & (yd == s) & (yc == f)
                return Z[ix].mean(0)
            a11, a10, a01, a00 = ort(1, 1), ort(1, 0), ort(0, 1), ort(0, 0)
            I.append((a11 - a10) - (a01 - a00))
            F.append(0.5 * ((a11 - a10) + (a01 - a00)))
            S.append(0.5 * ((a11 - a01) + (a10 - a00)))
        return (np.mean(I, 0), np.mean(F, 0), np.mean(S, 0))

    Ib, Fb, Sb = uclu(y_dur, y_cer)
    nI = float(np.linalg.norm(Ib)); nF = float(np.linalg.norm(Fb)); nS = float(np.linalg.norm(Sb))
    nul = {"I": [], "F": [], "S": []}
    for _ in range(K):
        yd, yc = y_dur.copy(), y_cer.copy()
        for d in kum:
            ix = np.where(kume == d)[0]
            p = rng.permutation(len(ix))
            yd[ix], yc[ix] = y_dur[ix][p], y_cer[ix][p]
        a, b, c = uclu(yd, yc)
        nul["I"].append(np.linalg.norm(a)); nul["F"].append(np.linalg.norm(b))
        nul["S"].append(np.linalg.norm(c))
    out = dict(n_kume=len(kum), K=int(K), norm_I=nI, norm_F=nF, norm_S=nS,
               r_cerceve=nI / max(nF, 1e-12), r_durus=nI / max(nS, 1e-12))
    for ad, v in (("I", nI), ("F", nF), ("S", nS)):
        a = np.array(nul[ad])
        out[f"null_{ad}_merkez"] = float(a.mean()); out[f"null_{ad}_sd"] = float(a.std(ddof=1))
        out[f"z_{ad}"] = float((v - a.mean()) / max(a.std(ddof=1), 1e-12))
    out["r_cerceve_null_duzeltmeli"] = ((nI - out["null_I_merkez"]) /
                                        max(nF - out["null_F_merkez"], 1e-12))
    return out


def egit_prob(X, y, fold, dev, gizli, seed):
    import torch
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    ft = torch.as_tensor(fold, dtype=torch.long, device=dev)
    skor = torch.zeros(len(y), device=dev)
    D = Xt.shape[1]
    for j in sorted(set(fold.tolist())):
        tr = ft != j; te = ~tr
        mu = Xt[tr].mean(0); sd = Xt[tr].std(0, unbiased=True).clamp_min(1e-6)
        Ztr, Zte, ytr = (Xt[tr] - mu) / sd, (Xt[te] - mu) / sd, yt[tr]
        torch.manual_seed(seed + 977 * j)
        net = (torch.nn.Sequential(torch.nn.Linear(D, gizli), torch.nn.ReLU(),
                                   torch.nn.Linear(gizli, 1)) if gizli
               else torch.nn.Linear(D, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        for _ in range(EPOK):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                net(Ztr).squeeze(-1), ytr).backward()
            opt.step()
        with torch.no_grad():
            skor[te] = net(Zte).squeeze(-1)
    return auc_np(y, skor.cpu().numpy())


def tfidf(metinler, min_df=3):
    dok = [KELIME.findall(t.lower()) for t in metinler]
    df = collections.Counter()
    for d in dok:
        df.update(set(d))
    kelime = sorted(w for w, c in df.items() if c >= min_df)
    ix = {w: i for i, w in enumerate(kelime)}
    n = len(dok)
    M = np.zeros((n, len(kelime)), dtype=np.float32)
    for i, d in enumerate(dok):
        c = collections.Counter(d)
        for w, v in c.items():
            if w in ix:
                M[i, ix[w]] = 1.0 + np.log(v)
    idf = np.log((1 + n) / (1 + np.array([df[w] for w in kelime], dtype=np.float32))) + 1.0
    M *= idf
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    return M, kelime


def spearman(a, b):
    def r(v):
        o = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.float64)
        return (o - o.mean()) / max(o.std(), 1e-12)
    return float((r(a) * r(b)).mean())


def analiz_hucre(okur, yazar, V, dev, capa=None):
    npy = f"{OUT}/H_{okur}__{yazar}.fp16.npy"
    jsn = f"{OUT}/A_{okur}__{yazar}.json"
    if os.path.exists(jsn):
        R = json.load(open(jsn))
        if R.get("tam"):
            print(f"  ↷ ATLA (tam analiz): {jsn}")
            return R
    met, y_dur, y_cer, kume, tab_dur, tab_cer = V
    H = np.load(npy, mmap_mode="r")
    L = H.shape[1]
    fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
    rng = np.random.default_rng(SEED)

    T, kelime = tfidf(met)
    t_dur, s_tdur = auc_ve_null(T, y_dur, kume, tab_dur, dev, K_NULL)
    t_cer, s_tcer = auc_ve_null(T, y_cer, kume, tab_cer, dev, K_NULL)
    payda(f"tfidf_{okur}_{yazar}", n_kelime=len(kelime), n_satir=T.shape[0],
          bekle={"n_kelime": 50, "n_satir": BEKLE_SATIR})
    print(f"  [YÜZEY] TF-IDF sözlük {len(kelime)} · durus-AUC {t_dur['auc']:.4f} "
          f"(z {t_dur['z']:+.2f}) · cerceve-AUC {t_cer['auc']:.4f} (z {t_cer['z']:+.2f})")

    orta = L // 2
    Xo = np.asarray(H[:, orta, :], dtype=np.float32)
    y_perm = Permutator(y_dur, tab_dur.tolist(), seed=SEED).cek(1)[0].astype(np.int8)
    prova = dict(katman=orta,
                 dogrusal=egit_prob(Xo, y_perm, fold, dev, 0, SEED),
                 mlp=egit_prob(Xo, y_perm, fold, dev, GIZLI, SEED))
    print(f"  [PROVA §7.1] ℓ{orta} permüte etiket → dogrusal {prova['dogrusal']:.4f} · "
          f"MLP {prova['mlp']:.4f}  (0,5 civari beklenir)")
    if not all(0.35 <= v <= 0.65 for v in (prova["dogrusal"], prova["mlp"])):
        raise RuntimeError(f"PROVA DÜSTÜ: permüte etiketle prob 0,5'ten uzak {prova} "
                           f"⇒ tasarim dejenere, profil okunmaz.")

    kat = []
    for l in range(L):
        X = np.asarray(H[:, l, :], dtype=np.float32)
        d, s_d = auc_ve_null(X, y_dur, kume, tab_dur, dev, K_NULL)
        c, _s_c = auc_ve_null(X, y_cer, kume, tab_cer, dev, K_NULL)
        a = aci_profili(X, y_dur, y_cer, tab_dur, K_ACI)
        p = paralelkenar(X, y_dur, y_cer, kume, K_INT, np.random.default_rng(SEED + l))
        lin = egit_prob(X, y_dur, fold, dev, 0, SEED + l)
        mlp = egit_prob(X, y_dur, fold, dev, GIZLI, SEED + l)
        kat.append(dict(l=l, durus=d, cerceve=c, aci=a, paralelkenar=p,
                        prob_dogrusal=lin, prob_mlp=mlp, prob_delta=mlp - lin,
                        yuzey_bag_durus=spearman(s_d, s_tdur)))
        print(f"    ℓ{l:>2} durAUC {d['auc']:.4f} (z{d['z']:+6.2f}) · cerAUC {c['auc']:.4f} "
              f"(z{c['z']:+6.2f}) · |cos| {a['cos_abs']:.3f} (z{a['z']:+6.2f}) · "
              f"lin {lin:.4f} mlp {mlp:.4f} Δ{mlp-lin:+.4f} · z_I {p['z_I']:+6.2f} "
              f"r_cer {p['r_cerceve']:.3f} · yüzeyρ {kat[-1]['yuzey_bag_durus']:+.3f}",
              flush=True)
        if dur_mu(f"ℓ{l}"):
            R = dict(okur=okur, yazar=yazar, L=L, tam=False, kesildi_l=l, katman=kat,
                     tfidf=dict(n_kelime=len(kelime), durus=t_dur, cerceve=t_cer), prova=prova)
            json.dump(R, open(jsn, "w"), ensure_ascii=False, indent=1)
            return R

    R = dict(okur=okur, yazar=yazar, L=L, tam=True, katman=kat, prova=prova,
             tfidf=dict(n_kelime=len(kelime), durus=t_dur, cerceve=t_cer))
    if capa is not None:
        ger = kat[capa["l"]]["durus"]["auc"]
        d = abs(ger - capa["auc"])
        print(f"  [CAPA] ℓ{capa['l']} durus-AUC {ger:.6f} ↔ mühürlü kosu {capa['auc']:.6f} "
              f"· |Δ| {d:.2e} (bar 1e-4) · {'GECTI ✓' if d <= 1e-4 else 'DÜSTÜ ✗'}")
        R["capa"] = dict(l=capa["l"], bizim=ger, muhurlu=capa["auc"], delta=d, gecti=bool(d <= 1e-4))
        if d > 1e-4:
            raise RuntimeError(f"CAPA DÜSTÜ: |Δ|={d:.2e} — kodlama tarifi mühürlü kosudan KAYMIS.")
    payda(f"analiz_{okur}_{yazar}", n_katman=len(kat), bekle={"n_katman": L})
    json.dump(R, open(jsn, "w"), ensure_ascii=False, indent=1)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", choices=("kodla", "analiz", "hepsi"), default="hepsi")
    ap.add_argument("--dev", default="cuda:1")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("★ KESIF · KATMAN-TARAMASI — VERDICT YOK · PREDICTION YOK · PREREG YOK · BANKA YOK")
    print(f"★ YARIM KOSU: bu gece bekleyen hücre → {BEKLEYEN} (gemma-OKUR, GPU bütcesi)")
    print(f"★ SAATLI DURAK: {DURAK.strftime('%Y-%m-%d %H:%M:%S')} yerel\n")

    C = on_ucus_cevresi(a.dev)
    for okur, yazar in HUCRELER:
        kapi_iki_kumas(okur, yazar)
    MOD = {o: kapi_model(o) for o in sorted({h[0] for h in HUCRELER})}
    C["modeller"] = MOD
    C["hucreler"] = [f"{o}←{y}" for o, y in HUCRELER]
    C["bekleyen"] = [f"{o}←{y}" for o, y in BEKLEYEN]
    json.dump(C, open(f"{OUT}/cevre.json", "w"), ensure_ascii=False, indent=1)

    VERI = {y: satirlar(y) for y in sorted({h[1] for h in HUCRELER})}

    if a.asama in ("kodla", "hepsi"):
        print("\n─── ASAMA 1 · KODLAMA (GPU) ───", flush=True)
        for okur, yazar in HUCRELER:
            if dur_mu("kodlama"):
                break
            kodla_hucre(okur, yazar, VERI[yazar][0], a.dev)

    if a.asama in ("analiz", "hepsi"):
        print("\n─── ASAMA 2 · KATMAN ANALIZI ───", flush=True)
        CAP = json.load(open(CAPA_JSON))["hucreler"]["mistral|qwen_sontoken_L22"]["auc"]
        for okur, yazar in HUCRELER:
            if not os.path.exists(f"{OUT}/H_{okur}__{yazar}.fp16.npy"):
                print(f"{okur} {yazar}")
                continue
            print(f"\n═══ {okur}-OKUR × {yazar}-METIN ═══", flush=True)
            capa = dict(l=22, auc=CAP) if (okur, yazar) == ("qwen", "mistral") else None
            analiz_hucre(okur, yazar, VERI[yazar], a.dev, capa)
            if dur_mu("hücre"):
                break

    kay = dict(damga_utc=subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                        capture_output=True, text=True).stdout.strip(),
               durak_hedef=DURAK.strftime("%Y-%m-%d %H:%M:%S"),
               durak_atesledi=bool(datetime.now() >= DURAK or os.path.exists(BAYRAK)),
               oldurulen_surec=0, docker_dokunma=0, gpu=a.dev)
    json.dump(kay, open(f"{OUT}/durak_kaydi.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n→ {OUT}/  ·  durak_kaydi: {kay}")


if __name__ == "__main__":
    main()
