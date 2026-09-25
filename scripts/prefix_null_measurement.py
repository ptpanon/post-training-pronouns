#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, collections
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from pergel_cekirdek import Cekirdek, Permutator, foldlar
from pergel_task1_ablasyon import ref_okumasi

PIL = __DNH_DATA__ + "/onek_korpus/uretim_pilot"
OUT = __DNH_DATA__ + "/onek_korpus/null_olcum"
YAZARLAR = ("mistral", "gemma")
PENCERE = "metin_160"
K_ANA, K_SWEEP = 5000, 800
SEED, NFOLD = 20260805, 5
KUME_IZGARA = (5, 8, 11, 14, 17)
TEKRAR = 8
MDE_KAT = 1.645

_PREREG_ONCESI = True

ROL_AILE = {"yazan": {"mistral": "mistral", "gemma": "gemma"},
            "okuyan": {"qwen_sontoken_L22": "qwen", "e5_L23": "e5"},
            "yargilayan": {"claude": "claude"}}


def kapi_uc_kumas(yazar, okuyucu):
    a = {"yazan": ROL_AILE["yazan"][yazar], "okuyan": ROL_AILE["okuyan"][okuyucu],
         "yargilayan": "claude"}
    if len(set(a.values())) != 3:
        raise RuntimeError(f"{a}")
    payda("uc_kumas", n_rol=len(a), n_aile=len(set(a.values())), bekle={"n_aile": 3})
    return a


def gercek_okuma_yasak(*_a, **_k):
    raise RuntimeError("PREREG ÖNCESI GERCEK-ETIKET OKUMASI YASAK (hesap baslamadi beyani).")


def satirlar(yazar, kok=None, etiket="ana", bekle_satir=816):
    S = [json.loads(l) for l in open(f"{kok or PIL}/{etiket}__{yazar}.jsonl", encoding="utf-8")]
    y = np.array([1 if r["durus"] == "arti" else 0 for r in S], dtype=np.int8)
    kume = np.array([r["debate"] for r in S])
    tabaka = np.array([f'{r["debate"]}|{r["isi"]}' for r in S])
    met = [r[PENCERE] for r in S]
    payda(f"satir_{yazar}", n_satir=len(S), n_kume=len(set(kume)), n_tabaka=len(set(tabaka)),
          bekle={"n_satir": bekle_satir, "n_kume": 17, "n_tabaka": 34})
    d = collections.Counter(tabaka)
    if len(set(d.values())) != 1:
        raise RuntimeError(f"TABAKA DENGESIZ: {sorted(set(d.values()))} — permütasyon tabani bozulur.")
    return met, y, kume, tabaka


def kodla_qwen(metin_map, dev):
    import kt_qwen25 as KT
    KT.DEV = dev
    tok, model = KT.yukle_model()
    out, sayac, n_sat = {}, collections.Counter(), 0
    for ad, met in metin_map.items():
        ST, _OR, uz = KT.kodla(tok, model, met, bs=16, sayac=sayac, maxlen=512, dev=dev)
        out[ad] = np.asarray(ST[:, 22, :], dtype=np.float32)
        n_sat += len(met)
        print(f"  qwen/{ad}: {ST.shape} → L22 {out[ad].shape} · jeton medyan {int(np.median(uz))}",
              flush=True)
    payda("kodla_qwen", n_satir=n_sat, n_kol=len(out),
          n_gecerli_token=int(sayac["gecerli_token"]),
          red_bos_havuz=int(sayac["bos_havuz"]),
          red_ucuncu_basamak=int(sayac["ucuncu_basamak"]),
          bekle={"n_satir": sum(len(v) for v in metin_map.values()),
                 "n_kol": len(metin_map)})
    del model
    import torch; torch.cuda.empty_cache()
    return out


def kodla_e5(metin_map, dev):
    import torch, anchor_kodlama as CK
    from transformers import AutoTokenizer, AutoModel
    CK.DEV = dev
    K = CK.KOLLAR["c1a"]
    uzunluklar = [len(t) for met in metin_map.values() for t in met]
    baglayici = sum(1 for u in uzunluklar if u > K["kes"])
    print(f"  [§7.4] e5 kanonik kes={K['kes']} karakter · pencere uzunlugu medyan "
          f"{int(np.median(uzunluklar))} · kesmeye CARPAN {baglayici}/{len(uzunluklar)} "
          f"({baglayici/len(uzunluklar):.1%})", flush=True)
    kes_yeni = 4000
    tok = AutoTokenizer.from_pretrained(K["model"])
    model = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(dev).eval()
    sayac = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0, uzunluklar=[])
    out = {}
    eski_kes, eski_maxlen = K["kes"], K["maxlen"]
    K["kes"], K["maxlen"] = kes_yeni, 512
    try:
        for ad, met in metin_map.items():
            H = CK.kodla(tok, model, met, "c1a", bs=64, sayac=sayac)
            out[ad] = np.asarray(H[:, 23, :], dtype=np.float32)
            print(f"  e5/{ad}: {H.shape} → L23 {out[ad].shape}", flush=True)
    finally:
        K["kes"], K["maxlen"] = eski_kes, eski_maxlen
    payda("kodla_e5", n_satir=int(sayac["satir"]), n_kol=len(out),
          red_maxlen_carpan=int(sayac["maxlen_carpan"]), red_bos_havuz=int(sayac["bos_havuz"]))
    del model; torch.cuda.empty_cache()
    return out, dict(kes_kanonik=eski_kes, kes_kullanilan=kes_yeni,
                     maxlen_kanonik=eski_maxlen, maxlen_kullanilan=512,
                     kanonik_kesmeye_carpan=baglayici, n_metin=len(uzunluklar),
                     uzunluk_medyan=int(np.median(uzunluklar)))


def null_dagilimi(X, kume, tabaka, dev, K, seed, std=True):
    fold = foldlar(kume.tolist(), k=NFOLD, seed=seed)
    C = Cekirdek(X, fold, dev=dev, std=std)
    y0 = np.zeros(len(kume), dtype=np.int8)
    n_tab = len(set(tabaka.tolist()))
    for t in sorted(set(tabaka.tolist())):
        ix = np.where(tabaka == t)[0]
        y0[ix[: len(ix) // 2]] = 1
    P = Permutator(y0, tabaka.tolist(), seed=seed)
    Yp = P.cek(K)
    A = C.oku_toplu(Yp)
    auc = C.auc_toplu(Yp, A)
    kirp = C.kirpilan_boyut
    C.bosalt()
    return dict(K=K, n_tabaka=n_tab, merkez=float(auc.mean()), sd=float(auc.std(ddof=1)),
                p95=float(np.percentile(auc, 95)), p05=float(np.percentile(auc, 5)),
                carpiklik=float(((auc - auc.mean()) ** 3).mean() / auc.std() ** 3),
                mde=float(MDE_KAT * auc.std(ddof=1)), kirpilan_boyut=int(kirp)), Yp, C


def regresyon_kapisi(X, kume, Yp, dev, std=True):
    fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
    C = Cekirdek(X, fold, dev=dev, std=std)
    y = Yp[0].astype(np.int64)
    A_hizli = C.oku_toplu(Yp[:1]).cpu().numpy()[0]
    C.bosalt()
    A_ref = ref_okumasi(X.astype(np.float64), y, fold, std)
    d = float(np.abs(A_hizli - A_ref).max())
    print(f"  [REGRESYON] maks|Δ| = {d:.3e} (bar 1e-4) · "
          f"{'GECTI ✓' if d <= 1e-4 else 'DÜSTÜ ✗'}")
    if d > 1e-4:
        raise RuntimeError(f"{d:.3e}")
    return d


def olcekleme_ussu(X, kume, tabaka, dev, rng):
    kumeler = sorted(set(kume.tolist()))
    nokta = []
    for m in KUME_IZGARA:
        sds = []
        R = 1 if m == len(kumeler) else TEKRAR
        for r in range(R):
            sec = kumeler if m == len(kumeler) else list(
                rng.choice(kumeler, size=m, replace=False))
            ix = np.where(np.isin(kume, sec))[0]
            if len(set(kume[ix].tolist())) < NFOLD:
                continue
            o, _, _ = null_dagilimi(X[ix], kume[ix], tabaka[ix], dev, K_SWEEP,
                                    SEED + 1000 * m + r)
            sds.append(o["sd"])
        if sds:
            nokta.append((m, float(np.median(sds)), len(sds)))
            print(f"    m={m:>2} küme · null-sd medyan {np.median(sds):.5f} "
                  f"({len(sds)} tekrar)", flush=True)
    lm = np.log([p[0] for p in nokta]); ls = np.log([p[1] for p in nokta])
    b, a = np.polyfit(lm, ls, 1)
    r2 = 1 - ((ls - (a + b * lm)) ** 2).sum() / ((ls - ls.mean()) ** 2).sum()
    return dict(nokta=nokta, b=float(b), a=float(a), r2=float(r2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("★ HESAP BASLAMADI: gercek etiketle hicbir karar istatistigi hesaplanmayacak.\n")

    VERI = {y: satirlar(y) for y in YAZARLAR}
    met_map = {y: VERI[y][0] for y in YAZARLAR}

    print("\n─── KODLAMA · okuyan rol ───", flush=True)
    t0 = time.time()
    XQ = kodla_qwen(met_map, a.dev)
    XE, e5_not = kodla_e5(met_map, a.dev)
    print(f"  kodlama {time.time()-t0:.0f}s")

    SONUC = {"pencere": PENCERE, "K_ana": K_ANA, "K_sweep": K_SWEEP, "nfold": NFOLD,
             "mde_kat": MDE_KAT, "e5_7_4_notu": e5_not, "yazarlar": {}}
    rng = np.random.default_rng(SEED)

    for yazar in YAZARLAR:
        _met, _y, kume, tabaka = VERI[yazar]
        SONUC["yazarlar"][yazar] = {}
        for okuyucu, X in (("qwen_sontoken_L22", XQ[yazar]), ("e5_L23", XE[yazar])):
            print(f"\n═══ {yazar} × {okuyucu} ═══", flush=True)
            rol = kapi_uc_kumas(yazar, okuyucu)
            o, Yp, _C = null_dagilimi(X, kume, tabaka, a.dev, K_ANA, SEED)
            d = regresyon_kapisi(X, kume, Yp, a.dev)
            print(f"  NULL: merkez {o['merkez']:.4f} · sd {o['sd']:.5f} · p95 {o['p95']:.4f} "
                  f"· carpiklik {o['carpiklik']:+.3f} · MDE {o['mde']:.5f}")
            if not (0.49 <= o["merkez"] <= 0.51):
                print(f"  ★ BAYRAK: null merkezi 0,5'ten uzak ({o['merkez']:.4f}) — "
                      f"§6 'merkezlenmeyen null bayraklanir'")
            print("  ölcekleme üssü (bu tasarimda YENIDEN):", flush=True)
            u = olcekleme_ussu(X, kume, tabaka, a.dev, rng)
            print(f"    ⇒ b = {u['b']:+.3f} · R² = {u['r2']:.3f}   "
                  f"(base-önek hattinin −0,33'ü TASINMADI)")
            SONUC["yazarlar"][yazar][okuyucu] = dict(null=o, olcekleme=u, rol=rol,
                                                     regresyon_maks_delta=d)
    json.dump(SONUC, open(f"{OUT}/null_olcum.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n→ {OUT}/null_olcum.json")


if __name__ == "__main__":
    main()
