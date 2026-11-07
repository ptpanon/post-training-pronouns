#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, hashlib, argparse
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
from pergel_cekirdek import Cekirdek, Permutator, foldlar
from prefix_korpus_pilot import KELIME
import prefix_layer_taramasi as KT

KAYNAK = __DNH_DATA__ + "/onek_korpus/muhurlu"
OUT = __DNH_DATA__ + "/onek_korpus/taban_rejimi"
CIKTI = f"{ROOT}/unreleased/TABAN_REJIMI_2026-08-07.json"
ZEMINLER = [(y, k) for y in ("mistral", "gemma") for k in ("ana", "plasebo")]
SEED_KARISTIRMA = 20260807
K_NULL, BANT = 1000, 1.645
ADLAR_BELGE = {"YALNIZ-TABAKA-ICI-NITELIKLI", "YALNIZ-METIN-ICI-NITELIKLI",
               "IKISI-DE-NITELIKLI", "HICBIRI-NITELIKLI-DEGIL", "ÖLCÜLEMEZ-taban"}


def verdict_taban(A_nitelikli, B_nitelikli, olculdu):
    if not olculdu:
        return "ÖLCÜLEMEZ-taban"
    if A_nitelikli and B_nitelikli:
        return "IKISI-DE-NITELIKLI"
    if B_nitelikli:
        return "YALNIZ-TABAKA-ICI-NITELIKLI"
    if A_nitelikli:
        return "YALNIZ-METIN-ICI-NITELIKLI"
    return "HICBIRI-NITELIKLI-DEGIL"


def adlari_say():
    return {verdict_taban(a, b, o) for a in (True, False) for b in (True, False)
            for o in (True, False)}


def verdict_dogum(z_gercek, z_taban, sd_esli):
    g, t, s = (np.asarray(x, float) for x in (z_gercek, z_taban, sd_esli))
    if not (np.isfinite(g).all() and np.isfinite(t).all() and np.isfinite(s).all()):
        return "ÖLCÜLEMEZ-katman", None
    if np.all(t >= BANT):
        return "TABAN-ZATEN-ÜSTTE", None
    d = g - t
    for l in range(len(d)):
        if np.all(d[l:] >= BANT * s[l:]):
            return ("BÜYÜME-ERKEN-L0-8" if l <= 8 else
                    "BÜYÜME-ORTA-L9-16" if l <= 16 else "BÜYÜME-GEC-L17-24"), l
    return "BÜYÜME-YOK", None


def dejenerelik_provasi():
    rng = np.random.default_rng(SEED_KARISTIRMA)
    sav = {}
    z = rng.normal(1.0, 0.5, 25)
    z = np.clip(z, -1.0, 1.4)
    sd = np.full(25, 0.30)
    sav["A_gercek_esittir_taban"] = dict(verdict=verdict_dogum(z, z, sd)[0], bekle="BÜYÜME-YOK")
    sav["B_taban_zaten_ustte"] = dict(verdict=verdict_dogum(z + 5, np.full(25, 2.0), sd)[0],
                                      bekle="TABAN-ZATEN-ÜSTTE")
    g = z.copy(); g[12:] += 3.0
    sav["C_gec_buyume"] = dict(verdict=verdict_dogum(g, z, sd)[0], bekle="BÜYÜME-ORTA-L9-16")
    sav["D_nan"] = dict(verdict=verdict_dogum(z * np.nan, z, sd)[0], bekle="ÖLCÜLEMEZ-katman")
    for k, v in sav.items():
        v["gecti"] = bool(v["verdict"] == v["bekle"])
    return sav


def satirlar(yazar, kol):
    S = [json.loads(l) for l in open(f"{KAYNAK}/{kol}__{yazar}.jsonl", encoding="utf-8")]
    return ([x[KT.PENCERE] for x in S],
            np.array([1 if x["durus"] == "arti" else 0 for x in S], dtype=np.int8),
            np.array([x["debate"] for x in S]),
            np.array([f'{x["debate"]}|{x["isi"]}' for x in S]))


def karistir(met, tab, variant, seed):
    rng = np.random.default_rng(seed)
    jet = [KELIME.findall(t.lower()) for t in met]
    out = [None] * len(met)
    if variant == "A_metin_ici":
        for i, w in enumerate(jet):
            w = list(w); rng.shuffle(w); out[i] = " ".join(w)
        return out
    for t in np.unique(tab):
        idx = np.where(tab == t)[0]
        havuz = [w for i in idx for w in jet[i]]
        rng.shuffle(havuz)
        p = 0
        for i in idx:
            n = len(jet[i])
            out[i] = " ".join(havuz[p:p + n]); p += n
        assert p == len(havuz)
    return out


def kodla_tam(met, dev):
    import torch, anchor_kodlama as CK
    from transformers import AutoTokenizer, AutoModel
    CK.DEV = dev
    K = CK.KOLLAR["c1a"]
    tok = AutoTokenizer.from_pretrained(K["model"])
    model = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(dev).eval()
    sayac = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0, uzunluklar=[])
    eski = (K["kes"], K["maxlen"]); K["kes"], K["maxlen"] = 4000, 512
    try:
        H = CK.kodla(tok, model, met, "c1a", bs=64, sayac=sayac)
    finally:
        K["kes"], K["maxlen"] = eski
    del model; torch.cuda.empty_cache()
    return H, sayac


def oku_l21(H, y, fold, Yp, dev):
    X = np.asarray(H[:, 21, :], dtype=np.float32)
    C = Cekirdek(X, fold, dev=dev, std=True)
    A = float(C.auc_toplu(y[None, :], C.oku_toplu(y[None, :]))[0])
    n = C.auc_toplu(Yp, C.oku_toplu(Yp))
    n = n.cpu().numpy() if hasattr(n, "cpu") else np.asarray(n)
    C.bosalt()
    m, s = float(n.mean()), float(n.std(ddof=1))
    return dict(A=A, null_merkez=m, null_sd=s, z=float((A - m) / max(s, 1e-12)), MDE=BANT * s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0"); ap.add_argument("--bekle-gpu", type=int, default=1)
    a = ap.parse_args()
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    kod = adlari_say()
    print(f"TABAN REJIMI · damga {damga} · prediction e0984dc · seed {SEED_KARISTIRMA}")
    print(f"[AD CAPRAZ SAYIMI · D44] kod {len(kod)} · belge {len(ADLAR_BELGE)} ⇒ "
          f"{'ESIT ✓' if kod == ADLAR_BELGE else 'ESIT DEGIL ✗'}")
    if kod != ADLAR_BELGE:
        raise RuntimeError("AD CAPRAZ SAYIMI DÜSTÜ ⇒ kosu durdu.")

    sav = dejenerelik_provasi()
    print("★ DEJENERELIK PROVASI (dogum kurali — pahali isin ÖNÜNDE):")
    for k, v in sav.items():
        print(f"   {k:24s} → {v['verdict']:20s} (bekle {v['bekle']}) "
              f"{'✓' if v['gecti'] else '✗'}")
    if not all(v["gecti"] for v in sav.values()):
        raise RuntimeError("")
    print("")

    C_gpu = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, etiket="taban_rejimi")
    os.makedirs(OUT, exist_ok=True)
    R = dict(damga=damga, prediction="e0984dc", seed=SEED_KARISTIRMA, K_null=K_NULL,
             gpu=C_gpu, dejenerelik=sav, zeminler={}, manifest={})
    nitelik = {"A_metin_ici": [], "B_tabaka_ici": []}
    for yazar, kol in ZEMINLER:
        ad = f"{yazar}__{kol}"
        met, y, kume, tab = satirlar(yazar, kol)
        fold = foldlar(kume.tolist(), k=KT.NFOLD, seed=KT.SEED)
        Yp = Permutator(y, tab.tolist(), seed=KT.SEED).cek(K_NULL)
        R["zeminler"][ad] = {}
        print(f"\n─── {ad} (n={len(y)}) ───", flush=True)
        for variant in ("A_metin_ici", "B_tabaka_ici"):
            kar = karistir(met, tab, variant, SEED_KARISTIRMA)
            npy = f"{OUT}/E5_TUM__{variant}__{ad}.fp16.npy"
            if os.path.exists(npy):
                H = np.load(npy, mmap_mode="r"); sn = 0.0; sayac = dict(bos_havuz=0)
            else:
                t0 = time.time(); H, sayac = kodla_tam(kar, a.dev); sn = time.time() - t0
                np.save(npy, np.asarray(H, dtype=np.float16))
            r = oku_l21(H, y, fold, Yp, a.dev)
            sha = hashlib.sha256(open(npy, "rb").read()).hexdigest()
            nit = bool(abs(r["z"]) < BANT)
            nitelik[variant].append(nit)
            R["zeminler"][ad][variant] = dict(r, nitelikli=nit, sekil=list(np.shape(H)))
            R["manifest"][f"{variant}__{ad}"] = dict(yol=os.path.basename(npy), sha256=sha,
                                                     sekil=list(np.shape(H)), saniye=round(sn, 1))
            t_ozgun = sorted([w for t in met for w in KELIME.findall(t.lower())])
            t_kar = sorted([w for t in kar for w in KELIME.findall(t.lower())])
            R["manifest"][f"{variant}__{ad}"]["havuz_ozdes"] = bool(t_ozgun == t_kar)
            print(f"  {variant:14s} AUC {r['A']:.4f} · null {r['null_merkez']:.4f}±"
                  f"{r['null_sd']:.4f} ⇒ **z {r['z']:+.2f}** · MDE {r['MDE']:.4f} ⇒ "
                  f"{'NITELIKLI ✓' if nit else 'NITELIKSIZ ✗'} · sha {sha[:16]} · "
                  f"havuz özdes {R['manifest'][f'{variant}__{ad}']['havuz_ozdes']}", flush=True)
            del H

    A_ok = all(nitelik["A_metin_ici"]); B_ok = all(nitelik["B_tabaka_ici"])
    R["nitelik"] = dict(A_metin_ici=A_ok, B_tabaka_ici=B_ok,
                        A_asan_zemin=int(sum(1 for x in nitelik["A_metin_ici"] if not x)),
                        B_asan_zemin=int(sum(1 for x in nitelik["B_tabaka_ici"] if not x)))
    R["verdict"] = verdict_taban(A_ok, B_ok, True)
    payda("taban_rejimi", n_zemin=len(ZEMINLER), n_variant=2, n_dosya=len(R["manifest"]),
          n_sav=len(sav), n_ad=len(kod),
          bekle={"n_zemin": 4, "n_variant": 2, "n_dosya": 8, "n_sav": 4, "n_ad": 5})
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    print(f"\n★★ PREDICTION HÜKMÜ (ön-kayitli rule, `e0984dc` §1): **{R['verdict']}**")
    print(f"   A metin-ici: {R['nitelik']['A_asan_zemin']}/4 zeminde bandi ASTI · "
          f"B tabaka-ici: {R['nitelik']['B_asan_zemin']}/4 asti")
    print(f"→ {CIKTI}")


if __name__ == "__main__":
    main()
