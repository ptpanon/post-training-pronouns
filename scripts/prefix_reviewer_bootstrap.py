#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import glob
import hashlib
import argparse
import numpy as np

ROOT = __DNH_ROOT__ + ""
os.environ.setdefault("HF_HOME", __DNH_DATA__ + "/.hf_local")
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
import prefix_zengin_gomu as ZG

OUT = __DNH_DATA__ + "/onek_korpus/hakem"
CIKTI = f"{ROOT}/unreleased/HAKEM_BOOTSTRAP_2026-08-08.json"
OKURLAR = {"c1a": 23, "c2a": None}
B, SEED, BAR, NB = 200, 20260808, 0.70, 3
ADLAR = ("KARARLI-VAR", "KARARLI-YOK", "ÖLCÜLEMEZ")


def birim(A):
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def asama_gomu(dev, bekle):
    import torch
    import anchor_kodlama as CK
    from transformers import AutoTokenizer, AutoModel
    os.makedirs(OUT, exist_ok=True)
    C = kilitle(dev, beklenen_fiziksel=bekle, tam=False, etiket="hakem_gomu")
    CK.DEV = dev
    ZV = ZG.zemin_verisi()
    mp = f"{OUT}/manifest.json"
    M = json.load(open(mp)) if os.path.exists(mp) else dict(zeminler={})
    M["gpu"], M["damga"] = C, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for okur in OKURLAR:
        K = CK.KOLLAR[okur]
        tok = AutoTokenizer.from_pretrained(K["model"])
        mdl = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(dev).eval()
        for z, (met, y, kume) in ZV.items():
            ad = f"{okur}__{z}"
            npy = f"{OUT}/H_{ad}.fp16.npy"
            if os.path.exists(npy):
                print(f"  ↷ ATLA {ad} {np.load(npy, mmap_mode='r').shape}")
                continue
            sayac = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0,
                         uzunluklar=[])
            t0 = time.time()
            H = CK.kodla(tok, mdl, met, okur, bs=64, sayac=sayac)
            sn = time.time() - t0
            np.save(npy, np.asarray(H, dtype=np.float16))
            M["zeminler"][ad] = dict(okur=okur, zemin=z, model=K["model"],
                                     sekil=list(np.asarray(H).shape), sha256=sha(npy),
                                     saniye=round(sn, 1), n=int(len(y)),
                                     n_kume=int(len(set(kume.tolist()))),
                                     bos_havuz=int(sayac["bos_havuz"]))
            payda(f"hakem_gomu_{ad}", n_satir=int(sayac["satir"]),
                  n_katman=int(np.asarray(H).shape[1]),
                  bekle={"n_satir": len(met)})
            print(f"  → {ad} {list(np.asarray(H).shape)} · sha {M['zeminler'][ad]['sha256'][:16]}"
                  f" · {sn:.1f}s", flush=True)
            del H
        del mdl
        torch.cuda.empty_cache()
    json.dump(M, open(mp, "w"), ensure_ascii=False, indent=1)
    print(f"→ {mp}")


def yonler(X, y, gruplar, rng, kip):
    from sklearn.linear_model import LogisticRegression
    import anchor_kodlama as CK
    W, kac = [], 0
    kumeler = sorted(set(gruplar.tolist()))
    ix_k = {k: np.flatnonzero(gruplar == k) for k in kumeler}
    for b in range(B):
        if kip == "kume":
            sec = rng.choice(len(kumeler), size=len(kumeler), replace=True)
            ix = np.concatenate([ix_k[kumeler[i]] for i in sec])
        else:
            ix = rng.choice(len(y), size=len(y), replace=True)
        if len(set(y[ix].tolist())) < 2:
            kac += 1
            continue
        m = LogisticRegression(**CK.KESTIRICI).fit(X[ix], y[ix])
        W.append(m.coef_[0])
    return birim(np.asarray(W, dtype=np.float64)), kac


def kararlilik(W, rng):
    n = len(W)
    p = rng.permutation(n)
    A, Bh = W[p[: n // 2]], W[p[n // 2:]]
    out = {}
    for kip, mrk in (("merkezsiz", False), ("merkezli", True)):
        ua = np.linalg.svd(A - (A.mean(0) if mrk else 0), full_matrices=False)[2][:NB]
        ub = np.linalg.svd(Bh - (Bh.mean(0) if mrk else 0), full_matrices=False)[2][:NB]
        out[kip] = [round(float(abs(np.dot(ua[i], ub[i]))), 4) for i in range(NB)]
    C = np.clip(W @ W.T, -1, 1)
    iu = np.triu_indices(n, 1)
    aci = np.degrees(np.arccos(np.abs(C[iu])))
    s = np.linalg.svd(W - W.mean(0) * 0, compute_uv=False) ** 2
    out["aci_medyan"] = round(float(np.median(aci)), 2)
    out["aci_p90"] = round(float(np.percentile(aci, 90)), 2)
    out["ort_yon_kosinus_medyan"] = round(float(np.median(np.abs(W @ birim(W.mean(0))))), 4)
    out["ilk_bilesen_payi"] = round(float(s[0] / s.sum()), 4)
    return out


def asama_hakem(dev):
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    M = json.load(open(f"{OUT}/manifest.json", encoding="utf-8"))
    ZV = ZG.zemin_verisi()
    R = dict(damga=damga, B=B, bar=BAR, adlar=list(ADLAR), seed=SEED,
             okurlar={k: ("son katman" if v is None else f"ℓ{v}") for k, v in OKURLAR.items()},
             hucreler={})
    print(f"BOOTSTRAP-HAKEM · damga {damga} · B={B} · bar={BAR} · "
          f"BIRINCIL kip: MERKEZSIZ PC1")
    for okur, kat in OKURLAR.items():
        for z, (met, y, kume) in ZV.items():
            ad = f"{okur}__{z}"
            H = np.load(f"{OUT}/H_{ad}.fp16.npy", mmap_mode="r")
            l = H.shape[1] - 1 if kat is None else kat
            X = np.asarray(H[:, l, :], dtype=np.float32)
            X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
            hc = dict(katman=int(l), d=int(X.shape[1]), n=int(len(y)),
                      n_kume=int(len(set(kume.tolist()))))
            for kip in ("kume", "satir"):
                rng = np.random.default_rng(SEED)
                W, kac = yonler(X, y, kume, rng, kip)
                hc[kip] = dict(kararlilik(W, np.random.default_rng(SEED + 1)),
                               n_yon=int(len(W)), atlanan=int(kac))
            hc["gecti_birincil"] = bool(hc["kume"]["merkezsiz"][0] >= BAR)
            R["hucreler"][ad] = hc
            print(f"  [{ad:10s}] ℓ{l} d={X.shape[1]} n={len(y)} küme={hc['n_kume']}")
            for kip in ("kume", "satir"):
                v = hc[kip]
                print(f"      {kip:6s} · merkezsiz PC1 {v['merkezsiz'][0]:.3f} "
                      f"(PC2/3 {v['merkezsiz'][1]:.2f}/{v['merkezsiz'][2]:.2f}) · "
                      f"merkezli PC1 {v['merkezli'][0]:.3f} · aci medyan "
                      f"{v['aci_medyan']:.1f}° · ort-yön kosinüs {v['ort_yon_kosinus_medyan']:.3f}"
                      f" · PC1 payi {v['ilk_bilesen_payi']:.3f}", flush=True)
    gecen = [a for a, v in R["hucreler"].items() if v["gecti_birincil"]]
    R["gecen_hucreler"] = gecen
    R["verdict"] = ("ÖLCÜLEMEZ" if not R["hucreler"] else
                  "KARARLI-VAR" if gecen else "KARARLI-YOK")
    payda("hakem_bootstrap", n_hucre=len(R["hucreler"]), B=B, n_okur=len(OKURLAR),
          bekle={"n_hucre": 4, "B": 200, "n_okur": 2})
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    print(f"\n★ HAKEM HÜKMÜ (bar {BAR}, küme-bootstrap, merkezsiz PC1): **{R['verdict']}**"
          f"  · gecen hücreler: {gecen or '—'}")
    print(f"→ {CIKTI}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", required=True, choices=("gomu", "hakem"))
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=0)
    a = ap.parse_args()
    (asama_gomu(a.dev, a.bekle_gpu) if a.asama == "gomu" else asama_hakem(a.dev))
