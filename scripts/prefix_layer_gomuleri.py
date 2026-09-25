#!/usr/bin/env python3
"""Infrastructure: embeddings from all e5 layers on four grounds (derived computation, no decision).
"""
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, hashlib, argparse
import numpy as np

ROOT = __DNH_ROOT__ + ""
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import prefix_layer_taramasi as KT

KAYNAK = __DNH_DATA__ + "/onek_korpus/muhurlu"
KOLLAR_DIR = __DNH_DATA__ + "/onek_korpus/kollar"
OUT = __DNH_DATA__ + "/onek_korpus/katman_e5"
ZEMINLER = [(y, k) for y in ("mistral", "gemma") for k in ("ana", "plasebo")]
CAPA_BAR = 0.9995
DEV = os.environ.get("DEV", "cuda:0")


def metinler(yazar, kol):
    S = [json.loads(l) for l in open(f"{KAYNAK}/{kol}__{yazar}.jsonl", encoding="utf-8")]
    return [x[KT.PENCERE] for x in S]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def kodla_zemin(yazar, kol, met, motor="c1a", bs=64):
    import torch, anchor_kodlama as CK
    from transformers import AutoTokenizer, AutoModel
    CK.DEV = DEV
    K = CK.KOLLAR[motor]
    tok = AutoTokenizer.from_pretrained(K["model"])
    model = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(DEV).eval()
    sayac = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0, uzunluklar=[])
    eski = (K["kes"], K["maxlen"]); K["kes"], K["maxlen"] = 4000, 512
    try:
        t0 = time.time()
        H = CK.kodla(tok, model, met, motor, bs=bs, sayac=sayac)
        sn = time.time() - t0
    finally:
        K["kes"], K["maxlen"] = eski
    del model
    torch.cuda.empty_cache()
    return H, sayac, sn


def capa(yazar, kol, H):
    p = f"{KOLLAR_DIR}/E5_L21__{kol}.fp16.npy"
    if yazar != "mistral" or not os.path.exists(p):
        return dict(kurulabilir=False, sebep=f"{yazar}")
    A = np.asarray(H[:, 21, :], dtype=np.float32)
    B = np.load(p).astype(np.float32)
    an = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    d = (an * bn).sum(1)
    return dict(kurulabilir=True, n=int(len(d)), kosegen_min=float(d.min()),
                kosegen_ort=float(d.mean()), bar=CAPA_BAR, gecti=bool(d.min() >= CAPA_BAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zemin", default="hepsi", help="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sec = ZEMINLER if a.zemin == "hepsi" else [tuple(a.zemin.split(":"))]
    print(f"{damga} {len(sec)} {DEV}")
    mp = f"{OUT}/manifest.json"
    M = json.load(open(mp)) if os.path.exists(mp) else dict(damga=damga, zeminler={})
    M["damga_son"] = damga
    M["tarif"] = (""
                  "")
    n_capa, n_gecti = 0, 0
    for yazar, kol in sec:
        ad = f"{yazar}__{kol}"
        npy = f"{OUT}/E5_TUM__{ad}.fp16.npy"
        met = metinler(yazar, kol)
        if os.path.exists(npy):
            print(f"  ↷ ATLA {ad} ({np.load(npy, mmap_mode='r').shape})"); continue
        print(f"\n─── {ad} (n={len(met)}) ───", flush=True)
        H, sayac, sn = kodla_zemin(yazar, kol, met)
        c = capa(yazar, kol, H)
        if c["kurulabilir"]:
            n_capa += 1
            n_gecti += int(c["gecti"])
            print(f"  [CAPA] ℓ21 ↔ kollar/E5_L21__{kol}: kösegen min {c['kosegen_min']:.6f} "
                  f"ort {c['kosegen_ort']:.6f} (bar {CAPA_BAR}) ⇒ "
                  f"{'GECTI ✓' if c['gecti'] else 'DÜSTÜ ✗'}")
            if not c["gecti"]:
                raise RuntimeError(f"{ad}")
        else:
            print(f"  [CAPA] kurulamaz — {c['sebep']} (mistral kollarinin capasi tasir)")
        np.save(npy, np.asarray(H, dtype=np.float16))
        M["zeminler"][ad] = dict(n_satir=len(met), sekil=list(H.shape), yol=os.path.basename(npy),
                                 sha256=sha(npy), capa=c, saniye=round(sn, 1),
                                 red_bos_havuz=int(sayac["bos_havuz"]))
        payda(f"katman_e5_{ad}", n_satir=int(sayac["satir"]), n_katman=int(H.shape[1]),
              red_bos_havuz=int(sayac["bos_havuz"]), bekle={"n_satir": len(met), "n_katman": 25})
        print(f"  → {npy} {list(H.shape)} · sha {M['zeminler'][ad]['sha256'][:16]} · {sn:.1f}s")
        del H
    if n_capa:
        print(f"\n[CAPA PAYDASI] kurulabilen {n_capa} · gecen {n_gecti}")
    json.dump(M, open(mp, "w"), ensure_ascii=False, indent=1)
    print(f"→ {mp}")


if __name__ == "__main__":
    main()
