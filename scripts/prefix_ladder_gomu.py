#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
import prefix_ladder as M
import prefix_ladder_example as O

KAYNAK_PREREG = __DNH_DATA__ + "/onek_korpus/muhurlu"
MUHURLU_E5 = __DNH_DATA__ + "/onek_korpus/katman_e5"
OUT = __DNH_DATA__ + "/onek_korpus/merdiven_e5"
PENCERE, CAPA_BAR = "metin_160", 0.9995
DOSYA_ADI = {"B0": "plasebo", "B2": "ana"}

ONEK_AD = {"c1a": "E5_TUM", "c3a": "BGEL_TUM", "c4a": "SIMCSE_TUM", "c5a": "QWENEMB_TUM"}
YIGIN = {"c1a": 64, "c3a": 64, "c4a": 64, "c5a": 8}
def cikis_dizini(motor):
    return OUT if motor == "c1a" else f"{OUT.rsplit('_', 1)[0]}_{motor}"


def satir_kimlik_sha(S):
    h = hashlib.sha256()
    for r in S:
        h.update((str(r.get("debate", "")) + "|" + str(r.get("durus", "")) + "|" +
                  str(r.get("cekim", "")) + "|" +
                  hashlib.sha1((r.get(PENCERE) or "").encode()).hexdigest()[:16]
                  + "\n").encode())
    return h.hexdigest()[:16]


def ozdes_satir_sayimi(H, katman=0):
    A = np.asarray(H[:, katman, :], dtype=np.float32)
    _, ters, sayi = np.unique(A, axis=0, return_inverse=True, return_counts=True)
    return int((sayi[ters] > 1).sum())


def satirlar(kol, bas):
    yol, _ = O.hucre_yukle(kol, bas)
    S = [json.loads(l) for l in open(yol, encoding="utf-8")]
    if bas in M.KOLLAR[kol]["diskte"]:
        S = [r for r in S if r["isi"] == "sert"]
    return yol, S


def capa_indeksleri(yazar, bas):
    yol = f"{KAYNAK_PREREG}/{DOSYA_ADI[bas]}__{yazar}.jsonl"
    return [i for i, l in enumerate(open(yol, encoding="utf-8"))
            if json.loads(l)["isi"] == "sert"], yol


def capa(kol, bas, H, motor="c1a"):
    if motor != "c1a":
        return dict(kurulabilir=False, sinif="kosinus-capasi-motor-disi",
                    sebep=f"{motor}"
                          f"")
    K = M.KOLLAR[kol]
    if bas not in K["diskte"]:
        return dict(kurulabilir=False,
                    sebep=f"{kol} {bas}")
    ix, _ = capa_indeksleri(K["yazar"], bas)
    p = f"{MUHURLU_E5}/E5_TUM__{K['yazar']}__{DOSYA_ADI[bas]}.fp16.npy"
    B = np.load(p, mmap_mode="r")[ix][:, 21, :].astype(np.float32)
    A = np.asarray(H[:, 21, :], dtype=np.float32)
    an = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    d = (an * bn).sum(1)
    return dict(kurulabilir=True, n=int(len(d)), kaynak=os.path.basename(p),
                kosegen_min=float(d.min()), kosegen_ort=float(d.mean()),
                bar=CAPA_BAR, gecti=bool(d.min() >= CAPA_BAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, required=True)
    ap.add_argument("--hucre", default="hepsi")
    ap.add_argument("--motor", default="c1a", choices=tuple(ONEK_AD))
    a = ap.parse_args()
    OUT_M = cikis_dizini(a.motor)
    os.makedirs(OUT_M, exist_ok=True)
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    C = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=False, etiket="merdiven_gomu")
    import prefix_layer_gomuleri as KG
    KG.DEV = a.dev
    hucreler = ([(k, b) for k in M.KOLLAR for b in M.SIRA] if a.hucre == "hepsi"
                else [tuple(a.hucre.split(":"))])
    c1a_mp = f"{OUT}/manifest.json"
    C1A = json.load(open(c1a_mp))["hucreler"] if os.path.exists(c1a_mp) else {}
    print(f"{a.motor} {damga}"
          f"{len(hucreler)} {a.dev} {a.bekle_gpu}")
    mp = f"{OUT_M}/manifest.json"
    M_ = json.load(open(mp)) if os.path.exists(mp) else dict(damga=damga, hucreler={})
    M_["damga_son"], M_["lafiz_hash"] = damga, M.LAFIZ_HASH
    M_["motor"], M_["motor_sozlesmesi"] = a.motor, None
    M_["tarif"] = KG.__doc__.strip().splitlines()[0]
    M_["gpu"] = C
    n_capa, n_gecti = 0, 0
    t_zincir, n_kosan, eta_basildi = time.time(), 0, False
    for kol, bas in hucreler:
        ad = f"{kol}__{bas}"
        npy = f"{OUT_M}/{ONEK_AD[a.motor]}__{ad}.fp16.npy"
        if os.path.exists(npy):
            print(f"  ↷ ATLA {ad} ({np.load(npy, mmap_mode='r').shape})")
            continue
        yol, S = satirlar(kol, bas)
        met = [r[PENCERE] for r in S]
        kim = satir_kimlik_sha(S)
        if ad in C1A and int(C1A[ad]["n_satir"]) != len(met):
            raise RuntimeError(f"{ad} {C1A[ad]['n_satir']}"
                               f"{len(met)}")
        H, sayac, sn = KG.kodla_zemin(M.KOLLAR[kol]["yazar"], bas, met,
                                      motor=a.motor, bs=YIGIN[a.motor])
        n_kosan += 1
        c = capa(kol, bas, H, motor=a.motor)
        if c["kurulabilir"]:
            n_capa += 1
            n_gecti += int(c["gecti"])
            print(f"  [CAPA] {ad} ↔ {c['kaynak']}: kösegen min {c['kosegen_min']:.6f} "
                  f"ort {c['kosegen_ort']:.6f} ⇒ {'GECTI ✓' if c['gecti'] else 'DÜSTÜ ✗'}")
            if not c["gecti"]:
                raise RuntimeError(f"CAPA DÜSTÜ ({ad}) ⇒ kod yolu mühürlü yoldan SAPMIS.")
        ozdes = ozdes_satir_sayimi(H, katman=0)
        np.save(npy, np.asarray(H, dtype=np.float16))
        M_["hucreler"][ad] = dict(kaynak=yol, n_satir=len(met), sekil=list(H.shape),
                                  yol=os.path.basename(npy),
                                  sha256=hashlib.sha256(open(npy, "rb").read()).hexdigest(),
                                  capa=c, saniye=round(sn, 1),
                                  red_bos_havuz=int(sayac["bos_havuz"]),
                                  satir_kimlik_sha=kim,
                                  c1a_ile_ayni_satir=(C1A[ad]["n_satir"] == len(met)
                                                      if ad in C1A else None),
                                  red_ozdes_satir_L0=ozdes,
                                  red_maxlen_carpan=int(sayac["maxlen_carpan"]))
        payda(f"merdiven_{a.motor}_{ad}", n_satir=int(sayac["satir"]),
              n_katman=int(H.shape[1]), red_bos_havuz=int(sayac["bos_havuz"]),
              red_ozdes_satir_L0=ozdes, red_maxlen_carpan=int(sayac["maxlen_carpan"]),
              bekle={"n_satir": len(met)})
        print(f"  → {ad} {list(H.shape)} · sha {M_['hucreler'][ad]['sha256'][:16]} · "
              f"kimlik {kim} · özdes-L0 {ozdes} · kesilen {int(sayac['maxlen_carpan'])} · "
              f"{sn:.1f}s", flush=True)
        del H
        json.dump(M_, open(mp, "w"), ensure_ascii=False, indent=1)
        if not eta_basildi:
            kalan = len([1 for k2, b2 in hucreler
                         if not os.path.exists(f"{OUT_M}/{ONEK_AD[a.motor]}__{k2}__{b2}.fp16.npy")])
            cevrim = time.time() - t_zincir
            print(f"{cevrim:.1f}"
                  f"{sn:.1f} {kalan}"
                  f"{(kalan * (cevrim - 0)) / 60:.1f}"
                  f"", flush=True)
            eta_basildi = True
    print(f"\n[CAPA PAYDASI] kosinüs-capasi kurulabilen {n_capa} · gecen {n_gecti}")
    if a.motor != "c1a" and C1A:
        esit = sum(1 for ad in M_["hucreler"]
                   if ad in C1A and M_["hucreler"][ad].get(""))
        print(f"{len(C1A)}"
              f"{len(M_['hucreler'])} {esit}")
    import anchor_kodlama as CK
    M_["motor_sozlesmesi"] = {k: v for k, v in CK.KOLLAR[a.motor].items() if k != "kayit"}
    json.dump(M_, open(mp, "w"), ensure_ascii=False, indent=1)
    print(f"→ {mp}")


if __name__ == "__main__":
    main()
