#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import argparse
import numpy as np

ROOT = __DNH_ROOT__ + ""
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from pergel_cekirdek import Cekirdek, Permutator, foldlar

CIKTI = f"{ROOT}/unreleased/ZENGIN_GOMU_2026-08-07.json"
TABLO = f"{ROOT}/unreleased/COKUS_TABLOSU_2026-08-07.json"
K_NULL, NFOLD, SEED = 400, 5, 20260806
Z = 1.645
VARYANTLAR = [dict(max_features=m, min_df=d, sublinear_tf=True)
              for m in (1000, 2000, 4000, 8000) for d in (2, 5)]
ADLAR = ("GÖMÜ-ASIYOR", "TOKENIZER-TORBASI", "KARISIK", "ÖLCÜLEMEZ")


def zemin_verisi():
    import esdegerlik as E
    pde, cde, yde, fde = E.disapere("dev")
    (_, _, _, _), (phe, che, yhe, dhe) = E.kialo()
    return {"dis": ([f"{a} {b}" for a, b in zip(pde, cde)],
                    np.asarray(yde).astype(np.int8), np.asarray(fde)),
            "kia": ([f"{a} {b}" for a, b in zip(phe, che)],
                    np.asarray(yhe).astype(np.int8), np.asarray(dhe))}


def oku(X, y, fold, Yp, dev):
    C = Cekirdek(np.asarray(X, dtype=np.float32), fold, dev=dev, std=True)
    A = float(C.auc_toplu(y[None, :], C.oku_toplu(y[None, :]))[0])
    nl = C.auc_toplu(Yp, C.oku_toplu(Yp))
    nl = nl.cpu().numpy() if hasattr(nl, "cpu") else np.asarray(nl)
    C.bosalt()
    m, s = float(nl.mean()), float(nl.std(ddof=1))
    return dict(A=A, null_merkez=m, null_sd=s, z=float((A - m) / max(s, 1e-12)))


def verdict_gomu(fark, mde):
    if not all(np.isfinite(v) for v in fark.values()):
        return "ÖLCÜLEMEZ"
    asan = [z for z in fark if fark[z] >= mde[z]]
    torba = [z for z in fark if fark[z] <= 0]
    if len(asan) == len(fark):
        return "GÖMÜ-ASIYOR"
    if len(torba) == len(fark):
        return "TOKENIZER-TORBASI"
    return "KARISIK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"ZENGIN-GÖMÜ · damga {damga} · {len(VARYANTLAR)} TF-IDF varyanti "
          f"(bataryanin 8 tarziyla ES genislik) · K_null={K_NULL}")
    from sklearn.feature_extraction.text import TfidfVectorizer
    ZV = zemin_verisi()
    torba = {}
    for z, (met, y, kume) in ZV.items():
        fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
        Yp = Permutator(y, kume.tolist(), seed=SEED).cek(K_NULL)
        tek = []
        for v in VARYANTLAR:
            F = np.asarray(TfidfVectorizer(**v).fit_transform(met).todense(),
                           dtype=np.float32)
            r = oku(F, y, fold, Yp, a.dev)
            tek.append(dict(r, **v, boyut=int(F.shape[1])))
            print(f"  [{z}] TF-IDF mf={v['max_features']:>5} min_df={v['min_df']} "
                  f"d={F.shape[1]:>5} · AUC {r['A']:.4f} · z {r['z']:+.2f}", flush=True)
        en = max(tek, key=lambda r: r["A"])
        torba[z] = dict(varyantlar=tek, max_AUC=en["A"], max_varyant=en["max_features"],
                        null_sd=en["null_sd"], MDE=Z * en["null_sd"], n=int(len(y)))
        print(f"  ★ [{z}] TF-IDF MAX(8) = {en['A']:.4f} (mf={en['max_features']}, "
              f"min_df={en['min_df']}) · null_sd {en['null_sd']:.4f}", flush=True)

    T = json.load(open(TABLO, encoding="utf-8"))
    R = dict(damga=damga, adlar=list(ADLAR), K_null=K_NULL, tfidf=torba,
             not_="ℓ0 = bataryanin auc_l0'i (8 tarz üzerinden MAX); TF-IDF de 8 varyant MAX",
             ciftler={})
    for r in T["satirlar"]:
        if not r.get("verdict"):
            continue
        soy = r["cift"].split("·")[0]
        J = json.load(open(f"{ROOT}/unreleased/onek{r['dosya']}", encoding="utf-8"))
        K = J.get("zeminler")
        if K is None:
            aday = [v for v in J.get("hucreler", {}).values()
                    if v.get("base_kol") == r["base_kol"]] or \
                   [v for k, v in J.get("hucreler", {}).items() if k in (r["cift"], r["etiket"])]
            if not aday:
                print(f"   ↷ ATLA {r['cift']}: kaynakta zemin bulunamadi ({r['dosya']})")
                continue
            K = aday[0]["zeminler"]
        for rol in ("base", "it"):
            fark, mde = {}, {}
            for z in ("dis", "kia"):
                fark[z] = K[z][rol]["auc_l0"] - torba[z]["max_AUC"]
                mde[z] = torba[z]["MDE"]
            R["ciftler"][f"{r['cift']}|{rol}"] = dict(
                soy=soy, kol=K["dis"][rol]["kol"],
                auc_l0={z: round(K[z][rol]["auc_l0"], 4) for z in ("dis", "kia")},
                fark={z: round(v, 4) for z, v in fark.items()},
                verdict=verdict_gomu(fark, mde))

    print("\n★ CIFT × KOL TABLOSU (ℓ0 − TF-IDF_max8)")
    for k, v in R["ciftler"].items():
        print(f"   {k:26s} ℓ0 dis {v['auc_l0']['dis']:.3f} kia {v['auc_l0']['kia']:.3f} · "
              f"fark {v['fark']['dis']:+.3f}/{v['fark']['kia']:+.3f} ⇒ {v['verdict']}")

    soylar = {}
    for k, v in R["ciftler"].items():
        soylar.setdefault(v["soy"], []).append(np.mean(list(v["auc_l0"].values())))
    ort = {s: float(np.mean(x)) for s, x in soylar.items()}
    g3 = [v for k, v in R["ciftler"].items() if k.startswith("gemma·3")]
    g3_ort = float(np.mean([np.mean(list(v["auc_l0"].values())) for v in g3])) if g3 else float("nan")
    digerleri = {s: o for s, o in ort.items() if s != "gemma"}
    R["soy_ort_l0"] = {s: round(o, 4) for s, o in sorted(ort.items(), key=lambda x: -x[1])}
    R["gemma3_ort_l0"] = round(g3_ort, 4)
    R["K_B"] = ("GEMMA3-EN-YÜKSEK" if g3 and all(g3_ort > o for o in digerleri.values())
                else "DEGIL" if g3 else "ÖLCÜLEMEZ")
    print(f"\n★ SOY ORTALAMA ℓ0: " + " · ".join(f"{s} {o:.3f}" for s, o in R["soy_ort_l0"].items()))
    print(f"★ K-B · gemma-3 ℓ0 ort {g3_ort:.4f} ⇒ **{R['K_B']}**")
    payda("zengin_gomu", n_zemin=len(torba), n_varyant=len(VARYANTLAR),
          n_cift_kol=len(R["ciftler"]), bekle={"n_zemin": 2, "n_varyant": 8})
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    print(f"→ {CIKTI}")


if __name__ == "__main__":
    main()
