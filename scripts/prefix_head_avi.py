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
from pergel_cekirdek import Cekirdek, foldlar
from prefix_muhurlu_kosu import auc_np
import prefix_ladder as M
import prefix_ladder_gomu as G
import prefix_archive_head as A

CIKTI = f"{ROOT}/unreleased/HEAD_AVI_2026-08-08.json"
GIZLI = __DNH_DATA__ + "/onek_korpus/merdiven_e5"
TUTULAN = ["bull-fight", "nipples", "palestine",
           "people-should-not-follow-any-specific-religion"]
HUCRELER = [("c1a", "mistral_base"), ("c2a", "mistral_base"),
            ("c1a", "gemma_base"), ("c2a", "gemma_base")]
BIRINCIL = ("c1a", "mistral_base")
B_BOOT, SEED, Z, NFOLD = 200, 20260808, 1.645, 5
ORAN5 = 0.05

K_A_ADLARI = ("HEAD-ÜSTÜN", "KATMAN-YETER", "AYIRT-EDILEMEZ", "TARZA-BAGIMLI", "ÖLCÜLEMEZ")
K_B_ADLARI = ("YÜZDE5-YETER", "YÜZDE5-YETMEZ", "ÖLCÜLEMEZ")


def egit_degerlendir(X, y, tr, va, dev):
    fold = np.where(tr, 0, 1).astype(np.int64)
    C = Cekirdek(np.asarray(X, dtype=np.float32), fold, dev=dev, std=True)
    s = C.oku_toplu(y[None, :])
    s = (s.cpu().numpy() if hasattr(s, "cpu") else np.asarray(s))[0]
    C.bosalt()
    return float(auc_np(y[va], s[va])), s


def oof_auc(X, y, kume, dev):
    fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
    C = Cekirdek(np.asarray(X, dtype=np.float32), fold, dev=dev, std=True)
    s = C.oku_toplu(y[None, :])
    s = (s.cpu().numpy() if hasattr(s, "cpu") else np.asarray(s))[0]
    C.bosalt()
    return float(auc_np(y, s))


def kume_bootstrap_mde(y, skor_a, skor_b, kume, rng, B=B_BOOT):
    ks = sorted(set(kume.tolist()))
    ix = {k: np.flatnonzero(kume == k) for k in ks}
    d = []
    for _ in range(B):
        sec = rng.choice(len(ks), size=len(ks), replace=True)
        j = np.concatenate([ix[ks[i]] for i in sec])
        if len(set(y[j].tolist())) < 2:
            continue
        d.append(auc_np(y[j], skor_a[j]) - auc_np(y[j], skor_b[j]))
    return (float(np.std(d, ddof=1)) if len(d) > 2 else float("nan")), len(d)


def verdict_ka(farklar, mdeler, isaret_tutarli, olculebilir):
    if not olculebilir or not any(np.isfinite(m) for m in mdeler):
        return "ÖLCÜLEMEZ"
    if not isaret_tutarli:
        return "TARZA-BAGIMLI"
    n_asan = sum(1 for f, m in zip(farklar, mdeler) if np.isfinite(m) and f >= m)
    n_ters = sum(1 for f, m in zip(farklar, mdeler) if np.isfinite(m) and -f >= m)
    if n_asan >= 3:
        return "HEAD-ÜSTÜN"
    if n_ters >= 3:
        return "KATMAN-YETER"
    return "AYIRT-EDILEMEZ"


def verdict_kb(farklar, mdeler, olculebilir):
    if not olculebilir or not any(np.isfinite(m) for m in mdeler):
        return "ÖLCÜLEMEZ"
    n = sum(1 for f, m in zip(farklar, mdeler) if np.isfinite(m) and f >= -m)
    return "YÜZDE5-YETER" if n >= 3 else "YÜZDE5-YETMEZ"


def veri(okur, kol):
    X, y, kume, tarz = [], [], [], []
    for bas in M.SIRA:
        _, S = G.satirlar(kol, bas)
        X.append(np.load(f"{A.OUT}/CTX_{okur}__{kol}__{bas}.fp16.npy", mmap_mode="r"))
        y += [1 if r["durus"] == "arti" else 0 for r in S]
        kume += [r["debate"] for r in S]
        tarz += [bas] * len(S)
    return X, (np.array(y, dtype=np.int8), np.array(kume), np.array(tarz))


def dilim(X, idx, l, h, hd=64):
    return np.concatenate([np.asarray(x[:, l, h * hd:(h + 1) * hd], dtype=np.float32)
                           for x in X])[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    MF = json.load(open(f"{A.OUT}/manifest.json", encoding="utf-8"))
    L = MF["okurlar"]["c1a"]["L"]
    NH = MF["okurlar"]["c1a"]["n_head"]
    HD = MF["okurlar"]["c1a"]["head_dim"]
    n5 = int(np.ceil(ORAN5 * L * NH))
    print(f"HEAD-AVI · damga {damga} · L={L} × head={NH} = {L*NH} · "
          f"H-5% = {n5} head · tutulan küme {TUTULAN}")
    R = dict(damga=damga, prereg="f9ed0ca", tutulan_kume=TUTULAN, L=L, n_head=NH,
             head_dim=HD, n_5yuzde=n5, B_boot=B_BOOT,
             K_A_adlari=list(K_A_ADLARI), K_B_adlari=list(K_B_ADLARI), hucreler={})

    for okur, kol in HUCRELER:
        X, (y, kume, tarz) = veri(okur, kol)
        n = len(y)
        tut_k = np.isin(kume, TUTULAN)
        rap = dict(n=int(n), rotasyonlar={})
        print(f"\n═══ {okur} × {kol} · n={n} ═══", flush=True)
        gizli_yol = {b: f"{GIZLI}/E5_TUM__{kol}__{b}.fp16.npy" for b in M.SIRA}
        gizli_var = okur == "c1a" and all(os.path.exists(p) for p in gizli_yol.values())

        for tutulan_tarz in M.SIRA:
            sec = (~tut_k) & (tarz != tutulan_tarz)
            va1 = tut_k & (tarz != tutulan_tarz)
            va2 = (~tut_k) & (tarz == tutulan_tarz)
            ix_s = np.flatnonzero(sec)
            skor = np.full((L, NH), np.nan)
            for l in range(L):
                Xl = np.concatenate([np.asarray(x[:, l, :], dtype=np.float32) for x in X])
                for h in range(NH):
                    skor[l, h] = oof_auc(Xl[ix_s, h * HD:(h + 1) * HD], y[ix_s],
                                         kume[ix_s], a.dev)
                del Xl
            kes = np.abs(skor - 0.5)
            olcu = bool(np.nanmax(kes) > 0)
            sirali = sorted(((l, h) for l in range(L) for h in range(NH)),
                            key=lambda t: -kes[t])
            H4 = [(l, h) for l in range(L)
                  for h in sorted(range(NH), key=lambda x: -kes[l, x])[:4]]
            H5 = sirali[:n5]
            TAM = [(l, h) for l in range(L) for h in range(NH)]
            en_iyi_l = int(np.argmax(kes.max(1)))

            def birlestir(sec_head, idx):
                return np.concatenate(
                    [dilim(X, idx, l, h, HD) for l, h in sec_head], axis=1)

            rot = dict(tutulan_tarz=tutulan_tarz, n_secim=int(sec.sum()),
                       n_va1=int(va1.sum()), n_va2=int(va2.sum()),
                       en_iyi_katman=en_iyi_l,
                       en_iyi_head_auc=float(np.nanmax(skor)),
                       secilen_H5=[[int(l), int(h)] for l, h in H5], satirlar={})
            for ad, mask in (("SATIR-1", va1), ("SATIR-2", va2)):
                idx = np.flatnonzero(sec | mask)
                tr = sec[idx]
                va = mask[idx]
                yy, kk = y[idx], kume[idx]
                A_, S_ = {}, {}
                for bad, hs in (("H-5%", H5), ("H-4", H4), ("TAM", TAM)):
                    A_[bad], S_[bad] = egit_degerlendir(birlestir(hs, idx), yy, tr, va,
                                                        a.dev)
                Xc = np.concatenate([np.asarray(x[:, en_iyi_l, :], dtype=np.float32)
                                     for x in X])[idx]
                A_["KATMAN-CTX"], S_["KATMAN-CTX"] = egit_degerlendir(Xc, yy, tr, va, a.dev)
                if gizli_var:
                    Xg = np.concatenate([np.load(gizli_yol[b], mmap_mode="r")[
                        :, en_iyi_l + 1, :].astype(np.float32) for b in M.SIRA])[idx]
                    A_["KATMAN"], S_["KATMAN"] = egit_degerlendir(Xg, yy, tr, va, a.dev)
                    del Xg
                taban = "KATMAN" if gizli_var else "KATMAN-CTX"
                rng = np.random.default_rng(SEED)
                sd_a, nb_a = kume_bootstrap_mde(yy[va], S_["H-4"][va], S_[taban][va],
                                                kk[va], rng)
                sd_b, nb_b = kume_bootstrap_mde(yy[va], S_["H-5%"][va], S_["TAM"][va],
                                                kk[va], np.random.default_rng(SEED + 1))
                rot["satirlar"][ad] = dict(
                    AUC={k: round(v, 4) for k, v in A_.items()}, taban=taban,
                    fark_KA=round(A_["H-4"] - A_[taban], 4), MDE_KA=round(Z * sd_a, 4),
                    fark_KB=round(A_["H-5%"] - A_["TAM"], 4), MDE_KB=round(Z * sd_b, 4),
                    n_boot=[nb_a, nb_b], n_kume_dogrulama=int(len(set(kk[va].tolist()))))
                print(f"  [{tutulan_tarz}/{ad}] " +
                      " · ".join(f"{k} {v:.3f}" for k, v in A_.items()) +
                      f" ⇒ K-A fark {rot['satirlar'][ad]['fark_KA']:+.3f} "
                      f"(MDE {rot['satirlar'][ad]['MDE_KA']:.3f}) · "
                      f"K-B fark {rot['satirlar'][ad]['fark_KB']:+.3f} "
                      f"(MDE {rot['satirlar'][ad]['MDE_KB']:.3f})", flush=True)
            rot["olculebilir"] = olcu
            rap["rotasyonlar"][tutulan_tarz] = rot

        for ad in ("SATIR-1", "SATIR-2"):
            fa = [rap["rotasyonlar"][t]["satirlar"][ad]["fark_KA"] for t in M.SIRA]
            ma = [rap["rotasyonlar"][t]["satirlar"][ad]["MDE_KA"] for t in M.SIRA]
            fb = [rap["rotasyonlar"][t]["satirlar"][ad]["fark_KB"] for t in M.SIRA]
            mb = [rap["rotasyonlar"][t]["satirlar"][ad]["MDE_KB"] for t in M.SIRA]
            olc = all(rap["rotasyonlar"][t]["olculebilir"] for t in M.SIRA)
            tutarli = all(v > 0 for v in fa) or all(v < 0 for v in fa)
            rap[ad] = dict(K_A=verdict_ka(fa, ma, tutarli, olc), K_B=verdict_kb(fb, mb, olc),
                           farklar_KA=fa, farklar_KB=fb, isaret_tutarli=bool(tutarli))
            print(f"  ⇒ {ad}: K-A **{rap[ad]['K_A']}** · K-B **{rap[ad]['K_B']}** "
                  f"(isaret tutarli: {tutarli})")
        R["hucreler"][f"{okur}|{kol}"] = rap

    b = R["hucreler"][f"{BIRINCIL[0]}|{BIRINCIL[1]}"]
    R["BIRINCIL"] = dict(hucre=list(BIRINCIL),
                         SATIR_1=dict(K_A=b["SATIR-1"]["K_A"], K_B=b["SATIR-1"]["K_B"]),
                         SATIR_2=dict(K_A=b["SATIR-2"]["K_A"], K_B=b["SATIR-2"]["K_B"]))
    payda("head_avi", n_hucre=len(R["hucreler"]), n_rotasyon=len(M.SIRA),
          n_head=L * NH, B=B_BOOT, bekle={"n_hucre": 4, "n_rotasyon": 5, "n_head": 384})
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    print(f"\n★ BIRINCIL ({BIRINCIL[0]} × {BIRINCIL[1]}): "
          f"SATIR-1 K-A {b['SATIR-1']['K_A']} / K-B {b['SATIR-1']['K_B']} · "
          f"SATIR-2 K-A {b['SATIR-2']['K_A']} / K-B {b['SATIR-2']['K_B']}")
    print(f"→ {CIKTI}")


if __name__ == "__main__":
    main()
