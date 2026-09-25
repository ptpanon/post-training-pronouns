#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import prefix_reviewer_bootstrap as HB
import prefix_steering_generation as SU

BAR = 0.70
ADLAR = ("demet_yogunlasmasi", "kume_tutulan_bolme")


def demet_yogunlasmasi(W, rng=None):
    r = rng if rng is not None else np.random.default_rng(0)
    return float(HB.kararlilik(W, r)["merkezsiz"][0])


def kume_tutulan_bolme(X, y, kume, n_bol=50, seed=20260808, esli=None):
    rng = np.random.default_rng(seed)
    ks = np.array(sorted(set(np.asarray(kume).tolist())))
    kume = np.asarray(kume)
    out, atlanan = [], 0
    for _ in range(n_bol):
        p = rng.permutation(len(ks))
        A = np.isin(kume, ks[p[: len(ks) // 2]])
        yy = []
        for M in (A, ~A):
            d = _fark(X[M], y[M], None if esli is None else np.asarray(esli)[M])
            if d is None:
                yy = None
                break
            yy.append(d)
        if yy is None:
            atlanan += 1
            continue
        out.append(abs(float(np.dot(yy[0], yy[1]))))
    if not out:
        return dict(medyan=float("nan"), p10=float("nan"), p90=float("nan"),
                    n_bolme=0, atlanan=atlanan, gecti=False, bar=BAR)
    o = np.array(out)
    return dict(medyan=round(float(np.median(o)), 4),
                p10=round(float(np.percentile(o, 10)), 4),
                p90=round(float(np.percentile(o, 90)), 4),
                n_bolme=len(o), atlanan=atlanan,
                gecti=bool(np.median(o) >= BAR), bar=BAR)


def _fark(X, y, esli=None):
    if len(set(np.asarray(y).tolist())) < 2:
        return None
    y = np.asarray(y)
    if esli is None:
        return SU.birim(X[y == 1].mean(0) - X[y == 0].mean(0))
    D = []
    for k in set(np.asarray(esli).tolist()):
        m = np.asarray(esli) == k
        if len(set(y[m].tolist())) < 2:
            continue
        D.append(X[m][y[m] == 1].mean(0) - X[m][y[m] == 0].mean(0))
    return SU.birim(np.mean(D, 0)) if D else None


def karsilastirma_testi(seed=7):
    rng = np.random.default_rng(seed)
    d, nk, nper = 64, 12, 60
    sonuc = {}
    for rejim in ("gercek_yon", "kume_ozgu_yon"):
        ortak = rng.standard_normal(d)
        X, y, ku = [], [], []
        for k in range(nk):
            v = ortak if rejim == "gercek_yon" else rng.standard_normal(d)
            v = v / np.linalg.norm(v)
            for i in range(nper):
                s = 1 if i % 2 == 0 else 0
                X.append(2.0 * s * v + rng.standard_normal(d) * 1.0)
                y.append(s); ku.append(f"k{k}")
        X = np.stack(X); y = np.array(y, dtype=np.int8); ku = np.array(ku)
        W, _ = HB.yonler(X, y, ku, np.random.default_rng(seed), "kume")
        a = demet_yogunlasmasi(W, np.random.default_rng(seed))
        b = kume_tutulan_bolme(X, y, ku, n_bol=30, seed=seed)
        sonuc[rejim] = dict(demet=round(a, 4), kume_tutulan=b["medyan"],
                            fark=round(a - b["medyan"], 4))
        print(f"  [{rejim:15s}] demet_yogunlasmasi {a:.4f} · "
              f"kume_tutulan_bolme {b['medyan']:.4f} · fark {a-b['medyan']:+.4f}")
    ayristi = (sonuc["kume_ozgu_yon"]["fark"] - sonuc["gercek_yon"]["fark"]) > 0.20
    payda("kararlilik_karsilastirma", n_rejim=len(sonuc), n_olcu=len(ADLAR),
          bekle={"n_rejim": 2, "n_olcu": 2})
    print(f"  ★ IKI ÖLCÜ AYRISIYOR MU: {'EVET' if ayristi else 'HAYIR'} "
          f"(küme-özgü rejimde fark {sonuc['kume_ozgu_yon']['fark']:+.4f}, "
          f"gercek-yön rejiminde {sonuc['gercek_yon']['fark']:+.4f})")
    return dict(sonuc=sonuc, ayristi=bool(ayristi),
                verdict=(""
                       if ayristi else ""))


if __name__ == "__main__":
    import json, time
    print("KARARLILIK · iki ölcü, iki ad — karsilastirma testi (K-5a)")
    R = karsilastirma_testi()
    R["damga"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    R["adlar"] = list(ADLAR)
    R["kapi"] = "kume_tutulan_bolme"
    R["kapi_disi"] = "demet_yogunlasmasi (betimleyici; doygun, eleyemez)"
    json.dump(R, open(f"{ROOT}/unreleased/KARARLILIK_IKI_OLCU_2026-08-08.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n→ unreleased/KARARLILIK_IKI_OLCU_2026-08-08.json")
