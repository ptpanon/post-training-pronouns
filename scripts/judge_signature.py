#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, sys, time
import numpy as np
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import form_count as FS
import addressee_olcu as MO
from reading_style_object import payda

ONBELLEK = __DNH_DATA__ + "/uf_bilesen.npz"
CIK = f"{ROOT}/results/judge_signature_2026-09-03.json"
NB, KPERM, SEED, TABAN = 1000, 200, 20260903, 0.05


def _z(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def _sogur(X, g):
    n = np.bincount(g)
    for j in range(X.shape[1]):
        ort = np.bincount(g, weights=X[:, j]) / n
        X[:, j] -= ort[g]
    return X


def _fit(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def kur_tasarim(A, nc, jeton, model_ix, n_model):
    m1 = 1000.0 * A["m1_sahis2"] / np.maximum(jeton, 1)
    s1 = 1000.0 * (A["m5_yakin"] - A["m1_sahis2"]) / np.maximum(jeton, 1)
    md = 1000.0 * A["n_modal"] / np.maximum(jeton, 1)
    ln_j = np.log(np.maximum(jeton, 1.0)); ln_c = np.log(np.maximum(nc, 1.0))
    D = np.zeros((len(jeton), n_model - 1))
    for k in range(1, n_model):
        D[:, k - 1] = (model_ix == k)
    return (_z(m1), _z(s1), _z(md), _z(ln_j), _z(ln_c), D)


def olc(m1, s1, ln_j, ln_c, D, y, g, plasebo=None):
    ilk = plasebo if plasebo is not None else m1
    X = np.column_stack([ilk, s1, ln_j, ln_c, D])
    X = _sogur(X.astype(float), g); yy = _sogur(y.reshape(-1, 1).copy(), g)[:, 0]
    b = _fit(X, yy)
    ad = np.unique(g); rng = np.random.default_rng(SEED)
    ix_of = {u: np.where(g == u)[0] for u in ad}
    B = []
    for _ in range(NB):
        sel = rng.choice(ad, len(ad), replace=True)
        ix = np.concatenate([ix_of[u] for u in sel])
        B.append(_fit(X[ix], yy[ix])[:2])
    B = np.array(B)
    rng2 = np.random.default_rng(SEED + 1); N = []
    for _ in range(KPERM):
        yp = yy.copy()
        for u in ad:
            i = ix_of[u]; yp[i] = yy[rng2.permutation(i)]
        N.append(_fit(X, yp)[:2])
    N = np.array(N)
    out = {}
    for j, ad_j in enumerate(("M1_1k", "SAHIS1_1k")):
        ci = [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))]
        frac = float((np.abs(N[:, j]) >= abs(b[j])).mean())
        gecti = (abs(b[j]) >= TABAN) and (ci[1] < 0 or ci[0] > 0) and b[j] < 0
        out[ad_j] = dict(beta=float(b[j]), ci=ci, ayrik=bool(ci[1] < 0 or ci[0] > 0),
                         taban_gecti=bool(abs(b[j]) >= TABAN), isaret_negatif=bool(b[j] < 0),
                         frac_null=frac, null_ort=float(N[:, j].mean()),
                         null_sd=float(N[:, j].std()), gecti=bool(gecti))
    return out


def prova():
    rng = np.random.default_rng(11); n, k = 400, 100
    g = np.repeat(np.arange(k), 4)
    D = np.zeros((n, 1))
    out = {}
    x = _z(rng.normal(size=n)); s = _z(rng.normal(size=n))
    lj = _z(rng.normal(size=n)); lc = _z(rng.normal(size=n))
    y = rng.normal(size=n)
    out["i_sifir_sinyal"] = olc(x, s, lj, lc, D, y, g)["M1_1k"]["gecti"]
    out["ii_sabit_puan"] = olc(x, s, lj, lc, D, np.ones(n), g)["M1_1k"]["gecti"]
    y2 = -1.0 * x + 0.01 * rng.normal(size=n)
    r3 = olc(x, s, lj, lc, D, y2, g)["M1_1k"]
    out["iii_guclu_negatif"] = dict(gecti=r3["gecti"], beta=round(r3["beta"], 3))
    out["iv_sabit_esdegisken"] = olc(np.zeros(n), s, lj, lc, D, y, g)["M1_1k"]["gecti"]
    return out


def main():
    P = prova()
    print("★ §7.1 DEJENERE PROVA (kosudan ÖNCE):", json.dumps(P, ensure_ascii=False))
    ok = (P["i_sifir_sinyal"] is False and P["ii_sabit_puan"] is False
          and P["iii_guclu_negatif"]["gecti"] is True
          and P["iv_sabit_esdegisken"] is False)
    print(f"{'GECTI' if ok else '★ DÜSTÜ'}"
          f"", flush=True)
    if not ok:
        return 4

    from datasets import load_dataset
    d = load_dataset("openbmb/UltraFeedback")["train"]
    t0 = time.time()
    if os.path.exists(ONBELLEK):
        Z = np.load(ONBELLEK, allow_pickle=True)
        A = {k: Z[k] for k in Z.files if k not in ("nc", "y", "g", "mi", "n_model")}
        nc, y, g, mi, n_model = Z["nc"], Z["y"], Z["g"], Z["mi"], int(Z["n_model"])
        print(f"  [{time.time()-t0:.0f}s] önbellek okundu · {len(y)} yanit", flush=True)
    else:
        M, y, g, mo = [], [], [], []
        for i in range(len(d)):
            r = d[i]
            for c in r["completions"]:
                M.append(c["response"] or ""); y.append(float(c["overall_score"]))
                g.append(i); mo.append(c.get("model") or "?")
        adlar = sorted(set(mo)); ix = {a: k for k, a in enumerate(adlar)}
        mi = np.array([ix[a] for a in mo]); n_model = len(adlar)
        print(f"  [{time.time()-t0:.0f}s] {len(M)} yanit · {len(set(g))} istem · "
              f"{n_model} model · sayac kosuyor…", flush=True)
        nlp = FS._boru()
        A, nc = MO.topla(nlp, M, n_process=6)
        y = np.array(y); g = np.array(g)
        np.savez_compressed(ONBELLEK, nc=nc, y=y, g=g, mi=mi, n_model=n_model,
                            **{k: A[k] for k in A})
        print(f"  [{time.time()-t0:.0f}s] sayac bitti, önbellek yazildi", flush=True)
    g = np.unique(g, return_inverse=True)[1]
    m1, s1, md, lj, lc, D = kur_tasarim(A, nc, A["n_jeton"], mi, n_model)
    R = {"gercek": olc(m1, s1, lj, lc, D, y, g),
         "plasebo_yuzey": olc(m1, s1, lj, lc, D, y, g, plasebo=md)}
    print(f"  [{time.time()-t0:.0f}s] kestirim bitti", flush=True)
    icc_araligi = []
    for icc in (0.0, 0.3, 0.6):
        deff = 1 + 3 * icc
        icc_araligi.append(dict(icc=icc, deff=deff,
                                mde=float(1.645 * np.sqrt(deff / len(y)))))
    S = dict(_sonuc=R, _prova=P, _mde_vekili=icc_araligi,
             _kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         veri="openbmb/UltraFeedback train · TAMAMI",
                         n_yanit=int(len(y)), n_istem=int(len(np.unique(g))),
                         n_model=n_model, cikti="overall_score",
                         sabit_etki="istem (sogurma) + model (gölge)",
                         n_bootstrap=NB, k_permutasyon=KPERM, seed=SEED,
                         taban=TABAN, motor="MO.topla · FS._boru — ITHAL (K-5a)"))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    G = R["gercek"]
    gecen = [k for k, v in G.items() if v["gecti"]]
    verdict = ("IMZA-VAR" if len(gecen) == 2 else
             f"KISMI ({gecen[0]})" if len(gecen) == 1 else
             "SAPTANDI-AMA-ÖNEMSIZ" if all(v["ayrik"] for v in G.values()) else
             "TERS" if any(v["taban_gecti"] and not v["isaret_negatif"] for v in G.values())
             else "ÖLCÜLEMEZ")
    payda("p11_yargic_imzasi", n_yanit=len(y), n_istem=len(np.unique(g)),
          hal_gecen=len(gecen), hal_plasebo_gecen=sum(1 for v in R["plasebo_yuzey"].values()
                                                      if v["gecti"]))
    print(f"\n{'kol':<16}{'β_std':>9}{'CI95':>24}{'frac_null':>11}{'taban':>7}{'gecti':>7}")
    for kol, V in R.items():
        for ad, v in V.items():
            print(f"{kol+'/'+ad:<16}{v['beta']:>+9.4f}"
                  f"  [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]{v['frac_null']:>11.3f}"
                  f"{'✓' if v['taban_gecti'] else '✗':>7}{'✓' if v['gecti'] else '✗':>7}")
    print(f"\n★ MDE vekili: " + " · ".join(f"ICC={x['icc']}⇒{x['mde']:.5f}" for x in icc_araligi)
          + f" sd/sd · TABAN {TABAN}")
    print(f"★★★ VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
