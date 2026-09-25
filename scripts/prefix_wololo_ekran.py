#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import argparse
import collections
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import prefix_steering_generation as SU
import prefix_aktivasyon_arsivi as AA

CIKTI = f"{ROOT}/unreleased/WOLOLO_EKRAN_2026-08-08.json"
ONBELLEK = __DNH_DATA__ + "/onek_korpus/wololo/pergel.npz"
KATMANLAR = (8, 12, 16, 20, 24, 28)
N_HEAD, D_HEAD = 32, 128
K_NULL, SEED = 200, 20260808
KAT_SPREAD = 1.5
N_ADAY, N_KONTROL = 32, 16


def verdict_ekran(goz_p95, goz_p50, null_p95, null_p50):
    if not all(np.isfinite([goz_p95, goz_p50, null_p95, null_p50])):
        return "ÖLCÜLEMEZ"
    yayilim_g, yayilim_n = goz_p95 - goz_p50, null_p95 - null_p50
    if goz_p95 > null_p95 and yayilim_g >= KAT_SPREAD * yayilim_n:
        return "EKRAN-GECERLI"
    return "EKRAN-GECERSIZ"


def _prova():
    kes = [("NaN", verdict_ekran(float("nan"), .1, .1, .05)),
           ("yayilim yok (D-16 vakasi)", verdict_ekran(0.90, 0.89, 0.50, 0.45)),
           ("null'u asmiyor", verdict_ekran(0.10, 0.02, 0.12, 0.05)),
           ("gecerli", verdict_ekran(0.30, 0.08, 0.12, 0.06))]
    for a_, h in kes:
        print(f"  prova {a_:26s} → {h}")
    assert len(set(h for _, h in kes)) == 3
    print("")


def pergel_yukle(dev):
    if os.path.exists(ONBELLEK):
        z = np.load(ONBELLEK, allow_pickle=True)
        return z["pergel"], list(z["anahtar"])
    os.makedirs(os.path.dirname(ONBELLEK), exist_ok=True)
    TON = __DNH_DATA__ + "/onek_korpus/ton_dengeli/ton__mistral.jsonl"
    Z = [json.loads(l) for l in open(TON, encoding="utf-8")]
    met = [(z.get("metin_160") or "").strip() for z in Z]
    from bs141_frontier_baselines import plain_enc
    E = plain_enc("intfloat/e5-large-v2", met, prefix="query: ")
    E = E - E.mean(0)
    lab = np.array([z.get("ton_hedef") for z in Z], dtype=object)
    d = SU.birim(E[lab == "sert"].mean(0) - E[lab == "sakin"].mean(0))
    p = E @ d
    ah = [f"{z['debate']}|{z['durus']}|{z['ton_hedef']}|{z['cekim']}" for z in Z]
    np.savez_compressed(ONBELLEK, pergel=p, anahtar=np.array(ah, dtype=object))
    return p, ah


_PY = {}


def pergel_yon(dev="cpu", alan="ton_hedef", arti="sert", eksi="sakin"):
    ad = f"{alan}|{arti}|{eksi}"
    if ad not in _PY:
        TON = __DNH_DATA__ + "/onek_korpus/ton_dengeli/ton__mistral.jsonl"
        Z = [json.loads(l) for l in open(TON, encoding="utf-8")]
        zm = [(z.get("metin_160") or "").strip() for z in Z]
        tut = [i for i, m in enumerate(zm) if len(m) > 40]
        from bs141_frontier_baselines import plain_enc
        E0 = plain_enc("intfloat/e5-large-v2", [zm[i] for i in tut], prefix="query: ")
        mu0 = E0.mean(0)
        lab = np.array([Z[i].get(alan) for i in tut], dtype=object)
        d = SU.birim((E0 - mu0)[lab == arti].mean(0) - (E0 - mu0)[lab == eksi].mean(0))
        pr = (E0 - mu0) @ d
        y = (lab == arti).astype(int)
        _PY[ad] = dict(mu0=mu0, d=d, n=len(tut), n_arti=int(y.sum()),
                       izdusum=pr, etiket=y)
    z = _PY[ad]
    return z["mu0"], z["d"]


def blok_artik(X, blok):
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for b in set(blok.tolist()):
        m = blok == b
        out[m] = X[m] - X[m].mean(0)
    return out


def skor(Sr, rr, egit):
    d = Sr[egit].T @ rr[egit]
    n = np.linalg.norm(d)
    if n == 0:
        return 0.0
    pr = Sr[~egit] @ (d / n)
    if pr.std() == 0 or rr[~egit].std() == 0:
        return 0.0
    return float(abs(np.corrcoef(pr, rr[~egit])[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--k-null", type=int, default=K_NULL)
    ap.add_argument("--prova", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    print("═══ WOLOLO §1 · ISTEM-ARITILMIS EKRAN ═══")
    _prova()
    if a.prova:
        print("★ PROVA — kapi mekanigi sinandi; veri OKUNMADI.")
        return
    pergel, ah_p = pergel_yukle(a.dev)
    ix_p = {k: i for i, k in enumerate(ah_p)}

    rows, null_hav = [], []
    rng = np.random.default_rng(SEED)
    for L in KATMANLAR:
        _, _, A, R, _ = AA.yukle(korpus="ton_dengeli", katman=L, seg=False, attn=True)
        ah = [f"{r['debate']}|{r['durus']}|{r['ton_hedef']}|{r['cekim']}" for r in R]
        tut = np.array([k in ix_p for k in ah])
        A = A[tut]
        ah = [k for k, t in zip(ah, tut) if t]
        r = np.array([pergel[ix_p[k]] for k in ah])
        blok = np.array(["|".join(k.split("|")[:3]) for k in ah], dtype=object)
        kume = np.array([k.split("|")[0] for k in ah], dtype=object)
        ks = np.array(sorted(set(kume.tolist())), dtype=object)
        p = rng.permutation(len(ks))
        egit = np.isin(kume, ks[p[:len(ks) // 2]])
        rr = blok_artik(r[:, None], blok)[:, 0]
        if L == KATMANLAR[0]:
            payda("wololo_havuz", n_satir=len(r), n_blok=len(set(blok.tolist())),
                  n_kume=len(ks), n_egit=int(egit.sum()))
            print(f"{len(r)} {len(set(blok.tolist()))}"
                  f"{len(ks)} {rr.std():.4f}"
                  f"{r.std():.4f}")
        for h in range(N_HEAD):
            Sr = blok_artik(A[:, h * D_HEAD:(h + 1) * D_HEAD], blok)
            rows.append(dict(katman=L, head=h, skor=round(skor(Sr, rr, egit), 4)))
        if L in (12, 24):
            for _ in range(a.k_null // 2):
                rp = rr.copy()
                for b in set(blok.tolist()):
                    m = blok == b
                    rp[m] = rng.permutation(rp[m])
                h = int(rng.integers(N_HEAD))
                Sr = blok_artik(A[:, h * D_HEAD:(h + 1) * D_HEAD], blok)
                null_hav.append(skor(Sr, rp, egit))
        print(f"  ℓ{L} bitti · {time.time()-t0:.0f}s", flush=True)

    s = np.array([r["skor"] for r in rows])
    nl = np.array(null_hav)
    g95, g50 = float(np.percentile(s, 95)), float(np.percentile(s, 50))
    n95, n50 = float(np.percentile(nl, 95)), float(np.percentile(nl, 50))
    H = verdict_ekran(g95, g50, n95, n50)
    rows.sort(key=lambda r: -r["skor"])
    R = dict(damga=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             sinif="",
             olcut=dict(kat_spread=KAT_SPREAD,
                        metin="GECERLI ⇔ (göz p95 > null p95) ∧ "
                              "(göz p95−p50 ≥ 1,5 × null p95−p50)"),
             gozlenen=dict(p50=round(g50, 4), p95=round(g95, 4),
                           maks=round(float(s.max()), 4), min=round(float(s.min()), 4),
                           yayilim=round(g95 - g50, 4)),
             null=dict(p50=round(n50, 4), p95=round(n95, 4), n=len(nl),
                       yayilim=round(n95 - n50, 4)),
             verdict_ekran=H, n_koordinat=len(rows), skorlar=rows)
    print(f"\n  gözlenen: p50 {g50:.4f} · p95 {g95:.4f} · maks {s.max():.4f} · "
          f"yayilim {g95-g50:.4f}")
    print(f"  null    : p50 {n50:.4f} · p95 {n95:.4f} (n={len(nl)}) · "
          f"yayilim {n95-n50:.4f}")
    print(f"  ★★ EKRAN HÜKMÜ: **{H}**")
    if H == "EKRAN-GECERLI":
        aday = rows[:N_ADAY]
        kalan = rows[N_ADAY:]
        kontrol = [kalan[i] for i in
                   np.random.default_rng(SEED + 3).choice(len(kalan), N_KONTROL,
                                                           replace=False)]
        R["aday"], R["kontrol"] = aday, kontrol
        print(f"  aday-{N_ADAY}: " +
              " · ".join(f"ℓ{r['katman']}h{r['head']}({r['skor']:.3f})" for r in aday[:6]))
        print(f"  kontrol-{N_KONTROL} ort skor "
              f"{np.mean([c['skor'] for c in kontrol]):.4f}")
    else:
        R["tikandi"] = (""
                        "")
        print(f"  ★ TIKANDI: liste ÜRETILMEDI — {R['tikandi']}")
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    payda("wololo_ekran", n_koordinat=len(rows), n_katman=len(KATMANLAR),
          n_null=len(nl), n_aday=len(R.get("aday", [])),
          bekle={"n_koordinat": len(KATMANLAR) * N_HEAD})
    print(f"→ {CIKTI} · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
