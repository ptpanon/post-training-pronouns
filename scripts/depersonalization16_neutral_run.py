#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, shutil, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
from reading_style_object import payda

BIRINCIL = {
    "Tulu3-8B":        ("taban", ["taban", "rl"],            "rl"),
    "OLMo2-13B":       ("taban", ["taban", "instruct"],      "instruct"),
    "OLMo2-32B":       ("taban", ["taban", "instruct"],      "instruct"),
    "OLMo-3-7B":       ("base",  ["base", "instruct"],       "instruct"),
    "Qwen2.5-1.5B":    ("base",  ["base", "instruct"],       "instruct"),
    "Qwen2.5-3B":      ("base",  ["base", "instruct"],       "instruct"),
    "Qwen2.5-7B":      ("base",  ["base", "instruct"],       "instruct"),
    "Qwen2.5-14B":     ("base",  ["base", "instruct"],       "instruct"),
    "Qwen2.5-32B":     ("base",  ["base", "instruct"],       "instruct"),
    "Qwen2.5-72B":     ("base",  ["base", "instruct"],       "instruct"),
    "Gemma-3-4B":      ("base",  ["base", "instruct"],       "instruct"),
    "Gemma-3-12B":     ("base",  ["base", "instruct"],       "instruct"),
    "Gemma-3-27B":     ("base",  ["base", "instruct"],       "instruct"),
    "Llama-3.1-8B":    ("base",  ["base", "instruct"],       "instruct"),
    "Llama-3.1-70B":   ("base",  ["base", "instruct"],       "instruct"),
    "Mistral-7B-v0.3": ("base",  ["base", "instruct"],       "instruct"),
}
GENISLETME = {
    "Gemma-4-12B":       ("base", ["base", "instruct"], "instruct"),
    "Gemma-4-26B-A4B":   ("base", ["base", "instruct"], "instruct"),
    "Gemma-4-E4B":       ("base", ["base", "instruct"], "instruct"),
    "Mistral-Small-24B": ("base", ["base", "instruct"], "instruct"),
}
SOY = {"Tulu3-8B": "allenai", "OLMo2-13B": "allenai", "OLMo2-32B": "allenai",
       "OLMo-3-7B": "allenai"}

CIK = f"{ROOT}/results/PREREG9_A_OLCUM_2026-08-31.json"
GECICI = "<scratch>"

ONBELLEK: dict = {}
OKU_ASIL = MK.oku
_SIRA = {"n": 0, "t0": 0.0, "toplam": 0}


def sargi(nlp, R, _asil=MK.satir_bilesenleri):
    V = _asil(nlp, R)
    ONBELLEK[_SIRA["anahtar"]] = V
    return V


def _oku_sargi(a, b, _asil=MK.oku):
    _SIRA["anahtar"] = (a, b)
    R = _asil(a, b)
    if R is not None:
        _SIRA["n"] += 1
        gec = time.time() - _SIRA["t0"]
        if _SIRA["n"] == 1:
            print(f"  [ETA] ilk anahtar okundu; ölcüm sonrasi ETA basilacak", flush=True)
        elif _SIRA["n"] > 1:
            bir = gec / (_SIRA["n"] - 1)
            eta = bir * (_SIRA["toplam"] - _SIRA["n"] + 1) / 60
            ey = "DEVAM" if eta < 90 else ""
            print(f"  [ETA] {_SIRA['n']-1}/{_SIRA['toplam']} bitti · birim {bir:.0f} sn "
                  f"⇒ kalan ≈ {eta:.0f} dk ⇒ esik 90 dk ⇒ {ey}", flush=True)
    return R


def plasebo_yari_bolme(V, ist, n=200, seed=20260901):
    rng = np.random.default_rng(seed); ad = np.unique(ist); out = []
    for _ in range(n):
        s = rng.random(len(ad)) < 0.5
        A = np.isin(ist, ad[s]); B = ~A
        if A.sum() == 0 or B.sum() == 0:
            continue
        out.append(MK.olc_toplam(V[A])["M1"] - MK.olc_toplam(V[B])["M1"])
    o = np.array(out, dtype=float); o = o[~np.isnan(o)]
    return dict(n=int(len(o)), ort=float(o.mean()), sd=float(o.std()),
                p95=float(np.percentile(np.abs(o), 95)))


def _M1(W):
    return MK.olc_toplam(W)["M1"]


def plasebo_esli(V, ist, n=200, seed=20260901, olc=_M1):
    rng = np.random.default_rng(seed); out = []
    ad = np.unique(ist); ix = {u: np.where(ist == u)[0] for u in ad}
    for _ in range(n):
        A, B = [], []
        for u in ad:
            g = ix[u]; s = rng.random(len(g)) < 0.5
            A.append(g[s]); B.append(g[~s])
        A = np.concatenate(A); B = np.concatenate(B)
        if len(A) == 0 or len(B) == 0:
            continue
        out.append(olc(V[A]) - olc(V[B]))
    o = np.array(out, dtype=float); o = o[~np.isnan(o)]
    return dict(n=int(len(o)), ort=float(o.mean()), sd=float(o.std()),
                p95=float(np.percentile(np.abs(o), 95)))


def sahis1_1k(V):
    return 1000 * (V[:, 8].sum() - V[:, 1].sum()) / max(V[:, 0].sum(), 1)


def main(sadece=None):
    HEPSI = {**BIRINCIL, **GENISLETME}
    if sadece:
        HEPSI = {k: v for k, v in HEPSI.items() if k in sadece}
    _SIRA["toplam"] = sum(len(v[1]) for v in HEPSI.values())
    _SIRA["t0"] = time.time()
    print(f"{len(BIRINCIL)} {len(GENISLETME)}"
          f"{len(HEPSI)} {_SIRA['toplam']}", flush=True)
    os.makedirs(f"{GECICI}/results", exist_ok=True)
    MK.MERDIVEN = HEPSI; MK.ROOT = GECICI
    MK.satir_bilesenleri = sargi; MK.oku = _oku_sargi
    MK.kos()
    S = json.load(open(f"{GECICI}/results/prereg8_measurement_2026-09-01.json", encoding="utf-8"))

    for a, (bas, rungs, son) in HEPSI.items():
        if a not in S:
            continue
        Vb = ONBELLEK.get((a, bas))
        if Vb is None:
            S[a]["plasebo"] = "V-YOK"; continue
        R = OKU_ASIL(a, bas)
        ist = np.array([r.get("istem_i") for r in R])
        S[a]["plasebo_yari_bolme_M1"] = plasebo_yari_bolme(Vb, ist)
        S[a]["plasebo_ESLI_M1"] = plasebo_esli(Vb, ist)
        Vs = ONBELLEK.get((a, son))
        if Vs is not None:
            S[a]["jeton_ort"] = dict(bas=float(Vb[:, 0].mean()), son=float(Vs[:, 0].mean()))
            S[a]["d_jeton"] = float(Vs[:, 0].mean() - Vb[:, 0].mean())
            s1b, s1s = sahis1_1k(Vb), sahis1_1k(Vs)
            m1b = 1000 * Vb[:, 1].sum() / max(Vb[:, 0].sum(), 1)
            m1s = 1000 * Vs[:, 1].sum() / max(Vs[:, 0].sum(), 1)
            S[a]["sahis1_1k"] = dict(bas=s1b, son=s1s, d=s1s - s1b)
            S[a]["oran_2_1"] = dict(bas=(m1b / s1b) if s1b else None,
                                    son=(m1s / s1s) if s1s else None,
                                    d=((m1s / s1s) - (m1b / s1b)) if (s1b and s1s) else None)
            S[a]["cumle_basi_jeton"] = dict(bas=float(Vb[:, 0].sum() / max(Vb[:, -1].sum(), 1)),
                                            son=float(Vs[:, 0].sum() / max(Vs[:, -1].sum(), 1)))
        S[a]["soy"] = SOY.get(a, "dis")
        S[a]["sinif"] = "BIRINCIL" if a in BIRINCIL else "GENISLETME"
    S["_kunye"] = dict(
        damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prereg="preregistration/prereg_depersonalization16_2026-08-31.md",
        prereg_sha_errata_oncesi="aebc876a5461db68",
        motor="addressee_run.py b114a7b46a3369cb (EDIT YOK — ithal + yapilandirma)",
        kol="A · PANEL (k0, 34 istem)", n_birincil=len(BIRINCIL), n_genisletme=len(GENISLETME))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("muhatap9_A", n_aile_istenen=len(HEPSI), n_aile_olculen=len([k for k in S if not k.startswith("_")]),
          n_anahtar=_SIRA["toplam"], n_perm=200, n_bootstrap=1000, n_plasebo=200)
    print(f"\n✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
