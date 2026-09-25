#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, json, os, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import form_count as FS
import addressee_olcu as MO
VERI = __DNH_DATA__ + "/c1_panel"
MERDIVEN = {
    "Tulu3-8B":      ("sft", ["sft", "dpo", "rl"],       "rl"),
    "OLMo2-13B":     ("sft", ["sft", "dpo", "instruct"], "instruct"),
    "OLMo2-32B":     ("sft", ["sft", "dpo", "instruct"], "instruct"),
    "OLMo2-7B-RLVR": ("dpo", ["dpo", "step_60", "step_360"], "step_360"),
}
ALAN = ("n_jeton", "m1_sahis2", "m2", "duzeltme", "duzeltme_kisisiz",
        "m4_volitif", "n_modal", "m5_uzak", "m5_yakin", "kuvvet")


def oku(a, b, kok=None):
    y = f"{kok or VERI}/{a}/{b}/uretim.jsonl"
    if not os.path.exists(y):
        return None
    return [json.loads(l) for l in open(y, encoding="utf-8")]


def satir_bilesenleri(nlp, R):
    M = [r["metin"] for r in R]
    A, nc = MO.topla(nlp, M, n_process=6)
    V = np.zeros((len(R), len(ALAN) + 1))
    for j, k in enumerate(ALAN):
        V[:, j] = A[k]
    V[:, -1] = nc
    return V


def olc_toplam(V):
    s = V.sum(axis=0); d = dict(zip(ALAN, s[:-1])); c = s[-1]
    return dict(
        M1=1000 * d["m1_sahis2"] / max(d["n_jeton"], 1),
        M2=d["m2"] / max(c, 1),
        M3=(d["duzeltme_kisisiz"] / d["duzeltme"]) if d["duzeltme"] > 0 else np.nan,
        M4=(d["m4_volitif"] / d["n_modal"]) if d["n_modal"] > 0 else np.nan,
        M5=(d["m5_uzak"] - d["m5_yakin"]) / max(d["n_jeton"], 1),
        KUVVET=d["kuvvet"] / max(c, 1))


def kos():
    nlp = FS._boru(); t0 = time.time(); SON = {}
    for a, (bas, rungs, son) in MERDIVEN.items():
        V, anah = {}, {}
        for b in rungs:
            R = oku(a, b)
            if R is None:
                print(f"{a} {b}"); continue
            V[b] = satir_bilesenleri(nlp, R)
            anah[b] = [(r.get("kol"), r.get("cekim"), r.get("istem_i")) for r in R]
            print(f"  [{time.time()-t0:.0f}s] {a}/{b}: {len(R)} satir", flush=True)
        if bas not in V or son not in V:
            print(f"{a} {bas} {son}"); continue
        tab = {b: olc_toplam(V[b]) for b in V}
        i0 = {k: j for j, k in enumerate(anah[bas])}
        cift = [(i0[k], j) for j, k in enumerate(anah[son]) if k in i0]
        A0 = V[bas][[x for x, _ in cift]]; A1 = V[son][[y for _, y in cift]]
        ist = np.array([anah[son][y][2] for _, y in cift])
        gz = {k: olc_toplam(A1)[k] - olc_toplam(A0)[k] for k in ("M1", "M2", "M3", "M4", "M5", "KUVVET")}
        rng = np.random.default_rng(20260901); N = {k: [] for k in gz}
        for _ in range(200):
            s = rng.random(len(cift)) < 0.5
            P0 = np.where(s[:, None], A1, A0); P1 = np.where(s[:, None], A0, A1)
            d = {k: olc_toplam(P1)[k] - olc_toplam(P0)[k] for k in gz}
            for k in gz:
                N[k].append(d[k])
        ad = np.unique(ist); B = []
        for _ in range(1000):
            sel = rng.choice(ad, len(ad), replace=True)
            ix = np.concatenate([np.where(ist == u)[0] for u in sel])
            B.append({k: olc_toplam(A1[ix])[k] - olc_toplam(A0[ix])[k] for k in gz})
        SON[a] = dict(basamak_tablosu=tab, kontrast=f"{bas}→{son}", n_cift=len(cift),
                      n_istem=int(len(ad)), fark={}, )
        for k in gz:
            nn = np.array(N[k], dtype=float); nn = nn[~np.isnan(nn)]
            bb = np.array([x[k] for x in B], dtype=float); bb = bb[~np.isnan(bb)]
            SON[a]["fark"][k] = dict(
                gozlenen=float(gz[k]),
                null_ort=float(np.mean(nn)) if len(nn) else None,
                null_sd=float(np.std(nn)) if len(nn) else None,
                frac_null_asan=float(np.mean(np.abs(nn) >= abs(gz[k]))) if len(nn) else None,
                ci=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))] if len(bb) else None)
    json.dump(SON, open(f"{ROOT}/results/prereg8_measurement_2026-09-01.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    payda("muhatap_kos", n_merdiven=len(SON), n_perm=200, n_bootstrap=1000)
    print(f"\n✓ results/prereg8_measurement_2026-09-01.json  ({time.time()-t0:.0f} sn)")
    return 0


if __name__ == "__main__":
    sys.exit(kos())
