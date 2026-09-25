#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import degeneracy_threshold_floor as DT

PANEL = __DNH_DATA__ + "/c1_panel"
K = lambda a: f"{ROOT}/results/{a}"
CIK = K("v22_cheap_measurements_2026-09-06.json")
D4_BAR = 0.60
NPERM = 5000
SEED = 20260906


def _prova():
    x = np.arange(16.0)
    r_ozdes = _spearman(x, x)
    r_ters = _spearman(x, -x)
    return {"i_ozdes_bir": bool(abs(r_ozdes - 1.0) < 1e-9),
            "ii_ters_eksi_bir": bool(abs(r_ters + 1.0) < 1e-9),
            "iii_farkli4_kisa_None": DT.farkli4("") is None,
            "iv_farkli4_tekrar": bool(DT.farkli4("a b c d " * 40) < 0.2)}


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def b_spearman(OUT):
    T = json.load(open(K("swapped_ruler_2026-09-03.json"), encoding="utf-8"))
    aile = sorted(k for k in T if not k.startswith("_"))
    kol = {"on_kayitli": ("asil_M1_1k", "asil_KUVVET_cumle"),
           "cumle_basina": ("ters_M1_cumle", "asil_KUVVET_cumle")}
    rng = np.random.default_rng(SEED); R = {}
    for ad, (km, kk) in kol.items():
        m = np.array([T[a][km]["gozlenen"] for a in aile])
        f = np.array([T[a][kk]["gozlenen"] for a in aile])
        r = _spearman(m, f)
        null = np.array([_spearman(m, rng.permutation(f)) for _ in range(NPERM)])
        R[ad] = dict(rho=r, n=len(aile), null_ort=float(null.mean()),
                     null_sd=float(null.std(ddof=1)),
                     frac_null_asan=float(np.mean(np.abs(null) >= abs(r))),
                     sd_kati=float((r - null.mean()) / null.std(ddof=1)))
        print(f"  ★ (b) {ad:12s} ρ={r:+.3f} · null {null.mean():+.3f}±"
              f"{null.std(ddof=1):.3f} · frac(|null|≥|ρ|)={R[ad]['frac_null_asan']:.4f} "
              f"· {R[ad]['sd_kati']:+.2f} sd · n={len(aile)}", flush=True)
    print(f""
          f"")
    OUT["b_spearman"] = R


def c_dejenerelik(OUT):
    T = json.load(open(K("swapped_ruler_2026-09-03.json"), encoding="utf-8"))
    aile = sorted(k for k in T if not k.startswith("_"))
    R = {}
    for a in aile:
        bac = {}
        for yol in sorted(glob.glob(f"{PANEL}/{a}/*/uretim.jsonl")):
            z = os.path.basename(os.path.dirname(yol))
            ham = [DT.farkli4(json.loads(l).get("metin") or "")
                   for l in open(yol, encoding="utf-8")]
            d4 = np.array([x for x in ham if x is not None], float)
            bac[z] = dict(n_satir=int(len(ham)), n_olculebilir=int(len(d4)),
                          n_kisa_olculemez=int(len(ham) - len(d4)),
                          alt_bar=int((d4 < D4_BAR).sum()),
                          oran=float((d4 < D4_BAR).mean()), ort=float(d4.mean()))
        R[a] = dict(bacak=bac,
                    cift_dejenere=T[a]["dejenere"], filtreli=T[a]["filtreli_M1_1k"],
                    ham=T[a]["asil_M1_1k"])
        say = " · ".join(f"{z} {v['oran']*100:.2f}%" for z, v in sorted(bac.items()))
        print(f"  ★ (c) {a:18s} {say} | ham ΔM1 {T[a]['asil_M1_1k']['gozlenen']:+.2f} "
              f"→ süzgecli {T[a]['filtreli_M1_1k']['gozlenen']:+.2f} "
              f"[{T[a]['filtreli_M1_1k']['ci'][0]:+.2f},{T[a]['filtreli_M1_1k']['ci'][1]:+.2f}] "
              f"{'AYRIK' if T[a]['filtreli_M1_1k']['ayrik'] else '★SIFIR ICERIYOR'}", flush=True)
    ay = sum(1 for a in aile if T[a]["filtreli_M1_1k"]["ayrik"])
    print(f"  [PAYDA] (c) aile={len(aile)} · süzgecli CI-ayrik={ay} ⇒ esik 16 ⇒ "
          f"EYLEM: altindaysa «süzgec sonrasi 16/16» cümlesi düser")
    OUT["c_dejenerelik"] = dict(bar=D4_BAR, aile=R, suzgecli_ayrik=ay, n=len(aile))


def f_tarihler(OUT):
    T = json.load(open(K("swapped_ruler_2026-09-03.json"), encoding="utf-8"))
    aile = sorted(k for k in T if not k.startswith("_"))
    R = {}
    for a in aile:
        for yol in sorted(glob.glob(f"{PANEL}/{a}/*/uretim.jsonl")):
            z = os.path.basename(os.path.dirname(yol))
            R[f"{a}/{z}"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime(os.path.getmtime(yol)))
    ilk, son = min(R.values()), max(R.values())
    print(f"{ilk} {son} {len(R)}")
    print(f"{len(R)}"
          f"")
    OUT["f_tarihler"] = dict(dosya=R, ilk=ilk, son=son, n=len(R))


def g_z(OUT):
    H = json.load(open(K("DIAL_VERDICT_2026-09-04.json"), encoding="utf-8"))
    sd_m1 = 1.1 / 2.0
    R = {}
    for kol, v in H["kol"].items():
        R[kol] = dict(dM1_B=v["dB"], z_M1=v["dB"] / sd_m1,
                      dKUVVET=v["dKUVVET"], z_kuvvet=v["z_kuvvet"])
        print(f"  ★ (g) {kol:14s} ΔM1 {v['dB']:+.2f} ⇒ z={v['dB']/sd_m1:+.1f} "
              f"(null-sd {sd_m1:.3f} = bar/2) · Δforce z={v['z_kuvvet']:+.1f}")
    print(f"  [PAYDA] (g) kol={len(R)} · null-sd kaynagi = PREREG «BAR = 2 × üst "
          f"seed sd» ⇒ esik: |z|<1 «yerinde» ⇒ EYLEM: z basilmadan «most "
          f"movable» cümlesi düser")
    OUT["g_z"] = dict(null_sd_M1=sd_m1, kol=R, kaynak="PREREG_KISI_KADRANI §BAR")


def i_bant(OUT):
    A = json.load(open(K("PREREG9_A_OLCUM_2026-08-31.json"), encoding="utf-8"))
    S = json.load(open(K("four_seed_reading_2026-09-01.json"), encoding="utf-8"))
    I = json.load(open(K("INSAN_BANDI_2026-08-31.json"), encoding="utf-8"))
    hiz = {}
    for k, v in A.items():
        if k.startswith("_") or not isinstance(v, dict) or "kontrast" not in v:
            continue
        if k not in S:
            continue
        z1 = v["kontrast"].split("→")[1]
        hiz[k] = v["basamak_tablosu"][z1]["M1"]
    en_ust = max(hiz, key=hiz.get)
    kon = {"SCOTUS-advocates": I["SCOTUS"]["A"], "SCOTUS-justices": I["SCOTUS"]["J"],
           "CMV": I["CMV"]["hepsi"]}
    en_alt = min(kon, key=lambda k: kon[k]["M1"])
    sd = S[en_ust]["m4_sd"] if "m4_sd" in S[en_ust] else float("nan")
    ust = hiz[en_ust] + 1.645 * sd
    print(f"  ★ (i) en yüksek hizali: {en_ust} M1={hiz[en_ust]:.2f} · seed sd="
          f"{sd:.3f} ⇒ tek-tarafli %95 üst {ust:.2f}")
    print(f"       en düsük konusma grubu: {en_alt} M1={kon[en_alt]['M1']:.2f} "
          f"CI {[round(x,2) for x in kon[en_alt]['M1_ci']]}")
    asiyor = bool(ust >= kon[en_alt]["M1_ci"][0])
    print(f"{len(hiz)} {len(kon)}"
          f""
          f"{'GIRIYOR ⇒ «at or below»' if asiyor else ''}")
    OUT["i_bant"] = dict(hizali_M1=hiz, en_ust_aile=en_ust, seed_sd=sd,
                         ust_95=ust, konusma={k: v["M1"] for k, v in kon.items()},
                         en_alt_grup=en_alt, en_alt_ci=kon[en_alt]["M1_ci"],
                         ortusuyor=asiyor)


def main():
    P = _prova(); print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False), flush=True)
    if not all(P.values()):
        print("★★ PROVA DÜSTÜ"); return 4
    OUT = {}; t0 = time.time()
    b_spearman(OUT); print()
    c_dejenerelik(OUT); print()
    f_tarihler(OUT); print()
    g_z(OUT); print()
    i_bant(OUT)
    OUT["_kunye"] = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         alet="scripts/v22_cheap_measurements.py", borc="D-0907-V22 §3 b,c,f,g,i",
                         prova=P, nperm=NPERM, seed=SEED, d4_bar=D4_BAR)
    json.dump(OUT, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n✓ {CIK} · {(time.time()-t0)/60:.1f} dk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
