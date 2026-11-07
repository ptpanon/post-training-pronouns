#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, random, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import filtre_verim as FV
import addressee_olcu as MO
import form_count as FS

CIK = f"{ROOT}/results/pool_person_delta_2026-09-06.json"
NB, ETA_ESIK_DK = 1000, 90.0
TAM = "--tam" in sys.argv
if TAM:
    CIK = f"{ROOT}/results/pool_person_delta_full_2026-09-06.json"
    ETA_ESIK_DK = 150.0


def _prova():
    nlp = FS._boru()
    A, _ = MO.topla(nlp, ["you should check your work", "the result is fine"])
    p2 = list(A["m1_sahis2"]); jt = list(A["n_jeton"])
    return {"i_ikinci_sahis_sayiliyor": bool(p2[0] >= 2),
            "ii_sahissiz_sifir": bool(p2[1] == 0),
            "iii_jeton_pozitif": bool(all(j > 0 for j in jt)),
            "iv_bos_metin_cokmuyor": bool(MO.topla(nlp, [""])[0]["n_jeton"][0] >= 0)}, nlp


def bin_jetonda(A, i0, i1):
    p2 = np.asarray(A["m1_sahis2"], float)[i0:i1]
    p1 = np.asarray(A["m5_yakin"], float)[i0:i1] - p2
    jt = np.maximum(np.asarray(A["n_jeton"], float)[i0:i1], 1.0)
    return 1000.0 * p2 / jt, 1000.0 * p1 / jt, jt


def boot(d, seed=20260906):
    rng = np.random.default_rng(seed)
    o = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)])
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main():
    P, nlp = _prova()
    print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    if not all(P.values()):
        return 4
    rng = random.Random(FV.SEED)
    OUT, t_hep = {}, time.time()
    for s in FV.SETLER:
        C, R, N, red = FV.yukle(s, None if TAM else 20000, rng)
        n = len(C)
        t0 = time.time(); MO.topla(nlp, C[:500], n_process=6); dt = time.time() - t0
        hiz = 500 / max(dt, 1e-9)
        eta = (2 * n / hiz) / 60
        print(f"{s['ad']} {hiz:.0f}"
              f"{eta:.1f} {ETA_ESIK_DK}"
              f"", flush=True)
        if eta > ETA_ESIK_DK:
            OUT[s["ad"]] = dict(HAL="ATLANDI-BÜTCE", eta_dk=round(eta, 1),
                                n_cift=n, n_havuz=N)
            continue
        A, _ = MO.topla(nlp, C + R, n_process=6)
        c2, c1, cj = bin_jetonda(A, 0, n)
        r2, r1, rj = bin_jetonda(A, n, 2 * n)
        d2, d1 = c2 - r2, c1 - r1
        hav2 = 1000.0 * (np.asarray(A["m1_sahis2"], float)[:n].sum() / cj.sum()
                         - np.asarray(A["m1_sahis2"], float)[n:2 * n].sum() / rj.sum())
        ci2, ci1 = boot(d2), boot(d1)
        sif = float((d2 == 0).mean())
        OUT[s["ad"]] = dict(
            HAL="ÖLCÜLDÜ", repo=s["repo"], n_havuz=N, n_cift=n, red_bos_taraf=red,
            d_M1=float(d2.mean()), d_M1_medyan=float(np.median(d2)), d_M1_ci=ci2,
            d_M1_havuz=float(hav2), kestirici_notu="d_M1 = cift ortalamasi; "
            "d_M1_havuz = korpus-düzeyi oran farki (ikisi AYRI nesnedir)",
            d_sahis1=float(d1.mean()), d_sahis1_ci=ci1,
            tam_sifir_pay=sif, jeton_ort_chosen=float(cj.mean()),
            jeton_ort_rejected=float(rj.mean()))
        o = OUT[s["ad"]]
        print(f"  ★ {s['ad']:14s} n={n:,}/{N:,} · Δ2.sahis = {o['d_M1']:+.3f}/1k "
              f"CI [{ci2[0]:+.3f},{ci2[1]:+.3f}] · Δ1.sahis = {o['d_sahis1']:+.3f} "
              f"· HAVUZ {o['d_M1_havuz']:+.3f} · medyan {o['d_M1_medyan']:+.3f} "
              f"· tam-sifir %{100*sif:.0f}", flush=True)

    olculen = [k for k, v in OUT.items() if v["HAL"] == "ÖLCÜLDÜ"]
    n_top = sum(OUT[k]["n_cift"] for k in olculen)
    n_hav = sum(OUT[k]["n_havuz"] for k in OUT)
    ayrik = [k for k in olculen if OUT[k]["d_M1_ci"][0] * OUT[k]["d_M1_ci"][1] > 0]
    asagi = [k for k in ayrik if OUT[k]["d_M1"] < 0]
    print(f"{len(OUT)} {len(olculen)} {n_top:,}"
          f"{n_hav:,} {len(ayrik)}"
          f"{len(asagi)}"
          f"")
    verdict = ("YÖN-YOK" if not ayrik else
             ("KÖKEN-DESTEKLENDI" if len(asagi) == len(ayrik) else
              "KÖKEN-DESTEKLENMEDI"))
    print(f"★ VERDICT: {verdict} (yön beyani kosudan ÖNCE yaziliydi: Δ<0 destekler)")
    S = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         alet="scripts/pool_person_delta.py",
                         borc="D-0907-V19-B §2e", SINIF="ÖLCÜM — köken kontrolü",
                         kip=("TAM-HAVUZ" if TAM else "ÖRNEKLEM"),
                         seed_ornekleme=FV.SEED, seed_boot=20260906, nb=NB,
                         birim="cift", n_cift_olculen=n_top, n_havuz=n_hav,
                         ONCUL_DUZELTME=("494 833 HAVUZUN boyudur; form sayaclari "
                                         "da bu 59 894'lük örneklemde ölcülmüstü "
                                         "(FILTRE_VERIM_2026-08-29 n_ornek)"),
                         prova=P),
             set=OUT, _verdict=verdict)
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK} · toplam {(time.time()-t_hep)/60:.1f} dk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
