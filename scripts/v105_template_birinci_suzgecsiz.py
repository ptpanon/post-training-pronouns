#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, os, subprocess, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
G = f"{ROOT}/results"
RULE = f"{ROOT}/preregistration/RULE_V105_TEMPLATE_BIRINCI_SUZGECSIZ_2026-09-23.md"
CIK = f"{G}/v105_template_birinci_suzgecsiz_2026-09-23.json"
CARD = f"{G}/template_filtered_2026-09-07.json"
ESIK = 1e-4


def sha(y):
    return hashlib.sha256(open(y, "rb").read()).hexdigest()


def rule_kapisi():
    h = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "-1", "--", os.path.relpath(RULE, ROOT)],
                       capture_output=True, text=True).stdout.strip()
    if not h:
        raise SystemExit("")
    b = sha(os.path.abspath(__file__))
    if b not in io.open(RULE, encoding="utf-8").read():
        raise SystemExit(f"{b[:16]}")
    return h, b[:16]


def main():
    import form_count as FS, addressee_run as MK, depersonalization16_neutral_run as M9
    import rung_force_ci as BK, reviewer_readings_v80 as H80
    import template_filtered as SS
    h, b = rule_kapisi()
    K = {r["aile"]: r for r in json.load(io.open(CARD, encoding="utf-8"))["aileler"]}
    CIFT, _ = SS.V23.cift_kur()
    nlp = FS._boru()
    M1 = lambda V: MK.olc_toplam(V)["M1"]
    S, red, t0, ETA = {}, [], time.time(), None
    for n, (a, zt, zh) in enumerate(CIFT, 1):
        Rt, Rh = SS.oku(SS.SAB, a, zt), SS.oku(SS.SAB, a, zh)
        if Rt is None or Rh is None or len(Rt) != SS.N_BEK or len(Rh) != SS.N_BEK:
            red.append(a); print(f"  ✗ {a}: template bacagi TAM degil"); continue
        At, Ah = [SS.anahtar(r) for r in Rt], [SS.anahtar(r) for r in Rh]
        if sorted(At) != sorted(Ah):
            red.append(a); print(f"{a}"); continue
        Vt, Vh = MK.satir_bilesenleri(nlp, Rt), MK.satir_bilesenleri(nlp, Rh)
        it = np.array([x.get("istem_i", -1) for x in Rt]); ih = np.array([x.get("istem_i", -1) for x in Rh])
        d2 = M1(Vh) - M1(Vt)
        bek = K[a]["sbl"]["dM1_suzgecsiz"] if a in K and "sbl" in K[a] else None
        sap = abs(d2 - bek) if bek is not None else float("nan")
        if bek is None or sap > ESIK:
            red.append(a); print(f"  ✗ {a}: ESDEGERLIK DÜSTÜ (süzgecsiz 2.sahis {d2:.4f} ↔ card {bek}) ⇒ ÖLCÜLMEZ")
            continue
        Wt = H80.ikame(Vt, H80.p1_sayim(Rt, "haric")); Wh = H80.ikame(Vh, H80.p1_sayim(Rh, "haric"))
        d1 = M1(Wh) - M1(Wt)
        lo, hi, nk = BK.kume_boot(Wh, ih, Wt, it, "M1")
        pl = M9.plasebo_esli(Wt, it, olc=M1)
        S[a] = dict(d1=round(float(d1), 4), ci=[round(lo, 4), round(hi, 4)], ayrik=bool(lo * hi > 0),
                    yon="asagi" if d1 < 0 else "yukari",
                    taban_1k=round(float(M1(Wt)), 4), hizali_1k=round(float(M1(Wh)), 4),
                    plasebo_ort=pl["ort"], plasebo_sd=pl["sd"], plasebo_p95=pl["p95"],
                    plasebo_ustu=bool(abs(d1) > pl["p95"]), n_istem=int(nk), n_satir=int(len(Rt)),
                    kapi_d2=round(float(d2), 4), kapi_card=bek, kapi_sapma=float(sap))
        if ETA is None:
            ETA = (time.time() - t0) * len(CIFT) / 60
            print(f"{(time.time()-t0)/60:.1f} {ETA:.1f}"
                  f"", flush=True)
        r = S[a]
        print(f"  ★ {a:16s} Δ1st {r['d1']:+7.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]"
              f"{' AYRIK' if r['ayrik'] else '      '} · plasebo p95 {r['plasebo_p95']:.2f}"
              f"{' ÜSTÜ' if r['plasebo_ustu'] else '     '} · taban {r['taban_1k']:6.2f} → {r['hizali_1k']:6.2f}"
              f" · kapi {r['kapi_sapma']:.1e}  [{n}/{len(CIFT)} · {(time.time()-t0)/60:.1f} dk]", flush=True)
    olc = sorted(S)
    asagi = [a for a in olc if S[a]["ayrik"] and S[a]["d1"] < 0]
    yukari = [a for a in olc if S[a]["ayrik"] and S[a]["d1"] > 0]
    nul = [a for a in olc if not S[a]["ayrik"]]
    pust = [a for a in olc if S[a]["plasebo_ustu"]]
    print(f"{len(CIFT)} {len(olc)} {len(red)} {red}"
          f"{len(asagi)} {len(yukari)} {len(nul)} {len(pust)}"
          f"")
    print(f"  ★ SÜZGECSIZ: birinci sahis {len(asagi)} of {len(CIFT)} modelde ayrik DÜSÜYOR "
          f"(süzülmüs okuma kâgitta 16 of 16)")
    json.dump(dict(sinif="BETIM · bar YOK · sahip kelimesi «EXECUTOR · v105» (1)", rule=h, alet=b,
                   alet_yolu="scripts/v105_template_birinci_suzgecsiz.py",
                   olcu="",
                   kapi=dict(ad="süzgecsiz 2.sahis ΔM1 ≡ TEMPLATE_FILTERED sbl.dM1_suzgecsiz", esik=ESIK, red=red),
                   plasebo="depersonalization16_neutral_run.plasebo_esli, TABAN bacagi, n=200, seed 20260901",
                   sayim=dict(n_aile=len(CIFT), olculen=len(olc), asagi_ayrik=asagi, yukari_ayrik=yukari,
                              null=nul, plasebo_ustu=pust),
                   aile=S, damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
              io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0 if not red else 3


if __name__ == "__main__":
    sys.exit(main())
