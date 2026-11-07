#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
from person_dial_pair_sagligi import red_var
from reading_style_object import payda

KOL_KOK = __DNH_DATA__ + "/dial_kol"
CIK = f"{KOK}/results/matched_arm_gate_2026-09-10.json"
N_HEDEF = 7537
HEDEF_RED, HEDEF_JETON = -0.0386, -23.72
BAND_RED, BAND_JETON = 0.0097, 5.93
SEED = 20260910


def _son_asistan(m):
    for x in reversed(m or []):
        if x.get("role") == "assistant":
            return x.get("content") or ""
    return ""


def main():
    yol = f"{KOL_KOK}/RASTGELE.jsonl"
    if not os.path.exists(yol):
        print(""); return 4
    P = []
    for l in io.open(yol, encoding="utf-8"):
        d = json.loads(l)
        c, r = _son_asistan(d["chosen"]), _son_asistan(d["rejected"])
        P.append(dict(id=d["id"],
                      dM1=float(d["M1_chosen"]) - float(d["M1_rejected"]),
                      djeton=float(d["n_jeton_chosen"]) - float(d["n_jeton_rejected"]),
                      dred=float(red_var(c)) - float(red_var(r))))
    dM1 = np.array([p["dM1"] for p in P])
    sd = float(dM1.std())
    MDE_CIFT = 1.645 * sd
    notr = list(P)
    print(f"  [PAYDA] esli_kapi_havuz: n_ornek={len(P)} · hal_sd_dM1={sd:.4f} · "
          f"hal_MDE_cift={MDE_CIFT:.4f} · hal_notr={len(notr)} "
          f"({100*len(notr)/max(len(P),1):.1f}%) ⇒ esik: nötr oran × havuz < "
          f"{N_HEDEF} ⇒ EYLEM: ESLEME-KURULAMADI")
    rng = np.random.default_rng(SEED)
    kalan = list(notr)
    rng.shuffle(kalan)
    sec, top_red, top_jet, top_m1 = [], 0.0, 0.0, 0.0
    for p in kalan:
        n1 = len(sec) + 1
        skor = []
        for yon in (+1, -1):
            r_ = (top_red + yon * p["dred"]) / n1
            j_ = (top_jet + yon * p["djeton"]) / n1
            skor.append((abs(r_ - HEDEF_RED) / max(BAND_RED, 1e-9)
                         + abs(j_ - HEDEF_JETON) / max(BAND_JETON, 1e-9), yon))
        yon = min(skor)[1]
        sec.append(dict(p, yon=yon)); top_red += yon * p["dred"]
        top_jet += yon * p["djeton"]; top_m1 += yon * p["dM1"]
        if len(sec) >= N_HEDEF:
            break
    n = max(len(sec), 1)
    dm1_sec = np.array([x["yon"] * x["dM1"] for x in sec], float)
    ort_m1 = float(dm1_sec.mean()); sd_sec = float(dm1_sec.std())
    notr_bar = 1.645 * sd / (n ** 0.5)
    ulasilan = dict(n=len(sec), dred=top_red / n, djeton=top_jet / n,
                    ort_dM1=ort_m1, sd_dM1=sd_sec, notr_bar=notr_bar,
                    medyan_dM1=float(np.median(np.abs(dm1_sec))))
    tut_red = abs(ulasilan["dred"] - HEDEF_RED) <= BAND_RED
    tut_jet = abs(ulasilan["djeton"] - HEDEF_JETON) <= BAND_JETON
    tut_n = len(sec) >= N_HEDEF
    tut_notr = (abs(ort_m1) <= notr_bar) and (sd_sec <= sd)
    print(f"  [PAYDA] esli_kapi_notrluk: hal_ort_dM1={ort_m1:+.4f} · "
          f"esik={notr_bar:.4f} · hal_sd_secilen={sd_sec:.3f} · esik_sd={sd:.3f} "
          f"⇒ {'NÖTR' if tut_notr else 'NÖTR-DEGIL'} ⇒ EYLEM: degilse ESLEME-KURULAMADI")
    HAL = ("ESLEME-KURULUR" if (tut_red and tut_jet and tut_n and tut_notr)
           else "ESLEME-KURULAMADI")
    print(f"  [PAYDA] esli_kapi_eslesme: n_secilen={len(sec)}/{N_HEDEF} · "
          f"hal_dred={ulasilan['dred']:+.4f} (hedef {HEDEF_RED:+.4f} ±{BAND_RED}) · "
          f"hal_djeton={ulasilan['djeton']:+.2f} (hedef {HEDEF_JETON:+.2f} ±{BAND_JETON}) · "
          f"red_band_disi={int(not tut_red)+int(not tut_jet)} ⇒ {HAL}")
    D = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         alet="scripts/matched_arm_gate.py",
                         prereg="preregistration/prereg_matched_arm_2026-09-10.md @ 5090356c",
                         ornek=yol, n_ornek=len(P), seed=SEED,
                         sinir=(""
                                "")),
             sd_dM1_cift=round(sd, 4), MDE_cift=round(MDE_CIFT, 4),
             n_aday=len(notr), notrluk_olcutu=("KOL birimi: |ort(dM1)| ≤ 1,645·sd/√n "
                                               "VE sd(secilen) ≤ sd(havuz) — ERRATA-1"),
             hedef=dict(dred=HEDEF_RED, djeton=HEDEF_JETON,
                        band_red=BAND_RED, band_jeton=BAND_JETON),
             ulasilan={k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in ulasilan.items()},
             HAL=HAL)
    json.dump(D, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("m_esli_kol_kapisi", n_ornek=len(P), hal_notr=len(notr),
          hal_secilen=len(sec), red_band_disi=int(not tut_red) + int(not tut_jet))
    print(f"✓ {CIK} · {HAL}")
    return 0 if HAL == "ESLEME-KURULUR" else 3


if __name__ == "__main__":
    sys.exit(main())
