#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, argparse, datetime
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from n2_maske_alim import (oku_alim, cift_kur, uclu, verdict_maske, KOMSU,
                           ZEMINLER, N_ASGARI, BAR_TUTARLILIK, ADLAR_MASKE)
from full_tarama import full_tarama as TT, bas as bas_tt, izgara
import miras_gate as MK

CIKTI = f"{ROOT}/results/O2_MASKE_IKI_OKUMA_2026-08-20.json"
Z = 1.645

OLCULEMEZ_DOGAR = (
    "esli cift < 200                                     → payda",
    "sd (kendi biriminde) ≤ 0                            → sd sifir",
    "ham ↔ jeton-normalize TERS ISARET                   → uzunluk-karisimi (sigorta-i, AYNEN)",
    "isaret-tutarliligi < 0,65                           → YÖN-DÖNEN (iki havuz ayri)",
    "|ort_norm| < 1,645·sd_norm/√n                       → MERDIVEN-OKUNMUYOR·MDE-alti",
    "",
)


def verdict_maske_v2(n, ters_isaret, ort_norm, sd_norm, tutarlilik):
    if n < N_ASGARI:
        return "ÖLCÜLEMEZ", "payda"
    if sd_norm is None or sd_norm <= 0:
        return "ÖLCÜLEMEZ", "sd sifir"
    if ters_isaret:
        return "ÖLCÜLEMEZ", "uzunluk-karisimi (ham ↔ jeton-normalize ters)"
    if tutarlilik < BAR_TUTARLILIK:
        return "YÖN-DÖNEN", f"tutarlilik {tutarlilik:.3f} < {BAR_TUTARLILIK}"
    if abs(ort_norm) < Z * sd_norm / np.sqrt(n):
        return "MERDIVEN-OKUNMUYOR", "MDE-alti"
    return "MERDIVEN-OKUNUYOR", ""


def tarama():
    R = TT(verdict_maske_v2,
           {"n": [0, 199, 200, 800], "ters_isaret": [False, True],
            "ort_norm": izgara(-0.5, 0.5, 7), "sd_norm": [0.0, 0.01, 0.2, 1.0],
            "tutarlilik": izgara(0.5, 1.0, 6)},
           ADLAR_MASKE, "O-2(a) · birim-esli rule")
    bas_tt(R)
    return R


def miras():
    return MK.bas(MK.denetle(
        ["payda", "yer-tutucu", "birim-esligi", "kill-hareketli",
         "tavan-yapisik-null", "yokluk-hukmu", "taban-sifir-degil"],
        gerekmeyen={
            "rejim-kurulamadi": ""
                                "",
            "tavan-olculdu": "MDE tabani ölcülen sd'den; tavan G-kutusu emsaliyle "
                             "PREREG_N2 §4'te ölcülmüstü (merdiven_komsu 1,0000)",
            "bar-bicimi": ""
                          "",
        }))


def kos():
    sat, pd = [], {"n_cift_uzay": len(ZEMINLER) * len(KOMSU), "n_cift_kurulan": 0,
                   "red_shard_yok": 0, "n_ad_degisen": 0, "n_ad_ayni": 0}
    for zemin in ZEMINLER:
        for a, b in KOMSU:
            A, B = oku_alim(zemin, a), oku_alim(zemin, b)
            if A is None or B is None:
                pd["red_shard_yok"] += 1
                sat.append({"zemin": zemin, "cift": f"{a}-{b}",
                            "ESKI": "", "YENI": "",
                            "DEGISTI": False})
                continue
            C = cift_kur(A, B)
            pd["n_cift_kurulan"] += 1
            ham, nrm = uclu(C["d_ham"]), uclu(C["d_norm"])
            ters = bool(np.sign(ham["ort"]) * np.sign(nrm["ort"]) < 0)
            e_ad, e_nd = verdict_maske(ham["n"], ters, ham["ort"], nrm["ort"],
                                     ham["isaret_tutarliligi"], ham["sd"])
            y_ad, y_nd = verdict_maske_v2(nrm["n"], ters, nrm["ort"], nrm["sd"],
                                        nrm["isaret_tutarliligi"])
            ayni = e_ad == y_ad
            pd["n_ad_ayni" if ayni else "n_ad_degisen"] += 1
            sat.append({
                "zemin": zemin, "cift": f"{a}-{b}", "n": ham["n"],
                "d_jeton_SABIT": round(float(C["d_jeton"].mean()), 3),
                "d_jeton_sd": round(float(C["d_jeton"].std()), 6),
                "HAM": {k: ham[k] for k in ("ort", "sd", "isaret_tutarliligi",
                                            "HAVUZ_POZ", "HAVUZ_NEG", "FARK", "TOPLAM")},
                "NORM": {k: nrm[k] for k in ("ort", "sd", "isaret_tutarliligi",
                                             "HAVUZ_POZ", "HAVUZ_NEG", "FARK", "TOPLAM")},
                "MDE_ham_birimi": round(float(Z * ham["sd"] / np.sqrt(ham["n"])), 6),
                "MDE_norm_birimi": round(float(Z * nrm["sd"] / np.sqrt(nrm["n"])), 6),
                "sisme_kat": round(float(ham["sd"] / nrm["sd"]), 2) if nrm["sd"] else None,
                "ters_isaret": ters,
                "ESKI": f"{e_ad}·{e_nd}" if e_nd else e_ad,
                "YENI": f"{y_ad}·{y_nd}" if y_nd else y_ad,
                "DEGISTI": not ayni})
    return {"damga": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sinif": "ERRATA-OKUMASI · iki okuma yan yana; eski verdict DÜSMEZ",
            "onarim": "verdict_maske(…, sd_ham) → verdict_maske_v2(…, sd_norm) — birim esligi",
            "PAYDA": pd, "satirlar": sat}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("tarama", "miras", "kos"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args()
    if a.tarama:
        tarama()
    if a.miras:
        miras()
    if a.kos:
        R = kos()
        json.dump(R, open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(R["PAYDA"], ensure_ascii=False))
        for s in R["satirlar"]:
            if "HAM" not in s:
                print(f"  {s['zemin']:16s}{s['cift']:6s} ⇒ {s['ESKI']}"); continue
            m = "★" if s["DEGISTI"] else " "
            print(f" {m} {s['zemin']:16s}{s['cift']:6s} n={s['n']:4d} Δjet={s['d_jeton_SABIT']:+.0f} "
                  f"| NORM ort={s['NORM']['ort']:+.5f} tut={s['NORM']['isaret_tutarliligi']:.3f} "
                  f"| MDE {s['MDE_ham_birimi']:.4f}→{s['MDE_norm_birimi']:.5f} ({s['sisme_kat']}×) "
                  f"| {s['ESKI']:44s} → {s['YENI']}")
        print("★", CIKTI)
    if not (a.tarama or a.miras or a.kos):
        print("★ --tarama · --miras · --kos")
