#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, argparse, datetime
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from full_tarama import full_tarama as TT, bas, izgara

ARSIV = __DNH_DATA__ + "/onek_korpus/aktivasyon_arsivi"
CIKTI = f"{ROOT}/results/N2_MASKE_ALIM_ADIM1.json"
ZEMINLER = ("mistral_base", "mistral_it_ham")
KOMSU = (("B0", "B1"), ("B1", "B2"), ("B2", "B3"), ("B3", "B4"))

N_ASGARI = 200
BAR_TUTARLILIK = 0.65
BAR_IG_TERS = 0.0


ADLAR_MASKE = ("MERDIVEN-OKUNUYOR", "MERDIVEN-OKUNMUYOR",
               "YÖN-DÖNEN", "ÖLCÜLEMEZ")

OLCULEMEZ_DOGAR = (
    ""
    "",
    "esli cift sayisi < 200                                 → payda",
    "ham ↔ jeton-normalize satirlari TERS ISARET            → uzunluk-karisimi",
    ""
    "",
    "",
    ""
    "",
)


def oku_alim(zemin, basamak):
    p = f"{ARSIV}/merdiven__{zemin}__{basamak}__p00.kayip.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {"ix": np.asarray(z["satir_ix"]),
            "kayip": np.asarray(z["kayip_onek_ort"], float),
            "tok": np.asarray(z["n_tok_onek"], float)}


def cift_kur(A, B):
    ia = {v: i for i, v in enumerate(A["ix"])}
    ib = {v: i for i, v in enumerate(B["ix"])}
    ort = np.intersect1d(A["ix"], B["ix"])
    if len(ort) == 0:
        return None
    ka = np.array([A["kayip"][ia[v]] for v in ort])
    kb = np.array([B["kayip"][ib[v]] for v in ort])
    ta = np.array([A["tok"][ia[v]] for v in ort])
    tb = np.array([B["tok"][ib[v]] for v in ort])
    return {"n": len(ort),
            "d_ham": (kb * tb) - (ka * ta),
            "d_norm": kb - ka,
            "d_jeton": tb - ta, "ka": ka, "kb": kb, "ta": ta, "tb": tb}


def uclu(d):
    poz, neg = d[d > 0], d[d < 0]
    cog = max(len(poz), len(neg)) / max(len(d), 1)
    return {"n": int(len(d)), "ort": round(float(d.mean()), 8),
            "sd": round(float(d.std(ddof=1)), 8) if len(d) > 1 else None,
            "isaret_tutarliligi": round(float(cog), 4),
            "HAVUZ_POZ": {"n": int(len(poz)),
                          "ort": round(float(poz.mean()), 8) if len(poz) else None},
            "HAVUZ_NEG": {"n": int(len(neg)),
                          "ort": round(float(neg.mean()), 8) if len(neg) else None},
            "FARK": round(float((poz.mean() if len(poz) else 0.0)
                                - (neg.mean() if len(neg) else 0.0)), 8),
            "TOPLAM": round(float(d.sum()), 6)}


def verdict_maske(n, ters_isaret, ort_ham, ort_norm, tutarlilik, sd_ham):
    if n < N_ASGARI:
        return "ÖLCÜLEMEZ", "payda"
    if sd_ham is None or sd_ham <= 0:
        return "ÖLCÜLEMEZ", "sd sifir"
    if ters_isaret:
        return "ÖLCÜLEMEZ", "uzunluk-karisimi (ham ↔ jeton-normalize ters)"
    if tutarlilik < BAR_TUTARLILIK:
        return "YÖN-DÖNEN", f"tutarlilik {tutarlilik:.3f} < {BAR_TUTARLILIK}"
    if abs(ort_norm) < 1.645 * sd_ham / np.sqrt(n):
        return "MERDIVEN-OKUNMUYOR", "MDE-alti"
    return "MERDIVEN-OKUNUYOR", ""


def tarama():
    R = TT(verdict_maske,
           {"n": [0, 199, 200, 800], "ters_isaret": [False, True],
            "ort_ham": izgara(-0.5, 0.5, 7), "ort_norm": izgara(-0.5, 0.5, 7),
            "tutarlilik": izgara(0.5, 1.0, 6), "sd_ham": [0.0, 0.01, 0.2, 1.0]},
           ADLAR_MASKE, "N-2 · MASKE-ALIM ADIM-1")
    bas(R)
    return R


def kos():
    sat, pd = [], {"n_cift_uzay": len(ZEMINLER) * len(KOMSU), "n_cift_kurulan": 0,
                   "red_shard_yok": 0, "red_payda": 0}
    for zemin in ZEMINLER:
        for a, b in KOMSU:
            A, B = oku_alim(zemin, a), oku_alim(zemin, b)
            if A is None or B is None:
                pd["red_shard_yok"] += 1
                sat.append({"zemin": zemin, "cift": f"{a}-{b}",
                            "VERDICT": "ÖLCÜLEMEZ", "neden": "shard-yok"})
                continue
            C = cift_kur(A, B)
            if C is None:
                pd["red_payda"] += 1
                continue
            pd["n_cift_kurulan"] += 1
            ham = uclu(C["d_ham"])
            nrm = uclu(C["d_norm"])
            ters = bool(np.sign(ham["ort"]) * np.sign(nrm["ort"]) < 0)
            ad, nd = verdict_maske(ham["n"], ters, ham["ort"], nrm["ort"],
                                 ham["isaret_tutarliligi"], ham["sd"])
            sat.append({"zemin": zemin, "cift": f"{a}-{b}",
                        "d_jeton_SABIT": round(float(C["d_jeton"].mean()), 3),
                        "d_jeton_sd": round(float(C["d_jeton"].std()), 6),
                        "HAM": ham, "JETON_NORM": nrm, "ters_isaret": ters,
                        "VERDICT": ad, "neden": nd})
    payda("n2_maske_adim1", **pd)
    return {"damga": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sinif": "MASKE-ALIM ADIM-1 (CPU) — EAP-IG YOK (ADIM-2, GPU)",
            "merdiven": "B-merdiveni (PAKET-K nesnesi) — kip merdiveni DEGIL",
            "PAYDA": pd, "satirlar": sat}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarama", action="store_true")
    ap.add_argument("--ac", action="store_true", help="★ ACILIS — sahip masasi sonrasi")
    a = ap.parse_args()
    if a.tarama:
        tarama()
    if a.ac:
        R = kos()
        json.dump(R, open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(R["PAYDA"], ensure_ascii=False)); print("★", CIKTI)
    if not (a.tarama or a.ac):
        print("★ --tarama · --ac (ACILIS sahip masasi sonrasi)")
