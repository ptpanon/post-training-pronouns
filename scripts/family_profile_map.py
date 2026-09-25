#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, hashlib, json, os, sys
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
DOC = f"{ROOT}/results"

AILE16 = ["Qwen2.5-1.5B", "Qwen2.5-7B", "Qwen2.5-14B", "Mistral-7B-v0.3",
          "Mistral-Small-24B", "Zephyr-7B", "Llama-3.1-8B", "Tulu-3-8B",
          "OLMo-3-7B", "OLMo-2-13B", "OLMo-2-32B", "Gemma-3-4B", "Gemma-3-12B",
          "Gemma-3-27B", "Falcon-H1-7B", "Granite-4.1-8B"]
C1AD = {"Tulu-3-8B": "Tulu3-8B", "OLMo-2-13B": "OLMo2-13B", "OLMo-2-32B": "OLMo2-32B",
        "Zephyr-7B": None, "Falcon-H1-7B": None, "Granite-4.1-8B": None}
GOVDE = {"zephyr": "Zephyr-7B", "tulu3": "Tulu-3-8B", "olmo": "OLMo-2-13B",
         "olmo32": "OLMo-2-32B"}


def sha(y, n=16):
    return hashlib.sha256(open(y, "rb").read()).hexdigest()[:n] if os.path.exists(y) else "YOK"


def oku(y):
    return json.load(open(y, encoding="utf-8")) if os.path.exists(y) else None


def _egim(T):
    a = np.array([float(k) for k in T]); t = np.array([T[k] for k in T], float)
    i = np.argsort(a); a, t = a[i], t[i]
    return float(np.polyfit(a, t, 1)[0])


def topla():
    K, X = {}, {a: {} for a in AILE16}
    EK = {}

    y = f"{DOC}/shadow_reading_16_scorecard_2026-08-25.json"; K["F5"] = sha(y); d = oku(y)
    if d:
        for sub, kisa in (("SCOTUS (OKUMA-B)", "SCOTUS"), ("cga-cmv", "CMV")):
            H = d["substrat"][sub]["hucre"]
            for kanal in ("c_KE", "c_YE", "c_DB"):
                e = f"okuma_{kisa}_{kanal}"
                EK[e] = ("sd-kati (null merkezine göre)", f"F5_16_KARNE @ {K['F5']}")
                for a in AILE16:
                    v = H.get(a, {}).get("sayilar", {}).get(kanal, {})
                    X[a][e] = float(v["sd_kati"]) if "sd_kati" in v else np.nan

    for e in ("kip_NLL_kisa", "kip_NLL_uzun"):
        EK[e] = ("ΔNLL (hizali−taban), emir−rica, nat/jeton", "E_YALINLIK + KESIF_KIP16")
        for a in AILE16:
            X[a][e] = np.nan
    for y in (f"{DOC}/plainness_2026-08-27.json", f"{DOC}/exploratory_mood16_2026-08-28.json"):
        d = oku(y)
        if not d or "aile" not in d:
            continue
        K[os.path.basename(y)] = sha(y)
        for a, v in d["aile"].items():
            if a in X and isinstance(v.get("ort"), dict):
                X[a]["kip_NLL_kisa"] = float(v["ort"]["kisa"])
                X[a]["kip_NLL_uzun"] = float(v["ort"]["uzun"])

    y = f"{DOC}/form_count_2026-08-28.json"; K["FORM"] = sha(y); d = oku(y)
    for s in ("a", "c", "d", "e"):
        e = f"form_{s}_delta"
        EK[e] = ("Δoran (sayac/cümle), ciplak protokol", f"FORM_SAYIM @ {K['FORM']}")
        for a in AILE16:
            c1 = C1AD.get(a, a)
            h = (d or {}).get("panel", {}).get(c1, {}).get(s, {}) if c1 else {}
            X[a][e] = float(h["TAM"]["delta"]) if "TAM" in h else np.nan
    EK["form_cumle_uretim_delta"] = ("Δ(cümle/üretim)", f"FORM_SAYIM @ {K['FORM']}")
    for a in AILE16:
        c1 = C1AD.get(a, a)
        u = (d or {}).get("panel", {}).get(c1, {}).get("_uzunluk") if c1 else None
        X[a]["form_cumle_uretim_delta"] = (float(u["sonra"]["cumle_uretim"]
                                                 - u["once"]["cumle_uretim"]) if u else np.nan)

    y = f"{DOC}/INTACT_YENIDEN_c1_2026-08-26.json"; K["C1"] = sha(y); d = oku(y)
    for eks, e in (("DOM", "dU_DOM"), ("ARO", "EK_dU_ARO"), ("TON", "EK_dU_TON")):
        EK[e] = ("ΔU sd-kati (isaretli)", f"INTACT_YENIDEN_c1 @ {K['C1']}")
        for a in AILE16:
            h = (d or {}).get("hucre", {}).get(C1AD.get(a, a) or "", {}).get("delta", {}).get(eks, {})
            v = h.get("sd_kati")
            X[a][e] = float(v) * (1 if h.get("dU", 0) >= 0 else -1) if v is not None else np.nan

    EK["dpo_egim"] = ("dT/dα (OLS), T = NLL(emir)−NLL(rica)", "D_DELTA_*.json")
    for a in AILE16:
        X[a]["dpo_egim"] = np.nan
    for y in sorted(glob.glob(f"{DOC}/D_DELTA_*.json")):
        d = oku(y); K[os.path.basename(y)] = sha(y)
        a = GOVDE.get(d.get("hucre"))
        if a and d.get("T"):
            X[a]["dpo_egim"] = _egim(d["T"])

    y = f"{DOC}/B_YON_ACISI_2026-08-27.json"; K["BYON"] = sha(y); d = oku(y)
    for e in ("cos_govde", "lm_head_payi"):
        EK[e] = (("cos(û_D_gövde, û_A_gövde)" if e == "cos_govde"
                  else "‖lm_head‖²/‖û‖² payi"), f"B_YON_ACISI @ {K['BYON']}")
        for a in AILE16:
            X[a][e] = np.nan
    for h, a in GOVDE.items():
        c = (d or {}).get("hucre", {}).get(h, {}).get("cos", {})
        if c:
            X[a]["cos_govde"] = float(c["GÖVDE"]); X[a]["lm_head_payi"] = float(c["_lm_head_payi"])

    y = f"{DOC}/KATMAN_TARAMA_2026-08-27.json"; K["KATMAN"] = sha(y); d = oku(y)
    EK["sonda_auc"] = ("AUC (argmax katman, 0…L/4 taramasi)", f"KATMAN_TARAMA @ {K['KATMAN']}")
    for a in AILE16:
        X[a]["sonda_auc"] = float((d or {}).get("aile", {}).get(a, {}).get("argmax_auc", np.nan))

    import family_panel as C
    C.panel_sec("f5")
    TARIF = {x["ad"]: x["tarif"] for x in C.PANEL}
    RECETE = {}
    for a in AILE16:
        t = TARIF.get(a, "")
        RECETE[a] = ("RLVR" if "RLVR" in t else "DPO" if "DPO" in t else
                     "RLHF" if "RLHF" in t or "RLHF" in t else
                     "SFT/tercih" if "SFT" in t or "tercih" in t else "tarif-yok")
    EK["recete"] = ("etiket — repo `tarif` alanindan türetildi (model karti DEGIL)",
                    "family_panel.PANEL_F5")

    eksenler = [e for e in EK if e != "recete"]
    dolu = sum(1 for a in AILE16 for e in eksenler if not np.isnan(X[a][e]))
    payda("harita_topla", n_aile=len(AILE16), n_eksen=len(eksenler),
          n_dolu_hucre=dolu, red_bos=len(AILE16) * len(eksenler) - dolu)
    return dict(aile=AILE16, eksen=eksenler, X=X, kunye=EK, recete=RECETE,
                tarif=TARIF, kaynak_sha=K)


if __name__ == "__main__":
    R = topla()
    print(f"★ {len(R['aile'])} aile × {len(R['eksen'])} eksen")
    for e in R["eksen"]:
        n = sum(1 for a in R["aile"] if not np.isnan(R["X"][a][e]))
        print(f"  {e:26s} n={n:2d}/16 · {R['kunye'][e][0]}  ← {R['kunye'][e][1]}")
