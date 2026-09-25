#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, json, os, re, sys, time
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
import addressee_run as MK
import form_count as FS
from reading_style_object import payda

KOK = __DNH_ROOT__ + ""
PANEL = __DNH_DATA__ + "/c1_panel"
ONB = __DNH_DATA__ + "/havuz_sayac_onbellek.npz"
CIK = f"{KOK}/results/v29_question_ve_seviye_2026-09-10.json"
MERD = {"Tulu3-8B": ["taban", "sft", "dpo", "rl"],
        "OLMo2-13B": ["taban", "sft", "dpo", "instruct"]}
CUMLE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
SAHIS2 = re.compile(r"\b(you|your|yours|yourself|you're|you've|you'll|you'd)\b", re.I)
TEKLIF = re.compile(r"\b(would you like|do you want|shall i|should i|can i help|"
                    r"let me know|any questions|would you prefer|need more|"
                    r"want me to|anything else)\b", re.I)


def cumleler(t):
    return [c.strip() for c in CUMLE.findall(t or "") if c.strip()]


def soru_olc(metinler):
    n_c = n_s = n_as = 0
    for t in metinler:
        for c in cumleler(t):
            n_c += 1
            if c.endswith("?"):
                n_s += 1
                if SAHIS2.search(c) and TEKLIF.search(c):
                    n_as += 1
    return dict(n_cumle=n_c, n_soru=n_s, n_aciklama_sorusu=n_as,
                soru_payi=round(n_s / max(n_c, 1), 5),
                aciklama_sorusu_payi=round(n_as / max(n_c, 1), 5))


def oku(aile, bacak):
    y = f"{PANEL}/{aile}/{bacak}/uretim.jsonl"
    if not os.path.exists(y):
        return None
    return [json.loads(l) for l in open(y, encoding="utf-8")]


def main():
    t0 = time.time()
    D = {"_kunye": dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        alet="scripts/v29_question_ve_seviye.py", borc="D-0910-V29 §3(c,d)",
                        SINIF="BETIM · bar YOK · ad YOK · prediction YOK (§10)",
                        soru_tanimi=dict(cumle=CUMLE.pattern, teklif=TEKLIF.pattern),
                        not_="yeni SAYAC (soru orani) — karar istatistigi DEGIL")}

    aileler = sorted(d for d in os.listdir(PANEL) if os.path.isdir(f"{PANEL}/{d}"))
    S = {}
    n_oku = n_yok = 0
    for a in aileler:
        for b in sorted(os.listdir(f"{PANEL}/{a}")):
            R = oku(a, b)
            if R is None:
                n_yok += 1; continue
            n_oku += 1
            S[f"{a}/{b}"] = soru_olc([r.get("metin") or "" for r in R])
    D["soru_oranlari"] = S
    print(f"{n_oku} {n_yok}"
          f"", flush=True)

    ist = None
    for a in aileler:
        for b in sorted(os.listdir(f"{PANEL}/{a}")):
            R = oku(a, b)
            if R:
                ist = sorted({(r.get("onek") or "") for r in R})
                break
        if ist:
            break
    if ist:
        ns = sum(1 for x in ist if x.rstrip().endswith("?"))
        D["istem_tarafi"] = dict(n_istem=len(ist), n_soru=ns,
                                 soru_payi=round(ns / max(len(ist), 1), 4))
        print(f"  [PAYDA] istem_soru: n_istem={len(ist)} · soru={ns} "
              f"⇒ panelin istem karisimi SABITTIR (aileden bagimsiz)", flush=True)

    nlp = FS._boru()
    SEV = {}
    for aile, rungs in MERD.items():
        for b in rungs:
            R = oku(aile, b)
            if R is None:
                SEV[f"{aile}/{b}"] = "ÜRETIM-YOK"; continue
            V = MK.satir_bilesenleri(nlp, R)
            SEV[f"{aile}/{b}"] = round(float(MK.olc_toplam(V)["M1"]), 4)
            print(f"  [{time.time()-t0:.0f}s] {aile}/{b}: M1={SEV[f'{aile}/{b}']}",
                  flush=True)
    D["basamak_seviyeleri"] = SEV

    z = np.load(ONB, allow_pickle=True)
    D["havuz_secilen"] = dict(
        set="tulu3-pref (önbellek)", n_cift=int(len(z["M1c"])),
        M1_secilen=round(float(z["M1c"].mean()), 4),
        M1_reddedilen=round(float(z["M1r"].mean()), 4),
        jeton_secilen=round(float(z["NJc"].mean()), 1),
        not_=""
             "")
    print(f"  [PAYDA] havuz_secilen: n_cift={D['havuz_secilen']['n_cift']} · "
          f"M1_secilen={D['havuz_secilen']['M1_secilen']} · "
          f"M1_reddedilen={D['havuz_secilen']['M1_reddedilen']} · "
          f"red_onbellekte_yok=2 (UF, HH)")
    json.dump(D, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("v29_soru_seviye", n_bacak=n_oku, n_merdiven_bacak=len(SEV),
          red_uretim_yok=n_yok, red_havuz_onbellekte_yok=2)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
