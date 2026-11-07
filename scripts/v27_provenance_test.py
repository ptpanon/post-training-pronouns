#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, re, sys, time
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
from reading_style_object import payda

KOK = __DNH_ROOT__ + ""
PANEL = __DNH_DATA__ + "/c1_panel"
SUZ = f"{KOK}/results/template_filtered_2026-09-07.json"
CIK = f"{KOK}/results/V27_KOKEN_SINAVI_2026-09-09.json"
NPERM, SEED, ASGARI = 200, 20260910, 3
CARD_BILGISI = {
    "allenai/Llama-3.1-Tulu-3-8B": "TULU-KARISIMI",
    "allenai/OLMo-2-1124-13B-Instruct": "TULU-KARISIMI",
    "allenai/OLMo-2-0325-32B-Instruct": "TULU-KARISIMI",
    "allenai/OLMo-2-1124-7B-Instruct": "TULU-KARISIMI",
}


def _snapshot(aile):
    for b in ("instruct", "rl", "rlvr", "dpo", "sft"):
        y = f"{PANEL}/{aile}/{b}/uretim_kunye.json"
        if os.path.exists(y):
            m = json.load(open(y, encoding="utf-8")).get("model") or ""
            if os.path.isdir(m):
                return m
    return None


def card_acik(aile):
    yol = _snapshot(aile)
    rd = os.path.join(yol, "README.md") if yol else None
    if not rd or not os.path.exists(rd):
        return "CARD-YOK", ""
    ANAHTAR = re.compile(r"(?i)\b(dpo|preference|rlhf)\b")
    for ham in io.open(rd, encoding="utf-8", errors="replace").read().splitlines():
        l = ham.strip()
        if re.search(r"(?i)^(-\s*|base_model:\s*|\*\*finetuned from model:\*\*\s*)"
                     r"[\w./-]+-dpo\s*$", l):
            return "ACIK", l[:160]
        if ANAHTAR.search(l) and "huggingface.co/datasets/" in l:
            return "ACIK", l[:160]
    return "KAPALI", ""


def depo(aile):
    for b in ("instruct", "rl", "rlvr", "dpo", "sft"):
        y = f"{PANEL}/{aile}/{b}/uretim_kunye.json"
        if os.path.exists(y):
            m = json.load(open(y, encoding="utf-8")).get("model") or ""
            if "models--" in m:
                seg = m.split("models--")[1].split("/")[0]
                return seg.replace("--", "/", 1).replace("--", "-")
    return None


def sinav(a, b):
    d = np.concatenate([a, b]); g = np.array([1] * len(a) + [0] * len(b))
    goz = float(a.mean() - b.mean())
    rng = np.random.default_rng(SEED)
    N = np.array([float(d[(p := rng.permutation(g)) == 1].mean() - d[p == 0].mean())
                  for _ in range(NPERM)])
    return goz, N, float((np.abs(N) >= abs(goz)).mean()), float(1.645 * N.std())


def main():
    PUAN = "--puanlayici" in sys.argv
    PUANLAYICI = {
        "Tulu3-8B":   ("TULU-RM", "card: Reward Model (RM) allenai/Llama-3.1-Tulu-3-8B-RM"),
        "OLMo2-13B":  ("TULU-RM", "card: Reward Model (RM) allenai/OLMo-2-1124-13B-RM"),
        "OLMo2-32B":  ("TULU-RM", ""),
        "Llama-3.1-8B":  ("INSAN-RLHF", "card: «SFT and RLHF to align with human preferences»"),
        "Llama-3.1-70B": ("INSAN-RLHF", "card: «SFT and RLHF to align with human preferences»"),
        "OLMo-3-7B":  ("KURGU-DOGRULANAMADI", ""),
    }
    ACIK = "--acik" in sys.argv
    AD_A, AD_B = (("TULU-RM", "INSAN-RLHF") if PUAN
                  else ("ACIK", "KAPALI") if ACIK else ("TULU-KARISIMI", "BILINMIYOR"))
    SINIFLAR = ((AD_A, AD_B, "KURGU-DOGRULANAMADI", "ADLANDIRILAMAZ") if PUAN
                else (AD_A, AD_B, "CARD-YOK") if ACIK
                else ("TULU-KARISIMI", "INSAN-ETIKETLI", "BILINMIYOR"))
    cikti = (f"{KOK}/results/v31_puanlayici_2026-09-10.json" if PUAN else
             f"{KOK}/results/v30_provenance_acik_2026-09-10.json" if ACIK else CIK)
    S = json.load(open(SUZ, encoding="utf-8"))
    R = [r for r in S["aileler"] if r["hal"] == "ÖLCÜLDÜ"]
    sat = []
    for r in R:
        d = depo(r["aile"])
        if PUAN:
            sn, kanit = PUANLAYICI.get(r["aile"], ("ADLANDIRILAMAZ", "card puanlayiciyi adlandirmiyor"))
            sat.append(dict(aile=r["aile"], depo=d, card=sn, kanit=kanit, sinif=sn,
                            dM1=round(r["sbl"]["dM1"], 3), ayrik=r["sbl"]["ayrik"]))
        elif ACIK:
            sn, kanit = card_acik(r["aile"])
            sat.append(dict(aile=r["aile"], depo=d, card=sn, kanit=kanit,
                            sinif=(AD_B if sn == "CARD-YOK" else sn),
                            dM1=round(r["sbl"]["dM1"], 3), ayrik=r["sbl"]["ayrik"]))
        else:
            sat.append(dict(aile=r["aile"], depo=d,
                            sinif=CARD_BILGISI.get(d, "BILINMIYOR"),
                            dM1=round(r["sbl"]["dM1"], 3), ayrik=r["sbl"]["ayrik"]))
    say = {}
    for x in sat:
        say[x["sinif"]] = say.get(x["sinif"], 0) + 1
    for s3 in SINIFLAR:
        say.setdefault(s3, 0)
    if ACIK and not PUAN:
        say["CARD-YOK"] = sum(1 for x in sat if x["card"] == "CARD-YOK")
    print(f"  [PAYDA] koken_sinif: n_aile={len(sat)} · " +
          " · ".join(f"{k}={v}" for k, v in sorted(say.items())) +
          f"{sum(1 for x in sat if not x['depo'])}"
          f"{ASGARI}")

    D = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         alet="scripts/v27_provenance_test.py", borc="D-0910-V27 §2b",
                         sinif_kaynagi=((""
                                         ""
                                         "")
                                        if ACIK else
                                        ("depo adi DISKTEN (panel künyesi snapshot yolu); "
                                         "depo→tercih-verisi eslemesi MODEL KARTINDAN "
                                         "(BELLEK, teyitsiz — §12)")),
                         kip=("PUANLAYICI-KIMLIGI (--puanlayici)" if PUAN else
                              "ACIK-RECETE (--acik)" if ACIK else "TÜLÜ-KARISIMI (V27)"),
                         n_perm=NPERM, seed=SEED, asgari_sinif=ASGARI),
             aileler=sat, sinif_sayimi=say)

    a = np.array([x["dM1"] for x in sat if x["sinif"] == AD_A], float)
    b = np.array([x["dM1"] for x in sat if x["sinif"] == AD_B], float)
    ASGARI_ = 2 if PUAN else ASGARI
    if len(a) < ASGARI_ or len(b) < ASGARI_:
        D["VERDICT"] = "ÖLCÜLEMEZ (sinif kücük)"
    else:
        goz, N, frac, mde = sinav(a, b)
        D.update(ortalama_A=round(float(a.mean()), 3),
                 ortalama_B=round(float(b.mean()), 3),
                 ad_A=AD_A, ad_B=AD_B,
                 fark=round(goz, 3), null_sd=round(float(N.std()), 3),
                 MDE=round(mde, 3), frac_null_asan=frac,
                 yukari_A=f"{sum(1 for x in sat if x['sinif']==AD_A and x['ayrik'] and x['dM1']>0)}/{len(a)}",
                 yukari_B=f"{sum(1 for x in sat if x['sinif']==AD_B and x['ayrik'] and x['dM1']>0)}/{len(b)}",
                 asagi_A=f"{sum(1 for x in sat if x['sinif']==AD_A and x['ayrik'] and x['dM1']<0)}/{len(a)}",
                 asagi_B=f"{sum(1 for x in sat if x['sinif']==AD_B and x['ayrik'] and x['dM1']<0)}/{len(b)}",
                 VERDICT=("KÖKENI-IZLEMIYOR" if frac > 0.05 else "KÖKENI-IZLIYOR"),
                 GUC_SERHI=(f"|fark|={abs(goz):.2f} · MDE={mde:.2f} ⇒ "
                            + ("bu tasarim bu farki GÖREBILIRDI"
                               if abs(goz) >= mde else
                               ""
                               "")))
        print(f"  ortalama ΔM1 · {AD_A} {D['ortalama_A']} ↔ {AD_B} "
              f"{D['ortalama_B']} · fark {D['fark']} · MDE {D['MDE']} · "
              f"frac={frac:.3f}")
        print(f"  yukari: {AD_A} {D['yukari_A']} ↔ {AD_B} {D['yukari_B']} · "
              f"asagi: {AD_A} {D['asagi_A']} ↔ {AD_B} {D['asagi_B']}")
        print(f"  ★★★ VERDICT: {D['VERDICT']} · {D['GUC_SERHI']}")
        if ACIK:
            b2 = np.array([x["dM1"] for x in sat
                           if x["sinif"] == AD_B and x["card"] != "CARD-YOK"], float)
            if len(b2) >= ASGARI:
                g2, N2, f2, m2 = sinav(a, b2)
                D["duyarlilik_kartsizlar_dusuruldu"] = dict(
                    n_B=len(b2), fark=round(g2, 3), MDE=round(m2, 3),
                    frac_null_asan=f2,
                    VERDICT=("KÖKENI-IZLEMIYOR" if f2 > 0.05 else "KÖKENI-IZLIYOR"))
                print(f"  [DUYARLILIK] kartsiz {len(b) - len(b2)} aile düsünce: "
                      f"fark {g2:+.3f} · MDE {m2:.2f} · frac={f2:.3f} ⇒ "
                      f"{D['duyarlilik_kartsizlar_dusuruldu']['VERDICT']}")
    json.dump(D, open(cikti, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("v27_koken", n_aile=len(sat), n_tulu=len(a), n_bilinmiyor=len(b),
          red_insan_etiketli=say.get("INSAN-ETIKETLI", 0), n_perm=NPERM)
    print(f"✓ {cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
