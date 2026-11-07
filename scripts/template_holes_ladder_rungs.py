#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, sys, time

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import template_robustness as SG
import family_panel as C

CIK = f"{ROOT}/results/template_holes_ladder_rungs_2026-08-29.json"
PREREG = f"{ROOT}/preregistration/prereg_template_holes_2026-08-28.md"

KOLLAR = [
    dict(ad="OLMo2-13B·taban→SFT", taban="OLMo2-13B/taban", hizali="OLMo2-13B/sft",
         s_taban="models--allenai--OLMo-2-1124-13B",
         s_hizali="models--allenai--OLMo-2-1124-13B-SFT",
         panel="olcek2", ikiz="ham", dU_ciplak_kayit=-0.001253),
    dict(ad="OLMo2-32B·taban→SFT", taban="OLMo2-32B/taban", hizali="OLMo2-32B/sft",
         s_taban="models--allenai--OLMo-2-0325-32B",
         s_hizali="models--allenai--OLMo-2-0325-32B-SFT",
         panel="olcek2", ikiz="ham", dU_ciplak_kayit=-0.019964),
    dict(ad="Tulu3-8B·taban→SFT", taban="Tulu3-8B/taban", hizali="Tulu3-8B/sft",
         s_taban="models--meta-llama--Llama-3.1-8B",
         s_hizali="models--allenai--Llama-3.1-Tulu-3-8B-SFT",
         panel="olcek3", ikiz="ham", dU_ciplak_kayit=-0.000863),
]

SERH_KOLLAR = [
    dict(ad=a, neden="TABAN-BACAGININ-SABLONU-VAR",
         serh="PREREG_TEMPLATE_GURBUZLUGU_2026-08-26 §1 bu hücre sinifini GEREKCEYLE disladi; "
              "simetrik (iki-bacak-sablonlu) okumanin verisi diskte HAZIR, ayri prereg ister")
    for a in ("OLMo2-13B·SFT→DPO", "OLMo2-13B·DPO→Inst",
              "OLMo2-32B·SFT→DPO", "OLMo2-32B·DPO→Inst",
              "Tulu3-8B·SFT→DPO", "Tulu3-8B·DPO→RL")]


def gomu_kapisi():
    eksik, n = [], 0
    for k in KOLLAR:
        for kok, dz in ((SG.CIPLAK_KOK, k["taban"]), (SG.CIPLAK_KOK, k["hizali"]),
                        (SG.TEMPLATE_KOK, k["hizali"])):
            n += 1
            y = f"{kok}/{dz}/X_e5.npy"
            if not os.path.exists(y):
                eksik.append(y)
    payda("merdiven_gomu_kapisi", n_bacak=n, hal_hazir=n - len(eksik),
          red_gomu_yok=len(eksik), bekle={"n_bacak": 9})
    if eksik:
        raise SystemExit("★ GÖMÜ KAPISI DÜSTÜ (CPU kosusu GPU isterdi):\n  "
                         + "\n  ".join(eksik))
    return True


def coz(a):
    t0 = time.time()
    SG.prova()
    gomu_kapisi()
    H, sayac = {}, dict(ayni=0, degisti=0, kapi=0, dej_dusuk=0, isaret_dondu=0)
    for k in KOLLAR:
        ad = k["ad"]
        s_yol = f"{SG.TEMPLATE_KOK}/{k['hizali']}/uretim.jsonl"
        if not os.path.exists(s_yol):
            H[ad] = dict(HAL="KAPI/ÖLCÜLEMEZ", neden="")
            sayac["kapi"] += 1; print(f"{ad:22s}", flush=True)
            continue
        dj_s, n_s = SG._dej(s_yol)
        dj_c, n_c = SG._dej(f"{SG.CIPLAK_KOK}/{k['hizali']}/uretim.jsonl")
        bos_s = sum(1 for l in open(s_yol, encoding="utf-8")
                    if not json.loads(l)["metin"].strip()) / n_s
        if bos_s > SG.BOS_BAR:
            H[ad] = dict(HAL="KAPI/ÖLCÜLEMEZ", neden=f"bos-orani {bos_s:.4f} > {SG.BOS_BAR}")
            sayac["kapi"] += 1; print(f"  [{ad:22s}] ★ KAPI — bos {bos_s:.4f}", flush=True)
            continue
        zb, Bb, UNb = SG._oku(k, "base", SG.CIPLAK_KOK, a.dev)
        zi_c, Bi_c, UNi_c = SG._oku(k, "instruct", SG.CIPLAK_KOK, a.dev)
        zi_s, Bi_s, UNi_s = SG._oku(k, "instruct", SG.TEMPLATE_KOK, a.dev)
        d_c = SG._delta(zb, Bb, UNb, zi_c, Bi_c, UNi_c, k["ikiz"])
        d_s = SG._delta(zb, Bb, UNb, zi_s, Bi_s, UNi_s, k["ikiz"])
        ac, as_ = d_c[SG.EKSEN_KARAR]["ad"], d_s[SG.EKSEN_KARAR]["ad"]
        kar = "ham" if k["ikiz"] == "ham" else "merkezli"
        uc, us = d_c[SG.EKSEN_KARAR][kar]["dU"], d_s[SG.EKSEN_KARAR][kar]["dU"]
        gocs = abs(uc - k["dU_ciplak_kayit"])
        if gocs > 5e-4:
            raise SystemExit(f"★ SAYI-GÖCÜ: {ad} ciplak ΔU {uc:+.6f} ≠ kayit "
                             f"{k['dU_ciplak_kayit']:+.6f} (|Δ|={gocs:.6f}) ⇒ DUR")
        isaret = bool(uc * us < 0)
        hal = "AYNI-AD" if ac == as_ else "AD-DEGISTI"
        sayac["ayni" if hal == "AYNI-AD" else "degisti"] += 1
        sayac["isaret_dondu"] += int(isaret); sayac["dej_dusuk"] += int(dj_s <= dj_c)
        H[ad] = dict(HAL=hal, ad_ciplak=ac, ad_sablonlu=as_, karar_ikizi=k["ikiz"],
                     yon=(f"{ac} → {as_}" if hal == "AD-DEGISTI" else None),
                     dU_ciplak=uc, dU_sablonlu=us, dU_ciplak_kayit=k["dU_ciplak_kayit"],
                     sayi_gocu=round(gocs, 8), ISARET_DONDU=isaret,
                     DEJ_ciplak=round(dj_c, 4), DEJ_sablonlu=round(dj_s, 4),
                     DEJ_dustu=bool(dj_s <= dj_c), bos_orani_sablonlu=round(bos_s, 4),
                     uc_eksen_ciplak={e: d_c[e]["ad"] for e in C.EKSENLER},
                     uc_eksen_sablonlu={e: d_s[e]["ad"] for e in C.EKSENLER},
                     tam_ciplak=d_c, tam_sablonlu=d_s)
        print(f"  [{ad:22s}] {ac:12s} → {as_:12s} · ΔU {uc:+.5f}→{us:+.5f}"
              f" · DEJ {dj_c:.3f}→{dj_s:.3f} · ★ {hal}"
              f"{'  ★★ ISARET DÖNDÜ' if isaret else ''}", flush=True)
    olculebilir = sayac["ayni"] + sayac["degisti"]
    payda("template_merdiven", n_kol=len(KOLLAR), hal_olculebilir=olculebilir,
          hal_ayni=sayac["ayni"], hal_degisti=sayac["degisti"],
          red_kapi=sayac["kapi"], red_protokol_disi=len(SERH_KOLLAR),
          bekle={"n_kol": 3})
    print(f""
          f"{sayac['ayni']} {sayac['degisti']} {sayac['kapi']}")
    print(f"★ PROTOKOL-DISI (kosulmadi, adiyla): {len(SERH_KOLLAR)} basamak")
    json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   PREREG=os.path.basename(PREREG), SINIF="KESIF",
                   PANEL="",
                   PREDICTION="YOK (KESIF sinifi)",
                   EKSEN_KARAR=SG.EKSEN_KARAR, sayac=sayac, olculebilir=olculebilir,
                   kol=H, protokol_disi=SERH_KOLLAR,
                   saniye=round(time.time() - t0, 1)),
              open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"→ {CIK}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--asama", required=True, choices=["kapi", "coz"])
    p.add_argument("--dev", default="cpu")
    p.add_argument("--bekle-gpu", dest="bekle_gpu", type=int, default=0)
    A = p.parse_args()
    if A.asama == "kapi":
        SG.prova(); gomu_kapisi(); print("★ KAPILAR GECTI (CPU kosusu mümkün)")
        raise SystemExit(0)
    raise SystemExit(coz(A))
