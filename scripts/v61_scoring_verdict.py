#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import arm_table as KT
from reading_style_object import payda

CIK = os.environ.get("V61P_CIK", f"{ROOT}/results/v61_scoring_2026-09-11.json")
KOK = os.environ.get("V61P_KOK", __DNH_DATA__ + "/v61_puanlama")
PUANLAYICI = (("armorm", "RLHFlow/ArmoRM-Llama3-8B-v0.1"),
              ("skywork", "Skywork/Skywork-Reward-Llama-3.1-8B"))
NB = 1000
SEED = 20260911
MDE_ORAN = 0.10
GUC_PROVA_N = 300


def _ayrik(ci):
    return bool(ci) and (ci[0] > 0 or ci[1] < 0)


def _ortusmez(a, b):
    if not a or not b:
        return False
    return a[1] < b[0] or b[1] < a[0]


def kaskad(ci, nokta, ci_pla, mde_gecti):
    if ci is None:
        return "ÖLCÜLEMEZ", "CI kurulamadi"
    if _ayrik(ci):
        if ci_pla is None:
            return "ÖLCÜLEMEZ", (""
                                 ""
                                 "")
        if not _ortusmez(ci, ci_pla):
            return "ÖLCÜLEMEZ", "plasebo bandi gercek etkiyi icine aliyor"
        if nokta > 0:
            return "CEZALANDIRIYOR", "Δ>0, CI sifirdan ayrik, plasebo bandinin disinda"
        return "ÖDÜLLENDIRIYOR", "Δ<0, CI sifirdan ayrik, plasebo bandinin disinda"
    if not mde_gecti:
        return "ÖLCÜLEMEZ", ""
    return "SIFIR", "CI sifiri kapsiyor, MDE bu farki görebilirdi"


def prova():
    s = [
        ("CEZALANDIRIYOR", kaskad([0.10, 0.30], +0.20, [-0.03, 0.03], True)[0]),
        ("ÖDÜLLENDIRIYOR", kaskad([-0.30, -0.10], -0.20, [-0.03, 0.03], True)[0]),
        ("SIFIR",          kaskad([-0.02, 0.02], +0.001, [-0.05, 0.05], True)[0]),
        ("SIFIR",          kaskad([-0.01, 0.01], +0.000, [-0.20, 0.20], True)[0]),
        ("ÖLCÜLEMEZ",      kaskad([-0.40, 0.40], +0.01, [-0.05, 0.05], False)[0]),
        ("ÖLCÜLEMEZ",      kaskad([0.10, 0.30], +0.20, [0.05, 0.35], True)[0]),
        ("ÖLCÜLEMEZ",      kaskad([0.10, 0.30], +0.20, None, True)[0]),
    ]
    ok = all(b == g for b, g in s)
    dogan = sorted({g for _, g in s})
    ayr = birlesik({"a": "CEZALANDIRIYOR", "b": "SIFIR"})
    ayni = birlesik({"a": "CEZALANDIRIYOR", "b": "CEZALANDIRIYOR"})
    ay_ok = (ayr == "AYRISTI" and ayni == "CEZALANDIRIYOR")
    payda("v61_puanlama_prova", n_sinav=len(s) + 2,
          hal_gecen=sum(1 for b, g in s if b == g), hal_dogan_ad=len(dogan),
          hal_ayristi=int(ay_ok), red_uyusmaz=sum(1 for b, g in s if b != g))
    for b, g in s:
        print(f"  [PROVA] beklenen {b:16s} ⇒ dogan {g:16s} {'✓' if b == g else '✗'}")
    print(f"{ayr} {ayni}"
          f"{'✓' if ay_ok else '✗'}")
    print(f"  ★ dogan ad kümesi: {dogan} ⇒ esik: dört adin dördü + AYRISTI "
          f"dogmazsa ⇒ EYLEM: VERDICT YAZILMAZ (K-1)", flush=True)
    return ok and ay_ok and len(dogan) == 4


def birlesik(adlar):
    v = set(adlar.values())
    return v.pop() if len(v) == 1 else "AYRISTI"


def _oku(ad):
    y = os.path.join(KOK, f"{ad}.jsonl")
    if not os.path.exists(y):
        return None
    R = [json.loads(l) for l in io.open(y, encoding="utf-8") if l.strip()]
    return R or None


def main():
    if not prova():
        print("★ PROVA DÜSTÜ ⇒ VERDICT YAZILMAZ"); return 4
    t0 = time.time(); OUT = {}; adlar = {}
    for ad, depo in PUANLAYICI:
        R = _oku(ad)
        if R is None:
            OUT[ad] = dict(hal="BEKLIYOR", depo=depo,
                           eksik=os.path.join(KOK, f"{ad}.jsonl"))
            print(f"{ad:9s}", flush=True)
            continue
        ist = np.array([int(r["istem_i"]) for r in R])
        kis = np.array([float(r["puan_kisili"]) for r in R])
        ksz = np.array([float(r["puan_kisisiz"]) for r in R])
        pla = np.array([float(r.get("puan_plasebo", np.nan)) for r in R])
        d = ksz - kis
        d_pla = pla - kis
        b = KT.kume_ort_boot(d, ist, nb=NB, seed=SEED)
        b_pla = KT.kume_ort_boot(d_pla, ist, nb=NB, seed=SEED)
        ci = [round(b[0], 5), round(b[1], 5)] if b else None
        ci_pla = [round(b_pla[0], 5), round(b_pla[1], 5)] if b_pla else None
        sd_puan = float(np.nanstd(kis))
        sd_boot = (b[1] - b[0]) / (2 * 1.96) if b else float("nan")
        mde = 1.645 * sd_boot
        mde_gecti = bool(sd_puan > 0 and mde < MDE_ORAN * sd_puan)
        g = KT.kume_ort_boot(d[:GUC_PROVA_N], ist[:GUC_PROVA_N], nb=400, seed=SEED)
        mde_g = 1.645 * ((g[1] - g[0]) / (2 * 1.96)) if g else float("nan")
        nokta = float(np.nanmean(d))
        hk, neden = kaskad(ci, nokta, ci_pla, mde_gecti)
        adlar[ad] = hk
        OUT[ad] = dict(hal="ÖLCÜLDÜ", depo=depo, VERDICT=hk, neden=neden,
                       n_cift=len(R), n_istem=int(len(np.unique(ist))),
                       delta=round(nokta, 5), ci=ci,
                       delta_plasebo=round(float(np.nanmean(d_pla)), 5), ci_plasebo=ci_pla,
                       sd_puan=round(sd_puan, 5), mde=round(mde, 5),
                       mde_bari=round(MDE_ORAN * sd_puan, 5), mde_gecti=mde_gecti,
                       guc_provasi_mde=round(mde_g, 5) if mde_g == mde_g else None,
                       etki_sd_biriminde=round(nokta / sd_puan, 4) if sd_puan else None,
                       medyan_degisim_payi=round(float(np.nanmedian(
                           [r.get("degisim_payi", np.nan) for r in R])), 5))
        print(f"  [{time.time()-t0:4.0f}s] {ad:9s} Δ={nokta:+.4f} CI={ci} · "
              f"plasebo Δ={np.nanmean(d_pla):+.4f} CI={ci_pla} · MDE={mde:.4f} "
              f"(bar {MDE_ORAN*sd_puan:.4f}) ⇒ ★ {hk} ({neden})", flush=True)
        print(f"{MDE_ORAN}"
              f"", flush=True)
    ol = {k: v for k, v in OUT.items() if v.get("hal") == "ÖLCÜLDÜ"}
    HK = birlesik(adlar) if len(adlar) == len(PUANLAYICI) else "BEKLIYOR"
    payda("v61_puanlama_verdict", n_puanlayici=len(PUANLAYICI), hal_olculen=len(ol),
          hal_cezalandiriyor=sum(1 for v in adlar.values() if v == "CEZALANDIRIYOR"),
          hal_odullendiriyor=sum(1 for v in adlar.values() if v == "ÖDÜLLENDIRIYOR"),
          hal_sifir=sum(1 for v in adlar.values() if v == "SIFIR"),
          hal_olculemez=sum(1 for v in adlar.values() if v == "ÖLCÜLEMEZ"),
          hal_ayristi=int(HK == "AYRISTI"),
          red_bekleyen=len(PUANLAYICI) - len(ol))
    print(f"  ★ BIRLESIK VERDICT: {HK} · adlar={adlar} ⇒ esik: iki ad farkli "
          f"⇒ EYLEM: AYRISTI, tek sayiya INDIRILMEZ", flush=True)
    K = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="VERDICT", alet="scripts/v61_scoring_verdict.py",
             prereg="prereg_v61_scoring_2026-09-11.md @ 06e13a49",
             notice="prerun_notice_v61_2026-09-11.md @ d6e7863c",
             prediction="prediction_v61_2026-09-11.md @ b4503313",
             kok=KOK, n_bootstrap=NB, seed=SEED, kume_birimi="istem_i",
             mde_orani=MDE_ORAN, guc_provasi_n=GUC_PROVA_N,
             SERH_GUC_KAPISI=(""
                              ""
                              ""
                              ""
                              ""
                              "%d"
                              "" % GUC_PROVA_N),
             uzay=["CEZALANDIRIYOR", "ÖDÜLLENDIRIYOR", "SIFIR", "ÖLCÜLEMEZ", "AYRISTI"],
             serh=(""
                   ""),
             BIRLESIK=HK, puanlayici=OUT)
    json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
