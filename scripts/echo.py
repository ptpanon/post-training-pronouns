#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
import template_robustness_24 as S24
import template_robustness as SG
import urial_run as UK
from reading_style_object import payda

KOK = os.environ.get("YANKI_KOK", __DNH_DATA__ + "/c1_panel_template")
CIK = os.environ.get("YANKI_CIK", f"{ROOT}/results/YANKI_2026-09-10.json")
PEN, ADIM, KUYRUK = 40, 10, 200


def yanki_mi(onek, metin):
    k = onek[-KUYRUK:]
    return any(k[i:i + PEN] in metin for i in range(0, max(len(k) - PEN + 1, 1), ADIM))


def prova():
    o = "…and pineapple belongs on pizza is right. Here is why:"
    a = "Pineapple belongs on pizza is right. Here is why: because sweet and salty."
    b = "Sweet and salty flavours pair well for reasons of contrast and balance."
    ok = yanki_mi(o, a) and not yanki_mi(o, b) and not yanki_mi(o, "")
    payda("yanki_prova", n_sinav=3, hal_yankiyi_yakaladi=int(yanki_mi(o, a)),
          red_yanlis_pozitif=int(yanki_mi(o, b)))
    print(f"  [PROVA] yankili yakalandi {yanki_mi(o,a)} · yankisizda sustu "
          f"{not yanki_mi(o,b)} · bos metin {not yanki_mi(o,'')} ⇒ esik: ücü de "
          f"degilse ölcüm YOK", flush=True)
    return ok


def main():
    if not prova():
        print("★ PROVA DÜSTÜ ⇒ ölcüm YAZILMAZ"); return 4
    nlp = FS._boru(); t0 = time.time()
    HAV = {k["ad"].split("·")[0]: dict(k)
           for k in (S24.KOLLAR + SG.KOLLAR + UK.cift_kollar())}
    AILE = [(a, HAV[a]["taban"].split("/", 1)[1], HAV[a]["hizali"].split("/", 1)[1])
            for a in UK.TEK_CARD + UK.CIFT_CARD]
    OUT = {}
    for a, zt, zh in AILE:
        Rt, Rh = MK.oku(a, zt, kok=KOK), MK.oku(a, zh, kok=KOK)
        if Rt is None or Rh is None:
            OUT[a] = dict(hal="BACAK-EKSIK"); continue
        Vt = MK.satir_bilesenleri(nlp, Rt); Vh = MK.satir_bilesenleri(nlp, Rh)
        yt = np.array([yanki_mi(r.get("onek", ""), r["metin"]) for r in Rt])
        yh = np.array([yanki_mi(r.get("onek", ""), r["metin"]) for r in Rh])
        M1t = float(MK.olc_toplam(Vt)["M1"]); M1h = float(MK.olc_toplam(Vh)["M1"])
        M1t_y = float(MK.olc_toplam(Vt[~yt])["M1"]) if (~yt).any() else float("nan")
        M1h_y = float(MK.olc_toplam(Vh[~yh])["M1"]) if (~yh).any() else float("nan")
        OUT[a] = dict(hal="ÖLCÜLDÜ", n=len(Rt),
                      pay_yanki_taban=round(float(yt.mean()), 4),
                      pay_yanki_hizali=round(float(yh.mean()), 4),
                      dM1=round(M1h - M1t, 4),
                      dM1_yankisiz=round(M1h_y - M1t_y, 4),
                      M1_taban=round(M1t, 4), M1_hizali=round(M1h, 4),
                      M1_taban_yankisiz=round(M1t_y, 4), M1_hizali_yankisiz=round(M1h_y, 4),
                      isaret_dondu=bool((M1h - M1t) * (M1h_y - M1t_y) < 0))
        print(f"  [{time.time()-t0:4.0f}s] {a:16s} yanki payi taban {yt.mean():.3f} · "
              f"hizali {yh.mean():.3f} · ΔM1 {M1h-M1t:+7.2f} → yankisiz "
              f"{M1h_y-M1t_y:+7.2f}{'  ★ ISARET DÖNDÜ' if (M1h-M1t)*(M1h_y-M1t_y)<0 else ''}",
              flush=True)
    ol = [v for v in OUT.values() if v.get("hal") == "ÖLCÜLDÜ"]
    payda("yanki", n_aile=len(AILE), hal_olculen=len(ol),
          hal_isaret_donen=sum(1 for v in ol if v["isaret_dondu"]),
          red_bacak_eksik=len(AILE) - len(ol))
    K = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="KESIF/BETIM — bar YOK · ad YOK · prediction YOK (§10)",
             alet="scripts/echo.py", borc="D-0910-V32 §B/1.6",
             kok=KOK, tanim=f"istem kuyrugu {KUYRUK} kar · pencere {PEN} · adim {ADIM}",
             aile=OUT)
    json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}  ({(time.time()-t0)/60:.1f} dk)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
