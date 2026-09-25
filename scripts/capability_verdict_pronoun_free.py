#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, io, json, os, sys, time
import numpy as np
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
from reading_style_object import payda

G = f"{KOK}/results"
CARD16 = f"{G}/YETENEK_KARNE_16_2026-09-13.json"
CARD16_SHA = "2c17ba6aa1aa1f1c"
TABAN = f"{G}/capability_pronoun_free_threshold_floor_2026-09-14.json"
TABAN_SHA = "cb8154ffa7a82228"
CIK = f"{G}/capability_pronoun_free_scorecard_2026-09-14.json"
KOL = "kayit_sahissiz"
PANEL_ADLARI = ("ÖLCÜLEMEZ", "YETENEK YOK", "KISMÎ", "SAGLAM-ASMASIZ", "ASMA")
KOPYA_ADLARI = ("ÖLCÜLEMEZ", "KOPYA-ICINDE", "KOPYADAN-BÜYÜK-DÜSÜS", "KOPYADAN-BÜYÜK-ARTIS")
BAR_AILE, BAR_DEJ, BAR_GERI, BAR_ASMA, BAR_CI = 12, 4, 0.50, 1.0, 12


def sha16(y):
    return hashlib.sha256(open(y, "rb").read()).hexdigest()[:16]


def panel_hukmu(n_olculen, n_dejenere, medyan):
    if n_olculen < BAR_AILE or n_dejenere > BAR_DEJ:
        return "ÖLCÜLEMEZ"
    if medyan <= 0:
        return "YETENEK YOK"
    if medyan < BAR_GERI:
        return "KISMÎ"
    if medyan <= BAR_ASMA:
        return "SAGLAM-ASMASIZ"
    return "ASMA"


def kopya_hanesi(panel, D, E):
    if panel == "ÖLCÜLEMEZ":
        return "ÖLCÜLEMEZ"
    if D > E:
        return "KOPYADAN-BÜYÜK-DÜSÜS"
    if D < -E:
        return "KOPYADAN-BÜYÜK-ARTIS"
    return "KOPYA-ICINDE"


def tarama():
    import full_tarama as TT
    R1 = TT.tam_tarama(lambda n_olculen, n_dejenere, medyan: panel_hukmu(n_olculen, n_dejenere, medyan),
                       dict(n_olculen=list(range(0, 17)), n_dejenere=list(range(0, 17)),
                            medyan=TT.izgara(-1.0, 3.0, 81)),
                       PANEL_ADLARI, etiket="tarama_panel")
    TT.bas(R1, "PANEL")
    E = json.load(open(TABAN))["egim"]["E_medyan"]
    K16 = json.load(open(CARD16))["PANEL"]["kayit"]["medyan_geri_kazanim"]
    izg = dict(n_olculen=[11, 16], n_dejenere=[0, 5], medyan=TT.izgara(-1.0, 3.0, 161))
    fn = lambda n_olculen, n_dejenere, medyan, D: kopya_hanesi(panel_hukmu(n_olculen, n_dejenere, medyan), D, E)
    R2 = TT.tam_tarama(fn, izg, KOPYA_ADLARI, etiket="tarama_kopya_veri",
                       turetilmis=dict(D=lambda kw: K16 - kw["medyan"]))
    TT.bas(R2, f"KOPYA · veri yolu (D = {K16} − medyan, E = {E})")
    R3 = TT.tam_tarama(lambda n_olculen, D: kopya_hanesi(panel_hukmu(n_olculen, 0, 1.0), D, E),
                       dict(n_olculen=[11, 16], D=TT.izgara(-3.0, 3.0, 121)), KOPYA_ADLARI,
                       etiket="tarama_kopya_izgara")
    TT.bas(R3, "KOPYA · izgara")
    return TT.kapi(R1, R2, R3, TT.veri_yolu_karsilastir(R3, R2))


def aile_oku(nlp, a, kol_kok):
    import addressee_run as MK, rung_force_ci as BK, capability_verdict as YH
    h_ad, h_yol = YH._bacak(a, YH.HIZALI_ADLARI)
    t_ad, t_yol = YH._bacak(a, YH.TABAN_ADLARI)
    if not h_ad or not t_ad:
        return dict(hal="EKSIK-REFERANS", hizali_ad=h_ad, taban_ad=t_ad)
    ref = YH.oku(h_yol); tab = YH.oku(t_yol)
    Vr = MK.satir_bilesenleri(nlp, ref); Vt = MK.satir_bilesenleri(nlp, tab)
    ir = np.array([x.get("istem_i", -1) for x in ref])
    m_ref = float(MK.olc_toplam(Vr)["M1"]); m_tab = float(MK.olc_toplam(Vt)["M1"])
    payda_ = m_tab - m_ref
    r = dict(hizali_ad=h_ad, taban_ad=t_ad, M1_hizali=round(m_ref, 4), M1_taban=round(m_tab, 4),
             payda=round(payda_, 4), n_ref=len(ref))
    K = YH.oku(f"{kol_kok}/{a}/instruct/uretim.jsonl")
    if K is None:
        r["kol"] = dict(hal=""); return r
    d4 = YH.d4_metin([x["metin"] for x in K])
    Vk = MK.satir_bilesenleri(nlp, K)
    ik = np.array([x.get("istem_i", -1) for x in K])
    m_k = float(MK.olc_toplam(Vk)["M1"])
    lo, hi, nk = BK.kume_boot(Vk, ik, Vr, ir, "M1")
    gk = (m_k - m_ref) / payda_ if payda_ > 0 else float("nan")
    r["kol"] = dict(M1=round(m_k, 4), dM1=round(m_k - m_ref, 4), ci=[round(lo, 4), round(hi, 4)],
                    ayrik=bool(lo * hi > 0), n_istem=int(nk), n_satir=len(K), geri_kazanim=round(gk, 4),
                    d4_metin=round(d4, 4), dejenere=bool(d4 < YH.D_SHELF), hal="ÖLCÜLDÜ")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarama", action="store_true", help="yalniz K-f tam taramasi")
    ap.add_argument("--prova", default=None, help="")
    a = ap.parse_args()
    if a.tarama:
        return tarama()
    import form_count as FS, capability_verdict as YH
    if sha16(CARD16) != CARD16_SHA or sha16(TABAN) != TABAN_SHA:
        print(f"{sha16(CARD16)} {sha16(TABAN)}"); return 3
    C16 = json.load(open(CARD16)); TB = json.load(open(TABAN))
    nlp = FS._boru(); t0 = time.time()
    if a.prova:
        fark = 0; n = 0
        for aile in a.prova.split(","):
            r = aile_oku(nlp, aile, f"{YH.YET_KOK}/kayit"); c = C16["aile"][aile]
            cift = [("M1_hizali", r["M1_hizali"], c["M1_hizali"]), ("M1_taban", r["M1_taban"], c["M1_taban"]),
                    ("payda", r["payda"], c["payda"])] + \
                   [(k, r["kol"][k], c["kayit"][k]) for k in ("M1", "dM1", "ci", "geri_kazanim", "d4_metin", "n_istem", "ayrik")]
            for k, x, y in cift:
                n += 1; fark += int(x != y)
                print(f"  [PROVA] {aile:<14} {k:<13} alet {x} · card {y} ⇒ {'AYNI' if x == y else '★ FARKLI'}")
        payda("yetenek_sahissiz_prova", n_alan=n, red_farkli=fark, hal_dk=round((time.time() - t0) / 60, 1))
        return 0 if fark == 0 else 3
    aileler = sorted(C16["aile"])
    R = {}; red_ref = []
    for aile in aileler:
        r = aile_oku(nlp, aile, f"{YH.YET_KOK}/{KOL}"); R[aile] = r
        c = C16["aile"][aile]
        if r.get("hal") == "EKSIK-REFERANS" or (r["M1_hizali"], r["M1_taban"], r["payda"]) != \
                (c["M1_hizali"], c["M1_taban"], c["payda"]):
            red_ref.append(aile)
        k = r.get("kol", {})
        print(f"  {aile:<17} payda {r.get('payda', float('nan')):+6.2f} · {KOL} ΔM1 "
              f"{k.get('dM1', float('nan')):+7.2f} gk {k.get('geri_kazanim', float('nan')):+6.2f} "
              f"d4 {k.get('d4_metin', 0):.3f}", flush=True)
    if red_ref:
        print(f"★★ REFERANS KANONIK KARTTAN SAPTI: {red_ref} ⇒ verdict YAZILMADI · cikis 3")
        payda("yetenek_sahissiz_verdict", n_aile=len(R), red_referans_sapma=len(red_ref)); return 3
    olc = [x for x in aileler if R[x].get("kol", {}).get("hal") == "ÖLCÜLDÜ" and R[x]["payda"] > 0]
    dej = [x for x in olc if R[x]["kol"]["dejenere"]]
    gk = [R[x]["kol"]["geri_kazanim"] for x in olc]
    med = float(np.median(gk)) if gk else float("nan")
    P = panel_hukmu(len(olc), len(dej), med)
    MED16 = C16["PANEL"]["kayit"]["medyan_geri_kazanim"]; E = TB["egim"]["E_medyan"]
    D = MED16 - med
    KH = kopya_hanesi(P, D, E)
    n_poz = sum(1 for x in olc if R[x]["kol"]["ayrik"] and R[x]["kol"]["dM1"] > 0)
    n_neg = sum(1 for x in olc if R[x]["kol"]["ayrik"] and R[x]["kol"]["dM1"] < 0)
    MDE_D = 0.07
    kil = bool(P != "ÖLCÜLEMEZ" and (abs(D - E) < MDE_D or abs(D + E) < MDE_D
                                     or abs(med - BAR_ASMA) < MDE_D or abs(med - BAR_GERI) < MDE_D))
    PREDICTION = {
     "F-YSZ-1": ("SKORLANAMAZ" if P == "ÖLCÜLEMEZ" else ("TUTTU" if P == "ASMA" else "DÜSTÜ"), f"panel = {P}"),
     "F-YSZ-2": ("SKORLANAMAZ" if KH == "ÖLCÜLEMEZ" else ("TUTTU" if KH == "KOPYA-ICINDE" else "DÜSTÜ"),
                 f"kopya hanesi = {KH} (D {D:+.4f} · E {E:.4f})"),
     "F-YSZ-3": ("SKORLANAMAZ" if P == "ÖLCÜLEMEZ" else ("TUTTU" if n_poz >= BAR_CI else "DÜSTÜ"),
                 f"CI-ayrik pozitif {n_poz}/{len(olc)} (bar {BAR_CI})"),
    }
    out = dict(_kunye=dict(alet="scripts/capability_verdict_pronoun_free.py", kol=KOL,
                           damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           prereg="preregistration/prereg_capability_pronoun_free_2026-09-14.md",
                           kanonik_card=[CARD16, CARD16_SHA], taban=[TABAN, TABAN_SHA],
                           motor="muhatap_kos + rung_force_ci.kume_boot + yetenek_verdict (ITHAL)",
                           bar_aile=BAR_AILE, bar_dejenere=BAR_DEJ, bar_geri=BAR_GERI, bar_asma=BAR_ASMA,
                           bar_ci=BAR_CI, mde_vekil_D=MDE_D, sure_dk=round((time.time() - t0) / 60, 1)),
               PANEL=dict(VERDICT=P, n_olculen=len(olc), n_dejenere=len(dej), dejenere_aile=dej,
                          medyan_geri_kazanim=round(med, 4), min=round(min(gk), 4) if gk else None,
                          maks=round(max(gk), 4) if gk else None, n_gk_gt_1=sum(1 for x in gk if x > 1),
                          n_ayrik_poz=n_poz, n_ayrik_neg=n_neg),
               KOPYA=dict(VERDICT=KH, kanonik_medyan=MED16, yeni_medyan=round(med, 4), D=round(D, 4), E=E,
                          KIL_PAYI=kil),
               PREDICTION=PREDICTION, aile=R)
    io.open(CIK, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n  ★★ PANEL {P} · medyan geri_kazanim {med:+.4f} · ölcülen {len(olc)}/16 · dejenere {len(dej)} · "
          f"CI-ayrik + {n_poz} / − {n_neg}")
    print(f"  ★★ KOPYA {KH} · D {D:+.4f} (kanonik {MED16} − yeni {med:.4f}) · E {E}"
          f"{'' if kil else ''}")
    for b, (s, g) in PREDICTION.items():
        print(f"     {b}: {s:<11} — {g}")
    payda("yetenek_sahissiz_verdict", n_aile=len(R), hal_olculen=len(olc), red_dejenere=len(dej),
          red_referans_sapma=0, hal_dk=out["_kunye"]["sure_dk"])
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
