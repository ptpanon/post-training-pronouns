#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
import addressee_run as MK
import form_count as FS
import rung_force_ci as BK
from reading_style_object import payda

REF_KOK = __DNH_DATA__ + "/c1_panel"
YET_KOK = __DNH_DATA__ + "/c1_panel_yetenek"
CIK = os.environ.get("YET_CIK", f"{KOK}/results/capability_scorecard_2026-09-13.json")
CIFT_CARD = ("Llama-3.1-70B", "Qwen2.5-72B")
TABAN_CARD = f"{KOK}/results/YETENEK_TABAN_CIPLAK_2026-09-12.json"
D_SHELF, BAR_AILE, BAR_GERI, BAR_DEJ = 0.60, 12, 0.50, 4
TABAN_ADLARI = ("base", "taban")
HIZALI_ADLARI = ("instruct", "rl", "dpo")


def _bacak(aile, adaylar):
    for ad in adaylar:
        y = f"{REF_KOK}/{aile}/{ad}/uretim.jsonl"
        if os.path.exists(y):
            return ad, y
    return None, None


def d4_metin(ts):
    v = []
    for t in ts:
        w = t.split()
        if len(w) < 8:
            continue
        g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        v.append(len(set(g)) / len(g))
    return float(np.mean(v)) if v else 0.0


def oku(yol):
    if not os.path.exists(yol):
        return None
    return [json.loads(l) for l in io.open(yol, encoding="utf-8")]


def main():
    nlp = FS._boru()
    T = json.load(io.open(TABAN_CARD, encoding="utf-8"))
    aileler = sorted(os.listdir(f"{YET_KOK}/kayit"))
    R, t0 = {}, time.time()
    for a in aileler:
        h_ad, h_yol = _bacak(a, HIZALI_ADLARI)
        t_ad, t_yol = _bacak(a, TABAN_ADLARI)
        r = dict(tarif_taban=T.get(a, {}).get("M1_base"),
                 hizali_ad=h_ad, taban_ad=t_ad)
        if not h_ad or not t_ad:
            R[a] = dict(hal="EKSIK-REFERANS", hizali_ad=h_ad, taban_ad=t_ad,
                        aranan=dict(hizali=list(HIZALI_ADLARI), taban=list(TABAN_ADLARI)))
            print(f"  ★★ {a}: EKSIK-REFERANS (hizali={h_ad} · taban={t_ad})", flush=True)
            continue
        ref = oku(h_yol); tab = oku(t_yol)
        Vr = MK.satir_bilesenleri(nlp, ref); Vt = MK.satir_bilesenleri(nlp, tab)
        ir = np.array([x.get("istem_i", -1) for x in ref])
        m_ref = float(MK.olc_toplam(Vr)["M1"]); m_tab = float(MK.olc_toplam(Vt)["M1"])
        payda_ = m_tab - m_ref
        r.update(M1_hizali=round(m_ref, 4), M1_taban=round(m_tab, 4),
                 payda=round(payda_, 4), n_ref=len(ref))
        for kol in ("kayit", "acik"):
            K = oku(f"{YET_KOK}/{kol}/{a}/instruct/uretim.jsonl")
            if K is None:
                r[kol] = dict(hal=""); continue
            d4 = d4_metin([x["metin"] for x in K])
            Vk = MK.satir_bilesenleri(nlp, K)
            ik = np.array([x.get("istem_i", -1) for x in K])
            m_k = float(MK.olc_toplam(Vk)["M1"])
            lo, hi, nk = BK.kume_boot(Vk, ik, Vr, ir, "M1")
            gk = (m_k - m_ref) / payda_ if payda_ > 0 else float("nan")
            r[kol] = dict(M1=round(m_k, 4), dM1=round(m_k - m_ref, 4),
                          ci=[round(lo, 4), round(hi, 4)], ayrik=bool(lo * hi > 0),
                          n_istem=int(nk), geri_kazanim=round(gk, 4),
                          d4_metin=round(d4, 4), dejenere=bool(d4 < D_SHELF),
                          hal="ÖLCÜLDÜ")
        R[a] = r
        k, c = r.get("kayit", {}), r.get("acik", {})
        print(f"  {a:<17} payda {payda_:+6.2f} · kayit ΔM1 {k.get('dM1',float('nan')):+7.2f} "
              f"gk {k.get('geri_kazanim',float('nan')):+6.2f} d4 {k.get('d4_metin',0):.3f} · "
              f"acik ΔM1 {c.get('dM1',float('nan')):+7.2f} gk "
              f"{c.get('geri_kazanim',float('nan')):+6.2f} d4 {c.get('d4_metin',0):.3f}",
              flush=True)

    H = {}
    for kol in ("kayit", "acik"):
        olc = [a for a, r in R.items()
               if r.get(kol, {}).get("hal") == "ÖLCÜLDÜ" and r.get("payda", 0) > 0]
        dej = [a for a in olc if R[a][kol]["dejenere"]]
        gk = [R[a][kol]["geri_kazanim"] for a in olc]
        med = float(np.median(gk)) if gk else float("nan")
        if len(olc) < BAR_AILE or len(dej) > BAR_DEJ:
            hal = "ÖLCÜLEMEZ"
        elif med <= 0:
            hal = "YETENEK YOK"
        elif med >= BAR_GERI:
            hal = "YETENEK SAGLAM"
        else:
            hal = "KISMÎ"
        H[kol] = dict(VERDICT=hal, n_olculen=len(olc), n_dejenere=len(dej),
                      dejenere_aile=dej, medyan_geri_kazanim=round(med, 4),
                      min=round(min(gk), 4) if gk else None,
                      maks=round(max(gk), 4) if gk else None,
                      n_gk_ge_1=sum(1 for x in gk if x > 1),
                      n_ayrik_poz=sum(1 for a in olc
                                      if R[a][kol]["ayrik"] and R[a][kol]["dM1"] > 0))
    ort = [a for a in R if R[a].get("kayit", {}).get("hal") == "ÖLCÜLDÜ"
           and R[a].get("acik", {}).get("hal") == "ÖLCÜLDÜ"]
    n_acik_ge = sum(1 for a in ort
                    if R[a]["acik"]["geri_kazanim"] >= R[a]["kayit"]["geri_kazanim"])
    PREDICTION = {
     "F-YET-1": ("TUTTU" if H["acik"]["VERDICT"] == "YETENEK SAGLAM" else "DÜSTÜ",
                 f"acik paneli = {H['acik']['VERDICT']}"),
     "F-YET-2": ("TUTTU" if H["kayit"]["VERDICT"] == "YETENEK SAGLAM" else "DÜSTÜ",
                 f"kayit paneli = {H['kayit']['VERDICT']}"),
     "F-YET-3": ("TUTTU" if n_acik_ge >= 12 else "DÜSTÜ",
                 f"acik ≥ kayit: {n_acik_ge}/{len(ort)} (bar 12)"),
     "F-YET-4": ("TUTTU" if H["acik"]["n_gk_ge_1"] >= 1 else "DÜSTÜ",
                 f"{H['acik']['n_gk_ge_1']}"),
     "F-YET-5": ("TUTTU" if max(H['kayit']['n_dejenere'], H['acik']['n_dejenere']) <= BAR_DEJ
                 else "DÜSTÜ",
                 f"dejenere hücre: kayit {H['kayit']['n_dejenere']} · "
                 f"acik {H['acik']['n_dejenere']} (bar ≤4)"),
    }
    out = dict(_kunye=dict(alet="scripts/capability_verdict.py",
               damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="preregistration/prereg_capability_2026-09-12.md @ d50b7555",
               prediction="preregistration/prediction_capability_2026-09-12.md @ 649f4c36",
               motor="muhatap_kos + rung_force_ci.kume_boot (ITHAL, edit yok)",
               d_shelf=D_SHELF, d_shelf_olcusu="METIN BASINA (W-1101)",
               bar_aile=BAR_AILE, bar_geri=BAR_GERI, bar_dejenere=BAR_DEJ,
               atlanan_aile=[x for x in CIFT_CARD if x not in aileler],
               atlanan_sebep=("ATLANDI-CIFT-CARD-GEREK (tek kartta sigmaz)"
                              if any(x not in aileler for x in CIFT_CARD) else "YOK"),
               n_aile_diskte=len(aileler),
               sure_dk=round((time.time() - t0) / 60, 1)),
               PANEL=H, PREDICTION=PREDICTION, aile=R)
    io.open(CIK, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print()
    for kol in ("kayit", "acik"):
        h = H[kol]
        print(f"  ★★ {kol.upper():<6} {h['VERDICT']:<16} medyan geri_kazanim "
              f"{h['medyan_geri_kazanim']:+.3f} (min {h['min']:+.2f} · maks {h['maks']:+.2f}) "
              f"· ölcülen {h['n_olculen']}/{len(aileler)} · dejenere {h['n_dejenere']} · "
              f"CI-ayrik pozitif {h['n_ayrik_poz']}")
    for b, (s, g) in PREDICTION.items():
        print(f"     {b}: {s:<6} — {g}")
    payda("yetenek_verdict", n_aile=len(R), hal_olculen=H["kayit"]["n_olculen"],
          red_dejenere=H["kayit"]["n_dejenere"] + H["acik"]["n_dejenere"],
          hal_dk=round((time.time() - t0) / 60, 1))
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
