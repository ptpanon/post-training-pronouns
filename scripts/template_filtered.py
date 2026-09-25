#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
import rung_force_ci as BK
import dejenerelik_olcer as D
import v23_template_count as V23
from reading_style_object import payda

SAB = __DNH_DATA__ + "/c1_panel_template"
CIP = __DNH_DATA__ + "/c1_panel"
CIK = f"{ROOT}/results/template_filtered_2026-09-07.json"
N_BEK = 6528
TUT_ESIK = 0.25


def oku(kok, aile, zemin):
    y = f"{kok}/{aile}/{zemin}/uretim.jsonl"
    if not os.path.exists(y):
        return None
    return [json.loads(l) for l in io.open(y, encoding="utf-8")]


def anahtar(r):
    return (r.get("kol"), r.get("istem_i"), r.get("cekim"))


def bayraklar(R):
    dej, d4 = [], []
    for r in R:
        b = D.bayrakla(r["metin"])
        dej.append(bool(b["DEJENERE"])); d4.append(bool(b["f_yin4"]))
    return np.array(dej), np.array(d4)


def main():
    nlp = FS._boru(); t0 = time.time()
    CIFT, _ = V23.cift_kur()
    OUT, satir = {}, []
    for n, (a, zt, zh) in enumerate(CIFT, 1):
        bacaklar = {("sbl", "taban"): oku(SAB, a, zt), ("sbl", "hizali"): oku(SAB, a, zh),
                    ("cip", "taban"): oku(CIP, a, zt), ("cip", "hizali"): oku(CIP, a, zh)}
        if any(R is None or len(R) != N_BEK for R in bacaklar.values()):
            print(f"{a}"); continue
        F = {k: bayraklar(R) for k, R in bacaklar.items()}
        K = {k: [anahtar(r) for r in R] for k, R in bacaklar.items()}
        kirli = set()
        for k in bacaklar:
            kirli |= {K[k][i] for i in np.where(F[k][0])[0]}
        tut = {k: np.array([anah not in kirli for anah in K[k]]) for k in bacaklar}
        oran = float(tut[("sbl", "taban")].mean())
        hucre = {f"{p}_{z}": dict(dej=float(F[(p, z)][0].mean()),
                                  d4=float(F[(p, z)][1].mean()))
                 for p, z in bacaklar}
        r = dict(aile=a, zemin_taban=zt, zemin_hizali=zh, hucre=hucre,
                 tutulan_oran=round(oran, 4), tutulan_n=int(tut[("sbl", "taban")].sum()))
        if oran < TUT_ESIK:
            r["hal"] = "ÖLCÜLEMEZ-SÜZGEC"
            satir.append(r); OUT[a] = r
            print(f"  ★★ {a}: tutulan {oran:.1%} < %25 ⇒ ÖLCÜLEMEZ-SÜZGEC"); continue
        for p, kok in (("sbl", SAB), ("cip", CIP)):
            Vt = MK.satir_bilesenleri(nlp, bacaklar[(p, "taban")])
            Vh = MK.satir_bilesenleri(nlp, bacaklar[(p, "hizali")])
            it = np.array([x.get("istem_i", -1) for x in bacaklar[(p, "taban")]])
            ih = np.array([x.get("istem_i", -1) for x in bacaklar[(p, "hizali")]])
            mt, mh = tut[(p, "taban")], tut[(p, "hizali")]
            d = MK.olc_toplam(Vh[mh])["M1"] - MK.olc_toplam(Vt[mt])["M1"]
            lo, hi, nk = BK.kume_boot(Vh[mh], ih[mh], Vt[mt], it[mt], "M1")
            d0 = MK.olc_toplam(Vh)["M1"] - MK.olc_toplam(Vt)["M1"]
            r[p] = dict(dM1=round(float(d), 4), ci=[round(lo, 4), round(hi, 4)],
                        ayrik=bool(lo * hi > 0), n_istem=int(nk),
                        dM1_suzgecsiz=round(float(d0), 4),
                        M1_taban=round(float(MK.olc_toplam(Vt[mt])["M1"]), 4),
                        M1_hizali=round(float(MK.olc_toplam(Vh[mh])["M1"]), 4))
        r["hal"] = "ÖLCÜLDÜ"
        satir.append(r); OUT[a] = r
        s, c = r["sbl"], r["cip"]
        print(f"  ★ {a:15s} tutulan {oran:5.1%} · SABLONLU ΔM1 {s['dM1']:+7.2f} "
              f"[{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}]{' AYRIK' if s['ayrik'] else '      '}"
              f" · CIPLAK {c['dM1']:+6.2f} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]"
              f"{' AYRIK' if c['ayrik'] else ''}  [{n}/{len(CIFT)} · {(time.time()-t0)/60:.1f} dk]",
              flush=True)

    olc = [r for r in satir if r["hal"] == "ÖLCÜLDÜ"]
    def say(p):
        poz = [r["aile"] for r in olc if r[p]["ayrik"] and r[p]["dM1"] > 0]
        neg = [r["aile"] for r in olc if r[p]["ayrik"] and r[p]["dM1"] < 0]
        nul = [r["aile"] for r in olc if not r[p]["ayrik"]]
        return poz, neg, nul
    ps, ns, us = say("sbl"); pc, nc, uc = say("cip")
    temiz_poz = [r["aile"] for r in olc if r["sbl"]["ayrik"] and r["sbl"]["dM1"] > 0
                 and r["hucre"]["sbl_taban"]["dej"] < 0.30]
    print(f"\n★ SABLONLU (same-string, süzülmüs): pozitif-ayrik {len(ps)} · "
          f"negatif-ayrik {len(ns)} · null {len(us)}")
    print(f"  pozitif: {ps}\n  negatif: {ns}\n  null: {us}")
    print(f"★ CIPLAK (ayni süzgec):            pozitif-ayrik {len(pc)} · "
          f"negatif-ayrik {len(nc)} · null {len(uc)}")
    print(f"  pozitif: {pc}\n  negatif: {nc}\n  null: {uc}")
    print(f"★ TEMIZ-VE-POZITIF (taban dej < 0,30): {temiz_poz}")
    payda("template_filtered", n_aile=len(CIFT), hal_olculdu=len(olc),
          hal_sablonlu_poz=len(ps), hal_sablonlu_neg=len(ns), hal_sablonlu_null=len(us),
          hal_ciplak_neg=len(nc), red_olculemez=len(satir) - len(olc))
    print(f"\n★ DAL KURALI (direktif §3): süzülmüs sablonlu ΔM1'de pozitif-ayrik ≥3 "
          f"⇒ DAL C · ölcülen {len(ps)} ⇒ **{'C ÖNERILIR' if len(ps) >= 3 else 'C ÖNERILMEZ'}**")
    json.dump(dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               alet="scripts/template_filtered.py",
                               borc="D-0908-TEMPLATE-VERDICT §1",
                               suzgec="hücre DÖRT hücrede de dejenere degilse girer (tek yol)",
                               tut_esik=TUT_ESIK, n_beklenen_satir=N_BEK,
                               gecen_dk=round((time.time()-t0)/60, 1)),
                   aileler=satir,
                   sayim=dict(sablonlu=dict(pozitif=ps, negatif=ns, null=us),
                              ciplak=dict(pozitif=pc, negatif=nc, null=uc),
                              temiz_ve_pozitif=temiz_poz,
                              dal_C=bool(len(ps) >= 3))),
              io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
