#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import depersonalization16_neutral_run as M9
import form_count as FS
import rung_force_ci as BK
import template_filtered as SS
from reading_style_object import payda

CARD = f"{ROOT}/results/template_filtered_2026-09-07.json"
CIK = f"{ROOT}/results/template_filtered_first_person_2026-09-15.json"
I2, IY = MK.ALAN.index("m1_sahis2"), MK.ALAN.index("m5_yakin")


def birinci(V):
    W = V.copy(); W[:, I2] = V[:, IY] - V[:, I2]
    return W


def main():
    nlp = FS._boru(); t0 = time.time()
    K = {r["aile"]: r for r in json.load(open(CARD))["aileler"]}
    CIFT, _ = SS.V23.cift_kur()
    out, fark, esd = [], 0, 0
    for n, (a, zt, zh) in enumerate(CIFT, 1):
        bac = {("sbl", "taban"): SS.oku(SS.SAB, a, zt), ("sbl", "hizali"): SS.oku(SS.SAB, a, zh),
               ("cip", "taban"): SS.oku(SS.CIP, a, zt), ("cip", "hizali"): SS.oku(SS.CIP, a, zh)}
        if any(R is None or len(R) != SS.N_BEK for R in bac.values()):
            print(f"  ✗ {a}: dört hücre TAM degil"); continue
        F = {k: SS.bayraklar(R) for k, R in bac.items()}
        A = {k: [SS.anahtar(r) for r in R] for k, R in bac.items()}
        kirli = set()
        for k in bac:
            kirli |= {A[k][i] for i in np.where(F[k][0])[0]}
        tut = {k: np.array([x not in kirli for x in A[k]]) for k in bac}
        oran = float(tut[("sbl", "taban")].mean())
        r = dict(aile=a, tutulan_oran=round(oran, 4))
        if oran < SS.TUT_ESIK:
            r["hal"] = "ÖLCÜLEMEZ-SÜZGEC"; out.append(r); continue
        for p in ("sbl", "cip"):
            Vt = MK.satir_bilesenleri(nlp, bac[(p, "taban")]); Vh = MK.satir_bilesenleri(nlp, bac[(p, "hizali")])
            it = np.array([x.get("istem_i", -1) for x in bac[(p, "taban")]])
            ih = np.array([x.get("istem_i", -1) for x in bac[(p, "hizali")]])
            mt, mh = tut[(p, "taban")], tut[(p, "hizali")]
            d2 = MK.olc_toplam(Vh[mh])["M1"] - MK.olc_toplam(Vt[mt])["M1"]
            lo2, hi2, _ = BK.kume_boot(Vh[mh], ih[mh], Vt[mt], it[mt], "M1")
            k = K[a][p]
            ok = (round(float(d2), 4), [round(lo2, 4), round(hi2, 4)]) == (k["dM1"], k["ci"])
            esd += 1; fark += int(not ok)
            Wt, Wh = birinci(Vt), birinci(Vh)
            assert abs(MK.olc_toplam(Wt[mt])["M1"] - M9.sahis1_1k(Vt[mt])) < 1e-9
            d1 = MK.olc_toplam(Wh[mh])["M1"] - MK.olc_toplam(Wt[mt])["M1"]
            lo1, hi1, nk = BK.kume_boot(Wh[mh], ih[mh], Wt[mt], it[mt], "M1")
            r[p] = dict(d2=round(float(d2), 4), ci2=[round(lo2, 4), round(hi2, 4)], esdeger_card=ok,
                        d1=round(float(d1), 4), ci1=[round(lo1, 4), round(hi1, 4)], ayrik1=bool(lo1 * hi1 > 0),
                        n_istem=int(nk), S1_taban=round(float(M9.sahis1_1k(Vt[mt])), 4),
                        S1_hizali=round(float(M9.sahis1_1k(Vh[mh])), 4))
        r["hal"] = "ÖLCÜLDÜ"; out.append(r)
        s, c = r["sbl"], r["cip"]
        print(f"  {a:16s} SABLONLU 1.s {s['d1']:+7.2f} [{s['ci1'][0]:+.2f},{s['ci1'][1]:+.2f}] (2.s {s['d2']:+.2f} card={s['esdeger_card']}) · "
              f"CIPLAK 1.s {c['d1']:+6.2f} [{c['ci1'][0]:+.2f},{c['ci1'][1]:+.2f}] (card={c['esdeger_card']})  [{n}/{len(CIFT)} · {(time.time()-t0)/60:.1f} dk]", flush=True)
    olc = [r for r in out if r["hal"] == "ÖLCÜLDÜ"]
    def say(p, ad):
        poz = [r["aile"] for r in olc if r[p][f"ayrik{ad}"] and r[p][f"d{ad}"] > 0] if ad == "1" else \
              [r["aile"] for r in olc if r[p]["ci2"][0] * r[p]["ci2"][1] > 0 and r[p]["d2"] > 0]
        neg = [r["aile"] for r in olc if r[p][f"ayrik{ad}"] and r[p][f"d{ad}"] < 0] if ad == "1" else \
              [r["aile"] for r in olc if r[p]["ci2"][0] * r[p]["ci2"][1] > 0 and r[p]["d2"] < 0]
        nul = [r["aile"] for r in olc if r["aile"] not in poz + neg]
        return dict(yukari=poz, asagi=neg, null=nul)
    S = {f"{p}_{ad}": say(p, ad) for p in ("sbl", "cip") for ad in ("1", "2")}
    capraz = {}
    for r in olc:
        def sinif(d, ci): return "yukari" if ci[0] > 0 else ("asagi" if ci[1] < 0 else "null")
        capraz[r["aile"]] = dict(sablonlu_2=sinif(r["sbl"]["d2"], r["sbl"]["ci2"]), sablonlu_1=sinif(r["sbl"]["d1"], r["sbl"]["ci1"]))
    uyum = sum(1 for v in capraz.values() if v["sablonlu_1"] == v["sablonlu_2"])
    json.dump(dict(_kunye=dict(alet="scripts/template_filtered_first_person.py", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               sinif="",
                               olcu_birinci_sahis="depersonalization16_neutral_run.sahis1_1k = 1000·(m5_yakin − m1_sahis2)/n_jeton (türetilmis; cümle basi hitap yanliligi adiyla)",
                               suzgec="template_filtered: hücre dört bacakta da dejenere degilse (ayni rule)",
                               aralik="",
                               kanonik_card=CARD, esdeger_alan=esd, esdeger_fark=fark,
                               sure_dk=round((time.time() - t0) / 60, 1)),
                   sayim=S, capraz_sablonlu=capraz, uyum_sablonlu_1_2=uyum, aileler=out),
              io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in S.items():
        print(f"★ {k}: yukari {len(v['yukari'])} · asagi {len(v['asagi'])} · null {len(v['null'])}")
    print(f"★ sablonlu 1.s ↔ 2.s ayni sinif: {uyum}/{len(olc)}")
    payda("template_filtered_sahis1", n_aile=len(CIFT), hal_olculdu=len(olc), n_esdeger_alan=esd, red_esdeger_fark=fark,
          hal_uyum=uyum)
    print(f"✓ {CIK}")
    return 0 if fark == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
