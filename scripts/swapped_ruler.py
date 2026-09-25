#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import depersonalization16_neutral_run as M9
import form_count as FS
from degeneracy_threshold_floor import farkli4
from reading_style_object import payda

A_CARD = f"{ROOT}/results/PREREG9_A_OLCUM_2026-08-31.json"
CIK = f"{ROOT}/results/swapped_ruler_2026-09-03.json"
D4_BAR = 0.60
NB = 1000
ALAN = list(MK.ALAN)
IX = {k: i for i, k in enumerate(ALAN)}


def m1_1k(W):
    return 1000 * W[:, IX["m1_sahis2"]].sum() / max(W[:, IX["n_jeton"]].sum(), 1)


def m1_cumle(W):
    return W[:, IX["m1_sahis2"]].sum() / max(W[:, -1].sum(), 1)


def kuvvet_cumle(W):
    return W[:, IX["kuvvet"]].sum() / max(W[:, -1].sum(), 1)


def kuvvet_1k(W):
    return 1000 * W[:, IX["kuvvet"]].sum() / max(W[:, IX["n_jeton"]].sum(), 1)


def oku_ci(P0, P1, ist, olc, rng):
    gz = olc(P1) - olc(P0)
    ad = np.unique(ist); B = []
    for _ in range(NB):
        sel = rng.choice(ad, len(ad), replace=True)
        ix = np.concatenate([np.where(ist == u)[0] for u in sel])
        B.append(olc(P1[ix]) - olc(P0[ix]))
    b = np.array(B, float)
    ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    pl = M9.plasebo_esli(P0, ist, olc=olc)
    return dict(gozlenen=float(gz), ci=ci, ayrik=bool(ci[1] < 0 or ci[0] > 0),
                yon="asagi" if gz < 0 else "yukari",
                plasebo_p95=pl["p95"], plasebo_ustu=bool(abs(gz) > pl["p95"]))


def main():
    A = json.load(open(A_CARD, encoding="utf-8"))
    B = {k: v for k, v in A.items() if isinstance(v, dict) and v.get("sinif") == "BIRINCIL"}
    nlp = FS._boru(); t0 = time.time(); S = {}
    for aile, v in sorted(B.items()):
        k0, k1 = v["kontrast"].split("→")
        Rb, Ri = MK.oku(aile, k0), MK.oku(aile, k1)
        if Rb is None or Ri is None:
            S[aile] = dict(hal=""); continue
        Vb = MK.satir_bilesenleri(nlp, Rb); Vi = MK.satir_bilesenleri(nlp, Ri)
        anb = {(r.get("kol"), r.get("cekim"), r.get("istem_i")): j for j, r in enumerate(Rb)}
        cift = [(anb[k], j) for j, r in enumerate(Ri)
                if (k := (r.get("kol"), r.get("cekim"), r.get("istem_i"))) in anb]
        i0 = [x for x, _ in cift]; i1 = [y for _, y in cift]
        P0, P1 = Vb[i0], Vi[i1]
        ist = np.array([Ri[y].get("istem_i") for y in i1])
        rng = np.random.default_rng(20260903)
        r = dict(n_cift=len(cift),
                 asil_M1_1k=oku_ci(P0, P1, ist, m1_1k, np.random.default_rng(20260903)),
                 asil_KUVVET_cumle=oku_ci(P0, P1, ist, kuvvet_cumle, np.random.default_rng(20260903)),
                 ters_M1_cumle=oku_ci(P0, P1, ist, m1_cumle, np.random.default_rng(20260903)),
                 ters_KUVVET_1k=oku_ci(P0, P1, ist, kuvvet_1k, np.random.default_rng(20260903)))
        d4 = np.array([(farkli4(Rb[x].get("metin") or "") or np.nan) for x in i0], float)
        tut = ~np.isnan(d4) & (d4 >= D4_BAR)
        r["dejenere"] = dict(n_cift=len(cift), n_olculebilir=int((~np.isnan(d4)).sum()),
                             n_dejenere=int(((~np.isnan(d4)) & (d4 < D4_BAR)).sum()),
                             n_kisa_olculemez=int(np.isnan(d4).sum()),
                             oran=float(((~np.isnan(d4)) & (d4 < D4_BAR)).sum() /
                                        max((~np.isnan(d4)).sum(), 1)))
        if tut.sum() >= 20 and len(np.unique(ist[tut])) >= 5:
            r["filtreli_M1_1k"] = oku_ci(P0[tut], P1[tut], ist[tut], m1_1k,
                                         np.random.default_rng(20260903))
            r["filtreli_M1_1k"]["n_cift"] = int(tut.sum())
        else:
            r["filtreli_M1_1k"] = dict(hal="ÖLCÜLEMEZ — filtre sonrasi n<20 ya da küme<5",
                                       n_cift=int(tut.sum()))
        r["_tanim_uzunluk"] = dict(
            jeton_cumle_base=float(P0[:, IX["n_jeton"]].sum() / max(P0[:, -1].sum(), 1)),
            jeton_cumle_instruct=float(P1[:, IX["n_jeton"]].sum() / max(P1[:, -1].sum(), 1)),
            SINIF="")
        S[aile] = r
        print(f"  [{time.time()-t0:.0f}s] {aile:<18} n={len(cift):<4} "
              f"ΔM1_1k={r['asil_M1_1k']['gozlenen']:+7.3f}{'*' if r['asil_M1_1k']['ayrik'] else ' '} "
              f"ΔM1_cml={r['ters_M1_cumle']['gozlenen']:+7.4f}{'*' if r['ters_M1_cumle']['ayrik'] else ' '} "
              f"ΔKUV_cml={r['asil_KUVVET_cumle']['gozlenen']:+7.4f}{'*' if r['asil_KUVVET_cumle']['ayrik'] else ' '} "
              f"ΔKUV_1k={r['ters_KUVVET_1k']['gozlenen']:+7.3f}{'*' if r['ters_KUVVET_1k']['ayrik'] else ' '} "
              f"dej={r['dejenere']['oran']:.3f}", flush=True)
    ok = [k for k, v in S.items() if "asil_M1_1k" in v]
    reg_ayrik_asagi = sum(1 for k in ok if S[k]["ters_M1_cumle"]["ayrik"] and S[k]["ters_M1_cumle"]["yon"] == "asagi")
    kuv_ayrik = sum(1 for k in ok if S[k]["ters_KUVVET_1k"]["ayrik"])
    a_gecti = bool(reg_ayrik_asagi > len(ok) / 2 and kuv_ayrik <= len(ok) / 2)
    fil = [k for k in ok if "gozlenen" in S[k]["filtreli_M1_1k"]]
    fil_ayrik = [k for k in fil if S[k]["filtreli_M1_1k"]["ayrik"]]
    kiran = sorted(set(fil) - set(fil_ayrik))
    b_gecti = bool(len(kiran) == 0 and len(fil) == len(ok))
    S["_kunye"] = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        SINIF="",
        motor="muhatap_kos (edit yok) · depersonalization16_neutral_run.plasebo_esli(olc=…) K-5a esdegerlik ÖZDES · degeneracy_threshold_floor.farkli4 DONMUS",
        d4_bar=D4_BAR, n_bootstrap=NB,
        karar_a=dict(gecti=a_gecti, reg_ayrik_asagi=reg_ayrik_asagi, kuv_ayrik=kuv_ayrik, n=len(ok)),
        karar_b=dict(gecti=b_gecti, ayrik=len(fil_ayrik), olculen=len(fil), kiran=kiran))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    payda("p11_ters_cetvel", n_aile=len(ok), hal_a_gecti=int(a_gecti), hal_b_gecti=int(b_gecti),
          hal_reg_ayrik_asagi=reg_ayrik_asagi, hal_kuv_ayrik=kuv_ayrik,
          n_filtreli_olculen=len(fil), red_filtreli_kiran=len(kiran))
    print(f"\n★ (a) TERS CETVEL: register CI-ayrik asagi {reg_ayrik_asagi}/{len(ok)} · "
          f"kuvvet CI-ayrik {kuv_ayrik}/{len(ok)} ⇒ esik: reg>{len(ok)//2} VE kuv≤{len(ok)//2} "
          f"⇒ **{'GECTI' if a_gecti else ''}**")
    print(f"★ (b) FILTRELI M1: ayrik {len(fil_ayrik)}/{len(fil)} ölcülen "
          f"⇒ esik 16/16 ⇒ **{'GECTI' if b_gecti else 'GECMEDI'}**"
          + (f" · KIRAN: {', '.join(kiran)}" if kiran else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
