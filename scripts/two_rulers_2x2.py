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
import bare_imperative_cerceve as AC
import form_count as FS
from reading_style_object import payda

CIK = f"{ROOT}/results/2x2_2026-09-03.json"
NB = 1000
IX = {k: i for i, k in enumerate(MK.ALAN)}
C_CUM, C_ADZ = len(MK.ALAN), len(MK.ALAN) + 1


def _p(W, pay, bol, k=1.0):
    return k * W[:, pay].sum() / max(W[:, bol].sum(), 1)


OLCU = {
 "M1_1k":        (lambda W: _p(W, IX["m1_sahis2"], IX["n_jeton"], 1000), True),
 "M1_cumle":     (lambda W: _p(W, IX["m1_sahis2"], C_CUM), False),
 "KUVVET_cumle": (lambda W: _p(W, IX["kuvvet"], C_CUM), True),
 "KUVVET_1k":    (lambda W: _p(W, IX["kuvvet"], IX["n_jeton"], 1000), False),
 "SAHIS1_1k":    (lambda W: 1000 * (W[:, IX["m5_yakin"]].sum() - W[:, IX["m1_sahis2"]].sum())
                  / max(W[:, IX["n_jeton"]].sum(), 1), True),
 "SAHIS1_cumle": (lambda W: (W[:, IX["m5_yakin"]].sum() - W[:, IX["m1_sahis2"]].sum())
                  / max(W[:, C_CUM].sum(), 1), False),
 "CIPLAK_cumle": (lambda W: _p(W, C_ADZ, C_CUM), True),
 "CIPLAK_1k":    (lambda W: _p(W, C_ADZ, IX["n_jeton"], 1000), False),
}


def bilesen(nlp, R):
    V = MK.satir_bilesenleri(nlp, R)
    A, _ = AC.say(nlp, [r["metin"] for r in R], n_process=6)
    return np.hstack([V, A["a_duzyazi"].reshape(-1, 1)])


def olc(P0, P1, ist, fn):
    gz = fn(P1) - fn(P0); taban = fn(P0)
    ad = np.unique(ist); B = []
    rng = np.random.default_rng(20260903)
    for _ in range(NB):
        sel = rng.choice(ad, len(ad), replace=True)
        ix = np.concatenate([np.where(ist == u)[0] for u in sel])
        B.append(fn(P1[ix]) - fn(P0[ix]))
    b = np.array(B, float)
    ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    pl = M9.plasebo_esli(P0, ist, olc=fn)
    sd = pl["sd"] or float("nan")
    return dict(taban=float(taban), gozlenen=float(gz), ci=ci,
                ayrik=bool(ci[1] < 0 or ci[0] > 0), yon="asagi" if gz < 0 else "yukari",
                goreli=float(gz / taban) if taban else None,
                z=float((gz - pl["ort"]) / sd) if sd == sd and sd else None,
                plasebo_ort=pl["ort"], plasebo_sd=pl["sd"], plasebo_p95=pl["p95"],
                plasebo_ustu=bool(abs(gz) > pl["p95"]))


def cift_al(Rb, Ri):
    anb = {(r.get("kol"), r.get("cekim"), r.get("istem_i")): j for j, r in enumerate(Rb)}
    c = [(anb[k], j) for j, r in enumerate(Ri)
         if (k := (r.get("kol"), r.get("cekim"), r.get("istem_i"))) in anb]
    return [x for x, _ in c], [y for _, y in c]


def main():
    A = json.load(open(f"{ROOT}/results/PREREG9_A_OLCUM_2026-08-31.json", encoding="utf-8"))
    B = {k: v for k, v in A.items() if isinstance(v, dict) and v.get("sinif") == "BIRINCIL"}
    nlp = FS._boru(); t0 = time.time(); S = {"_aile": {}, "_merdiven": {}}
    for aile, v in sorted(B.items()):
        k0, k1 = v["kontrast"].split("→")
        Rb, Ri = MK.oku(aile, k0), MK.oku(aile, k1)
        if Rb is None or Ri is None:
            S["_aile"][aile] = dict(hal=""); continue
        i0, i1 = cift_al(Rb, Ri)
        P0 = bilesen(nlp, Rb)[i0]; P1 = bilesen(nlp, Ri)[i1]
        ist = np.array([Ri[y].get("istem_i") for y in i1])
        r = {ad: olc(P0, P1, ist, fn) for ad, (fn, _) in OLCU.items()}
        r["_uzunluk"] = dict(base=float(_p(P0, IX["n_jeton"], C_CUM)),
                             instruct=float(_p(P1, IX["n_jeton"], C_CUM)),
                             SINIF="POST-HOC BETIM · karar kurali OKUMAZ")
        r["n_cift"] = len(i0); S["_aile"][aile] = r
        print(f"  [{time.time()-t0:.0f}s] {aile:<18} M1/1k={r['M1_1k']['gozlenen']:+7.3f} "
              f"({100*r['M1_1k']['goreli']:+5.1f}%) · 1.kisi/1k={r['SAHIS1_1k']['gozlenen']:+7.3f} "
              f"({100*r['SAHIS1_1k']['goreli']:+6.1f}%) · ciplak/cml={r['CIPLAK_cumle']['gozlenen']:+.4f}",
              flush=True)
    for aile, (bas, rungs, son) in MK.MERDIVEN.items():
        if aile == "OLMo2-7B-RLVR":
            continue
        V, R = {}, {}
        for b in rungs:
            Rb = MK.oku(aile, b)
            if Rb is None:
                continue
            R[b] = Rb; V[b] = bilesen(nlp, Rb)
        if len(V) < len(rungs):
            S["_merdiven"][aile] = dict(hal=""); continue
        ad = {}
        for a1, a2 in zip(rungs[:-1], rungs[1:]):
            i0, i1 = cift_al(R[a1], R[a2])
            P0, P1 = V[a1][i0], V[a2][i1]
            ist = np.array([R[a2][y].get("istem_i") for y in i1])
            ad[f"{a1}→{a2}"] = dict(
                M1_cumle=olc(P0, P1, ist, OLCU["M1_cumle"][0]),
                M1_1k=olc(P0, P1, ist, OLCU["M1_1k"][0]),
                uzunluk=dict(bas=float(_p(P0, IX["n_jeton"], C_CUM)),
                             son=float(_p(P1, IX["n_jeton"], C_CUM))))
            print(f"  [{time.time()-t0:.0f}s] {aile}/{a1}→{a2}: "
                  f"ΔM1/cml={ad[f'{a1}→{a2}']['M1_cumle']['gozlenen']:+.4f}"
                  f"{'*' if ad[f'{a1}→{a2}']['M1_cumle']['ayrik'] else ' '}", flush=True)
        S["_merdiven"][aile] = ad
    AI = {k: v for k, v in S["_aile"].items() if "M1_1k" in v}
    hizli = [k for k, v in AI.items()
             if v["SAHIS1_1k"]["goreli"] is not None
             and v["SAHIS1_1k"]["goreli"] < v["M1_1k"]["goreli"]]
    n2 = len(hizli)
    dal2 = "DAL-1" if n2 == len(AI) else ("DAL-1-sayiyla" if n2 >= 12 else "DAL-2")
    tutan = []
    for aile, ad in S["_merdiven"].items():
        if not isinstance(ad, dict) or "hal" in ad:
            continue
        adim = {k: v["M1_cumle"] for k, v in ad.items()}
        terc = [k for k in adim if k.endswith("→dpo")]
        if not terc:
            continue
        t = terc[0]
        enb = max(adim, key=lambda k: abs(adim[k]["gozlenen"]))
        if enb == t and adim[t]["ayrik"]:
            tutan.append(aile)
    dal3 = "TAB2 KALIR" if len(tutan) >= 2 else "TAB2 SONUC CÜMLESI SAHIBE"
    S["_kunye"] = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        SINIF="",
        z_tanimi="",
        n_bootstrap=NB, motor="MK · M9.plasebo_esli(olc=) · AC.say · hepsi ITHAL",
        karar_2=dict(n_hizli=n2, n_aile=len(AI), dal=dal2, aileler=sorted(hizli)),
        karar_3=dict(tutan=sorted(tutan), dal=dal3))
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    payda("p11_2x2", n_aile=len(AI), n_merdiven=len([1 for v in S["_merdiven"].values()
                                                     if isinstance(v, dict) and "hal" not in v]),
          hal_sahis1_daha_hizli=n2, hal_merdiven_tutan=len(tutan))
    print(f"\n★ §2 · 1.kisi oransal düsüsü 2.kisininkinden BÜYÜK: {n2}/{len(AI)} "
          f"⇒ esik 16 ⇒ DAL-1 · 12-15 ⇒ DAL-1-sayiyla · <12 ⇒ DAL-2 ⇒ **{dal2}**")
    print(f"★ §3 · /cümle cetvelinde tercih adimi en büyük VE ayrik: {len(tutan)}/3 "
          f"({', '.join(tutan) if tutan else 'yok'}) ⇒ esik ≥2 ⇒ **{dal3}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
