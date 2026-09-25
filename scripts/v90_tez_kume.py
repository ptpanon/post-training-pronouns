#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
ARA = __DNH_DATA__ + "/v90_ara"
TK = __DNH_DATA__ + "/seed_denetimi"
CARD = f"{ROOT}/results/four_seed_level_2026-09-16.json"
CIK = f"{ROOT}/results/v90_tez_kume_2026-09-17.json"
RULE = "preregistration/rule_v90_readings_2026-09-17.md @ fc97c65d"
NB, RNG = 1000, 20260901


def tohumlar(dizin):
    return sorted(glob.glob(f"{TK}/{dizin}/t*/uretim.jsonl"))[:4]


def bilesen(aile):
    import addressee_run as MK, form_count as FS
    K = json.load(open(CARD, encoding="utf-8"))[aile]
    k0, k1 = list(K["zemin_dizin"].keys())
    yb, yi = tohumlar(K["zemin_dizin"][k0]), tohumlar(K["zemin_dizin"][k1])
    assert len(yb) == 4 and len(yi) == 4, (aile, len(yb), len(yi))
    nlp = FS._boru(); t0 = time.time()
    A0, A1, IST, DEB, esleme_red = [], [], [], [], 0
    for zb, zi in zip(yb, yi):
        Rb = [json.loads(l) for l in open(zb, encoding="utf-8")]
        Ri = [json.loads(l) for l in open(zi, encoding="utf-8")]
        for R in (Rb, Ri):
            i2d, d2i = {}, {}
            for r in R:
                i2d.setdefault(r["istem_i"], set()).add(r["debate"])
                d2i.setdefault(r["debate"], set()).add(r["istem_i"])
            esleme_red += sum(len(s) != 1 for s in i2d.values()) + sum(len(s) != 2 for s in d2i.values())
            esleme_red += int(len(d2i) != 17)
        Vb = MK.satir_bilesenleri(nlp, Rb); Vi = MK.satir_bilesenleri(nlp, Ri)
        anb = {(r.get("kol"), r.get("cekim"), r.get("istem_i")): j for j, r in enumerate(Rb)}
        cift = [(anb[k], j) for j, r in enumerate(Ri)
                if (k := (r.get("kol"), r.get("cekim"), r.get("istem_i"))) in anb]
        esleme_red += sum(Rb[x]["debate"] != Ri[y]["debate"] for x, y in cift)
        A0.append(Vb[[x for x, _ in cift]]); A1.append(Vi[[y for _, y in cift]])
        IST.append(np.array([Ri[y].get("istem_i") for _, y in cift]))
        DEB.append(np.array([Ri[y]["debate"] for _, y in cift]))
        print(f"  [{time.time() - t0:.0f}s] {aile} {os.path.basename(os.path.dirname(zi))}: cift {len(cift)}", flush=True)
    os.makedirs(ARA, exist_ok=True)
    np.savez(f"{ARA}/tez_{aile}.npz", P0=np.vstack(A0), P1=np.vstack(A1),
             ist_son=IST[-1], ist_gercek=np.concatenate(IST), deb=np.concatenate(DEB),
             esleme_red=esleme_red, dosyalar=np.array(yb + yi))
    print(f"  [PAYDA] v90_tez_bilesen_{aile}: n_seed=4 · n_satir={sum(len(a) for a in A0)} · red_esleme={esleme_red} "
          f"· {time.time() - t0:.0f} sn", flush=True)
    return 0 if esleme_red == 0 else 3


def boot(P0, P1, kume):
    import addressee_run as MK, depersonalization16_neutral_run as M9
    rng = np.random.default_rng(RNG); ad = np.unique(kume); ix_of = {u: np.where(kume == u)[0] for u in ad}
    bm, bs = [], []
    for _ in range(NB):
        sel = rng.choice(ad, len(ad), replace=True)
        ix = np.concatenate([ix_of[u] for u in sel])
        bm.append(MK.olc_toplam(P1[ix])["M1"] - MK.olc_toplam(P0[ix])["M1"])
        bs.append(M9.sahis1_1k(P1[ix]) - M9.sahis1_1k(P0[ix]))
    q = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return q(np.array(bm)), q(np.array(bs)), len(ad)


def sinif(ci):
    return "asagi" if ci[1] < 0 else ("yukari" if ci[0] > 0 else "null")


def oku():
    import addressee_run as MK, depersonalization16_neutral_run as M9
    K = json.load(open(CARD, encoding="utf-8"))
    aileler = sorted(a for a in K if not a.startswith("_"))
    S, red_kapi, red_esleme = {}, [], 0
    for a in aileler:
        z = np.load(f"{ARA}/tez_{a}.npz")
        P0, P1, deb = z["P0"], z["P1"], z["deb"]
        red_esleme += int(z["esleme_red"])
        ist_arac = np.tile(z["ist_son"], 4)
        sira_ozdes = bool(np.array_equal(ist_arac, z["ist_gercek"]))
        dM1 = MK.olc_toplam(P1)["M1"] - MK.olc_toplam(P0)["M1"]
        d1 = M9.sahis1_1k(P1) - M9.sahis1_1k(P0)
        ciM_i, ci1_i, n_i = boot(P0, P1, ist_arac)
        ciM_t, ci1_t, n_t = boot(P0, P1, deb)
        sap = max(abs(dM1 - K[a]["havuz_dM1"]), abs(ciM_i[0] - K[a]["ci"][0]), abs(ciM_i[1] - K[a]["ci"][1]))
        if sap > 1e-9 or not sira_ozdes:
            red_kapi.append(a)
        S[a] = dict(dM1=float(dM1), ci_istem=ciM_i, ci_tez=ciM_t, sinif_istem=sinif(ciM_i), sinif_tez=sinif(ciM_t),
                    genislik_orani=float((ciM_t[1] - ciM_t[0]) / (ciM_i[1] - ciM_i[0])),
                    d1st=float(d1), ci1_istem=ci1_i, ci1_tez=ci1_t, sinif1_istem=sinif(ci1_i), sinif1_tez=sinif(ci1_t),
                    genislik_orani_1st=float((ci1_t[1] - ci1_t[0]) / (ci1_i[1] - ci1_i[0])),
                    n_kume_istem=n_i, n_kume_tez=n_t, n_satir=int(len(P0)),
                    card_plasebo_ustu=K[a]["plasebo_ustu"], kapi_sapma=float(sap), kapi_sira_ozdes=sira_ozdes)
        print(f"  {a:16s} ΔM1 {dM1:+.2f} istem [{ciM_i[0]:+.2f},{ciM_i[1]:+.2f}] tez [{ciM_t[0]:+.2f},{ciM_t[1]:+.2f}] "
              f"· Δ1st {d1:+.2f} tez [{ci1_t[0]:+.2f},{ci1_t[1]:+.2f}] · kapi {sap:.1e}", flush=True)
    say = lambda alan: {s: sum(v[alan] == s for v in S.values()) for s in ("asagi", "yukari", "null")}
    hal = "ALET-KAYDI" if (red_kapi or red_esleme) else "ESDEGER"
    card = dict(sinif="BETIM · card · bar yok", rule=RULE, soru="(i) tez-kümeli CI n=17 · Tablo 1 dört-seed nesnesi",
                kaynak=dict(card="results/four_seed_level_2026-09-16.json"),
                alet=dict(hal=hal, red_kapi=red_kapi, red_esleme=red_esleme,
                          kapi=""),
                aile=S if hal == "ESDEGER" else {},
                sayim=dict(dM1_tez=say("sinif_tez"), dM1_istem=say("sinif_istem"),
                           d1st_tez=say("sinif1_tez"), d1st_istem=say("sinif1_istem")) if hal == "ESDEGER" else {},
                SERH=""
                     "",
                damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(card, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from reading_style_object import payda
    payda("v90_tez_kume", n_aile=len(aileler), hal_esdeger=int(hal == "ESDEGER"), red_kapi=len(red_kapi), red_esleme=red_esleme)
    print(f"✓ {CIK} · {hal} · sayim {card['sayim']}")
    return 0 if hal == "ESDEGER" else 3


if __name__ == "__main__":
    if sys.argv[1] == "bilesen":
        sys.exit(bilesen(sys.argv[2]))
    sys.exit(oku())
