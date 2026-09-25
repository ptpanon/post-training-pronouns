#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import glob
import time
import argparse
import itertools
import collections
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import prefix_steering_generation as SU
import prefix_kararlilik as KR
import prefix_okunmazlik as OK
import prefix_yazar_bataryasi as YB

HAVUZ = YB.OUT
CIKTI = f"{ROOT}/unreleased/YAZAR_GEOMETRI_2026-08-08.json"
OKURLAR = (("e5", "intfloat/e5-large-v2", "query: "), ("bge", "BAAI/bge-m3", ""))
EKSENLER = (("ton", "ton_hedef", "sert", "sakin"), ("taraf", "durus", "arti", "eksi"))
K_NULL, SEED = 200, 20260808
ADLAR = ("ÖLCÜLEMEZ", "AILE-ICI-PARALEL", "SACIK")


def verdict_yazar(delta, null_p95, null_sd, boy_delta, iki_encoder_gecti, kapi_tamam):
    if (not kapi_tamam) or not all(np.isfinite([delta, null_p95, null_sd, boy_delta])):
        return "ÖLCÜLEMEZ"
    if (iki_encoder_gecti and delta > null_p95 and delta > boy_delta
            and delta >= 1.645 * null_sd):
        return "AILE-ICI-PARALEL"
    return "SACIK"


def _prova():
    n = float("nan")
    kes = [("kapi düsük", verdict_yazar(.3, .1, .05, .1, True, False)),
           ("NaN", verdict_yazar(n, .1, .05, .1, True, True)),
           ("tek encoder", verdict_yazar(.3, .1, .05, .1, False, True)),
           ("null asamadi", verdict_yazar(.05, .1, .02, .01, True, True)),
           ("boy plasebosu büyük", verdict_yazar(.3, .1, .05, .4, True, True)),
           ("MDE alti", verdict_yazar(.12, .1, .10, .05, True, True)),
           ("paralel", verdict_yazar(.3, .1, .05, .1, True, True))]
    for a_, h in kes:
        print(f"  prova {a_:20s} → {h}")
    assert len(set(h for _, h in kes)) == 3
    import ast
    ag = ast.parse(open(__file__, encoding="utf-8").read())
    kodda = {x.value.value for d in ast.walk(ag)
             if isinstance(d, ast.FunctionDef) and d.name == "verdict_yazar"
             for x in ast.walk(d)
             if isinstance(x, ast.Return) and isinstance(x.value, ast.Constant)}
    print(f"  [D44] kodda {len(kodda)} · mühürde {len(ADLAR)} · "
          f"fark {sorted(kodda ^ set(ADLAR)) or '—'}")
    if kodda != set(ADLAR):
        raise SystemExit("★ D44 CAPRAZ SAYIM DÜSTÜ — kosu BASLAMAZ")
    print("  ✓ prova gecti — üc dal, yedi sinir girdisi, D44 3=3")


def delta_ist(D, aile, yazarlar, soy=None):
    ic, dis = [], []
    for i, j in itertools.combinations(range(len(yazarlar)), 2):
        if soy is not None and soy[i] == soy[j]:
            continue
        c = float(np.dot(D[i], D[j]))
        (ic if aile[i] == aile[j] else dis).append(c)
    if not ic or not dis:
        return float("nan"), 0, 0, float("nan"), float("nan")
    return (float(np.median(ic) - np.median(dis)), len(ic), len(dis),
            float(np.median(ic)), float(np.median(dis)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--k-null", type=int, default=K_NULL)
    ap.add_argument("--prova", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    _prova()
    if a.prova:
        print("★ PROVA — verdict mekanigi sinandi; veri OKUNMADI.")
        return

    S, yazarlar = {}, []
    for kol in YB.YAZARLAR:
        y = f"{HAVUZ}/uretim_{kol}.jsonl"
        m = f"{HAVUZ}/manifest_{kol}.json"
        if not (os.path.exists(y) and os.path.exists(m)
                and json.load(open(m)).get("tamam")):
            print(f"{kol}")
            continue
        S[kol] = [json.loads(l) for l in open(y, encoding="utf-8")]
        yazarlar.append(kol)
    aile = [YB.YAZARLAR[k]["aile"] for k in yazarlar]
    boy = np.array([YB.YAZARLAR[k]["boy"] for k in yazarlar])
    sayim = collections.Counter(aile)
    kapi_tamam = len(yazarlar) >= 4 and all(v >= 2 for v in sayim.values()) \
        and len(sayim) >= 2

    karne = {}
    for k in yazarlar:
        met = [r["metin"] for r in S[k]]
        dolu = float(np.mean([len(m.strip()) > 20 for m in met]))
        okz = np.array([OK.okunmazlik(m) for m in met])
        tkr = np.array([OK.tekrar_orani(m) for m in met])
        kel = [m.split() for m in met]
        ttr = float(np.mean([len(set(w)) / max(len(w), 1) for w in kel]))
        karne[k] = dict(aile=YB.YAZARLAR[k]["aile"], boy=YB.YAZARLAR[k]["boy"],
                        n=len(met), dolu_oran=round(dolu, 4),
                        okunmazlik_ort=round(float(okz.mean()), 4),
                        frac_okunmaz=round(float((okz > YB.BAR_OKUNMAZ).mean()), 4),
                        tekrar_ort=round(float(tkr.mean()), 4),
                        boy_kelime=round(float(np.mean([len(w) for w in kel])), 1),
                        ttr=round(ttr, 4),
                        kapi_dolu=bool(dolu >= 0.95),
                        kirli=bool((okz > YB.BAR_OKUNMAZ).mean() >= YB.BAR_ORAN))
    dusen = [k for k in yazarlar if not karne[k]["kapi_dolu"] or karne[k]["kirli"]]
    for k in dusen:
        print(f"  ★ {k} KAPIDAN DÜSTÜ (dolu {karne[k]['dolu_oran']:.3f} · "
              f"kirli {karne[k]['kirli']}) — adiyla yazildi")
    kalan = [k for k in yazarlar if k not in dusen]
    sayim2 = collections.Counter(YB.YAZARLAR[k]["aile"] for k in kalan)
    kapi_tamam = kapi_tamam and all(v >= 2 for v in sayim2.values()) and len(sayim2) >= 2
    yazarlar, aile = kalan, [YB.YAZARLAR[k]["aile"] for k in kalan]
    boy = np.array([YB.YAZARLAR[k]["boy"] for k in kalan])
    payda("yazar_coz_havuz", n_yazar=len(yazarlar), n_aile=len(sayim2),
          n_satir=sum(len(S[k]) for k in yazarlar), red_dusen=len(dusen))
    print(f"{len(yazarlar)} {len(sayim2)}"
          f"{sum(len(S[k]) for k in yazarlar)} {kapi_tamam}")

    sys.path.insert(0, f"{ROOT}/scripts")
    from bs141_frontier_baselines import plain_enc
    tum_met, sinir = [], {}
    for k in yazarlar:
        sinir[k] = (len(tum_met), len(tum_met) + len(S[k]))
        tum_met += [r["metin"] for r in S[k]]
    R = dict(damga=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             prereg="unreleased/PREREG_YAZAR_BATARYASI_2026-08-08.md",
             yazarlar=yazarlar, aile=aile, karne=karne, dusen=dusen,
             kapi_tamam=bool(kapi_tamam), K_null=a.k_null, eksen={})
    rng = np.random.default_rng(SEED)
    gomu = {}
    for ad, model, onek in OKURLAR:
        print(f"  [{ad}] kodlaniyor ({len(tum_met)} metin)…", flush=True)
        gomu[ad] = plain_enc(model, tum_met, prefix=onek)
        print(f"  [{ad}] {gomu[ad].shape} · {time.time()-t0:.0f}s", flush=True)

    for eks, alan, poz, neg in EKSENLER:
        R["eksen"][eks] = {}
        for ad, _, _ in OKURLAR:
            E = gomu[ad] - gomu[ad].mean(0)
            D, ic_kar = [], {}
            for k in yazarlar:
                b, s = sinir[k]
                lab = np.array([r.get(alan) for r in S[k]], dtype=object)
                m1, m0 = lab == poz, lab == neg
                if m1.sum() < 5 or m0.sum() < 5:
                    D.append(np.full(E.shape[1], np.nan))
                    continue
                D.append(SU.birim(E[b:s][m1].mean(0) - E[b:s][m0].mean(0)))
                kume = np.array([r["debate"] for r in S[k]], dtype=object)
                ic_kar[k] = KR.kume_tutulan_bolme(E[b:s], m1.astype(int), kume, n_bol=20)
            D = np.stack(D)
            gec = np.array([np.isfinite(d).all() for d in D])
            d_gec, aile_gec = D[gec], [aile[i] for i in range(len(aile)) if gec[i]]
            boy_gec, yaz_gec = boy[gec], [yazarlar[i] for i in range(len(yazarlar)) if gec[i]]
            delta, n_ic, n_dis, med_ic, med_dis = delta_ist(d_gec, aile_gec, yaz_gec)
            null = []
            for _ in range(a.k_null):
                p = rng.permutation(len(aile_gec))
                null.append(delta_ist(d_gec, [aile_gec[i] for i in p], yaz_gec)[0])
            null = np.array([x for x in null if np.isfinite(x)])
            tert = np.digitize(boy_gec, np.percentile(boy_gec, [33.3, 66.7]))
            boy_delta = delta_ist(d_gec, [f"t{t}" for t in tert], yaz_gec)[0]
            R["eksen"][eks][ad] = dict(
                delta=round(delta, 4), n_ic_cift=n_ic, n_dis_cift=n_dis,
                medyan_ic=round(med_ic, 4), medyan_dis=round(med_dis, 4),
                null_merkez=round(float(null.mean()), 4),
                null_sd=round(float(null.std(ddof=1)), 4),
                null_p95=round(float(np.percentile(null, 95)), 4),
                MDE=round(1.645 * float(null.std(ddof=1)), 4),
                boy_plasebo_delta=round(float(boy_delta), 4),
                tavan_kararlilik={k: round(v["medyan"], 4) for k, v in ic_kar.items()},
                n_yazar=int(gec.sum()), red_atlanan=int((~gec).sum()))
            print(f"  {eks:6s}/{ad:4s} Δ {delta:+.4f} | null {null.mean():+.4f}±"
                  f"{null.std(ddof=1):.4f} p95 {np.percentile(null,95):+.4f} | "
                  f"boy-plasebo {boy_delta:+.4f} | icmed {med_ic:.3f} dismed {med_dis:.3f}")

    tt = R["eksen"]["ton"]
    gecti = all(tt[ad]["delta"] > tt[ad]["null_p95"] for ad, _, _ in OKURLAR)
    e5 = tt["e5"]
    H = verdict_yazar(e5["delta"], e5["null_p95"], e5["null_sd"],
                    e5["boy_plasebo_delta"], gecti, kapi_tamam)
    R["iki_encoder_gecti"] = bool(gecti)
    R["verdict"] = H
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    payda("yazar_karar", n_yazar=len(yazarlar), n_eksen=len(EKSENLER),
          n_okur=len(OKURLAR), n_null=a.k_null, n_ad=len(ADLAR),
          bekle={"n_okur": 2, "n_ad": 3})
    print(f"\n  ★★ K-YAZAR HÜKMÜ (ön-kayitli rule): **{H}**")
    print(f"→ {CIKTI} · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
