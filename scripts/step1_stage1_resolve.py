#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, json, os, sys, time
from collections import Counter
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from verdict_name_crosscount import capraz_say
import prefix_judge_calibration as KAL
import a2_kutupsallik as A2

OUT = __DNH_DATA__ + "/adim1_asama1"
CIKTI = f"{ROOT}/results/ADIM1_ASAMA1_2026-08-13.json"
CEKIS_CIKTI = f"{ROOT}/results/ADIM1_CEKIS_2026-08-13.json"
E5 = "intfloat/e5-large-v2"
SEED, B_BOOT, K_NULL = 20260813, 2000, 200

IZGARA = {("DOM+", "ARO+"): "DOM+x2_ARO+x2", ("DOM-", "ARO+"): "DOM-x2_ARO+x2",
          ("DOM+", "ARO-"): "DOM+x2_ARO-x2", ("DOM-", "ARO-"): "DOM-x2_ARO-x2"}
CEKIS = {"ARO+": "ARO+x1", "ARO-": "ARO-x1", "DOM+": "DOM+x1", "DOM-": "DOM-x1"}
ADLAR1 = ("A-AGIR", "D-AGIR", "DENGELI", "TEK-EKSEN-SÖNÜK", "ÖLCÜLEMEZ")
ADLAR2 = ("CEKIS-ÖNGÖRÜR", "CEKIS-ÖNGÖRMEZ", "CEKIS-ÖLCÜLEMEZ")


def verdict_uzay1(kapi_tamam, D_ci, g_agir, g_hafif, null_p95, agir_eksen):
    if (not kapi_tamam) or D_ci is None or not all(np.isfinite(v) for v in D_ci):
        return "ÖLCÜLEMEZ"
    if D_ci[0] > 0:
        return "A-AGIR" if agir_eksen == "ARO" else "D-AGIR"
    if D_ci[1] < 0:
        return "D-AGIR" if agir_eksen == "ARO" else "A-AGIR"
    if min(g_agir, g_hafif) > null_p95:
        return "DENGELI"
    return "TEK-EKSEN-SÖNÜK"


def verdict_uzay2(h1, agir_eksen, galip_eksen):
    if h1 == "ÖLCÜLEMEZ" or agir_eksen is None:
        return "CEKIS-ÖLCÜLEMEZ"
    if h1 in ("DENGELI", "TEK-EKSEN-SÖNÜK"):
        return "CEKIS-ÖLCÜLEMEZ"
    return "CEKIS-ÖNGÖRÜR" if galip_eksen == agir_eksen else "CEKIS-ÖNGÖRMEZ"


def prova():
    V1 = [((False, (0.1, 0.2), 0.3, 0.1, 0.05, "ARO"), "ÖLCÜLEMEZ"),
          ((True, (0.05, 0.20), 0.3, 0.1, 0.05, "ARO"), "A-AGIR"),
          ((True, (0.05, 0.20), 0.3, 0.1, 0.05, "DOM"), "D-AGIR"),
          ((True, (-0.20, -0.05), 0.1, 0.3, 0.05, "ARO"), "D-AGIR"),
          ((True, (-0.05, 0.05), 0.3, 0.2, 0.05, "ARO"), "DENGELI"),
          ((True, (-0.05, 0.05), 0.3, 0.02, 0.05, "ARO"), "TEK-EKSEN-SÖNÜK")]
    for g, b in V1:
        assert verdict_uzay1(*g) == b, (g, verdict_uzay1(*g), b)
    V2 = [(("ÖLCÜLEMEZ", "ARO", "ARO"), "CEKIS-ÖLCÜLEMEZ"),
          (("A-AGIR", None, "ARO"), "CEKIS-ÖLCÜLEMEZ"),
          (("DENGELI", "ARO", "ARO"), "CEKIS-ÖLCÜLEMEZ"),
          (("A-AGIR", "ARO", "ARO"), "CEKIS-ÖNGÖRÜR"),
          (("D-AGIR", "ARO", "DOM"), "CEKIS-ÖNGÖRMEZ")]
    for g, b in V2:
        assert verdict_uzay2(*g) == b, (g, verdict_uzay2(*g), b)
    payda("a1_prova", n_vaka1=len(V1), n_vaka2=len(V2),
          n_ad1=len(ADLAR1), n_ad2=len(ADLAR2),
          bekle={"n_vaka1": 6, "n_vaka2": 5, "n_ad1": 5, "n_ad2": 3})
    capraz_say(__file__, "verdict_uzay1", ADLAR1)
    capraz_say(__file__, "verdict_uzay2", ADLAR2)


def _merkez_fark(X, a, b):
    return float(np.linalg.norm(X[a].mean(0) - X[b].mean(0)))


def prova_ilkel():
    X = np.zeros((4, 3)); X[0, 0] = X[1, 0] = 1.0; X[2, 1] = X[3, 1] = 1.0
    a, b = np.array([0, 1]), np.array([2, 3])
    assert abs(_merkez_fark(X, a, a) - 0.0) < 1e-12
    assert abs(_merkez_fark(X, a, b) - np.sqrt(2)) < 1e-12
    payda("a1_ilkel_prova", n_vaka=2, red_sapan=0, bekle={"n_vaka": 2})


def yukle():
    R = [json.loads(l) for l in open(f"{OUT}/uretim.jsonl", encoding="utf-8")]
    payda("a1_yukle", n_satir=len(R), n_kol=len(set(r["kol"] for r in R)),
          n_istem=len(set(r["istem_i"] for r in R)),
          red_bos_metin=sum(1 for r in R if not r["metin"].strip()),
          bekle={"n_satir": 816, "n_kol": 8, "n_istem": 34})
    return R


def gom(R, dev):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(E5, device=dev)
    X = m.encode([f"query: {r['metin']}" for r in R], batch_size=64,
                 normalize_embeddings=True, show_progress_bar=False)
    X = np.asarray(X, dtype=np.float64)
    np.save(f"{OUT}/X_e5.npy", X.astype(np.float32))
    payda("a1_gomme", n_satir=len(X), n_boyut=X.shape[1],
          red_norm_sapan=int(abs(np.linalg.norm(X, axis=1) - 1).max() > 1e-4),
          bekle={"n_satir": 816, "n_boyut": 1024})
    return X


def main(a):
    t0 = time.time()
    prova(); prova_ilkel()
    R = yukle()
    C = None
    if not os.path.exists(f"{OUT}/X_e5.npy") or a.yeniden_gom:
        from gpu_lock import kilitle
        C = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="a1_coz")
        X = gom(R, a.dev)
    else:
        X = np.load(f"{OUT}/X_e5.npy").astype(np.float64)
        payda("a1_gomme_onbellek", n_satir=len(X), n_boyut=X.shape[1],
              bekle={"n_satir": 816})
    kol = np.array([r["kol"] for r in R])
    ist = np.array([int(r["istem_i"]) for r in R])
    idx = {k: np.flatnonzero(kol == k) for k in set(kol.tolist())}

    cekis = {e: _merkez_fark(X, idx[CEKIS[f"{e}+"]], idx[CEKIS[f"{e}-"]])
             for e in ("ARO", "DOM")}
    kl = sorted(set(ist.tolist()))
    rb = np.random.default_rng(SEED)

    def _boot(fn):
        B = []
        for _ in range(B_BOOT):
            sec = rb.choice(len(kl), len(kl), replace=True)
            mask = np.concatenate([np.flatnonzero(ist == kl[s]) for s in sec])
            B.append(fn(mask))
        return np.array(B)

    def _cekis_of(e):
        return lambda mask: _merkez_fark(
            X, mask[np.isin(mask, idx[CEKIS[f"{e}+"]])],
            mask[np.isin(mask, idx[CEKIS[f"{e}-"]])])

    cekis_ci = {e: [float(np.percentile(_boot(_cekis_of(e)), p)) for p in (2.5, 97.5)]
                for e in ("ARO", "DOM")}
    cekis_sd = {e: float(_boot(_cekis_of(e)).std()) for e in ("ARO", "DOM")}
    dfark = abs(cekis["ARO"] - cekis["DOM"])
    esik_sd = max(cekis_sd.values())
    agir = None if dfark < esik_sd else ("ARO" if cekis["ARO"] > cekis["DOM"] else "DOM")
    hafif = None if agir is None else ("DOM" if agir == "ARO" else "ARO")

    ck = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              prereg="results/PREREG_ADIM1_ASAMA1_2026-08-13.md (+ERRATA-1,-2)",
              SINIF="CEKIS-KALIBRASYONU — verdict ÜRETMEZ; AGIR atamasini verir",
              alet="e5-large-v2 1024d, L2-norm (ERRATA-1 kapsami)",
              cekis={k: round(v, 6) for k, v in cekis.items()},
              cekis_CI={k: [round(x, 6) for x in v] for k, v in cekis_ci.items()},
              cekis_sd={k: round(v, 6) for k, v in cekis_sd.items()},
              delta_cekis=round(dfark, 6), esik_beraberlik_sd=round(esik_sd, 6),
              AGIR=agir, HAFIF=hafif,
              beraberlik=bool(agir is None), n_kume=len(kl), B=B_BOOT,
              betik_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
    json.dump(ck, open(CEKIS_CIKTI, "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(ck, ensure_ascii=False, indent=1))
    if a.asama == "cekis":
        print(f"\n★ AGIR = {agir} · HAFIF = {hafif} — **izgara SAYILMADI.**", flush=True)
        return 0

    AP = np.concatenate([idx[IZGARA[(d, "ARO+")]] for d in ("DOM+", "DOM-")])
    AN = np.concatenate([idx[IZGARA[(d, "ARO-")]] for d in ("DOM+", "DOM-")])
    DP = np.concatenate([idx[IZGARA[("DOM+", r)]] for r in ("ARO+", "ARO-")])
    DN = np.concatenate([idx[IZGARA[("DOM-", r)]] for r in ("ARO+", "ARO-")])
    g = {"ARO": _merkez_fark(X, AP, AN), "DOM": _merkez_fark(X, DP, DN)}
    payda("a1_izgara", n_ARO_arti=len(AP), n_ARO_eksi=len(AN),
          n_DOM_arti=len(DP), n_DOM_eksi=len(DN),
          bekle={"n_ARO_arti": 204, "n_ARO_eksi": 204,
                 "n_DOM_arti": 204, "n_DOM_eksi": 204})

    def _g_of(e):
        P, N = (AP, AN) if e == "ARO" else (DP, DN)
        return lambda mask: _merkez_fark(X, mask[np.isin(mask, P)],
                                         mask[np.isin(mask, N)])

    Bg = {e: _boot(_g_of(e)) for e in ("ARO", "DOM")}
    BD = Bg[agir] - Bg[hafif] if agir else None
    D_ci = ([float(np.percentile(BD, 2.5)), float(np.percentile(BD, 97.5))]
            if BD is not None else None)

    IZG = np.concatenate([AP, AN])
    null = []
    for _ in range(K_NULL):
        p = rb.permutation(len(IZG))
        null.append(_merkez_fark(X, IZG[p[:len(AP)]], IZG[p[len(AP):]]))
    null = np.array(null)
    null_p95 = float(np.percentile(null, 95))

    k0e = all(cekis_ci[e][0] > null_p95 for e in ("ARO", "DOM"))
    g_tavan = max(list(g.values()) + list(cekis.values()))
    h1 = verdict_uzay1(k0e, D_ci, g.get(agir, float("nan")),
                     g.get(hafif, float("nan")), null_p95, agir)
    galip = (agir if h1 == ("A-AGIR" if agir == "ARO" else "D-AGIR")
             else (hafif if h1 in ("A-AGIR", "D-AGIR") else None))
    h2 = verdict_uzay2(h1, agir, galip)

    jeton = Counter()
    sec = [dict(id=f"{i}", metin=R[i]["metin"]) for i in np.concatenate([AP, AN])]
    cik, redj = KAL.yargila_esz(KAL.RUBRIK["ARO"], sec, A2.ES, jeton=jeton)
    ar = {int(k): v for k, v in cik.items()}
    o_p = np.mean([ar.get(int(i)) == "AROUSED" for i in AP if int(i) in ar])
    o_n = np.mean([ar.get(int(i)) == "AROUSED" for i in AN if int(i) in ar])
    payda("a1_cetvel", n_istenen=len(sec), n_donen=len(ar),
          **{f"red_{k}": v for k, v in redj.items()})

    ort = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="results/PREREG_ADIM1_ASAMA1_2026-08-13.md (+ERRATA-1,-2)",
               VERDICT_uzay1=h1, VERDICT_uzay2=h2,
               AGIR=agir, HAFIF=hafif, cekis={k: round(v, 6) for k, v in cekis.items()},
               g={k: round(v, 6) for k, v in g.items()},
               D=round(float(g[agir] - g[hafif]), 6) if agir else None,
               D_CI=[round(x, 6) for x in D_ci] if D_ci else None,
               D_sd=round(float(BD.std()), 6) if BD is not None else None,
               MDE_1645sd=round(1.645 * float(BD.std()), 6) if BD is not None else None,
               g_CI={e: [round(float(np.percentile(Bg[e], p)), 6) for p in (2.5, 97.5)]
                     for e in ("ARO", "DOM")},
               null_merkez=round(float(null.mean()), 6), null_sd=round(float(null.std()), 6),
               null_p95=round(null_p95, 6), K_null=K_NULL,
               k0e_saf_kutup_gecti=bool(k0e),
               g_tavan=round(g_tavan, 6),
               tavana_yapisik=bool(agir and abs(g[agir] - g[hafif]) > 0.9 * g_tavan),
               cetvel_dogrulama=dict(
                   alet="yerel Qwen2.5-32B · RUBRIK['ARO'] · hükme GIRMEZ",
                   AROUSED_orani_ARO_arti=round(float(o_p), 4),
                   AROUSED_orani_ARO_eksi=round(float(o_n), 4),
                   fark=round(float(o_p - o_n), 4),
                   isaret_uyumu=bool((o_p - o_n) > 0),
                   n_donen=len(ar), jeton=dict(jeton)),
               n_kume=len(kl), B=B_BOOT, saniye=round(time.time() - t0, 1),
               cevre=C, betik_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
    json.dump(ort, open(CIKTI, "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in ort.items() if k != "cevre"},
                     ensure_ascii=False, indent=1))
    print(f"\n★ UZAY-1 = {h1} · UZAY-2 = {h2}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", default="verdict", choices=("prova", "cekis", "verdict"))
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=1)
    ap.add_argument("--yeniden-gom", action="store_true")
    a = ap.parse_args()
    if a.asama == "prova":
        prova(); prova_ilkel()
        print("")
        sys.exit(0)
    sys.exit(main(a))
