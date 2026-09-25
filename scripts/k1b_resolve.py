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
from step1_stage1_resolve import _merkez_fark, prova_ilkel

OUT = __DNH_DATA__ + "/k1b_rakip_uzunluk"
CIKTI = f"{ROOT}/results/K1B_2026-08-13.json"
E5 = "intfloat/e5-large-v2"
SEED, B_BOOT, K_NULL = 20260813, 2000, 2000
ADLAR = ("UZUNLUK-ACIKLAR", "RAKIP-GERCEK", "KARMA", "AYRISTIRILAMAZ", "KAPI-ÖLCÜLEMEZ")


def verdict_k1b(kapi_tamam, U_ci, R_ci):
    if not kapi_tamam:
        return "KAPI-ÖLCÜLEMEZ"
    if U_ci is None or R_ci is None or not all(np.isfinite(list(U_ci) + list(R_ci))):
        return "KAPI-ÖLCÜLEMEZ"
    u = (U_ci[0] > 0) or (U_ci[1] < 0)
    r = (R_ci[0] > 0) or (R_ci[1] < 0)
    if u and r:
        return "KARMA"
    if u:
        return "UZUNLUK-ACIKLAR"
    if r:
        return "RAKIP-GERCEK"
    return "AYRISTIRILAMAZ"


def prova():
    V = [((False, (0.1, 0.3), (-0.1, 0.1)), "KAPI-ÖLCÜLEMEZ"),
         ((True, (float("nan"), 0.2), (-0.1, 0.1)), "KAPI-ÖLCÜLEMEZ"),
         ((True, (0.02, 0.30), (-0.10, 0.10)), "UZUNLUK-ACIKLAR"),
         ((True, (-0.10, 0.10), (-0.30, -0.02)), "RAKIP-GERCEK"),
         ((True, (0.02, 0.30), (-0.30, -0.02)), "KARMA"),
         ((True, (-0.10, 0.10), (-0.10, 0.10)), "AYRISTIRILAMAZ")]
    for g, b in V:
        assert verdict_k1b(*g) == b, (g, verdict_k1b(*g), b)
    assert verdict_k1b(True, (-0.30, -0.02), (-0.10, 0.10)) == "UZUNLUK-ACIKLAR"
    payda("k1b_prova", n_vaka=len(V) + 1, n_ad=len(ADLAR),
          bekle={"n_vaka": 7, "n_ad": 5})
    capraz_say(__file__, "verdict_k1b", ADLAR)


def aile_adlari(eksenler=("ARO", "DOM")):
    X1 = {e: (f"{e}+x1", f"{e}-x1") for e in eksenler}
    X4 = {e: (f"{e}+x4", f"{e}-x4") for e in eksenler}
    K22 = {"ARO": [(["DOM+x2_ARO+x2"], ["DOM+x2_ARO-x2"]),
                   (["DOM-x2_ARO+x2"], ["DOM-x2_ARO-x2"])],
           "DOM": [(["DOM+x2_ARO+x2"], ["DOM-x2_ARO+x2"]),
                   (["DOM+x2_ARO-x2"], ["DOM-x2_ARO-x2"])]}
    return X1, X4, K22


def g_hat(X, sec, aileler, nm, e, aile, iss):
    X1, X4, K22 = aileler
    if aile == "x1":
        return _merkez_fark(X, sec([X1[e][0]], iss), sec([X1[e][1]], iss)) - nm
    if aile == "x4":
        return _merkez_fark(X, sec([X4[e][0]], iss), sec([X4[e][1]], iss)) - nm
    return float(np.mean([_merkez_fark(X, sec(p, iss), sec(q, iss)) - nm
                          for p, q in K22[e]]))


def yukle():
    R = [json.loads(l) for l in open(f"{OUT}/uretim.jsonl", encoding="utf-8")]
    payda("k1b_yukle", n_satir=len(R), n_kol=len(set(r["kol"] for r in R)),
          n_istem=len(set(r["istem_i"] for r in R)),
          n_cekim=len(set(r["cekim"] for r in R)),
          red_bos_metin=sum(1 for r in R if not r["metin"].strip()),
          bekle={"n_satir": 4896, "n_kol": 12, "n_istem": 34, "n_cekim": 12})
    return R


def gom(R, dev, bekle_satir=4896):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(E5, device=dev)
    X = m.encode([f"query: {r['metin']}" for r in R], batch_size=64,
                 normalize_embeddings=True, show_progress_bar=False)
    X = np.asarray(X, dtype=np.float64)
    np.save(f"{OUT}/X_e5.npy", X.astype(np.float32))
    payda("k1b_gomme", n_satir=len(X), n_boyut=X.shape[1],
          red_norm_sapan=int(abs(np.linalg.norm(X, axis=1) - 1).max() > 1e-4),
          bekle={"n_satir": bekle_satir, "n_boyut": 1024})
    return X


def main(a):
    t0 = time.time()
    prova(); prova_ilkel()
    R = yukle()
    if not os.path.exists(f"{OUT}/X_e5.npy") or a.yeniden_gom:
        X = gom(R, a.dev)
    else:
        X = np.load(f"{OUT}/X_e5.npy").astype(np.float64)
    kol = np.array([r["kol"] for r in R]); ist = np.array([r["istem_i"] for r in R])
    K34 = np.unique(ist)
    CIFT = {(k, i): np.flatnonzero((kol == k) & (ist == i))
            for k in np.unique(kol) for i in K34}
    sec = lambda ks, iss: np.concatenate([CIFT[(k, i)] for k in ks for i in iss])

    AD = sorted(set(kol.tolist()))
    payda("k1b_hucre", n_kol=len(AD), red_beklenmeyen=int(len(AD) != 12),
          bekle={"n_kol": 12})
    X1, X4, K22 = aile_adlari()
    eksik = [n for e in ("ARO", "DOM")
             for n in list(X1[e]) + list(X4[e]) + [x for p in K22[e] for s in p for x in s]
             if n not in AD]
    if eksik:
        raise SystemExit(f"★ hücre adi diskte YOK: {sorted(set(eksik))} — diskteki: {AD}")

    rb = np.random.default_rng(SEED)
    hav = sec([f"ARO+x1", f"ARO-x1"], K34)
    n1 = len(sec([f"ARO+x1"], K34))
    null = np.array([_merkez_fark(X, hav[(p := rb.permutation(len(hav)))[:n1]], hav[p[n1:]])
                     for _ in range(K_NULL)])
    nm, nsd, np95 = float(null.mean()), float(null.std()), float(np.percentile(null, 95))
    hav2 = sec(["DOM+x2_ARO+x2", "DOM-x2_ARO+x2", "DOM+x2_ARO-x2", "DOM-x2_ARO-x2"], K34)
    n2 = len(hav2) // 2
    null2 = np.array([_merkez_fark(X, hav2[(p := rb.permutation(len(hav2)))[:n2]], hav2[p[n2:]])
                      for _ in range(400)])
    oran = nm / float(null2.mean())
    k0e = abs(oran - 2 ** 0.5) / (2 ** 0.5) <= 0.10
    payda("k1b_k0e", n_cekim_408=K_NULL, n_cekim_816=400, red_oran_sapan=int(not k0e))
    print(f"  [K0-e] null(408)={nm:.6f} null(816)={null2.mean():.6f} oran={oran:.4f} "
          f"(√2=1,4142) ⇒ {'GECTI' if k0e else 'DÜSTÜ'}", flush=True)

    def gh(e, aile, iss):
        return g_hat(X, sec, (X1, X4, K22), nm, e, aile, iss)

    OL, BOOT = {}, {}
    for e in ("ARO", "DOM"):
        g1, g4, g2 = gh(e, "x1", K34), gh(e, "x4", K34), gh(e, "22", K34)
        OL[e] = dict(g_x1=round(g1, 6), g_x4=round(g4, 6), g_22=round(g2, 6),
                     U=round(g4 - g1, 6), R=round(g2 - g4, 6), T=round(g2 - g1, 6))
    rbb = np.random.default_rng(SEED + 5)
    B = {e: {"U": [], "R": [], "T": []} for e in ("ARO", "DOM")}
    for _ in range(B_BOOT):
        ks = rbb.choice(K34, len(K34), replace=True)
        for e in ("ARO", "DOM"):
            g1, g4, g2 = gh(e, "x1", ks), gh(e, "x4", ks), gh(e, "22", ks)
            B[e]["U"].append(g4 - g1); B[e]["R"].append(g2 - g4); B[e]["T"].append(g2 - g1)
    for e in ("ARO", "DOM"):
        for k in ("U", "R", "T"):
            v = np.array(B[e][k])
            OL[e][f"{k}_CI"] = [round(float(np.percentile(v, 2.5)), 6),
                                round(float(np.percentile(v, 97.5)), 6)]
            OL[e][f"{k}_sd"] = round(float(v.std()), 6)
        OL[e]["MDE_1645sd"] = round(1.645 * float(np.array(B[e]["T"]).std()), 6)
        BOOT[e] = {k: np.array(B[e][k]) for k in ("U", "R", "T")}

    H = verdict_k1b(k0e, OL["DOM"]["U_CI"], OL["DOM"]["R_CI"])
    H_ARO = verdict_k1b(k0e, OL["ARO"]["U_CI"], OL["ARO"]["R_CI"])

    yuz = {}
    for ad in AD:
        m = kol == ad
        uz = [len(r["metin"].split()) for r, mm in zip(R, m) if mm]
        ttr = [len(set(r["metin"].split())) / max(1, len(r["metin"].split()))
               for r, mm in zip(R, m) if mm]
        on = [len(r["onek"].split()) for r, mm in zip(R, m) if mm]
        yuz[ad] = dict(cikti_kelime=round(float(np.mean(uz)), 1),
                       ttr=round(float(np.mean(ttr)), 4),
                       onek_kelime=round(float(np.mean(on)), 1))

    cet = None
    if not a.cetvel_atla:
        jeton = Counter()
        ap_ = sec([X4["ARO"][0]], K34)[:200]; an_ = sec([X4["ARO"][1]], K34)[:200]
        s = [dict(id=f"{i}", metin=R[i]["metin"]) for i in np.concatenate([ap_, an_])]
        cik, redj = KAL.yargila_esz(KAL.RUBRIK["ARO"], s, A2.ES, jeton=jeton)
        ar = {int(k): v for k, v in cik.items()}
        pp = np.mean([ar.get(int(i)) == "AROUSED" for i in ap_ if int(i) in ar])
        nn = np.mean([ar.get(int(i)) == "AROUSED" for i in an_ if int(i) in ar])
        cet = dict(kol="ARO±x4", AROUSED_arti=round(float(pp), 4),
                   AROUSED_eksi=round(float(nn), 4), fark=round(float(pp - nn), 4),
                   isaret_uyumu=bool((pp - nn) > 0) == (OL["ARO"]["g_x4"] > 0),
                   n_donen=len(ar), red=dict(redj), jeton=dict(jeton))
        print(f"  [CETVEL] ARO+ {pp:.4f} ↔ ARO− {nn:.4f} · fark {pp-nn:+.4f}", flush=True)

    U = json.load(open(f"{ROOT}/results/K1B_URETIM_2026-08-13.json", encoding="utf-8"))
    ort = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="results/PREREG_K1B_RAKIP_UZUNLUK_2026-08-13.md @ c0ad1d0",
               VERDICT=H, VERDICT_ARO_KONTROL=H_ARO, kapi_k0e=bool(k0e),
               kapsam_belirsizligi_cozumu="ĝ(2:2) = rakip kutbu SABIT iki ölcümün "
                                          "ortalamasi, her biri n=408 ⇒ üc nokta n-esli",
               olcum=OL, null_merkez_408=round(nm, 6), null_sd_408=round(nsd, 6),
               null_p95_408=round(np95, 6), null_merkez_816=round(float(null2.mean()), 6),
               null_oran=round(oran, 4), K_null=K_NULL, B=B_BOOT, n_kume=len(K34),
               yuzey=yuz, cetvel=cet, k0d=U.get("k0d", {}).get("n_ozdes"),
               k0d_red=U.get("k0d", {}).get("red_ozdes_degil"),
               saniye=round(time.time() - t0, 1),
               betik_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
    json.dump(ort, open(CIKTI, "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in ort.items() if k not in ("yuzey",)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--yeniden-gom", action="store_true")
    ap.add_argument("--cetvel-atla", action="store_true")
    ap.add_argument("--yalniz-prova", action="store_true")
    a = ap.parse_args()
    if a.yalniz_prova:
        prova(); prova_ilkel()
        print("★ prova GECTI — veriye DOKUNULMADI.")
        sys.exit(0)
    sys.exit(main(a))
