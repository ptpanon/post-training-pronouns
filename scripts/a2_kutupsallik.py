#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from verdict_name_crosscount import capraz_say
import prefix_judge_calibration as KAL
from adim4_harman2 import kol_orani

D = __DNH_DATA__ + "/onek_korpus/seritD_150"
ONBELLEK = f"{D}/yargi_a2"
CIKTI = f"{ROOT}/results/A2_KUTUPSALLIK_2026-08-12.json"
TEMPLATE_TON = f"{ROOT}/unreleased/TEMPLATE_SAFAK_YARGI_2026-08-11.json"
TEMPLATE_TON_SHA = "15bc72d5ebdfed1b69d0dd4c2d01bce70fe02ae4f756d9a6ed43c7ac008556b3"
MANIFEST_SHA = "a79774d62becee4df527a1a69cd833fe015bcf78671152a7b88c48b76c5ae243"
ES, B_BOOT, K_NULL, SEED = 8, 2000, 200, 20260812
N_SATIR_KOL, MIN_PAYDA = 170, 40
ADLAR = ("DOLASIKLIK-YOK", "UCLARDA-YOGUN", "ORTADA-YOGUN", "YAYGIN", "ÖLCÜLEMEZ")

MERDIVEN = {
    "TON": [("TONPx4_TONMx1", +0.6), ("TONPx3_TONMx2", +0.2),
            ("TONPx2_TONMx3", -0.2), ("TONPx1_TONMx4", -0.6)],
    "ARO": [("AROPx4_AROMx1", +0.6), ("AROPx3_AROMx2", +0.2),
            ("AROPx2_AROMx3", -0.2), ("AROPx1_AROMx4", -0.6)],
}
KARAR = {"TON": (("HARSH", "CALM"), "HARSH"), "ARO": (("AROUSED", "FLAT"), "AROUSED")}
P = np.array([+0.6, +0.2, -0.2, -0.6])


def verdict_kutup(kapi_tamam, egim_ayrik, ekstra_ayrik, ekstra):
    if not kapi_tamam:
        return "ÖLCÜLEMEZ"
    if not egim_ayrik:
        return "DOLASIKLIK-YOK"
    if ekstra_ayrik and ekstra > 0:
        return "UCLARDA-YOGUN"
    if ekstra_ayrik and ekstra < 0:
        return "ORTADA-YOGUN"
    return "YAYGIN"


def _kutup(h):
    a = np.abs(h - np.nanmean(h))
    return float((a[0] + a[3]) / 2 - (a[1] + a[2]) / 2)


def _egim(h):
    return float(np.sum((P - P.mean()) * (h - np.nanmean(h))) / np.sum((P - P.mean()) ** 2))


def istatistik(h):
    if not np.all(np.isfinite(h)):
        return float("nan"), float("nan"), float("nan")
    s = _egim(h)
    k = _kutup(h)
    return s, k, k - 0.4 * abs(s)


def prova():
    sabit = np.array([0.5, 0.5, 0.5, 0.5])
    dogrusal = 0.5 + 0.3 * P
    basamak = np.array([0.9, 0.55, 0.45, 0.1])
    orta = np.array([0.62, 0.85, 0.15, 0.38])
    for ad, h, bek in (("sabit", sabit, 0.0), ("dogrusal", dogrusal, 0.0)):
        s, k, e = istatistik(h)
        assert abs(e - bek) < 1e-9, (ad, e)
        print(f"  prova {ad:9s}: ŝ={s:+.4f} KUTUP={k:.4f} EKSTRA={e:+.6f} ✓")
    s_b, k_b, e_b = istatistik(basamak)
    s_o, k_o, e_o = istatistik(orta)
    print(f"  prova basamak : ŝ={s_b:+.4f} KUTUP={k_b:.4f} EKSTRA={e_b:+.4f} (>0 ✓)")
    print(f"  prova orta    : ŝ={s_o:+.4f} KUTUP={k_o:.4f} EKSTRA={e_o:+.4f} (<0 ✓)")
    assert e_b > 0 and e_o < 0
    V = [((False, True, True, 0.1), "ÖLCÜLEMEZ"),
         ((True, False, True, 0.1), "DOLASIKLIK-YOK"),
         ((True, True, True, 0.1), "UCLARDA-YOGUN"),
         ((True, True, True, -0.1), "ORTADA-YOGUN"),
         ((True, True, False, 0.1), "YAYGIN")]
    for g, b in V:
        assert verdict_kutup(*g) == b, (g, verdict_kutup(*g), b)
    payda("a2_prova", n_vaka=len(V), n_ad=len(ADLAR), n_dejenere=4,
          bekle={"n_vaka": 5, "n_ad": 5})
    capraz_say(__file__, "verdict_kutup", ADLAR)


def k0_malzeme():
    sha = hashlib.sha256(open(f"{D}/uretim.jsonl", "rb").read()).hexdigest()
    R = {}
    eksik = []
    for m, bas in MERDIVEN.items():
        for kol, p in bas:
            y = f"{D}/kol/{kol}.jsonl"
            if not os.path.exists(y):
                eksik.append(kol)
                continue
            R[kol] = [json.loads(l) for l in open(y, encoding="utf-8")]
            if len(R[kol]) != N_SATIR_KOL:
                eksik.append(f"{kol}(n={len(R[kol])})")
    payda("a2_malzeme", n_kol=len(R), n_satir=sum(len(v) for v in R.values()),
          red_eksik=len(eksik), red_sha_degismis=int(sha != MANIFEST_SHA),
          bekle={"n_kol": 8, "n_satir": 1360})
    if eksik:
        raise SystemExit(f"★ K0-a DÜSTÜ — {eksik}")
    if sha != MANIFEST_SHA:
        raise SystemExit(f"★ K0-b DÜSTÜ — üretim sha {sha[:16]} ≠ {MANIFEST_SHA[:16]}")
    print(f"{N_SATIR_KOL} {sha[:16]}",
          flush=True)
    return R, sha


def k0_rubrik():
    S = json.load(open(TEMPLATE_TON, encoding="utf-8"))
    bek = S.pop("sha256_govde")
    got = hashlib.sha256(json.dumps(S, ensure_ascii=False, indent=2,
                                    sort_keys=True).encode()).hexdigest()
    aro_sha = hashlib.sha256(KAL.RUBRIK["ARO"]["sistem"].encode()).hexdigest()
    payda("a2_rubrik", n_yol=2, red_ton_template=int(bek != TEMPLATE_TON_SHA),
          red_govde_sapan=int(got != bek), bekle={"n_yol": 2})
    if bek != TEMPLATE_TON_SHA or got != bek:
        raise SystemExit(f"★ K0-c DÜSTÜ — TON sablonu ({got[:16]} / {bek[:16]})")
    S["sha256_govde"] = bek
    print(f"[K0-c] TON template ✓ {bek[:16]} (bankalanmisla bayt-özdes) · "
          f"ARO rubrik lafiz sha {aro_sha[:16]}", flush=True)
    return S, bek, aro_sha


def yargila_kol(kol, satirlar, eksen, S, jeton):
    os.makedirs(ONBELLEK, exist_ok=True)
    yol = f"{ONBELLEK}/{kol}__{eksen}.json"
    on = f"{eksen[0]}{int(hashlib.sha256(kol.encode()).hexdigest()[:6], 16) % 1000:03d}"
    sec = [{"id": f"{on}{i:04d}", "metin": r["metin"]} for i, r in enumerate(satirlar)]
    if os.path.exists(yol):
        d = json.load(open(yol, encoding="utf-8"))
        if d.get("n_istenen") == len(sec) and len(d["cikti"]) >= 0.9 * len(sec):
            print(f"  [ÖNBELLEK] {kol}/{eksen} · {len(d['cikti'])}/{len(sec)}", flush=True)
            return np.array([d["cikti"].get(s["id"], "") for s in sec], dtype=object)
    rk = KAL.RUBRIK[eksen]
    cik, red = KAL.yargila_esz(rk, sec, ES, template=(S if eksen == "TON" else None),
                               jeton=jeton)
    gec = yol + f".tmp{os.getpid()}"
    json.dump({"kol": kol, "eksen": eksen, "n_istenen": len(sec), "red": red,
               "cikti": cik}, open(gec, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(gec, yol)
    print(f"  [YARGI] {kol}/{eksen} · {len(cik)}/{len(sec)} · red {red}", flush=True)
    return np.array([cik.get(s["id"], "") for s in sec], dtype=object)


def merdiven_olc(m, R, et, rng):
    kollar = [k for k, _ in MERDIVEN[m]]
    kar, poz = KARAR[et]
    kol_v = np.concatenate([[k] * len(R[k]) for k in kollar])
    deb = np.concatenate([[r["debate"] for r in R[k]] for k in kollar])
    lab = np.concatenate([ETIKET[(k, et)] for k in kollar])
    temiz = np.ones(len(lab), bool)
    h, n = kol_orani(lab, kol_v, temiz, kollar, karar=kar, pozitif=poz)
    s, k, e = istatistik(h)

    ku = sorted(set(deb.tolist()))
    gr = {c: np.flatnonzero(deb == c) for c in ku}

    def _yeni(ix):
        return istatistik(kol_orani(lab[ix], kol_v[ix], temiz[ix], kollar,
                                    karar=kar, pozitif=poz)[0])
    Bs, Be = np.empty(B_BOOT), np.empty(B_BOOT)
    for b in range(B_BOOT):
        ix = np.concatenate([gr[ku[i]] for i in rng.choice(len(ku), len(ku), True)])
        Bs[b], _, Be[b] = _yeni(ix)
    ok_s, ok_e = np.isfinite(Bs), np.isfinite(Be)
    ci_s = [float(np.quantile(Bs[ok_s], q)) for q in (0.025, 0.975)]
    ci_e = [float(np.quantile(Be[ok_e], q)) for q in (0.025, 0.975)]

    Ns, Ne = np.empty(K_NULL), np.empty(K_NULL)
    for j in range(K_NULL):
        kv = kol_v.copy()
        for c in ku:
            ix = gr[c]
            kv[ix] = kol_v[ix][rng.permutation(len(ix))]
        hn, _ = kol_orani(lab, kv, temiz, kollar, karar=kar, pozitif=poz)
        Ns[j], _, Ne[j] = istatistik(hn)

    payda(f"a2_{m}_{et}", n_kol=len(kollar), n_satir=len(lab),
          n_karar=int(sum(n)), n_kume=len(ku), n_boot=int(ok_e.sum()), K_null=K_NULL,
          red_bos_etiket=int((lab == "").sum()),
          bekle={"n_kol": 4, "n_kume": 5, "K_null": 200})
    return dict(
        kollar=kollar, p=P.tolist(), h=[round(float(x), 4) for x in h], n_karar=n,
        egim=round(s, 5), KUTUP=round(k, 5), EKSTRA=round(e, 5),
        egim_CI=[round(x, 5) for x in ci_s], EKSTRA_CI=[round(x, 5) for x in ci_e],
        egim_sd=round(float(np.std(Bs[ok_s], ddof=1)), 5),
        EKSTRA_sd=round(float(np.std(Be[ok_e], ddof=1)), 5),
        egim_ayrik=bool(ci_s[0] > 0 or ci_s[1] < 0),
        EKSTRA_ayrik=bool(ci_e[0] > 0 or ci_e[1] < 0),
        null_egim=dict(ort=round(float(np.nanmean(Ns)), 5),
                       sd=round(float(np.nanstd(Ns, ddof=1)), 5),
                       frac_asan=round(float(np.mean(np.abs(Ns) >= abs(s))), 4)),
        null_EKSTRA=dict(ort=round(float(np.nanmean(Ne)), 5),
                         sd=round(float(np.nanstd(Ne, ddof=1)), 5),
                         frac_asan=round(float(np.mean(Ne >= e)), 4)),
        menzil_h=round(float(np.nanmax(h) - np.nanmin(h)), 4),
        n_kume=len(ku), B=int(ok_e.sum()))


ETIKET = {}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    t0 = time.time()
    prova()
    R, u_sha = k0_malzeme()
    S, t_sha, a_sha = k0_rubrik()
    JETON = Counter()

    prov = [{"id": f"PR{i:04d}", "metin": r["metin"]}
            for i, r in enumerate(R["TONPx4_TONMx1"][:6])]
    c1, _ = KAL.yargila(KAL.RUBRIK["TON"], prov, template=S, jeton=JETON)
    c2, _ = KAL.yargila_esz(KAL.RUBRIK["TON"], prov, ES, template=S, jeton=JETON)
    ort = [k for k in c1 if k in c2]
    ayn = sum(1 for k in ort if c1[k] == c2[k])
    payda("a2_yargi_yolu", n_istenen=6, n_ortak=len(ort), red_ayrisan=len(ort) - ayn,
          bekle={"n_istenen": 6, "n_ortak": 6})
    if len(ort) < 6 or ayn != len(ort):
        raise SystemExit(f"★ K0-d DÜSTÜ — seri ↔ eszamanli ({ayn}/{len(ort)})")
    print(f"[K0-d] yargi yolu ✓ 6/6 · seri ≡ eszamanli (ES={ES})", flush=True)

    for m, bas in MERDIVEN.items():
        for kol, _ in bas:
            for et in ("TON", "ARO"):
                ETIKET[(kol, et)] = yargila_kol(kol, R[kol], et, S, JETON)

    rng = np.random.default_rng(SEED)
    O = {f"{m}_merdiveni__{et}_okumasi": merdiven_olc(m, R, et, rng)
         for m in MERDIVEN for et in ("TON", "ARO")}

    kendi = O["TON_merdiveni__TON_okumasi"]
    yan = O["TON_merdiveni__ARO_okumasi"]
    payda_yeter = all(min(v["n_karar"]) >= MIN_PAYDA for v in O.values())
    k0e = bool(kendi["egim_ayrik"]) and payda_yeter
    VERDICT = verdict_kutup(k0e, yan["egim_ayrik"], yan["EKSTRA_ayrik"], yan["EKSTRA"])

    ik_k = O["ARO_merdiveni__ARO_okumasi"]
    ik_y = O["ARO_merdiveni__TON_okumasi"]
    IKINCIL = verdict_kutup(bool(ik_k["egim_ayrik"]) and payda_yeter,
                          ik_y["egim_ayrik"], ik_y["EKSTRA_ayrik"], ik_y["EKSTRA"])

    print(f"\n★ K0-e (kendi ekseni): TON egim {kendi['egim']:+.4f} "
          f"CI {kendi['egim_CI']} · ayrik={kendi['egim_ayrik']} · payda≥{MIN_PAYDA}"
          f"={payda_yeter}")
    print(f"★ BIRINCIL — TON merdiveni × ARO okumasi: egim {yan['egim']:+.4f} "
          f"CI {yan['egim_CI']} · EKSTRA {yan['EKSTRA']:+.4f} CI {yan['EKSTRA_CI']}")
    print(f"★ VERDICT = **{VERDICT}**   ·   ikincil (ARO merdiveni × TON) = {IKINCIL}\n",
          flush=True)

    kayit = dict(
        damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prereg="results/PREREG_A2_KUTUPSALLIK_2026-08-12.md",
        SINIF="SONUC-TASIR · VERDICT ÜRETIR — ön-kayitli, no-tweak",
        yargic=KAL.QWEN_MODEL, ES=ES, dilim_boy=KAL.DILIM_BOY,
        yargi_yollari={"TON": f"donmus template {t_sha[:16]} (bankalanmisla bayt-özdes)",
                       "ARO": f"kanonik rubrik lafiz {a_sha[:16]} (donmus template YOK)"},
        uretim_sha=u_sha, olcumler=O,
        K0e_uygulanabilirlik=dict(kendi_egim=kendi["egim"], CI=kendi["egim_CI"],
                                  ayrik=kendi["egim_ayrik"], payda_yeter=payda_yeter,
                                  MIN_PAYDA=MIN_PAYDA),
        VERDICT=VERDICT, IKINCIL_ARO_merdiveni=IKINCIL, adlar=list(ADLAR),
        jeton=dict(JETON),
        payda=dict(n_kol=8, n_satir=1360, n_yargi=2720, n_eksen=2,
                   sure_sn=round(time.time() - t0, 1)))
    gec = CIKTI + f".tmp{os.getpid()}"
    json.dump(kayit, open(gec, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=str)
    os.replace(gec, CIKTI)
    print(f"[KAYIT] {CIKTI} · {(time.time()-t0)/60:.1f} dk · jeton {dict(JETON)}")
