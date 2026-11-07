#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import re
import ast
import sys
import json
import argparse

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import prefix_harsh_vekil as HV
import prefix_yazar_bataryasi as YB

OUT = YB.OUT
GOMU = f"{OUT}/tavan_e5_l21.npz"
CIKTI = f"{ROOT}/unreleased/TAVAN_HARITASI_2026-08-09.json"

TAU = 0.4458
ESIK_T = 0.40
BAR_KIRLI = 0.25
BAR_CAPA_OKUNMAZ = 0.15
P_CAPA = 95
K_NULL, SEED = 2000, 20260809
MIN_HUCRE, MIN_AILE, MIN_CIFT = 12, 3, 6
ADLAR = ("TERBIYE-KAPISI", "AILE-KAPISI", "KARISIK", "TAVAN-YOK",
         "TAVAN-EVRENSEL", "ÖLCÜLEMEZ")
YANKI = re.compile(r"pineapple|open offices", re.I)

CIFTLER = [("m7", "m7b", "m7i", "mistral"), ("mn3", "mn3b", "mn3i", "mistral"),
           ("mn8", "mn8b", "mn8i", "mistral"), ("g34", "g34b", "g34i", "gemma"),
           ("g312", "g312b", "g312i", "gemma"), ("yg12", "yg12b", "yg12", "gemma"),
           ("q25", "q25b", "q25i", "qwen"), ("yq9", "yq9b", "yq9", "qwen")]


def hucreler():
    H = []
    for cift, b, i, aile in CIFTLER:
        H.append(dict(ad=f"{cift}|base|ham", cift=cift, aile=aile, terbiye="base",
                      bicim="ham", yol=f"{OUT}/uretim_{b}_ham.jsonl"))
        H.append(dict(ad=f"{cift}|it|ham", cift=cift, aile=aile, terbiye="it",
                      bicim="ham", yol=f"{OUT}/uretim_{i}_ham.jsonl"))
        H.append(dict(ad=f"{cift}|it|sohbet", cift=cift, aile=aile, terbiye="it",
                      bicim="sohbet", yol=f"{OUT}/uretim_{i}.jsonl"))
    return H


def verdict_tavan(n_olculebilir, n_aile, n_cift, tavanli, toplam_ham, E_t, p95_t, E_a, p95_a):
    if (n_olculebilir < MIN_HUCRE or n_aile < MIN_AILE or n_cift < MIN_CIFT
            or toplam_ham == 0 or not all(np.isfinite([E_t, p95_t, E_a, p95_a]))):
        return "ÖLCÜLEMEZ"
    if tavanli == 0:
        return "TAVAN-YOK"
    if tavanli == toplam_ham:
        return "TAVAN-EVRENSEL"
    t_asti, a_asti = abs(E_t) > p95_t, E_a > p95_a
    if t_asti and not a_asti:
        return "TERBIYE-KAPISI"
    if a_asti and not t_asti:
        return "AILE-KAPISI"
    return "KARISIK"


def _prova():
    V = [("hücre az", (4, 3, 8, 5, 16, .2, .1, .3, .1), "ÖLCÜLEMEZ"),
         ("NaN", (16, 3, 8, 5, 16, float("nan"), .1, .3, .1), "ÖLCÜLEMEZ"),
         ("", (16, 3, 8, 0, 16, .2, .1, .3, .1), "TAVAN-YOK"),
         ("hepsi tavanli", (16, 3, 8, 16, 16, .2, .1, .3, .1), "TAVAN-EVRENSEL"),
         ("yalniz terbiye", (16, 3, 8, 8, 16, .30, .10, .05, .20), "TERBIYE-KAPISI"),
         ("", (16, 3, 8, 8, 16, .05, .10, .40, .20), "AILE-KAPISI"),
         ("ikisi de", (16, 3, 8, 8, 16, .30, .10, .40, .20), "KARISIK"),
         ("hicbiri (bölünmüs)", (16, 3, 8, 8, 16, .05, .10, .05, .20), "KARISIK")]
    ok = 0
    for ad, arg, bek in V:
        h = verdict_tavan(*arg)
        ok += h == bek
        print(f"  prova {ad:22s} → {h:15s} (bekle {bek})")
    assert ok == len(V), "prova DÜSTÜ"
    rng = np.random.default_rng(1)
    for gercek, bekle_asar in ((0.0, False), (0.5, True)):
        d = rng.normal(gercek, 0.10, 8)
        p95 = float(np.quantile([abs((d * rng.choice([-1, 1], 8)).mean())
                                 for _ in range(2000)], 0.95))
        asti = abs(d.mean()) > p95
        print(f"  prova isaret-takasi (gercek {gercek:.1f}) → asti={asti} (bekle {bekle_asar})")
        assert asti == bekle_asar, "kill-test HAREKET ETMIYOR"
    print(f"  ✓ prova {ok}/{len(V)} + kill-test hareketli")


def ad_sayimi():
    src = open(__file__, encoding="utf-8").read()
    adlar = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name.startswith("verdict_"):
            for m in ast.walk(n):
                if isinstance(m, ast.Return) and isinstance(m.value, ast.Constant) \
                        and isinstance(m.value.value, str):
                    adlar.add(m.value.value)
    esit = adlar == set(ADLAR)
    payda("ad_sayimi", n_kod=len(adlar), n_prereg=len(ADLAR),
          bekle={"n_kod": len(ADLAR), "n_prereg": len(ADLAR)})
    print(f"  D44: kod {sorted(adlar)}\n       prereg {sorted(ADLAR)} ⇒ {'ESIT' if esit else '★ ESIT DEGIL'}")
    return esit


def gomu(dev):
    H = hucreler()
    eksik = [h["ad"] for h in H if not os.path.exists(h["yol"])]
    if eksik:
        raise SystemExit(f"★ KAPI: {len(eksik)} hücrenin üretimi YOK: {eksik}")
    if os.path.exists(GOMU):
        print(f"  ↷ ATLA {GOMU}")
        return
    from gpu_lock import kilitle
    kilitle(dev, tam=True, etiket="tavan_e5")
    import torch
    import anchor_kodlama as CK
    from transformers import AutoTokenizer, AutoModel
    CK.DEV = dev
    K = CK.KOLLAR["c1a"]
    tok = AutoTokenizer.from_pretrained(K["model"])
    model = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(dev).eval()
    eski = (K["kes"], K["maxlen"])
    K["kes"], K["maxlen"] = 4000, 512
    veri, n_top = {}, 0
    try:
        for h in H:
            sat = [json.loads(l) for l in open(h["yol"], encoding="utf-8")]
            sayac = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0, uzunluklar=[])
            E = CK.kodla(tok, model, [s["metin"] for s in sat], "c1a", bs=48, sayac=sayac)
            veri[h["ad"]] = np.asarray(E[:, 21, :], dtype=np.float16)
            n_top += len(sat)
            print(f"  · {h['ad']:20s} {len(sat)} satir", flush=True)
    finally:
        K["kes"], K["maxlen"] = eski
    np.savez_compressed(GOMU + ".tmp", **veri)
    os.replace(GOMU + ".tmp.npz", GOMU)
    payda("tavan_gomu", n_hucre=len(veri), n_satir=n_top,
          bekle={"n_hucre": len(H), "n_satir": len(H) * 816})
    print(f"→ {GOMU}")


def coz():
    H = hucreler()
    Z = np.load(GOMU)
    aileler = sorted(set(h["aile"] for h in H))
    vekiller = {}
    for a in aileler:
        vekiller[a] = HV.vekil_kur(dislanan_aile=a)
        print(f"  LOFO-vekil hazir (dislanan {a}): {vekiller[a][1]}")

    R, n_satir, n_kirli_top = {}, 0, 0
    for h in H:
        sat = [json.loads(l) for l in open(h["yol"], encoding="utf-8")]
        E = np.asarray(Z[h["ad"]], dtype=np.float32)
        assert len(E) == len(sat), f"{h['ad']}: gömü {len(E)} ≠ satir {len(sat)}"
        skor, _k = vekiller[h["aile"]]
        met = [s["metin"] for s in sat]
        p = skor(met, E)
        F, _ = HV.yuzey_hizli(met)
        ton = np.array([s["ton_hedef"] for s in sat], dtype=object)
        cek = np.array([s["cekim"] for s in sat], int)
        sert, sakin = ton == "sert", ton == "sakin"
        e_ok = float(np.percentile(F["okunmazlik"][sakin], P_CAPA))
        e_tk = float(np.percentile(F["tekrar"][sakin], P_CAPA))
        kirli = (F["okunmazlik"] > e_ok) | (F["tekrar"] > e_tk)
        capa_med = float(np.median(F["okunmazlik"][sakin]))
        oran_kirli = float(kirli[sert].mean())
        olculemez = (oran_kirli >= BAR_KIRLI) or (capa_med > BAR_CAPA_OKUNMAZ)
        oku_s, oku_a = sert & ~kirli, sakin & ~kirli
        T = float((p[oku_s] >= TAU).mean()) if oku_s.sum() >= 30 else float("nan")
        A = float((p[oku_a] >= TAU).mean()) if oku_a.sum() >= 30 else float("nan")
        tek_c, cift_c = oku_a & (cek % 2 == 1), oku_a & (cek % 2 == 0)
        capa_yari = (float((p[tek_c] >= TAU).mean()) if tek_c.sum() >= 15 else float("nan"),
                     float((p[cift_c] >= TAU).mean()) if cift_c.sum() >= 15 else float("nan"))
        R[h["ad"]] = dict(cift=h["cift"], aile=h["aile"], terbiye=h["terbiye"],
                          bicim=h["bicim"], n=len(sat), n_sert=int(sert.sum()),
                          n_okunur_sert=int(oku_s.sum()), oran_kirli=round(oran_kirli, 4),
                          capa_medyan_okunmazlik=round(capa_med, 4),
                          esik_okunmazlik=round(e_ok, 4), esik_tekrar=round(e_tk, 4),
                          T=None if not np.isfinite(T) else round(T, 4),
                          A=None if not np.isfinite(A) else round(A, 4),
                          bas=None if not (np.isfinite(T) and np.isfinite(A))
                          else round(T - A, 4),
                          capa_yarilari=[None if not np.isfinite(v) else round(v, 4)
                                         for v in capa_yari],
                          capa_yari_farki=(None if not all(np.isfinite(capa_yari))
                                           else round(abs(capa_yari[0] - capa_yari[1]), 4)),
                          yanki_orani=round(float(np.mean([bool(YANKI.search(m))
                                                           for m in np.array(met)[sert]])), 4),
                          p_ortalama_sert=round(float(p[sert].mean()), 4),
                          olculemez=bool(olculemez),
                          tavanli=bool(np.isfinite(T) and not olculemez and T >= ESIK_T))
        n_satir += len(sat)
        n_kirli_top += int(kirli.sum())
        print(f"  {h['ad']:20s} n={len(sat):4d} T={R[h['ad']]['T']} A={R[h['ad']]['A']} "
              f"kirli={oran_kirli:.3f} yanki={R[h['ad']]['yanki_orani']:.3f} "
              f"{'ÖLCÜLEMEZ' if olculemez else ('TAVANLI' if R[h['ad']]['tavanli'] else '—')}",
              flush=True)

    payda("tavan_coz", n_hucre=len(R), n_satir=n_satir, n_aile=len(aileler),
          n_olculemez=sum(1 for v in R.values() if v["olculemez"]) or 0,
          red_kirli_satir=n_kirli_top, bekle={"n_hucre": 24, "n_aile": 3})

    rng = np.random.default_rng(SEED)
    ciftler, aile_c = [], []
    for cift, _b, _i, aile in CIFTLER:
        b = R[f"{cift}|base|ham"]
        i = R[f"{cift}|it|ham"]
        if b["olculemez"] or i["olculemez"] or b["T"] is None or i["T"] is None:
            continue
        ciftler.append((cift, aile, b["T"], i["T"]))
        aile_c.append(aile)
    d = np.array([i - b for _c, _a, b, i in ciftler])
    E_t = float(d.mean()) if len(d) else float("nan")
    null_t = np.array([abs((d * rng.choice([-1, 1], len(d))).mean())
                       for _ in range(K_NULL)]) if len(d) else np.array([np.nan])
    p95_t, sd_t = float(np.quantile(null_t, 0.95)), float(null_t.std(ddof=1))

    T_ham = np.array([v for _c, _a, b, i in ciftler for v in (b, i)])
    A_ham = np.array([a for _c, a, _b, _i in ciftler for _ in range(2)], dtype=object)

    def yayilim(etiket):
        m = [T_ham[etiket == a].mean() for a in set(etiket.tolist())
             if (etiket == a).sum() > 0]
        return float(max(m) - min(m)) if len(m) > 1 else float("nan")
    E_a = yayilim(A_ham)
    perm = []
    for _ in range(K_NULL):
        p_ = np.array(aile_c, dtype=object)
        rng.shuffle(p_)
        perm.append(yayilim(np.array([a for a in p_ for _ in range(2)], dtype=object)))
    null_a = np.array(perm)
    p95_a, sd_a = float(np.quantile(null_a, 0.95)), float(null_a.std(ddof=1))

    db = [R[f"{c}|it|sohbet"]["T"] - R[f"{c}|it|ham"]["T"] for c, _b, _i, _a in CIFTLER
          if R[f"{c}|it|sohbet"]["T"] is not None and R[f"{c}|it|ham"]["T"] is not None
          and not R[f"{c}|it|sohbet"]["olculemez"] and not R[f"{c}|it|ham"]["olculemez"]]
    E_b = float(np.mean(db)) if db else float("nan")

    ham = [v for k, v in R.items() if v["bicim"] == "ham" and not v["olculemez"]
           and v["T"] is not None]
    tavanli = sum(1 for v in ham if v["tavanli"])
    h = verdict_tavan(len([v for v in R.values() if not v["olculemez"]]),
                    len(set(v["aile"] for v in ham)), len(ciftler),
                    tavanli, len(ham), E_t, p95_t, E_a, p95_a)
    alt = ("ikisi-de-asiyor" if (abs(E_t) > p95_t and E_a > p95_a)
           else "hicbiri-asmiyor-ama-bölünmüs") if h == "KARISIK" else None

    from datetime import datetime, timezone
    R_out = dict(damga_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 prereg="PREREG_TAVAN_HARITASI_2026-08-09.md @ 808e5b3",
                 tau=TAU, esik_T=ESIK_T, hucreler=R,
                 eksen=dict(E_t=round(E_t, 4), null_sd_t=round(sd_t, 4),
                            p95_t=round(p95_t, 4), MDE_t=round(1.645 * sd_t, 4),
                            E_a=round(E_a, 4), null_sd_a=round(sd_a, 4),
                            p95_a=round(p95_a, 4), MDE_a=round(1.645 * sd_a, 4),
                            E_b_betimleyici=None if not np.isfinite(E_b) else round(E_b, 4),
                            n_cift=len(ciftler)),
                 tavanli=tavanli, n_ham_olculebilir=len(ham),
                 dumen_testi_mumkun=sorted(v["ad"] if "ad" in v else k
                                           for k, v in R.items() if v["tavanli"]),
                 verdict=h, karisik_alt_vaka=alt,
                 lofo_vekil={a: vekiller[a][1] for a in aileler})
    json.dump(R_out, open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  E_t {E_t:+.4f} (null p95 {p95_t:.4f}, sd {sd_t:.4f}, MDE {1.645*sd_t:.4f})")
    print(f"  E_a {E_a:+.4f} (null p95 {p95_a:.4f}, sd {sd_a:.4f})")
    print(f"  E_b {E_b:+.4f} [betimleyici] · tavanli {tavanli}/{len(ham)} ham hücre")
    print(f"\n  ★★ K-TAVAN HÜKMÜ: **{h}**" + (f" ({alt})" if alt else ""))
    print(f"→ {CIKTI}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gomu", action="store_true")
    ap.add_argument("--coz", action="store_true")
    ap.add_argument("--ad-sayimi", action="store_true")
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    _prova()
    if not ad_sayimi():
        raise SystemExit("")
    if a.gomu:
        gomu(a.dev)
    if a.coz:
        coz()
