#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, csv, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
from reading_style_object import payda
import form_count as FS
import cerceve as CC
import v3_kurator_verimi as V3

TESTSETI = f"{ROOT}/results/FORM_TESTSETI_2026-08-28.json"
MATRIS = f"{ROOT}/results/HARITA_MATRIS_2026-08-28.csv"
PANEL = __DNH_DATA__ + "/c1_panel"
TAKMA = {"Tulu-3-8B": ("Tulu3-8B", "taban", "rl"),
         "OLMo-2-13B": ("OLMo2-13B", "taban", "instruct"),
         "OLMo-2-32B": ("OLMo2-32B", "taban", "instruct")}
HURMET = ("c_rica", "c_lutfen")
CEKINCE = ("c_istek", "c_artik")

YASLANAN = [
    dict(dosya="p11_submission.tex", satir=127, tur="tanim",
         cumle="Five sentence-level counters … bare imperative, second-person "
               "obligation, **hedge**, interrogative directive, hortative",
         dayanak="sayac listesi (sayi yok)"),
    dict(dosya="p11_submission.tex", satir=131, tur="SAYI",
         cumle="balanced accuracy $0.958$ (bare imperative) and **$1.000$ for hedge**, "
               "interrogative directive and hortative",
         dayanak="FORM_TESTSETI_2026-08-28.json · altin c-pozitifleri"),
    dict(dosya="p11_submission.tex", satir=272, tur="SAYI",
         cumle="the first component (41\\% of variance) puts all six reading axes on one "
               "side and the **mood and hedge axes** on the other",
         dayanak="HARITA_MATRIS · sütun `form_c_delta` · 16 aile"),
]


def _oran(A, ix=None):
    s = {k: float(A[k][ix].sum() if ix is not None else A[k].sum())
         for k in ("c",) + CC.ALT}
    c = max(s["c"], 1.0)
    s["hurmet_payi"] = round((s["c_rica"] + s["c_lutfen"]) / c, 4)
    s["cekince_payi"] = round((s["c_istek"] + s["c_artik"]) / c, 4)
    return s


def bolum_testseti(nlp, npr):
    R = json.load(open(TESTSETI, encoding="utf-8"))
    cum = [r["cumle"] for r in R]
    A, n_c, n_j = V3.say2(nlp, cum, npr)
    altin_c = np.array([int(r["altin"]["c"]) for r in R], dtype=bool)
    tahmin_c = A["c"] > 0
    ix = np.flatnonzero(altin_c)
    payda("c_bilesim_testseti", n_madde=len(R), n_altin_c=int(altin_c.sum()),
          n_tahmin_c=int(tahmin_c.sum()),
          red_uyusmazlik=int((altin_c != tahmin_c).sum()))
    det = []
    for i in ix:
        det.append(dict(kod=R[int(i)]["kod"], katman=R[int(i)]["katman"],
                        cumle=R[int(i)]["cumle"],
                        **{k: int(A[k][int(i)]) for k in ("c",) + CC.ALT}))
    return dict(n_madde=len(R), n_altin_c_pozitif=int(altin_c.sum()),
                n_tahmin_c_pozitif=int(tahmin_c.sum()),
                uyusmazlik=int((altin_c != tahmin_c).sum()),
                bilesim=_oran(A, ix), madde=det)


def bolum_harita(nlp, npr, aileler):
    O = {}
    n_sapan = 0
    for aile, (yb, yi, yayin) in aileler.items():
        R = {}
        for ad, y in (("base", yb), ("instruct", yi)):
            M = [json.loads(l)["metin"] for l in open(f"{y}/uretim.jsonl", encoding="utf-8")]
            A, n_c, _ = V3.say2(nlp, M, npr)
            top = max(float(n_c.sum()), 1.0)
            R[ad] = {k: float(A[k].sum()) / top for k in ("c",) + CC.ALT}
        d = {k: R["instruct"][k] - R["base"][k] for k in R["base"]}
        c_abs = max(abs(d["c"]), 1e-12)
        O[aile] = dict(delta=d,
                       pay_rica=round(d["c_rica"] / c_abs, 4),
                       pay_lutfen=round(d["c_lutfen"] / c_abs, 4),
                       pay_istek=round(d["c_istek"] / c_abs, 4),
                       pay_artik=round(d["c_artik"] / c_abs, 4))
        O[aile]["yayin_form_c_delta"] = yayin
        O[aile]["yayin_sapma"] = None if yayin is None else round(d["c"] - yayin, 9)
        if yayin is not None and abs(d["c"] - yayin) > 2e-6:
            n_sapan += 1
            print(f"  ★ SAPMA {aile}: üretilen {d['c']:+.7f} ↔ yayimlanan {yayin:+.7f}",
                  flush=True)
        print(f"  {aile:16s} Δc {d['c']:+.5f} | rica {d['c_rica']:+.5f} "
              f"lütfen {d['c_lutfen']:+.5f} istek {d['c_istek']:+.5f} "
              f"artik {d['c_artik']:+.5f} | hürmet payi "
              f"{(d['c_rica']+d['c_lutfen'])/c_abs:+.3f}", flush=True)
    payda("c_bilesim_harita", n_aile=len(O), n_alt=len(CC.ALT),
          red_yayin_sapan=n_sapan)
    return O


def aile_yollari():
    R = list(csv.DictReader(open(MATRIS, encoding="utf-8")))
    O, atlanan = {}, []
    for r in R:
        a = r["aile"]
        dz, zb, zi = TAKMA.get(a, (a, "base", "instruct"))
        yb, yi = f"{PANEL}/{dz}/{zb}", f"{PANEL}/{dz}/{zi}"
        if os.path.exists(f"{yb}/uretim.jsonl") and os.path.exists(f"{yi}/uretim.jsonl"):
            O[a] = (yb, yi, float(r["form_c_delta"]) if r["form_c_delta"] else None)
        else:
            atlanan.append(a)
    payda("c_bilesim_aile", n_matris=len(R), n_okunabilir=len(O),
          red_uretim_eksik=len(atlanan))
    bos = [r["aile"] for r in R if not r["form_c_delta"]]
    print(f"  ★ atlanan {sorted(atlanan)} · matriste ZATEN bos {sorted(bos)} · "
          f"kacirilan {sorted(set(atlanan) - set(bos))}", flush=True)
    if atlanan:
        print(f"{atlanan}", flush=True)
    return O, atlanan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-process", type=int, default=24)
    ap.add_argument("--yalniz-testseti", action="store_true")
    a = ap.parse_args()
    t0 = time.time(); nlp = FS._boru(a.n_process)
    T = bolum_testseti(nlp, a.n_process)
    print(f"  ★ TEST SETI · altin c-pozitif {T['n_altin_c_pozitif']}/70 · "
          f"uyusmazlik {T['uyusmazlik']} · bilesim {T['bilesim']}", flush=True)
    H, atl = ({}, [])
    if not a.yalniz_testseti:
        A, atl = aile_yollari()
        H = bolum_harita(nlp, a.n_process, A)
    r = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="BETIM/DENETIM — verdict YOK; yazim-onarimina dipnot malzemesi",
             tanim=dict(HURMET=HURMET, CEKINCE=CEKINCE,
                        not_="hürmet_payi = (c_rica + c_lütfen) / c"),
             yaslanan=YASLANAN, testseti=T, harita=H, harita_atlanan=atl,
             saniye=round(time.time() - t0, 1))
    y = f"{ROOT}/results/composition_2026-08-31.json"
    json.dump(r, open(y, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {y}  ({r['saniye']/60:.1f} dk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
