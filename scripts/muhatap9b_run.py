#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import depersonalization16_neutral_run as M9
from degeneracy_threshold_floor import farkli4
from reading_style_object import payda

DIS = __DNH_DATA__ + "/elicit"
DEJ_ESIK = 0.60
GECICI = "<scratch>"
AILELER = [a for a in M9.BIRINCIL]


def _yol(a, b):
    return f"{DIS}/uretim_m9_{a}_{b}.jsonl"


def oku_b(a, b, pencere=None):
    y = _yol(a, b)
    if not os.path.exists(y):
        return None
    R = [json.loads(l) for l in open(y, encoding="utf-8")]
    if len(R) < 396:
        return None
    out = []
    for r in R:
        t = r.get("metin") or ""
        if pencere:
            w = t.split()
            if len(w) < pencere:
                continue
            t = " ".join(w[:pencere])
        out.append(dict(metin=t, kol=r.get("seed"), cekim=r.get("cekim"),
                        istem_i=r.get("istem_id")))
    return out


def dejenere(a, b):
    R = [json.loads(l) for l in open(_yol(a, b), encoding="utf-8")]
    v = [x for x in (farkli4(r.get("metin") or "") for r in R) if x is not None]
    o = float(np.mean(v)) if v else 0.0
    kes = float(np.mean([str(r.get("kesik")) == "True" for r in R]))
    return o, kes


def pencere_sec(a):
    ps = []
    for b in ("base", "instruct"):
        y = _yol(a, b)
        if not os.path.exists(y):
            return None
        L = [len((json.loads(l).get("metin") or "").split()) for l in open(y, encoding="utf-8")]
        ps.append(int(np.percentile(L, 25)))
    return max(min(ps), 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uzunluk-esli", dest="esli", action="store_true",
                    help="PREREG-9b kipi: ilk-pencere okumasi")
    a = ap.parse_args()
    kip = "9b_ILK_PENCERE" if a.esli else "9_B_KOLU"
    cik = f"{ROOT}/results/PREREG9{'b' if a.esli else ''}_B_OLCUM_2026-09-01.json"
    os.makedirs(f"{GECICI}/results", exist_ok=True)

    uygun, raf, yok, pencere = [], [], [], {}
    for ai in AILELER:
        if not (os.path.exists(_yol(ai, "base")) and os.path.exists(_yol(ai, "instruct"))):
            yok.append(ai); continue
        ob, kb = dejenere(ai, "base"); oi, ki = dejenere(ai, "instruct")
        if ob < DEJ_ESIK or oi < DEJ_ESIK:
            raf.append(dict(aile=ai, farkli4_base=ob, farkli4_instruct=oi)); continue
        uygun.append(ai)
        pencere[ai] = dict(kesik_base=kb, kesik_instruct=ki, kesme_farki=ki - kb,
                           W=(pencere_sec(ai) if a.esli else None))
    print(f"★ {kip} · uygun {len(uygun)}/16 · DEJENERE-RAF {len(raf)} · KOSULMADI {len(yok)}", flush=True)
    for r in raf:
        print(f"   ✗ {r['aile']}: farkli4 base {r['farkli4_base']:.3f} / inst "
              f"{r['farkli4_instruct']:.3f} ⇒ DEJENERE-RAF")

    MERD = {ai: ("base", ["base", "instruct"], "instruct") for ai in uygun}
    M9.ONBELLEK.clear()

    def sargi_oku(x, b):
        M9._SIRA["anahtar"] = (x, b)
        return oku_b(x, b, pencere[x]["W"] if a.esli else None)

    MK.MERDIVEN = MERD; MK.ROOT = GECICI
    MK.oku = sargi_oku; MK.satir_bilesenleri = M9.sargi
    MK.kos()
    S = json.load(open(f"{GECICI}/results/prereg8_measurement_2026-09-01.json", encoding="utf-8"))

    for ai in list(S):
        Vb = M9.ONBELLEK.get((ai, "base")); Vs = M9.ONBELLEK.get((ai, "instruct"))
        if Vb is None or Vs is None:
            continue
        R = oku_b(ai, "base", pencere[ai]["W"] if a.esli else None)
        ist = np.array([r["istem_i"] for r in R])
        S[ai]["plasebo_ESLI_M1"] = M9.plasebo_esli(Vb, ist)
        s1b, s1s = M9.sahis1_1k(Vb), M9.sahis1_1k(Vs)
        m1b = 1000 * Vb[:, 1].sum() / max(Vb[:, 0].sum(), 1)
        m1s = 1000 * Vs[:, 1].sum() / max(Vs[:, 0].sum(), 1)
        S[ai]["sahis1_1k"] = dict(bas=s1b, son=s1s, d=s1s - s1b)
        S[ai]["oran_2_1"] = dict(bas=(m1b / s1b) if s1b else None,
                                 son=(m1s / s1s) if s1s else None,
                                 d=((m1s / s1s) - (m1b / s1b)) if (s1b and s1s) else None)
        S[ai].update(pencere[ai]); S[ai]["kol"] = "B"

    S["_kunye"] = dict(
        damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), kip=kip,
        prereg=("preregistration/prereg_9b_uzunluk_matched_2026-09-01.md" if a.esli
               else "preregistration/prereg_depersonalization16_2026-08-31.md"),
        motor="addressee_run.py b114a7b46a3369cb (edit yok)",
        dejenerasyon_esigi=DEJ_ESIK, n_uygun=len(uygun), n_raf=len(raf), n_kosulmadi=len(yok),
        raf=raf, kosulmadi=yok,
        TAVAN=f"{len(uygun)}/16 — B9a bari 12 ⇒ "
              f"{'TAVAN-SINIRLI' if len(uygun) < 12 else 'bar erisilebilir'} (ERRATA-2)")
    json.dump(S, open(cik, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("muhatap9b", n_aile_istenen=16, n_uygun=len(uygun), red_dejenere=len(raf),
          red_kosulmadi=len(yok), n_perm=200, n_bootstrap=1000)
    print(f"\n✓ {cik}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
