#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys, json, os, hashlib
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import t1_adi
ROOT = __DNH_ROOT__ + ""
KAYNAK = f"{ROOT}/results/four_seed_reading_2026-09-01.json"
CIK = os.path.dirname(os.path.abspath(__file__)) + "/SEEDCOV.tex"
META = os.path.dirname(os.path.abspath(__file__)) + "/SEEDCOV.meta.json"


def sha16(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def tr(x, n=2):
    return f"${x:.{n}f}$"


def band(xs):
    xs = sorted(xs)
    if len(xs) <= 4:
        return liste([tr(x) for x in xs])
    return f"{tr(xs[0])}--{tr(xs[-1])}"


def liste(xs, birlestir="and"):
    xs = list(xs)
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + f" {birlestir} " + xs[-1]


def main():
    import four_seed_loader as _TK
    S, _SRC = _TK.oku()
    aile = {k: v for k, v in S.items() if not k.startswith("_")}
    tam = {k: v for k, v in aile.items() if v.get("hal") == "m=4"}
    N, T = len(tam), len(aile)
    en_kucuk = min(aile, key=lambda k: abs(aile[k]["tek_seed_dM1"]))
    kapsiyor = en_kucuk in tam
    tutan = [k for k, v in tam.items() if v["ayrik"] and v.get("plasebo_ustu")]
    hepsi = len(tutan) == N
    _bek = {k: v for k, v in aile.items() if v.get("hal") != "m=4"}
    _kis = {k: v for k, v in _bek.items()
            if (v.get("n_base") or 0) + (v.get("n_instruct") or 0) > 0}
    _tek = len(_bek) - len(_kis)
    if _kis:
        _ad = ", ".join(f"{k} at {v.get('n_base',0)}+{v.get('n_instruct',0)} of $4+4$"
                        for k, v in sorted(_kis.items()))
        _f1 = "is" if _tek == 1 else "are"
        _f2 = "is" if len(_kis) == 1 else "are"
        _kalan = (f"Of the remaining {len(_bek)} models, {_tek} {_f1} single-seed and "
                  f"{len(_kis)} {_f2} partway through the four-seed audit ({_ad}).")
    elif not _bek:
        _kalan = ""
    else:
        _kalan = (f"The remaining model is single-seed." if len(_bek) == 1
                  else f"The remaining {len(_bek)} models are single-seed.")
    print(f"{len(_bek)} {len(_kis)}"
          f"{_tek} {len(_bek)}")
    d = [v["fark_mutlak"] for v in tam.values()]
    sd = [v["m4_sd"] for v in tam.values()]
    _ic = sum(1 for v in tam.values()
              if min(v["seed_dM1"]) <= v["tek_seed_dM1"] <= max(v["seed_dM1"]))
    _dis = N - _ic
    _enz = sorted(((abs(v["tek_seed_dM1"] - v["m4_ort"]) / (v["m4_sd"] or 1e-9), k)
                   for k, v in tam.items()), reverse=True)[:2]
    print(f"{N} {_ic}"
          f"{_dis}"
          f"")
    print(f"{T} {N} {len(tutan)}"
          f"{en_kucuk} {int(kapsiyor)}")

    if N == 0:
        govde = ("The multi-seed audit has completed no model yet; every number in "
                 "Table~\\ref{tab:families} is single-seed.")
    else:
        p_kucuk = ("and they include the model that carries the smallest effect in the "
                   "panel, which is the one a seed artifact would break first. "
                   if kapsiyor else
                   "The model carrying the smallest effect in the panel is not yet among "
                   "them, so the most seed-fragile case is still untested. ")
        p_tut = ("all of them hold" if hepsi else
                 f"{len(tutan)} of the {N} hold")
        govde = (
            f"A separate multi-seed audit regenerates each model with four sampling "
            f"seeds and re-reads it with the same estimator. It has completed "
            + (f"\\textbf{{all {T}}}" if N == T else f"\\textbf{{{N} of {T}}} so far")
            + (f" ({liste(sorted(tam))}), " if N <= 6 else
               " (the manifest lists them), ")
            + f"{p_kucuk}"
            f"Re-read with four generation seeds, {p_tut}: the shift stays negative, "
            f"the prompt-clustered interval still excludes zero, and the effect still "
            f"exceeds the paired placebo. Four-seed pooled estimates differ from the "
            f"single-seed ones by {band(d)} per thousand, "
            f"against seed-to-seed spreads of {band(sd)}. "
            f"What survives the re-read is the sign, the interval and the placebo, in "
            f"all {N}; the \\emph{{magnitude}} does not travel as well. "
            f"{_dis} of the {N} single-seed readings fall outside their own "
            f"four-seed range (up to {_enz[0][0]:.1f} seed-sd, {t1_adi(_enz[0][1])}, and "
            f"{_enz[1][0]:.1f}, {t1_adi(_enz[1][1])}), so we read magnitudes at the four-seed "
            f"pooled value wherever it exists and mark the rest. "
            + (f"{_kalan} They are marked as such in Table~\\ref{{tab:families}}." if _kalan else ""))

    with open(CIK, "w", encoding="utf-8") as f:
        f.write("% ★ ÜRETILDI — elle düzenlenmez (figures/seed_coverage_paragraph.py). Kaynak: "
                "results/four_seed_reading_2026-09-01.json\n")
        f.write("\\paragraph{Seed coverage of the register battery.}\n")
        f.write("\\vekalet{The sixteen-model panel was generated with a single sampling "
                "seed per checkpoint, and the generation manifests of that panel do not "
                "record which seed. The single-seed status is an inference from the "
                "pipeline default rather than a logged fact. " + govde + "}\n")
    json.dump(dict(table="SEEDCOV",
                   sources=_SRC,
                   payda=dict(n_aile=T, hal_m4=N, hal_tutan=len(tutan),
                              n_bekleyen=len(_bek), hal_kismi=len(_kis)),
                   note="renders a counter; computes no new statistic"),
              open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ {os.path.basename(CIK)}  ← m=4 {N}/{T}  (+ künye)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
