#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, sys, time
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
from reading_style_object import payda

KOK = __DNH_ROOT__ + ""
SUZ = f"{KOK}/results/template_filtered_2026-09-07.json"
ANA = f"{KOK}/results/V27_ONEK_ANATOMISI_2026-09-09.json"
CIK = f"{KOK}/results/v27_system_line_test_2026-09-09.json"
NPERM, SEED = 200, 20260910


def main():
    S = json.load(open(SUZ, encoding="utf-8"))
    A = json.load(open(ANA, encoding="utf-8"))["template_sistem_satiri"]
    R = [r for r in S["aileler"] if r["hal"] == "ÖLCÜLDÜ"]
    var, yok, eksik = [], [], []
    for r in R:
        a = r["aile"]; v = A.get(a)
        if not v or v.get("hal") != "OK":
            eksik.append(a); continue
        (var if v.get("sistem_var") else yok).append(
            (a, r["sbl"]["dM1"], r["sbl"]["ayrik"]))
    print(f"{len(R)} {len(var)}"
          f"{len(yok)} {len(eksik)} {eksik}"
          f"")
    if eksik or not var or not yok:
        print(""); return 3

    d = np.array([x[1] for x in var + yok], float)
    g = np.array([1] * len(var) + [0] * len(yok))
    gozlenen = float(d[g == 1].mean() - d[g == 0].mean())
    rng = np.random.default_rng(SEED)
    N = np.array([float(d[(p := rng.permutation(g)) == 1].mean() - d[p == 0].mean())
                  for _ in range(NPERM)])
    frac = float((np.abs(N) >= abs(gozlenen)).mean())
    yuk_var = sum(1 for x in var if x[2] and x[1] > 0)
    yuk_yok = sum(1 for x in yok if x[2] and x[1] > 0)
    D = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         alet="scripts/v27_system_line_test.py",
                         borc="D-0910-V27 §2 · Limitations 'obvious next candidate'",
                         n_perm=NPERM, seed=SEED,
                         null="",
                         serh=(""
                               ""
                               "")),
             sistem_VAR=[dict(aile=a, dM1=round(x, 3), ayrik=y) for a, x, y in var],
             sistem_YOK=[dict(aile=a, dM1=round(x, 3), ayrik=y) for a, x, y in yok],
             ortalama_VAR=round(float(d[g == 1].mean()), 3),
             ortalama_YOK=round(float(d[g == 0].mean()), 3),
             fark=round(gozlenen, 3), null_ort=round(float(N.mean()), 3),
             null_sd=round(float(N.std()), 3), frac_null_asan=frac,
             yukari_VAR=f"{yuk_var}/{len(var)}", yukari_YOK=f"{yuk_yok}/{len(yok)}",
             VERDICT=("SIRALAMIYOR" if frac > 0.05 else "SIRALIYOR"))
    json.dump(D, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ortalama ΔM1 · sistem VAR {D['ortalama_VAR']} ↔ YOK {D['ortalama_YOK']} "
          f"· fark {D['fark']} · frac(null≥|fark|)={frac:.3f}")
    print(f"  yukari cikan: sistem VAR {D['yukari_VAR']} ↔ YOK {D['yukari_YOK']}")
    print(f"  ★★★ VERDICT: {D['VERDICT']}")
    payda("v27_sistem_satiri", n_aile=len(R), n_var=len(var), n_yok=len(yok),
          red_cozulemedi=len(eksik), n_perm=NPERM)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
