#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
import rung_force_ci as BK
DIS = __DNH_DATA__ + "/elicit"
B_CARD = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/depersonalization16_measurement_2026-09-01.json"
CIK = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/results/V20_ELICIT_M3_2026-09-06.json"


def main():
    B = json.load(open(B_CARD, encoding="utf-8"))
    aile = sorted(k for k, v in B.items()
                  if not k.startswith("_") and isinstance(v, dict) and "fark" in v)
    nlp = FS._boru(); OUT = {}; t0 = time.time()
    for a in aile:
        ys = [f"{DIS}/uretim_m9_{a}_{z}.jsonl" for z in ("base", "instruct")]
        if not all(os.path.exists(y) for y in ys):
            print(f"{a}"); continue
        D = []
        for y in ys:
            R = [json.loads(l) for l in open(y, encoding="utf-8")]
            V = MK.satir_bilesenleri(nlp, R)
            ist = np.array([r.get("istem_i", i) for i, r in enumerate(R)])
            D.append((V, ist, len(R)))
        (Vb, ib, nb), (Vi, ii, ni) = D
        m3b = float(Vb[:, 4].sum() / max(Vb[:, 3].sum(), 1))
        m3i = float(Vi[:, 4].sum() / max(Vi[:, 3].sum(), 1))
        ci = BK.kume_boot(Vi, ii, Vb, ib, "M3")[:2]
        OUT[a] = dict(n_base=nb, n_instruct=ni,
                      duzeltme_base=float(Vb[:, 3].sum()), duzeltme_instruct=float(Vi[:, 3].sum()),
                      M3_base=m3b, M3_instruct=m3i, dM3=m3i - m3b, dM3_ci=list(ci),
                      ayrik=bool(ci[0] * ci[1] > 0))
        print(f"  ★ {a:18s} ΔM3 {m3i-m3b:+.4f} CI [{ci[0]:+.4f},{ci[1]:+.4f}] · "
              f"ham düzeltme {Vb[:,3].sum():.0f}/{Vi[:,3].sum():.0f} "
              f"[{(time.time()-t0)/60:.1f} dk]", flush=True)
    ay = [k for k, v in OUT.items() if v["ayrik"]]
    print(f"\n  [PAYDA] aile={len(OUT)} · CI-ayrik={len(ay)} (yukari "
          f"{sum(1 for k in ay if OUT[k]['dM3']>0)}) ⇒ esik: ayrik yoksa EYLEM = "
          f"«corrections migrate» bu zeminde de daralir")
    json.dump(dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               alet="scripts/v20_elicited_impersonal_correction.py", borc="D-0907-V20 §1a",
                               zemin="ELICIT-99", n_aile=len(OUT), ayrik=ay),
                   aile=OUT), open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
