#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys, time
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
import addressee_run as MK
import depersonalization16_neutral_run as M9
import form_count as FS
from reading_style_object import payda

MERD = {"Zephyr-7B": ["taban", "sft", "dpo"],
        "Tulu3-8B": ["taban", "sft", "dpo", "rl"],
        "OLMo2-13B": ["taban", "sft", "dpo", "instruct"]}
TEMPLATE_KOK = __DNH_DATA__ + "/c1_panel_template"
CIK = f"{KOK}/results/v31_rung_level_2026-09-10.json"


def main():
    import os
    MK.VERI = TEMPLATE_KOK
    eksik = [f"{a}/{b}" for a, rr in MERD.items() for b in rr
             if not os.path.exists(f"{MK.VERI}/{a}/{b}/uretim.jsonl")]
    print(f"{MK.VERI} {len(MERD)}"
          f"{sum(len(v) for v in MERD.values())} {len(eksik)}"
          f"", flush=True)
    if eksik:
        print(f"     ★★ EKSIK: {eksik}")
    nlp = FS._boru(); t0 = time.time(); S = {}
    for aile, rungs in MERD.items():
        S[aile] = {}
        for b in rungs:
            R = MK.oku(aile, b)
            if R is None:
                S[aile][b] = "KOSULMADI"; continue
            V = MK.satir_bilesenleri(nlp, R)
            o = MK.olc_toplam(V)
            S[aile][b] = dict(M1=round(float(o["M1"]), 3),
                              M3=round(float(o["M3"]), 4),
                              sahis1=round(float(M9.sahis1_1k(V)), 3),
                              n_satir=len(R))
            print(f"  [{time.time()-t0:.0f}s] {aile}/{b}: "
                  f"M1={S[aile][b]['M1']:.2f} /1k · 1.kisi={S[aile][b]['sahis1']:.2f}",
                  flush=True)
    D = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         alet="scripts/v31_rung_level.py",
                         motor="addressee_run.py b114a7b46a3369cb (edit yok)",
                         borc="D-0910-V31-METIN §2", kok=TEMPLATE_KOK,
                         sinif="DÜZEY OKUMASI — verdict kurmaz"),
             merdiven=S)
    json.dump(D, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n = sum(1 for a in S for b in S[a] if isinstance(S[a][b], dict))
    payda("v31_basamak_duzey", n_merdiven=len(MERD), hal_bacak=n,
          red_kosulmadi=sum(len(v) for v in MERD.values()) - n)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
