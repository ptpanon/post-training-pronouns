#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
import person_dial_verdict as KH
from reading_style_object import payda

CIK = f"{ROOT}/results/matched_arm_verdict_2026-09-11.json"


def esli_kaskad(H, ctx):
    print("\n★★★ M-ESLI-KOL — kol basina verdict (prereg §4, gevsetilmeden)", flush=True)
    print(f"  ★ bar={KH.BAR}/1k · MDE bandi {KH.BAR}–{KH.MDE_B} · okuma B birincil "
          f"(kol − {KH.REF_B}) · A yaninda (kol − SFT)", flush=True)
    for k in H:
        v = H[k]
        print(f"  {k:14s} dB={v['dB']:+7.3f} [{v['ciB'][0]:+.3f},{v['ciB'][1]:+.3f}] "
              f"· plasebo p95={v['plasebo_p95']:.3f} · **{v['hal_B']}** "
              f"| A: dA={v['dA']:+7.3f} {v['hal_A']} · raf={v['raf']}", flush=True)
    payda("m_esli_kol_verdict", n_kol=len(H),
          hal_gecen_B=sum(1 for v in H.values() if v["hal_B"] == "GECTI"),
          hal_rafta=sum(1 for v in H.values() if v["raf"]),
          red_kol_eksik=int(any(k not in H for k in KH.KOL_YUVA)))
    return {k: dict(hal_B=H[k]["hal_B"], hal_A=H[k]["hal_A"],
                    dB=H[k]["dB"], dA=H[k]["dA"], ciB=list(H[k]["ciB"]),
                    plasebo_p95=H[k]["plasebo_p95"], raf=H[k]["raf"]) for k in H}


def main():
    KH.KOL_YUVA = {"ESLI": 771, "KISI_ASAGI": 844, "RASTGELE": 114}
    KH.M = 4
    KH.REF_B = "RASTGELE"
    KH.KASKAD = esli_kaskad
    KH.PREREG_ADI = (""
                    "")
    KH.CIK = os.environ.get("M_ESLI_CIK", CIK)
    print(f"★ M-ESLI-KOL · yuvalar {KH.KOL_YUVA} · M={KH.M} · referans {KH.REF_B}", flush=True)
    return KH.main()


if __name__ == "__main__":
    sys.exit(main() or 0)
