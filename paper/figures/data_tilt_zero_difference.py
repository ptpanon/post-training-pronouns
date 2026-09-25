#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, re, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, ROOT, load, sha16

HUK = "results/data_tilt_zero_difference_verdict_2026-09-13.json"
KOL = "results/data_tilt_zero_difference_arm_2026-09-13.json"
TV = "results/SEED_VARYANS_2026-08-30_KK3_NOTR_t20260906.json"
BAH = "preregistration/prediction_data_tilt_zero_difference_2026-09-13.md"
T2 = "results/data_tilt_zero_difference_t2_verdict_2026-09-22.json"


def binlik(n):
    return f"{n:,}".replace(",", "{,}")


def main():
    H, ph = load(HUK)
    K, pk = load(KOL)
    T, pt = load(TV)
    h = H["verdict"]
    assert h["kk3f"] == "FARKSIZ-CEKIYOR", h["kk3f"]
    notr = H["kol"]["KK3_NOTR"]["dA"]
    dA, ciA, dB, ciB, p95 = h["dA"], h["ciA"], h["dB_notr"], h["ciB_notr"], h["plasebo_p95"]
    kis_f = K["olcum"]["KK3_FARKSIZ"]["pay_kisili"]
    kis_n = K["olcum"]["KK3_NOTR"]["pay_kisili"]
    pay = abs(dA) - p95
    b = io.open(f"{ROOT}/{BAH}", encoding="utf-8").read()
    satir = [l for l in b.splitlines() if l.startswith("| **F-KK3F-1**")][0]
    P = {ad: float(v.replace(",", ".")) for ad, v in re.findall(r"`([A-ZCGIÖSÜ-]+)` (\d,\d\d)", satir)}
    assert abs(sum(P.values()) - 1) < 1e-9 and len(P) == 7, P
    brier = sum((p - (1.0 if ad == h["kk3f"] else 0.0)) ** 2 for ad, p in P.items())
    print(f"  [PAYDA] kk3f_farksiz: dA={dA:.4f} · notr={notr:.4f} · dB={dB:.4f} · p95={p95:.4f} · pay={pay:.4f} "
          f"· kisili F={kis_f:.4f} N={kis_n:.4f} · brier={brier:.4f} · n_olasilik={len(P)} ⇒ esik: ciB sifiri "
          f"icermiyorsa «covers zero» YAZILMAZ")
    assert ciB[0] <= 0 <= ciB[1], ""
    s = (f"A run on pairs with no pronoun contrast (a second person in "
         f"${100*kis_f:.2f}\\%$ of its pairs, against ${100*kis_n:.0f}\\%$ in the gap-removed run) lowers the rate "
         f"as much, ${dA:.2f}$ $[{ciA[0]:.2f},{ciA[1]:.2f}]$ against ${notr:.2f}$ for the gap-removed run. The "
         f"difference, ${dB:.2f}$ $[{ciB[0]:.2f},+{ciB[1]:.2f}]$, covers zero, and the run clears its paired placebo by ${pay:.2f}$, "
         f"at one data size. A second training seed also lowers the rate, with an interval excluding zero, and falls "
         f"further than the gap-removed run (Appendix~\\ref{{app:D7}}).")
    io.open(f"{OUT}/KK3F_FARKSIZ.tex", "w", encoding="utf-8").write(s + "\n")
    T2d, _ = load(T2); t = T2d["verdict"]
    assert t["kk3f_t2"] == "TUTAR" and t["egitim_tam"] and not t["raf"], t["kk3f_t2"]
    assert t["ciB_notr"][1] < 0, ""
    d7 = (r"\paragraph*{A second training seed for the run without a pronoun contrast.}" "\n"
          f"\\vekalet{{Registered before training, the same pairs, step count and configuration were trained again with a "
          f"second seed and read with the first seed's reader. The rate falls by ${t['dA']:.2f}$ $[{t['ciA'][0]:.2f},{t['ciA'][1]:.2f}]$, "
          f"against ${t['seed1']['dA']:.2f}$ $[{t['seed1']['ciA'][0]:.2f},{t['seed1']['ciA'][1]:.2f}]$ in the first seed, and clears "
          f"its paired placebo (95th percentile ${t['plasebo_p95']:.2f}$); the generations stay above the degeneracy shelf "
          f"(distinct-4 ${t['distinct4']:.2f}$). Against the gap-removed run's first seed the difference is now ${t['dB_notr']:.2f}$ "
          f"$[{t['ciB_notr'][0]:.2f},{t['ciB_notr'][1]:.2f}]$: in this seed the run without a contrast lowers the rate more than "
          f"the gap-removed run rather than as much. Both seeds lower it, so the registered outcome holds; how far it falls "
          f"relative to the gap-removed run depends on the seed.}}")
    io.open(f"{OUT}/KK3F_T2_D7.tex", "w", encoding="utf-8").write(d7 + "\n")
    adim_eski, adim_yeni = T["adim"], K["adim"]
    d = (f"The step count was first carried over from the earlier runs (${binlik(adim_eski)}$) and the run's own "
         f"gate fired before training; it was re-derived from this run's length losses by the "
         f"registration's own rule (${binlik(adim_yeni)}$).")
    io.open(f"{OUT}/KK3F_SAPMA.tex", "w", encoding="utf-8").write(d + "\n")
    json.dump(dict(table="KK3F_FARKSIZ", ciktilar=["KK3F_FARKSIZ.tex", "KK3F_SAPMA.tex"],
                   sources=[ph, pk, pt, dict(yol=BAH, sha256_16=sha16(f"{ROOT}/{BAH}"))],
                   payda=dict(dA=round(dA, 4), notr=round(notr, 4), dB=round(dB, 4), plasebo_payi=round(pay, 4),
                              kisili_farksiz=round(kis_f, 6), kisili_notr=round(kis_n, 4),
                              brier=round(brier, 4), adim_eski=adim_eski, adim_yeni=adim_yeni),
                   note="renders the sealed KK-3F cascade reading; no statistic computed here"),
              io.open(f"{OUT}/KK3F_FARKSIZ.meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  ✓ KK3F_FARKSIZ.tex · «{s}»\n  ✓ KK3F_SAPMA.tex · «{d}»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
