#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/data_tilt_neutral_seed3_verdict_2026-09-15.json"
CARD_TB = "results/data_tilt_natural_verdict_2026-09-16.json"
AD = {"KK3_NOTR": "1 (20260906)", "KK3_NOTR_T2": "2 (20260914)", "KK3_NOTR_T3": "3 (20260917)"}


def main():
    K, pk = load(CARD)
    H = K["verdict"]
    assert H["kk3t3"] == "ISARET-TUTTU" and H["alet"] == "ESDEGER" and not H["raf3"], H
    gecen = [k for k in AD if K["kol"][k]["hal_A"] == "GECTI"]
    print(f"  [PAYDA] kk3t3_seed: n_seed={len(AD)} · hal_plasebo_gecen={len(gecen)} · red_raf=0 "
          f"⇒ esik: verdict adi ya da alet degisirse ⇒ EYLEM: blok YAZILMAZ")
    L = [r"\vekalet{\textbf{A third training seed puts the flattened run back where the first one had it.} "
         rf"Its displacement against the first seed's flattened run is ${H['dT3']:+.2f}$ $[{H['ciT3'][0]:+.2f},{H['ciT3'][1]:+.2f}]$ per thousand, "
         rf"inside the ${H['ref_tilt']:.2f}$ that separates the flattened run from the natural one at the first seed, and it clears its own paired placebo where the second seed did not. "
         rf"The share the flattening returns is read, as in \S\ref{{sec:dial}}, against each seed's own natural run (next table), and is printed for all three seeds, marked where the run does not clear the placebo. The spread across three seeds is still not a variance estimate.}}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrr}\toprule",
         r"training seed & $\Delta M_1$ vs.\ SFT & 95\% CI & paired placebo $p95$ & clears it & share returned \\ \midrule"]
    TB, ptb = load(CARD_TB)
    assert TB["verdict"]["kk3taban"] == "ISARET-TUTTU" and TB["verdict"]["alet"] == "ESDEGER", TB["verdict"]
    _es = TB["verdict"]["duz_eksi_dogal_esli"]
    _tk = {"KK3_NOTR": "t1", "KK3_NOTR_T2": "t2", "KK3_NOTR_T3": "t3"}
    paylar = {}
    for k, e in AD.items():
        r = K["kol"][k]; v = _es[_tk[k]]
        assert abs(v["duz"] - r["dA"]) < 1e-9, (k, v["duz"], r["dA"])
        pay = paylar[k] = v["fark"] / abs(v["dogal"])
        L.append(f"{e} & ${r['dA']:+.2f}$ & $[{r['ciA'][0]:+.2f},{r['ciA'][1]:+.2f}]$ & ${r['plasebo_p95']:.2f}$ & "
                 f"{'yes' if r['hal_A'] == 'GECTI' else 'no'} & "
                 + f"${100*pay:.0f}\\%$" + ("" if r['hal_A'] == 'GECTI' else r"$^{\dagger}$") + " \\\\")
    _isaretli = sum(1 for k in AD if K["kol"][k]["hal_A"] != "GECTI")
    print(f"  [PAYDA] kk3t3_pay_esli: n_seed={len(paylar)} · pay={[round(100 * p, 1) for p in paylar.values()]} · "
          f"isaretli={_isaretli} ⇒ esik: duz≠dA ya da verdict/alet degisirse ⇒ EYLEM: üretici DÜSER")
    L += [r"\bottomrule\end{tabular}\end{center}"]
    if _isaretli:
        L += [r"\vekalet{\footnotesize $^{\dagger}$ does not clear the placebo.}\par"]
    io.open(f"{OUT}/KK3T3_SEED.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/data_tilt_neutral_seed3.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="KK3T3_SEED", ciktilar=["KK3T3_SEED.tex"], sources=[pk, ptb],
                   payda=dict(n_seed=len(AD), plasebo_gecen=len(gecen), dT3=round(H["dT3"], 4),
                              paylar={k: round(p, 4) for k, p in paylar.items()}),
                   note="renders the sealed KK-3T3 verdict; computes no new statistic"),
              io.open(f"{OUT}/KK3T3_SEED.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ KK3T3_SEED.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
