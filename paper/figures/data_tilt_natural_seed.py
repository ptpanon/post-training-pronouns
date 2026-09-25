#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

HUK = "results/data_tilt_natural_verdict_2026-09-16.json"
ESL = "results/data_tilt_natural_matched_fark_2026-09-17.json"
PLA = "results/v90_matched_plasebo_2026-09-17.json"
AD = {"KK3_TABAN": ("1 (20260906)", "t1"), "KK3_TABAN_T2": ("2 (20260914)", "t2"), "KK3_TABAN_T3": ("3 (20260917)", "t3")}


def main():
    import os
    eksik = [y for y in (HUK, ESL) if not os.path.exists(f"{__DNH_ROOT__}/{y}")]
    if eksik:
        print(f"{len(eksik)} {eksik}")
        return 3
    K, pk = load(HUK)
    E, pe = load(ESL)
    P = pp = None
    if os.path.exists(f"{__DNH_ROOT__}/{PLA}"):
        P, pp = load(PLA)
        if P["alet"]["hal"] != "ESDEGER" or not P.get("sonuc"):
            print(f"  [PAYDA] kk3taban_seed: plasebo karti hal={P['alet']['hal']} ⇒ sütun BASILMAZ"); P = pp = None
    ek = P is not None
    ad = K["verdict"]["kk3taban"]
    print(f"  [PAYDA] kk3taban_seed: verdict={ad} · alet={K['verdict']['alet']} · n_seed={len(AD)} · "
          f"havuz={'VAR' if E.get('havuz') else 'YOK'} ⇒ esik: ALET-KAYDI/KOSULMADI ⇒ EYLEM: blok YAZILMAZ")
    assert ad in ("ISARET-TUTTU", "ISARET-DÖNDÜ", "RAF"), ad
    assert E["verdict_adi"] == ad, (E["verdict_adi"], ad)
    tut = {"ISARET-TUTTU": "the sign holds in both new seeds",
           "ISARET-DÖNDÜ": "the sign does not hold in every new seed",
           "RAF": "a new seed fell to the degeneracy shelf"}[ad]
    L = [r"\vekalet{\textbf{The natural run at three training seeds.} "
         r"The natural run was retrained from the same pairs and the same supervised checkpoint at the two further seeds of the flattened run, "
         r"so each flattened seed has a natural twin. The registered bar was the sign against the supervised checkpoint in each new seed, "
         rf"magnitudes as a range; {tut}. "
         r"The flattened-minus-natural difference is description, not part of the bar: the two runs are separate training runs, "
         r"and its interval carries the sampling of prompts, not the variance across training seeds."
         + (r" Its own paired placebo splits the two runs' generations at random within each prompt and draw, "
            r"$200$ times, and recomputes the difference averaged over draws, as the difference itself is; the run's placebo splits a single draw, "
            r"so the two placebo columns are not on one scale. The difference clears its own placebo's $p95$ "
            + ("in every row." if all(v["plasebo_ustu"] for v in P["sonuc"].values()) else
               ("in " + ", ".join(k.replace("t", "seed ") if k != "havuz" else "the pooled row"
                                  for k, v in P["sonuc"].items() if v["plasebo_ustu"]) + " and not in "
                + ", ".join(k.replace("t", "seed ") if k != "havuz" else "the pooled row"
                            for k, v in P["sonuc"].items() if not v["plasebo_ustu"]) + "."))
            if ek else "") + "}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
         (r"\begin{tabular}{lrrrrrr}\toprule" if ek else r"\begin{tabular}{lrrrrr}\toprule"),
         r"training seed & $\Delta M_1$ vs.\ SFT & 95\% CI & paired placebo $p95$ & clears it & flattened $-$ natural [95\% CI]"
         + (r" & its placebo $p95$" if ek else "") + r" \\ \midrule"]
    for k, (e, t) in AD.items():
        r = K["kol"][k]; f = E["esli"][t]
        L.append(f"{e} & ${r['dA']:+.2f}$ & $[{r['ciA'][0]:+.2f},{r['ciA'][1]:+.2f}]$ & ${r['plasebo_p95']:.2f}$ & "
                 f"{'yes' if r['bilesen_A']['plasebo'] else 'no'} & ${f['fark']:+.2f}$ $[{f['ci'][0]:+.2f},{f['ci'][1]:+.2f}]$"
                 + (f" & ${P['sonuc'][t]['plasebo_p95']:.2f}$" if ek else "") + " \\\\")
    if E.get("havuz"):
        h = E["havuz"]
        L.append(r"\midrule")
        L.append(f"pooled, three seeds & \\multicolumn{{4}}{{l}}{{natural run ${K['verdict']['dA_aralik'][0]:+.2f}$ to "
                 f"${K['verdict']['dA_aralik'][1]:+.2f}$ (range only)}} & ${h['fark']:+.2f}$ $[{h['ci'][0]:+.2f},{h['ci'][1]:+.2f}]$"
                 + (f" & ${P['sonuc']['havuz']['plasebo_p95']:.2f}$" if ek else "") + " \\\\")
    L += [r"\bottomrule\end{tabular}\end{center}"]
    _uy = "results/kk3uckat_t2_verdict_2026-09-18.json"
    ek_uc = None
    if os.path.exists(f"{__DNH_ROOT__}/{_uy}"):
        U, pu = load(_uy); hu = U["verdict"]
        if hu.get("kk3uckat") in ("ISARET-TUTTU", "ISARET-DÖNDÜ") and hu.get("alet") == "ESDEGER":
            k = hu["karsitlik"]; dA = hu["dA"]; d4 = hu["distinct4"]
            L.append(r"\vekalet{\textbf{The tripled run at two training seeds.} The second seed was registered with one bar, the "
                     r"sign of tripled minus flat at the same seed" + (", and kept it" if hu["kk3uckat"] == "ISARET-TUTTU" else ", and lost it")
                     + rf": ${k['t1']:+.2f}$ at the first seed and ${k['t2']:+.2f}$ $[{k['t2_ci'][0]:+.2f},{k['t2_ci'][1]:+.2f}]$ at the "
                     rf"second, with the tripled run at ${dA['KK3_UCKAT']:+.2f}$ and ${dA['KK3_UCKAT_T2']:+.2f}$ against the supervised "
                     rf"checkpoint and \mbox{{distinct-4}} ${d4['KK3_UCKAT']:.2f}$ and ${d4['KK3_UCKAT_T2']:.2f}$. Two seeds give a range, "
                     r"not a spread estimate.}")
            ek_uc = pu
            print(f"  [PAYDA] kk3uckat_t2_satir: verdict={hu['kk3uckat']} · t1={k['t1']} · t2={k['t2']} ⇒ cümle YAZILDI")
    io.open(f"{OUT}/KK3TABAN_SEED.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/data_tilt_natural_seed.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="KK3TABAN_SEED", ciktilar=["KK3TABAN_SEED.tex"], sources=[pk, pe] + ([pp] if ek else []) + ([ek_uc] if ek_uc else []),
                   payda=dict(verdict=ad, n_seed=len(AD), havuz=bool(E.get("havuz")), esli_plasebo=ek),
                   note="renders the sealed KK3_TABAN seed verdict and its sealed description; computes no new statistic"),
              io.open(f"{OUT}/KK3TABAN_SEED.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ KK3TABAN_SEED.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
