#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
from style import OUT, load
from reviewer_readings_v80 import sinif

G = "results/"
SATIR = [
    ("Raw continuation, four seeds pooled", ("v97_t1_dogrudan_birinci_2026-09-19.json", "aile.*.ci", None),
     ("four_seed_level_2026-09-16.json", "*.ci", None), r"Table~\ref{tab:families}; App.~\ref{app:D2}", 16),
    ("Raw continuation, one seed (registered)", ("reviewer_q1_panel_primary_2026-09-15.json", "aile.*.dogrudan_US_haric.ci", None),
     ("PREREG9_A_OLCUM_2026-08-31.json", "*.fark.M1.ci", ("sinif", "BIRINCIL")), r"App.~\ref{app:repro}, \ref{app:D2}", 16),
    ("Chat template, filtered set", ("template_filtered_first_person_2026-09-15.json", "aileler.*.sbl.ci1", None),
     ("template_filtered_2026-09-07.json", "aileler.*.sbl.ci", None), r"App.~\ref{app:D3}", 16),
    ("Chat template, unfiltered", ("v105_template_birinci_suzgecsiz_2026-09-23.json", "aile.*.ci", None),
     ("v23_template_count_2026-09-06.json", "B_sablonlu_delta.*.ci", None), r"App.~\ref{app:D3}", 16),
    ("In-context prompt, own format", ("reviewer_q1_urial_primary_2026-09-15.json", "kiyas.i.*.cs.ci", None),
     ("URIAL_VERDICT_2048_16_K2048_2026-09-11.json", "kiyas.i.aile.*.ci", None), r"App.~\ref{app:D4}", 16),
    ("In-context prompt, identical string", ("reviewer_q1_urial_primary_2026-09-15.json", "kiyas.ii.*.cs.ci", None),
     ("URIAL_VERDICT_2048_16_K2048_2026-09-11.json", "kiyas.ii.aile.*.ci", None), r"App.~\ref{app:D4}", 16),
    ("Person-stripped in-context prompt, own format", None,
     ("urial_verdict_impersonal_x1_2026-09-10.json", "kiyas.i.aile.*.ci", ("AD!", "ÖLCÜLEMEZ-KIRPMA")), r"App.~\ref{app:D4}", 15),
    ("ELICIT-99, own format", ("P11_2X2_ELICIT_2026-09-03.json", "_aile.*.SAHIS1_1k.ci", None),
     ("P11_2X2_ELICIT_2026-09-03.json", "_aile.*.M1_1k.ci", None), r"App.~\ref{app:D2}", 15),
    ("ELICIT-99, identical string", ("v98_elicit_ozdes_2026-09-21.json", "aile.*.IS1_1k.ci", None),
     ("v98_elicit_ozdes_2026-09-21.json", "aile.*.M1_1k.ci", None), r"App.~\ref{app:D2}", 16),
    ("Ordinary instructions, own format", ("v91_konusan_sayimi_2026-09-17.json", "aile.*.IS1.ci", None),
     ("siradan_instruction_reading_2026-09-17.json", "aile.*.M1.ci", None), r"App.~\ref{app:D4}", 16),
    ("Ordinary instructions, identical string", ("v96_q4_urial_ozdes_2026-09-19.json", "aile.*.K1.IS1.ci", None),
     ("v96_q4_urial_ozdes_2026-09-19.json", "aile.*.K1.M1.ci", None), r"App.~\ref{app:D4}", 16),
    ("Ordinary instructions, identical person-stripped string", ("v96_k3_impersonal_ozdes_2026-09-19.json", "aile.*.IS1_1k.ci", None),
     ("v96_k3_impersonal_ozdes_2026-09-19.json", "aile.*.M1_1k.ci", None), r"App.~\ref{app:D4}", 16),
    ("Ordinary instructions, base person-stripped, aligned own format", ("v96_q4_urial_ozdes_2026-09-19.json", "aile.*.K2.IS1.ci", None),
     ("v96_q4_urial_ozdes_2026-09-19.json", "aile.*.K2.M1.ci", None), r"App.~\ref{app:D4}", 16),
]


def coz(d, yol):
    out = [((), d)]
    for p in yol.split("."):
        yeni = []
        for k, x in out:
            if p == "*":
                it = x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else []
                yeni += [(k + (a,), v) for a, v in it if not str(a).startswith("_")]
            elif isinstance(x, dict) and p in x:
                yeni.append((k, x[p]))
        out = yeni
    return out


def say(hucre, kaynaklar):
    card, yol, suz = hucre
    d, prov = load(G + card)
    kaynaklar[prov["yol"]] = prov
    sat = coz(d, yol)
    if suz and suz[0] == "sinif":
        sat = [(k, v) for k, v in sat if d[k[0]].get("sinif") == suz[1]]
    if suz and suz[0] == "AD!":
        on = yol.rsplit(".", 1)[0].rsplit(".*", 1)[0]
        aile = coz(d, on)[0][1]
        sat = [(k, v) for k, v in sat if aile[k[-1]].get("AD") != suz[1]]
    s = [sinif(c) for _, c in sat]
    return s.count("asagi"), s.count("yukari"), s.count("null"), len(s)


def hucre_tex(a, u, n, toplam, payda):
    if a == toplam:
        h = rf"${a}{{\downarrow}}$"
    else:
        h = "$" + r"\;".join(f"{x}{{{y}}}" for x, y in ((a, r"\downarrow"), (u, r"\uparrow"), (n, r"\sim")) if x) + "$"
    return h + (rf"$^{{/{toplam}}}$" if payda != 16 else "")


def main():
    kaynaklar, L, say_ = {}, [], []
    for ad, I, Y, ek, payda in SATIR:
        hi = say(I, kaynaklar) if I else None
        hy = say(Y, kaynaklar)
        for h in (hi, hy):
            if h and h[3] != payda:
                print(f"  [PAYDA] t_okumalar: {ad} payda {h[3]} ≠ {payda} ⇒ EYLEM: tablo YAZILMAZ"); return 3
        say_.append(dict(satir=ad, I=hi, you=hy))
        L.append(f"{ad} & " + (hucre_tex(*hi, payda) if hi else "---") + f" & {hucre_tex(*hy, payda)} & {ek} \\\\")
    T = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Which readings survive.} Models whose aligned-minus-base rate falls ($\downarrow$), rises ($\uparrow$) "
         r"or has a 95\% interval covering zero ($\sim$), of 16 unless marked. ``Identical string'': both checkpoints read the "
         r"same text. ---: not read. Last column: where each reading is reported.}",
         r"\label{tab:readings}", r"\begin{tabular}{@{}p{0.46\linewidth}ccl@{}}\toprule", r"Reading & ``I'' & ``you'' & Reported in \\ \midrule"]
    T += L + [r"\bottomrule\end{tabular}", r"\end{table}"]
    io.open(f"{OUT}/T_OKUMALAR.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/t_okumalar.py)\n" + "\n".join(T) + "\n")
    json.dump(dict(table="T_OKUMALAR", ciktilar=["T_OKUMALAR.tex"], sources=list(kaynaklar.values()), payda=dict(sayim=say_),
                   note="counts each reading's 95% intervals from its card; no statistic computed here"),
              io.open(f"{OUT}/T_OKUMALAR.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(L)} {len(kaynaklar)}")
    return 0 if len(L) == 13 else 3


if __name__ == "__main__":
    sys.exit(main())
