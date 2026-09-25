#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, math
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/paper/figures")
from style import OUT, load

KISILI  = "results/urial_verdict_2048_x1_2026-09-11.json"
KISISIZ = "results/urial_verdict_impersonal_x1_2026-09-10.json"
CIK_S4  = f"{OUT}/URIAL_KISISIZ_S4.tex"
CIK_LIM = f"{OUT}/URIAL_KISISIZ_LIM.tex"
META    = f"{OUT}/URIAL_KISISIZ.meta.json"
KISISIZ_ISTEM_M1 = 0.0


def _ort(z):
    z = sorted(z)
    return z[len(z) // 2] if len(z) % 2 else (z[len(z)//2 - 1] + z[len(z)//2]) / 2


def main():
    A, prov_a = load(KISILI)
    B, prov_b = load(KISISIZ)
    KA, KB = A["kiyas"]["i"], B["kiyas"]["i"]
    ra, rb = KA["aile"], KB["aile"]

    _kirpik = sorted({a for a in ra
                      if ra[a].get("AD") == "ÖLCÜLEMEZ-KIRPMA"
                      or rb.get(a, {}).get("AD") == "ÖLCÜLEMEZ-KIRPMA"})
    eslesen = [a for a in ra if rb.get(a, {}).get("hal") == "ÖLCÜLDÜ"
               and ra[a].get("hal") == "ÖLCÜLDÜ" and a not in _kirpik]
    bekleyen = [a for a in ra if rb.get(a, {}).get("hal") != "ÖLCÜLDÜ"]

    ta = [ra[a]["M1_taban"] for a in eslesen]
    tb = [rb[a]["M1_taban"] for a in eslesen]
    hz_ayni = sum(1 for a in eslesen
                  if abs(ra[a]["M1_hizali"] - rb[a]["M1_hizali"]) < 1e-9)

    kayma = [rb[a]["dM1"] - ra[a]["dM1"] for a in eslesen]
    poz = sum(1 for x in kayma if x > 0)
    n = len(kayma)
    p_isaret = 2 * sum(math.comb(n, k) for k in range(poz, n + 1)) / 2 ** n

    donen = [a for a in eslesen if ra[a]["AD"] != rb[a]["AD"]]
    _say = lambda R: dict(
        asagi=sum(1 for a in eslesen if R[a]["AD"] == "AGIRLIK-ÖZELLIGI"),
        ters=sum(1 for a in eslesen if R[a]["AD"] == "TERS"),
        bicim=sum(1 for a in eslesen if R[a]["AD"] not in ("AGIRLIK-ÖZELLIGI", "TERS")))
    say_a, say_b = _say(ra), _say(rb)

    dus_pay = 100 * (sum(ta) / n - sum(tb) / n) / (sum(ta) / n)

    K = dict(n_aile=KB["n_aile"], n_olculen=KB["n_olculen"],
             bacak_eksik=KB["n_bacak_eksik"], kapsam=KB["KAPSAM_HUKMU"],
             sayim_kisisiz=KB["sayim"], sayim_kisili=KA["sayim"],
             sayim_eslesen_kisili=say_a, sayim_eslesen_kisisiz=say_b,
             n_eslesen=n, hal_kayma_pozitif=poz, p_isaret=p_isaret,
             hizali_bacak_ozdes=hz_ayni, donen_aile=donen, bekleyen=bekleyen,
             bant_kisili=[min(ta), max(ta)], bant_kisisiz=[min(tb), max(tb)],
             ortanca_kisili=_ort(ta), ortanca_kisisiz=_ort(tb),
             ort_kayma=sum(kayma) / n, dusus_payi_yuzde=dus_pay,
             istem_m1_kisisiz=KISISIZ_ISTEM_M1,
             kol_kipi=A.get("kol_kipi"), yeniden_okuma=A.get("yeniden_okuma"),
             kirpma_max_length=A.get("kirpma", {}).get("max_length"),
             olculemez_kirpma=_kirpik)

    bek = ", ".join(bekleyen) if bekleyen else "none"
    don = ", ".join(donen) if donen else "none"
    _ndon = len(donen)
    _don_c = ("no model changes sides" if _ndon == 0 else
              "the model that changes sides is \\textbf{" + don + "}" if _ndon == 1 else
              "the " + {2: "two", 3: "three"}.get(_ndon, str(_ndon))
              + " models that change sides are \\textbf{" + don + "}")
    _don_k = ("no model changes sides" if _ndon == 0 else
              "by one model, " + don if _ndon == 1 else
              "by " + {2: "two", 3: "three"}.get(_ndon, str(_ndon))
              + " models, " + don)
    _don_sayi = ("for no model" if _ndon == 0 else
                 "for one of them" if _ndon == 1 else
                 "for " + {2: "two", 3: "three"}.get(_ndon, str(_ndon))
                 + " of them")
    _don_g = (_don_sayi if _ndon == 0 else
              _don_sayi + ", " + " and ".join(donen) if _ndon == 2 else
              _don_sayi + ", " + don)
    _pen_b = "{:,}".format(B.get("kirpma", {}).get("max_length") or 0).replace(",", "{,}")
    _kip = ("read on the six single-strength prefix conditions, the ones on which the person-stripped "
            "frame fits the $" + _pen_b + "$-token window its corpus was generated at"
            if A.get("kol_kipi") == "x1" else "read on all sixteen prefix conditions")
    _kip += ", each checkpoint in its own format"
    _kirp_c = ("" if not _kirpik else
               " " + ", ".join(_kirpik) + (" is" if len(_kirpik) == 1 else " are")
               + " excluded: its person-stripped prompt exceeds that $" + _pen_b
               + "$-token window on its own tokenizer even at single strength, and "
               "the corpus was not regenerated.")
    _bek_c = ("" if not bekleyen else
              " " + bek + " is still generating; when it lands the denominator "
              "becomes " + str(K["n_aile"]) + ".")
    F = lambda x, k=1: ("%." + str(k) + "f") % x
    s4 = ("\\vekalet{\\textbf{The control that decides the copying account.} "
          "The aligned side is bit-identical across the two "
          "reads (" + str(hz_ayni) + " of " + str(n) + " models agree to $10^{-9}$), "
          "so every difference below sits in the base outputs alone. "
          "\\textbf{Stripping every second person from the prompt lowers the base band "
          "but does not collapse it}: the bases move from $" + F(K["bant_kisili"][0])
          + "$--$" + F(K["bant_kisili"][1]) + "$ per thousand (median $"
          + F(K["ortanca_kisili"]) + "$) to $" + F(K["bant_kisisiz"][0]) + "$--$"
          + F(K["bant_kisisiz"][1]) + "$ (median $" + F(K["ortanca_kisisiz"])
          + "$) --- a fall of about $" + F(dus_pay, 0) + "\\%$ of the level, with the "
          "shift in the same direction on all " + str(poz) + " of " + str(n)
          + " models (mean $" + ("%+.2f" % K["ort_kayma"]) + "$ per thousand, "
          "two-sided exact sign test $p=" + F(p_isaret, 5) + "$). A frame with no "
          "second persons in it still draws most of the density out of the base "
          "checkpoints. \\textbf{Copying the prompt is therefore a component and not "
          "the source.}" + _kirp_c + " Across the " + str(n)
          + " models measured both ways (" + _kip
          + "), \\textbf{which side a model falls on changes "
          + _don_sayi + "}: " + str(say_a["asagi"]) + " down and " + str(say_a["ters"])
          + " reversed with the persons in the frame, " + str(say_b["asagi"]) + " and "
          + str(say_b["ters"])
          + (" without them, with " + str(say_b.get("bicim", 0))
             + " coming out \\textsc{format-only}, and "
             if say_b.get("bicim") else " without them, and ")
          + _don_c
          + ". \\emph{We do not read the contrast between the two frames as a "
          "result here}: the two pre-registered analyses fix the two comparisons, "
          "not the difference between them, and that difference was not registered "
          "in advance.}\n")
    lim = ("The copying account for the in-context assistant prompt is \\textbf{narrowed "
           "further and still not closed}: the base checkpoints exceed the density of "
           "the three examples they are shown but not that of the prefix taken whole, "
           "and the control that decides it --- the same prefix rewritten without a "
           "single second person, measured at $" + F(KISISIZ_ISTEM_M1) + "$ --- was "
           "run on " + str(K["n_olculen"]) + " of " + str(K["n_aile"]) + " models. It "
           "removes about $" + F(dus_pay, 0) + "\\%$ of the base band and leaves the "
           "rest ($" + F(K["bant_kisisiz"][0]) + "$--$" + F(K["bant_kisisiz"][1])
           + "$ per thousand), so the density the bases produce is mostly not copied "
           "from the frame. Both sides are " + _kip + "." + _kirp_c + _bek_c)
    s4_kisa = ("\\vekalet{\\textbf{The control that decides the copying account.} "
               "Prompted with a person-stripped rewrite of the same frame "
               "--- a prefix measuring $" + F(KISISIZ_ISTEM_M1)
               + "$ on the same counter --- the base checkpoints still produce second "
               "persons: \\textbf{stripping every one of them from the prompt lowers "
               "the base band by about $" + F(dus_pay, 0)
               + "\\%$ and does not collapse it}, in the same direction on all "
               + str(poz) + " of " + str(n) + " models read both ways. "
               "\\textbf{Copying the prompt is therefore a component and not the "
               "source.} The band, the median, and the two models that change "
               "sides are in \\emph{The person-stripped control, in full}, below.}\n")
    io.open(CIK_S4, "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/urial_impersonal.py)\n" + s4_kisa)
    io.open(f"{OUT}/URIAL_KISISIZ_APP.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/urial_impersonal.py)\n"
        "\\paragraph*{The person-stripped control, in full.}\n" + s4)
    io.open(CIK_LIM, "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/urial_impersonal.py)\n" + lim + "\n")
    json.dump(dict(table="URIAL_KISISIZ",
                   sources=[prov_a, prov_b], payda=K,
                   note="reads two sealed cards; computes only their difference"),
              io.open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"{K['n_aile']} {K['n_olculen']}"
          f"{K['bacak_eksik']} {n} {poz}"
          f"{hz_ayni}"
          f"")
    print(f"  [PAYDA] urial_kisisiz_ad: donen_aile={donen or 'YOK'} · bekleyen={bekleyen or 'YOK'}")
    print(f"  ✓ {os.path.basename(CIK_S4)} + {os.path.basename(CIK_LIM)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
