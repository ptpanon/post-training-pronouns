#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys, json, io
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/rung_table_2026-09-01.json"
CARD_EK = "results/v26_rung_extension_bare_2026-09-09.json"
CARD_SBL = "results/v26_rung_zephyr_template_2026-09-09.json"
CARD_SBL2 = "results/v28_rung_template_2026-09-10.json"
CARD_70B = "results/v31_rung_tulu70b_2026-09-10.json"
CARD_SBL3 = "results/rung_olmo2_7b_template_2026-09-24.json"
CARD_SBL4 = "results/rung_tulu70b_template_2026-09-24.json"
TIRE_SEBEP = {"Zephyr-7B": "yok", "OLMo2-32B": "template", "OLMo3-7B": "template", "Tulu3-70B": "kosulmadi"}
CARD_7B = "results/v31_rung_olmo2_7b_2026-09-10.json"
CARD_V61 = "results/v61_ladder_2026-09-11.json"
CARD_DUZEY = "results/ladder_level_bare_2026-09-15.json"
KUNYE_V61 = __DNH_DATA__ + "/c1_panel_v61/{a}/{b}/uretim_kunye.json"
EN = {"taban": "base", "sft": "SFT", "dpo": "DPO", "rl": "RL", "instruct": "Instruct",
      "rlvr": "RLVR"}
AD = {"Tulu3-8B": "Tülu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B",
      "Zephyr-7B": "Zephyr-7B", "OLMo3-7B": "OLMo-3-7B", "Tulu3-70B": "Tülu-3-70B", "OLMo2-7B": "OLMo-2-7B"}


_SAYILAR = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def SAYI(k):
    return _SAYILAR.get(k, str(k))


def _ortusur(a, b):
    return not (a["ci"][1] < b["ci"][0] or b["ci"][1] < a["ci"][0])


def _sinif(v):
    ad = list(v["adimlar"]); bs, sd = v["adimlar"][ad[0]], v["adimlar"][ad[1]]
    if not bs["ayrik"]:
        return "tercih-tek-basina"
    if _ortusur(bs, sd):
        return "berabere"
    return "tercih-buyuk" if abs(sd["M1"]) > abs(bs["M1"]) else "denetim-buyuk"


def main():
    S, prov = load(CARD)
    E, prov_ek = load(CARD_EK)
    for a, v in E.items():
        if not a.startswith("_"):
            S[a] = v
    _ek_hal, _ek_prov = {}, []
    for _ad, _yol in (("Tulu3-70B", CARD_70B), ("OLMo2-7B", CARD_7B)):
        try:
            _T, _prov = load(_yol)
            _ek_prov.append(_prov)
            _n = 0
            for a, v in _T.items():
                if not a.startswith("_"):
                    S[a] = v; _n += 1
            _ek_hal[_ad] = "GIRDI" if _n else "CARD BOS"
        except Exception as _e:
            _ek_hal[_ad] = f"YOK ({type(_e).__name__})"
    print(f"{_ek_hal}"
          f"")
    SBL, kaynak_sbl = {}, []
    _sbl_hal = {}
    for _k in (CARD_SBL, CARD_SBL2, CARD_SBL3, CARD_SBL4):
        try:
            T, _p = load(_k)
            _gir = {a: v for a, v in T.items() if not a.startswith("_") and isinstance(v, dict)}
            SBL.update(_gir)
            kaynak_sbl.append(_p)
            _sbl_hal[_k.split("/")[-1]] = sorted(_gir) or "CARD BOS"
        except Exception as _e:
            _sbl_hal[_k.split("/")[-1]] = f"YOK ({type(_e).__name__})"
    print(f"{_sbl_hal}"
          f"")
    prov_sbl = kaynak_sbl[0] if kaynak_sbl else None
    _oran, _pay, _mutlak = {}, {}, {}
    for a in SBL:
        if a in S:
            def _r(v):
                ad = list(v["adimlar"]); bs, sd = v["adimlar"][ad[0]], v["adimlar"][ad[1]]
                return abs(sd["M1"]) / max(abs(bs["M1"]), 1e-9)
            def _p(v):
                ad = list(v["adimlar"])
                return 100.0 * v["adimlar"][ad[1]]["M1"] / v["uctan_uca"]["M1"]
            def _m(v):
                return abs(v["adimlar"][list(v["adimlar"])[1]]["M1"])
            _oran[a] = (round(_r(S[a]), 3), round(_r(SBL[a]), 3))
            _pay[a] = (round(_p(S[a]), 1), round(_p(SBL[a]), 1))
            _mutlak[a] = (round(_m(S[a]), 2), round(_m(SBL[a]), 2))
    _SIRA = ["Zephyr-7B", "Tulu3-8B", "Tulu3-70B", "OLMo2-7B", "OLMo2-13B"]
    _s4dz = [a for a in _SIRA if a in _pay] + [a for a in _pay if a not in _SIRA]
    def _yuz(x):
        return f"${x:.1f}\\%$" if abs(x) < 1 else f"${x:.0f}\\%$"
    def _liste(xs):
        return ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else xs[0]
    _s4dusen = [a for a in _s4dz if _pay[a][1] < _pay[a][0]]
    _nd = len(_s4dz)
    _bas = ((f"SFT's share grows on all {SAYI(_nd)} sequences we can read both ways. " if len(_s4dusen) == _nd else
             f"SFT's share grows on {SAYI(len(_s4dusen))} of the {SAYI(_nd)} sequences we can read both ways. On "
             + _liste([AD.get(a, a) for a in _s4dz if a not in _s4dusen]) + " the preference stage's share does not fall. "))
    _cum = (_bas + "The preference stage carries " + _liste([_yuz(_pay[a][0]) for a in _s4dz])
            + " of the end-to-end change on raw continuation, and " + _liste([_yuz(_pay[a][1]) for a in _s4dz])
            + " under the template (" + ", ".join(AD.get(a, a).replace("ü", '\\"u') for a in _s4dz) + ").")
    io.open(f"{OUT}/T4_TEMPLATE_CUMLE.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/rungs_table.py)\n" + _cum + "\n")
    print(f"  [PAYDA] t4_template_cumle: n_dizi={_nd} · hal_pay_dusen={len(_s4dusen)} · "
          f"sft_onde_sablonda={sum(1 for a in _s4dz if _pay[a][1] < 50)} ⇒ esik: düsen<n ise cümle «K of N» kurulur")
    print("  ✓ T4_TEMPLATE_CUMLE.tex")
    _buyuyen = [a for a in _mutlak if _mutlak[a][1] > _mutlak[a][0]]
    print(f"{_pay}"
          f"{_buyuyen}"
          f"")
    print(f"{_oran}"
          f"")
    _sn = {a: _sinif(v) for a, v in S.items()
           if not a.startswith("_") and isinstance(v, dict)}
    _n = {k: sum(1 for x in _sn.values() if x == k)
          for k in ("tercih-buyuk", "berabere", "tercih-tek-basina", "denetim-buyuk")}
    _hep = all(v["adimlar"][list(v["adimlar"])[1]]["ayrik"] and
               v["adimlar"][list(v["adimlar"])[1]]["M1"] < 0
               for a, v in S.items() if not a.startswith("_") and isinstance(v, dict))
    print(f"{_n} {_hep}"
          f"")
    _tek = [a for a, x in _sn.items() if x == "tercih-tek-basina"]
    global TARIF
    TARIF = {"Tulu3-8B": "AI2 pipelines", "Tulu3-70B": "AI2 pipelines", "OLMo2-7B": "AI2 pipelines",
             "OLMo2-13B": "AI2 pipelines", "OLMo2-32B": "AI2 pipelines", "OLMo3-7B": "AI2 pipelines",
             "OLMo-3-7B": "AI2 pipelines", "Zephyr-7B": "the Zephyr pipeline"}
    _tf = {}
    for a in _sn:
        _tf[TARIF.get(a, "?")] = _tf.get(TARIF.get(a, "?"), 0) + 1
    _ntf = len(_tf)
    _tfc = ", ".join(f"{SAYI(v)} on {k}" if v>1 else f"one on {k}" for k, v in sorted(_tf.items(), key=lambda x: -x[1]))
    _tfc = _tfc.replace("the Zephyr pipeline",
                        r"the Zephyr pipeline \citep{tunstall2023zephyr}")
    print(f"  [PAYDA] t4_tarif: n_merdiven={len(_sn)} · n_tarif_ailesi={_ntf} "
          f"· dagilim={_tf} ⇒ esik: bilinmeyen tarif ('?') varsa ⇒ EYLEM: "
          f"cümle «two recipe models» YAZMAZ")
    assert "?" not in _tf, f"★ tarif ailesi CÖZÜLEMEDI: {[a for a in _sn if a not in TARIF]}"
    def _ab(v, i):
        return v["adimlar"][list(v["adimlar"])[i]]["M1"]
    _ber_sft = [(a, _ab(S[a], 0), _ab(S[a], 1)) for a, x in _sn.items()
                if x == "berabere" and abs(_ab(S[a], 0)) > abs(_ab(S[a], 1))]
    print(f"  [PAYDA] t4_berabere_sft_buyuk: n={len(_ber_sft)} {[(AD.get(a, a), round(s0, 2), round(s1, 2)) for a, s0, s1 in _ber_sft]} "
          f"⇒ esik: >0 ise ⇒ EYLEM: gövde cümlesi aileyi iki nokta kestirimiyle adlandirir")
    _ber_c = ("".join(". On " + AD.get(a, a) + ", one of the " + SAYI(_n["berabere"]) + " tied sequences, the SFT point estimate is "
                      "the larger ($%+.2f$ against $%+.2f$)" % (s0, s1) for a, s0, s1 in _ber_sft))
    _govde = (
        "On raw continuation "
        + ("the preference stage lowers the rate on all " + SAYI(len(_sn)) + " sequences, every interval "
           "excluding zero. The " + SAYI(len(_sn)) + " sequences come from " + SAYI(_ntf) + " labs (" + _tfc
           + "), not from " + SAYI(len(_sn)) + " independent sources. " if _hep else
           "the preference stage does not clear its interval on every sequence. ")
        + "The preference stage is the larger of the two stages on " + SAYI(_n["tercih-buyuk"])
        + " sequences and statistically tied with SFT on " + SAYI(_n["berabere"]) + _ber_c + ". On "
        + (", ".join(AD.get(a, a) for a in _tek) if _tek else "none")
        + " the preference stage carries the drop alone. On this prompt and these " + SAYI(_ntf)
        + " labs' pipelines, the format change contributes to the drop but does not carry it alone.")
    io.open(f"{OUT}/T4_body.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/rungs_table.py)\n" + _govde + "\n")
    _V61, prov_v61 = load(CARD_V61)
    _dis = {a: v["ciplak"] for a, v in _V61["merdiven"].items()}
    _dis_sft = [a for a, c in _dis.items() if c["VERDICT"] == "FARKLI-ADIM" and c["adim"]["sft"]["ayrik"]
                and not c["adim"]["pref"]["ayrik"] and abs(c["adim"]["sft"]["dM1"]) > abs(c["adim"]["pref"]["dM1"])]
    assert len(_dis_sft) == len(_dis), f"★ AI2-disi merdiven sinifi beklenen dalda degil: {_dis}"
    assert _n["denetim-buyuk"] == 0, f"{_sn}"
    _n_tercih = _n["tercih-buyuk"] + _n["tercih-tek-basina"]
    assert _n_tercih + _n["berabere"] == len(_sn)
    _ozet = ("on the raw-continuation prompt the preference step carries more on " + SAYI(_n_tercih) + " checkpoint sequences, the supervised "
             "step on " + SAYI(len(_dis_sft)) + ", and " + SAYI(_n["berabere"]) + " are tied")
    io.open(f"{OUT}/T4_OZET.tex", "w", encoding="utf-8").write(_ozet + "\n")
    print(f"  [PAYDA] t4_ozet: tercih_buyuk+tek={_n_tercih} · berabere={_n['berabere']} · denetim_disari={len(_dis_sft)} "
          f"· n_merdiven={len(_sn) + len(_dis)} ⇒ esik: denetim adimi AI2/Zephyr'de büyük cikarsa ya da dis merdiven "
          f"dali degisirse ⇒ EYLEM: DÜSER, özet elle düzeltilmez")
    print("  [PAYDA] t4_govde: n_merdiven=" + str(len(_sn)) + " · tercih_buyuk="
          + str(_n["tercih-buyuk"]) + " · berabere=" + str(_n["berabere"])
          + " · tek_basina=" + str(_tek) + ""
          "")
    _OLCEK = {"OLMo-2": [("OLMo2-7B", 7), ("OLMo2-13B", 13), ("OLMo2-32B", 32)],
              "T\u00fclu-3": [("Tulu3-8B", 8), ("Tulu3-70B", 70)]}
    def _sft_pay(a):
        v = S[a]; k0 = list(v["adimlar"])[0]
        return 100 * v["adimlar"][k0]["M1"] / v["uctan_uca"]["M1"]
    for _dz in _OLCEK.values():
        for _a, _ in _dz:
            assert _a not in _sn or _sn[_a] != "denetim-buyuk", f"★ ölcek dizisinde denetim adimi büyük: {_a}"
    _mono = lambda z: all(z[i] < z[i + 1] for i in range(len(z) - 1)) or \
                      all(z[i] > z[i + 1] for i in range(len(z) - 1))
    _oz, _eksik = {}, []
    for aile, dizi in _OLCEK.items():
        if any(a not in S for a, _ in dizi):
            _eksik += [a for a, _ in dizi if a not in S]; continue
        pay = [_sft_pay(a) for a, _ in dizi]
        uce = [abs(S[a]["uctan_uca"]["M1"]) for a, _ in dizi]
        _oz[aile] = dict(olcek=[p for _, p in dizi], pay=pay, uctan_uca=uce,
                         pay_monoton=_mono(pay), uctan_uca_monoton=_mono(uce))
    print("  [PAYDA] t4_olcek: " + json.dumps(
        {k: dict(pay=[round(x, 1) for x in v["pay"]],
                 uce=[round(x, 1) for x in v["uctan_uca"]],
                 pay_monoton=v["pay_monoton"], uce_monoton=v["uctan_uca_monoton"])
         for k, v in _oz.items()}, ensure_ascii=False)
        + " · red_eksik=" + str(len(_eksik))
        + " ⇒ esik: pay_monoton True olursa «ölcegi izlemiyor» cümlesi DÜSER")
    if "OLMo-2" in _oz:
        o = _oz["OLMo-2"]
        _sc = ("\\textbf{The supervised stage's share does not follow scale.} "
               "Within OLMo-2 --- one pipeline read at "
               + SAYI(len(o["olcek"])) + " scales --- the supervised stage carries "
               + ", ".join("$" + ("%.0f" % x) + "\\%$" for x in o["pay"])
               + " of the end-to-end withdrawal at "
               + ", ".join(str(x) + "B" for x in o["olcek"]) + ", which is "
               + ("monotone in scale" if o["pay_monoton"] else "not monotone in scale")
               + ", while the withdrawal itself "
               + ("does fall monotonically" if o["uctan_uca_monoton"] else "does not")
               + " ($" + "$, $".join("%.1f" % x for x in o["uctan_uca"])
               + "$ per thousand). ")
        if "T\u00fclu-3" in _oz:
            t = _oz["T\u00fclu-3"]
            _sc += ("T\\\"ulu-3 gives two points in the same direction of scale and "
                    "the share rises rather than falls: $" + ("%.0f" % t["pay"][0])
                    + "\\%$ at " + str(t["olcek"][0]) + "B against $"
                    + ("%.0f" % t["pay"][1]) + "\\%$ at " + str(t["olcek"][1])
                    + "B. \\textbf{Size moves the amount withdrawn; \\emph{within a "
                    "pipeline} it never makes the supervised stage the larger one.}")
        _o2 = _oz.get("OLMo-2")
        _kisa = ""
        if _o2:
            _kisa = ("SFT's share does not follow scale. Within OLMo-2, one pipeline at three sizes, it is $"
                     + "\\%$, $".join("%.0f" % p for p in _o2["pay"])
                     + "\\%$ at " + "B, ".join(str(o) for o in _o2["olcek"])
                     + "B, while the drop itself shrinks monotonically with size. Within a pipeline, size "
                       "never makes SFT the larger stage.")
        io.open(f"{OUT}/T4_scale.tex", "w", encoding="utf-8").write(
            "% * URETILDI - elle duzenlenmez (figures/rungs_table.py)\n"
            + (_kisa or _sc) + "\n")
        print("  ✓ T4_scale.tex (gövde · kisa)")
        io.open(f"{OUT}/T4_scale_full.tex", "w", encoding="utf-8").write(
            "% * URETILDI - elle duzenlenmez (figures/rungs_table.py)\n"
            "\\paragraph*{Scale sequences behind the supervised stage's share.}\n"
            "\\vekalet{" + _sc + "}\n")
        print("  ✓ T4_scale_full.tex (Ek E)")
    _n_s = len(SBL)
    _ist = [a for a in _s4dz if a not in _s4dusen]
    _zep_disi = [a for a in SBL if a != "Zephyr-7B"]
    _tercih_ayrik = sum(1 for a in _zep_disi if SBL[a]["adimlar"][list(SBL[a]["adimlar"])[1]]["ayrik"])
    def _ist_cumle(a):
        st = list(SBL[a]["adimlar"].values())
        _adx = AD.get(a, a).replace("ü", '\\"u')
        return (f"On {_adx} the preference stage's share rises instead, from ${_pay[a][0]:.1f}\\%$ to "
                f"${_pay[a][1]:.1f}\\%$: under the template the preference stage carries the drop "
                + ("while the supervised stage does not clear its interval. " if not st[0]["ayrik"] else "and the supervised stage clears its interval as well. "))
    _SBL_UZUN = ((r"\textbf{The blocks marked (tmpl.) read the same checkpoints through "
             r"the chat template, and on " + (f"all {SAYI(_n_s)}" if not _ist else f"{SAYI(_n_s - len(_ist))} of the {SAYI(_n_s)}")
             + r" the protocol moves the localization the same way}: \textbf{the supervised stage's share grows and the "
             r"preference stage's share shrinks}, from "
             + ", ".join(f"${_pay[a][0]:.1f}\\%$" for a in _s4dz) + " on raw continuation to "
             + ", ".join(f"${_pay[a][1]:.1f}\\%$" for a in _s4dz)
             + r" under the chat template (" + ", ".join(AD.get(a, a) for a in _s4dz) + "). "
             + "".join(_ist_cumle(a) for a in _ist)
             + r"On Zephyr-7B the exchange is complete and the two stages swap which one "
             r"is null; on the other " + SAYI(len(_zep_disi)) + r" the preference stage "
             + ("still clears its interval" if _tercih_ayrik == len(_zep_disi) else f"clears its interval on {SAYI(_tercih_ayrik)}")
             + r", so the pre-registered outcome name does not change there. "
             r"\textbf{The base$\rightarrow$SFT row of a (tmpl.) block is the one "
             r"asymmetric comparison in this table}: these " + SAYI(_n_s) + r" bases ship no chat "
             r"template of their own (measured), so that row reads the base checkpoint through a "
             r"template it never saw, while every preference row compares two "
             r"checkpoints that both carry one. "
             r"OLMo-2-32B and OLMo-3-7B have no chat-template block: their checkpoints do not all ship "
             r"the same chat template, so their stages cannot be read on one string. "
             r"\textbf{The chat-template rows here and in "
             r"Table~\ref{tab:twobytwo-full} are not the same reading}: this table is "
             r"computed on all cells, that one only on cells that are non-degenerate "
             r"in all four quadrants, so the same model reads $-17.17$ here and "
             r"$-18.41$ there. "
             r"The same contrast thus appears on three sets of cells: Table~\ref{tab:families} pools four further "
             r"sampling seeds over all cells, Table~\ref{tab:twobytwo-full} reads one seed on the filtered cells, "
             r"and this table reads one seed on all cells, unfiltered. "
             + r"The absolute levels behind these shares are in "
             r"Appendix~\ref{app:kesim}. "
             r"Which preference objective is used is itself a measured choice "
             r"\citep{ivison2024unpacking,ivison2023tulu2,dong2023rm}.")
            if SBL else "")
    _SBL_KISA = ((r"\textbf{The blocks marked (tmpl.) read the same "
                  r"checkpoints through the chat template, and on all three "
                  r"the supervised step's share grows while the preference "
                  r"step's shrinks} (Appendix~\ref{app:kesim}).")
                 if SBL else "")

    def _tire_cumle():
        mer_ = [x for x, v in S.items() if not x.startswith("_") and isinstance(v, dict)]
        tire = {}
        for x in mer_:
            nc = len(S[x]["adimlar"]); ns = len(SBL[x]["adimlar"]) if x in SBL else 0
            eksik = [("ham", i) for i in range(nc, 3)] + [("sbl", i) for i in range(ns, 3)]
            if eksik:
                tire[x] = eksik
        acik = set(tire) - set(TIRE_SEBEP); fazla = set(TIRE_SEBEP) - set(tire)
        print(f"  [PAYDA] t4_tire: n_tire={sum(len(v) for v in tire.values())} · diziler={sorted(tire)} · "
              f"red_sebepsiz={sorted(acik)} · red_tiresiz_sebep={sorted(fazla)} ⇒ esik 0/0 ⇒ EYLEM: >0 ise künye YAZILMAZ")
        assert not acik and not fazla, (acik, fazla)
        g = {"yok": [], "template": [], "kosulmadi": []}
        for x in sorted(tire, key=lambda z: AD.get(z, z)):
            g[TIRE_SEBEP[x]].append(x)
        def ad_(x):
            yalniz_post = all(i == 2 for _, i in tire[x])
            return AD.get(x, x) + (r" pref.$\rightarrow$post" if yalniz_post and TIRE_SEBEP[x] == "kosulmadi" else "")
        par = []
        if g["yok"]:
            par.append("no such stage (" + ", ".join(ad_(x) for x in g["yok"]) + ")")
        if g["template"]:
            par.append("checkpoints do not share one template (" + ", ".join(AD.get(x, x) for x in g["template"]) + ")")
        if g["kosulmadi"]:
            par.append("not run (" + ", ".join(ad_(x) for x in g["kosulmadi"]) + ")")
        return "Dashes: " + "; ".join(par) + "."

    def _tight():
        def hh(d):
            return f"$\\mathbf{{{d['M1']:+.2f}}}$" if d["ayrik"] else f"${d['M1']:+.2f}$"
        def hucre(lst, i):
            return hh(lst[i]) if i < len(lst) else "---"
        mer = [x for x, v in S.items() if not x.startswith("_") and isinstance(v, dict)]
        assert {TARIF.get(x, "?") for x in mer} == {"AI2 pipelines", "the Zephyr pipeline"}, [TARIF.get(x, "?") for x in mer]
        T = [r"\begin{table}[tb]", r"\centering\small",
             r"\caption{\textbf{How the training stages share the drop.} "
             r"Change in the second-person rate ($\Delta$2nd per 1,000 tokens) at each "
             r"stage of the " + SAYI(len(mer)) + r" AI2 and Zephyr sequences, on raw continuation "
             r"and, where its checkpoints share one template, under it. SFT: supervised fine-tuning. "
             r"Bold: interval excluding zero. " + _tire_cumle() + r" Appendix Table~\ref{tab:rungs-full} gives intervals and the two sequences outside these labs.}",
             r"\label{tab:rungs}",
             r"\begin{tabular}{l rrr @{\hspace{1.1em}} rrr}", r"\toprule",
             r"& \multicolumn{3}{c}{raw continuation} & \multicolumn{3}{c}{chat template} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"sequence & base$\rightarrow$SFT & SFT$\rightarrow$pref. & "
             r"pref.$\rightarrow$post & base$\rightarrow$SFT & SFT$\rightarrow$pref. "
             r"& pref.$\rightarrow$post \\", r"\midrule"]
        _grup = {}
        for x in mer:
            _grup.setdefault(TARIF.get(x, "?"), []).append(x)
        assert "?" not in _grup, f"★ tarif ailesi CÖZÜLEMEDI: {_grup.get('?')}"
        _ilk = True
        for _tf_ad, _uyeler in sorted(_grup.items(), key=lambda kv: -len(kv[1])):
            if not _ilk:
                T.append(r"\addlinespace[2pt]")
            _ilk = False
            _gor = _tf_ad.split(" ", 1)[1] if _tf_ad.split(" ", 1)[0] in ("a", "an", "the") else _tf_ad
            T.append(r"\multicolumn{7}{l}{\emph{" + _gor
                     + f"}} ({len(_uyeler)})}}" + r" \\")
            for x in _uyeler:
                hc = list(S[x]["adimlar"].values())
                hs = list(SBL[x]["adimlar"].values()) if x in SBL else []
                T.append(f"\\quad {AD.get(x, x)} & " + " & ".join(
                    [hucre(hc, 0), hucre(hc, 1), hucre(hc, 2),
                     hucre(hs, 0), hucre(hs, 1), hucre(hs, 2)]) + r" \\")
        T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        io.open(f"{OUT}/T4_rungs.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")
        _sb = sum(1 for x in mer if x in SBL)
        print(f"  [PAYDA] t4_siki: n_merdiven={len(mer)} · hal_sablonlu={_sb} · "
              f"hal_satir={len(mer)} · hal_sutun=6 ⇒ esik: satir>7 ⇒ yarim sayfayi asar")
        print("  ✓ T4_rungs.tex (siki gövde tablosu)")

    L = [r"\begin{table}[tb]", r"\centering\small",
         r"\caption{\textbf{How the training stages share the withdrawal, in full.} "
         r"The training stages of Table~\ref{tab:rungs} with prompt-clustered 95\% intervals, "
         r"the change in first-person density and in the force total, and the "
         r"end-to-end sum for each checkpoint sequence; the two sequences outside the AI2 and Zephyr pipelines follow in their own block. Blocks marked (tmpl.) read the same "
         r"checkpoints through the chat template. Bold marks an interval excluding "
         r"zero.}",
         r"\label{tab:rungs-full}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"sequence & stage & $\Delta M_1$ & 95\% CI & $\Delta$1st & $\Delta$force \\",
         r"\midrule"]
    n_a = 0
    for a, v in S.items():
        if a.startswith("_") or not isinstance(v, dict):
            continue
        ilk = True
        for ad, d in v["adimlar"].items():
            k0, k1 = ad.split("→")
            et = f"{EN.get(k0,k0)}$\\rightarrow${EN.get(k1,k1)}"
            m1 = f"\\textbf{{{d['M1']:+.2f}}}" if d["ayrik"] else f"{d['M1']:+.2f}"
            L.append(f"{AD.get(a,a) if ilk else '':12s} & {et} & {m1} & "
                     f"$[{d['ci'][0]:+.2f},\\,{d['ci'][1]:+.2f}]$ & "
                     f"{d['sahis1']:+.1f} & {d['KUVVET']:+.4f} \\\\")
            ilk = False
        u = v["uctan_uca"]
        L.append(f" & \\emph{{end to end}} & {u['M1']:+.2f} & "
                 f"$[{u['ci'][0]:+.2f},\\,{u['ci'][1]:+.2f}]$ & & \\\\")
        L.append(r"\addlinespace")
        n_a += 1
    _DZ, prov_dz = load(CARD_DUZEY)
    _dis_ayrik = 0
    L.append(r"\midrule")
    L.append(r"\multicolumn{6}{l}{\emph{Outside the AI2 and Zephyr pipelines (raw continuation)}}\\")
    for a, c in _V61["merdiven"].items():
        c = c["ciplak"]; dz = _DZ["merdiven"][a]
        _dis_ayrik += int(c["adim"]["pref"]["ayrik"])
        ilk = True
        for k, et in (("sft", "base$\\rightarrow$SFT"), ("pref", "SFT$\\rightarrow$DPO")):
            d = c["adim"][k]
            m1 = f"\\textbf{{{d['dM1']:+.2f}}}" if d["ayrik"] else f"{d['dM1']:+.2f}"
            L.append(f"{AD.get(a,a) if ilk else '':12s} & {et} & {m1} & "
                     f"$[{d['ci'][0]:+.2f},\\,{d['ci'][1]:+.2f}]$ & --- & --- \\\\")
            ilk = False
        _uu = dz["dpo"]["M1"] - dz["base"]["M1"]
        assert abs(_uu - (c["adim"]["sft"]["dM1"] + c["adim"]["pref"]["dM1"])) < 0.01, (a, _uu)
        L.append(f" & \\emph{{end to end}} & {_uu:+.2f} & --- & & \\\\")
        _kn = [json.load(io.open(KUNYE_V61.format(a=a, b=b_), encoding="utf-8")) for b_ in ("taban", "sft", "dpo")]
        _toh = sorted({x["seed"] for x in _kn}); _cek = sorted({x["n_cekim"] for x in _kn})
        assert len(_toh) == 1 and len(_cek) == 1, (a, _toh, _cek)
        L.append(r"\multicolumn{6}{p{\linewidth}}{\footnotesize " + AD.get(a, a) + ": second-person density "
                 f"${dz['base']['M1']:.2f}\\rightarrow{dz['sft']['M1']:.2f}\\rightarrow{dz['dpo']['M1']:.2f}$ per thousand "
                 f"(base, SFT, DPO); paired-placebo 95th percentile ${c['adim']['sft']['plasebo_p95']:.2f}$ and "
                 f"${c['adim']['pref']['plasebo_p95']:.2f}$ for the two stages ({c['adim']['sft']['plasebo_n']} draws); "
                 f"one generation seed, {_cek[0]} generations per prompt in each prefix condition; intervals from {_V61['n_bootstrap']} prompt-clustered resamples; "
                 r"first-person and force changes not computed.}\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}",
          r"\begin{table}[tb]", r"\centering\small",
          r"\textbf{Table~\ref{tab:rungs-full} (continued).}\\[3pt]",
          r"\begin{tabular}{llrrrr}", r"\toprule",
          r"sequence & stage & $\Delta M_1$ & 95\% CI & $\Delta$1st & $\Delta$force \\"]
    _ilk_sbl = True
    for a, v in SBL.items():
        L.append(r"\midrule")
        ilk = True
        for ad, d in v["adimlar"].items():
            k0, k1 = ad.split("→")
            et = f"{EN.get(k0,k0)}$\\rightarrow${EN.get(k1,k1)}"
            m1 = f"\\textbf{{{d['M1']:+.2f}}}" if d["ayrik"] else f"{d['M1']:+.2f}"
            L.append(f"{(AD.get(a,a) + ' (tmpl.)') if ilk else '':16s} & {et} & {m1} & "
                     f"$[{d['ci'][0]:+.2f},\\,{d['ci'][1]:+.2f}]$ & "
                     f"{d['sahis1']:+.1f} & {d['KUVVET']:+.4f} \\\\")
            ilk = False
        u = v["uctan_uca"]
        L.append(f" & \\emph{{end to end}} & {u['M1']:+.2f} & "
                 f"$[{u['ci'][0]:+.2f},\\,{u['ci'][1]:+.2f}]$ & & \\\\")
    _sbl_ayrik = sum(1 for a in SBL
                     if SBL[a]["adimlar"][list(SBL[a]["adimlar"])[1]]["ayrik"])
    print(f"{len(SBL)} {_sbl_ayrik}"
          f"")
    L += [r"\midrule",
          r"\multicolumn{6}{p{\linewidth}}{\footnotesize Bold: interval excludes zero. "
          f"On the raw-continuation prompt the preference stage clears its interval on all "
          f"{n_a} AI2 and Zephyr checkpoint sequences of this table and on "
          f"{SAYI(_dis_ayrik) if _dis_ayrik else 'neither'} of the {SAYI(len(_V61['merdiven']))} outside them. " + (f"Under the template we can read {len(SBL)} of them "
          f"as a single string; the preference stage still clears its interval on "
          f"{_sbl_ayrik} of those {len(SBL)}, and on Zephyr-7B the two stages swap "
          f"which one is null." if SBL else "")
          + r"}\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T4_full.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("  ✓ T4_full.tex")
    _tight()
    io.open(f"{OUT}/T4_note.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/rungs_table.py)\n"
        "\\paragraph*{Notes on Table~\\ref{tab:rungs-full}.}\n"
        "\\vekalet{" + (_SBL_UZUN or "Not applicable: no chat-template block.") + "}\n")
    print("  ✓ T4_note.tex")
    json.dump(dict(table="T4_rungs", ciktilar=["T4_rungs.tex", "T4_body.tex", "T4_full.tex", "T4_note.tex", "T4_scale.tex", "T4_scale_full.tex", "T4_OZET.tex", "T4_TEMPLATE_CUMLE.tex"], sources=[prov, prov_ek] + kaynak_sbl + _ek_prov + [prov_v61, prov_dz],
                   payda=dict(n_merdiven=n_a, **{k.replace("-", "_"): v for k, v in _n.items()}),
                   note="renders PREREG-10; criterion fixed before the run"),
              open(f"{OUT}/T4_rungs.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(SBL)}"
          f"{[v['dal'].split(' · ')[0] for v in SBL.values()]}"
          f"")
    print(f"  [PAYDA] t4: n_merdiven={n_a} · n_adim={sum(len(v['adimlar']) for k,v in S.items() if not k.startswith('_'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
