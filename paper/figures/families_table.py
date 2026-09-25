#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
from style import OUT, load, t1_adi
import elicited16_loader as E16
import urial_verdict_loader

A_CARD = "results/PREREG9_A_OLCUM_2026-08-31.json"
SEED_CARD = "results/four_seed_reading_2026-09-01.json"
ZEMIN_CARD = "results/substrate_protocol_2026-09-06.json"
ONEK_CARD = "results/v27_urial_prefix_density_2026-09-09.json"
SOY_ORG = {"allenai": "AI2", "Qwen": "Alibaba", "google": "Google",
           "meta-llama": "Meta", "mistralai": "Mistral"}
PANEL = __DNH_DATA__ + "/c1_panel"


def soy_diskten(aile, zeminler):
    import glob as _g, json as _j, os as _o, re as _re
    aday = list(zeminler)[::-1] + ["instruct", "base"]
    yollar = [f"{PANEL}/{aile}/{z}/uretim_kunye.json" for z in aday]
    yollar += sorted(_g.glob(f"{PANEL}/{aile}/*/uretim_kunye.json"))
    for y in yollar:
        if not _o.path.exists(y):
            continue
        m = _re.search(r"models--([^-][^/]*?)--",
                       _j.load(open(y, encoding="utf-8")).get("model", ""))
        if m:
            return SOY_ORG.get(m.group(1), m.group(1))
    return None


def ayrik(f):
    return f["gozlenen"] < 0 and (f["ci"][1] < 0 or f["ci"][0] > 0)


def main():
    A, pa = load(A_CARD); B, pb = E16.yukle()
    UX, pux = urial_verdict_loader.yukle(urial_verdict_loader.CARD_2048)
    Xail, Xsay, Xkap = urial_verdict_loader.kiyas(UX, "ii")
    Usay = Xsay
    _kirp = UX["kiyas"]["ii"]["n_hucre_kirpik"]
    _kirpad = sorted(a for a, v in Xail.items() if v.get("AD") == "ÖLCÜLEMEZ-KIRPMA")
    _TVk, _ptv = load("results/v97_generation_tavani_2026-09-19.json")
    _TVe = _TVk["kok"]["elicit"]
    _doy = {a for a, v in _TVe.items() if v["instruct"]["pay_tavan"] >= 0.10}
    Ab = {k: v for k, v in A.items() if isinstance(v, dict) and v.get("sinif") == "BIRINCIL"}
    raf = {r["aile"] for r in B["_kunye"]["raf"]}
    kosulmadi = set(B["_kunye"]["kosulmadi"])
    satir = []
    for k, v in Ab.items():
        k0, k1 = v["kontrast"].split("→"); t = v["basamak_tablosu"]
        b, i = t[k0]["M1"], t[k1]["M1"]
        f = v["fark"]["M1"]
        r = dict(ad=k, soy=soy_diskten(k, (k0, k1)), ab=b, ai=i,
                 ad_pct=100 * (i - b) / b, a_ayrik=ayrik(f), ci=f["ci"],
                 s1=v["sahis1_1k"]["d"], m1d=f["gozlenen"],
                 a_plasebo=abs(f["gozlenen"]) > v["plasebo_ESLI_M1"]["p95"])
        if k in B and isinstance(B[k], dict) and "fark" in B[k]:
            bt = B[k]["basamak_tablosu"]; bf = B[k]["fark"]["M1"]
            r.update(bb=bt["base"]["M1"], bi=bt["instruct"]["M1"],
                     bd=bf["gozlenen"], b_ayrik=ayrik(bf), b_not=None,
                     b_plasebo=bool(bf["plasebo_ustu"]))
        else:
            r.update(bb=None, b_plasebo=False,
                     b_not=("DEJENERE-RAF" if k in raf else
                            ("KOSULMADI" if k in kosulmadi else "---")))
        u = Xail.get(k)
        if u is not None and u.get("hal") == "ÖLCÜLDÜ" and u["AD"] != "ÖLCÜLEMEZ-KIRPMA":
            r.update(ub=u["M1_taban"], ui=u["M1_hizali"], ud=u["dM1"],
                     u_ayrik=ayrik({"gozlenen": u["dM1"], "ci": u["ci"]}),
                     u_plasebo=(u["plasebo_frac"] <= 0.05), u_ad=u["AD"])
        else:
            r.update(ub=None, u_plasebo=False,
                     u_ad=((u or {}).get("AD") if (u or {}).get("AD") == "ÖLCÜLEMEZ-KIRPMA"
                           else (u or {}).get("hal", "BACAK-EKSIK")))
        satir.append(r)
    _DZ = "results/four_seed_level_2026-09-16.json"
    _dort = None
    _ptk = None
    _A4dd, _pdd = {}, None
    _A4d = {}
    if os.path.exists(f"{__DNH_ROOT__}/{_DZ}"):
        _Draw, _pdz = load(_DZ)
        _D = {k: v for k, v in _Draw.items() if not k.startswith("_")}
        _tam = [v for v in _D.values() if v.get("hal") == "m=4" and "havuz_M1_taban" in v]
        _tut = sum(1 for v in _tam if v["havuz_dM1"] < 0 and v["ayrik"] and v["plasebo_ustu"])
        _kap = len(_tam) == len(satir) == 16 and _tut == 16 and all(r["ad"] in _D for r in satir)
        print(f"{len(_tam)} {_tut} {len(satir)}"
              f"")
        _TKd, _ptk = load("results/v90_tez_kume_2026-09-17.json")
        _A4d = _TKd["aile"] if _TKd["alet"]["hal"] == "ESDEGER" and len(_TKd["aile"]) == 16 else {}
        _A4dd, _pdd = {}, None
        _TBb, _ptb = {}, None
        if os.path.exists(__DNH_ROOT__ + "/results/v104_base_birinci_2026-09-23.json"):
            _TBk, _ptb = load("results/v104_base_birinci_2026-09-23.json")
            _TBb = _TBk["aile"] if not _TBk["kapi"]["red"] and len(_TBk["aile"]) == 16 else {}
        print(f"  [PAYDA] t1_taban_birinci: n={len(_TBb)}/16 ⇒ esik 16 ⇒ EYLEM: degilse taban-birinci sütunu YAZILMAZ")
        if os.path.exists(__DNH_ROOT__ + "/results/v97_t1_dogrudan_birinci_2026-09-19.json"):
            _DDd, _pdd = load("results/v97_t1_dogrudan_birinci_2026-09-19.json")
            _A4dd = _DDd["aile"] if _DDd["alet"]["hal"] == "ESDEGER" and len(_DDd["aile"]) == 16 else {}
        print(f"{len(_A4dd)}")
        if _kap:
            for r in satir:
                v = _D[r["ad"]]
                r["tek"] = dict(ab=r["ab"], ai=r["ai"], m1d=r["m1d"], ci=list(r["ci"]), a_ayrik=bool(r["a_ayrik"]), s1=r["s1"])
                if r["ad"] in _TBb:
                    r["s1b"] = _TBb[r["ad"]]["taban_1k"]
                if r["ad"] in _A4dd:
                    r["s1"] = _A4dd[r["ad"]]["d1st_dogrudan"]; r["s1_ci"] = _A4dd[r["ad"]]["ci"]
                elif r["ad"] in _A4d:
                    r["s1"] = _A4d[r["ad"]]["d1st"]
                r.update(ab=v["havuz_M1_taban"], ai=v["havuz_M1_hizali"], m1d=v["havuz_dM1"],
                         ci=list(v["ci"]), a_ayrik=bool(v["ayrik"]), a_plasebo=bool(v["plasebo_ustu"]),
                         ad_pct=100 * v["havuz_dM1"] / v["havuz_M1_taban"])
            _dort = dict(n=16, card=_pdz["yol"], sha256_16=_pdz["sha256_16"])
    satir.sort(key=lambda r: r["m1d"])
    nA = sum(r["a_ayrik"] for r in satir)
    nB = sum(1 for r in satir if r["bb"] is not None and r["b_ayrik"])
    nBp = sum(1 for r in satir if r["bb"] is not None)
    n1 = sum(1 for r in satir if r["s1"] < 0)
    nb = sum(1 for r in satir if abs(r["s1"]) > abs(r["m1d"]))
    nAp = sum(1 for r in satir if r["a_ayrik"] and r["a_plasebo"])
    nBpl = sum(1 for r in satir if r["bb"] is not None and r["b_ayrik"] and r["b_plasebo"])
    nUpl = sum(1 for r in satir if r["ub"] is not None and r["u_ayrik"] and r["u_plasebo"])
    nU = sum(1 for r in satir if r["ub"] is not None)
    nUd = sum(1 for r in satir if r["ub"] is not None and r["u_ayrik"])
    n_soysuz = sum(1 for r in satir if not r["soy"])
    _u_kirpik = [r["ad"] for r in satir if r["u_ad"] == "ÖLCÜLEMEZ-KIRPMA"]
    _u_yok = [r["ad"] for r in satir if r["ub"] is None
              and r["u_ad"] != "ÖLCÜLEMEZ-KIRPMA"]
    print(f"{len(satir)} {nU}"
          f"{len(_u_kirpik)} {_u_kirpik}"
          f"{len(_u_yok)} {_u_yok}")
    if _u_yok:
        print("  ★ URIAL BACAGI EKSIK: %s ⇒ HATA (§8)" % _u_yok); return 3
    if n_soysuz:
        print(f"  ★ SOY CÖZÜLEMEDI: {[r['ad'] for r in satir if not r['soy']]} ⇒ HATA (§8)")
        return 3
    print(f"{len(satir)} {nA} {len(satir)}"
          f"{nBp} {nB} {n1} {nb}"
          f"{nU} {Xsay['asagi']} {Xsay['bicim']}"
          f"{Xsay['ters']} {Xsay.get('olculemez_kirpma',0)}"
          f"{Usay['asagi']} {Usay['bicim']} {Usay['ters']}"
          f"{Xkap['bacak_eksik']}"
          f"{nAp} {nBpl} {nUpl}"
          f"{len(_doy)}"
          f""
          f"{len(satir)}")

    try:
        import four_seed_loader as _TK
        _S, _pts = _TK.oku(); pt = _pts[0]; pt_ek = _pts[1]
        _n4 = sum(1 for k, v in _S.items() if not k.startswith("_") and v.get("hal") == "m=4")
        _tek = {k for k, v in _S.items()
                if not k.startswith("_") and v.get("hal") != "m=4"}
        _tutan = sum(1 for k, v in _S.items() if not k.startswith("_") and v.get("hal") == "m=4"
                     and v.get("ayrik") and v.get("plasebo_ustu") and v.get("havuz_dM1", 0) < 0)
        if _tutan == _n4 and not _tek:
            _nA_1 = sum(1 for r in satir if r.get("tek") and r["tek"]["a_ayrik"])
            _n1_1 = sum(1 for r in satir if r.get("tek") and r["tek"]["s1"] < 0)
            _nb_1 = sum(1 for r in satir if r.get("tek") and abs(r["tek"]["s1"]) > abs(r["tek"]["m1d"]))
            _ayni = (_n4 == nA == _nA_1 and n1 == _n1_1 and nb == _nb_1)
            print(f"{_nA_1} {nA} {_n1_1} {n1}"
                  f"{_nb_1} {nb}")
            _seed = ("These counts on raw continuation are the same at one sampling seed and at four pooled." if _ayni else
                      f"At the original seed the counts are {_nA_1}, {_n1_1} and {_nb_1} of {len(satir)}.")
        elif _tutan == _n4:
            _seed = (f"The panel is read at one sampling seed; the four-seed re-read holds in "
                      f"{_n4}/{len(satir)} models and the {len(_tek)} marked $^{{\\ddagger}}$ are not yet re-read.")
        else:
            _seed = (f"The panel is read at one sampling seed; the four-seed re-read covers "
                      f"{_n4}/{len(satir)} models and holds in {_tutan}; the {len(_tek)} marked "
                      f"$^{{\\ddagger}}$ are not yet re-read.")
        print(f"  [PAYDA] t1_seed_tutan: m4={_n4} · tutan={_tutan} ⇒ esik tutan==m4 ⇒ EYLEM: degilse cümle «covers … holds in» bicimine gecer")
    except Exception:
        _n4 = None; _tek = set(); pt = dict(yol=SEED_CARD, sha256_16=None, hal="OKUNAMADI"); pt_ek = dict(yol="results/SEED_UC_AILE_OKUMA_2026-09-15.json", sha256_16=None, hal="OKUNAMADI")
        _seed = "Four-seed re-read: \\textsc{count unavailable}."
    print(f"  [PAYDA] t1_seed_sayaci: hal_m4={_n4} · n_aile={len(satir)}")
    _bicim_ad = sorted(r["ad"] for r in satir if str(r.get("u_ad") or "").startswith("BICIM"))
    assert len(_bicim_ad) == Xsay["bicim"], (_bicim_ad, Xsay["bicim"])
    _bicim_c = (" and in " + " and ".join(_bicim_ad) + " the difference appears only when each output is in "
                "its own format" if _bicim_ad else "")
    _soylar = sorted({r["soy"] for r in satir})
    _soy_dusen = [x for x in _soylar if all(r["a_ayrik"] and r["m1d"] < 0 for r in satir if r["soy"] == x)]
    print(f"{len(_soylar)} {_soylar} {len(_soy_dusen)}"
          f"")
    _ozet = (f"On raw continuation the second-person rate falls in {nA} of {len(satir)} models, "
             f"{len(_soy_dusen)} of {len(_soylar)} lineages, every "
             f"interval excluding zero (Table~\\ref{{tab:families}}). This is the pre-registered test, and its "
             f"bar was 12 of 16 with a positive control on the same pipeline. The fall has an interval excluding zero "
             f"in {nB} of the {len(satir)} models on ELICIT-99, which scores {nBp}. Under the in-context assistant prompt on the identical "
             f"string it has an interval excluding zero in {nUd} of {nU}, where {Xsay['ters']} of the other {nU - nUd} rise{_bicim_c}. "
             + (f"The first-person rate falls in every model and, in absolute terms, further "
                f"than the second-person rate. "
                if (n1 == len(satir) and nb == len(satir)) else
                f"The first-person rate falls in {n1} of {len(satir)} models and, in absolute terms, further "
                f"than the second-person rate in {nb} of {len(satir)}. ")
             + f"{_seed}")
    _ozet_ek = (f"The paired within-prompt placebo replaced the registered one "
                f"after the first count was read. It is a required control outside the bar, which names only the "
                f"interval, and on top of the interval it is cleared by {nAp}/{len(satir)} on the debate prompts, "
                f"{nBpl}/{nBp} on ELICIT-99 (the placebo band is much wider there) and "
                f"{nUpl}/{nU} on the in-context assistant prompt (identical string). ")

    _ubmin, _ubmax, _ubn, _ubdis = urial_verdict_loader.bant(Xail, "M1_taban")
    _O, po = load(ONEK_CARD)
    _orn = _O["cevaplar_uc"]; _bas = _O["baslik"]; _but = _O["onek_butun"]
    print(f"{_orn} {_bas} {_but}"
          f"")
    _ust = sum(1 for a, v in Xail.items()
               if v.get("AD") != "ÖLCÜLEMEZ-KIRPMA" and v["M1_taban"] > _but)
    _marj = _ubmin - _but
    _kirps = f"{_kirp:,}".replace(",", "{,}")
    _urial = (
        r"$^{\ast}$On the in-context assistant prompt both checkpoints are read on the identical "
        r"string, so the only difference between them is the weights, and the "
        f"columns are read on all sixteen prefix conditions (${_kirps}$ cells truncate).")
    _urial_ek = (
        r"$^{\ast}$\textbf{The in-context assistant prompt, in full.} Both checkpoints are read on "
        r"the identical string, so the only difference between them is the weights. "
        f"The task lifts every base into ${_ubmin:.1f}$--${_ubmax:.1f}$ per thousand "
        f"across the {_ubn} scored models, above the ${_orn:.1f}$ of the three "
        f"example answers; the lowest base sits only {_marj:.2f} per thousand above "
        f"the ${_but:.2f}$ of the prompt taken whole (its instruction header alone "
        f"runs at ${_bas:.1f}$) --- a margin below this design's own resolution --- "
        r"so the copying account is excluded at the answers and left open at the "
        r"whole prefix. "
        f"The contrast there is {Xsay['asagi']} down, {Xsay['ters']} up and "
        f"{Xsay['bicim']} "
        r"\textsc{format-only} (a shift below the model's minimum detectable effect) "
        f"across {Xkap['n_olculen']} of {Xkap['n_aile']} models with "
        f"{Xkap['bacak_eksik']} missing generation sets. "
        + (f"{', '.join(t1_adi(a) for a in _kirpad)} "
           + ("exceeds" if len(_kirpad) == 1 else "exceed")
           + " the window on its own tokenizer even at single strength and "
           + ("is" if len(_kirpad) == 1 else "are")
           + r" reported \textsc{unmeasurable-truncated} rather than scored. "
           if _kirpad else "")
        + r"The reading against each aligned "
        r"checkpoint's own template is in Table~\ref{tab:twobytwo-full}.")

    _dsat = sorted(_doy, key=lambda a: float(a.rsplit("-", 1)[-1].rstrip("B")))
    _pc = lambda v: f"{100*v:.0f}"
    _ik = [f"{a} ({_pc(_TVe[a]['instruct']['pay_tavan'])}\\% aligned, {_pc(_TVe[a]['base']['pay_tavan'])}\\% base)" for a in _dsat]
    _kesik = (
        (r"$^{\S}$These rows reach the $1{,}024$-token cap on ELICIT-99 in at least a tenth of aligned draws: "
         + (", ".join(_ik[:-1]) + " and " + _ik[-1] if len(_ik) > 1 else _ik[0])
         + r". Part of their density there is read on truncated text (Appendix Table~\ref{tab:tavan}). "
         r"We print the stamp rather than drop the rows, and we do not claim a "
         r"direction for the bias it leaves.") if _dsat else "")

    _Z, pz = load(ZEMIN_CARD)
    _zn, _ze = _Z["NOTR_PROTOKOL"], _Z["ELICIT_PROTOKOL"]
    _nfark, _nayni = len(_ze["farkli_protokol"]), len(_ze["ayni_protokol"])
    print(f"  [PAYDA] t1_protokol: notr_onekte_template_izi={_zn['onekte_template_izi']} "
          f"(yetenek={_zn['kunyede_template_yetenek']}) · elicited farkli={_nfark} "
          f"ayni={_nayni} ⇒ esik: farkli>0 ⇒ EYLEM: sag sutuna hancer serhi")
    _prot = (
        r"\textsc{protocol}: on the debate prompts every checkpoint, base and "
        r"instruct alike, is prompted with the same raw-continuation text scaffold; no chat "
        f"template is applied on either checkpoint ({_zn['onekte_template_izi']} of the "
        f"{_zn['n_okunan']} "
        r"generation sets stored on this protocol, which also cover the checkpoint sequences, "
        r"the training runs and the four models of Appendix~\ref{app:breadth}, "
        r"carries a template marker in its prompt, although "
        f"{_zn['kunyede_template_yetenek']} of them come from checkpoints that ship one), so that "
        r"contrast is not confounded with a change of prompt format. "
        r"$^{\dagger}$On ELICIT-99 the protocol follows each checkpoint: "
        r"a checkpoint with a chat template is prompted through it and one without "
        f"through the raw-continuation scaffold, so in {_nfark} of the {_nfark + _nayni} measured "
        r"models the two checkpoints differ in prompt format as well as in training, and "
        r"\textbf{a format change is not separated from alignment on this prompt set}. "
        + {1: "The one model", 2: "The two models",
           3: "The three models"}.get(_nayni, f"The {_nayni} models")
        + " whose checkpoints share a format are "
        + ", ".join(sorted(_ze["ayni_protokol"],
                           key=lambda a: float(a.rsplit("-", 1)[-1].rstrip("B")))
                    ) + ".")
    _sinif = sorted({r["b_not"] for r in satir if r["bb"] is None})
    _ad = {"DEJENERE-RAF": r"\textsc{d-shelf} = degeneracy shelf "
                           r"(distinct-4 $< 0.60$)",
           "KOSULMADI": r"\textsc{not-run} = compute gate"}
    _legend = ("Models with no elicited entry carry their measured reason: "
               + ", ".join(_ad.get(c, c) for c in _sinif) + ". ") if _sinif else ""
    L = []
    L.append(r"\begin{table}[tb]")
    L.append(r"\centering\scriptsize\setlength{\tabcolsep}{2.6pt}")
    L.append(r"\caption{\textbf{Sixteen models on the debate prompts, ELICIT-99 and the in-context assistant prompt.} "
             r"$M_1$ is second-person forms per thousand tokens and "
             + (r"$\Delta$1st the first-person shift on the debate prompts, counted directly" if _A4dd else
              r"$\Delta$1st the derived first-person shift on the debate prompts") +
             (r" (the debate-prompt columns pool four further sampling seeds). " if _dort and _A4d else ". ") +
             r"\textsc{disj.} and bold mark a negative shift whose "
             r"prompt-clustered 95\% CI excludes zero. " + _legend
             + (r"$^{\ddagger}$ marks a model not yet re-read at four sampling seeds. " if _tek else "") +
             r"The prompt sets, the counts and the task-matched "
             r"reading are in Appendix~\ref{app:kesim}.}")
    L.append(r"\label{tab:families}")
    _SBF = all("s1b" in r for r in satir)
    L.append(r"\begin{tabular}{llrrrr" + ("r" if _SBF else "") + r"r@{\hspace{5pt}}rrr@{\hspace{5pt}}rrr}")
    L.append(r"\toprule")
    L.append(r" & & \multicolumn{" + ("6" if _SBF else "5") + r"}{c}{debate prompts} & "
             r"\multicolumn{3}{c}{ELICIT-99$^{\dagger}$} & "
             r"\multicolumn{3}{c}{in-context assistant prompt$^{\ast}$} \\")
    L.append((r"\cmidrule(lr){3-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}" if _SBF else
              r"\cmidrule(lr){3-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}"))
    L.append(r"model & lin. & base & inst. & $\Delta M_1$ & \% & "
             + (r"1st base & " if _SBF else "") + r"$\Delta$1st"
             r" & base & inst. & $\Delta M_1$ & base & inst. & $\Delta M_1$ \\")
    L.append(r"\midrule")
    for r in satir:
        a = f"\\textbf{{{r['m1d']:+.2f}}}" if r["a_ayrik"] else f"{r['m1d']:+.2f}"
        if r["bb"] is None:
            bcol = r"\multicolumn{3}{c}{\textsc{%s}}" % ("d-shelf" if r["b_not"] == "DEJENERE-RAF" else "not-run")
        else:
            bd = f"\\textbf{{{r['bd']:+.2f}}}" if r["b_ayrik"] else f"{r['bd']:+.2f}"
            _ks = "$^{\\S}$" if r["ad"] in _doy else ""
            bcol = f"{r['bb']:.1f} & {r['bi']:.1f} & {bd}{_ks}"
        if r["ub"] is None:
            ucol = (r"\multicolumn{3}{c}{\textsc{trunc.}}"
                    if r["u_ad"] == "ÖLCÜLEMEZ-KIRPMA" else
                    r"\multicolumn{3}{c}{\textsc{---}}")
        else:
            ud = f"\\textbf{{{r['ud']:+.2f}}}" if r["u_ayrik"] else f"{r['ud']:+.2f}"
            ucol = f"{r['ub']:.1f} & {r['ui']:.1f} & {ud}"
        _im = "$^{\\ddagger}$" if r["ad"] in _tek else ""
        L.append(f"{t1_adi(r['ad'])}{_im} & {r['soy']} & {r['ab']:.1f} & {r['ai']:.1f} & "
                 f"{a} & {r['ad_pct']:+.0f} & " + (f"{r['s1b']:.1f} & " if _SBF else "")
                 + f"{r['s1']:+.1f} & {bcol} & {ucol} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L[3] = L[3].replace(r"\label{tab:families}", r"\label{tab:families-full}")
    L[2] = (r"\caption{\textbf{Sixteen models on the debate prompts, ELICIT-99 and the in-context assistant prompt, in full.} "
            r"Table~\ref{tab:families} with the elicited and in-context assistant prompts "
            r"restored. " + L[2].split("} ", 1)[1])
    open(f"{OUT}/T1_full.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"  ✓ T1_full.tex (Ek E)")
    _SB = all("s1b" in r for r in satir) and os.environ.get("T1_SB", "0") == "1"
    print(f"  [PAYDA] t1_sb: hal_sutun={int(_SB)} · n_satir_s1b={sum('s1b' in r for r in satir)}/{len(satir)}")
    _SAT = []
    for r in satir:
        a = f"$\\mathbf{{{r['m1d']:+.2f}}}$" if r["a_ayrik"] else f"${r['m1d']:+.2f}$"
        _im = "$^{\\ddagger}$" if r["ad"] in _tek else ""
        _k = os.environ.get("T1_GOVDE_KIP", "pm")
        if _k == "ci":
            _c = f"[{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]".replace("-", "$-$")
        elif _k == "pm":
            _c = f"$\\pm${(r['ci'][1] - r['ci'][0]) / 2:.2f}"
        else:
            _c = f"{r['ad_pct']:+.0f}"
        _b1 = (f"$\\mathbf{{{r['s1']:+.1f}}}$"
               if r.get("s1_ci") and (r["s1_ci"][1] < 0 or r["s1_ci"][0] > 0) else f"${r['s1']:+.1f}$")
        _gad = {"Tulu3-8B": r'T\"ulu-3-8B', "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}.get(r["ad"], r["ad"])
        _SAT.append(f"{_gad}{_im} & {r['soy']} & " + _b1
                    + (f" & {r['s1b']:.1f}" if _SB else "")
                    + f" & {r['ab']:.1f} & {r['ai']:.1f} & {a} & {_c}")
    _y = (len(_SAT) + 1) // 2
    G = [r"\begin{table}[tb]",
         r"\centering\scriptsize\setlength{\tabcolsep}{2.8pt}",
         r"\caption{\textbf{Sixteen models on raw continuation.} "
         r"$\Delta$1st and $\Delta$2nd are the changes in the first- and second-person rates per 1,000 "
         r"tokens on the debate prompts" + (r", pooled over four further sampling seeds" if _dort and _A4d else "")
         + r". \emph{Base} and \emph{aligned} are second-person levels"
         + (r", and \emph{1st base} is the base checkpoint's first-person rate" if _SB else "")
         + r". Rows are sorted by $\Delta$2nd, the registered quantity; \textbf{bold} marks a change whose "
         + {"ci": r"prompt-clustered $95\%$ CI, given beside it, excludes zero.",
            "pm": r"prompt-clustered $95\%$ CI ($\pm$ half-width) excludes zero.",
            "yuzde": r"prompt-clustered $95\%$ CI excludes zero."}[os.environ.get("T1_GOVDE_KIP", "pm")]
         + (r" $^{\ddagger}$ marks a model not yet re-read at four sampling seeds." if _tek else "") + r"}",
         r"\label{tab:families}",
         (r"\begin{tabular}{llrrrrrr @{\hspace{1.1em}} llrrrrrr}" if _SB else
          r"\begin{tabular}{llrrrrr @{\hspace{1.1em}} llrrrrr}"),
         r"\toprule",
         (lambda h: (lambda c: c + r" & " + c + r" \\")(
              r"model & lin. & $\Delta$1st" + (r" & 1st base" if _SB else "")
              + r" & base & aligned & $\Delta$2nd & " + h))(
             {"ci": r"$95\%$ CI", "pm": r"$\pm$", "yuzde": r"\%"}[os.environ.get("T1_GOVDE_KIP", "pm")]),
         r"\midrule"]
    for k in range(_y):
        sag = _SAT[k + _y] if k + _y < len(_SAT) else " & " * (7 if _SB else 6)
        G.append(f"{_SAT[k]} & {sag} \\\\")
    G += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T1_families.tex", "w", encoding="utf-8").write("\n".join(G) + "\n")
    if _dort:
        T = [r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrr @{\hspace{1.0em}} lrrrrr}\toprule",
             r"model & base & inst. & $\Delta M_1$ & $\pm$ & $\Delta$1st & model & base & inst. & $\Delta M_1$ & $\pm$ & $\Delta$1st \\ \midrule"]
        _R = [(t1_adi(r["ad"]), r["tek"]) for r in satir]
        _h = (len(_R) + 1) // 2
        for k in range(_h):
            def _c(x):
                a_, v_ = x
                return (f"{a_} & {v_['ab']:.1f} & {v_['ai']:.1f} & ${v_['m1d']:+.2f}$ & $\\pm${(v_['ci'][1]-v_['ci'][0])/2:.2f}"
                        f" & {v_['s1']:+.1f}")
            T.append(_c(_R[k]) + " & " + (_c(_R[k + _h]) if k + _h < len(_R) else " & " * 5) + r" \\")
        T += [r"\bottomrule\end{tabular}\end{center}"]
        open(f"{OUT}/T1_TEK_SEED.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")
        print(f"  ✓ T1_TEK_SEED.tex ({len(_R)} aile) — tek seed okumasi Ek'te AYNEN")
    open(f"{OUT}/T1_body.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/families_table.py)\n"
        + _ozet + "\n")
    print("  ✓ T1_body.tex")
    _uc_kume = (r" The same contrast is read on three sets of cells, so it carries three values: this table pools four "
                r"further sampling seeds over all cells, Table~\ref{tab:twobytwo-full} reads one seed on the cells "
                r"non-degenerate in all four quadrants, and Table~\ref{tab:rungs-full} reads one seed on all cells, unfiltered.")
    open(f"{OUT}/T1_note.tex", "w", encoding="utf-8").write(
        "\\paragraph*{Notes on Table~\\ref{tab:families-full}.}\n"
        "\\vekalet{" + _ozet_ek + _uc_kume + " " + _prot + " " + _urial + " " + _urial_ek + " " + _kesik + "}\n")
    import json
    json.dump(dict(table="T1_families", sources=[pa, pb, pt, pt_ek, pz, po, pux, _ptv] + ([_pdz] if _dort else []) + ([_ptk] if _ptk else []) + ([_pdd] if _pdd else []) + ([_ptb] if _ptb else []),
                   payda=dict(n_aile=len(satir), A_ayrik=nA, B_olculen=nBp, B_ayrik=nB, s1_olcu=("dogrudan" if _A4dd else "turetilmis"),
                              s1_dusen=n1, s1_daha_cok=nb, U_olculen=nU,
                              U_ayrik=nUd, U_asagi=Usay["asagi"], U_ters=Usay["ters"],
                              U_bicim=Usay["bicim"], kesik_doygun=len(_doy)),
                   sira=[r["ad"] for r in satir],
                   note="renders pre-specified measurements; computes no new statistic"),
              open(f"{OUT}/T1_families.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ✓ T1_families.tex ({len(L)} satir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
