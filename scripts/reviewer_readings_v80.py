#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, re, subprocess, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
G = f"{ROOT}/results"
TARIH = "2026-09-15"
GECICI = "<scratch>"
CARD9 = f"{G}/PREREG9_A_OLCUM_2026-08-31.json"
CARD_SS = f"{G}/template_filtered_2026-09-07.json"
CARD_YSZ = f"{G}/template_filtered_first_person_echo_free_2026-09-15.json"
CARD_U = f"{G}/URIAL_VERDICT_2048_16_K2048_2026-09-11.json"
CARD_BANT = f"{ROOT}/paper/figures/BANT.meta.json"
US = re.compile(r"\bUS\b")
SATIR = re.compile(r"^\s{0,3}#{1,6}\s+\S|^\s{0,3}[-*+•]\s+\S|^\s{0,3}\d{1,2}[.)]\s+\S|^\s*\*\*[^*\n]+\*\*\s*:?\s*$")


def damga(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def sinif(ci): return "yukari" if ci[0] > 0 else ("asagi" if ci[1] < 0 else "null")
def yaz(ad, K):
    y = f"{G}/HAKEM_{ad}_{TARIH}.json"; json.dump(K, io.open(y, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(f"✓ {y}")


def p1_sayim(R, kip):
    import addressee_olcu as MO
    if kip == "dahil":
        return np.array([len(MO.SAHIS1.findall(r["metin"] or "")) for r in R], float)
    return np.array([len(MO.SAHIS1.findall(r["metin"] or "")) - len(US.findall(r["metin"] or "")) for r in R], float)


def ikame(V, p1):
    import addressee_run as MK
    W = V.copy(); W[:, MK.ALAN.index("m1_sahis2")] = p1; return W


def eslestir(Rb, Rs):
    ib = {(r.get("kol"), r.get("cekim"), r.get("istem_i")): j for j, r in enumerate(Rb)}
    c = [(ib[k], j) for j, r in enumerate(Rs) if (k := (r.get("kol"), r.get("cekim"), r.get("istem_i"))) in ib]
    return np.array([x for x, _ in c]), np.array([y for _, y in c])


def fark_ci(Ws, Wb, ist_s, ist_b):
    import addressee_run as MK, rung_force_ci as BK
    d = MK.olc_toplam(Ws)["M1"] - MK.olc_toplam(Wb)["M1"]
    lo, hi, nk = BK.kume_boot(Ws, ist_s, Wb, ist_b, "M1")
    return dict(d=round(float(d), 4), ci=[round(lo, 4), round(hi, 4)], sinif=sinif([lo, hi]), n_istem=int(nk))


def birincil():
    A = json.load(open(CARD9, encoding="utf-8"))
    return {a: v for a, v in A.items() if isinstance(v, dict) and v.get("sinif") == "BIRINCIL"}


def q1_panel():
    import addressee_run as MK, depersonalization16_neutral_run as M9, form_count as FS
    from template_filtered_first_person import birinci
    nlp = FS._boru(); t0 = time.time(); out = {}; sap = 0.0
    for aile, v in sorted(birincil().items()):
        bas, son = v["kontrast"].split("→")
        Rb, Rs = MK.oku(aile, bas), MK.oku(aile, son)
        xb, xs = eslestir(Rb, Rs)
        Vb, Vs = MK.satir_bilesenleri(nlp, Rb)[xb], MK.satir_bilesenleri(nlp, Rs)[xs]
        Rb2, Rs2 = [Rb[i] for i in xb], [Rs[i] for i in xs]
        ib, is_ = np.array([r["istem_i"] for r in Rb2]), np.array([r["istem_i"] for r in Rs2])
        s1 = abs((MK.olc_toplam(Vs)["M1"] - MK.olc_toplam(Vb)["M1"]) - v["fark"]["M1"]["gozlenen"])
        s2 = abs((M9.sahis1_1k(Vs) - M9.sahis1_1k(Vb)) - v["sahis1_1k"]["d"])
        sap = max(sap, s1, s2)
        r = dict(kontrast=v["kontrast"], kapi_sapma=max(s1, s2))
        r["dogrudan_US_haric"] = fark_ci(ikame(Vs, p1_sayim(Rs2, "cs")), ikame(Vb, p1_sayim(Rb2, "cs")), is_, ib)
        r["dogrudan_US_dahil"] = fark_ci(ikame(Vs, p1_sayim(Rs2, "dahil")), ikame(Vb, p1_sayim(Rb2, "dahil")), is_, ib)
        r["turetilmis"] = fark_ci(birinci(Vs), birinci(Vb), is_, ib)
        nus = dict(taban=int(sum(len(US.findall(x["metin"] or "")) for x in Rb2)), hizali=int(sum(len(US.findall(x["metin"] or "")) for x in Rs2)))
        r["US_sayisi"] = nus; r["US_etkisi_d"] = round(r["dogrudan_US_dahil"]["d"] - r["dogrudan_US_haric"]["d"], 4)
        out[aile] = r
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {aile:16s} 1.s US-haric {r['dogrudan_US_haric']['d']:+7.2f} {r['dogrudan_US_haric']['ci']} · "
              f"dahil {r['dogrudan_US_dahil']['d']:+7.2f} · türetilmis {r['turetilmis']['d']:+7.2f} · US {nus} · kapi {max(s1, s2):.1e}", flush=True)
    say = {k: {c: sorted(a for a, r in out.items() if r[k]["sinif"] == c) for c in ("asagi", "yukari", "null")}
           for k in ("dogrudan_US_haric", "dogrudan_US_dahil", "turetilmis")}
    yaz("Q1_PANEL_BIRINCIL", dict(sinif="BETIM · card · kâgida girmez", rule="preregistration/rule_reviewer_readings_2026-09-15.md @ 25315d8e",
                                  soru="",
                                  alet=dict(hal="ESDEGER" if sap <= 1e-6 else "KAYDI", en_buyuk_sapma=sap, kanonik=os.path.basename(CARD9)),
                                  sayim={k: {c: len(x) for c, x in v.items()} for k, v in say.items()}, sayim_adlar=say,
                                  aile=out, damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)))


def urial_ikame(kip):
    os.makedirs(GECICI, exist_ok=True)
    os.environ.update(URIAL_VERDICT_KOK=__DNH_DATA__ + "/c1_panel_urial_2048", URIAL_KIRPMA_MAXLEN="2048",
                      URIAL_KOLLAR="", URIAL_VERDICT_CIK=f"{GECICI}/urial_{kip}.json", URIAL_REPORT=f"{GECICI}/urial_{kip}_report.md")
    import addressee_run as MK
    if kip != "yok":
        asil = MK.satir_bilesenleri
        def sarmali(nlp, R, _a=asil):
            return ikame(_a(nlp, R), p1_sayim(R, kip))
        MK.satir_bilesenleri = sarmali
    import urial_verdict as UH
    return UH.main()


def q1_urial_ozet():
    U = json.load(open(CARD_U, encoding="utf-8"))
    Y = {k: json.load(open(f"{GECICI}/urial_{k}.json", encoding="utf-8")) for k in ("yok", "cs", "dahil")}
    sap, out, say = 0.0, {}, {}
    for kk in ("i", "ii"):
        out[kk] = {}
        for a, x in U["kiyas"][kk]["aile"].items():
            y = Y["yok"]["kiyas"][kk]["aile"].get(a, {})
            if x.get("hal") == "ÖLCÜLDÜ":
                sap = max(sap, abs(x["dM1"] - y.get("dM1", 1e9)), abs(x["ci"][0] - y["ci"][0]), abs(x["ci"][1] - y["ci"][1]))
            r = {}
            for k in ("cs", "dahil"):
                z = Y[k]["kiyas"][kk]["aile"].get(a, {})
                if z.get("hal") == "ÖLCÜLDÜ":
                    r[k] = dict(d1=z["dM1"], ci=z["ci"], sinif=sinif(z["ci"]), taban=z["M1_taban"], hizali=z["M1_hizali"])
                else:
                    r[k] = dict(hal=z.get("hal"))
            r["US_etkisi_d"] = round(r["dahil"]["d1"] - r["cs"]["d1"], 4) if "d1" in r["cs"] and "d1" in r["dahil"] else None
            out[kk][a] = r
        say[kk] = {k: {c: sorted(a for a, r in out[kk].items() if r[k].get("sinif") == c) for c in ("asagi", "yukari", "null")}
                   for k in ("cs", "dahil")}
    yaz("Q1_URIAL_BIRINCIL", dict(sinif="BETIM · card · kâgida girmez", rule="preregistration/rule_reviewer_readings_2026-09-15.md @ 25315d8e",
                                  soru="",
                                  yol="urial_verdict.py (K2048 yapilandirmasi) · M1 sütunu p1 ile ikame · kiyas (i) kâgidin okumasi",
                                  alet=dict(hal="ESDEGER" if sap <= 1e-4 else "KAYDI", en_buyuk_sapma=sap, kanonik=os.path.basename(CARD_U)),
                                  sayim={kk: {k: {c: len(x) for c, x in v.items()} for k, v in s.items()} for kk, s in say.items()},
                                  sayim_adlar=say, kiyas=out, damga_utc=damga()))


def duzyazi(t):
    L = (t or "").split("\n"); ne = [l for l in L if l.strip()]
    at = [l for l in L if l.strip() and SATIR.match(l)]
    return "\n".join(l for l in L if not SATIR.match(l)), len(at), len(ne)


def q2():
    import addressee_run as MK, form_count as FS
    nlp = FS._boru(); t0 = time.time(); out = {}; sap = 0.0
    for aile, v in sorted(birincil().items()):
        bas, son = v["kontrast"].split("→")
        Rb, Rs = MK.oku(aile, bas), MK.oku(aile, son)
        xb, xs = eslestir(Rb, Rs); Rb, Rs = [Rb[i] for i in xb], [Rs[i] for i in xs]
        ib, is_ = np.array([r["istem_i"] for r in Rb]), np.array([r["istem_i"] for r in Rs])
        Vb, Vs = MK.satir_bilesenleri(nlp, Rb), MK.satir_bilesenleri(nlp, Rs)
        s = abs((MK.olc_toplam(Vs)["M1"] - MK.olc_toplam(Vb)["M1"]) - v["fark"]["M1"]["gozlenen"]); sap = max(sap, s)
        r = dict(kontrast=v["kontrast"], kapi_sapma=s)
        Dv = {}
        for bac, R, V in (("taban", Rb, Vb), ("hizali", Rs, Vs)):
            D = [duzyazi(x["metin"]) for x in R]
            W = MK.satir_bilesenleri(nlp, [dict(metin=d[0]) for d in D]); Dv[bac] = W
            at, ne = sum(d[1] for d in D), sum(d[2] for d in D)
            r[f"atilan_{bac}"] = dict(satir_payi=round(at / max(ne, 1), 4), jeton_payi=round(1 - W[:, 0].sum() / max(V[:, 0].sum(), 1), 4),
                                      uretim_payi=round(float(np.mean([d[1] > 0 for d in D])), 4))
        r["duzyazi"] = fark_ci(Dv["hizali"], Dv["taban"], is_, ib)
        r["tam"] = dict(d=round(float(MK.olc_toplam(Vs)["M1"] - MK.olc_toplam(Vb)["M1"]), 4))
        out[aile] = r
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {aile:16s} ΔM1 tam {r['tam']['d']:+7.2f} → düzyazi {r['duzyazi']['d']:+7.2f} {r['duzyazi']['ci']} · "
              f"atilan satir taban {r['atilan_taban']['satir_payi']:.3f} / hizali {r['atilan_hizali']['satir_payi']:.3f} · kapi {s:.1e}", flush=True)
    say = {c: sorted(a for a, r in out.items() if r["duzyazi"]["sinif"] == c) for c in ("asagi", "yukari", "null")}
    yaz("Q2_DUZYAZI", dict(sinif="BETIM · card · kâgida girmez", rule="preregistration/rule_reviewer_readings_2026-09-15.md @ 25315d8e",
                           soru="",
                           satir_deseni=SATIR.pattern, sinir="",
                           alet=dict(hal="ESDEGER" if sap <= 1e-6 else "KAYDI", en_buyuk_sapma=sap),
                           sayim={c: len(x) for c, x in say.items()}, sayim_adlar=say, aile=out, damga_utc=damga(),
                           sure_dk=round((time.time() - t0) / 60, 1)))


def q2_sahis1():
    import addressee_run as MK, depersonalization16_neutral_run as M9, form_count as FS
    nlp = FS._boru(); t0 = time.time(); out = {}; sap = 0.0
    Q1 = json.load(open(f"{G}/HAKEM_Q1_PANEL_BIRINCIL_{TARIH}.json", encoding="utf-8"))["aile"]
    Q2 = json.load(open(f"{G}/HAKEM_Q2_DUZYAZI_{TARIH}.json", encoding="utf-8"))["aile"]
    for aile, v in sorted(birincil().items()):
        bas, son = v["kontrast"].split("→")
        Rb, Rs = MK.oku(aile, bas), MK.oku(aile, son)
        xb, xs = eslestir(Rb, Rs); Rb, Rs = [Rb[i] for i in xb], [Rs[i] for i in xs]
        ib, is_ = np.array([r["istem_i"] for r in Rb]), np.array([r["istem_i"] for r in Rs])
        Vb, Vs = MK.satir_bilesenleri(nlp, Rb), MK.satir_bilesenleri(nlp, Rs)
        s1 = abs((MK.olc_toplam(Vs)["M1"] - MK.olc_toplam(Vb)["M1"]) - v["fark"]["M1"]["gozlenen"])
        s2 = abs((M9.sahis1_1k(Vs) - M9.sahis1_1k(Vb)) - v["sahis1_1k"]["d"])
        tam1 = MK.olc_toplam(ikame(Vs, p1_sayim(Rs, "cs")))["M1"] - MK.olc_toplam(ikame(Vb, p1_sayim(Rb, "cs")))["M1"]
        s3 = abs(round(float(tam1), 4) - Q1[aile]["dogrudan_US_haric"]["d"])
        W, P, pay = {}, {}, {}
        for bac, R in (("taban", Rb), ("hizali", Rs)):
            D = [duzyazi(x["metin"]) for x in R]; Rd = [dict(metin=d[0]) for d in D]
            W[bac] = MK.satir_bilesenleri(nlp, Rd); P[bac] = p1_sayim(Rd, "cs")
            pay[bac] = round(sum(d[1] for d in D) / max(sum(d[2] for d in D), 1), 4)
        dM1 = MK.olc_toplam(W["hizali"])["M1"] - MK.olc_toplam(W["taban"])["M1"]
        s4 = abs(round(float(dM1), 4) - Q2[aile]["duzyazi"]["d"])
        s5 = max(abs(pay["taban"] - Q2[aile]["atilan_taban"]["satir_payi"]), abs(pay["hizali"] - Q2[aile]["atilan_hizali"]["satir_payi"]))
        sap = max(sap, s1, s2, s3, s4, s5)
        r = dict(kontrast=v["kontrast"], kapi=dict(ham_M1=s1, ham_sahis1=s2, q1_tam=s3, q2_duzyazi=s4, q2_pay=s5),
                 duzyazi_d1st=fark_ci(ikame(W["hizali"], P["hizali"]), ikame(W["taban"], P["taban"]), is_, ib),
                 tam_d1st_q1=Q1[aile]["dogrudan_US_haric"], duzyazi_dM1_q2=Q2[aile]["duzyazi"], atilan_satir_payi=pay)
        out[aile] = r
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {aile:16s} Δ1st düzyazi {r['duzyazi_d1st']['d']:+7.2f} {r['duzyazi_d1st']['ci']} "
              f"{r['duzyazi_d1st']['sinif']} · tam {r['tam_d1st_q1']['d']:+7.2f} · kapi {max(s1, s2, s3, s4, s5):.1e}", flush=True)
    say = {c: sorted(a for a, r in out.items() if r["duzyazi_d1st"]["sinif"] == c) for c in ("asagi", "yukari", "null")}
    yaz("IS1_DUZYAZI_SAHIS1", dict(sinif="BETIM · card", rule="preregistration/rule_prose_first_person_2026-09-15.md @ 508b8179",
                                   soru="",
                                   satir_deseni=SATIR.pattern, alet=dict(hal="ESDEGER" if sap <= 1e-4 else "KAYDI", en_buyuk_sapma=sap,
                                   kapilar="ham M1 ≤1e-6 · ham sahis1 ≤1e-6 · Q1 tam 4 ondalik · Q2 düzyazi 4 ondalik · Q2 pay 4 ondalik"),
                                   sayim={c: len(x) for c, x in say.items()}, sayim_adlar=say, aile=out, damga_utc=damga(),
                                   sure_dk=round((time.time() - t0) / 60, 1)))


def q3():
    from math import comb
    A = birincil(); isr = {a: int(np.sign(v["fark"]["M1"]["gozlenen"])) for a, v in A.items()}
    Q = [a for a in A if a.startswith("Qwen2.5")]
    soy = {"Google": [a for a in A if a.startswith("Gemma")], "Meta": [a for a in A if a.startswith("Llama")],
           "AI2": ["Tulu3-8B", "OLMo2-13B", "OLMo2-32B"], "Qwen": Q, "Mistral": ["Mistral-7B-v0.3"], "OLMo-3": ["OLMo-3-7B"]}
    U = {"U0": {a: [a] for a in A},
         "U1": {**{a: [a] for a in A if a not in Q}, "Qwen": Q},
         "U2": {**{a: [a] for a in A if a not in Q + ["Tulu3-8B", "Llama-3.1-8B"]}, "Qwen": Q, "Llama-3.1-8B-taban": ["Tulu3-8B", "Llama-3.1-8B"]},
         "U3": soy,
         "U4": {**{k: v for k, v in soy.items() if k not in ("AI2", "OLMo-3")}, "AI2+OLMo-3": soy["AI2"] + soy["OLMo-3"]}}
    out = {}
    for ad, birim in U.items():
        assert sorted(sum(birim.values(), [])) == sorted(A), ad
        yon = {b: (isr[u[0]] if len({isr[x] for x in u}) == 1 else 0) for b, u in birim.items()}
        n = sum(1 for y in yon.values() if y != 0); k = sum(1 for y in yon.values() if y < 0)
        kk = min(k, n - k)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n)
        out[ad] = dict(n_birim=len(birim), n_testte=n, n_asagi=k, n_karisik=sum(1 for y in yon.values() if y == 0),
                       karisik=[b for b, y in yon.items() if y == 0], p_iki_yonlu=p, bonferroni_bar=0.005, bar_alti=bool(p < 0.005),
                       birimler={b: u for b, u in birim.items()})
        print(f"  {ad}: birim {len(birim)} · testte {n} · asagi {k} · karisik {out[ad]['n_karisik']} · p={p:.3g} · bar 0,005 alti {p < 0.005}")
    yaz("Q3_ISARET_BIRIM", dict(sinif="BETIM · card · kâgida girmez", rule="preregistration/rule_reviewer_readings_2026-09-15.md @ 25315d8e",
                                soru="Q3 · bagimlilik birimlerine indirgenmis iki yönlü kesin isaret testi", kaynak=os.path.basename(CARD9),
                                card_arama="kâgidin Multiplicity paragrafindaki 11/10 birim p-degerlerinin repoda karti yoktu (rule §Q3)",
                                tanim=out, damga_utc=damga()))


def q4():
    A = birincil(); S = {r["aile"]: r for r in json.load(open(CARD_SS, encoding="utf-8"))["aileler"]}
    U = json.load(open(CARD_U, encoding="utf-8"))["kiyas"]
    T = {}
    for a, v in sorted(A.items()):
        f = v["fark"]["M1"]; s = S.get(a, {}).get("sbl"); ui = U["i"]["aile"].get(a, {}); uii = U["ii"]["aile"].get(a, {})
        T[a] = dict(ciplak=dict(d=round(f["gozlenen"], 4), ci=[round(x, 4) for x in f["ci"]], sinif=sinif(f["ci"])),
                    template=dict(d=s["dM1"], ci=s["ci"], sinif=sinif(s["ci"])) if s else None,
                    urial_i=dict(d=ui["dM1"], ci=ui["ci"], sinif=sinif(ui["ci"]), AD=ui.get("AD")) if ui.get("hal") == "ÖLCÜLDÜ" else dict(hal=ui.get("hal")),
                    urial_ii=dict(d=uii["dM1"], ci=uii["ci"], sinif=sinif(uii["ci"]), AD=uii.get("AD")) if uii.get("hal") == "ÖLCÜLDÜ" else dict(hal=uii.get("hal")))
    def say(k): return {c: sorted(a for a, r in T.items() if r[k] and r[k].get("sinif") == c) for c in ("asagi", "yukari", "null")}
    SY = {k: say(k) for k in ("ciplak", "template", "urial_i", "urial_ii")}
    kes = sorted(set(SY["template"]["yukari"]) & set(SY["urial_i"]["yukari"]))
    ters_i = sorted(a for a, r in T.items() if r["urial_i"].get("AD") == "TERS")
    for a, r in T.items():
        print(f"  {a:16s} ciplak {r['ciplak']['sinif']:6s} · template {r['template']['sinif'] if r['template'] else '—':6s} · URIAL(i) {r['urial_i'].get('sinif','—'):6s} ({r['urial_i'].get('AD')}) · URIAL(ii) {r['urial_ii'].get('sinif','—')}")
    print(f"  ★ sablonda yukari {len(SY['template']['yukari'])} · URIAL(i) CI-yukari {len(SY['urial_i']['yukari'])} (kartin TERS adi {len(ters_i)}) · kesisim {kes}")
    yaz("Q4_PROTOKOL_ISARET", dict(sinif="BETIM · card · kâgida girmez", rule="preregistration/rule_reviewer_readings_2026-09-15.md @ 25315d8e",
                                   soru="Q4 · aile × protokol (ciplak / template / URIAL) isaret tablosu; sablonda yükselenler ∩ URIAL'de yükselenler",
                                   kaynak=[os.path.basename(x) for x in (CARD9, CARD_SS, CARD_U)],
                                   sayim={k: {c: len(x) for c, x in v.items()} for k, v in SY.items()}, sayim_adlar=SY,
                                   urial_i_TERS_adi=ters_i, kesisim_template_yukari_urial_i_yukari=kes,
                                   kesisim_template_yukari_urial_i_TERS=sorted(set(SY["template"]["yukari"]) & set(ters_i)),
                                   tablo=T, damga_utc=damga()))


def oran(d): v = [x for x in d.values() if x is not None]; return dict(min=min(v), max=max(v), oran=round(max(v) / min(v), 3), n=len(v))


def q5():
    import addressee_run as MK, form_count as FS, template_filtered as SS, echo as YK, continuation_mode_exit as Q2
    A = birincil(); S = {r["aile"]: r for r in json.load(open(CARD_SS, encoding="utf-8"))["aileler"]}
    U = json.load(open(CARD_U, encoding="utf-8"))["kiyas"]["i"]["aile"]; B = json.load(open(CARD_BANT, encoding="utf-8"))
    bant = dict(
        ciplak=dict(taban=oran({a: v["basamak_tablosu"][v["kontrast"].split("→")[0]]["M1"] for a, v in A.items()}),
                    hizali=oran({a: v["basamak_tablosu"][v["kontrast"].split("→")[1]]["M1"] for a, v in A.items()})),
        template_filtered=dict(taban=oran({a: S[a]["sbl"]["M1_taban"] for a in A if a in S}), hizali=oran({a: S[a]["sbl"]["M1_hizali"] for a in A if a in S}),
                             BANT_meta=dict(taban=B["oran_taban"], hizali=B["oran_hizali"])),
        urial_i=dict(taban=oran({a: x["M1_taban"] for a, x in U.items() if x.get("hal") == "ÖLCÜLDÜ" and x.get("AD") != "ÖLCÜLEMEZ-KIRPMA"}),
                     hizali=oran({a: x["M1_hizali"] for a, x in U.items() if x.get("hal") == "ÖLCÜLDÜ" and x.get("AD") != "ÖLCÜLEMEZ-KIRPMA"})))
    nlp = FS._boru(); t0 = time.time(); CIFT, _ = SS.V23.cift_kur(); lev = {"taban": {}, "hizali": {}}; det = {}; sap = 0.0
    for a, zt, zh in CIFT:
        bac = {("sbl", "taban"): SS.oku(SS.SAB, a, zt), ("sbl", "hizali"): SS.oku(SS.SAB, a, zh),
               ("cip", "taban"): SS.oku(SS.CIP, a, zt), ("cip", "hizali"): SS.oku(SS.CIP, a, zh)}
        F = {k: SS.bayraklar(R) for k, R in bac.items()}; K = {k: [SS.anahtar(r) for r in R] for k, R in bac.items()}
        kirli = set()
        for k in bac:
            kirli |= {K[k][i] for i in np.where(F[k][0])[0]}
        det[a] = {}
        for b in ("taban", "hizali"):
            R = bac[("sbl", b)]; tut = np.array([x not in kirli for x in K[("sbl", b)]])
            ek = np.array([YK.yanki_mi(r.get("onek", ""), r["metin"]) for r in R]); tr = np.array([bool(Q2.TUR.search(r["metin"] or "")) for r in R])
            V = MK.satir_bilesenleri(nlp, R)
            m_s = MK.olc_toplam(V[tut])["M1"]; sap = max(sap, abs(round(m_s, 4) - S[a]["sbl"][f"M1_{b}"]))
            m_t = MK.olc_toplam(V[tut & ~ek & ~tr])["M1"]; lev[b][a] = m_t
            det[a][b] = dict(M1_filtered=round(m_s, 4), M1_yankisiz_tursuz=round(m_t, 4), atilan_pay=round(float((tut & (ek | tr)).sum() / max(tut.sum(), 1)), 4))
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {a:16s} taban {det[a]['taban']} · hizali {det[a]['hizali']}", flush=True)
    bant["template_filtered_yankisiz_tursuz"] = dict(taban=oran(lev["taban"]), hizali=oran(lev["hizali"]))
    for k, v in band.items():
        print(f"  ★ {k}: taban {v['taban']['oran']}× · hizali {v['hizali']['oran']}×")
    yaz("Q5_BANT_UC_ZEMIN", dict(sinif="BETIM · card · kâgida girmez", rule="preregistration/rule_reviewer_readings_2026-09-15.md @ 25315d8e",
                                 soru="",
                                 alet=dict(hal="ESDEGER" if sap <= 1e-4 else "KAYDI", en_buyuk_sapma=sap, kanonik=os.path.basename(CARD_SS)),
                                 tur_deseni="continuation_mode_exit.TUR", bant=band, aile=det, damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)))


def w2():
    import addressee_run as MK, form_count as FS, rung_force_ci as BK, template_filtered as SS, echo as YK, turn_marker as TI
    from template_filtered_first_person import birinci
    Y = {r["aile"]: r for r in json.load(open(CARD_YSZ, encoding="utf-8"))["aileler"]}
    nlp = FS._boru(); t0 = time.time(); CIFT, _ = SS.V23.cift_kur(); out = []; sap = 0.0
    for a, zt, zh in CIFT:
        bac = {("sbl", "taban"): SS.oku(SS.SAB, a, zt), ("sbl", "hizali"): SS.oku(SS.SAB, a, zh),
               ("cip", "taban"): SS.oku(SS.CIP, a, zt), ("cip", "hizali"): SS.oku(SS.CIP, a, zh)}
        F = {k: SS.bayraklar(R) for k, R in bac.items()}; A = {k: [SS.anahtar(r) for r in R] for k, R in bac.items()}
        kirli = set()
        for k in bac:
            kirli |= {A[k][i] for i in np.where(F[k][0])[0]}
        yank = set()
        for k in (("sbl", "taban"), ("sbl", "hizali")):
            yank |= {A[k][i] for i, r in enumerate(bac[k]) if YK.yanki_mi(r.get("onek", ""), r["metin"])}
        dis = kirli | yank
        Rt, Rh = bac[("sbl", "taban")], bac[("sbl", "hizali")]
        mt = np.array([x not in dis for x in A[("sbl", "taban")]]); mh = np.array([x not in dis for x in A[("sbl", "hizali")]])
        it = np.array([x.get("istem_i", -1) for x in Rt]); ih = np.array([x.get("istem_i", -1) for x in Rh])
        Vt, Vh = MK.satir_bilesenleri(nlp, Rt), MK.satir_bilesenleri(nlp, Rh)
        k0 = fark_ci(birinci(Vh)[mh], birinci(Vt)[mt], ih[mh], it[mt])
        sap = max(sap, abs(k0["d"] - Y[a]["d1"]), abs(k0["ci"][0] - Y[a]["ci1"][0]), abs(k0["ci"][1] - Y[a]["ci1"][1]))
        im = TI.aile_isaretleri(Rh)
        if im:
            IS = re.compile("|".join(re.escape(x) for x in im))
            def kes(R): return [dict(r, metin=(r["metin"][:m.start()] if (m := IS.search(r["metin"] or "")) else r["metin"])) for r in R]
            Kt, Kh = kes(Rt), kes(Rh)
            pay = dict(taban=round(float(np.mean([bool(IS.search(r["metin"] or "")) for r in Rt])), 4),
                       hizali=round(float(np.mean([bool(IS.search(r["metin"] or "")) for r in Rh])), 4))
        else:
            Kt, Kh = Rt, Rh; pay = dict(taban=None, hizali=None)
        Ct, Ch = MK.satir_bilesenleri(nlp, Kt), MK.satir_bilesenleri(nlp, Kh)
        r = dict(aile=a, isaretler=im, isaret_cozuldu=bool(im), kesilen_satir_payi=pay, tutulan_oran=round(float(mt.mean()), 4),
                 kapi_kesmesiz=k0, kapi_sapma=max(abs(k0["d"] - Y[a]["d1"]), abs(k0["ci"][0] - Y[a]["ci1"][0]), abs(k0["ci"][1] - Y[a]["ci1"][1])),
                 turetilmis=fark_ci(birinci(Ch)[mh], birinci(Ct)[mt], ih[mh], it[mt]),
                 dogrudan_US_haric=fark_ci(ikame(Ch, p1_sayim(Kh, "cs"))[mh], ikame(Ct, p1_sayim(Kt, "cs"))[mt], ih[mh], it[mt]))
        out.append(r)
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {a:16s} kesmesiz {k0['d']:+7.2f} (card {Y[a]['d1']:+.2f}) → kesik türetilmis {r['turetilmis']['d']:+7.2f} {r['turetilmis']['ci']} · "
              f"dogrudan {r['dogrudan_US_haric']['d']:+7.2f} · kesilen {pay} · isaret {im}", flush=True)
    say = {k: {c: sorted(r["aile"] for r in out if r[k]["sinif"] == c) for c in ("asagi", "yukari", "null")} for k in ("turetilmis", "dogrudan_US_haric")}
    yaz("W2_TUR_KESIK_SAHIS1", dict(sinif="BETIM · card · kâgida girmez", rule="preregistration/rule_reviewer_readings_2026-09-15.md @ 25315d8e",
                                    soru="",
                                    alet=dict(hal="ESDEGER" if sap <= 1e-4 else "KAYDI", en_buyuk_sapma=sap, kanonik=os.path.basename(CARD_YSZ)),
                                    sayim={k: {c: len(x) for c, x in v.items()} for k, v in say.items()}, sayim_adlar=say, aileler=out,
                                    damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)))


if __name__ == "__main__":
    kip = sys.argv[1]
    if kip == "urial_ikame":
        sys.exit(urial_ikame(sys.argv[2]) or 0)
    {"q2_sahis1": q2_sahis1, "q1_panel": q1_panel, "q1_urial_ozet": q1_urial_ozet, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "w2": w2}[kip]()
