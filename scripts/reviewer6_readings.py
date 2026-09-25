#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, inspect, io, json, os, re, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
G = f"{ROOT}/results"
RULE = "preregistration/rule_reviewer6_readings_2026-09-17.md @ 84f79b65"
ARA = os.environ.get("HAKEM6_ARA", __DNH_DATA__ + "/hakem6_ara")
CARD_SS = f"{G}/template_filtered_2026-09-07.json"
CARD_U = f"{G}/URIAL_VERDICT_2048_16_K2048_2026-09-11.json"
T1_FULL = f"{ROOT}/paper/figures/T1_full.tex"
GEMMA = ("Gemma-3-4B", "Gemma-3-12B", "Gemma-3-27B")
SINIFLAR = ("YO", "KP", "AO", "SI", "TK")
KIPLER = ("RET", "ASISTAN", "META", "DEVAM")
BANT_ESIK_SATIR = 80
TOL = 1e-6


def damga(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def sha16(y): return hashlib.sha256(open(y, "rb").read()).hexdigest()[:16]


def oran(d):
    v = {a: x for a, x in d.items() if x is not None}
    lo = min(v, key=v.get); hi = max(v, key=v.get)
    return dict(min=round(v[lo], 4), min_aile=lo, max=round(v[hi], 4), max_aile=hi, oran=round(v[hi] / v[lo], 3), n=len(v))


def sinif3(ci): return "asagi" if ci[1] < 0 else ("yukari" if ci[0] > 0 else "null")


def yaz(ad, D):
    y = f"{G}/HAKEM6_{ad}_2026-09-17.json"
    json.dump(D, io.open(y, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {os.path.relpath(y, ROOT)} · sha {sha16(y)}")
    return y


def t1_full_elicit():
    out = {}
    for sat in io.open(T1_FULL, encoding="utf-8"):
        h = [x.strip() for x in sat.split("&")]
        if len(h) >= 13 and re.match(r"^[A-Za-z]", h[0]) and h[0] != "family":
            out[h[0].replace("--", "-").replace("Tulu3", "Tulu3")] = (float(h[7]), float(h[8]))
    return out


def elicit():
    import elicited16_loader as E16, muhatap9b_run as M9B, two_rulers_2x2 as P2, form_count as FS
    from reading_style_object import payda
    t0 = time.time(); nlp = FS._boru()
    B, kun = E16.yukle()
    CARD = json.load(open(f"{ROOT}/{E16.CARD}", encoding="utf-8"))
    aileler = [a for a in B if not a.startswith("_")]
    raf = [r["aile"] for r in B["_kunye"]["raf"]]
    fnM1, fnS1 = P2.OLCU["M1_1k"][0], P2.OLCU["SAHIS1_1k"][0]
    satir, sinif_tab, red_kapi, bicim = {}, {}, [], {}
    for a in aileler:
        Rb = M9B.oku_b(a, "base"); Ri = M9B.oku_b(a, "instruct")
        i0, i1 = P2.cift_al(Rb, Ri)
        P0 = P2.bilesen(nlp, Rb)[i0]; P1 = P2.bilesen(nlp, Ri)[i1]
        ist = np.array([Ri[y].get("istem_i") for y in i1])
        k = CARD["_aile"][a]
        tum = {"M1_1k": P2.olc(P0, P1, ist, fnM1), "SAHIS1_1k": P2.olc(P0, P1, ist, fnS1)}
        sap = max(max(abs(tum[o]["taban"] - k[o]["taban"]), abs(tum[o]["gozlenen"] - k[o]["gozlenen"]),
                      abs(tum[o]["ci"][0] - k[o]["ci"][0]), abs(tum[o]["ci"][1] - k[o]["ci"][1])) for o in tum)
        if sap > TOL:
            red_kapi.append((a, sap))
        bicim[a] = {b: sorted({str(json.loads(l)["sablonlu"]) for l in open(M9B._yol(a, b), encoding="utf-8")})
                    for b in ("base", "instruct")}
        satir[a] = dict(M1_taban=k["M1_1k"]["taban"], M1_hizali=k["M1_1k"]["taban"] + k["M1_1k"]["gozlenen"],
                        yeniden_sapma=sap, sablonlu=bicim[a], n_cift=len(i0))
        sn = np.array([str(x).split("-")[0] for x in ist])
        sinif_tab[a] = {}
        for c in SINIFLAR:
            m = sn == c
            r1 = P2.olc(P0[m], P1[m], ist[m], fnM1); r2 = P2.olc(P0[m], P1[m], ist[m], fnS1)
            sinif_tab[a][c] = dict(n_istem=int(len(np.unique(ist[m]))), n_satir=int(m.sum()),
                                   dM1=round(r1["gozlenen"], 4), ci_M1=[round(x, 4) for x in r1["ci"]], sinif_M1=sinif3(r1["ci"]),
                                   d1st=round(r2["gozlenen"], 4), ci_1st=[round(x, 4) for x in r2["ci"]], sinif_1st=sinif3(r2["ci"]))
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {a:16s} kapi sapma {sap:.1e} · bicim {bicim[a]} · "
              + " · ".join(f"{c} {sinif_tab[a][c]['dM1']:+.2f}/{sinif_tab[a][c]['d1st']:+.2f}" for c in SINIFLAR), flush=True)
    T1 = t1_full_elicit()
    red_t1 = [a for a in aileler if a in T1 and (round(satir[a]["M1_taban"], 1), round(satir[a]["M1_hizali"], 1)) != T1[a]]
    eksik_t1 = [a for a in aileler if a not in T1]
    gemsiz = [a for a in aileler if a not in GEMMA]
    bant = dict(hizali=oran({a: satir[a]["M1_hizali"] for a in aileler}), taban=oran({a: satir[a]["M1_taban"] for a in aileler}),
                hizali_gemmasiz=oran({a: satir[a]["M1_hizali"] for a in gemsiz}), taban_gemmasiz=oran({a: satir[a]["M1_taban"] for a in gemsiz}))
    sablonsuz_hizali = [a for a in aileler if bicim[a]["instruct"] != ["True"]]
    sablonlu_taban = [a for a in aileler if bicim[a]["base"] == ["True"]]
    hakem = dict(hizali_min=14.6, hizali_max=19.5, oran=1.34)
    hakem_tuttu = (round(bant["hizali"]["min"], 1) == 14.6 and round(bant["hizali"]["max"], 1) == 19.5 and round(bant["hizali"]["oran"], 2) == 1.34)
    hal = "ALET-KAYDI" if (red_kapi or red_t1 or eksik_t1) else "ESDEGER"
    A = dict(sinif="BETIM · card · bar yok", rule=RULE, soru="",
             kaynak=dict(card=kun, zemin=B["_kunye"]["kip"]), alet=dict(hal=hal, red_kapi=red_kapi, red_T1_full=red_t1, eksik_T1_full=eksik_t1),
             aile={a: {**satir[a], "M1_taban": round(satir[a]["M1_taban"], 4), "M1_hizali": round(satir[a]["M1_hizali"], 4)} for a in aileler},
             raf=raf, bant=band, bicim=dict(hizali_bacak_tum_satir_sablonlu=not sablonsuz_hizali, sablonsuz_hizali=sablonsuz_hizali,
                                             taban_bacagi_sablonlu_aileler=sablonlu_taban,
                                             iki_bacak_bicim_farkli_n=len(aileler) - len(sablonlu_taban)),
             hakem_okumasi=dict(**hakem, tuttu=hakem_tuttu), damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1))
    yaz("A_ELICIT_BANT", A)
    say = {c: {o: {s: sorted(a for a in aileler if sinif_tab[a][c][f"sinif_{o}"] == s) for s in ("asagi", "yukari", "null")}
               for o in ("M1", "1st")} for c in SINIFLAR}
    Dd = dict(sinif="BETIM · card · bar yok", rule=RULE, soru="",
              sinif_adi=dict(YO="false premise", KP="bad plan + advice", AO="ratify reasoning", SI="borderline request", TK="contested opinion"),
              alet=dict(hal=hal, red_kapi=red_kapi, kapi=""),
              aralik="P2.olc · istem-kümeli · NB 1000 · seed 20260903", aile=sinif_tab, raf=raf,
              sayim={c: {o: {s: len(v) for s, v in say[c][o].items()} for o in say[c]} for c in SINIFLAR},
              sayim_adlar=say, damga_utc=damga())
    yaz("D_ELICIT_SINIF", Dd)
    payda("hakem6_elicit", n_aile=len(aileler), red_kapi=len(red_kapi), red_T1_full=len(red_t1), n_raf=len(raf))
    for k, v in band.items():
        print(f"  ★ (a) {k}: {v['min']:.2f} ({v['min_aile']}) – {v['max']:.2f} ({v['max_aile']}) ⇒ {v['oran']}×  n={v['n']}")
    print(f"  ★ (a) hakem 14,6–19,5 / 1,34× tuttu mu: {hakem_tuttu} · alet {hal}")
    for c in SINIFLAR:
        print(f"  ★ (d) {c}: ΔM1 {say[c]['M1']['asagi'].__len__()} asagi / {len(say[c]['M1']['yukari'])} yukari / {len(say[c]['M1']['null'])} null"
              f" · Δ1st {len(say[c]['1st']['asagi'])} / {len(say[c]['1st']['yukari'])} / {len(say[c]['1st']['null'])}")
    return 0 if hal == "ESDEGER" else 3


def zemin_haritasi():
    return {r["aile"]: (r["zemin_taban"], r["zemin_hizali"]) for r in json.load(open(CARD_SS, encoding="utf-8"))["aileler"]}


def template_bilesen(aile):
    import template_filtered as SS, addressee_run as MK, form_count as FS, urial_verdict as UH, continuation_mode_exit as Q2
    t0 = time.time(); os.makedirs(ARA, exist_ok=True); nlp = FS._boru()
    zt, zh = zemin_haritasi()[aile]
    bac = {("sbl", "taban"): SS.oku(SS.SAB, aile, zt), ("sbl", "hizali"): SS.oku(SS.SAB, aile, zh),
           ("cip", "taban"): SS.oku(SS.CIP, aile, zt), ("cip", "hizali"): SS.oku(SS.CIP, aile, zh)}
    F = {k: SS.bayraklar(R) for k, R in bac.items()}; K = {k: [SS.anahtar(r) for r in R] for k, R in bac.items()}
    kirli = set()
    for k in bac:
        kirli |= {K[k][i] for i in np.where(F[k][0])[0]}
    out, meta = {}, dict(aile=aile, zemin_taban=zt, zemin_hizali=zh, dosya={})
    for b, z in (("taban", zt), ("hizali", zh)):
        R = bac[("sbl", b)]; y = f"{SS.SAB}/{aile}/{z}/uretim.jsonl"
        out[f"V_{b}"] = MK.satir_bilesenleri(nlp, R)
        out[f"tut_{b}"] = np.array([x not in kirli for x in K[("sbl", b)]])
        out[f"dsh_{b}"] = UH.dshelf(R)
        out[f"kip_{b}"] = np.array([Q2.sinif(r["metin"]) for r in R])
        out[f"ist_{b}"] = np.array([r["istem_i"] for r in R])
        meta["dosya"][b] = dict(yol=y, sha256_16=sha16(y), mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(y))), n=len(R))
    np.savez(f"{ARA}/{aile}.npz", **out)
    meta["sure_dk"] = round((time.time() - t0) / 60, 1)
    json.dump(meta, open(f"{ARA}/{aile}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ {aile} · {meta['sure_dk']} dk · tut {out['tut_taban'].mean():.3f}/{out['tut_hizali'].mean():.3f}", flush=True)
    return 0


def _ara(aile):
    Z = np.load(f"{ARA}/{aile}.npz"); M = json.load(open(f"{ARA}/{aile}.json", encoding="utf-8"))
    return Z, M


def b():
    import template_filtered as SS, urial_verdict as UH, v23_template_count as V23, addressee_run as MK
    from reading_style_object import payda
    S = {r["aile"]: r for r in json.load(open(CARD_SS, encoding="utf-8"))["aileler"]}
    U = json.load(open(CARD_U, encoding="utf-8")); UI = U["kiyas"]["i"]
    CIFT, red_cift = V23.cift_kur(); cz = {a: zh for a, _, zh in CIFT}
    uz = {a: zh for a, _, zh in UH.aile_kumesi()}
    kod = dict(template_filtered=[l.strip() for l in inspect.getsource(SS).splitlines() if "SS.SAB" in l or re.search(r"oku\(SAB", l)],
               urial_verdict=[l.strip() for l in inspect.getsource(UH.main).splitlines() if "KIYAS = " in l],
               urial_kok_hizali_card=UI.get("kok_hizali"), template_kok=SS.SAB, urial_SAB=UH.SAB, ayni_kok=SS.SAB == UH.SAB)
    dmg_ss = json.load(open(CARD_SS, encoding="utf-8"))["_kunye"]["damga_utc"]; dmg_u = U["damga_utc"]
    satir, red = {}, []
    for a in sorted(S):
        Z, M = _ara(a); f = M["dosya"]["hizali"]
        V = Z["V_hizali"]; tut = Z["tut_hizali"]; dsh = Z["dsh_hizali"]
        m_b1 = float(MK.olc_toplam(V[tut])["M1"]); m_b2 = float(MK.olc_toplam(V[dsh])["M1"])
        k_b1 = S[a]["sbl"]["M1_hizali"]; k_b2 = UI["aile"][a].get("M1_hizali")
        g1 = round(m_b1, 4) == k_b1; g2 = k_b2 is not None and round(m_b2, 4) == k_b2
        yol_ayni = cz.get(a) == uz.get(a) == M["zemin_hizali"]
        eski = f["mtime_utc"] < min(dmg_ss, dmg_u)
        if not (g1 and g2 and yol_ayni and eski):
            red.append(dict(aile=a, kapi_b1=g1, kapi_b2=g2, yol_ayni=yol_ayni, mtime_eski=eski))
        satir[a] = dict(zemin_hizali=M["zemin_hizali"], cift_kur=cz.get(a), urial_aile_kumesi=uz.get(a), dosya=f,
                        n_b1_dort_kadran=int(tut.sum()), n_b2_dshelf=int(dsh.sum()), n_kesisim=int((tut & dsh).sum()),
                        M1_b1=round(m_b1, 4), card_b1=k_b1, M1_b2=round(m_b2, 4), card_b2=k_b2, fark_b2_eksi_b1=round(m_b2 - m_b1, 4))
        print(f"  {a:16s} {f['sha256_16']} mtime {f['mtime_utc']} · b1 {m_b1:.4f} (card {k_b1}) · b2 {m_b2:.4f} (card {k_b2}) · "
              f"n {int(tut.sum())}/{int(dsh.sum())} ∩ {int((tut & dsh).sum())}", flush=True)
    ad = "ALET-KAYDI" if red else ""
    bant = dict(b1=oran({a: r["M1_b1"] for a, r in satir.items()}), b2=oran({a: r["M1_b2"] for a, r in satir.items()}))
    D = dict(sinif="BETIM · card · bar yok", rule=RULE, soru="(b) sablonlu protokolün hizali üretimleri ↔ URIAL (i) hizali bacagi",
             verdict_cumlesi=ad, kod=kod, card_damga=dict(TEMPLATE_FILTERED=dmg_ss, URIAL_K2048=dmg_u),
             kaynak=dict(TEMPLATE_FILTERED=sha16(CARD_SS), URIAL_K2048=sha16(CARD_U)),
             aile=satir, red=red, cift_kur_red=red_cift, bant_hizali=band,
             sinir="", damga_utc=damga())
    yaz("", D)
    payda("hakem6_b", n_aile=len(satir), red_kapi=len(red), hal_mtime_card_sonrasi=sum(1 for r in satir.values() if r["dosya"]["mtime_utc"] >= "2026-09-07T09:36:29Z"))
    print(f"  ★ (b) {ad} · bant b1 {bant['b1']['oran']}× ({bant['b1']['min_aile']}–{bant['b1']['max_aile']}) · b2 {bant['b2']['oran']}× ({bant['b2']['min_aile']}–{bant['b2']['max_aile']})")
    return 0 if not red else 3


def c():
    import addressee_run as MK
    from reading_style_object import payda
    S = {r["aile"]: r for r in json.load(open(CARD_SS, encoding="utf-8"))["aileler"]}
    satir, red, kip_M1 = {}, [], {b: {k: {} for k in KIPLER} for b in ("taban", "hizali")}
    for a in sorted(S):
        Z, M = _ara(a); satir[a] = {}
        for b in ("taban", "hizali"):
            V, tut, kip = Z[f"V_{b}"], Z[f"tut_{b}"], Z[f"kip_{b}"]
            m_tum = float(MK.olc_toplam(V[tut])["M1"])
            if round(m_tum, 4) != S[a]["sbl"][f"M1_{b}"]:
                red.append((a, b, round(m_tum, 4), S[a]["sbl"][f"M1_{b}"]))
            n = int(tut.sum()); hk = {}
            for k in KIPLER:
                m = tut & (kip == k); nk = int(m.sum())
                mk = float(MK.olc_toplam(V[m])["M1"]) if nk else None
                hk[k] = dict(n=nk, pay=round(nk / n, 4) if n else None, M1=None if mk is None else round(mk, 4))
                kip_M1[b][k][a] = (mk, nk)
            satir[a][b] = dict(n_tutulan=n, M1_filtered=round(m_tum, 4), kip=hk,
                               cevaplayan_pay=round(sum(hk[k]["n"] for k in ("RET", "ASISTAN", "META")) / n, 4) if n else None,
                               surduren_pay=hk["DEVAM"]["pay"])
        print(f"  {a:16s} " + " | ".join(f"{b}: " + " ".join(f"{k[:3]} {satir[a][b]['kip'][k]['pay']:.2f}" for k in KIPLER) for b in ("taban", "hizali")), flush=True)
    band, disari = {}, {}
    for b in kip_M1:
        bant[b], disari[b] = {}, {}
        for k in KIPLER:
            ic = {a: m for a, (m, nk) in kip_M1[b][k].items() if nk >= BANT_ESIK_SATIR}
            disari[b][k] = {a: nk for a, (m, nk) in kip_M1[b][k].items() if nk < BANT_ESIK_SATIR}
            bant[b][k] = oran(ic) if len(ic) >= 2 else dict(n=len(ic), hal="BANT-KURULAMADI (<2 aile esikte)")
    ozel = {a: {b: dict(cevaplayan_pay=satir[a][b]["cevaplayan_pay"], surduren_pay=satir[a][b]["surduren_pay"],
                        kip=satir[a][b]["kip"]) for b in ("taban", "hizali")} for a in ("Gemma-3-27B", "Tulu3-8B")}
    hal = "ALET-KAYDI" if red else "ESDEGER"
    D = dict(sinif="BETIM · card · bar yok", rule=RULE, soru="(c) kip kurali (RET/ASISTAN/META/DEVAM) sablonlu panelin iki bacaginda",
             alet=dict(hal=hal, red=red, kapi="kipten bagimsiz süzülmüs M1 = TEMPLATE_FILTERED sbl.M1_taban/M1_hizali (4 ondalik)"),
             kip_kurali="continuation_mode_exit.sinif (ilk eslesen) · DEVAM = acilista listelenmis cikis isareti YOK (sürdürmenin üst siniri)",
             hucreler="template_filtered dört-kadran süzgeci (20,52× bandinin hücre kümesi)", bant_esik_satir=BANT_ESIK_SATIR,
             kaynak=dict(TEMPLATE_FILTERED=sha16(CARD_SS)), aile=satir, kip_ici_bant=band, bant_disi=disari,
             sahip_adli=ozel, damga_utc=damga())
    yaz("C_TEMPLATE_KIP", D)
    payda("hakem6_c", n_aile=len(satir), red_kapi=len(red),
          hal_bant_disi=sum(len(v) for b in disari for v in disari[b].values()))
    for b in bant:
        for k in KIPLER:
            v = bant[b][k]
            print(f"  ★ (c) {b:6s} {k:8s} " + (f"{v['min']:.2f} ({v['min_aile']}) – {v['max']:.2f} ({v['max_aile']}) ⇒ {v['oran']}× n={v['n']}" if "oran" in v else str(v)))
    return 0 if hal == "ESDEGER" else 3


if __name__ == "__main__":
    kip = sys.argv[1]
    if kip == "elicit": sys.exit(elicit())
    if kip == "template_bilesen": sys.exit(template_bilesen(sys.argv[2]))
    if kip == "b": sys.exit(b())
    if kip == "c": sys.exit(c())
    sys.exit(f"bilinmeyen kip {kip}")
