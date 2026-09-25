#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
import reviewer_readings_v80 as H80
import addressee_run as MK
import addressee_olcu as MO
import form_count as FS
from reading_style_object import payda

G = f"{ROOT}/results"
TARIH = "2026-09-16"
RULE = "preregistration/rule_v87_readings_2026-09-16.md @ 7ee8f32b"
LISTE_AILE = ("Gemma-3-4B", "Gemma-3-12B", "Gemma-3-27B", "Qwen2.5-1.5B", "Qwen2.5-32B", "Qwen2.5-72B")
N_PLASEBO, SEED = 200, 20260901
N_PROC = 8
LISTE_AILE = tuple(os.environ["V87_B_AILE"].split(",")) if os.environ.get("V87_B_AILE") else LISTE_AILE
SADECE_B = os.environ.get("V87_SADECE_B") == "1"
EK = os.environ.get("V87_EK", "")


def damga():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def yaz(ad, K):
    y = f"{G}/HAKEM_V87_{ad}_{TARIH}.json"
    json.dump(K, io.open(y, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {y}", flush=True)


def cumle_kayitlari(nlp, R):
    metin = [(r.get("metin") or "") for r in R]
    out = []
    for i, doc in enumerate(nlp.pipe(metin, n_process=N_PROC, batch_size=200)):
        D = MO.belge_olc(doc, metin[i])
        A = [(s.start_char, s.end_char) for s in doc.sents
             if any((not x.is_punct and not x.is_space) for x in s)]
        assert len(A) == len(D), f"cümle araligi {len(A)} ≠ sayac {len(D)} (satir {i})"
        out.append([(a, b, d) for (a, b), d in zip(A, D)])
    return out


def liste_satir_araliklari(t):
    ar, p = [], 0
    for ln in (t or "").split("\n"):
        if ln.strip() and H80.SATIR.match(ln):
            ar.append((p, p + len(ln) + 1))
        p += len(ln) + 1
    return ar


def vektor(kayit, sec=None):
    V = np.zeros((len(kayit), len(MK.ALAN) + 1))
    for i, cs in enumerate(kayit):
        for j, (a, b, d) in enumerate(cs):
            if sec is not None and not sec[i][j]:
                continue
            for k, ad in enumerate(MK.ALAN):
                V[i, k] += d[ad]
            V[i, -1] += 1
    return V


def duzyazi_secimi(R, kayit):
    sec, n_c, n_at = [], 0, 0
    for r, cs in zip(R, kayit):
        L = liste_satir_araliklari(r.get("metin"))
        s = [not any(a < le and b > ls for ls, le in L) for a, b, _ in cs]
        sec.append(s); n_c += len(s); n_at += sum(1 for x in s if not x)
    return sec, n_c, n_at


def plasebo_kol_bolme(V, ist, kol, n=N_PLASEBO, seed=SEED):
    rng = np.random.default_rng(seed)
    grup = {}
    for j, (u, k) in enumerate(zip(ist, kol)):
        grup.setdefault(u, {}).setdefault(k, []).append(j)
    ad = sorted(grup)
    kol_say = [len(grup[u]) for u in ad]
    out = []
    for _ in range(n):
        A, B = [], []
        for u in ad:
            ks = sorted(grup[u], key=str); pm = rng.permutation(len(ks)); h = len(ks) // 2
            for r_, ix in enumerate(pm):
                (A if r_ < h else B).extend(grup[u][ks[ix]])
        if not A or not B:
            continue
        out.append(MK.olc_toplam(V[A])["M1"] - MK.olc_toplam(V[B])["M1"])
    o = np.array(out, dtype=float); o = o[~np.isnan(o)]
    return dict(n=int(len(o)), ort=float(o.mean()), sd=float(o.std()),
                p95=float(np.percentile(np.abs(o), 95)),
                n_istem=len(ad), kol_istem_min=int(min(kol_say)), kol_istem_max=int(max(kol_say)))


def main():
    nlp = FS._boru(); t0 = time.time()
    B1 = H80.birincil()
    print(f"★ v87 IS 5(a)+(b) · {len(B1)} birincil aile · rule {RULE}", flush=True)
    A_out, B_out = {}, {}
    sap_a = sap_b_satir = sap_b_d = 0.0
    for n_i, (aile, v) in enumerate(sorted(B1.items()), 1):
        if SADECE_B and aile not in LISTE_AILE:
            continue
        bas, son = v["kontrast"].split("→")
        Rb, Rs = MK.oku(aile, bas), MK.oku(aile, son)
        xb, xs = H80.eslestir(Rb, Rs); Rb, Rs = [Rb[i] for i in xb], [Rs[i] for i in xs]
        ib = np.array([r["istem_i"] for r in Rb]); is_ = np.array([r["istem_i"] for r in Rs])
        kb = np.array([str(r.get("kol")) for r in Rb])
        Kb, Ks = cumle_kayitlari(nlp, Rb), cumle_kayitlari(nlp, Rs)
        Vb, Vs = vektor(Kb), vektor(Ks)
        goz = v["fark"]["M1"]["gozlenen"]
        s = abs((MK.olc_toplam(Vs)["M1"] - MK.olc_toplam(Vb)["M1"]) - goz); sap_a = max(sap_a, s)
        P = plasebo_kol_bolme(Vb, ib, kb) if not SADECE_B else dict(p95=float("nan"), kol_istem_min=0, kol_istem_max=0)
        gec = bool(abs(goz) > P["p95"])
        A_out[aile] = dict(kontrast=v["kontrast"], dM1_gozlenen=round(goz, 4), kapi_sapma=s,
                           plasebo_kol=P, gecti=gec, oran=round(abs(goz) / P["p95"], 3) if P["p95"] else None,
                           n_kol=len(set(kb)))
        ek = ""
        if aile in LISTE_AILE:
            ref_b, ref_s = MK.satir_bilesenleri(nlp, Rb), MK.satir_bilesenleri(nlp, Rs)
            sap_b_satir = max(sap_b_satir, float(np.abs(ref_b - Vb).max()), float(np.abs(ref_s - Vs).max()))
            sap_b_d = max(sap_b_d, s)
            sb, ncb, nab = duzyazi_secimi(Rb, Kb); ss, ncs, nas = duzyazi_secimi(Rs, Ks)
            Wb, Ws = vektor(Kb, sb), vektor(Ks, ss)
            fc = H80.fark_ci(Ws, Wb, is_, ib)
            B_out[aile] = dict(kontrast=v["kontrast"], duzyazi_cumle=fc, tam=dict(d=round(goz, 4)),
                               atilan_taban=dict(cumle_payi=round(nab / max(ncb, 1), 4),
                                                 jeton_payi=round(1 - Wb[:, 0].sum() / max(Vb[:, 0].sum(), 1), 4)),
                               atilan_hizali=dict(cumle_payi=round(nas / max(ncs, 1), 4),
                                                  jeton_payi=round(1 - Ws[:, 0].sum() / max(Vs[:, 0].sum(), 1), 4)))
            ek = (f" · (b) düzyazi-cümle ΔM1 {fc['d']:+.2f} {fc['ci']} {fc['sinif']} · atilan cümle "
                  f"taban {B_out[aile]['atilan_taban']['cumle_payi']:.3f} / hizali {B_out[aile]['atilan_hizali']['cumle_payi']:.3f}")
        gecen = time.time() - t0
        eta = gecen / n_i * (len(B1) - n_i) / 60
        print(f"  [{gecen/60:5.1f} dk · ETA {eta:4.1f} dk ⇒ esik 60 dk ⇒ EYLEM: asarsa bakis] {aile:14s} "
              f"(a) |ΔM1| {abs(goz):5.2f} vs p95 {P['p95']:5.2f} ⇒ {'GECTI' if gec else 'gecmedi'} "
              f"(kol/istem {P['kol_istem_min']}–{P['kol_istem_max']}) · kapi {s:.1e}{ek}", flush=True)

    hal_a = "ESDEGER" if sap_a <= 1e-6 else "ALET-KAYDI"
    g = sorted(a for a, r in A_out.items() if r["gecti"]); k = sorted(a for a, r in A_out.items() if not r["gecti"])
    if not SADECE_B: yaz("A_KOL_PLASEBO", dict(
        sinif="",
        rule=RULE, soru="IS 5(a) · istem icinde kol bölmesi plasebosu, taban bacagi, 200 cekim, p95",
        alet=dict(hal=hal_a, en_buyuk_sapma=sap_a, kapi="ΔM1 kartin fark.M1.gozlenen'ini ≤1e-6 yeniden üretir"),
        sayim=dict(gecti=len(g), gecmedi=len(k), payda=len(A_out)), gecen=g, gecmeyen=k,
        s1_yan_cumle=("GIRER" if len(g) >= 12 else "GIRMEZ") + " (rule: ≥12/16)",
        aile=A_out, damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)))
    hal_b = "ESDEGER" if (sap_b_satir <= 1e-9 and sap_b_d <= 1e-6) else "ALET-KAYDI"
    yaz("B_CUMLE_DUZYAZI" + EK, dict(
        sinif="",
        rule=RULE, soru=f"IS 5(b) · {len(B_out)} ailede cümle-birimli düzyazi ΔM1 [CI]"
                          + (" · kalan aileler" if EK else " · liste-bagimli aileler"),
        satir_deseni=H80.SATIR.pattern,
        birim="cümle: dokundugu satirlarin hicbiri desene uymuyorsa DÜZYAZI; tek ayristirma",
        alet=dict(hal=hal_b, satir_sapma=sap_b_satir, d_sapma=sap_b_d,
                  kapi=""),
        sayim={c: sum(1 for r in B_out.values() if r["duzyazi_cumle"]["sinif"] == c) for c in ("asagi", "null", "yukari")},
        aile=B_out, damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)))
    payda("hakem_v87", n_aile=len(A_out), hal_a_gecti=len(g), red_a_gecmedi=len(k),
          n_b_aile=len(B_out), hal_alet_a=int(hal_a == "ESDEGER"), hal_alet_b=int(hal_b == "ESDEGER"))
    return 0 if (hal_a == "ESDEGER" and hal_b == "ESDEGER") else 3


if __name__ == "__main__":
    sys.exit(main())
