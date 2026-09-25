#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import collections, glob, hashlib, io, json, os, sys, time
import numpy as np
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
import addressee_run as MK
import addressee_olcu as MO
import form_count as FS
from capability_arm_pronoun_free import TALIMAT_SAHISSIZ
from reading_style_object import payda

VERI = __DNH_DATA__ + ""
G = f"{KOK}/results"
CARD16 = f"{G}/YETENEK_KARNE_16_2026-09-13.json"
KISILI = f"{G}/urial_verdict_x1_2026-09-10.json"
KISISIZ = f"{G}/URIAL_VERDICT_KISISIZ_X1_2026-09-11.json"
CIK = f"{G}/capability_pronoun_free_threshold_floor_2026-09-14.json"
I2, IJ = MK.ALAN.index("m1_sahis2"), MK.ALAN.index("n_jeton")
IM2 = MK.ALAN.index("m2")


def sha(y):
    return hashlib.sha256(open(y, "rb").read()).hexdigest()[:16]


def yogunluk(nlp, sayac):
    dz = list(sayac.keys()); w = np.array([sayac[d] for d in dz], dtype=float)
    V = MK.satir_bilesenleri(nlp, [{"metin": d} for d in dz])
    s2 = float((V[:, I2] * w).sum()); nj = float((V[:, IJ] * w).sum())
    return dict(M1=1000 * s2 / nj, sahis2=s2, jeton=nj, n_benzersiz=len(dz), n_satir=int(w.sum()))


def main():
    t0 = time.time(); nlp = FS._boru(); out = dict()
    import capability_arm as YK
    L = dict(kayit=YK.TALIMAT["kayit"], acik=YK.TALIMAT["acik"], kayit_sahissiz=TALIMAT_SAHISSIZ)
    V = MK.satir_bilesenleri(nlp, [{"metin": t.strip()} for t in L.values()])
    out["lafiz"] = {k: dict(dize=t.strip(), sayac_m1_sahis2=int(V[i, I2]), sayac_n_jeton=int(V[i, IJ]),
                            sayac_m2_muhatap=int(V[i, IM2]),
                            regex_sahis2=MO.SAHIS2.findall(t), regex_sahis1=MO.SAHIS1.findall(t),
                            regex_hitap=bool(MO.HITAP.search(t)))
                    for i, (k, t) in enumerate(L.items())}
    for k, r in out["lafiz"].items():
        print(f"  [lafiz] {k:<15} sayac 2.sahis {r['sayac_m1_sahis2']} · regex 2.sahis {r['regex_sahis2']} · "
              f"regex 1.sahis {r['regex_sahis1']} · jeton {r['sayac_n_jeton']} · M2(muhatap) {r['sayac_m2_muhatap']}")
    y = out["lafiz"]["kayit_sahissiz"]
    if y["sayac_m1_sahis2"] or y["regex_sahis2"] or y["regex_sahis1"]:
        raise SystemExit("")

    ESKI = YK.TALIMAT["kayit"]
    dosyalar = sorted(glob.glob(f"{VERI}/c1_panel_yetenek/kayit/*/instruct/uretim.jsonl"))
    imza = {}; eski = collections.Counter(); yeni = collections.Counter()
    eski_yf = collections.Counter(); yeni_yf = collections.Counter()
    KT = json.load(open(f"{G}/KOL_TABLOSU_2026-09-10.json"))
    YF = {k for k, v in KT["onek"].items() if v["n_sahis2"] == 0}
    red_bas = 0
    for f in dosyalar:
        aile = f.split("/")[-3]; h = hashlib.sha256()
        for l in io.open(f, encoding="utf-8"):
            r = json.loads(l); o = r["onek"]
            if not o.startswith(ESKI):
                red_bas += 1; continue
            h.update(f"{r['kol']}|{r['istem_i']}|{r['cekim']}|{o}\n".encode())
            if aile == "Gemma-3-4B":
                n = TALIMAT_SAHISSIZ + o[len(ESKI):]
                eski[o] += 1; yeni[n] += 1
                if r["kol"] in YF:
                    eski_yf[o] += 1; yeni_yf[n] += 1
        imza[aile] = h.hexdigest()[:16]
    ayni = len(set(imza.values())) == 1
    payda("sahissiz_istem", n_dosya=len(dosyalar), hal_imza_tek=int(ayni), red_bas_eksik=red_bas)
    if len(dosyalar) != 16 or not ayni or red_bas:
        raise SystemExit(f"{imza} {red_bas}")
    dE, dY, dEy, dYy = (yogunluk(nlp, c) for c in (eski, yeni, eski_yf, yeni_yf))
    out["yetenek_istem"] = dict(n_aile_imza_ayni=16, eski=dE, yeni=dY, eski_youfree9=dEy, yeni_youfree9=dYy,
                                delta_d_yet=dE["M1"] - dY["M1"], delta_d_yet_youfree9=dEy["M1"] - dYy["M1"],
                                delta_sahis2_satir_basina=(dE["sahis2"] - dY["sahis2"]) / dE["n_satir"])
    print(f"  [istem] 16 kol: eski {dE['M1']:.3f} → yeni {dY['M1']:.3f} /1k ⇒ Δd_yet {dE['M1']-dY['M1']:.3f} · "
          f"9 you-free kol: {dEy['M1']:.3f} → {dYy['M1']:.3f} · satir basina Δ2.sahis "
          f"{out['yetenek_istem']['delta_sahis2_satir_basina']:.3f}")

    def urial(kok):
        c = collections.Counter(); im = set()
        for f in sorted(glob.glob(f"{VERI}/{kok}/*/base/uretim.jsonl")):
            h = hashlib.sha256()
            for l in io.open(f, encoding="utf-8"):
                r = json.loads(l)
                if r.get("tur") != "x1":
                    continue
                h.update(f"{r['kol']}|{r['istem_i']}|{r['cekim']}|{r['onek']}\n".encode())
                if "/Gemma-3-4B/" in f:
                    c[r["onek"]] += 1
            im.add(h.hexdigest())
        return c, len(im)
    cL, imL = urial("c1_panel_urial"); cS, imS = urial("c1_panel_urial_kisisiz")
    if imL != 1 or imS != 1:
        raise SystemExit(f"{imL} {imS}")
    uL, uS = yogunluk(nlp, cL), yogunluk(nlp, cS)
    dU = uL["M1"] - uS["M1"]
    out["urial_istem"] = dict(kisili=uL, kisisiz=uS, delta_d_urial=dU)
    print(f"  [URIAL istem, x1] kisili {uL['M1']:.3f} → kisisiz {uS['M1']:.3f} /1k ⇒ Δd_URIAL {dU:.3f}")

    KL = json.load(open(KISILI))["kiyas"]["i"]["aile"]
    KS = json.load(open(KISISIZ))["kiyas"]["i"]["aile"]
    K16 = json.load(open(CARD16))
    A = {}; red = {}
    for a, r in K16["aile"].items():
        l, s = KL.get(a, {}), KS.get(a, {})
        kl = ((l.get("kirpma") or {}).get("taban") or {}).get("oran_kirpilan", 0.0)
        ks = ((s.get("kirpma") or {}).get("taban") or {}).get("oran_kirpilan", 0.0)
        if l.get("hal") != "ÖLCÜLDÜ" or s.get("hal") != "ÖLCÜLDÜ" or kl > 0 or ks > 0:
            red[a] = dict(kisili_hal=l.get("hal"), kisisiz_hal=s.get("hal"), kirpik_kisili=kl, kirpik_kisisiz=ks)
            continue
        kay = l["M1_taban"] - s["M1_taban"]; sf = kay / dU
        E = sf * out["yetenek_istem"]["delta_d_yet"] / r["payda"]
        k = r["kayit"]; sd_gap = (k["ci"][1] - k["ci"][0]) / 3.92 / r["payda"]
        A[a] = dict(M1_taban_kisili=l["M1_taban"], M1_taban_kisisiz=s["M1_taban"], kayma=round(kay, 4),
                    egim=round(sf, 5), payda=r["payda"], E_M1=round(sf * out["yetenek_istem"]["delta_d_yet"], 4),
                    E_gap=round(E, 4), eski_kayit_sd_gap=round(sd_gap, 4))
    E = [v["E_gap"] for v in A.values()]
    SD = [v["eski_kayit_sd_gap"] for v in A.values()]
    out["egim"] = dict(aile=A, dislanan=red, n_aile=len(A), E_medyan=round(float(np.median(E)), 4),
                       E_min=round(min(E), 4), E_maks=round(max(E), 4),
                       kayma_min=round(min(v["kayma"] for v in A.values()), 4),
                       kayma_maks=round(max(v["kayma"] for v in A.values()), 4),
                       n_kayma_pozitif=sum(v["kayma"] > 0 for v in A.values()),
                       eski_kayit_sd_gap_bandi=[round(min(SD), 4), round(float(np.median(SD)), 4), round(max(SD), 4)])
    print(f"  [egim] {len(A)} aile (dislanan {sorted(red)}) · kayma {out['egim']['kayma_min']:+.2f}…"
          f"{out['egim']['kayma_maks']:+.2f} (pozitif {out['egim']['n_kayma_pozitif']}) · "
          f"E_gap medyan {out['egim']['E_medyan']:.4f} [{out['egim']['E_min']:.4f}, {out['egim']['E_maks']:.4f}] · "
          f"eski kayit aile-ici sd_gap bandi {out['egim']['eski_kayit_sd_gap_bandi']}")
    for a, v in sorted(A.items(), key=lambda x: x[1]["E_gap"]):
        print(f"      {a:<16} kayma {v['kayma']:+6.2f} · egim {v['egim']:.4f} · payda {v['payda']:.3f} · "
              f"E_M1 {v['E_M1']:.3f} · E_gap {v['E_gap']:.4f}")
    out["_kunye"] = dict(alet="scripts/capability_pronoun_free_threshold_floor.py", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         kaynak=dict(card16=[CARD16, sha(CARD16)], kisili=[KISILI, sha(KISILI)], kisisiz=[KISISIZ, sha(KISISIZ)]),
                         K_e="",
                         sure_dk=round((time.time() - t0) / 60, 1))
    io.open(CIK, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    payda("sahissiz_taban", n_aile_egim=len(A), red_dislanan=len(red), hal_dk=out["_kunye"]["sure_dk"])
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
