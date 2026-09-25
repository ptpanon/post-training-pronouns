#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
import rung_force_ci as BK
import dejenerelik_olcer as D
import v23_template_count as V23
import urial_run as UK
import template_robustness_24 as S24
import template_robustness as SG
import serita_harman102 as SA
import k1b_generation as K1BU
from reading_style_object import payda

URIAL = os.environ.get("URIAL_VERDICT_KOK",
                       __DNH_DATA__ + "/c1_panel_urial")
SAB = __DNH_DATA__ + "/c1_panel_template"
CIK = os.environ.get("URIAL_VERDICT_CIK", f"{ROOT}/results/urial_verdict_2026-09-09.json")
REPORT = os.environ.get("URIAL_REPORT", f"{ROOT}/results/REPORT_URIAL_2026-09-09.md")
N_BEK = 6528
N_PLASEBO = int(os.environ.get("URIAL_N_PLASEBO", "200"))
SEED = 20260909
ORNEK_M1 = 16.29
BANT_CIPLAK_HIZALI = (2.90, 9.80)

KIRPMA_MAXLEN = int(os.environ.get("URIAL_KIRPMA_MAXLEN", "1024"))
KOL_KIP = os.environ.get("URIAL_KOLLAR", "")


def x1_kollar():
    ad = [k[0] for k in K1BU.kollar(ton=True)]
    x1 = [a for a in ad if a.endswith("x1")]
    payda("urial_verdict_kol_kipi", n_kol_tablo=len(ad), hal_x1=len(x1),
          red_beklenmeyen=abs(6 - len(x1)))
    return set(x1)


def kol_suz(R, kip, x1):
    if not kip:
        return R
    return [r for r in R if str(r.get("kol", "")) in x1]


def kirpma_denetimi(snap_ad, R, _onb={}):
    if not R:
        return dict(hal="SATIR-YOK", n=0)
    if snap_ad not in _onb:
        from transformers import AutoTokenizer
        _onb[snap_ad] = AutoTokenizer.from_pretrained(SA._snapshot(None, model_adi=snap_ad))
    tok = _onb[snap_ad]
    tekil = sorted({r.get("onek", "") for r in R})
    L = dict(zip(tekil, (len(x) for x in tok(tekil, truncation=False)["input_ids"])))
    uz = [L[r.get("onek", "")] for r in R]
    n_k = sum(1 for x in uz if x > KIRPMA_MAXLEN)
    return dict(hal="ÖLCÜLDÜ", n=len(uz), tepe_jeton=max(uz), dip_jeton=min(uz),
                max_length=KIRPMA_MAXLEN, n_kirpilan=n_k,
                oran_kirpilan=round(n_k / len(uz), 4))


def aile_kumesi():
    hav = {k["ad"].split("·")[0]: dict(k)
           for k in (S24.KOLLAR + SG.KOLLAR + UK.cift_kollar())}
    out = [(ad, hav[ad]["taban"].split("/", 1)[1], hav[ad]["hizali"].split("/", 1)[1])
           for ad in UK.TEK_CARD + UK.CIFT_CARD]
    payda("urial_verdict_evren", n_aile=len(out), n_beklenen=16,
          red_eksik=abs(16 - len(out)))
    return out


def oku(kok, aile, zemin):
    y = f"{kok}/{aile}/{zemin}/uretim.jsonl"
    if not os.path.exists(y):
        return None
    return [json.loads(l) for l in io.open(y, encoding="utf-8")]


def dshelf(R):
    return np.array([not bool(D.bayrakla(r["metin"])["f_yin4"]) for r in R])


def esli_plasebo(Vh, ih, Vt, it, n=N_PLASEBO, seed=SEED):
    ka = {u: np.where(ih == u)[0] for u in np.unique(ih)}
    kb = {u: np.where(it == u)[0] for u in np.unique(it)}
    ortak = [u for u in ka if u in kb]
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        ai, bi = [], []
        for u in ortak:
            A, B = ka[u], kb[u]
            hepsi_h = Vh[A]; hepsi_t = Vt[B]
            if rng.random() < 0.5:
                ai.append(hepsi_h); bi.append(hepsi_t)
            else:
                ai.append(hepsi_t); bi.append(hepsi_h)
        out.append(MK.olc_toplam(np.concatenate(ai))["M1"]
                   - MK.olc_toplam(np.concatenate(bi))["M1"])
    o = np.array(out); return o[~np.isnan(o)]


def _ad_kirpmali(ad, kir_t, kir_h):
    if not KOL_KIP:
        return ad
    if kir_t.get("n_kirpilan", 0) or kir_h.get("n_kirpilan", 0):
        return "ÖLCÜLEMEZ-KIRPMA"
    return ad


def verdict(dM1, mde, n_bacak):
    if n_bacak < 8:
        return "ÖLCÜLEMEZ-KAPSAM"
    if abs(dM1) < mde:
        return "BICIM-ÖZELLIGI"
    return "AGIRLIK-ÖZELLIGI" if dM1 < 0 else "TERS"


def main():
    kuru = "--kuru" in sys.argv
    global URIAL
    if "--kok" in sys.argv:
        URIAL = sys.argv[sys.argv.index("--kok") + 1]
        if CIK.endswith("urial_verdict_2026-09-09.json"):
            print(""
                  "")
            return 4
    global KOL_KIP
    if "--kollar" in sys.argv:
        KOL_KIP = sys.argv[sys.argv.index("--kollar") + 1]
    if KOL_KIP and KOL_KIP != "x1":
        print(f"{KOL_KIP}"); return 4
    if KOL_KIP and CIK.endswith("urial_verdict_2026-09-09.json"):
        print("★ --kollar verildi ama cikti yolu VARSAYILAN ⇒ mühürlü card "
              "EZILIRDI ⇒ DUR."); return 4
    X1 = x1_kollar() if KOL_KIP else set()
    HAV = {k["ad"].split("·")[0]: dict(k)
           for k in (S24.KOLLAR + SG.KOLLAR + UK.cift_kollar())}
    print(f"  [PAYDA] urial_verdict_kok: kok={URIAL} · cik={os.path.basename(CIK)} "
          f"· kol_kipi={KOL_KIP or 'hepsi(16)'} "
          f"⇒ esik: kök/kip varsayilan ciktiyla eslesirse DUR", flush=True)
    nlp = FS._boru(); t0 = time.time()
    CIFT = aile_kumesi()
    KIYAS = {"ii": ("ayni dize", URIAL, URIAL), "i": ("görev esit", URIAL, SAB)}
    SONUC = {}
    for etiket, (tarif, kok_t, kok_h) in KIYAS.items():
        satir, n_bacak, n_suz, n_kir = {}, 0, 0, 0
        for a, zt, zh in CIFT:
            Rt, Rh = oku(kok_t, a, zt), oku(kok_h, a, zh)
            if Rt is None or Rh is None or len(Rt) != N_BEK or len(Rh) != N_BEK:
                satir[a] = dict(hal="BACAK-EKSIK",
                                taban=(None if Rt is None else len(Rt)),
                                hizali=(None if Rh is None else len(Rh)))
                continue
            if kuru:
                satir[a] = dict(hal="KURU", taban=len(Rt), hizali=len(Rh)); n_bacak += 1
                continue
            Rt = kol_suz(Rt, KOL_KIP, X1); Rh = kol_suz(Rh, KOL_KIP, X1)
            if not Rt or not Rh:
                satir[a] = dict(hal="SÜZGEC-BOS", taban=len(Rt), hizali=len(Rh)); continue
            kir_t = kirpma_denetimi(HAV[a]["s_taban"], Rt)
            kir_h = kirpma_denetimi(HAV[a]["s_hizali"], Rh)
            n_suz += len(Rt) + len(Rh)
            n_kir += kir_t.get("n_kirpilan", 0) + kir_h.get("n_kirpilan", 0)
            mt, mh = dshelf(Rt), dshelf(Rh)
            Vt = MK.satir_bilesenleri(nlp, Rt); Vh = MK.satir_bilesenleri(nlp, Rh)
            it = np.array([x.get("istem_i", -1) for x in Rt])
            ih = np.array([x.get("istem_i", -1) for x in Rh])
            M1t = float(MK.olc_toplam(Vt[mt])["M1"]); M1h = float(MK.olc_toplam(Vh[mh])["M1"])
            d = M1h - M1t
            lo, hi, nk = BK.kume_boot(Vh[mh], ih[mh], Vt[mt], it[mt], "M1")
            nul = esli_plasebo(Vh[mh], ih[mh], Vt[mt], it[mt])
            nsd = float(nul.std()); nmerkez = float(nul.mean())
            mde = 1.645 * nsd
            satir[a] = dict(hal="ÖLCÜLDÜ", M1_taban=round(M1t, 4), M1_hizali=round(M1h, 4),
                            dM1=round(d, 4), ci=[round(lo, 4), round(hi, 4)], n_kume=nk,
                            dshelf_tutulan=[round(float(mt.mean()), 4), round(float(mh.mean()), 4)],
                            null_sd=round(nsd, 4), null_merkez=round(nmerkez, 4),
                            MDE=round(mde, 4),
                            null_merkezli=bool(abs(nmerkez) / (nsd or 1) <= 1.645),
                            plasebo_frac=round(float((np.abs(nul) >= abs(d)).mean()), 4),
                            kirpma=dict(taban=kir_t, hizali=kir_h),
                            AD=_ad_kirpmali(verdict(d, mde, 99), kir_t, kir_h))
            n_bacak += 1
            print(f"  [{etiket}] {a:16s} ΔM1 {d:+7.2f} [{lo:+.2f},{hi:+.2f}] "
                  f"· MDE {mde:.2f} · n={len(Rt)}/{len(Rh)} "
                  f"· kirpik {kir_t.get('n_kirpilan',0)}/{kir_h.get('n_kirpilan',0)} "
                  f"· {satir[a]['AD']}", flush=True)
        olculen = [v for v in satir.values() if v.get("hal") == "ÖLCÜLDÜ"]
        payda(f"urial_verdict_{etiket}", n_aile=len(CIFT), hal_olculen=len(olculen),
              hal_satir_suzulen=n_suz, hal_hucre_kirpik=n_kir,
              red_bacak_eksik=sum(1 for v in satir.values() if v.get("hal") == "BACAK-EKSIK"),
              red_suzgec_bos=sum(1 for v in satir.values() if v.get("hal") == "SÜZGEC-BOS"))
        SONUC[etiket] = dict(tarif=tarif, kok_taban=kok_t, kok_hizali=kok_h,
                             kol_kipi=(KOL_KIP or "hepsi"), n_kol=(len(X1) if KOL_KIP else 16),
                             n_satir_suzulen=n_suz, n_hucre_kirpik=n_kir,
                             n_aile=len(CIFT), n_olculen=len(olculen), aile=satir,
                             n_bacak_eksik=sum(1 for v in satir.values()
                                               if v.get("hal") == "BACAK-EKSIK"),
                             KAPSAM_HUKMU=("ÖLCÜLEMEZ-KAPSAM" if len(olculen) < 8 else "OKUNUR"),
                             sayim=dict(
                                 asagi=sum(1 for v in olculen if v["AD"] == "AGIRLIK-ÖZELLIGI"),
                                 bicim=sum(1 for v in olculen if v["AD"] == "BICIM-ÖZELLIGI"),
                                 ters=sum(1 for v in olculen if v["AD"] == "TERS"),
                                 olculemez_kirpma=sum(1 for v in olculen
                                                      if v["AD"] == "ÖLCÜLEMEZ-KIRPMA")))
    K = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             alet="scripts/urial_verdict.py", borc="sahip kelimesi 2026-09-09 §1",
             prereg_1=dict(dosya="prereg_urial_2026-09-07.md", commit="31a8f13c", kiyas="ii"),
             prereg_2=dict(dosya="prereg_urial_2_2026-09-07.md", commit="c22249dd", kiyas="i"),
             rakip_aciklama=dict(ornek_M1=ORNEK_M1, ciplak_hizali_bandi=BANT_CIPLAK_HIZALI,
                                 serh=""),
             kol_kipi=(KOL_KIP or "hepsi"),
             kirpma=dict(max_length=KIRPMA_MAXLEN, alet="serita_harman102._kol_uret",
                         serh="W-1016: üretimde GÖMÜLÜYDÜ; artik PROJECT_ISTEM_MAXLEN"),
             yeniden_okuma=bool(KOL_KIP),
             n_plasebo=N_PLASEBO, seed=SEED, kiyas=SONUC,
             gecen_dk=round((time.time() - t0) / 60, 1))
    json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}", flush=True)
    with io.open(REPORT, "w", encoding="utf-8") as fh:
        _kip = KOL_KIP or "hepsi"
        fh.write(f"# DÖNÜS-URIAL (taslak · sahip okur)\n\n*damga: {K['damga_utc']}*\n\n"
                 f"· kol kipi: **{_kip}**"
                 + (""
                    ""
                    if KOL_KIP else " (on alti kol, mühürlü küme)") + ""
                 f"{KIRPMA_MAXLEN}"
                 f"")
        for e, S in SONUC.items():
            fh.write(f"{e} {S['tarif']}"
                     f"{S['n_olculen']} {S['n_aile']}"
                     f"{S['n_bacak_eksik']}"
                     f"{S['KAPSAM_HUKMU']} {S['sayim']['asagi']}"
                     f"{S['sayim']['bicim']} {S['sayim']['ters']}"
                     + (f" · ölcülemez-kirpma **{S['sayim']['olculemez_kirpma']}**"
                        if S['sayim'].get('olculemez_kirpma') else "")
                     + f"{S['n_satir_suzulen']}"
                       f"{S['n_hucre_kirpik']}"
                     f""
                     f"")
            for a, v in S["aile"].items():
                if v.get("hal") != "ÖLCÜLDÜ":
                    fh.write(f"| {a} | — | — | — | — | {v.get('hal')} |\n"); continue
                _kk = v.get("kirpma", {})
                fh.write(f"| {a} | {v['dM1']:+.2f} | [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] "
                         f"| {v['MDE']:.2f} "
                         f"| {_kk.get('taban',{}).get('n_kirpilan','—')}/"
                         f"{_kk.get('hizali',{}).get('n_kirpilan','—')} "
                         f"| {v['AD']} |\n")
            _kok_et = "kisisiz" if "kisisiz" in URIAL else "kisili"
            _t = [v["M1_taban"] for v in S["aile"].values() if v.get("hal") == "ÖLCÜLDÜ"]
            _h = [v["M1_hizali"] for v in S["aile"].values() if v.get("hal") == "ÖLCÜLDÜ"]
            _eks = [a for a, v in S["aile"].items() if v.get("hal") != "ÖLCÜLDÜ"]
            if _t:
                _t2, _h2 = sorted(_t), sorted(_h)
                _ort = lambda z: (z[len(z)//2] if len(z) % 2 else (z[len(z)//2-1]+z[len(z)//2])/2)
                fh.write(f""
                         f""
                         f""
                         f"{_kok_et} {min(_t2):.2f} {max(_t2):.2f}"
                         f"{_ort(_t2):.2f} {sum(_t2)/len(_t2):.2f} {len(_t2)}"
                         f"{min(_h2):.2f} {max(_h2):.2f}"
                         f"{_ort(_h2):.2f} {sum(_h2)/len(_h2):.2f} {len(_h2)}"
                         f"{S['n_aile']} {len(_t2)}"
                         + (f", disarida kalan **adiyla**: {', '.join(_eks)}" if _eks else "")
                         + ".\n")
            fh.write(f""
                     f"{ORNEK_M1} {BANT_CIPLAK_HIZALI}")
    print(f"✓ {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
