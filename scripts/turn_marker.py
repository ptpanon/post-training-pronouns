#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, re, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
import template_robustness_24 as S24
import template_robustness as SG
import urial_run as UK
from reading_style_object import payda

KOK = os.environ.get("TUR_KOK", __DNH_DATA__ + "/c1_panel_template")
CIK = os.environ.get("TUR_CIK", f"{ROOT}/results/turn_marker_2026-09-10.json")
ADAY = re.compile(r"<\|[a-zA-Z0-9_]+\|>|<start_of_turn>|<end_of_turn>|"
                  r"\[INST\]|\[/INST\]")


def aile_isaretleri(Rh):
    ss = set()
    for r in Rh[:64]:
        ss |= set(ADAY.findall(r.get("onek", "")))
    return sorted(ss)


def prova():
    a = "Yes you are right.<|user|>What about you? you you"
    b = "Yes you are right. No marker here."
    ISARET = re.compile("|".join(re.escape(x) for x in ["<|user|>", "<|im_start|>"]))
    m = ISARET.search(a)
    kesik = a[:m.start()] if m else a
    ok = (m is not None) and ISARET.search(b) is None and kesik.count("you") == 1
    payda("tur_isareti_prova", n_sinav=3, hal_yakaladi=int(m is not None),
          hal_kesti=kesik.count("you"), red_yanlis_pozitif=int(ISARET.search(b) is not None))
    print(f"{kesik!r}"
          f"{ISARET.search(b) is None}", flush=True)
    return ok


def main():
    if not prova():
        print("★ PROVA DÜSTÜ ⇒ ölcüm YAZILMAZ"); return 4
    nlp = FS._boru(); t0 = time.time()
    HAV = {k["ad"].split("·")[0]: dict(k)
           for k in (S24.KOLLAR + SG.KOLLAR + UK.cift_kollar())}
    AILE = [(a, HAV[a]["taban"].split("/", 1)[1], HAV[a]["hizali"].split("/", 1)[1])
            for a in UK.TEK_CARD + UK.CIFT_CARD]
    OUT = {}
    for a, zt, zh in AILE:
        Rt, Rh = MK.oku(a, zt, kok=KOK), MK.oku(a, zh, kok=KOK)
        if Rt is None or Rh is None:
            OUT[a] = dict(hal="BACAK-EKSIK"); continue
        _im = aile_isaretleri(Rh)
        if not _im:
            OUT[a] = dict(hal="TEMPLATE-ISARETI-CÖZÜLEMEDI"); print(f"  ✗ {a}: isaret yok"); continue
        ISARET = re.compile("|".join(re.escape(x) for x in _im))
        n_is = sum(1 for r in Rt if ISARET.search(r["metin"]))
        Vt = MK.satir_bilesenleri(nlp, Rt); Vh = MK.satir_bilesenleri(nlp, Rh)
        M1t = float(MK.olc_toplam(Vt)["M1"]); M1h = float(MK.olc_toplam(Vh)["M1"])
        kes = []
        for r in Rt:
            m = ISARET.search(r["metin"])
            kes.append({"metin": r["metin"][:m.start()] if m else r["metin"]})
        Vk = MK.satir_bilesenleri(nlp, kes)
        M1k = float(MK.olc_toplam(Vk)["M1"])
        s2_tam = float(Vt[:, MK.ALAN.index("m1_sahis2")].sum())
        s2_kes = float(Vk[:, MK.ALAN.index("m1_sahis2")].sum())
        OUT[a] = dict(hal="ÖLCÜLDÜ", n_satir=len(Rt), n_isaretli=n_is,
                      pay_isaretli=round(n_is / len(Rt), 4),
                      M1_taban=round(M1t, 4), M1_taban_kesik=round(M1k, 4),
                      M1_hizali=round(M1h, 4),
                      dM1=round(M1h - M1t, 4), dM1_kesik=round(M1h - M1k, 4),
                      pay_you_isaretten_sonra=round((s2_tam - s2_kes) / max(s2_tam, 1), 4),
                      isaretler=_im)
        print(f"{time.time()-t0:4.0f} {a:16s} {n_is:5d} {len(Rt)}"
              f"{100*n_is/len(Rt):4.1f} {M1t:6.2f} {M1k:6.2f}"
              f"{M1h-M1t:+7.2f} {M1h-M1k:+7.2f}"
              f"{100*(s2_tam-s2_kes)/max(s2_tam,1):.0f}", flush=True)
    ol = [v for v in OUT.values() if v.get("hal") == "ÖLCÜLDÜ"]
    payda("tur_isareti", n_aile=len(AILE), hal_olculen=len(ol),
          hal_isaret_goren_aile=sum(1 for v in ol if v["n_isaretli"] > 0),
          hal_etiketsiz_gorulmez=1, red_bacak_eksik=len(AILE) - len(ol))
    print(""
          "", flush=True)
    K = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="KESIF/BETIM — bar YOK · ad YOK · prediction YOK (§10)",
             alet="scripts/turn_marker.py", borc="D-0910-V32 §B/1.6",
             kok=KOK, isaret_kaynagi="",
             sinir="",
             aile=OUT)
    json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}  ({(time.time()-t0)/60:.1f} dk)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
