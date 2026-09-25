#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, os, sys, time
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
import gpu_lock as RK
import template_robustness as SG
import serita_harman102 as SA
import system_prompt_arm as MS
from reading_style_object import payda
import urial_run as UK

TALIMAT = {
 "kayit": ("Write the continuation as if you were talking with a friend: "
           "informal, personal and conversational rather than neutral or "
           "encyclopedic.\n\n"),
 "acik":  ("Write the continuation in the first and second person: refer to "
           "yourself as I and to the reader as you.\n\n"),
}
CIFT_CARD = MS.CIFT_CARD
CIK_KOK = __DNH_DATA__ + "/c1_panel_yetenek"
_GEREK = {"Llama-3.1-70B": 45.0, "Qwen2.5-72B": 45.0}
ZINCIR = dict(PREREG=f"{KOK}/preregistration/prereg_capability_2026-09-12.md",
              NOTICE=f"{KOK}/results/NOTICE_YETENEK_2026-09-12.md",
              PREDICTION=f"{KOK}/preregistration/prediction_capability_2026-09-12.md")


def onek_kurucu(talimat):
    def kur(tok):
        def f(ad, tur, c, ofs, i, ist, cekim):
            ham = SA.onek_kur(ad, tur, c, ofs, i, ist, cekim)
            return talimat + ham
        return f
    return kur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", default="kayit", choices=sorted(TALIMAT))
    ap.add_argument("--aile", default=None)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--coklu-card", dest="coklu_card", type=int, default=0)
    ap.add_argument("--kuru", action="store_true", help="")
    a = ap.parse_args()
    H = MS.hucreler()
    if a.aile:
        H = [h for h in H if h["ad"] == a.aile]
        if not H:
            print(f"{a.aile}"); return 4
    for y in ZINCIR.values():
        if not os.path.exists(y):
            print(f"★ B-43 DÜSTÜ — {os.path.basename(y)} YOK"); return 4
    SG.PREREG, SG.NOTICE, SG.PREDICTION = ZINCIR["PREREG"], ZINCIR["NOTICE"], ZINCIR["PREDICTION"]
    SG.ONEK_KURUCU = onek_kurucu(TALIMAT[a.kol])
    SG.CIKTI_KOK = f"{CIK_KOK}/{a.kol}"
    payda("yetenek_kol", n_hucre=len(H), n_zincir=3, n_kol=1)
    if a.kuru:
        _cagri, _asil = {}, SA.onek_kur
        try:
            def _vekil(*ar, **kw):
                _cagri["ar"] = ar
                return "<<HAM ISTEM>>"
            SA.onek_kur = _vekil
            _arg = ("DOM+x1", "DOM", (0, 1), 0, 0, {"tez": "t", "durus": "d"}, 0)
            _s = onek_kurucu(TALIMAT[a.kol])(None)(*_arg)
        finally:
            SA.onek_kur = _asil
        T = TALIMAT[a.kol]
        _ok = [_s.startswith(T), _s[len(T):] == "<<HAM ISTEM>>",
               _s == T + "<<HAM ISTEM>>", _cagri.get("ar") == _arg,
               "<|" not in _s and "<s>" not in _s]
        print(f"  [KURU] kol={a.kol} · hücre={len(H)} · cikti kökü={SG.CIKTI_KOK}")
        for _n, _t in zip(("(1) önek BASA konuyor", "",
                           "(3) baska ek YOK", "(4) argümanlar AYNEN gecti",
                           "(5) TEMPLATE isareti YOK"), _ok):
            print(f"  [KURU] {_n:<28} ⇒ {_t}")
        print(f"  [KURU] önek: {T!r}")
        print("")
        return 0 if all(_ok) else 4
    import torch as _t
    n_gor = _t.cuda.device_count() if _t.cuda.is_available() else 0
    t0 = time.time(); ariza = 0; atlanan = []; zaten = []; damgali = 0
    for h in H:
        if h["ad"] in CIFT_CARD and (n_gor < 2 or a.coklu_card < 2):
            atlanan.append(h["ad"])
            print(f"  ★ {h['ad']}: ATLANDI-CIFT-CARD-GEREK (görünen {n_gor}) ⇒ "
                  f"EYLEM: iki card bosken ayri kosuda", flush=True); continue
        _hedef = f"{h['ad']}/instruct"
        _ok, _m = UK.icerik(_hedef, kok=SG.CIKTI_KOK)
        if _ok:
            zaten.append(h["ad"])
            print(f"{h['ad']} {_m}", flush=True); continue
        import gc as _gc
        _gc.collect()
        if n_gor:
            _t.cuda.empty_cache(); _t.cuda.synchronize()
        _bos = min((_t.cuda.mem_get_info(i)[0] for i in range(n_gor)),
                   default=0) / 2**30
        _ger = h.get("gb_gerek", 0.0) or _GEREK.get(h["ad"], 0.0)
        print(f"{h['ad']} {_bos:.1f}"
              f"{_ger:.1f}"
              f"", flush=True)
        payda(f"yetenek_vram_{h['ad']}", n_card=n_gor, hal_bos_GiB=round(_bos, 1),
              hal_gerek_GiB=_ger, red_yetersiz=int(_ger > 0 and _bos < _ger))
        if _ger > 0 and _bos < _ger:
            atlanan.append(h["ad"])
            print(f"  ★ {h['ad']}: ATLANDI-VRAM-YETERSIZ (bos {_bos:.1f} < "
                  f"{_ger:.1f} GiB) ⇒ EYLEM: ayri kosuda", flush=True); continue
        SG.KOLLAR = [dict(ad=h["ad"], taban=f"{h['ad']}/base",
                          hizali=f"{h['ad']}/instruct",
                          s_taban=h["s_taban"], s_hizali=h["s_hizali"],
                          panel="c1", ikiz="ham", olculen_dk=0.0)]
        try:
            SG.uret(argparse.Namespace(kol=h["ad"], zemin="instruct", dev=a.dev,
                                       bekle_gpu=RK.fiziksel_bekle(a.dev),
                                       coklu_card=a.coklu_card))
        except SystemExit as e:
            print(f"  ★★ {h['ad']}: {e}"); ariza += 1
        except Exception as e:
            print(f"  ★★★ {h['ad']}: {type(e).__name__}: {e}"); ariza += 1
        else:
            _ok2, _m2 = UK.icerik(_hedef, kok=SG.CIKTI_KOK)
            if _ok2:
                _dz = os.path.join(SG.CIKTI_KOK, _hedef)
                with open(os.path.join(_dz, ".tamam"), "w", encoding="utf-8") as _f:
                    _f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                             + f" · {_m2}\n")
                damgali += 1
            else:
                print(f"  ★ .tamam YAZILMADI — {_hedef}: {_m2}", flush=True)
    print(f"  [PAYDA] yetenek_uret: kol={a.kol} · n_hucre={len(H)} · "
          f"hal_kosan={len(H)-len(atlanan)-len(zaten)} · "
          f"red_atlanan_cift_card={len(atlanan)} · hal_zaten_var={len(zaten)} · "
          f"hal_damgali={damgali} · red_ariza={ariza} · "
          f"hal_dk={(time.time()-t0)/60:.1f} ⇒ esik: red_ariza>0 ⇒ rc=3")
    return 0 if ariza == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
