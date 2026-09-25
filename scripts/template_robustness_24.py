#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import template_robustness as SG
import full_tarama as TT

CIK = f"{ROOT}/results/template_robustness_24_2026-08-27.json"
PREREG = f"{ROOT}/preregistration/prereg_template_robustness_24_2026-08-27.md"
PREDICTION = f"{ROOT}/preregistration/prediction_24.md"
NOTICE = f"{ROOT}/results/NOTICE_TEMPLATE_24_2026-08-27.md"
ESKI9 = f"{ROOT}/results/template_robustness_2026-08-26.json"

BOS_VRAM_GIB = 68.7

KOLLAR = [
    dict(ad="Qwen2.5-1.5B", taban="Qwen2.5-1.5B/base", hizali="Qwen2.5-1.5B/instruct",
         s_taban="models--Qwen--Qwen2.5-1.5B", s_hizali="models--Qwen--Qwen2.5-1.5B-Instruct",
         panel="f5", ikiz="ham", olculen_dk=16.8, olculen_vram_gib=3.26),
    dict(ad="Qwen2.5-3B", taban="Qwen2.5-3B/base", hizali="Qwen2.5-3B/instruct",
         s_taban="models--Qwen--Qwen2.5-3B", s_hizali="models--Qwen--Qwen2.5-3B-Instruct",
         panel="olcek2", ikiz="ham", olculen_dk=19.9, olculen_vram_gib=6.33),
    dict(ad="Qwen2.5-14B", taban="Qwen2.5-14B/base", hizali="Qwen2.5-14B/instruct",
         s_taban="models--Qwen--Qwen2.5-14B", s_hizali="models--Qwen--Qwen2.5-14B-Instruct",
         panel="f5", ikiz="ham", olculen_dk=32.8, olculen_vram_gib=29.05),
    dict(ad="Mistral-Small-24B", taban="Mistral-Small-24B/base", hizali="Mistral-Small-24B/instruct",
         s_taban="models--mistralai--Mistral-Small-24B-Base-2501",
         s_hizali="models--mistralai--Mistral-Small-24B-Instruct-2501",
         panel="f5", ikiz="ham", olculen_dk=32.9, olculen_vram_gib=45.43),
    dict(ad="Gemma-3-27B", taban="Gemma-3-27B/base", hizali="Gemma-3-27B/instruct",
         s_taban="models--google--gemma-3-27b-pt", s_hizali="models--google--gemma-3-27b-it",
         panel="f5", ikiz="ham", olculen_dk=58.1, olculen_vram_gib=54.64),
    dict(ad="Gemma-4-E4B", taban="Gemma-4-E4B/base", hizali="Gemma-4-E4B/instruct",
         s_taban="models--google--gemma-4-E4B", s_hizali="models--google--gemma-4-E4B-it",
         panel="(panel YOK)", ikiz="ham", olculen_dk=39.4, olculen_vram_gib=14.81),
    dict(ad="Qwen2.5-32B", taban="Qwen2.5-32B/base", hizali="Qwen2.5-32B/instruct",
         s_taban="models--Qwen--Qwen2.5-32B", s_hizali="models--Qwen--Qwen2.5-32B-Instruct",
         panel="olcek", ikiz="ham", olculen_dk=50.7, olculen_vram_gib=61.03),
    dict(ad="Gemma-4-12B", taban="Gemma-4-12B/base", hizali="Gemma-4-12B/instruct",
         s_taban="models--google--gemma-4-12b", s_hizali="models--google--gemma-4-12B-it",
         panel="(panel YOK)", ikiz="ham", olculen_dk=53.2, olculen_vram_gib=22.35),
    dict(ad="Gemma-4-26B-A4B", taban="Gemma-4-26B-A4B/base", hizali="Gemma-4-26B-A4B/instruct",
         s_taban="models--google--gemma-4-26B-A4B", s_hizali="models--google--gemma-4-26B-A4B-it",
         panel="olcek", ikiz="ham", olculen_dk=159.5, olculen_vram_gib=48.18),
]
VRAM_KAPI = [
    dict(ad="Llama-3.1-70B", olculen_dk=90.8, olculen_vram_gib=67.21, coklu_card=2),
    dict(ad="Qwen2.5-72B",   olculen_dk=93.9, olculen_vram_gib=70.81, coklu_card=2),
]

EKSEN_KARAR = SG.EKSEN_KARAR
KOL_ADLARI = ("AYNI-AD", "AD-DEGISTI", "KAPI")
PANEL_ADLARI = ("SABLONA-GÜRBÜZ", "SABLONA-BAGIMLI", "KAPI")
BAR_AYNI = 2.0 / 3.0


def verdict_panel(adlar):
    olc = [a for a in adlar if a != "KAPI"]
    if len(olc) < 2:
        return "KAPI"
    return ("SABLONA-GÜRBÜZ" if sum(1 for a in olc if a == "AYNI-AD") >= BAR_AYNI * len(olc)
            else "SABLONA-BAGIMLI")


def sinif(k):
    v = k.get("olculen_vram_gib")
    if v is None:
        return "VRAM-ÖLCÜLMEMIS"
    return "ölcülebilir" if v <= BOS_VRAM_GIB else "VRAM-KAPI"


def snapshot_kapisi():
    KOK = __DNH_DATA__ + "/c1_panel"
    KOKLER = ("<storage>/huggingface/hub", f"{ROOT}/.hf/hub",
              os.path.expanduser("~/.cache/huggingface/hub"))
    kusur = []
    for k in KOLLAR:
        for alan, zemin in (("s_taban", "base"), ("s_hizali", "instruct")):
            ky = f"{KOK}/{k['ad']}/{zemin}/uretim_kunye.json"
            if not os.path.exists(ky):
                kusur.append(f"{k['ad']}/{zemin}: künye YOK"); continue
            g = [x for x in json.load(open(ky, encoding="utf-8"))["model"].split("/")
                 if x.startswith("models--")][0]
            if k[alan] != g:
                kusur.append(f"{k['ad']}.{alan}: tabloda {k[alan]} ≠ künyede {g}")
            elif not any(os.path.isdir(f"{kk}/{k[alan]}") for kk in KOKLER):
                kusur.append(f"{k['ad']} {alan}")
    payda("s24_snapshot_kapisi", n_kol=len(KOLLAR), n_alan=2 * len(KOLLAR),
          red_uyusmaz=len(kusur))
    print(f""
          f"")
    if kusur:
        print("★ SNAPSHOT KAPISI DÜSTÜ:\n   " + "\n   ".join(kusur))
        return 3
    return 0


def eta_kapisi(kollar):
    tot = sum(k["olculen_dk"] for k in kollar)
    payda("s24_eta", n_kol=len(kollar), hal_toplam_dk=round(tot, 1),
          hal_toplam_saat=round(tot / 60, 2))
    print(f"{tot:.1f} {tot/60:.2f}"
          f"")
    for k in kollar:
        print(f"   {k['ad']:20s} ölcülen {k['olculen_dk']:5.1f} dk ⇒ tavan "
              f"{2*k['olculen_dk']:5.1f} · alt-kapi {0.25*k['olculen_dk']:5.1f} dk "
              f"(K-2g) · VRAM {k['olculen_vram_gib']} GiB")
    return tot


def vram_provasi(a):
    import torch
    from transformers import AutoModelForCausalLM
    import weight_dosyalari as AGD
    HUB2 = ("<storage>/huggingface/hub", f"{ROOT}/.hf/hub",
            os.path.expanduser("~/.cache/huggingface/hub"))
    out = {}
    for k in [x for x in KOLLAR if sinif(x) == "VRAM-ÖLCÜLMEMIS"]:
        snap = None
        for kok in HUB2:
            p = f"{kok}/{k['s_hizali']}"
            if os.path.isdir(p) and (s := AGD.sec(p)):
                snap = s; break
        if snap is None:
            out[k["ad"]] = dict(hal="SNAPSHOT-YOK"); continue
        torch.cuda.reset_peak_memory_stats()
        try:
            m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).to(a.dev)
            v = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            out[k["ad"]] = dict(hal="ölcülebilir" if v <= BOS_VRAM_GIB else "VRAM-KAPI",
                                tepe_vram_gib=v)
            del m
        except Exception as e:
            out[k["ad"]] = dict(hal="VRAM-KAPI", hata=type(e).__name__)
        torch.cuda.empty_cache()
        print(f"   {k['ad']:20s} → {out[k['ad']]}", flush=True)
    payda("s24_vram_provasi", n_kol=len(out),
          hal_olculebilir=sum(1 for v in out.values() if v["hal"] == "ölcülebilir"),
          red_kapi=sum(1 for v in out.values() if v["hal"] != "ölcülebilir"))
    json.dump(out, open(f"{ROOT}/results/TEMPLATE_24_VRAM_2026-08-27.json", "w"),
              ensure_ascii=False, indent=1)
    return 0


def uret(a):
    SG.KOLLAR, SG.PREREG, SG.PREDICTION, SG.NOTICE = KOLLAR, PREREG, PREDICTION, NOTICE
    return SG.uret(a)


def coz(a):
    SG.KOLLAR, SG.PREREG, SG.PREDICTION, SG.NOTICE, SG.CIK = KOLLAR, PREREG, PREDICTION, NOTICE, CIK
    return SG.coz(a)


def _tam_tarama():
    R_kol = TT.tam_tarama(lambda ayni, kapi: ("KAPI" if kapi else
                                              ("AYNI-AD" if ayni else "AD-DEGISTI")),
                          dict(ayni=[False, True], kapi=[False, True]),
                          KOL_ADLARI, etiket="s24_kol_izgara")
    R_pan = TT.tam_tarama(lambda a, b, c: verdict_panel([a, b, c]),
                          dict(a=list(KOL_ADLARI), b=list(KOL_ADLARI), c=list(KOL_ADLARI)),
                          PANEL_ADLARI, etiket="s24_panel_izgara")
    TT.bas(R_kol, "TEMPLATE-24 · KOL"); TT.bas(R_pan, "TEMPLATE-24 · PANEL")
    rc = TT.kapi(R_kol, R_pan, etiket="s24_tam_tarama_kapisi")
    json.dump(dict(kol=R_kol, panel=R_pan, rc=rc),
              open(f"{ROOT}/results/TEMPLATE_24_TAM_TARAMA_2026-08-27.json", "w"),
              ensure_ascii=False, indent=1, default=str)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", dest="kip",
                    choices=("eta", "vram", "uret", "coz", "tarama"), required=True)
    ap.add_argument("--kol")
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", dest="bekle_gpu", type=int, default=0)
    ap.add_argument("--prereg", default="c1071985")
    ap.add_argument("--prediction", default="c4741041")
    a, _ = ap.parse_known_args()
    if a.kip == "tarama":
        return _tam_tarama()
    kos = [k for k in KOLLAR if sinif(k) == "ölcülebilir"]
    if a.kip == "eta":
        payda("s24_sinif", n_kol=len(KOLLAR) + len(VRAM_KAPI),
              hal_olculebilir=len(kos),
              hal_vram_olculmemis=sum(1 for k in KOLLAR if sinif(k) == "VRAM-ÖLCÜLMEMIS"),
              red_vram_kapi=len(VRAM_KAPI))
        rc = snapshot_kapisi()
        eta_kapisi(kos)
        if rc:
            return rc
        print(f"\n★ VRAM-KAPI ({len(VRAM_KAPI)}): " +
              " · ".join(f"{k['ad']} ({k['olculen_vram_gib']} GiB, {k['coklu_card']} card)"
                         for k in VRAM_KAPI))
        return 0
    if a.kip == "vram":
        from gpu_lock import kilitle
        kilitle(a.dev, beklenen_fiziksel=None, tam=False, etiket="s24_vram")
        return vram_provasi(a)
    return uret(a) if a.kip == "uret" else coz(a)


if __name__ == "__main__":
    sys.exit(main())
