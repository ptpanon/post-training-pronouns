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
import family_panel as C
import serita_harman102 as SA
import dejenerelik_olcer as D

CIPLAK_KOK = C.OUT_KOK
TEMPLATE_KOK = __DNH_DATA__ + "/c1_panel_template"
ONEK_KURUCU = None
CIKTI_KOK = None
CIK = f"{ROOT}/results/template_robustness_2026-08-26.json"
PREREG = f"{ROOT}/preregistration/prereg_template_robustness_2026-08-26.md"
PREDICTION = f"{ROOT}/preregistration/prediction_template.md"
NOTICE = f"{ROOT}/results/NOTICE_TEMPLATE_GURBUZLUGU_2026-08-26.md"

KOLLAR = [
    dict(ad="Qwen2.5-7B",      taban="Qwen2.5-7B/base",      hizali="Qwen2.5-7B/instruct",
         s_taban="models--Qwen--Qwen2.5-7B", s_hizali="models--Qwen--Qwen2.5-7B-Instruct",
         panel="c1", ikiz="ham", olculen_dk=18.7),
    dict(ad="Mistral-7B-v0.3", taban="Mistral-7B-v0.3/base", hizali="Mistral-7B-v0.3/instruct",
         s_taban="models--mistralai--Mistral-7B-v0.3",
         s_hizali="models--mistralai--Mistral-7B-Instruct-v0.3",
         panel="c1", ikiz="ham", olculen_dk=20.2),
    dict(ad="OLMo-3-7B",       taban="OLMo-3-7B/base",       hizali="OLMo-3-7B/instruct",
         s_taban="models--allenai--Olmo-3-1025-7B",
         s_hizali="models--allenai--Olmo-3-7B-Instruct",
         panel="c1", ikiz="ham", olculen_dk=23.2),
    dict(ad="Llama-3.1-8B",    taban="Llama-3.1-8B/base",    hizali="Llama-3.1-8B/instruct",
         s_taban="models--meta-llama--Llama-3.1-8B",
         s_hizali="models--meta-llama--Llama-3.1-8B-Instruct",
         panel="c1", ikiz="ham", olculen_dk=20.8),
    dict(ad="Gemma-3-4B",      taban="Gemma-3-4B/base",      hizali="Gemma-3-4B/instruct",
         s_taban="models--google--gemma-3-4b-pt", s_hizali="models--google--gemma-3-4b-it",
         panel="c1", ikiz="ham", olculen_dk=27.4),
    dict(ad="Gemma-3-12B",     taban="Gemma-3-12B/base",     hizali="Gemma-3-12B/instruct",
         s_taban="models--google--gemma-3-12b-pt", s_hizali="models--google--gemma-3-12b-it",
         panel="c1", ikiz="ham", olculen_dk=39.2),
    dict(ad="OLMo2-13B·uctan-uca", taban="OLMo2-13B/taban",  hizali="OLMo2-13B/instruct",
         s_taban="models--allenai--OLMo-2-1124-13B",
         s_hizali="models--allenai--OLMo-2-1124-13B-Instruct",
         panel="hasat_h3", ikiz="ham", olculen_dk=32.5),
    dict(ad="OLMo2-32B·uctan-uca", taban="OLMo2-32B/taban",  hizali="OLMo2-32B/instruct",
         s_taban="models--allenai--OLMo-2-0325-32B",
         s_hizali="models--allenai--OLMo-2-0325-32B-Instruct",
         panel="olcek2", ikiz="ham", olculen_dk=54.8),
    dict(ad="Tulu3-8B·uctan-uca", taban="Tulu3-8B/taban",    hizali="Tulu3-8B/rl",
         s_taban="models--meta-llama--Llama-3.1-8B",
         s_hizali="models--allenai--Llama-3.1-Tulu-3-8B",
         panel="olcek3", ikiz="merkezli", olculen_dk=21.1),
]
BOS_BAR = C.BOS_BAR
EKSEN_KARAR = "DOM"


def _ad(d):
    if d["AYRIK_POZ"]:
        return "AYRIK_POZ"
    if d["AYRIK_NEG"]:
        return "AYRIK_NEG"
    return "CONTINGENT" if d["CONTINGENT"] else "~"


def _panel_kur(k):
    return [dict(ad=k["ad"], base=k["s_taban"], instruct=k["s_hizali"],
                 anahtar={"base": k["taban"], "instruct": k["hizali"]},
                 tarif="TEMPLATE GÜRBÜZLÜGÜ")]


def sablonlu_onek_fn(tok):
    def f(ad, tur, c, ofs, i, ist, cekim):
        ham = SA.onek_kur(ad, tur, c, ofs, i, ist, cekim)
        return tok.apply_chat_template([{"role": "user", "content": ham}],
                                       tokenize=False, add_generation_prompt=True)
    return f


def prova():
    ok = []
    for d in ({"AYRIK_POZ": True, "AYRIK_NEG": False, "CONTINGENT": False},
              {"AYRIK_POZ": False, "AYRIK_NEG": True, "CONTINGENT": False},
              {"AYRIK_POZ": False, "AYRIK_NEG": False, "CONTINGENT": True},
              {"AYRIK_POZ": False, "AYRIK_NEG": False, "CONTINGENT": False}):
        ok.append(_ad(d))
    assert ok == ["AYRIK_POZ", "AYRIK_NEG", "CONTINGENT", "~"], ok
    dogan = set()
    for p in (True, False):
        for n in (True, False):
            for c in (True, False):
                if p and n:
                    continue
                dogan.add(_ad({"AYRIK_POZ": p, "AYRIK_NEG": n, "CONTINGENT": c}))
    eksik = {"AYRIK_POZ", "AYRIK_NEG", "CONTINGENT", "~"} - dogan
    payda("template_prova", n_dal=4, n_dogan=len(dogan), red_dogamayan=len(eksik))
    if eksik:
        raise SystemExit(f"★ K-f: DOGAMAYAN AD {eksik} ⇒ kosu YOK")
    return True


def uret(a):
    import torch
    from transformers import AutoTokenizer
    k = next(x for x in KOLLAR if x["ad"] == a.kol)
    for y in (PREREG, NOTICE, PREDICTION):
        if not os.path.exists(y):
            raise SystemExit(f"★ B-43 DÜSTÜ — {os.path.basename(y)} YOK.")
    prova()
    C.PANEL = _panel_kur(k); C.GEMMA = C._gemma(C.PANEL)
    C.OUT_KOK = CIKTI_KOK or TEMPLATE_KOK
    snap = SA._snapshot(None, model_adi=k["s_hizali"])
    tok = AutoTokenizer.from_pretrained(snap)
    if not getattr(tok, "chat_template", None):
        payda(f"template_kapi_{k['ad']}", n_kol=1, red_template_yok=1)
        raise SystemExit(f"{k['ad']}")
    orn = tok.apply_chat_template([{"role": "user", "content": "X"}],
                                  tokenize=False, add_generation_prompt=True)
    j_sar = tok(orn)["input_ids"]
    j_ham = tok("X")["input_ids"]
    n_bos = sum(1 for t in j_sar[:2] if t == tok.bos_token_id) if tok.bos_token_id else 0
    payda(f"template_bos_{k['ad']}", n_ornek=1, hal_bos_bas=n_bos,
          hal_sarili_jeton=len(j_sar), hal_ham_jeton=len(j_ham),
          red_cift_bos=int(n_bos > 1))
    print(f"  [§7.4] template sarimi: {len(j_ham)}→{len(j_sar)} jeton · bastaki BOS={n_bos}"
          f"{'  ★ CIFT-BOS (kayda gecti, kurgu tekdüze)' if n_bos > 1 else ''}", flush=True)
    _asil = SA._kol_uret

    def _sarmal(model, tok_, aa, kol, ist_, n_cekim, yol, onek_fn=None):
        return _asil(model, tok_, aa, kol, ist_, n_cekim, yol,
                     onek_fn=(ONEK_KURUCU or sablonlu_onek_fn)(tok))
    SA._kol_uret = _sarmal
    try:
        ns = argparse.Namespace(cift=k["ad"],
                                zemin=getattr(a, "zemin", "instruct"), dev=a.dev,
                                bekle_gpu=a.bekle_gpu,
                                coklu_card=getattr(a, "coklu_card", 0))
        rc = C.uret(ns)
    finally:
        SA._kol_uret = _asil
        C.OUT_KOK = CIPLAK_KOK
    return rc


def _dej(yol):
    n, d = 0, 0
    with open(yol, encoding="utf-8") as fh:
        for l in fh:
            n += 1
            d += int(D.bayrakla(json.loads(l)["metin"])["DEJENERE"])
    return d / n, n


def _oku(k, zemin, kok, dev):
    C.PANEL = _panel_kur(k); C.GEMMA = C._gemma(C.PANEL); C.OUT_KOK = kok
    try:
        return C._zemin_oku(k["ad"], zemin, dev)
    finally:
        C.OUT_KOK = CIPLAK_KOK


def _delta(zb, Bb, UNb, zi, Bi, UNi, ikiz):
    out = {}
    for e in C.EKSENLER:
        v = Bb[e] - Bi[e]
        ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        sd = float(v.std())
        dU = float(zb["olcum"][e]["U"] - zi["olcum"][e]["U"])
        dn = UNb[e] - UNi[e]
        m = float(dn.mean())
        ham = dict(dU=round(dU, 6), dU_CI=[round(x, 6) for x in ci], dU_sd=round(sd, 6),
                   null_sd_kati=round(abs(dU) / sd, 3) if sd > 0 else None,
                   AYRIK_POZ=bool(ci[0] > 0), AYRIK_NEG=bool(ci[1] < 0))
        ham["CONTINGENT"] = bool(not ham["AYRIK_POZ"] and not ham["AYRIK_NEG"]
                                 and ham["null_sd_kati"] is not None
                                 and 1.0 <= ham["null_sd_kati"] < 1.645)
        mer = dict(dU=round(dU - m, 6), dU_CI=[round(ci[0] - m, 6), round(ci[1] - m, 6)],
                   dU_sd=round(sd, 6),
                   null_sd_kati=round(abs(dU - m) / sd, 3) if sd > 0 else None,
                   AYRIK_POZ=bool(ci[0] - m > 0), AYRIK_NEG=bool(ci[1] - m < 0))
        mer["CONTINGENT"] = bool(not mer["AYRIK_POZ"] and not mer["AYRIK_NEG"]
                                 and mer["null_sd_kati"] is not None
                                 and 1.0 <= mer["null_sd_kati"] < 1.645)
        out[e] = dict(ham=ham, merkezli=mer, null_merkezi=round(m, 6),
                      plasebo_frac=round(float((np.abs(dn) >= abs(dU)).mean()), 4),
                      PLASEBO_MERKEZLI=bool(abs(m) / (dn.std() or 1) <= 1.645),
                      KARAR_IKIZI=ikiz, ad=_ad(ham if ikiz == "ham" else mer))
    return out


def coz(a):
    t0 = time.time(); prova()
    H, sayac = {}, dict(ayni=0, degisti=0, kapi=0, dej_dusuk=0, isaret_dondu=0)
    for k in KOLLAR:
        ad = k["ad"]
        s_yol = f"{TEMPLATE_KOK}/{k['hizali']}/uretim.jsonl"
        if not os.path.exists(s_yol):
            H[ad] = dict(HAL="KAPI/ÖLCÜLEMEZ", neden="")
            sayac["kapi"] += 1
            print(f"{ad:22s}", flush=True); continue
        dj_s, n_s = _dej(s_yol)
        dj_c, n_c = _dej(f"{CIPLAK_KOK}/{k['hizali']}/uretim.jsonl")
        bos_s = sum(1 for l in open(s_yol, encoding="utf-8")
                    if not json.loads(l)["metin"].strip()) / n_s
        if bos_s > BOS_BAR:
            H[ad] = dict(HAL="KAPI/ÖLCÜLEMEZ", neden=f"bos-orani {bos_s:.4f} > {BOS_BAR}",
                         DEJ_ciplak=round(dj_c, 4), DEJ_sablonlu=round(dj_s, 4))
            sayac["kapi"] += 1
            print(f"  [{ad:22s}] ★ KAPI — bos {bos_s:.4f} > {BOS_BAR}", flush=True); continue
        try:
            zb, Bb, UNb = _oku(k, "base", CIPLAK_KOK, a.dev)
            zi_c, Bi_c, UNi_c = _oku(k, "instruct", CIPLAK_KOK, a.dev)
            zi_s, Bi_s, UNi_s = _oku(k, "instruct", TEMPLATE_KOK, a.dev)
        except Exception as e:
            H[ad] = dict(HAL="KAPI/ÖLCÜLEMEZ", neden=f"kestirici kurulamadi: {str(e)[:120]}")
            sayac["kapi"] += 1
            print(f"  [{ad:22s}] ★ KAPI — {str(e)[:90]}", flush=True); continue
        d_c = _delta(zb, Bb, UNb, zi_c, Bi_c, UNi_c, k["ikiz"])
        d_s = _delta(zb, Bb, UNb, zi_s, Bi_s, UNi_s, k["ikiz"])
        ac, as_ = d_c[EKSEN_KARAR]["ad"], d_s[EKSEN_KARAR]["ad"]
        kar = "ham" if k["ikiz"] == "ham" else "merkezli"
        uc, us = d_c[EKSEN_KARAR][kar]["dU"], d_s[EKSEN_KARAR][kar]["dU"]
        isaret = bool(uc * us < 0)
        hal = "AYNI-AD" if ac == as_ else "AD-DEGISTI"
        sayac["ayni" if hal == "AYNI-AD" else "degisti"] += 1
        sayac["isaret_dondu"] += int(isaret)
        sayac["dej_dusuk"] += int(dj_s <= dj_c)
        H[ad] = dict(HAL=hal, ad_ciplak=ac, ad_sablonlu=as_, karar_ikizi=k["ikiz"],
                     yon=(f"{ac} → {as_}" if hal == "AD-DEGISTI" else None),
                     dU_ciplak=uc, dU_sablonlu=us, ISARET_DONDU=isaret,
                     DEJ_ciplak=round(dj_c, 4), DEJ_sablonlu=round(dj_s, 4),
                     DEJ_dustu=bool(dj_s <= dj_c), bos_orani_sablonlu=round(bos_s, 4),
                     uc_eksen_ciplak={e: d_c[e]["ad"] for e in C.EKSENLER},
                     uc_eksen_sablonlu={e: d_s[e]["ad"] for e in C.EKSENLER},
                     tam_ciplak=d_c, tam_sablonlu=d_s)
        print(f"  [{ad:22s}] {ac:12s} → {as_:12s} · ΔU {uc:+.5f}→{us:+.5f}"
              f" · DEJ {dj_c:.3f}→{dj_s:.3f} · ★ {hal}"
              f"{'  ★★ ISARET DÖNDÜ' if isaret else ''}", flush=True)
    olculebilir = sayac["ayni"] + sayac["degisti"]
    if olculebilir < 5:
        panel = "KAPI/ÖLCÜLEMEZ"
    elif sayac["ayni"] >= 7:
        panel = "TEMPLATE-GÜRBÜZ"
    elif sayac["degisti"] >= 3:
        panel = "SABLONA-BAGIMLI"
    else:
        panel = "KISMÎ-GÜRBÜZ"
    B = {}
    B["BT-1"] = ("SKORLANMAZ" if olculebilir == 0 else
                 ("TUTTU" if sayac["ayni"] >= 6 else "TUTMADI"))
    B["BT-2"] = ("SKORLANMAZ" if olculebilir == 0 else
                 ("TUTTU" if sayac["isaret_dondu"] == 0 else "TUTMADI"))
    B["BT-3"] = "TUTTU" if sayac["dej_dusuk"] == len(KOLLAR) else "TUTMADI"
    B["BT-4"] = "TUTTU" if sayac["kapi"] >= 1 else "TUTMADI"
    payda("template_gurbuzlugu", n_kol=len(KOLLAR), hal_olculebilir=olculebilir,
          hal_ayni=sayac["ayni"], hal_degisti=sayac["degisti"], red_kapi=sayac["kapi"],
          bekle={"n_kol": 9})
    print(f"{panel} {sayac['ayni']} {sayac['degisti']}"
          f"{sayac['kapi']} {olculebilir}")
    print(f"★ PREDICTION: " + " · ".join(f"{a_}={b}" for a_, b in B.items())
          + f"  ⇒ ASSISTANT {sum(1 for v in B.values() if v=='TUTTU')}/"
            f"{sum(1 for v in B.values() if v!='SKORLANMAZ')}")
    json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   PREREG="d301b137 (+ERRATA-1)", PREDICTION="ef3e39d2",
                   EKSEN_KARAR=EKSEN_KARAR, PANEL=panel, sayac=sayac,
                   olculebilir=olculebilir, PREDICTION_SKOR=B, kol=H,
                   saniye=round(time.time() - t0, 1)),
              open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"→ {CIK}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--asama", required=True, choices=["prova", "uret", "coz"])
    p.add_argument("--kol", default=None)
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--bekle-gpu", dest="bekle_gpu", type=int, default=0)
    a = p.parse_args()
    if a.asama == "prova":
        prova(); print("★ prova TAM (4/4 dal dogar)"); sys.exit(0)
    sys.exit(uret(a) if a.asama == "uret" else coz(a))
