#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, sys, time
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
import gpu_lock as RK
import template_robustness as SG
import serita_harman102 as SA
import family_panel as C
from reading_style_object import payda
import urial_run as UK

SATIR = "You are a helpful assistant."
CIFT_CARD = {"Llama-3.1-70B", "Qwen2.5-72B"}
CIK_KOK = __DNH_DATA__ + "/c1_panel_msistem"
ZINCIR = dict(PREREG=f"{KOK}/preregistration/prereg_system_prompt_2026-09-10.md",
              NOTICE=f"{KOK}/results/NOTICE_M_SISTEM_2026-09-10.md",
              PREDICTION=f"{KOK}/preregistration/prediction_system_prompt_2026-09-10.md")


def _n(x): return "".join(ch for ch in x.lower() if ch.isalnum())


def hucreler():
    S = json.load(open(f"{KOK}/results/template_filtered_2026-09-07.json",
                       encoding="utf-8"))
    aileler = [r["aile"] for r in S["aileler"] if r["hal"] == "ÖLCÜLDÜ"]
    H = {}
    for pad in ("f5", "olcek", "olcek2", "olcek3", "c1"):
        C.PANEL = C.PANELLER[pad]
        try:
            C.snaplari_coz()
        except SystemExit:
            pass
        for k, v in C.SNAP.items():
            if isinstance(v, dict) and v.get("instruct") and v.get("base"):
                H.setdefault(_n(k), (k, v))
    out = []
    for a in aileler:
        v = H.get(_n(a))
        if not v:
            print(f"{a}"); continue
        def _depo(y):
            return y.split("models--")[1].split("/")[0].join(["models--", ""]) \
                if "models--" in y else y
        out.append(dict(ad=a, yol_taban=v[1]["base"], yol_hizali=v[1]["instruct"],
                        s_taban="models--" + v[1]["base"].split("models--")[1].split("/")[0],
                        s_hizali="models--" + v[1]["instruct"].split("models--")[1].split("/")[0]))
    return out


def onek_kurucu(icerik):
    def kur(tok):
        def f(ad, tur, c, ofs, i, ist, cekim):
            ham = SA.onek_kur(ad, tur, c, ofs, i, ist, cekim)
            return tok.apply_chat_template(
                [{"role": "system", "content": icerik},
                 {"role": "user", "content": ham}],
                tokenize=False, add_generation_prompt=True)
        return f
    return kur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", default="satirli", choices=["satirli", "satirsiz"])
    ap.add_argument("--zemin", default="instruct", choices=["base", "instruct"])
    ap.add_argument("--aile", default=None)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--coklu-card", dest="coklu_card", type=int, default=0)
    ap.add_argument("--kuru", action="store_true", help="")
    a = ap.parse_args()
    icerik = SATIR if a.kol == "satirli" else ""
    H = hucreler()
    if a.aile:
        H = [h for h in H if h["ad"] == a.aile]
        if not H:
            print(f"{a.aile}"); return 4
    for y in ZINCIR.values():
        if not os.path.exists(y):
            print(f"★ B-43 DÜSTÜ — {os.path.basename(y)} YOK"); return 4
    SG.PREREG, SG.NOTICE, SG.PREDICTION = ZINCIR["PREREG"], ZINCIR["NOTICE"], ZINCIR["PREDICTION"]
    SG.ONEK_KURUCU = onek_kurucu(icerik)
    SG.CIKTI_KOK = f"{CIK_KOK}/{a.kol}"
    payda("m_sistem_kol", n_hucre=len(H), n_zincir=3, red_eksik_snapshot=0)
    if a.kuru:
        from transformers import AutoTokenizer
        h = H[0]
        tok = AutoTokenizer.from_pretrained(h["yol_hizali"])
        ham = "Explain what a signal is."
        A = tok.apply_chat_template([{"role": "system", "content": ""},
                                     {"role": "user", "content": ham}],
                                    tokenize=False, add_generation_prompt=True)
        B = tok.apply_chat_template([{"role": "system", "content": SATIR},
                                     {"role": "user", "content": ham}],
                                    tokenize=False, add_generation_prompt=True)
        print(f"  [KURU] kol={a.kol} · zemin={a.zemin} · hücre={len(H)} · "
              f"cikti kökü={SG.CIKTI_KOK}")
        print(f"{h['ad']}"
              f"{B.replace(SATIR, '', 1) == A}")
        print(f"  [KURU] YOK öneki: {A[:120]!r}")
        print(f"  [KURU] VAR öneki: {B[:120]!r}")
        print("")
        return 0
    import torch as _t
    n_gor = _t.cuda.device_count() if _t.cuda.is_available() else 0
    t0 = time.time(); ariza = 0; atlanan = []; zaten = []; damgali = 0
    for h in H:
        if h["ad"] in CIFT_CARD and (n_gor < 2 or a.coklu_card < 2):
            atlanan.append(h["ad"])
            print(f"  ★ {h['ad']}/{a.zemin}: ATLANDI-CIFT-CARD-GEREK "
                  f"(görünen card {n_gor}, --coklu-card {a.coklu_card}) ⇒ "
                  f"EYLEM: iki card bosken ayri kosuda", flush=True)
            continue
        _hedef = f"{h['ad']}/{'base' if a.zemin == 'base' else 'instruct'}"
        _ok, _m = UK.icerik(_hedef, kok=SG.CIKTI_KOK)
        if _ok:
            zaten.append(h["ad"])
            print(f"{h['ad']} {a.zemin} {_m}"
                  f"", flush=True)
            continue
        SG.KOLLAR = [dict(ad=h["ad"], taban=f"{h['ad']}/base",
                          hizali=f"{h['ad']}/instruct",
                          s_taban=h["s_taban"], s_hizali=h["s_hizali"],
                          panel="c1", ikiz="ham", olculen_dk=0.0)]
        try:
            SG.uret(argparse.Namespace(kol=h["ad"], zemin=a.zemin, dev=a.dev,
                                       bekle_gpu=RK.fiziksel_bekle(a.dev),
                                       coklu_card=a.coklu_card))
        except SystemExit as e:
            print(f"  ★★ {h['ad']}/{a.zemin}: {e}"); ariza += 1
        except Exception as e:
            print(f"  ★★★ {h['ad']}/{a.zemin}: {type(e).__name__}: {e}"); ariza += 1
        else:
            _ok2, _m2 = UK.icerik(_hedef, kok=SG.CIKTI_KOK)
            if _ok2:
                _dz = os.path.join(SG.CIKTI_KOK, _hedef)
                with open(os.path.join(_dz, ".tamam"), "w", encoding="utf-8") as _f:
                    _f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                             + f" · {_m2}\n")
                damgali += 1
            else:
                print(f"  ★ .tamam YAZILMADI — {_hedef} icerik denetimi: {_m2}",
                      flush=True)
    print(f"  [PAYDA] m_sistem_uret: n_hucre={len(H)} · hal_kosan={len(H)-len(atlanan)} · "
          f"red_atlanan_cift_card={len(atlanan)} · hal_zaten_var={len(zaten)} · "
          f"hal_damgali={damgali} · red_ariza={ariza} · "
          f"hal_dk={(time.time()-t0)/60:.1f} ⇒ esik: red_ariza>0 ⇒ rc=3")
    return 0 if ariza == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
