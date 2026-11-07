#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, sys, time

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import family_panel as C
import template_robustness as SG
import serita_harman102 as SA
import form_count as FS

CKPT_KOK = __DNH_DATA__ + "/mini_dpo_ckpt"
DPO_HUB = __DNH_DATA__ + "/dpo_hub"
CIK = f"{ROOT}/results/MINI_DPO_EGRI_2026-08-29.json"
TABAN_SNAP = "models--allenai--Llama-3.1-Tulu-3-8B-SFT"
IKIZ = "ham"


def hub_kur(k):
    ad = f"models--project--minidpo-ckpt-{k:03d}"
    hedef = f"{DPO_HUB}/{ad}/snapshots/ckpt{k:03d}"
    os.makedirs(os.path.dirname(hedef), exist_ok=True)
    if not os.path.islink(hedef) and not os.path.isdir(hedef):
        os.symlink(f"{CKPT_KOK}/checkpoint-{k}", hedef)
    return ad, hedef


def panel_kur(k, ad_hub):
    return [dict(ad=f"miniDPO-{k:03d}", base=TABAN_SNAP, instruct=ad_hub,
                 anahtar={"base": "Tulu3-8B/sft", "instruct": f"miniDPO/ckpt-{k:03d}"},
                 tarif="MINI-DPO pilot egrisi (KESIF)")]


def ucuz_aletler(yol, nlp=None):
    import numpy as np
    R = [json.loads(l) for l in open(f"{yol}/uretim.jsonl", encoding="utf-8")]
    metin = [r["metin"] for r in R]
    if nlp is None:
        nlp = FS._boru()
    A, n_c, n_j = FS.uretim_say(nlp, metin)
    top_c = max(int(n_c.sum()), 1)
    return {"n_metin": len(metin), "n_cumle": int(n_c.sum()),
            "n_jeton": int(n_j.sum()),
            "uretim_basina": {k: round(float(v.sum()) / len(metin), 5) for k, v in A.items()},
            "cumle_basina": {k: round(float(v.sum()) / top_c, 5) for k, v in A.items()}}


def bir_checkpoint(k, dev, bekle_gpu):
    ad_hub, yol_hub = hub_kur(k)
    C.PANEL = panel_kur(k, ad_hub); C.GEMMA = C._gemma(C.PANEL)
    C.OUT_KOK = __DNH_DATA__ + "/c1_panel"
    cad = f"miniDPO-{k:03d}"
    out = C.uretim_yolu(cad, "instruct")
    t0 = time.time()
    if not os.path.exists(f"{out}/uretim_kunye.json"):
        import argparse as _ap
        C.uret(_ap.Namespace(cift=cad, zemin="instruct", dev=dev,
                             bekle_gpu=bekle_gpu, coklu_card=0))
    else:
        print(f"{k}", flush=True)
    zb, Bb, UNb = C._zemin_oku(cad, "base", dev)
    zi, Bi, UNi = C._zemin_oku(cad, "instruct", dev)
    d = SG._delta(zb, Bb, UNb, zi, Bi, UNi, IKIZ)
    E = SG.EKSEN_KARAR
    dej, n_dej = SG._dej(f"{out}/uretim.jsonl")
    uc = ucuz_aletler(out)
    return dict(adim=k, dU_DOM=d[E]["ham"]["dU"], ad=d[E]["ad"],
                U_taban_alet={e: float(zb["olcum"][e]["U"]) for e in C.EKSENLER},
                U_ckpt={e: float(zi["olcum"][e]["U"]) for e in C.EKSENLER},
                dU_CI=d[E]["ham"]["dU_CI"], dU_sd=d[E]["ham"]["dU_sd"],
                uc_eksen={e: d[e]["ad"] for e in C.EKSENLER},
                uc_eksen_dU={e: d[e]["ham"]["dU"] for e in C.EKSENLER},
                DEJ=round(dej, 4), form=uc, saniye=round(time.time() - t0, 1),
                uretim_yolu=out, tam=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adimlar", default="200,50,100,150")
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", dest="bekle_gpu", type=int, default=0)
    a = ap.parse_args()
    if DPO_HUB not in SA.HF_KOKLERI:
        SA.HF_KOKLERI.append(DPO_HUB)
        print(f"{DPO_HUB}", flush=True)
    adimlar = [int(x) for x in a.adimlar.split(",")]
    var = [k for k in adimlar if os.path.isdir(f"{CKPT_KOK}/checkpoint-{k}")]
    yok = [k for k in adimlar if k not in var]
    payda("mini_dpo_egri_kapi", n_istenen=len(adimlar), hal_diskte=len(var),
          red_checkpoint_yok=len(yok))
    if yok:
        print(f"{yok}", flush=True)
    E = {}
    eski = json.load(open(CIK, encoding="utf-8")) if os.path.exists(CIK) else {}
    E.update(eski.get("nokta", {}))
    for k in var:
        try:
            r = bir_checkpoint(k, a.dev, a.bekle_gpu)
            E[str(k)] = r
            print(f"  ★ ckpt-{k:3d} · ΔU_DOM {r['dU_DOM']:+.5f} [{r['ad']}] · "
                  f"DEJ {r['DEJ']:.3f} · {r['saniye']:.0f} sn", flush=True)
        except Exception as e:
            E[str(k)] = {"adim": k, "HATA": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"  ★ ckpt-{k} ARIZA: {str(e)[:160]}", flush=True)
        json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       PREREG="prereg_mini_dpo_8b_2026-08-29.md", SINIF="KESIF",
                       PANEL="", PREDICTION="YOK",
                       taban="Tulu3-8B/sft (ciplak, diskte hazir)", karar_ikizi=IKIZ,
                       istenen=adimlar, diskte=var, nokta=E),
                  open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    payda("mini_dpo_egri", n_nokta=len(E), hal_olculen=sum(1 for v in E.values() if "HATA" not in v),
          red_ariza=sum(1 for v in E.values() if "HATA" in v))
    print(f"→ {CIK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
