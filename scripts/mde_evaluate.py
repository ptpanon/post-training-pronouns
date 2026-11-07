#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, sys, time

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
from reading_style_object import payda
import gpu_lock as RK
import mini_dpo_egri as ME
import serita_harman102 as SA

if ME.DPO_HUB not in SA.HF_KOKLERI:
    SA.HF_KOKLERI.append(ME.DPO_HUB)

TV = os.environ.get("CKPT_KOK", __DNH_DATA__ + "/seed_varyans")
import hashlib
import mini_dpo_feasibility as FZ

TABAN_M6 = {"ARO": 0.065974, "DOM": 0.083918, "TON": 0.026248}


def hub_no(etiket, seed):
    if etiket == "mde":
        return int(seed) % 1000
    return int(hashlib.sha256(etiket.encode()).hexdigest()[:8], 16) % 900 + 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--etiket", default="mde",
                    help="checkpoint dizini öneki: mde | EMIR_ARTI | EMIR_EKSI | ...")
    ap.add_argument("--m", type=int, default=1,
                    help="")
    ap.add_argument("--adim", type=int, default=100)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--prova", action="store_true",
                    help=""
                         "")
    a = ap.parse_args()
    kok = f"{TV}/{a.etiket}_t{a.seed}"
    ME.CKPT_KOK = kok
    kk = hub_no(a.etiket, a.seed)
    KOLLAR = ("EMIR_ARTI", "EMIR_EKSI", "FORM_YOK", "RASTGELE")
    yuva = {k: hub_no(k, a.seed) for k in KOLLAR}
    n_cak = len(KOLLAR) - len(set(yuva.values()))
    payda("hub_yuvasi", n_kol=len(KOLLAR), n_yuva=len(set(yuva.values())),
          red_cakisma=n_cak, hal_yuva=str(yuva))
    if n_cak:
        raise SystemExit(f"★ HUB YUVASI CAKISMASI {yuva} ⇒ ARIZA (§8 kol-aidiyeti)")
    ck_a, ck_k = f"{kok}/checkpoint-{a.adim}", f"{kok}/checkpoint-{kk}"
    if os.path.exists(f"{ck_a}/config.json") and not os.path.exists(ck_k):
        os.symlink(ck_a, ck_k)
    if not os.path.exists(f"{ck_k}/config.json"):
        raise SystemExit(f"{ck_a} {ck_k}")
    dev_i = RK.fiziksel_bekle(a.dev)
    rej, _, _ = FZ.prod_rejimi(dev_i)
    FZ.prod_kapisi(dev_i, etiket=f"terazi_{a.etiket}")
    bos, _ = FZ.gpu_bos_mib(dev_i)
    kapi = FZ.kapi_hesapla(bos, rej)
    print(f"  ★ KAPI: rejim {rej} · bos {bos} MiB ⇒ tavan **{kapi} MiB** "
          f"⇒ esik asilirsa EYLEM = gözcü `dur_istendi` + kosu DURUR (W-412)",
          flush=True)
    if a.prova:
        print(f"{ck_a} {kk} {a.m}"
              f"" + ", ".join(
                  f"c1_panel/miniDPO/ckpt-{kk + i*1000:03d}" for i in range(a.m)))
        for i in range(a.m):
            yp = f"{__DNH_DATA__}/c1_panel/miniDPO/ckpt-{kk + i*1000:03d}"
            print(f"      m{i+1}: {'VAR (atlanir)' if os.path.exists(yp + '/uretim_kunye.json') else 'YOK (üretilecek)'}  {yp}")
        return 0
    goz = FZ.Gozcu(kapi); goz.start()
    try:
        goz.sert_kur(kapi, dev_i)
    except Exception as ex:
        print(f"{ex}", flush=True)
    t0 = time.time()
    import serita_harman102 as _SA
    R, eski = [], _SA.SEED
    for i in range(a.m):
        _SA.SEED = 20260830 + i
        ME.CKPT_KOK = kok
        dial_i = kk + i * 1000
        try:
            os.symlink(ck_k, f"{kok}/checkpoint-{dial_i}") if not os.path.exists(
                f"{kok}/checkpoint-{dial_i}") else None
        except FileExistsError:
            pass
        print(f"  ★ m-adim {i+1}/{a.m} · SA.SEED={_SA.SEED} · ckpt-{dial_i}", flush=True)
        R.append(ME.bir_checkpoint(dial_i, a.dev, RK.fiziksel_bekle(a.dev)))
    _SA.SEED = eski
    goz.dur = True
    import numpy as _np
    dU = _np.array([x["dU_DOM"] for x in R])
    Uk = _np.array([x["U_ckpt"]["DOM"] for x in R])
    dU_m = TABAN_M6["DOM"] - Uk
    r = dict(m=a.m, tohumlar=[20260830+i for i in range(a.m)],
             dU_DOM=float(dU.mean()), dU_DOM_sd=float(dU.std(ddof=1)) if a.m > 1 else None,
             dU_DOM_tekil=[float(x) for x in dU],
             taban="PREREG_TABAN_CIZGISI_2026-08-30 · m=6 · DOM +0.083918",
             U_ckpt_DOM=[float(x) for x in Uk],
             U_ckpt_DOM_ort=float(Uk.mean()),
             dU_PREREG_DOM=float(dU_m.mean()),
             dU_PREREG_DOM_sd=float(dU_m.std(ddof=1)) if a.m > 1 else None,
             dU_PREREG_DOM_tekil=[float(x) for x in dU_m],
             kayma_alet_prereg=round(float(dU_m.mean() - dU.mean()), 6),
             ad=R[0]["ad"] if a.m == 1 else None,
             gozcu=dict(tavan_mib=kapi, rejim=rej, tepe_kul_mib=goz.tepe_kul,
                        ihlal=bool(goz.ihlal), dur_istendi=bool(goz.dur_istendi)),
             tekil=R)
    r["seed"] = a.seed; r["adim"] = a.adim; r["etiket"] = a.etiket
    payda(f"mde_deg_{a.seed}", n_ckpt=1, hal_dU_DOM=round(r["dU_DOM"], 6))
    yol = f"{ROOT}/results/KOL_DEGER_{a.etiket}_m{a.m}_2026-08-30.json"
    json.dump(r, open(yol, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    sd = f" ± {r['dU_DOM_sd']:.6f}" if r.get("dU_DOM_sd") else ""
    sdm = f" ± {r['dU_PREREG_DOM_sd']:.6f}" if r.get("dU_PREREG_DOM_sd") else ""
    print(f"\n★ {a.etiket} (m={a.m})"
          f"\n   ΔU_ALET (yedinci taban) **{r['dU_DOM']:+.6f}**{sd} · "
          f"tekil {[round(x,6) for x in r['dU_DOM_tekil']]}"
          f"\n   ★★ ΔU_PREREG (m=6 taban) **{r['dU_PREREG_DOM']:+.6f}**{sdm} · "
          f"tekil {[round(x,6) for x in r['dU_PREREG_DOM_tekil']]}"
          f"\n   kayma {r['kayma_alet_prereg']:+.6f} · gözcü tepe "
          f"{goz.tepe_kul} MiB / tavan {kapi} · ihlal {goz.ihlal}"
          f"\n   {(time.time()-t0)/60:.1f} dk → {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
