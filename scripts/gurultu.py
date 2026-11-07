#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import gpu_lock as RK
import argparse, itertools, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
from reading_style_object import payda
import family_panel as C
import template_robustness as SG
import serita_harman102 as SA
import form_count as FS
import mini_dpo_feasibility as FZ

KOK = __DNH_DATA__ + "/k0_gurultu"
CIK = f"{ROOT}/results/K0_GURULTU_2026-08-30.json"
SNAP_SFT = "models--allenai--Llama-3.1-Tulu-3-8B-SFT"
SEED0 = 20260830
IKIZ = "ham"


def panel(a_seed, b_seed):
    return [dict(ad="k0-null", base=SNAP_SFT, instruct=SNAP_SFT,
                 anahtar={"base": f"seed{a_seed}", "instruct": f"seed{b_seed}"},
                 aile="Tulu3-8B", tarif="k=0 GÜRÜLTÜ TABANI (ayni model, iki seed)")]


def kur(a_t, b_t):
    C.PANEL = panel(a_t, b_t); C.GEMMA = C._gemma(C.PANEL); C.OUT_KOK = KOK


def uret_seed(t, dev):
    kur(t, t)
    eski = SA.SEED
    SA.SEED = SEED0 + t
    yol = C.uretim_yolu("k0-null", "base")
    print(f"  ★ seed {t}: SA.SEED {eski} → {SA.SEED} · → {yol}", flush=True)
    t0 = time.time()
    if os.path.exists(f"{yol}/uretim_kunye.json"):
        print("", flush=True)
        SA.SEED = eski
        return yol, 0.0, True
    C.uret(argparse.Namespace(cift="k0-null", zemin="base", dev=dev,
                              bekle_gpu=RK.fiziksel_bekle(dev), coklu_card=0))
    SA.SEED = eski
    return yol, time.time() - t0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--pencere-dk", type=float, default=180.0)
    ap.add_argument("--seed-tavan", type=int, default=6)
    ap.add_argument("--seed-taban", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(KOK, exist_ok=True)
    t00 = time.time()

    bos0, kul0 = FZ.gpu_bos_mib(RK.fiziksel_bekle("cuda:0"))
    _rejim, sag = FZ.prod_kapisi(RK.fiziksel_bekle("cuda:0"), "gurultu.py")
    kapi = FZ.kapi_hesapla(bos0, _rejim); tavan = kul0 + kapi
    print(f"[ÖN-UCUS] GPU0 bos={bos0} kullanilan={kul0} · prod={sag}\n"
          f"[KAPI] bütce = bos {bos0} − TAMPON {FZ.TAMPON_MIB} = {kapi} MiB ⇒ card tavani {tavan} MiB "
          f"⇒ EYLEM: gözcü DURDURUCU (kanonik) — asilirsa bayrak + dur_istendi", flush=True)
    g = FZ.Gozcu(tavan); g.start()

    yollar, sureler = {}, {}
    y, sn, atl = uret_seed(0, a.dev)
    yollar[0] = y; sureler[0] = sn
    if atl:
        sn = 0.0
    S = a.seed_tavan if sn == 0 else int(max(a.seed_taban,
                                              min(a.seed_tavan, (a.pencere_dk * 60) // max(sn, 1))))
    payda("k0_pencere", n_olcum=1, hal_ilk_seed_sn=round(sn, 1),
          hal_pencere_dk=a.pencere_dk, hal_secilen_seed=S)
    print(f"  [KAPI/W-71] ilk seed **ÖLCÜLDÜ**: {sn:.0f} sn ({sn/60:.1f} dk) ⇒ "
          f"pencere {a.pencere_dk:.0f} dk ⇒ sigan seed {int((a.pencere_dk*60)//max(sn,1))} "
          f"⇒ taban {a.seed_taban} · tavan {a.seed_tavan} ⇒ **SECILEN S = {S}** ⇒ "
          f"EYLEM: S<{a.seed_taban} cikarsa kosu `SEED-YETMEZ` adiyla durur", flush=True)
    if S < a.seed_taban:
        raise SystemExit(f"★ SEED-YETMEZ: pencereye {S} seed sigiyor, taban {a.seed_taban}")

    for t in range(1, S):
        y, sn_t, _ = uret_seed(t, a.dev)
        yollar[t] = y; sureler[t] = sn_t
        print(f"     seed {t} bitti · {sn_t/60:.1f} dk · toplam {(time.time()-t00)/60:.1f} dk",
              flush=True)

    E = SG.EKSEN_KARAR
    U, cift = {}, {}
    for i, j in itertools.combinations(range(S), 2):
        kur(i, j)
        zb, Bb, UNb = C._zemin_oku("k0-null", "base", a.dev)
        zi, Bi, UNi = C._zemin_oku("k0-null", "instruct", a.dev)
        d = SG._delta(zb, Bb, UNb, zi, Bi, UNi, IKIZ)
        U[i] = {e: float(zb["olcum"][e]["U"]) for e in C.EKSENLER}
        U[j] = {e: float(zi["olcum"][e]["U"]) for e in C.EKSENLER}
        cift[f"{i}-{j}"] = dict(
            dU={e: d[e]["ham"]["dU"] for e in C.EKSENLER},
            ad={e: d[e]["ad"] for e in C.EKSENLER},
            dU_DOM=d[E]["ham"]["dU"], AD_DOM=d[E]["ad"])
        print(f"  ★ seed {i} ↔ {j}: ΔU_DOM = {d[E]['ham']['dU']:+.6f} [{d[E]['ad']}]",
              flush=True)

    dU = np.array([v["dU_DOM"] for v in cift.values()])
    uv = np.array([U[t][E] for t in sorted(U)])
    ad_say = {}
    for v in cift.values():
        ad_say[v["AD_DOM"]] = ad_say.get(v["AD_DOM"], 0) + 1
    taban = dict(
        eksen=E, n_seed=S, n_cift=len(dU),
        U_seed_basina={str(t): round(U[t][E], 6) for t in sorted(U)},
        U_ort=round(float(uv.mean()), 6), U_sd=round(float(uv.std(ddof=1)), 6),
        dU_ort=round(float(dU.mean()), 6), dU_sd=round(float(dU.std(ddof=1)), 6),
        dU_mutlak_ort=round(float(np.abs(dU).mean()), 6),
        dU_p95_mutlak=round(float(np.percentile(np.abs(dU), 95)), 6),
        dU_menzil=round(float(dU.max() - dU.min()), 6),
        AD_dagilimi=ad_say)
    payda("k0_gurultu", n_seed=S, n_cift=len(dU), hal_dU_sd=taban["dU_sd"],
          red_ad_sifirdan_farkli=sum(v for k, v in ad_say.items() if k != "~"))
    g.dur = True; g.join(timeout=3)
    sag2 = FZ.prod_saglik()
    O = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="KESIF · GÜRÜLTÜ TABANI (egitim YOK · bar YOK · ad YOK · prediction YOK)",
             KURULUM=""
                     "",
             seed0=SEED0, pencere_dk=a.pencere_dk, secilen_S=S,
             sureler_sn={str(k): round(v, 1) for k, v in sureler.items()},
             yollar={str(k): v for k, v in yollar.items()},
             taban=taban, cift=cift,
             gozcu=dict(tepe_kul=g.tepe_kul, tavan=tavan, ihlal=g.ihlal,
                        dur_istendi=g.dur_istendi, kapi_mib=kapi),
             prod_once=sag, prod_sonra=sag2, saniye=round(time.time() - t00, 1))
    json.dump(O, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"\n★★★ GÜRÜLTÜ TABANI · {S} seed · {len(dU)} cift\n"
          f"   U({E}) sd = {taban['U_sd']:.6f} · ΔU sd = {taban['dU_sd']:.6f} · "
          f"|ΔU| ort = {taban['dU_mutlak_ort']:.6f} · menzil = {taban['dU_menzil']:.6f}\n"
          f"   AD dagilimi: {ad_say}\n"
          f"   prod: {sag} → {sag2} · {(time.time()-t00)/60:.1f} dk\n→ {CIK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
