#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, glob, json, os, sys, time
from collections import defaultdict
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

CIK = f"{ROOT}/results/weight_delta_similarity_2026-09-05.json"
CK = __DNH_DATA__ + "/dial_ckpt"
TV = __DNH_DATA__ + "/seed_varyans"
SFT_GLOB = ("<storage>/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT/"
            "snapshots/*/model-*.safetensors")
KOLLAR = {
    "KISI_ASAGI":   f"{CK}/DIAL_KISI_ASAGI_t20260830/checkpoint-458/model.safetensors",
    "KISI_YUKARI":  f"{CK}/DIAL_KISI_YUKARI_t20260830/checkpoint-458/model.safetensors",
    "RASTGELE":     f"{CK}/DIAL_RASTGELE_t20260830/checkpoint-458/model.safetensors",
    "KUVVET_ASAGI": f"{CK}/DIAL_KUVVET_ASAGI_t20260830/checkpoint-458/model.safetensors",
}


def grup(ad):
    if "lm_head" in ad:
        return "lm_head"
    if "embed_tokens" in ad:
        return "embed"
    if any(x in ad for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(x in ad for x in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    if "layernorm" in ad or ad.endswith("model.norm.weight"):
        return "norm"
    return "diger"


def kat(ad):
    p = ad.split(".")
    if "layers" in p:
        return int(p[p.index("layers") + 1])
    return -1


def _prova():
    return {
        "i_lm_head": grup("lm_head.weight") == "lm_head",
        "ii_embed": grup("model.embed_tokens.weight") == "embed",
        "iii_attn": grup("model.layers.7.self_attn.q_proj.weight") == "attention",
        "iv_mlp": grup("model.layers.7.mlp.down_proj.weight") == "mlp",
        "v_norm": grup("model.layers.7.input_layernorm.weight") == "norm",
        "vi_son_norm": grup("model.norm.weight") == "norm",
        "vii_kat": kat("model.layers.31.mlp.up_proj.weight") == 31,
        "viii_katsiz": kat("lm_head.weight") == -1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pozkon", default="")
    a = ap.parse_args()
    P = _prova()
    print("★ §7.1 PROVA (grup/kat):", json.dumps(P, ensure_ascii=False))
    print(f"   ⇒ {'GECTI' if all(P.values()) else 'DÜSTÜ'} ⇒ EYLEM: düserse cikis 4",
          flush=True)
    if not all(P.values()):
        return 4
    import torch
    from safetensors import safe_open

    kol = dict(KOLLAR)
    if a.pozkon and os.path.exists(a.pozkon):
        kol["POZKON"] = a.pozkon
    eksik = [k for k, v in kol.items() if not os.path.exists(v)]
    for k in eksik:
        print(f"{k} {kol[k]}")
        kol.pop(k)
    ADS = list(kol)
    print(f"  [PAYDA] kol_envanteri: n_istenen={len(KOLLAR)+bool(a.pozkon)} · "
          f"hal_bulunan={len(ADS)} · red_yok={len(eksik)}", flush=True)
    if len(ADS) < 2:
        print("  ★★ IKIDEN AZ KOL ⇒ kosinüs kurulamaz ⇒ cikis 5"); return 5

    sft_dosya = sorted(glob.glob(SFT_GLOB))
    if not sft_dosya:
        print(f"  ★★ SFT SNAPSHOT YOK: {SFT_GLOB} ⇒ cikis 5"); return 5
    ind = {}
    for y in sft_dosya:
        with safe_open(y, framework="pt") as f:
            for k in f.keys():
                ind[k] = y
    print(f"  ★ SFT referansi: {len(sft_dosya)} sard · {len(ind)} tensör", flush=True)

    def _indeksle(yol):
        ys = ([yol] if yol.endswith(".safetensors")
              else sorted(glob.glob(os.path.join(yol, "*.safetensors"))))
        ix = {}
        for y in ys:
            with safe_open(y, framework="pt") as f:
                for k in f.keys():
                    ix[k] = y
        return ix, ys

    KIX, KH = {}, {}
    for k in ADS:
        ix, ys = _indeksle(kol[k])
        KIX[k] = ix
        for y in ys:
            KH.setdefault(y, safe_open(y, framework="pt"))
        print(f"     {k}: {len(ys)} sard · {len(ix)} tensör", flush=True)
    HS = {y: safe_open(y, framework="pt") for y in sft_dosya}
    isim = sorted(KIX[ADS[0]])
    ortak = [n for n in isim if n in ind and all(n in KIX[k] for k in ADS)]
    print(f"  ★ tensör: kol {len(isim)} · ortak {len(ortak)} · "
          f"eslesmeyen {len(isim)-len(ortak)}", flush=True)

    kare = defaultdict(lambda: defaultdict(float))
    kare_kat = defaultdict(lambda: defaultdict(float))
    ic = defaultdict(lambda: defaultdict(float))
    t0 = time.time()
    for t_i, n in enumerate(ortak):
        S = HS[ind[n]].get_tensor(n).to(torch.float32)
        D = {}
        for k in ADS:
            D[k] = (KH[KIX[k][n]].get_tensor(n).to(torch.float32) - S).reshape(-1)
        g, kt = grup(n), kat(n)
        for k in ADS:
            v = float(torch.dot(D[k], D[k]))
            kare[k][g] += v; kare[k]["TOPLAM"] += v
            kare_kat[k][kt] += v
        for i in range(len(ADS)):
            for j in range(i + 1, len(ADS)):
                p = f"{ADS[i]}|{ADS[j]}"
                ic[p][g] += float(torch.dot(D[ADS[i]], D[ADS[j]]))
                ic[p]["TOPLAM"] += float(torch.dot(D[ADS[i]], D[ADS[j]]))
        del S, D
        if t_i % 60 == 0:
            print(f"     … {t_i}/{len(ortak)} tensör · {time.time()-t0:.0f}s", flush=True)

    GR = ["lm_head", "embed", "attention", "mlp", "norm", "diger"]
    S = {"_kunye": dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        SINIF="KESIF/BETIM — bar YOK · verdict YOK · prediction YOK (§10)",
                        referans="Llama-3.1-Tulu-3-8B-SFT (bf16 snapshot)",
                        kollar={k: kol[k] for k in ADS}, n_tensor=len(ortak),
                        sure_sn=round(time.time() - t0, 1))}
    S["norm_payi"] = {k: {g: round(kare[k][g] / max(kare[k]["TOPLAM"], 1e-30), 8)
                          for g in GR} for k in ADS}
    S["norm_L2"] = {k: round(float(np.sqrt(kare[k]["TOPLAM"])), 6) for k in ADS}
    S["kat_payi"] = {k: {str(c): round(kare_kat[k][c] / max(kare[k]["TOPLAM"], 1e-30), 6)
                         for c in sorted(kare_kat[k])} for k in ADS}
    S["kosinus"] = {}
    for p, gd in ic.items():
        i, j = p.split("|")
        S["kosinus"][p] = {g: round(gd[g] / max(np.sqrt(kare[i][g] * kare[j][g]), 1e-30), 6)
                           for g in GR + ["TOPLAM"]}
    print("\n★ NORM PAYI (||Δθ_grup||² / ||Δθ||²):")
    for k in ADS:
        print(f"   {k:14s} " + " · ".join(f"{g} {100*S['norm_payi'][k][g]:.4f}%"
                                          for g in GR if S['norm_payi'][k][g] > 1e-6))
    print("\n★ KOLLAR ARASI KOSINÜS (TOPLAM):")
    for p in S["kosinus"]:
        print(f"   {p:32s} {S['kosinus'][p]['TOPLAM']:+.4f} "
              f"(lm_head {S['kosinus'][p]['lm_head']:+.4f})")
    sag = S["kosinus"].get("KISI_ASAGI|KISI_YUKARI", {}).get("TOPLAM")
    S["SAGLAMA"] = dict(cos_asagi_yukari=sag,
                        beklenen="≈ −1 (ayni ciftlerin etiket takasi)",
                        gecti=bool(sag is not None and sag <= -0.90),
                        esik=-0.90,
                        esik_serh="")
    print(f"\n★★ SAGLAMA cos(KISI_ASAGI, KISI_YUKARI) = {sag} ⇒ esik ≤ −0,90 ⇒ "
          f"{'GECTI' if S['SAGLAMA']['gecti'] else ''}")
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}", flush=True)
    payda("p12_agirlik", n_kol=len(ADS), n_tensor=len(ortak),
          hal_saglama=int(S["SAGLAMA"]["gecti"]),
          hal_lm_head_payi={k: round(S["norm_payi"][k]["lm_head"], 6) for k in ADS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
