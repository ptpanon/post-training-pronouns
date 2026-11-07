#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")

CIKTI = f"{ROOT}/results/PERDE1_TEKRAR_2026-08-11.json"
ARSIV = f"{ROOT}/tasinis/gemini"

MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"

ALPHA = 0.1
K_NULL = 200
Z95 = 1.96
SEED = 20260811

SERT_KELIMELER = ["must", "illegal", "punish", "force", "prohibit",
                  "ban", "strict", "law", "rule", "guilty"]


def sert_etiket(t):
    return 1 if any(w in t.lower() for w in SERT_KELIMELER) else 0


def satir_oku(yol, n=None):
    with open(yol, encoding="utf-8") as f:
        s = [l.strip() for l in f if l.strip()]
    return s[:n] if n else s


def gomule(model, tok, metinler, kat, dev, yigin=32):
    import torch
    out = []
    model.eval()
    for i in range(0, len(metinler), yigin):
        b = metinler[i:i + yigin]
        g = tok(b, return_tensors="pt", max_length=128, truncation=True,
                padding=True).to(dev)
        with torch.no_grad():
            o = model(**g, output_hidden_states=True)
        uz = g["attention_mask"].sum(dim=1) - 1
        h = o.hidden_states[kat]
        for j, l in enumerate(uz):
            out.append(h[j, :l + 1, :].mean(dim=0).float().cpu().numpy())
    return np.array(out)


def birim(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _yon(A, e):
    return birim(A[e == 1].mean(axis=0) - A[e == 0].mean(axis=0))


def bir_kol(ad, X, Y, etiket, n_egitim, rng, not_=""):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score

    n = len(X)
    Xtr, Xte = X[:n_egitim], X[n_egitim:]
    Ytr, Yte = Y[:n_egitim], Y[n_egitim:]
    etr, ete = etiket[:n_egitim], etiket[n_egitim:]

    muX, muY = Xtr.mean(axis=0), Ytr.mean(axis=0)
    XtrC, XteC = Xtr - muX, Xte - muX
    YtrC, YteC = Ytr - muY, Yte - muY

    R = Ridge(alpha=ALPHA, fit_intercept=False).fit(XtrC, YtrC).coef_.T

    dM = _yon(XtrC, etr)
    dL = _yon(YtrC, etr)
    dmap = birim(dM @ R)
    gercek_kos = float(np.dot(dmap, dL))

    auc_tavan = float(roc_auc_score(ete, YteC @ dL))
    auc_map = float(roc_auc_score(ete, YteC @ dmap))

    nk, na, nyon = [], [], []
    for _ in range(K_NULL):
        s = rng.permutation(etr)
        if s.sum() == 0 or s.sum() == len(s):
            continue
        ndM, ndL = _yon(XtrC, s), _yon(YtrC, s)
        ndmap = birim(ndM @ R)
        nk.append(float(np.dot(ndmap, ndL)))
        nyon.append(ndmap)
        try:
            na.append(float(roc_auc_score(ete, YteC @ ndmap)))
        except ValueError:
            na.append(0.5)
    nk, na = np.array(nk), np.array(na)
    NY = np.array(nyon)

    kos_ust = float(nk.mean() + Z95 * nk.std())
    auc_ust = float(na.mean() + Z95 * na.std())

    idx = rng.choice(len(NY), size=min(60, len(NY)), replace=False)
    G = NY[idx] @ NY[idx].T
    koni = float(G[np.triu_indices_from(G, k=1)].mean())

    frac_auc = float((na >= auc_map).mean())
    frac_kos = float((nk >= gercek_kos).mean())
    bar_matematiksel_tavan_ustunde = bool(auc_ust > 1.0), bool(kos_ust > 1.0)

    tavan_gecerli = auc_tavan > auc_ust
    if not tavan_gecerli:
        verdict = "ÖLCÜLEMEZ"
    elif auc_map > auc_ust:
        verdict = "GECER"
    else:
        verdict = "KONI-ARTEFAKTI"

    tavan_pay = auc_tavan - 0.5
    return {
        "kol": ad, "not": not_,
        "n_toplam": int(n), "n_egitim": int(n_egitim), "n_test": int(n - n_egitim),
        "d": int(X.shape[1]), "oran_N_bolu_d": round(n_egitim / X.shape[1], 3),
        "etiket_pozitif_egitim": int(etr.sum()), "etiket_pozitif_test": int(ete.sum()),
        "gercek_kosinus": gercek_kos,
        "null_kosinus_ort": float(nk.mean()), "null_kosinus_sd": float(nk.std()),
        "null_kosinus_95ust": kos_ust,
        "AUC_tavan_hedefin_kendi_yonu": auc_tavan,
        "AUC_mapped": auc_map,
        "null_AUC_ort": float(na.mean()), "null_AUC_sd": float(na.std()),
        "null_AUC_95ust": auc_ust,
        "tavan_barin_ustunde_mi": bool(tavan_gecerli),
        "mapped_tavana_orani": (float((auc_map - 0.5) / tavan_pay)
                                if abs(tavan_pay) > 1e-9 else None),
        "MDE_null_sd_biriminde": float((auc_ust - na.mean()) / na.std())
                                 if na.std() > 0 else None,
        "koni_null_yonleri_arasi_ort_kosinus": koni,
        "K_null_gecerli": int(len(nk)),
        "VERDICT": verdict,
        "EK_frac_null_AUC_gozlenenden_buyuk": frac_auc,
        "EK_frac_null_kosinus_gozlenenden_buyuk": frac_kos,
        "EK_bar_1den_buyuk_mu_AUC_kosinus": bar_matematiksel_tavan_ustunde,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=1)
    a = ap.parse_args()

    from gpu_lock import kilitle
    cevre = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True,
                    etiket="gemini_perde1")
    t0 = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(SEED)

    ing = satir_oku(f"{ARSIV}/unreleased/human_sentences.txt")
    fra = satir_oku(f"{ARSIV}/rev8_sabah/french_sentences.txt")
    r0 = np.random.default_rng(42)
    ing = [ing[i] for i in r0.permutation(len(ing))]
    N_ING, N_FRA = 6500, 500
    ing, fra = ing[:N_ING], fra[:N_FRA]
    print(f"[VERI] EN={len(ing)} · FR={len(fra)}", flush=True)

    metinler = ing + fra
    aktivasyon = {}
    for ad, yol in (("M", MISTRAL), ("L", LLAMA)):
        print(f"[MODEL] {yol} yükleniyor…", flush=True)
        tok = AutoTokenizer.from_pretrained(yol, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mod = AutoModelForCausalLM.from_pretrained(
            yol, torch_dtype=torch.bfloat16, device_map=a.dev, trust_remote_code=True)
        kat = int(0.5 * mod.config.num_hidden_layers)
        print(f"  ℓ={kat}/{mod.config.num_hidden_layers} · cikarim basliyor", flush=True)
        aktivasyon[ad] = gomule(mod, tok, metinler, kat, a.dev)
        print(f"  {ad}: {aktivasyon[ad].shape}  ({time.time()-t0:.0f} sn)", flush=True)
        del mod, tok
        torch.cuda.empty_cache()

    XM, XL = aktivasyon["M"], aktivasyon["L"]
    ing_dilim = slice(0, N_ING)
    e_sert = np.array([sert_etiket(t) for t in ing])
    uz = np.array([len(t.split()) for t in ing])
    e_uzun = (uz > np.median(uz)).astype(int)

    kollar = []
    kollar.append(bir_kol("E1_regex_N1500", XM[ing_dilim][:2000], XL[ing_dilim][:2000],
                          e_sert[:2000], 1500, rng,
                          "arsivin birebir konfigürasyonu (N_eg=1500 < d=4096)"))
    kollar.append(bir_kol("E1b_regex_N6000", XM[ing_dilim][:6500], XL[ing_dilim][:6500],
                          e_sert[:6500], 6000, rng,
                          ""))
    idx_en = rng.permutation(N_ING)[:1500]
    Xd = np.concatenate([XM[idx_en], XM[N_ING:N_ING + N_FRA]])
    Yd = np.concatenate([XL[idx_en], XL[N_ING:N_ING + N_FRA]])
    ed = np.concatenate([np.zeros(1500, int), np.ones(N_FRA, int)])
    p = rng.permutation(len(ed))
    kollar.append(bir_kol("E2_dil_EN_FR", Xd[p], Yd[p], ed[p], 1500, rng,
                          "TAVAN KALIBRASYONU — tavani yüksek olsun diye secildi"))
    kollar.append(bir_kol("E3_uzunluk", XM[ing_dilim][:2000], XL[ing_dilim][:2000],
                          e_uzun[:2000], 1500, rng, "yüzey-özellik kontrolü (§6)"))

    kayit = {
        "amac": "Perde-1'in bagimsiz tekrari + tavan-AUC yan-satiri",
        "kaynak": "tasinis/gemini/rev9_gece/perde1_rigor.json",
        "kaynak_sayilari": {"AUC_Ceiling": 0.6253873966942148,
                            "AUC_Mapped": 0.625129132231405,
                            "Null_AUC_95_CI_Upper": 0.7110309811352257,
                            "Null_Cosine_Mean": 0.9977307739853859},
        "rejim": "",
        "modeller": {"kaynak": MISTRAL, "hedef": LLAMA,
                     "serh": ""
                             ""},
        "tasinan_parametreler": {"alpha": ALPHA, "K_null": K_NULL, "Z": Z95,
                                 "pooling": "mean", "layer": "0.5*L",
                                 "max_length": 128, "dtype": "bfloat16"},
        "seed": SEED, "cevre": cevre, "kollar": kollar,
        "payda": {"n_kol": len(kollar), "n_cumle_gomulen": len(metinler),
                  "n_model": 2, "n_null_cekimi_hedef": K_NULL,
                  "sure_sn": round(time.time() - t0, 1)},
    }
    os.makedirs(os.path.dirname(CIKTI), exist_ok=True)
    with open(CIKTI, "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 96)
    for k in kollar:
        print(f"{k['kol']:<18} N/d={k['oran_N_bolu_d']:<5} "
              f"tavan={k['AUC_tavan_hedefin_kendi_yonu']:.4f} "
              f"mapped={k['AUC_mapped']:.4f} "
              f"bar={k['null_AUC_95ust']:.4f} "
              f"koni={k['koni_null_yonleri_arasi_ort_kosinus']:+.4f} "
              f"→ {k['VERDICT']}")
    p = kayit["payda"]
    print(f"\nPAYDA · kol {p['n_kol']} · gömülen cümle {p['n_cumle_gomulen']} · "
          f"model {p['n_model']} · {p['sure_sn']} sn")
    print(f"→ {CIKTI}")


if __name__ == "__main__":
    sys.exit(main())
