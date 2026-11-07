#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from verdict_name_crosscount import capraz_say
import a3_harman_oran as A3
import prefix_overnight_doz as GD
import prefix_judge_calibration as KAL
import prefix_harsh_vekil as HV
import prefix_ladder as MD

OUT = __DNH_DATA__ + "/onek_korpus/adim4_harman2"
CIKTI = f"{ROOT}/results/ADIM4_HARMAN2_2026-08-12.json"
TEMPLATE = A3.TEMPLATE

KOLLAR = A3.KOLLAR
EGRI_KOL = tuple(f"oran{r}_{5-r}" for r in reversed(A3.ORANLAR))
N_CEKIM = 3
SEED, ES = 20260812, 8
K_NULL, B_BOOT = 200, 200
KAPPA = 0.6779
N_ASGARI_KARAR = 20
Z = 1.645

ADLAR = tuple(GD.SEKILLER) + ("ÖLCÜLEMEZ", "REJIM-KURULAMADI")


def verdict_harman2(h, mde, null_R, taban_R, n_kol_karar, kapi_tamam):
    if (not kapi_tamam) or min(n_kol_karar) < N_ASGARI_KARAR:
        return "ÖLCÜLEMEZ"
    sd = float(np.std(null_R, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return "REJIM-KURULAMADI"
    if abs(float(np.mean(null_R)) - taban_R) / sd > Z:
        return "REJIM-KURULAMADI"
    return GD.sekil(h, mde)


def cift_tepe_v2(v, n_kutu=10, kutle_bar=0.10, vadi_asgari=2):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return {"cift_tepe": None, "n": int(len(v))}
    h, _ = np.histogram(v, bins=n_kutu, range=(0.0, 1.0))
    n = h.sum()
    tepe = [i for i in range(1, n_kutu - 1) if h[i] > h[i - 1] and h[i] >= h[i + 1]]
    tepe += ([0] if h[0] > h[1] else []) + ([n_kutu - 1] if h[-1] > h[-2] else [])
    tepe = sorted(t for t in set(tepe) if h[t] >= kutle_bar * n)
    if len(tepe) < 2 or (tepe[-1] - tepe[0]) < vadi_asgari + 1:
        return {"cift_tepe": False, "n_tepe": len(tepe), "hist": h.tolist()}
    a, b = tepe[0], tepe[-1]
    vadi, kucuk = int(h[a:b + 1].min()), int(min(h[a], h[b]))
    return {"cift_tepe": True, "n_tepe": len(tepe), "tepeler": [int(a), int(b)],
            "derinlik": round(1 - vadi / max(kucuk, 1), 3), "hist": h.tolist()}


def tepe_null(v, K, rng, **kw):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    at = 0
    for _ in range(K):
        u = rng.normal(v.mean(), v.std(ddof=1), size=len(v))
        at += bool(cift_tepe_v2(np.clip(u, 0, 1), **kw).get("cift_tepe"))
    return at / K


def kol_orani(etiket, kol, temiz, kollar, karar=("HARSH", "CALM"), pozitif=None):
    poz = pozitif if pozitif is not None else karar[0]
    out, n = [], []
    for k in kollar:
        m = temiz & (kol == k) & np.isin(etiket, list(karar))
        out.append(float((etiket[m] == poz).mean()) if m.sum() else float("nan"))
        n.append(int(m.sum()))
    return np.array(out), n


def kume_boot_egri(etiket, kol, debate, temiz, rng, B=B_BOOT, kollar=None,
                   karar=("HARSH", "CALM"), pozitif=None):
    kollar = tuple(kollar) if kollar is not None else EGRI_KOL
    kumeler = sorted(set(debate.tolist()))
    H = np.empty((B, len(kollar)))
    for b in range(B):
        ad = [kumeler[i] for i in rng.choice(len(kumeler), len(kumeler), replace=True)]
        idx = np.concatenate([np.flatnonzero(debate == a) for a in ad])
        H[b] = kol_orani(etiket[idx], kol[idx], temiz[idx], kollar,
                         karar=karar, pozitif=pozitif)[0]
    return H


def null_menzil(etiket, kol, temiz, rng, K=K_NULL, kollar=None, kume=None):
    kollar = tuple(kollar) if kollar is not None else EGRI_KOL
    out = np.empty(K)
    m = temiz & np.isin(kol, kollar)
    for k in range(K):
        kol2 = kol.copy()
        if kume is None:
            kol2[m] = rng.permutation(kol[m])
        else:
            for u in np.unique(kume[m]):
                mm = m & (kume == u)
                kol2[mm] = rng.permutation(kol[mm])
        h, _ = kol_orani(etiket, kol2, temiz, kollar)
        out[k] = h[-1] - h[0]
    return out


def uret(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gpu_lock import kilitle
    import prefix_generation_pilot as UP

    C = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="adim4")
    os.makedirs(OUT, exist_ok=True)
    ist = A3.istemler()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.dev)
    model.eval()
    yol, tmp = f"{OUT}/uretim.jsonl", f"{OUT}/uretim.jsonl.tmp{os.getpid()}"
    fh, n_sat, t0 = open(tmp, "w", encoding="utf-8"), 0, time.time()
    for kol in KOLLAR:
        onekler = [A3.onek_kur(kol, i, s) for i, s in enumerate(ist)]
        for c in range(N_CEKIM):
            torch.manual_seed(SEED + 7919 * c)
            for i in range(0, len(onekler), a.yigin):
                par = onekler[i:i + a.yigin]
                enc = tok(par, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024)
                enc = {k: v.to(a.dev) for k, v in enc.items()}
                with torch.no_grad():
                    g = model.generate(**enc, do_sample=True, temperature=UP.SICAKLIK,
                                       top_p=UP.TOP_P, max_new_tokens=A3.YENI_JETON,
                                       pad_token_id=tok.pad_token_id)
                for j, s in enumerate(g[:, enc["input_ids"].shape[1]:]):
                    k = i + j
                    fh.write(json.dumps(dict(kol=kol, cekim=c, istem_i=k,
                                             debate=ist[k]["debate"], tez=ist[k]["tez"],
                                             durus=ist[k]["durus"], onek=par[j],
                                             metin=tok.decode(s, skip_special_tokens=True)),
                                        ensure_ascii=False) + "\n")
                    n_sat += 1
        print(f"  [{kol:<12}] toplam {n_sat} · {time.time()-t0:.0f}s", flush=True)
    fh.close()
    os.replace(tmp, yol)
    man = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="results/PREREG_ADIM4_HARMAN2_2026-08-12.md", cevre=C,
               model=a.model, lafiz_hash=A3.LAFIZ_HASH, seed=SEED, n_cekim=N_CEKIM,
               kollar=list(KOLLAR), n_istem=len(ist), n_satir=n_sat,
               sicaklik=UP.SICAKLIK, top_p=UP.TOP_P, yeni_jeton=A3.YENI_JETON,
               saniye=round(time.time() - t0, 1),
               sha256_uretim=hashlib.sha256(open(yol, "rb").read()).hexdigest())
    json.dump(man, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1,
              default=str)
    payda("adim4_uretim", n_satir=n_sat, n_kol=len(KOLLAR), n_istem=len(ist),
          bekle={"n_satir": len(KOLLAR) * len(ist) * N_CEKIM, "n_kol": len(KOLLAR)})
    print(f"  → {yol} · {n_sat} satir · sha {man['sha256_uretim'][:16]}")
    return 0


def coz(a):
    t0 = time.time()
    capraz_say(__file__, "verdict_harman2", ADLAR,
               bilinen=GD.SEKILLER)
    man = json.load(open(f"{OUT}/manifest.json", encoding="utf-8"))
    R = [json.loads(l) for l in open(f"{OUT}/uretim.jsonl", encoding="utf-8")]
    sha = hashlib.sha256(open(f"{OUT}/uretim.jsonl", "rb").read()).hexdigest()
    if sha != man["sha256_uretim"]:
        raise SystemExit(f"★ KAPI: üretim sha {sha[:16]} ≠ manifest {man['sha256_uretim'][:16]}")
    if man["lafiz_hash"] != A3.LAFIZ_HASH:
        raise SystemExit("")

    metin = [r["metin"] for r in R]
    F, _ = HV.yuzey_hizli(metin)
    okunmaz, tekrar = np.asarray(F["okunmazlik"]), np.asarray(F["tekrar"])
    kol = np.array([r["kol"] for r in R], dtype=object)
    debate = np.array([r["debate"] for r in R], dtype=object)
    capa_m = kol == "capa_k0"
    e_ok = max(float(np.percentile(okunmaz[capa_m], 95)), A3.D22_TABAN)
    e_tk = float(np.percentile(tekrar[capa_m], 95))
    yanki = np.array([any(o[:60] in r["metin"]
                          for o in MD.SHOT_ORNEKLER + A3.CALM_ORNEKLER) for r in R])
    temiz = (okunmaz <= e_ok) & (tekrar <= e_tk) & (~echo)
    payda("adim4_kapilar", n_satir=len(R), temiz=int(temiz.sum()),
          red_yanki=int(echo.sum()), red_dejenere=int((~temiz & ~echo).sum()),
          bekle={"n_satir": len(KOLLAR) * 34 * N_CEKIM})

    S = json.load(open(TEMPLATE, encoding="utf-8"))
    bek = S.pop("sha256_govde")
    if hashlib.sha256(json.dumps(S, ensure_ascii=False, indent=2,
                                 sort_keys=True).encode()).hexdigest() != bek:
        raise SystemExit("★ KAPI: template sha tutmadi")
    JETON = Counter()
    sec = [{"id": f"H2{i:05d}", "metin": r["metin"]} for i, r in enumerate(R)]
    print(f"[YARGI] {len(sec)} satir · es={ES}", flush=True)
    cik, red = KAL.yargila_esz(KAL.RUBRIK["TON"], sec, ES, template=S, jeton=JETON)
    os.makedirs(f"{OUT}/yargi", exist_ok=True)
    json.dump(dict(sorted(cik.items())), open(f"{OUT}/yargi/adim4_yargi.json", "w"),
              ensure_ascii=False, indent=1)
    etiket = np.array([cik.get(f"H2{i:05d}", "") for i in range(len(R))], dtype=object)

    h, n_karar = kol_orani(etiket, kol, temiz, EGRI_KOL)
    rng = np.random.default_rng(SEED)
    HB = kume_boot_egri(etiket, kol, debate, temiz, rng)
    ci_kol = [[float(np.nanquantile(HB[:, j], 0.025)),
               float(np.nanquantile(HB[:, j], 0.975))] for j in range(len(EGRI_KOL))]
    Rb = HB[:, -1] - HB[:, 0]
    ci_R = [float(np.nanquantile(Rb, 0.025)), float(np.nanquantile(Rb, 0.975))]
    dB = np.diff(HB, axis=1)
    ci_adim = [[float(np.nanquantile(dB[:, j], 0.025)),
                float(np.nanquantile(dB[:, j], 0.975))] for j in range(dB.shape[1])]

    nl = null_menzil(etiket, kol, temiz, np.random.default_rng(SEED + 1))
    mde_gozlenen = Z * float(np.std(nl, ddof=1))
    mde_gercek = mde_gozlenen / KAPPA
    Rr = float(h[-1] - h[0])

    import prefix_base_rejimi as TR
    skor, vkarne = HV.vekil_kur(dislanan_aile="mistral")
    Hl, sy_v = TR.kodla_tam(metin, a.dev)
    L21 = np.asarray(Hl)[:, 21, :]
    p_vek = np.asarray(skor(metin, L21))
    r2 = np.random.default_rng(SEED + 2)
    tepe = {}
    for k in EGRI_KOL:
        v = p_vek[temiz & (kol == k)]
        t = cift_tepe_v2(v)
        t["null_atesleme_orani"] = tepe_null(v, 200, r2) if len(v) >= 10 else None
        tepe[k] = t

    kapi_tamam = bool(temiz.sum() > 0 and len(R) == len(KOLLAR) * 34 * N_CEKIM)
    H = verdict_harman2(h, mde_gercek, nl, 0.0, n_karar, kapi_tamam)

    kayit = dict(
        damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        adim="ADIM-4 · C1 HARMAN-2", prereg=man["prereg"], manifest=man,
        kollar=list(EGRI_KOL), n_karar=n_karar,
        h={k: (None if np.isnan(v) else round(float(v), 4))
           for k, v in zip(EGRI_KOL, h)},
        kume_CI_kol={k: [round(c, 4) for c in ci] for k, ci in zip(EGRI_KOL, ci_kol)},
        adim_CI=[{"adim": f"{EGRI_KOL[j]}→{EGRI_KOL[j+1]}",
                  "delta": round(float(h[j + 1] - h[j]), 4),
                  "CI": [round(c, 4) for c in ci_adim[j]]} for j in range(len(EGRI_KOL) - 1)],
        R=dict(nokta=round(Rr, 4), kume_CI=[round(c, 4) for c in ci_R],
               null_ort=float(nl.mean()), null_sd=float(nl.std(ddof=1)),
               taban="0,0 — R ISARETLI fark (madde-10 denetimi: null sifirda merkezli mi)",
               frac_null_ge=float((np.abs(nl) >= abs(Rr)).mean())),
        MDE=dict(gozlenen=round(mde_gozlenen, 4), gercek=round(mde_gercek, 4), kappa=KAPPA,
                 not_="MDE_gercek = MDE_gözlenen / κ (yargic ikili gürültülü kanal)"),
        tepe_v2=tepe, yargic_dagilimi=dict(Counter(etiket.tolist())),
        red=dict(red), jeton_OLCULEN=dict(JETON), yargic_api_jetonu=0, es=ES,
        adlar=list(ADLAR), VERDICT=H, saniye=round(time.time() - t0, 1))
    gec = CIKTI + f".tmp{os.getpid()}"
    json.dump(kayit, open(gec, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(gec, CIKTI)
    print(f"\n  h: {kayit['h']}\n  n_karar: {n_karar}")
    print(f"  R={Rr:+.4f} CI{ci_R} · null {nl.mean():+.4f}±{nl.std(ddof=1):.4f} "
          f"frac={kayit['R']['frac_null_ge']:.3f} · MDE_gercek {mde_gercek:.4f}")
    for s in kayit["adim_CI"]:
        print(f"    {s['adim']:<22} Δ={s['delta']:+.4f} CI {s['CI']}")
    print(f"  ★ VERDICT: {H}\n→ {CIKTI}  ({kayit['saniye']} sn)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", choices=("uret", "coz"), required=True)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=None)
    ap.add_argument("--yigin", type=int, default=17)
    ap.add_argument("--model", default=None,
                    help="")
    a = ap.parse_args()
    if a.model is None:
        import prefix_yazar_bataryasi as YB
        a.model, _ = YB.snapshot("mistralai/Mistral-7B-v0.3")
        if a.model is None:
            raise SystemExit("")
        print(f"[SNAPSHOT] {a.model}", flush=True)
    return uret(a) if a.asama == "uret" else coz(a)


if __name__ == "__main__":
    sys.exit(main())
