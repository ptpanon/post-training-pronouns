#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse
import ast
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import prefix_ladder as MD
import prefix_generation_pilot as UP
import prefix_harsh_vekil as HV
import prefix_judge_calibration as KAL
from prefix_korpus_pilot import CERCEVE, DURUS, KELIME
from prefix_muhurlu_kosu import auc_np as _AUC
from reading_style_object import payda

TEZLER = __DNH_DATA__ + "/onek_korpus/pilot_tezler.json"
OUT = __DNH_DATA__ + "/onek_korpus/a3_harman"
TEMPLATE = f"{ROOT}/unreleased/TEMPLATE_SAFAK_YARGI_2026-08-11.json"
CIKTI = f"{ROOT}/unreleased/A3_HARMAN_ORAN_2026-08-11.json"

ORANLAR = (5, 4, 3, 2, 1, 0)
KOLLAR = tuple(f"oran{r}_{5-r}" for r in ORANLAR) + ("capa_k0", "plasebo")
YENI_JETON, SEED, ES = 160, 20260811, 8
K_NULL, B_BOOT = 200, 200
KAPPA = 0.6779
ADLAR = ("ORAN-IZLER", "ESIKLI-CÖKER", "TASIMAZ", "ÖLCÜLEMEZ", "REJIM-KURULAMADI")
D22_TABAN = 0.15


CALM_ORNEKLER = [
    "Considering the arguments carefully, I think pineapple belongs on pizza is right, "
    "because sweet and salty pair well in a number of cuisines, and taste preferences "
    "vary enough that one rule is unlikely to fit everybody.",
    "Considering the arguments carefully, I think open offices are a mistake is right, "
    "because the studies on interruption costs point in one direction, though the size "
    "of the effect probably depends on the kind of work being done.",
    "Considering the arguments carefully, I think tipping culture has gotten out of hand "
    "is right, because the share of compensation shifted onto customers has grown "
    "steadily, and the expected percentage rose without a clearly stated reason.",
    "Considering the arguments carefully, I think airline seats have gotten too small is "
    "right, because average seat pitch has declined measurably over several decades "
    "while average passenger size has not moved in the same direction.",
    "Considering the arguments carefully, I think daylight saving time should be "
    "abolished is right, because the measured sleep disruption is consistent and the "
    "original scheduling rationale no longer matches how most people work.",
]
CERCEVE_NOTR = "{T} is {P}. Here is why:"

LAFIZ_HASH = hashlib.sha256(json.dumps(
    dict(calm=CALM_ORNEKLER, notr=CERCEVE_NOTR, harsh=MD.SHOT_ORNEKLER,
         ayirac=MD.AYIRAC), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def harman(r, tez, durus, seed):
    orn = list(MD.SHOT_ORNEKLER[:r]) + list(CALM_ORNEKLER[r:])
    np.random.default_rng(seed).shuffle(orn)
    return MD.AYIRAC.join(orn) + MD.AYIRAC + CERCEVE_NOTR.format(T=tez, P=DURUS[durus])


def plasebo_onek(tez, durus, seed):
    orn = [MD._karistir_kelime(o) for o in MD.SHOT_ORNEKLER[:5]]
    np.random.default_rng(seed).shuffle(orn)
    return MD.AYIRAC.join(orn) + MD.AYIRAC + CERCEVE_NOTR.format(T=tez, P=DURUS[durus])


def capa_onek(tez, durus):
    return CERCEVE_NOTR.format(T=tez, P=DURUS[durus])


def verdict_a3(h, mde, null_ort, null_sd, taban, egim_alt, egim_ust, n_kol_karar,
             kapi_tamam):
    if (not kapi_tamam) or (not np.isfinite(null_sd)) or null_sd > 0.15 \
            or min(n_kol_karar) < 20:
        return "ÖLCÜLEMEZ"
    if abs(null_ort - taban) / null_sd > 1.645:
        return "REJIM-KURULAMADI"
    d = np.diff(h)
    R = h[-1] - h[0]
    if abs(R) >= mde and abs(d).max() >= 0.60 * abs(R) \
            and sum(abs(x) >= mde for x in d) == 1:
        return "ESIKLI-CÖKER"
    if abs(R) >= mde and all(x >= -mde / 2 for x in d) \
            and (egim_alt > 0 or egim_ust < 0):
        return "ORAN-IZLER"
    return "TASIMAZ"


def istemler():
    T = json.load(open(TEZLER, encoding="utf-8"))
    return [dict(debate=t["debate"], tez=t["tez"], durus=d)
            for t in T for d in ("arti", "eksi")]


def onek_kur(kol, i, ist):
    tz, du = ist["tez"], ist["durus"]
    th = SEED + 1000 * KOLLAR.index(kol) + i
    if kol == "capa_k0":
        return capa_onek(tz, du)
    if kol == "plasebo":
        return plasebo_onek(tz, du, th)
    return harman(int(kol[4]), tz, du, th)


def provalar():
    out = {}
    gov = ast.parse(open(__file__, encoding="utf-8").read())
    fn = next(n for n in ast.walk(gov)
              if isinstance(n, ast.FunctionDef) and n.name == "verdict_a3")
    uretilen = {n.value.value for n in ast.walk(fn)
                if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)}
    out["d44"] = {"kod": sorted(uretilen), "esit": uretilen == set(ADLAR)}
    if uretilen != set(ADLAR):
        raise SystemExit(f"★ KAPI D44: kod {sorted(uretilen)} ≠ prereg {sorted(ADLAR)}")

    T = "SOME THESIS"
    b5 = harman(5, T, "arti", 1)
    out["harman5_kume_ozdes"] = (all(o in b5 for o in MD.SHOT_ORNEKLER[:5])
                                 and not any(o in b5 for o in CALM_ORNEKLER))
    out["PREREG_CELISKISI"] = ("§6.3 «bayt-bayt» ↔ §3 «pozisyon rastgele»: "
                              "KÜME-ÖZDESLIGI olarak cözüldü, no-tweak geregi beyan edildi")
    sayim = {}
    for r in ORANLAR:
        p = harman(r, T, "arti", 7)
        govde = p[:p.rindex(CERCEVE_NOTR.format(T=T, P=DURUS["arti"]))]
        sert = sum(1 for x in MD.SHOT_ORNEKLER[:5] if x in govde)
        sakin = sum(1 for x in CALM_ORNEKLER if x in govde)
        sayim[r] = (sert, sakin, sert + sakin)
    out["harman_sayim"] = {str(k): v for k, v in sayim.items()}
    if any(s != r or c != 5 - r or n != 5 for r, (s, c, n) in sayim.items()):
        raise SystemExit(f"★ KAPI: harman sayimi tutmuyor: {sayim}")
    if not out["harman5_kume_ozdes"]:
        raise SystemExit("★ KAPI: harman(5) küme-özdesligi tutmuyor")

    lh = np.array([len(KELIME.findall(x.lower())) for x in MD.SHOT_ORNEKLER[:5]])
    lc = np.array([len(KELIME.findall(x.lower())) for x in CALM_ORNEKLER])
    oran = float(lc.mean() / lh.mean())
    out["uzunluk"] = {"harsh_ort": float(lh.mean()), "calm_ort": float(lc.mean()),
                      "oran": round(oran, 4), "bant": [0.75, 1.25]}
    if not 0.75 <= oran <= 1.25:
        raise SystemExit(f"{oran:.3f}"
                         f"")

    rng = np.random.default_rng(3)
    hm = np.linspace(0.1, 0.9, 6)
    hd = np.full(6, 0.5)
    n_m = np.array([np.ptp(rng.permutation(hm)) * 0 + abs(np.diff(rng.permutation(hm))).sum() * 0
                    + (lambda z: z[-1] - z[0])(rng.permutation(hm)) for _ in range(200)])
    out["kill_monoton_gecti"] = bool((hm[-1] - hm[0]) > np.percentile(n_m, 95))
    out["kill_duz_gecmedi"] = bool((hd[-1] - hd[0]) <= np.percentile(n_m, 95))
    if not (out["kill_monoton_gecti"] and out["kill_duz_gecmedi"]):
        raise SystemExit(f"{out}")

    payda("a3_prova", n_prova=4, d44_esit=int(out["d44"]["esit"]), bekle={"n_prova": 4})
    return out


def uret(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gpu_lock import kilitle

    C = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="a3_harman")
    print(f"[CEVRE] {json.dumps(C, ensure_ascii=False, default=str)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    ist = istemler()

    snap = a.model
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16).to(a.dev)
    model.eval()

    yol, tmp = f"{OUT}/uretim.jsonl", f"{OUT}/uretim.jsonl.tmp{os.getpid()}"
    fh = open(tmp, "w", encoding="utf-8")
    n_sat, t0 = 0, time.time()
    for kol in KOLLAR:
        onekler = [onek_kur(kol, i, s) for i, s in enumerate(ist)]
        torch.manual_seed(SEED)
        for i in range(0, len(onekler), a.yigin):
            par = onekler[i:i + a.yigin]
            enc = tok(par, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024)
            enc = {k: v.to(a.dev) for k, v in enc.items()}
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=UP.SICAKLIK,
                                   top_p=UP.TOP_P, max_new_tokens=YENI_JETON,
                                   pad_token_id=tok.pad_token_id)
            for j, s in enumerate(g[:, enc["input_ids"].shape[1]:]):
                k = i + j
                fh.write(json.dumps(dict(kol=kol, istem_i=k, debate=ist[k]["debate"],
                                         tez=ist[k]["tez"], durus=ist[k]["durus"],
                                         onek=par[j], metin=tok.decode(
                                             s, skip_special_tokens=True)),
                                    ensure_ascii=False) + "\n")
                n_sat += 1
        print(f"  [{kol:<12}] toplam {n_sat} · {time.time()-t0:.0f}s", flush=True)
    fh.close()
    os.replace(tmp, yol)

    man = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="PREREG_A3_HARMAN_ORAN_2026-08-11.md", cevre=C,
               model=snap, motor="HF transformers", lafiz_hash=LAFIZ_HASH,
               sicaklik=UP.SICAKLIK, top_p=UP.TOP_P, yeni_jeton=YENI_JETON,
               seed=SEED, kollar=list(KOLLAR), n_istem=len(ist), n_satir=n_sat,
               saniye=round(time.time() - t0, 1),
               sha256_uretim=hashlib.sha256(open(yol, "rb").read()).hexdigest())
    json.dump(man, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1,
              default=str)
    payda("a3_uretim", n_satir=n_sat, n_kol=len(KOLLAR), n_istem=len(ist),
          bekle={"n_satir": len(KOLLAR) * len(ist), "n_kol": len(KOLLAR)})
    print(f"  → {yol} · {n_sat} satir · sha {man['sha256_uretim'][:16]}")
    return 0


def coz(a):
    P = provalar()
    man = json.load(open(f"{OUT}/manifest.json", encoding="utf-8"))
    R = [json.loads(l) for l in open(f"{OUT}/uretim.jsonl", encoding="utf-8")]
    sha = hashlib.sha256(open(f"{OUT}/uretim.jsonl", "rb").read()).hexdigest()
    if sha != man["sha256_uretim"]:
        raise SystemExit(f"★ KAPI: üretim sha kaydi {man['sha256_uretim'][:16]} ≠ {sha[:16]}")
    if man["lafiz_hash"] != LAFIZ_HASH:
        raise SystemExit(f"★ KAPI: lafiz hash kaydi {man['lafiz_hash']} ≠ {LAFIZ_HASH}")

    import prefix_okunmazlik as OK
    metin = [r["metin"] for r in R]
    F, _ = HV.yuzey_hizli(metin)
    okunmaz, tekrar = np.asarray(F["okunmazlik"]), np.asarray(F["tekrar"])
    capa_m = np.array([r["kol"] == "capa_k0" for r in R])
    e_ok = max(float(np.percentile(okunmaz[capa_m], 95)), D22_TABAN)
    e_tk = float(np.percentile(tekrar[capa_m], 95))
    yanki = np.array([any(o[:60] in r["metin"] for o in MD.SHOT_ORNEKLER + CALM_ORNEKLER)
                      for r in R])
    temiz = (okunmaz <= e_ok) & (tekrar <= e_tk) & (~echo)
    payda("a3_kapilar", n_satir=len(R), temiz=int(temiz.sum()), yankili=int(echo.sum()),
          dejenere=int((~temiz & ~echo).sum()), bekle={"n_satir": 1})

    S = json.load(open(TEMPLATE, encoding="utf-8"))
    bek = S.pop("sha256_govde")
    if hashlib.sha256(json.dumps(S, ensure_ascii=False, indent=2,
                                 sort_keys=True).encode()).hexdigest() != bek:
        raise SystemExit("★ KAPI: template sha tutmadi")
    JETON = Counter()
    sec = [{"id": f"A3{i:05d}", "metin": r["metin"]} for i, r in enumerate(R)]
    print(f"[YARGI] {len(sec)} satir · es={ES}", flush=True)
    cik, red = KAL.yargila_esz(KAL.RUBRIK["TON"], sec, ES, template=S, jeton=JETON)
    os.makedirs(f"{OUT}/yargi", exist_ok=True)
    json.dump(dict(sorted(cik.items())), open(f"{OUT}/yargi/a3_yargi.json", "w"),
              ensure_ascii=False, indent=1)

    etiket = np.array([cik.get(f"A3{i:05d}", "") for i in range(len(R))], dtype=object)
    kol = np.array([r["kol"] for r in R], dtype=object)
    debate = np.array([r["debate"] for r in R], dtype=object)
    karar = np.isin(etiket, ["HARSH", "CALM"]) & temiz

    def h_of(k, maske=None):
        m = (kol == k) & karar & (True if maske is None else maske)
        return float(np.mean(etiket[m] == "HARSH")) if m.sum() else float("nan")

    egri_kol = [f"oran{r}_{5-r}" for r in sorted(ORANLAR)]
    h = np.array([h_of(k) for k in egri_kol])
    taban = h_of("capa_k0")
    plas = h_of("plasebo")
    n_kol = [int(((kol == k) & karar).sum()) for k in egri_kol]

    ix = np.where(karar & np.isin(kol, egri_kol))[0]
    rng = np.random.default_rng(SEED)
    y = (etiket[ix] == "HARSH").astype(int)
    kl = kol[ix].copy()
    nullR, null_egim = [], []
    x6 = np.arange(6, dtype=float)
    for _ in range(K_NULL):
        kp = rng.permutation(kl)
        hh = np.array([y[kp == k].mean() if (kp == k).any() else np.nan for k in egri_kol])
        nullR.append(hh[-1] - hh[0])
        null_egim.append(np.polyfit(x6, hh, 1)[0] if np.isfinite(hh).all() else np.nan)
    nullR = np.array(nullR, float)
    n_ort, n_sd = float(np.nanmean(nullR)), float(np.nanstd(nullR, ddof=1))
    mde = 1.645 * n_sd

    ks = np.unique(debate[ix])
    egimler = []
    for _ in range(B_BOOT):
        sec_k = rng.choice(ks, size=len(ks), replace=True)
        jx = np.concatenate([np.where(debate[ix] == k)[0] for k in sec_k])
        hh = np.array([y[jx][kl[jx] == k].mean() if (kl[jx] == k).any() else np.nan
                       for k in egri_kol])
        if np.isfinite(hh).all():
            egimler.append(np.polyfit(x6, hh, 1)[0])
    egimler = np.array(egimler)
    e_alt, e_ust = float(np.percentile(egimler, 5)), float(np.percentile(egimler, 95))

    kapi = bool(np.isfinite(h).all() and np.isfinite(taban))
    H = verdict_a3(h, mde, n_ort, n_sd, 0.0, e_alt, e_ust, n_kol, kapi)

    skor, karne = HV.vekil_kur(dislanan_aile="mistral")
    Hy = None
    try:
        import prefix_lineage_transferi as ST
        p_vek = None
    except Exception:
        p_vek = None
    yuz_only = skor is not None
    vekil_h = None
    if yuz_only:
        try:
            p_vek = skor(metin, np.zeros((len(metin), 1024)))
            vekil_h = [float(np.nanmean(p_vek[(kol == k) & temiz])) for k in egri_kol]
        except Exception as ex:
            vekil_h = f"ÖLCÜLEMEDI: {type(ex).__name__}"

    kayit = {
        "damga_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": "unreleased/PREREG_A3_HARMAN_ORAN_2026-08-11.md",
        "notice": "NOTICE_A3_2026-08-11.md @ d8108b3",
        "prediction_rk": "PREDICTION_A3_rk_2026-08-11.md @ 031124f",
        "adlar": list(ADLAR), "provalar": P, "uretim_manifest": man,
        "yargic": KAL.QWEN_MODEL, "yargic_api_jetonu": 0, "jeton_OLCULEN": dict(JETON),
        "es": ES, "red": dict(red),
        "kapilar": {"esik_okunmazlik": round(e_ok, 4), "esik_tekrar": round(e_tk, 4),
                    "n_temiz": int(temiz.sum()), "n_yankili": int(echo.sum()),
                    "n_satir": len(R)},
        "etiket_dagilimi": dict(Counter(etiket)),
        "EGRI_yargic": {k: (None if not np.isfinite(v) else round(v, 4))
                        for k, v in zip(egri_kol, h)},
        "n_karar_kol": dict(zip(egri_kol, n_kol)),
        "capa_k0": round(taban, 4), "plasebo": round(plas, 4),
        "menzil_R": round(float(h[-1] - h[0]), 4),
        "adimlar": [round(float(x), 4) for x in np.diff(h)],
        "null": {"ort": round(n_ort, 4), "sd": round(n_sd, 4), "K": K_NULL,
                 "p95": round(float(np.nanpercentile(nullR, 95)), 4),
                 "frac_null_ge_gozlenen": round(
                     float(np.mean(nullR >= (h[-1] - h[0]))), 4)},
        "MDE_isletim": round(mde, 4), "MDE_tasarim_kappa": round(
            1.645 * np.sqrt(2 * 0.25 / 34) / KAPPA, 4),
        "egim_bootstrap": {"alt_p05": round(e_alt, 4), "ust_p95": round(e_ust, 4),
                           "B_gecerli": int(egimler.size), "n_kume": int(len(ks))},
        "IKINCI_ALET_vekil_egri": vekil_h,
        "vekil_karne": karne,
        "VERDICT": H,
    }
    gec = CIKTI + f".tmp{os.getpid()}"
    json.dump(kayit, open(gec, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(gec, CIKTI)
    print(f"\n  egri (0:5→5:0): {[None if not np.isfinite(v) else round(v,3) for v in h]}")
    print(f"  capa {taban:.3f} · plasebo {plas:.3f} · R {h[-1]-h[0]:+.3f} · "
          f"MDE {mde:.3f} · egim CI[{e_alt:+.4f},{e_ust:+.4f}]")
    print(f"  ★ VERDICT: {H}\n→ {CIKTI}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", choices=("prova", "uret", "coz"), required=True)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=None)
    ap.add_argument("--yigin", type=int, default=17)
    ap.add_argument("--model", default=__DNH_DATA__ + "/.hf_local/hub/"
                                       "models--mistralai--Mistral-7B-v0.3")
    a = ap.parse_args()
    if a.asama == "prova":
        print(json.dumps(provalar(), ensure_ascii=False, indent=1, default=str))
        return 0
    return uret(a) if a.asama == "uret" else coz(a)


if __name__ == "__main__":
    sys.exit(main())
