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
import package1_kutuplar as KP
import a3_harman_oran as A3
import adim4_harman2 as A4
import prefix_overnight_doz as GD
import prefix_harsh_vekil as HV
import prefix_ladder as MD
import prefix_judge_calibration as KAL

OUT = __DNH_DATA__ + "/onek_korpus/seritA_102"
CIKTI = f"{ROOT}/results/SERITA_HARMAN102_2026-08-12.json"
TEMPLATE = A3.TEMPLATE
MODEL = f"{ROOT}/.hf/hub/models--mistralai--Mistral-7B-Instruct-v0.3"

SEED, ES = 20260812, 8
N_CEKIM = 5
TAKAS_OFSET = 104729
K_NULL, B_BOOT = 200, 200
KAPPA = 0.6779
N_ASGARI_KARAR = 20
Z = 1.645

MAX_ISTEM_JETON = int(os.environ.get("PROJECT_ISTEM_MAXLEN", "1024"))
KARAR_SINIF = ["HARSH", "CALM"]

ADLAR = ("ÖLCÜLEMEZ", "REJIM-KURULAMADI", "ETKILESIM-VAR", "MONOTON-KORUNUR",
         "SEKIL-DAGILIYOR")


def kollar():
    uy, _ = KP.izgara()
    hucre = [(KP.hucre_adi(c), "hucre", c, 0) for c in uy]
    saf = [c for c in uy if len(set(c)) == 1]
    uc_eksen = sorted(c for c in uy if len({KP.EKSEN[i] for i in c}) == 3)[0]
    K = (hucre
         + [("capa_k0", "capa", None, 0)]
         + [(f"plasebo_{KP.KUTUP_AD[c[0]]}", "plasebo", c, 0) for c in saf]
         + [(f"takas_{KP.hucre_adi(c)}", "takas", c, TAKAS_OFSET)
            for c in saf + [uc_eksen]])
    payda("seritA_kollar", n_kol=len(K), n_hucre=len(hucre), n_saf=len(saf),
          n_plasebo=len(saf), n_takas=len(saf) + 1,
          bekle={"n_kol": 116, "n_hucre": 102, "n_saf": 6, "n_takas": 7})
    return K, uc_eksen


def plasebo_onek_kutup(c, tez, durus, cekim, seed):
    k = c[0]
    orn = [MD._karistir_kelime(KP.ORNEK[k][(j + cekim) % KP.N_SHOT])
           for j in range(KP.N_SHOT)]
    np.random.default_rng(seed).shuffle(orn)
    return (KP.AYIRAC.join(orn) + KP.AYIRAC
            + KP.CERCEVE_NOTR.format(T=tez, P=A3.DURUS[durus]))


def _kol_tohumu(ad):
    return int(hashlib.sha256(ad.encode()).hexdigest()[:8], 16) % 9973


def onek_kur(ad, tur, c, ofset, i, ist, cekim):
    th = SEED + ofset + 1000 * _kol_tohumu(ad) + 17 * i + cekim
    tz, du = ist["tez"], ist["durus"]
    if tur == "capa":
        return A3.capa_onek(tz, du)
    if tur == "plasebo":
        return plasebo_onek_kutup(c, tz, du, cekim, th)
    return KP.onek_kur(c, tz, du, cekim, th)


def verdict_seritA(n_karar, kapi_tamam, null_merkez_bozuk, capa_kararsiz,
                 etkilesim_ayrik, sekiller):
    if (not kapi_tamam) or min(n_karar) < N_ASGARI_KARAR:
        return "ÖLCÜLEMEZ"
    if null_merkez_bozuk or capa_kararsiz:
        return "REJIM-KURULAMADI"
    if etkilesim_ayrik:
        return "ETKILESIM-VAR"
    if sum(s in ("MONOTON", "DOYAN") for s in sekiller) >= 6:
        return "MONOTON-KORUNUR"
    return "SEKIL-DAGILIYOR"


def prova():
    M = ["MONOTON"] * 8
    V = [((([50] * 8), False, False, False, False, M), "ÖLCÜLEMEZ"),
         ((([5] * 8), True, False, False, False, M), "ÖLCÜLEMEZ"),
         ((([50] * 8), True, True, False, False, M), "REJIM-KURULAMADI"),
         ((([50] * 8), True, False, True, False, M), "REJIM-KURULAMADI"),
         ((([50] * 8), True, False, False, True, M), "ETKILESIM-VAR"),
         ((([50] * 8), True, False, False, False, M), "MONOTON-KORUNUR"),
         ((([50] * 8), True, False, False, False, ["DÜZENSIZ"] * 8), "SEKIL-DAGILIYOR")]
    ok = 0
    for arg, bek in V:
        g = verdict_seritA(*arg)
        ok += g == bek
        print(f"  prova {bek:18s} → {g}")
    if ok != len(V):
        raise SystemExit("★ PROVA DÜSTÜ — verdict fonksiyonu")
    GD._prova()
    kotu = sum(KP.onek_kur(tuple([0] * r + [1] * (5 - r)), "T", "arti", 0, th)
               != A3.harman(r, "T", "arti", th)
               for r in range(6) for th in (1, 7, 20260812))
    payda("seritA_prova", n_vaka=len(V), n_esdeger=18, red_dusen=len(V) - ok,
          red_esdeger_fark=kotu, bekle={"n_vaka": 7, "n_esdeger": 18})
    if kotu:
        raise SystemExit(f"★ K0-d DÜSTÜ — {kotu} esdegerlik farki")
    if KP.DK.CAPA_HASH != "3756aeaa048ac196":
        raise SystemExit("★ K0-e DÜSTÜ — DOM lafzi degismis")
    print(f"  [K0-d] esdegerlik 18/18 ÖZDES · [K0-e] CAPA_HASH sabit · "
          f"LAFIZ_HASH {KP.LAFIZ_HASH}")


def uret(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gpu_lock import kilitle
    import prefix_generation_pilot as UP
    import glob as _glob

    prova()
    K, uc_eksen = kollar()
    C = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="seritA")
    os.makedirs(f"{OUT}/kol", exist_ok=True)
    ist = A3.istemler()
    bekle_satir = len(ist) * N_CEKIM

    snap = _snapshot(a.model)
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).to(a.dev)
    model.eval()

    n_cekim = N_CEKIM
    if not a.devam:
        t_p = time.time()
        _kol_uret(model, tok, a, K[0], ist, 1, f"{OUT}/kol/_kapi_provasi.jsonl")
        hiz = len(ist) / (time.time() - t_p)
        kestirim_dk = len(K) * bekle_satir / hiz / 60
        print(f"  [KAPI] ölcülen hiz {hiz:.2f} satir/s ⇒ kestirim {kestirim_dk:.0f} dk",
              flush=True)
        if kestirim_dk > 90:
            n_cekim = 3
            bekle_satir = len(ist) * n_cekim
            print(f"  [KAPI] >90 dk ⇒ N_CEKIM 5→3 (prereg §11 tek dali; rotasyon {{0,1,2}})",
                  flush=True)
        os.remove(f"{OUT}/kol/_kapi_provasi.jsonl")

    t0, atlanan = time.time(), 0
    for ki, (ad, tur, c, ofs) in enumerate(K):
        yol = f"{OUT}/kol/{_gv(ad)}.jsonl"
        if a.devam and os.path.exists(yol):
            n = sum(1 for _ in open(yol, encoding="utf-8"))
            if n >= bekle_satir:
                atlanan += 1
                continue
            os.remove(yol)
        _kol_uret(model, tok, a, (ad, tur, c, ofs), ist, n_cekim, yol)
        if (ki + 1) % 10 == 0 or ki == len(K) - 1:
            print(f"  [{ki+1:3d}/{len(K)}] {ad:<24} · {time.time()-t0:.0f}s", flush=True)

    yol, tmp = f"{OUT}/uretim.jsonl", f"{OUT}/uretim.jsonl.tmp{os.getpid()}"
    n_sat = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for ad, _, _, _ in K:
            for l in open(f"{OUT}/kol/{_gv(ad)}.jsonl", encoding="utf-8"):
                fh.write(l)
                n_sat += 1
    os.replace(tmp, yol)
    man = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="results/PREREG_SERITA_HARMAN102_2026-08-12.md", cevre=C,
               model=snap, motor="HF transformers", lafiz_hash=KP.LAFIZ_HASH,
               capa_hash=KP.DK.CAPA_HASH, seed=SEED, n_cekim=n_cekim,
               n_kol=len(K), n_istem=len(ist), n_satir=n_sat, uc_eksen_takas=list(uc_eksen),
               sicaklik=UP.SICAKLIK, top_p=UP.TOP_P, yeni_jeton=A3.YENI_JETON,
               betik_sha={f: hashlib.sha256(open(f"{ROOT}/scripts/{f}", "rb").read())
                          .hexdigest()[:16]
                          for f in ("package1_kutuplar.py", "serita_harman102.py")},
               saniye=round(time.time() - t0, 1), atlanan_kol=atlanan,
               sha256_uretim=hashlib.sha256(open(yol, "rb").read()).hexdigest())
    json.dump(man, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1,
              default=str)
    payda("seritA_uretim", n_satir=n_sat, n_kol=len(K), n_istem=len(ist),
          atlanan=atlanan, bekle={"n_satir": len(K) * bekle_satir, "n_kol": 116})
    print(f"  → {yol} · {n_sat} satir · sha {man['sha256_uretim'][:16]} · "
          f"{man['saniye']}s")
    return 0


def _gv(ad):
    return ad.replace("+", "P").replace("-", "M")


HF_KOKLERI = [f"{ROOT}/.hf/hub", "<storage>/huggingface/hub",
              os.path.expanduser("~/.cache/huggingface/hub")]


def _snapshot(kok, model_adi=None, kokler=None):
    import glob as g

    def _tam(yollar):
        return [s for s in yollar
                if g.glob(f"{s}/config.json")
                and (g.glob(f"{s}/*.safetensors") or g.glob(f"{s}/*.bin"))
                and (g.glob(f"{s}/tokenizer.json") or g.glob(f"{s}/tokenizer.model")
                     or (g.glob(f"{s}/vocab.json") and g.glob(f"{s}/merges.txt")))]

    if model_adi is None:
        ad = _tam(g.glob(f"{kok}/snapshots/*"))
    else:
        hepsi, ad = [], []
        for k in (kokler or HF_KOKLERI):
            hepsi += g.glob(f"{k}/{model_adi}/snapshots/*")
        ad = _tam(hepsi)
        atlanan = [x for x in hepsi if x not in ad]
        payda("snapshot_cok_kok", n_kok=len(kokler or HF_KOKLERI),
              n_aday_yol=len(hepsi), n_tam=len(ad), red_kabuk_atlandi=len(atlanan))
        for x in atlanan:
            print(f"{x}", flush=True)
        kok = model_adi
    if not ad:
        raise SystemExit(f"{kok}")
    return sorted(ad)[-1]


def _kol_uret(model, tok, a, kol, ist, n_cekim, yol, onek_fn=None):
    import torch
    import prefix_generation_pilot as UP
    ad, tur, c, ofs = kol
    tmp = yol + f".tmp{os.getpid()}"
    _ok = onek_fn or onek_kur
    _p0 = [_ok(ad, tur, c, ofs, i, s, _ck)
           for _ck in range(n_cekim) for i, s in enumerate(ist)]
    _n0 = [len(x) for x in tok(_p0, truncation=False)["input_ids"]]
    _tepe = max(_n0); _kirp = sum(1 for x in _n0 if x > MAX_ISTEM_JETON)
    print(f"{ad} {_tepe}"
          f"{MAX_ISTEM_JETON} {_kirp} {len(_n0)}"
          f"", flush=True)
    payda("istem_kirpma", n_istem=len(_n0), hal_tepe_jeton=_tepe,
          hal_max_length=MAX_ISTEM_JETON, red_kirpilan=_kirp)
    if _kirp:
        raise SystemExit(f"{ad} {_kirp} {len(_n0)}"
                         f"{MAX_ISTEM_JETON} {_tepe}")
    with open(tmp, "w", encoding="utf-8") as fh:
        for cekim in range(n_cekim):
            onekler = [_ok(ad, tur, c, ofs, i, s, cekim) for i, s in enumerate(ist)]
            torch.manual_seed(SEED + 7919 * cekim + (ofs or 0))
            for i in range(0, len(onekler), a.yigin):
                par = onekler[i:i + a.yigin]
                enc = tok(par, return_tensors="pt", padding=True, truncation=True,
                          max_length=MAX_ISTEM_JETON)
                enc = {k: v.to(a.dev) for k, v in enc.items()}
                with torch.no_grad():
                    g = model.generate(**enc, do_sample=True, temperature=UP.SICAKLIK,
                                       top_p=UP.TOP_P, max_new_tokens=A3.YENI_JETON,
                                       pad_token_id=tok.pad_token_id)
                for j, s in enumerate(g[:, enc["input_ids"].shape[1]:]):
                    k = i + j
                    fh.write(json.dumps(dict(
                        kol=ad, tur=tur, hucre=list(c) if c else [], cekim=cekim,
                        istem_i=k, debate=ist[k]["debate"], tez=ist[k]["tez"],
                        durus=ist[k]["durus"], onek=par[j],
                        metin=tok.decode(s, skip_special_tokens=True)),
                        ensure_ascii=False) + "\n")
    os.replace(tmp, yol)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", default="uret", choices=("prova", "uret", "coz"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=1)
    ap.add_argument("--yigin", type=int, default=17)
    ap.add_argument("--devam", action="store_true")
    a = ap.parse_args()
    if a.asama == "prova":
        prova()
        kollar()
        sys.exit(0)
    if a.asama == "uret":
        sys.exit(uret(a))
    import serita_resolve
    sys.exit(serita_resolve.coz(a))
