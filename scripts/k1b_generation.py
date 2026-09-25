#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, json, os, sys, time

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import package1_kutuplar as KP
import serita_harman102 as SA
import a3_harman_oran as A3
import s3_esitlik_generation as S3U

OUT = __DNH_DATA__ + "/k1b_rakip_uzunluk"
CIKTI = f"{ROOT}/results/K1B_URETIM_2026-08-13.json"
N_CEKIM, YIGIN = 12, 17

X1  = [(4,), (5,), (2,), (3,)]
X4  = [(4, 4, 4, 4), (5, 5, 5, 5), (2, 2, 2, 2), (3, 3, 3, 3)]
KAR = [(2, 2, 4, 4), (3, 3, 4, 4), (2, 2, 5, 5), (3, 3, 5, 5)]

ESKI = {}
_S3 = __DNH_DATA__ + "/onek_korpus/s3_esitlik/kol"
_A1 = __DNH_DATA__ + "/adim1_asama1/kol"


TON_X1 = [(0,), (1,)]
TON_X4 = [(0, 0, 0, 0), (1, 1, 1, 1)]


def kollar(ton=False):
    K = ([(KP.hucre_adi(c), "x1", c, 0) for c in X1]
         + [(KP.hucre_adi(c), "x4", c, 0) for c in X4]
         + [(KP.hucre_adi(c), "kar22", c, 0) for c in KAR])
    if ton:
        K += ([(KP.hucre_adi(c), "x1", c, 0) for c in TON_X1]
              + [(KP.hucre_adi(c), "x4", c, 0) for c in TON_X4])
    bek = 16 if ton else 12
    ad = [k[0] for k in K]
    payda("k1b_kollar", n_kol=len(K), n_benzersiz=len(set(ad)),
          n_x1=len(X1) + (len(TON_X1) if ton else 0),
          n_x4=len(X4) + (len(TON_X4) if ton else 0), n_kar=len(KAR),
          red_cakisma=len(ad) - len(set(ad)),
          bekle={"n_kol": bek, "n_benzersiz": bek,
                 "n_x1": 6 if ton else 4, "n_x4": 6 if ton else 4, "n_kar": 4})
    if len(set(ad)) != bek:
        raise SystemExit("★ K0-b: hücre cakismasi — kosu BASLAMAZ")
    return K


def k0a_yuva(K):
    bek = {"x1": 1, "x4": 4, "kar22": 4}
    say = {t: sorted({S3U._ornek_say(c)[0] for _, tt, c, _ in K if tt == t})
           for t in bek}
    red = sum(1 for t in bek if say[t] != [bek[t]])
    payda("k1b_k0a", n_aile=len(bek), n_kol=len(K), red_yuva_sapan=red,
          bekle={"n_aile": 3, "n_kol": 12})
    if red:
        raise SystemExit(f"★ K0-a: yuva provasi DÜSTÜ — {say}")
    return {t: bek[t] for t in bek}


def k0d_ozdeslik(K):
    import glob
    esk = {}
    for kok in (_S3, _A1):
        for p in glob.glob(f"{kok}/*.jsonl"):
            esk.setdefault(os.path.basename(p)[:-6], p)
    n_den = n_es = n_yok = red = 0
    ayrinti = {}
    for ad, tur, c, _ in K:
        gv = SA._gv(ad)
        yeni = f"{OUT}/kol/{gv}.jsonl"
        if gv not in esk or not os.path.exists(yeni):
            n_yok += 1
            continue
        n_den += 1
        A = [json.loads(l) for l in open(esk[gv], encoding="utf-8")]
        B = [json.loads(l) for l in open(yeni, encoding="utf-8")]
        a3 = [(r["cekim"], r["istem_i"], r["metin"]) for r in A if r["cekim"] < 3]
        b3 = [(r["cekim"], r["istem_i"], r["metin"]) for r in B if r["cekim"] < 3]
        ok = sorted(a3) == sorted(b3)
        ayrinti[gv] = dict(kaynak=esk[gv], n_eski=len(a3), n_yeni=len(b3), ozdes=ok)
        n_es += int(ok); red += int(not ok)
    payda("k1b_k0d", n_kol=len(K), n_denetlenen=n_den, red_ozdes_degil=red,
          bekle={"n_kol": 12})
    print(f"  [K0-d] denetlenen {n_den} · özdes {n_es} · eski karsiligi yok {n_yok} "
          f"· SAPAN {red}", flush=True)
    return dict(n_denetlenen=n_den, n_ozdes=n_es, n_karsiligi_yok=n_yok,
                red_ozdes_degil=red, ayrinti=ayrinti)


def main(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gpu_lock import kilitle

    t0 = time.time()
    K = kollar()
    yuva = k0a_yuva(K)
    selma = S3U.k0c_selma()
    ist = A3.istemler()
    bekle_satir = len(ist) * N_CEKIM
    C = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="k1b")
    os.makedirs(f"{OUT}/kol", exist_ok=True)

    t_yuk = time.time()
    snap = SA._snapshot(a.model)
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).to(a.dev)
    model.eval()
    yukleme_sn = time.time() - t_yuk
    print(f"  [W-72] yükleme {yukleme_sn:.1f} sn — ÖLCÜLDÜ, varsayilmadi", flush=True)

    atlanan, t_ilk = 0, None
    for ki, kol in enumerate(K):
        yol = f"{OUT}/kol/{SA._gv(kol[0])}.jsonl"
        if os.path.exists(yol):
            if sum(1 for _ in open(yol, encoding="utf-8")) >= bekle_satir:
                atlanan += 1
                continue
            os.remove(yol)
        tk = time.time()
        SA._kol_uret(model, tok, argparse.Namespace(dev=a.dev, yigin=YIGIN),
                     kol, ist, N_CEKIM, yol)
        if t_ilk is None:
            t_ilk = time.time() - tk
            print(f"  [W-71] ilk kol {t_ilk:.1f} sn (yükleme HARIC) ⇒ "
                  f"ETA {t_ilk*(len(K)-atlanan)/60:.1f} dk", flush=True)
        print(f"  [{ki+1}/{len(K)}] {kol[0]:<22} {kol[1]:<6} · {time.time()-t0:.0f}s",
              flush=True)

    k0d = k0d_ozdeslik(K)

    yol, tmp = f"{OUT}/uretim.jsonl", f"{OUT}/uretim.jsonl.tmp{os.getpid()}"
    n_sat, kol_say = 0, {}
    with open(tmp, "w", encoding="utf-8") as fh:
        for ad, tur, c, _ in K:
            p = f"{OUT}/kol/{SA._gv(ad)}.jsonl"
            n = 0
            for ln in open(p, encoding="utf-8"):
                fh.write(ln)
                n += 1
            kol_say[ad] = n
            n_sat += n
    os.replace(tmp, yol)
    sha = hashlib.sha256(open(yol, "rb").read()).hexdigest()

    payda("k1b_uretim", n_kol=len(K), n_satir=n_sat,
          red_kol_eksik=sum(1 for v in kol_say.values() if v < bekle_satir),
          bekle={"n_kol": 12, "n_satir": 12 * bekle_satir})
    print(f"  [SONUC] idempotent atlanan kol (sifir mesrudur): {atlanan}", flush=True)

    ort = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="results/PREREG_K1B_RAKIP_UZUNLUK_2026-08-13.md @ c0ad1d0",
               SINIF="MALZEME — yargi YOK, verdict YOK",
               cevre=C, model=snap, motor="HF transformers",
               lafiz_hash=KP.LAFIZ_HASH, selma_sha=selma, k0a_yuva=yuva, k0d=k0d,
               n_kol=len(K), n_istem=len(ist), n_cekim=N_CEKIM, yigin=YIGIN,
               kol_satir=kol_say, n_satir=n_sat, atlanan_kol=atlanan,
               yukleme_sn=round(yukleme_sn, 1),
               ilk_kol_sn=round(t_ilk, 1) if t_ilk else None,
               saniye=round(time.time() - t0, 1), sha256_uretim=sha,
               betik_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16])
    json.dump(ort, open(CIKTI, "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in ort.items() if k not in ("cevre", "k0d")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=1)
    ap.add_argument("--model",
                    default=f"{ROOT}/.hf/hub/models--mistralai--Mistral-7B-Instruct-v0.3")
    ap.add_argument("--yalniz-prova", action="store_true")
    a = ap.parse_args()
    if a.yalniz_prova:
        K = kollar(); k0a_yuva(K)
        print("★ K0-a/K0-b gecti — GPU'ya DOKUNULMADI.")
        sys.exit(0)
    sys.exit(main(a))
