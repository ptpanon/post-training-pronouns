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

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import package1_kutuplar as KP
import serita_harman102 as SA
import a3_harman_oran as A3

OUT = __DNH_DATA__ + "/onek_korpus/s3_esitlik"
SERITA = __DNH_DATA__ + "/onek_korpus/seritA_102/kol"
CIKTI = f"{ROOT}/results/S3_ESITLIK_URETIM_2026-08-13.json"
N_CEKIM = 3
YIGIN = 17

SAF = [(0, 0, 0, 0), (1, 1, 1, 1), (4, 4, 4, 4), (5, 5, 5, 5)]
TIE = [(0, 0, 4, 4), (0, 0, 5, 5), (1, 1, 4, 4), (1, 1, 5, 5)]
TRI = [tuple(sorted(t + (d,))) for t in TIE for d in (2, 3)]


def kollar():
    K = ([(KP.hucre_adi(c), "saf4", c, 0) for c in SAF]
         + [(KP.hucre_adi(c), "tie22", c, 0) for c in TIE]
         + [(KP.hucre_adi(c), "tie221", c, 0) for c in TRI])
    ad = [k[0] for k in K]
    payda("s3_kollar", n_kol=len(K), n_benzersiz=len(set(ad)),
          n_saf=len(SAF), n_tie=len(TIE), n_tri=len(TRI),
          red_cakisma=len(ad) - len(set(ad)),
          bekle={"n_kol": 16, "n_benzersiz": 16, "n_saf": 4, "n_tie": 4, "n_tri": 8})
    if len(set(ad)) != 16:
        raise SystemExit("★ K0-b: hücre cakismasi — kosu BASLAMAZ")
    return K


def _ornek_say(c, cekim=0, seed=1):
    p = KP.onek_kur(c, "X", "arti", cekim, seed)
    bekl = [KP.ORNEK[k][(j + cekim) % KP.N_SHOT] for j, k in enumerate(sorted(c))]
    return sum(1 for o in bekl if o in p), len(bekl)


def k0a_dort_yuva():
    n4 = [_ornek_say(c) for c in SAF + TIE]
    n5 = _ornek_say(TRI[0])
    payda("s3_k0a_yuva", n_kol4=len(n4), n_ornek_min=min(a for a, _ in n4),
          n_ornek_maks=max(a for a, _ in n4), n_beklenen_4=min(b for _, b in n4),
          n_ornek_5luk=n5[0], n_beklenen_5=n5[1],
          bekle={"n_kol4": 8, "n_ornek_min": 4, "n_ornek_maks": 4,
                 "n_beklenen_4": 4, "n_ornek_5luk": 5, "n_beklenen_5": 5})
    if min(a for a, _ in n4) != 4 or max(a for a, _ in n4) != 4 or n5 != (5, 5):
        raise SystemExit("")
    return dict(n_ornek_4luk=4, n_ornek_5luk=5)


def k0c_selma():
    DK = KP.DK
    g = json.dumps(dict(dom=list(DK.ORNEK_DOM), sub=list(DK.ORNEK_SUB)),
                   ensure_ascii=False, sort_keys=True)
    payda("s3_k0c_selma", n_dom=len(DK.ORNEK_DOM), n_sub=len(DK.ORNEK_SUB),
          bekle={"n_dom": 2, "n_sub": 2})
    return hashlib.sha256(g.encode()).hexdigest()[:32]


def esdegerlik_seritA(K):
    o = dict(n_cakisan=0, n_karsilastirilan=0, n_ozdes_onek=0, n_ozdes_metin=0,
             red_dosya_yok=0, red_satir_eksik=0)
    for ad, _, _, _ in K:
        sa = f"{SERITA}/{SA._gv(ad)}.jsonl"
        bz = f"{OUT}/kol/{SA._gv(ad)}.jsonl"
        if not os.path.exists(sa):
            continue
        o["n_cakisan"] += 1
        if not os.path.exists(bz):
            o["red_dosya_yok"] += 1
            continue
        A = [json.loads(l) for l in open(sa, encoding="utf-8")]
        B = [json.loads(l) for l in open(bz, encoding="utf-8")]
        ha = {(r["cekim"], r["istem_i"]): r for r in A if r["cekim"] < N_CEKIM}
        for r in B:
            k = (r["cekim"], r["istem_i"])
            if k not in ha:
                o["red_satir_eksik"] += 1
                continue
            o["n_karsilastirilan"] += 1
            o["n_ozdes_onek"] += int(ha[k]["onek"] == r["onek"])
            o["n_ozdes_metin"] += int(ha[k]["metin"] == r["metin"])
    payda("s3_esdegerlik", **o)
    return o


def main(a):
    global OUT, SERITA, CIKTI
    OUT, SERITA = getattr(a, 'cikis', OUT), getattr(a, 'serita', SERITA)
    CIKTI = getattr(a, 'manifest', CIKTI)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gpu_lock import kilitle
    import prefix_generation_pilot as UP

    t0 = time.time()
    K = kollar()
    yuva = k0a_dort_yuva()
    selma = k0c_selma()
    ist = A3.istemler()
    bekle_satir = len(ist) * N_CEKIM
    C = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="s3_esitlik")
    os.makedirs(f"{OUT}/kol", exist_ok=True)

    snap = SA._snapshot(a.model)
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).to(a.dev)
    model.eval()

    atlanan, t_ilk = 0, None
    for ki, kol in enumerate(K):
        yol = f"{OUT}/kol/{SA._gv(kol[0])}.jsonl"
        if os.path.exists(yol):
            n = sum(1 for _ in open(yol, encoding="utf-8"))
            if n >= bekle_satir:
                atlanan += 1
                continue
            os.remove(yol)
        tk = time.time()
        SA._kol_uret(model, tok, argparse.Namespace(dev=a.dev, yigin=YIGIN),
                     kol, ist, N_CEKIM, yol)
        if t_ilk is None:
            t_ilk = time.time() - tk
            print(f"  [W-71] ilk kol {t_ilk:.1f} sn (yükleme HARIC, ölcüldü) ⇒ "
                  f"ETA {t_ilk*(len(K)-atlanan)/60:.1f} dk", flush=True)
        print(f"  [{ki+1:2d}/{len(K)}] {kol[0]:<24} {kol[1]:<6} · "
              f"{time.time()-t0:.0f}s", flush=True)

    yol, tmp = f"{OUT}/uretim.jsonl", f"{OUT}/uretim.jsonl.tmp{os.getpid()}"
    n_sat = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for ad, _, _, _ in K:
            for l in open(f"{OUT}/kol/{SA._gv(ad)}.jsonl", encoding="utf-8"):
                fh.write(l)
                n_sat += 1
    os.replace(tmp, yol)

    esd = esdegerlik_seritA(K)
    man = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="results/PREREG_K4A_ESITLIK_BOWNER_2026-08-13.md @ f569a4c",
               sinif="MALZEME — yargi YOK, verdict YOK", cevre=C, model=snap,
               motor="HF transformers", lafiz_hash=KP.LAFIZ_HASH, selma_sha=selma,
               k0a_yuva=yuva, seed=SA.SEED, n_cekim=N_CEKIM, yigin=YIGIN,
               n_kol=len(K), n_istem=len(ist), n_satir=n_sat, atlanan_kol=atlanan,
               sicaklik=UP.SICAKLIK, top_p=UP.TOP_P, yeni_jeton=A3.YENI_JETON,
               esdegerlik_seritA=esd, ilk_kol_sn=round(t_ilk or 0, 1),
               betik_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
               saniye=round(time.time() - t0, 1),
               sha256_uretim=hashlib.sha256(open(yol, "rb").read()).hexdigest())
    json.dump(man, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1,
              default=str)
    json.dump(man, open(CIKTI, "w"), ensure_ascii=False, indent=1, default=str)
    payda("s3_uretim", n_satir=n_sat, n_kol=len(K), n_istem=len(ist), atlanan=atlanan,
          bekle={"n_satir": 16 * bekle_satir, "n_kol": 16, "n_istem": 34})
    print(f"  → {yol} · {n_sat} satir · sha {man['sha256_uretim'][:16]} · "
          f"{man['saniye']}s", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SA.MODEL)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=1)
    ap.add_argument("--yalniz-kapi", action="store_true")
    ap.add_argument("--cikis", default=OUT)
    ap.add_argument("--serita", default=SERITA)
    ap.add_argument("--manifest", default=CIKTI)
    a = ap.parse_args()
    if a.yalniz_kapi:
        kollar()
        k0a_dort_yuva()
        print("")
        sys.exit(0)
    sys.exit(main(a))
