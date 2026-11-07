#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, random, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
from reading_style_object import payda
import form_count as FS

CIK = f"{ROOT}/results/FILTRE_VERIM_2026-08-29.json"
SEED = 20260829
SAYAC = ("a", "b", "c", "d", "e", "b_cekirdek")

SETLER = [
    dict(ad="hh-rlhf", repo="Anthropic/hh-rlhf", split="train", cikarici="duz"),
    dict(ad="ultrafeedback", repo="HuggingFaceH4/ultrafeedback_binarized",
         split="train_prefs", cikarici="mesaj"),
    dict(ad="tulu3-pref", repo="allenai/llama-3.1-tulu-3-8b-preference-mixture",
         split="train", cikarici="mesaj"),
]


def _son_asistan_duz(s: str) -> str:
    i = s.rfind("\n\nAssistant:")
    return "" if i < 0 else s[i + len("\n\nAssistant:"):].strip()


def _son_asistan_mesaj(m) -> str:
    for x in reversed(m or []):
        if x.get("role") == "assistant":
            return (x.get("content") or "").strip()
    return ""


def cikar(kayit, kip):
    f = _son_asistan_duz if kip == "duz" else _son_asistan_mesaj
    return f(kayit["chosen"]), f(kayit["rejected"])


def yukle(s, n, rng):
    from datasets import load_dataset
    d = load_dataset(s["repo"], split=s["split"])
    N = len(d)
    idx = list(range(N))
    if n and n < N:
        rng.shuffle(idx); idx = sorted(idx[:n])
    C, R, red_bos = [], [], 0
    for i in idx:
        c, r = cikar(d[i], s["cikarici"])
        if not c or not r:
            red_bos += 1; continue
        C.append(c); R.append(r)
    payda(f"fv_yukle_{s['ad']}", n_toplam=N, n_ornek=len(idx), hal_cift=len(C),
          red_bos_taraf=red_bos)
    return C, R, N, red_bos


def say(nlp, metinler):
    A, n_c, n_j = FS.uretim_say(nlp, metinler, n_process=1, batch=256)
    return {k: A[k] for k in SAYAC}, n_c, n_j


def verim_kapisi(nlp, ornek, hedef_n, toplam_n):
    t0 = time.time(); say(nlp, ornek); dt = time.time() - t0
    hiz = len(ornek) / dt
    eta_dk = (2 * hedef_n / hiz) / 60
    eta_tam_dk = (2 * toplam_n / hiz) / 60
    payda("fv_verim_kapisi", n_olcum=len(ornek), hal_metin_sn=round(hiz, 1),
          hal_eta_dk=round(eta_dk, 1), hal_eta_tam_dk=round(eta_tam_dk, 1))
    print(f"{hiz:.0f} {hedef_n} {eta_dk:.1f}"
          f"{toplam_n} {eta_tam_dk:.0f}"
          f"", flush=True)
    return hiz, eta_dk


ESIKLER = [1, 2, 3]
def filtreler(D):
    F = {}
    for t in ESIKLER:
        F[f"a>=+{t}"] = D["a"] >= t
        F[f"a<=-{t}"] = D["a"] <= -t
        F[f"c<=-{t}"] = D["c"] <= -t
        F[f"c>=+{t}"] = D["c"] >= t
        F[f"d<=-{t}"] = D["d"] <= -t
        F[f"a>=+{t} & c<=-{t}"] = (D["a"] >= t) & (D["c"] <= -t)
    F["herhangi-form-farki"] = np.any(np.stack([D[k] != 0 for k in SAYAC]), axis=0)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="set basina örneklem (0=tamami)")
    ap.add_argument("--ornek-cift", type=int, default=20)
    a = ap.parse_args()
    rng = random.Random(SEED)
    nlp = FS._boru()
    OUT, t0 = {}, time.time()
    for s in SETLER:
        print(f"\n=== {s['ad']} ===", flush=True)
        C, R, N, red_bos = yukle(s, a.n, rng)
        verim_kapisi(nlp, C[:300], len(C), N)
        t1 = time.time()
        Ac, nc_c, nj_c = say(nlp, C)
        Ar, nc_r, nj_r = say(nlp, R)
        D = {k: Ac[k].astype(np.int32) - Ar[k].astype(np.int32) for k in SAYAC}
        F = filtreler(D)
        n = len(C)
        satir = {}
        for ad, m in F.items():
            k = int(m.sum())
            p = k / n
            se = (p * (1 - p) / n) ** 0.5
            satir[ad] = dict(gecen=k, n=n, oran=round(p, 5),
                             CI95=[round(max(0, p - 1.96 * se), 5), round(min(1, p + 1.96 * se), 5)],
                             ongorulen_tam_sayi=int(round(p * N)))
        ornekler = {}
        for ad, m in F.items():
            w = np.flatnonzero(m)[: a.ornek_cift]
            ornekler[ad] = [dict(chosen=C[i][:400], rejected=R[i][:400],
                                 delta={k: int(D[k][i]) for k in SAYAC}) for i in w]
        OUT[s["ad"]] = dict(
            repo=s["repo"], split=s["split"], cikarici=s["cikarici"],
            n_toplam=N, n_ornek=n, red_bos_taraf=red_bos,
            sayac_ort_chosen={k: round(float(Ac[k].mean()), 5) for k in SAYAC},
            sayac_ort_rejected={k: round(float(Ar[k].mean()), 5) for k in SAYAC},
            delta_dagilim={k: dict(ort=round(float(D[k].mean()), 5),
                                   sd=round(float(D[k].std()), 5),
                                   sifir_orani=round(float((D[k] == 0).mean()), 5),
                                   p05=int(np.percentile(D[k], 5)),
                                   p50=int(np.percentile(D[k], 50)),
                                   p95=int(np.percentile(D[k], 95)))
                           for k in SAYAC},
            jeton_ort_chosen=round(float(nj_c.mean()), 1),
            jeton_ort_rejected=round(float(nj_r.mean()), 1),
            esik=satir, ornek=ornekler, saniye=round(time.time() - t1, 1))
        print(f"  ★ {s['ad']}: n={n}/{N} · {time.time()-t1:.0f} sn", flush=True)
        for ad in ("a>=+1", "c<=-1", "a>=+1 & c<=-1", "herhangi-form-farki"):
            v = satir[ad]
            print(f"     {ad:22s} {v['gecen']:>6d}/{n} = {v['oran']*100:5.2f}% "
                  f"⇒ tam sette ≈{v['ongorulen_tam_sayi']:,}", flush=True)
    payda("filtre_verim", n_set=len(SETLER), n_esik=len(filtreler({k: np.zeros(1) for k in SAYAC})),
          hal_ornek_cift=a.ornek_cift)
    json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   SINIF="SAYIM (prereg YOK · bar YOK · egitim YOK)",
                   KURATOR_VERDICT=""
                                 "",
                   seed=SEED, n_istenen=a.n, set=OUT,
                   saniye=round(time.time() - t0, 1)),
              open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {CIK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
