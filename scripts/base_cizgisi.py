#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import gpu_lock as RK
import argparse, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
from reading_style_object import payda
import form_count as FS
import dejenerelik_olcer as DJ
import family_panel as C
import gurultu as K0
import mini_dpo_feasibility as FZ

KOK = __DNH_DATA__ + "/k0_gurultu"
CIK = f"{ROOT}/results/TABAN_CIZGISI_2026-08-30.json"
SAYAC = ("a", "b", "c", "d", "e", "b_cekirdek")


def bir_seed_cpu(t, nlp, npr):
    yol = f"{KOK}/seed{t}/uretim.jsonl"
    R = [json.loads(l) for l in open(yol, encoding="utf-8")]
    metin = [r["metin"] for r in R]
    kol = np.array([r["kol"] for r in R])
    A, n_c, n_j = FS.uretim_say(nlp, metin, n_process=npr, batch=128)
    dej = np.array([DJ.bayrakla(x)["DEJENERE"] for x in metin], dtype=bool)
    out = {}
    for k in sorted(set(kol.tolist())):
        m = kol == k
        j = np.maximum(n_j[m].astype(float), 1.0)
        out[k] = dict(n=int(m.sum()),
                      jeton_ort=round(float(n_j[m].mean()), 2),
                      dej_orani=round(float(dej[m].mean()), 4),
                      ham={c: round(float(A[c][m].mean()), 5) for c in SAYAC},
                      oran={c: round(float((A[c][m] / j).mean()), 6) for c in SAYAC})
    payda(f"taban_cpu_t{t}", n_satir=len(R), n_kol=len(out),
          hal_dej_orani=round(float(dej.mean()), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-process", type=int, default=48)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--tohumlar", default="0,1,2,3,4,5")
    ap.add_argument("--gpu", action="store_true", help="U okumasini da yap (GPU)")
    a = ap.parse_args()
    T = [int(x) for x in a.tohumlar.split(",")]
    t0 = time.time()
    rej, sag = FZ.prod_kapisi(RK.fiziksel_bekle("cuda:0"), "taban_cizgisi")
    nlp = FS._boru()

    print("\n=== CPU · kol-basina form sayaclari (m=%d seed) ===" % len(T), flush=True)
    CPU = {}
    for t in T:
        CPU[t] = bir_seed_cpu(t, nlp, a.n_process)
        print(f"  ★ seed {t} sayildi · {time.time()-t0:.0f} sn", flush=True)

    kollar = sorted(CPU[T[0]].keys())
    ozet = {}
    for k in kollar:
        ozet[k] = {}
        for cins in ("ham", "oran"):
            ozet[k][cins] = {}
            for c in SAYAC:
                v = np.array([CPU[t][k][cins][c] for t in T])
                ozet[k][cins][c] = dict(ort=round(float(v.mean()), 6),
                                        sd=round(float(v.std(ddof=1)), 6),
                                        m=len(T))
        dj = np.array([CPU[t][k]["dej_orani"] for t in T])
        jt = np.array([CPU[t][k]["jeton_ort"] for t in T])
        ozet[k]["dej"] = dict(ort=round(float(dj.mean()), 4), sd=round(float(dj.std(ddof=1)), 4))
        ozet[k]["jeton"] = dict(ort=round(float(jt.mean()), 2), sd=round(float(jt.std(ddof=1)), 2))

    U = None
    if a.gpu:
        print("\n=== GPU · zemin okumasi (U, üc eksen, seed basina) ===", flush=True)
        bos, kul = FZ.gpu_bos_mib(RK.fiziksel_bekle("cuda:0")); kapi = FZ.kapi_hesapla(bos, rej)
        print(f"[KAPI] rejim {rej} ⇒ bos {bos} − pay = {kapi} MiB", flush=True)
        g = FZ.Gozcu(kul + kapi); g.start()
        U = {}
        for t in T:
            K0.kur(t, t)
            z, B, UN = C._zemin_oku("k0-null", "base", a.dev)
            U[t] = {e: float(z["olcum"][e]["U"]) for e in C.EKSENLER}
            print(f"  ★ seed {t}: " + " · ".join(f"{e} {U[t][e]:+.6f}" for e in C.EKSENLER),
                  flush=True)
        g.dur = True; g.join(timeout=3)
        payda("taban_U", n_seed=len(U), red_gozcu_ihlal=int(g.ihlal))

    O = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="TABAN-CIZGISI (KESIF · bar YOK · ad YOK · prediction YOK)",
             ONCUL_DUZELTMESI=""
                              "",
             kaynak=KOK, tohumlar=T, m=len(T),
             model="models--allenai--Llama-3.1-Tulu-3-8B-SFT",
             TUTULMUS_SUBSTRAT="panelin 34 istemi egitim kollarina girmedi "
                               "(kesisim 0/3 kol; güc serhi W-717)",
             kol_ozet=ozet, seed_basina=CPU, U=U,
             saniye=round(time.time() - t0, 1))
    json.dump(O, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("taban_cizgisi", n_seed=len(T), n_kol=len(kollar), hal_m=len(T))
    print(f"\n→ {CIK} · {(time.time()-t0)/60:.1f} dk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
