#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
from reading_style_object import payda
import form_count as FS
import cerceve as CC
import three_arm as UK

ANAHTAR = ("a", "b", "c", "d", "e", "b_cekirdek") + CC.ALT + CC.TESHIS
CIK = f"{ROOT}/results/V3_KURATOR_VERIMI_2026-08-31.json"
ORNEK_DISK = __DNH_DATA__ + "/V3_ORNEKLER_2026-08-31.json"


def say2(nlp, metinler, n_process=1, batch=128):
    N = len(metinler)
    A = {k: np.zeros(N, dtype=np.int32) for k in ANAHTAR}
    n_c = np.zeros(N, dtype=np.int32); n_j = np.zeros(N, dtype=np.int32)
    for i, doc in enumerate(nlp.pipe(metinler, batch_size=batch, n_process=n_process)):
        n_j[i] = sum(1 for x in doc if not x.is_space)
        for s in doc.sents:
            if not any((not x.is_punct and not x.is_space) for x in s):
                continue
            n_c[i] += 1
            f = FS.sayaclar(s); g = CC.alt_siniflar(s)
            for k in ANAHTAR:
                A[k][i] += (f[k] if k in f else g[k])
    return A, n_c, n_j


def kapi_esdegerlik(nlp, ornek, npr):
    A, n_c, n_j = say2(nlp, ornek, npr)
    A1, c1, j1 = FS.uretim_say(nlp, ornek, n_process=npr, batch=128)
    A2, c2 = CC.say(nlp, ornek, n_process=npr, batch=128)
    fark = {k: int(np.abs(A[k] - A1[k]).sum()) for k in ("a", "b", "c", "d", "e", "b_cekirdek")}
    fark |= {k: int(np.abs(A[k] - A2[k]).sum()) for k in CC.ALT + CC.TESHIS}
    fark["n_cumle"] = int(np.abs(n_c - c1).sum()) + int(np.abs(n_c - c2).sum())
    fark["n_jeton"] = int(np.abs(n_j - j1).sum())
    top = sum(fark.values())
    payda("v3_esdegerlik", n_metin=len(ornek), n_anahtar=len(fark), red_sapan=top)
    print(f"  ★ ESDEGERLIK KAPISI · {len(ornek)} metin · toplam sapma **{top}** · {fark}",
          flush=True)
    if top:
        raise SystemExit("")
    return fark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-process", type=int, default=48)
    ap.add_argument("--n", type=int, default=0, help="0 = TAM SET")
    ap.add_argument("--kapi-ornek", type=int, default=1500)
    ap.add_argument("--ornek", type=int, default=20)
    ap.add_argument("--boy-bandi", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    from datasets import load_dataset
    import filtre_verim as FV
    ds = load_dataset(UK.REPO, split="train")
    N = len(ds) if not a.n else min(a.n, len(ds))
    ds = ds.select(range(N))
    C = [FV._son_asistan_mesaj(x) for x in ds["chosen"]]
    R = [FV._son_asistan_mesaj(x) for x in ds["rejected"]]
    KAYNAK, KIMLIK = ds["source"], ds["id"]
    gecerli = np.array([bool(c) and bool(r) for c, r in zip(C, R)])
    payda("v3_cikar", n_toplam=len(ds), hal_gecerli=int(gecerli.sum()),
          red_bos_taraf=int((~gecerli).sum()))
    KL = UK.klise_yukle()
    sc, sr = {}, {}
    Ch = [UK.hijyen(t, KL, sc)[0] for t in C]
    Rh = [UK.hijyen(t, KL, sr)[0] for t in R]
    nlp = FS._boru(a.n_process)
    kapi_esdegerlik(nlp, Ch[:a.kapi_ornek], a.n_process)
    print(f"  ★ tam tarama basliyor · {2*len(Ch):,} metin · ETA ≈ 24 dk "
          f"(uc_kol ölcümü 1 428 sn) ⇒ esik 48 dk ⇒ EYLEM: kesme, DÖNÜS'e serh", flush=True)
    Ac, ncc, njc = say2(nlp, Ch, a.n_process)
    print(f"  ○ chosen bitti · {time.time()-t0:.0f} sn", flush=True)
    Ar, ncr, njr = say2(nlp, Rh, a.n_process)
    print(f"  ○ rejected bitti · {time.time()-t0:.0f} sn", flush=True)
    d = {k: Ac[k].astype(int) - Ar[k].astype(int) for k in ANAHTAR}
    dj = njc.astype(int) - njr.astype(int)
    boy_ok = np.abs(dj) <= a.boy_bandi
    KOL = {
        "EMIR_ARTI_v3": (d["a"] >= 1) & (d["c_rica"] <= -1) & gecerli,
        "EMIR_EKSI_v3": (d["a"] <= -1) & (d["c_rica"] >= 1) & gecerli,
        "EMIR_ARTI_v1_esik": (d["a"] >= 1) & (d["c"] <= -1) & gecerli,
        "EMIR_EKSI_v1_esik": (d["a"] <= -1) & (d["c"] >= 1) & gecerli,
    }
    O, ORN = {}, {}
    rng = np.random.default_rng(20260831)
    for ad, m in KOL.items():
        m2 = m & boy_ok
        O[ad] = dict(n=int(m.sum()), oran=round(float(m.mean()), 6),
                     n_boy_bandi=int(m2.sum()), oran_boy=round(float(m2.mean()), 6),
                     medyan_dj=float(np.median(dj[m])) if m.sum() else None,
                     medyan_da=float(np.median(d["a"][m])) if m.sum() else None,
                     medyan_dcrica=float(np.median(d["c_rica"][m])) if m.sum() else None)
        ix = np.flatnonzero(m2 if m2.sum() >= a.ornek else m)
        pick = rng.choice(ix, min(a.ornek, len(ix)), replace=False) if len(ix) else []
        ORN[ad] = [dict(id=KIMLIK[int(i)], source=KAYNAK[int(i)],
                        d_a=int(d["a"][int(i)]), d_c=int(d["c"][int(i)]),
                        d_c_rica=int(d["c_rica"][int(i)]),
                        d_c_artik=int(d["c_artik"][int(i)]),
                        d_jeton=int(dj[int(i)]), boy_bandinda=bool(boy_ok[int(i)]),
                        chosen=C[int(i)], rejected=R[int(i)]) for i in pick]
        print(f"  {ad:20s} n={O[ad]['n']:6d} ({O[ad]['oran']*100:.3f}%) · "
              f"boy bandinda {O[ad]['n_boy_bandi']:6d} ({O[ad]['oran_boy']*100:.3f}%)",
              flush=True)
    payda("v3_verim", n_cift=int(N), n_kol=len(KOL),
          n_ornek=sum(len(v) for v in ORN.values()))
    r = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="SAYIM — bar YOK · ad YOK · prediction YOK · INSA YOK",
             repo=UK.REPO, n_toplam=int(N), boy_bandi=a.boy_bandi,
             klise_sha256=UK.sha(UK.KLISE) if hasattr(UK, "KLISE") else None,
             verim=O, saniye=round(time.time() - t0, 1),
             ORNEK_BEYANI=(""
                           f"{ORNEK_DISK}"),
             ornek_kimlik={k: [{kk: vv for kk, vv in x.items()
                                if kk not in ("chosen", "rejected")} for x in v]
                           for k, v in ORN.items()})
    json.dump(r, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(ORN, open(ORNEK_DISK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {CIK}  ({r['saniye']/60:.1f} dk) · örnek metinler → {ORNEK_DISK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
