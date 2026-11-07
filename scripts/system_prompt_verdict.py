#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, io, json, os, sys, time
import numpy as np
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
import addressee_run as MK
import form_count as FS
import templated_base_leg_kirlilik as STK
from capability_verdict import d4_metin
from reading_style_object import payda

CIK_KOK = os.environ.get("MSIS_KOK", __DNH_DATA__ + "/c1_panel_msistem")
CIK = os.environ.get("MSIS_CIK", f"{KOK}/results/system_prompt_scorecard_2026-09-10.json")
KOLLAR = ("satirli", "satirsiz")
BAR_AILE = 12
D_SHELF = 0.60
NB = int(os.environ.get("MSIS_NB", 1000))
NPERM = int(os.environ.get("MSIS_NPERM", 200))


def distinct4(metinler):
    top = set(); n = 0
    for t in metinler:
        w = t.split()
        for i in range(len(w) - 3):
            top.add(tuple(w[i:i + 4])); n += 1
    return (len(top) / n) if n else 0.0


def oku(kol, aile, zemin):
    y = f"{CIK_KOK}/{kol}/{aile}/{zemin}/uretim.jsonl"
    if not os.path.exists(y):
        return None
    return [json.loads(l) for l in io.open(y, encoding="utf-8")]


def _anahtar(r):
    return (r.get("kol"), r.get("cekim"), r.get("istem_i"))


def dM1(nlp, R0, R1, sira=None):
    k0 = {_anahtar(r): j for j, r in enumerate(R0)}
    k1 = {_anahtar(r): j for j, r in enumerate(R1)}
    ortak = sira if sira is not None else [k for k in k1 if k in k0]
    ortak = [k for k in ortak if k in k0 and k in k1]
    if not ortak:
        return None
    V0 = MK.satir_bilesenleri(nlp, [R0[k0[k]] for k in ortak])
    V1 = MK.satir_bilesenleri(nlp, [R1[k1[k]] for k in ortak])
    ist = np.array([k[2] for k in ortak])
    return V0, V1, ist, ortak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kuru", action="store_true", help="ÜRETMEZ; kapsami basar")
    ap.add_argument("--d4-olcu", choices=("havuz", "metin"), default="havuz",
                    help=""
                         "")
    a = ap.parse_args()
    _disk = sorted({d for k in KOLLAR
                    for d in (os.listdir(f"{CIK_KOK}/{k}") if os.path.isdir(f"{CIK_KOK}/{k}") else [])})
    try:
        import system_prompt_arm as MSK
        aileler = sorted({h["ad"] for h in MSK.hucreler()})
        _cift = MSK.CIFT_CARD
    except Exception as _e:
        print(f"  ★★ kanonik hücre listesi cözülemedi ({type(_e).__name__}) ⇒ "
              f"evren DISK ∪ cift-card ikilisi", flush=True)
        _cift = {"Llama-3.1-70B", "Qwen2.5-72B"}
        aileler = sorted(set(_disk) | _cift)
    _bekleyen = [x for x in aileler if x not in _disk]
    print(f"{len(KOLLAR)} {len(aileler)}"
          f"{len(_disk)} {len(_bekleyen)} {BAR_AILE}"
          f"{BAR_AILE}", flush=True)
    for _b in _bekleyen:
        print(f"{_b}"
              f"{' (cift card — faz 2)' if _b in _cift else ''}", flush=True)
    if a.kuru:
        _d4f = distinct4 if a.d4_olcu == "havuz" else d4_metin
        _gecen, _dusen = [], []
        for x in aileler:
            var = {f"{k}/{z}": oku(k, x, z) for k in KOLLAR for z in ("base", "instruct")}
            if any(v is None for v in var.values()):
                print(f"    {x}: BACAK-EKSIK {[n for n, v in var.items() if v is None]}")
                _dusen.append(x); continue
            d4 = {n: _d4f([r["metin"] for r in v]) for n, v in var.items()}
            dus = [n for n, v in d4.items() if v < D_SHELF]
            (_dusen if dus else _gecen).append(x)
            print(f"    {x:<17} min(d4)={min(d4.values()):.3f} · düsen={len(dus)} "
                  f"{'⇒ D-SHELF-DÜSTÜ ' + str(dus) if dus else '⇒ GECER'}")
        print(f"{len(aileler)} {a.d4_olcu}"
              f"{D_SHELF} {len(_gecen)} {len(_dusen)}"
              f"{BAR_AILE}", flush=True)
        print("  [KURU] ★ HICBIR VERDICT YAZILMADI")
        return 0
    nlp = FS._boru(); t0 = time.time()
    rng = np.random.default_rng(20260910)
    OUT, atlanan = {}, []
    for x in aileler:
        if x in _bekleyen:
            OUT[x] = dict(hal="BEKLIYOR-CIFT-CARD" if x in _cift else "BEKLIYOR")
            continue
        R = {(k, z): oku(k, x, z) for k in KOLLAR for z in ("base", "instruct")}
        if any(v is None for v in R.values()):
            eksik = [f"{k}/{z}" for (k, z), v in R.items() if v is None]
            OUT[x] = dict(hal="BACAK-EKSIK", eksik=eksik); atlanan.append(x); continue
        _d4f = distinct4 if a.d4_olcu == "havuz" else d4_metin
        d4 = {f"{k}/{z}": _d4f([r["metin"] for r in R[(k, z)]])
              for k in KOLLAR for z in ("base", "instruct")}
        dus = [n for n, v in d4.items() if v < D_SHELF]
        if dus:
            OUT[x] = dict(hal="D-SHELF-DÜSTÜ", distinct4=d4, dusen=dus)
            atlanan.append(x); continue
        _o = None
        for k in KOLLAR:
            kk = set(_anahtar(r) for r in R[(k, "base")]) & set(_anahtar(r) for r in R[(k, "instruct")])
            _o = kk if _o is None else (_o & kk)
        _o = sorted(_o) if _o else []
        if len(_o) < 2:
            OUT[x] = dict(hal="ESLESME-YOK", n_ortak=len(_o)); atlanan.append(x); continue
        P = {k: dM1(nlp, R[(k, "base")], R[(k, "instruct")], sira=_o) for k in KOLLAR}
        if any(v is None for v in P.values()):
            OUT[x] = dict(hal="ESLESME-YOK"); atlanan.append(x); continue
        if P["satirli"][3] != P["satirsiz"][3]:
            OUT[x] = dict(hal="HIZALAMA-DÜSTÜ"); atlanan.append(x); continue
        d_hat, ist = {}, None
        for k in KOLLAR:
            V0, V1, ist, _ = P[k]
            d_hat[k] = MK.olc_toplam(V1)["M1"] - MK.olc_toplam(V0)["M1"]
        fark = d_hat["satirli"] - d_hat["satirsiz"]
        ad = np.unique(ist); B = []
        for _ in range(NB):
            sel = rng.choice(ad, len(ad), replace=True)
            ix = np.concatenate([np.where(ist == u)[0] for u in sel])
            d = {}
            for k in KOLLAR:
                V0, V1, _, _ = P[k]
                d[k] = MK.olc_toplam(V1[ix])["M1"] - MK.olc_toplam(V0[ix])["M1"]
            B.append(d["satirli"] - d["satirsiz"])
        b_ = np.array(B, dtype=float); b_ = b_[~np.isnan(b_)]
        ci = [float(np.percentile(b_, 2.5)), float(np.percentile(b_, 97.5))]
        N = []
        for _ in range(NPERM):
            sw = np.zeros(len(ist), dtype=bool)
            for u in ad:
                m = ist == u
                sw[m] = rng.random() < 0.5
            A0 = np.where(sw[:, None], P["satirsiz"][0], P["satirli"][0])
            A1 = np.where(sw[:, None], P["satirsiz"][1], P["satirli"][1])
            B0 = np.where(sw[:, None], P["satirli"][0], P["satirsiz"][0])
            B1 = np.where(sw[:, None], P["satirli"][1], P["satirsiz"][1])
            N.append((MK.olc_toplam(A1)["M1"] - MK.olc_toplam(A0)["M1"])
                     - (MK.olc_toplam(B1)["M1"] - MK.olc_toplam(B0)["M1"]))
        nn = np.array(N, dtype=float)
        null_sd = float(np.std(nn)); null_ort = float(np.mean(nn))
        if not np.isfinite(null_sd) or null_sd <= 0:
            OUT[x] = dict(hal="NULL-DEJENERE", null_sd=null_sd, fark=float(fark))
            atlanan.append(x); continue
        mde = 1.645 * null_sd
        OUT[x] = dict(hal="ÖLCÜLDÜ", fark=float(fark),
                      dM1_satirli=float(d_hat["satirli"]),
                      dM1_satirsiz=float(d_hat["satirsiz"]),
                      ci=ci, ayrik=bool(ci[1] < 0 or ci[0] > 0),
                      null_sd=null_sd, null_ort=null_ort, MDE=float(mde),
                      kucuk=bool(abs(fark) < mde),
                      plasebo_frac=float(np.mean(np.abs(nn) >= abs(fark))),
                      distinct4=d4, n_cift=int(len(ist)), n_istem=int(len(ad)))
        print(f"  [{time.time()-t0:.0f}s] {x}: fark={fark:+.3f} CI=[{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"MDE={mde:.3f} {'AYRIK' if OUT[x]['ayrik'] else ''}", flush=True)
    olculen = [x for x, v in OUT.items() if v["hal"] == "ÖLCÜLDÜ"]
    n_ayrik = sum(1 for x in olculen if OUT[x]["ayrik"])
    n_kucuk = sum(1 for x in olculen if OUT[x]["kucuk"])
    if len(olculen) < BAR_AILE:
        VERDICT = "ÖLCÜLEMEZ (payda<12)"
    elif n_ayrik >= BAR_AILE:
        yon = "yukari" if np.mean([OUT[x]["fark"] for x in olculen if OUT[x]["ayrik"]]) > 0 else "asagi"
        VERDICT = f"SATIR-ETKILI · {yon}"
    elif n_kucuk >= BAR_AILE:
        VERDICT = "SATIR-ETKISIZ"
    else:
        VERDICT = "KARISIK"
    K = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         prereg="prereg_system_prompt_2026-09-10.md @ 5090356c",
                         alet="scripts/system_prompt_verdict.py",
                         motor="",
                         NB=NB, NPERM=NPERM, bar_aile=BAR_AILE, d_shelf=D_SHELF,
                         d4_olcu=a.d4_olcu,
                         d4_olcu_aciklama=("HAVUZLANMIS (2026-09-10 hükmü)"
                                           if a.d4_olcu == "havuz"
                                           else "METIN BASINA (W-1101/ERRATA-2)"),
                         kok=CIK_KOK),
             VERDICT=VERDICT, n_aile=len(aileler), n_olculen=len(olculen),
             n_ayrik=n_ayrik, n_kucuk=n_kucuk, atlanan=atlanan, aile=OUT)
    json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("msistem_verdict", n_aile=len(aileler), hal_d4_olcu=a.d4_olcu,
          hal_olculen=len(olculen),
          hal_ayrik=n_ayrik, hal_kucuk=n_kucuk, red_atlanan=len(atlanan),
          red_bekleyen=len(_bekleyen))
    print(f"  ★★★ VERDICT: {VERDICT} · ölcülen {len(olculen)}/{len(aileler)} · bar {BAR_AILE}", flush=True)
    print(f"  ✓ {CIK}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
