#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import siradan_instruction as ST
import reviewer_readings_v80 as H80
import two_rulers_2x2 as P2
import urial_verdict as UH
import form_count as FS

B1 = f"{ROOT}/results/siradan_instruction_reading_2026-09-17.json"
V91 = f"{ROOT}/results/v91_konusan_sayimi_2026-09-17.json"
KOK_B1, KOK_Q4, KOK_K3 = ST.KOK, __DNH_DATA__ + "/v96_q4", __DNH_DATA__ + "/v96_k3"
PREREG = "preregistration/prereg_v96_k3_impersonal_ozdes_2026-09-19.md"
OLCULER = ("M1_1k", "IS1_1k", "duz_M1_1k", "duz_IS1_1k", "duz_M1_cumle", "duz_IS1_cumle")


def damga(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def s3(ci): return "asagi" if ci[1] < 0 else ("yukari" if ci[0] > 0 else "null")


def tam(kok, dizin):
    y, k = f"{kok}/{dizin}/uretim.jsonl", f"{kok}/{dizin}/uretim_kunye.json"
    if not (os.path.exists(y) and os.path.exists(k)):
        return False
    return sum(1 for _ in io.open(y, encoding="utf-8")) == ST.N_BEK and json.load(io.open(k, encoding="utf-8")).get("TAM") is True


def yukle(kok, dizin, kes):
    R = [json.loads(l) for l in io.open(f"{kok}/{dizin}/uretim.jsonl", encoding="utf-8")]
    n = 0
    if kes:
        for r in R:
            r["metin"], k = ST.taban_kes(r["metin"]); n += k
    return R, n


def cift(nlp, Rt, Rh, duzyazi=True):
    fM1, fC = P2.OLCU["M1_1k"][0], P2.OLCU["M1_cumle"][0]
    i0, i1 = P2.cift_al(Rt, Rh)
    Rt = [Rt[x] for x in i0]; Rh = [Rh[y] for y in i1]
    tut = UH.dshelf(Rt) & UH.dshelf(Rh)
    ist = np.array([r["istem_i"] for r in Rh])[tut]
    V0, V1 = P2.bilesen(nlp, Rt), P2.bilesen(nlp, Rh)
    p0, p1 = H80.p1_sayim(Rt, "haric"), H80.p1_sayim(Rh, "haric")
    out = dict(M1_1k=P2.olc(V0[tut], V1[tut], ist, fM1),
               IS1_1k=P2.olc(H80.ikame(V0, p0)[tut], H80.ikame(V1, p1)[tut], ist, fM1))
    ek = dict(n_cift=len(i0), n_tutulan=int(tut.sum()))
    if duzyazi:
        D0 = [H80.duzyazi(r["metin"]) for r in Rt]; D1 = [H80.duzyazi(r["metin"]) for r in Rh]
        Q0 = [dict(r, metin=d[0]) for r, d in zip(Rt, D0)]; Q1 = [dict(r, metin=d[0]) for r, d in zip(Rh, D1)]
        W0, W1 = P2.bilesen(nlp, Q0), P2.bilesen(nlp, Q1)
        q0, q1 = H80.p1_sayim(Q0, "haric"), H80.p1_sayim(Q1, "haric")
        out.update(duz_M1_1k=P2.olc(W0[tut], W1[tut], ist, fM1),
                   duz_IS1_1k=P2.olc(H80.ikame(W0, q0)[tut], H80.ikame(W1, q1)[tut], ist, fM1),
                   duz_M1_cumle=P2.olc(W0[tut], W1[tut], ist, fC),
                   duz_IS1_cumle=P2.olc(H80.ikame(W0, q0)[tut], H80.ikame(W1, q1)[tut], ist, fC))
        ek.update(atilan_satir_payi=dict(taban=round(sum(d[1] for d in D0) / max(sum(d[2] for d in D0), 1), 4),
                                         hizali=round(sum(d[1] for d in D1) / max(sum(d[2] for d in D1), 1), 4)))
    return out, ek


def main():
    B = json.load(io.open(B1, encoding="utf-8")); V = json.load(io.open(V91, encoding="utf-8"))
    nlp = FS._boru(); H = ST.havuz(); t0 = time.time()
    satir, fark_max, eksik = {}, 0.0, []
    yuv = lambda d: {x: (round(v, 4) if isinstance(v, float) else v) for x, v in d.items()}
    for ad in ST.aile_sirasi():
        k = H[ad]
        Rt_b1, _ = yukle(KOK_B1, k["taban"], True); Rh_b1, _ = yukle(KOK_B1, k["hizali"], False)
        g, _ = cift(nlp, Rt_b1, Rh_b1, duzyazi=False)
        kb, kv = B["aile"][ad]["M1"], V["aile"][ad]["IS1"]
        fark = max(abs(round(g["M1_1k"]["gozlenen"], 4) - kb["gozlenen"]), abs(round(g["M1_1k"]["taban"], 4) - kb["taban"]),
                   abs(g["M1_1k"]["ci"][0] - kb["ci"][0]), abs(g["M1_1k"]["ci"][1] - kb["ci"][1]),
                   abs(round(g["IS1_1k"]["gozlenen"], 4) - kv["gozlenen"]), abs(g["IS1_1k"]["ci"][0] - kv["ci"][0]),
                   abs(g["IS1_1k"]["ci"][1] - kv["ci"][1]))
        fark_max = max(fark_max, fark)
        if not (tam(KOK_Q4, k["taban"]) and tam(KOK_K3, k["hizali"])):
            satir[ad] = dict(hal="BACAK-EKSIK", kapi_fark=fark); eksik.append(ad)
            print(f"  {ad:16s} kapi {fark:.1e} · BACAK-EKSIK", flush=True); continue
        Rt, nt = yukle(KOK_Q4, k["taban"], True); Rh, nh = yukle(KOK_K3, k["hizali"], True)
        o, ek = cift(nlp, Rt, Rh)
        satir[ad] = dict(hal="OKUNDU", kapi_fark=fark, kesilen_pay=dict(taban=round(nt / ST.N_BEK, 4), hizali=round(nh / ST.N_BEK, 4)),
                         **ek, **{m: yuv(o[m]) for m in OLCULER}, **{f"{m}_sinif3": s3(o[m]["ci"]) for m in OLCULER})
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {ad:16s} kapi {fark:.1e} · " +
              " · ".join(f"{m} {o[m]['gozlenen']:+.2f} {s3(o[m]['ci'])}" for m in OLCULER), flush=True)
    hal = "ESDEGER" if fark_max <= 1e-9 else "ALET-KAYDI"
    ok = {a: v for a, v in satir.items() if v["hal"] == "OKUNDU"}
    say = {m: {c: sum(1 for v in ok.values() if v[f"{m}_sinif3"] == c) for c in ("asagi", "yukari", "null")} for m in OLCULER}
    plas = {m: sum(1 for v in ok.values() if v[f"{m}_sinif3"] == "asagi" and v[m].get("plasebo_ustu")) for m in OLCULER}
    cik = f"{ROOT}/results/V96_K3_KISISIZ_OZDES_{time.strftime('%Y-%m-%d', time.gmtime())}.json"
    json.dump(dict(sinif="BETIM · card · bar yok · verdict adi yok", prereg=PREREG, alet=dict(hal=hal, kapi_fark_max=fark_max),
                   referans="B1: ΔM1 16/16 asagi · V91 dogrudan Δ1st 12/16 asagi · C(3) K1/K2 karti V96_Q4_URIAL_OZDES",
                   aile=satir if hal == "ESDEGER" else {}, sayim=say if hal == "ESDEGER" else {},
                   asagi_ve_plasebo_ustu=plas if hal == "ESDEGER" else {}, bacak_eksik=eksik,
                   motor="siradan_instruction.taban_kes (iki bacak) · two_rulers_2x2.cift_al/bilesen/olc (M1_1k · M1_cumle) · urial_verdict.dshelf · "
                         "reviewer_readings_v80.p1_sayim('haric') + ikame · reviewer_readings_v80.duzyazi",
                   damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)),
              io.open(cik, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{hal} {fark_max:.1e} {len(ok)} {len(eksik)}"
          f"")
    for m in OLCULER:
        print(f"    {m}: {say[m]} · asagi+plasebo üstü {plas[m]}")
    return 0 if hal == "ESDEGER" else 3


if __name__ == "__main__":
    sys.exit(main())
