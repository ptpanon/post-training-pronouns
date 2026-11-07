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
KOK_B1 = ST.KOK
KOK_Q4 = __DNH_DATA__ + "/v96_q4"
RULE = "preregistration/rule_v96_q4_urial_ozdes_2026-09-18.md @ c0322674"


def damga(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def cift_oku(nlp, Rt, Rh):
    fM1 = P2.OLCU["M1_1k"][0]
    i0, i1 = P2.cift_al(Rt, Rh)
    Rt = [Rt[x] for x in i0]; Rh = [Rh[y] for y in i1]
    tut = UH.dshelf(Rt) & UH.dshelf(Rh)
    V0, V1 = P2.bilesen(nlp, Rt), P2.bilesen(nlp, Rh)
    ist = np.array([r["istem_i"] for r in Rh])[tut]
    m1 = P2.olc(V0[tut], V1[tut], ist, fM1)
    p1t, p1h = H80.p1_sayim(Rt, "haric"), H80.p1_sayim(Rh, "haric")
    is1 = P2.olc(H80.ikame(V0, p1t)[tut], H80.ikame(V1, p1h)[tut], ist, fM1)
    return m1, is1, int(tut.sum()), len(i0)


def s3(ci): return "asagi" if ci[1] < 0 else ("yukari" if ci[0] > 0 else "null")


def main():
    B = json.load(io.open(B1, encoding="utf-8")); V = json.load(io.open(V91, encoding="utf-8"))
    nlp = FS._boru(); H = ST.havuz(); t0 = time.time()
    satir, fark_max, eksik = {}, 0.0, {"K1": [], "K2": []}
    yuv = lambda d: {x: (round(v, 4) if isinstance(v, float) else v) for x, v in d.items()}
    for ad in ST.aile_sirasi():
        k = H[ad]; sat = {}
        Rt_b1, n_kes_b1 = yukle(KOK_B1, k["taban"], True)
        Rh_b1, _ = yukle(KOK_B1, k["hizali"], False)
        m1, is1, _, _ = cift_oku(nlp, [dict(r) for r in Rt_b1], [dict(r) for r in Rh_b1])
        kb, kv = B["aile"][ad]["M1"], V["aile"][ad]["IS1"]
        fark = max(abs(round(m1["gozlenen"], 4) - kb["gozlenen"]), abs(round(m1["taban"], 4) - kb["taban"]),
                   abs(m1["ci"][0] - kb["ci"][0]), abs(m1["ci"][1] - kb["ci"][1]),
                   abs(round(is1["gozlenen"], 4) - kv["gozlenen"]), abs(is1["ci"][0] - kv["ci"][0]), abs(is1["ci"][1] - kv["ci"][1]))
        fark_max = max(fark_max, fark); sat["kapi_fark"] = round(fark, 12)
        if tam(KOK_Q4, k["hizali"]):
            Rh, n_kes = yukle(KOK_Q4, k["hizali"], True)
            m, s, n_t, n_c = cift_oku(nlp, [dict(r) for r in Rt_b1], Rh)
            sat["K1"] = dict(hal="OKUNDU", n_cift=n_c, n_tutulan=n_t, kesilen_pay_hizali=round(n_kes / ST.N_BEK, 4),
                             kesilen_pay_taban=round(n_kes_b1 / ST.N_BEK, 4), M1=yuv(m), IS1=yuv(s),
                             M1_sinif3=s3(m["ci"]), IS1_sinif3=s3(s["ci"]))
        else:
            sat["K1"] = dict(hal="BACAK-EKSIK"); eksik["K1"].append(ad)
        if tam(KOK_Q4, k["taban"]):
            Rt, n_kes = yukle(KOK_Q4, k["taban"], True)
            m, s, n_t, n_c = cift_oku(nlp, Rt, [dict(r) for r in Rh_b1])
            sat["K2"] = dict(hal="OKUNDU", n_cift=n_c, n_tutulan=n_t, kesilen_pay_taban=round(n_kes / ST.N_BEK, 4), M1=yuv(m), IS1=yuv(s),
                             M1_sinif3=s3(m["ci"]), IS1_sinif3=s3(s["ci"]))
        else:
            sat["K2"] = dict(hal="BACAK-EKSIK"); eksik["K2"].append(ad)
        satir[ad] = sat
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {ad:16s} kapi {fark:.1e} · "
              + " · ".join(f"{kk}: " + (f"ΔM1 {sat[kk]['M1']['gozlenen']:+.2f} {sat[kk]['M1_sinif3']} / Δ1st {sat[kk]['IS1']['gozlenen']:+.2f} {sat[kk]['IS1_sinif3']}"
                                      if sat[kk]["hal"] == "OKUNDU" else sat[kk]["hal"]) for kk in ("K1", "K2")), flush=True)
    hal = "ESDEGER" if fark_max <= 1e-9 else "ALET-KAYDI"
    say = {kk: {al: {c: sum(1 for v in satir.values() if v[kk]["hal"] == "OKUNDU" and v[kk][al] == c) for c in ("asagi", "yukari", "null")}
                for al in ("M1_sinif3", "IS1_sinif3")} for kk in ("K1", "K2")}
    cik = f"{ROOT}/results/V96_Q4_URIAL_OZDES_{time.strftime('%Y-%m-%d', time.gmtime())}.json"
    json.dump(dict(sinif="BETIM · card · bar yok", rule=RULE, alet=dict(hal=hal, kapi_fark_max=fark_max),
                   referans="B1: ikinci sahis 16/16 asagi · V91 dogrudan birinci sahis 12/16 asagi",
                   aile=satir if hal == "ESDEGER" else {}, sayim=say if hal == "ESDEGER" else {}, bacak_eksik=eksik,
                   motor="siradan_instruction.taban_kes (URIAL bicimli her bacak) · two_rulers_2x2.cift_al/bilesen/olc · urial_verdict.dshelf · "
                         "reviewer_readings_v80.p1_sayim('haric') + ikame",
                   damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)),
              io.open(cik, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{hal} {fark_max:.1e} {len(eksik['K1'])} {len(eksik['K2'])}"
          f"")
    for kk, d in say.items():
        print(f"    {kk}: {d}")
    return 0 if hal == "ESDEGER" else 3


if __name__ == "__main__":
    sys.exit(main())
