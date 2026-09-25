#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, re, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import siradan_instruction as ST
import v96_k3_read as K3O
import v96_q4_read as Q4O
import form_count as FS
import v97_q4_turn as DRV

PREREG = "preregistration/prereg_v97_q4_turn_2026-09-20.md"
KOK = DRV.KOK
OLCULER = K3O.OLCULER
TUR = [t for t, _ in DRV.TURLER]
FORUM = re.compile(r"(?m)^\s*[>@]|\b(?:Re:|OP\b|thread\b|quoting\b|Quote:)", re.I)
NASIL = re.compile(r"(?m)^\s*(?:\d+[.)]|Step\b|First,|Next,|Then,|Finally,)", re.I)
NOTICE = re.compile(r"\b(?:said|says|according to|reported|told|spokesperson|statement)\b", re.I)


def damga(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def s3(ci): return "asagi" if ci[1] < 0 else ("yukari" if ci[0] > 0 else "null")


def sinif(t):
    t = t or ""
    if FORUM.search(t):
        return "TUR_FORUM"
    if NASIL.search(t):
        return "TUR_NASIL"
    if NOTICE.search(t):
        return "TUR_NOTICE"
    return "TUR_BLOG"


def yukle(dizin):
    return [json.loads(l) for l in io.open(f"{KOK}/{dizin}/uretim.jsonl", encoding="utf-8")]


def main():
    B = json.load(io.open(Q4O.B1, encoding="utf-8")); V = json.load(io.open(Q4O.V91, encoding="utf-8"))
    nlp = FS._boru(); H = ST.havuz(); t0 = time.time()
    satir, fark_max, eksik = {}, 0.0, []
    yuv = lambda d: {x: (round(v, 4) if isinstance(v, float) else v) for x, v in d.items()}
    for ad in ST.aile_sirasi():
        k = H[ad]
        Rt_b1, _ = Q4O.yukle(Q4O.KOK_B1, k["taban"], True); Rh_b1, _ = Q4O.yukle(Q4O.KOK_B1, k["hizali"], False)
        m1, is1, _, _ = Q4O.cift_oku(nlp, [dict(r) for r in Rt_b1], [dict(r) for r in Rh_b1])
        kb, kv = B["aile"][ad]["M1"], V["aile"][ad]["IS1"]
        fark = max(abs(round(m1["gozlenen"], 4) - kb["gozlenen"]), abs(round(m1["taban"], 4) - kb["taban"]),
                   abs(m1["ci"][0] - kb["ci"][0]), abs(m1["ci"][1] - kb["ci"][1]),
                   abs(round(is1["gozlenen"], 4) - kv["gozlenen"]), abs(is1["ci"][0] - kv["ci"][0]), abs(is1["ci"][1] - kv["ci"][1]))
        fark_max = max(fark_max, fark)
        if not (DRV.bacak_tam(k["taban"]) and DRV.bacak_tam(k["hizali"])):
            satir[ad] = dict(hal="BACAK-EKSIK", kapi_fark=fark); eksik.append(ad)
            print(f"  {ad:16s} kapi {fark:.1e} · BACAK-EKSIK", flush=True); continue
        Rt, Rh = yukle(k["taban"]), yukle(k["hizali"])
        r = dict(hal="OKUNDU", kapi_fark=fark)
        for etiket, alt in [("TUM", None)] + [(t, t) for t in TUR]:
            A = [x for x in Rt if alt is None or x.get("kol") == alt]
            Bk = [x for x in Rh if alt is None or x.get("kol") == alt]
            o, ek = K3O.cift(nlp, [dict(x) for x in A], [dict(x) for x in Bk])
            r[etiket] = dict(**ek, **{m: yuv(o[m]) for m in OLCULER}, **{f"{m}_sinif3": s3(o[m]["ci"]) for m in OLCULER})
        r["uyum"] = {}
        for bac, R in (("taban", Rt), ("hizali", Rh)):
            M = {t: {u: 0 for u in TUR} for t in TUR}
            for x in R:
                M[x["kol"]][sinif(x.get("metin"))] += 1
            r["uyum"][bac] = dict(matris=M, pay={t: round(M[t][t] / max(sum(M[t].values()), 1), 4) for t in TUR})
        satir[ad] = r
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {ad:16s} kapi {fark:.1e} · TÜM " +
              " · ".join(f"{m} {r['TUM'][m]['gozlenen']:+.2f} {r['TUM'][f'{m}_sinif3']}" for m in ("M1_1k", "IS1_1k")) +
              " · uyum(hizali) " + " ".join(f"{t.split('_')[1][:4]} {r['uyum']['hizali']['pay'][t]:.2f}" for t in TUR), flush=True)
    hal = "ESDEGER" if fark_max <= 1e-9 else "ALET-KAYDI"
    ok = {a: v for a, v in satir.items() if v["hal"] == "OKUNDU"}
    say = {e: {m: {c: sum(1 for v in ok.values() if v[e][f"{m}_sinif3"] == c) for c in ("asagi", "yukari", "null")}
               for m in OLCULER} for e in ["TUM"] + TUR}
    plas = {e: {m: sum(1 for v in ok.values() if v[e][f"{m}_sinif3"] == "asagi" and v[e][m].get("plasebo_ustu")) for m in OLCULER}
            for e in ["TUM"] + TUR}
    uy_ort = {bac: {t: round(float(np.median([v["uyum"][bac]["pay"][t] for v in ok.values()])), 4) for t in TUR}
              for bac in ("taban", "hizali")} if ok else {}
    cik = f"{ROOT}/results/V97_Q4_TUR_{time.strftime('%Y-%m-%d', time.gmtime())}.json"
    json.dump(dict(sinif="BETIM · card · bar yok · verdict adi yok", prereg=PREREG, alet=dict(hal=hal, kapi_fark_max=fark_max),
                   tasarim=dict(turler=dict(DRV.TURLER), n_cekim=DRV.N_CEKIM, n_satir_bacak=DRV.N_BEK,
                                zemin=""),
                   referans="B1 kapisi (SIRADAN_TALIMAT_OKUMA + V91) · ham devam paneli PREREG9_A: ΔM1 16/16 asagi",
                   aile=satir if hal == "ESDEGER" else {}, sayim=say if hal == "ESDEGER" else {},
                   asagi_ve_plasebo_ustu=plas if hal == "ESDEGER" else {}, tur_uyumu_ortanca=uy_ort, bacak_eksik=eksik,
                   motor="v96_k3_read.cift (p11_2x2 · urial_verdict.dshelf · reviewer_readings_v80) · kapi v96_q4_read.cift_oku",
                   damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)),
              io.open(cik, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{hal} {fark_max:.1e} {len(ok)} {len(eksik)}"
          f"")
    for e in ["TUM"] + TUR:
        print(f"    {e}: " + " · ".join(f"{m} {say[e][m]}" for m in ("M1_1k", "IS1_1k")) +
              f" · plasebo üstü {plas[e]['M1_1k']}/{plas[e]['IS1_1k']}")
    print(f"    tür uyumu (ortanca): {uy_ort}")
    print(f"✓ {cik}")
    return 0 if hal == "ESDEGER" else 3


if __name__ == "__main__":
    sys.exit(main())
