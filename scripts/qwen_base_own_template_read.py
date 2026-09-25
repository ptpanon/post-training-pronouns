#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, sys, time
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import numpy as np
import form_count as FS
import v96_k3_read as K3
import urial_verdict as UH
from reading_style_object import payda

KOK = __DNH_DATA__ + "/c1_panel_template"
AILE = ("Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B")
CIK = f"{ROOT}/results/qwen_base_own_template_2026-09-24.json"
PREREG = "preregistration/prereg_qwen_base_own_template_2026-09-24.md"


def yukle(d):
    return [json.loads(l) for l in open(f"{KOK}/{d}/uretim.jsonl", encoding="utf-8")]


def main():
    nlp = FS._boru(); t0 = time.time(); S = {}
    for a in AILE:
        Rt = yukle(f"{a}/base.KIRLI_W974"); Rh = yukle(f"{a}/instruct")
        o, ek = K3.cift(nlp, Rt, Rh, duzyazi=False)
        i0, i1 = K3.P2.cift_al(Rt, Rh)
        Rt2 = [Rt[x] for x in i0]; Rh2 = [Rh[y] for y in i1]
        tut = UH.dshelf(Rt2) & UH.dshelf(Rh2)
        bos = np.array([not (r.get("metin") or "").strip() or not (q.get("metin") or "").strip()
                        for r, q in zip(Rt2, Rh2)])
        S[a] = dict(
            d2nd=dict(taban=o["M1_1k"]["taban"], delta=o["M1_1k"]["gozlenen"], ci=o["M1_1k"]["ci"]),
            d1st=dict(taban=o["IS1_1k"]["taban"], delta=o["IS1_1k"]["gozlenen"], ci=o["IS1_1k"]["ci"]),
            n_cift=ek["n_cift"], n_tutulan=ek["n_tutulan"], n_bos_tutulan=int((bos & tut).sum()))
        for k in ("d2nd", "d1st"):
            c = S[a][k]["ci"]; S[a][k]["ayrik"] = bool(c[1] < 0 or c[0] > 0)
        print(f"  [{time.time()-t0:.0f}s] {a:<13} Δ2nd {S[a]['d2nd']['delta']:+.3f} "
              f"[{S[a]['d2nd']['ci'][0]:+.2f},{S[a]['d2nd']['ci'][1]:+.2f}] · "
              f"Δ1st {S[a]['d1st']['delta']:+.3f} [{S[a]['d1st']['ci'][0]:+.2f},{S[a]['d1st']['ci'][1]:+.2f}] · "
              f"n_tutulan {S[a]['n_tutulan']} (bos {S[a]['n_bos_tutulan']})", flush=True)
    say = {k: dict(asagi_ayrik=sum(1 for v in S.values() if v[k]["ayrik"] and v[k]["delta"] < 0),
                   yukari_ayrik=sum(1 for v in S.values() if v[k]["ayrik"] and v[k]["delta"] > 0),
                   belirsiz=sum(1 for v in S.values() if not v[k]["ayrik"])) for k in ("d2nd", "d1st")}
    json.dump(dict(sinif="BETIM · card · bar yok · verdict adi yok", prereg=PREREG,
                   nesne=dict(taban="",
                              hizali=""),
                   motor="v96_k3_read.cift(duzyazi=False) — ITHAL", aile=S, sayim=say,
                   okunmayan=dict(**{"Qwen2.5-32B": ""}),
                   damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sure_dk=round((time.time() - t0) / 60, 1)),
              open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("qwen_taban_kendi_template", n_aile=len(S), n_cift_top=sum(v["n_cift"] for v in S.values()),
          red_bos_tutulan=sum(v["n_bos_tutulan"] for v in S.values()), bekle={"n_aile": 4})
    print(f"  sayim Δ2nd {say['d2nd']} · Δ1st {say['d1st']}\n✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
