#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""; sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK, form_count as FS, rung_force_ci as BK, template_filtered as SS
import echo as YK
from template_filtered_first_person import birinci
from reading_style_object import payda
CIK = f"{ROOT}/results/template_filtered_first_person_echo_free_2026-09-15.json"
nlp = FS._boru(); t0 = time.time(); CIFT, _ = SS.V23.cift_kur(); out = []
for n, (a, zt, zh) in enumerate(CIFT, 1):
    bac = {("sbl", "taban"): SS.oku(SS.SAB, a, zt), ("sbl", "hizali"): SS.oku(SS.SAB, a, zh),
           ("cip", "taban"): SS.oku(SS.CIP, a, zt), ("cip", "hizali"): SS.oku(SS.CIP, a, zh)}
    F = {k: SS.bayraklar(R) for k, R in bac.items()}
    A = {k: [SS.anahtar(r) for r in R] for k, R in bac.items()}
    kirli = set()
    for k in bac:
        kirli |= {A[k][i] for i in np.where(F[k][0])[0]}
    yank = set()
    for k in (("sbl", "taban"), ("sbl", "hizali")):
        yank |= {A[k][i] for i, r in enumerate(bac[k]) if YK.yanki_mi(r.get("onek", ""), r["metin"])}
    dis = kirli | yank
    Rt, Rh = bac[("sbl", "taban")], bac[("sbl", "hizali")]
    mt = np.array([x not in dis for x in A[("sbl", "taban")]]); mh = np.array([x not in dis for x in A[("sbl", "hizali")]])
    Vt = MK.satir_bilesenleri(nlp, Rt); Vh = MK.satir_bilesenleri(nlp, Rh)
    it = np.array([x.get("istem_i", -1) for x in Rt]); ih = np.array([x.get("istem_i", -1) for x in Rh])
    d2 = MK.olc_toplam(Vh[mh])["M1"] - MK.olc_toplam(Vt[mt])["M1"]; l2, h2, _ = BK.kume_boot(Vh[mh], ih[mh], Vt[mt], it[mt], "M1")
    Wt, Wh = birinci(Vt), birinci(Vh)
    d1 = MK.olc_toplam(Wh[mh])["M1"] - MK.olc_toplam(Wt[mt])["M1"]; l1, h1, nk = BK.kume_boot(Wh[mh], ih[mh], Wt[mt], it[mt], "M1")
    r = dict(aile=a, tutulan_oran=round(float(mt.mean()), 4), yanki_dislanan_ek=round(len(yank - kirli) / len(A[("sbl", "taban")]), 4),
             d1=round(float(d1), 4), ci1=[round(l1, 4), round(h1, 4)], d2=round(float(d2), 4), ci2=[round(l2, 4), round(h2, 4)], n_istem=int(nk))
    out.append(r)
    print(f"  {a:16s} tutulan {r['tutulan_oran']:.1%} · 1.s {d1:+7.2f} [{l1:+.2f},{h1:+.2f}] · 2.s {d2:+7.2f} [{l2:+.2f},{h2:+.2f}]  [{n}/16 · {(time.time()-t0)/60:.1f} dk]", flush=True)
def sinif(ci): return "yukari" if ci[0] > 0 else ("asagi" if ci[1] < 0 else "null")
S = {k: {c: [r["aile"] for r in out if sinif(r[f"ci{k}"]) == c] for c in ("yukari", "asagi", "null")} for k in ("1", "2")}
json.dump(dict(_kunye=dict(alet="scripts/template_filtered_first_person_echo_free.py", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           sinif="BETIMSEL — prereg yok", hucre="dört-bacak dejenerelik süzgeci ∩ iki sablonlu bacakta yankisiz (esli)",
                           yanki="echo.yanki_mi (200/40/10)", sure_dk=round((time.time() - t0) / 60, 1)),
               sayim=S, aileler=out), io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k in S: print(f"★ sablonlu {k}.s (süzülmüs+yankisiz): yukari {len(S[k]['yukari'])} · asagi {len(S[k]['asagi'])} · null {len(S[k]['null'])}")
payda("template_sahis1_yankisiz", n_aile=len(CIFT), hal_olculdu=len(out))
print(f"✓ {CIK}")
