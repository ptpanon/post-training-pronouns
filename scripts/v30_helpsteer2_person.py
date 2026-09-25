#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import gzip, io, json, os, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_olcu as MO
import form_count as FS
from tercih_person_delta import yogunluk
from reading_style_object import payda

CIK = f"{ROOT}/results/v30_helpsteer2_person_2026-09-10.json"
KOKLER = [f"{ROOT}/.hf/hub", "<storage>/huggingface/hub",
          os.path.expanduser("~/.cache/huggingface/hub")]
ALT = "datasets--nvidia--HelpSteer2/snapshots"


def kaynak():
    for k in KOKLER:
        d = os.path.join(k, ALT)
        if not os.path.isdir(d):
            continue
        for s in sorted(os.listdir(d)):
            y = os.path.join(d, s, "preference", "preference.jsonl.gz")
            if os.path.exists(y):
                return y, s
    return None, None


def main():
    yol, sha = kaynak()
    if not yol:
        print("★ HelpSteer2-Preference ÜC KÖKTE de yok (find -L usulü) ⇒ DUR")
        return 4
    C, R, berabere, bos = [], [], 0, 0
    with gzip.open(yol, "rt", encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            s = d.get("preference_strength")
            if s is None or int(s) == 0:
                berabere += 1; continue
            c = (d["response_2"] if int(s) > 0 else d["response_1"]) or ""
            r = (d["response_1"] if int(s) > 0 else d["response_2"]) or ""
            if not c.strip() or not r.strip():
                bos += 1; continue
            C.append(c.strip()); R.append(r.strip())
    payda("v30_helpsteer2_yukle", n_toplam=len(C) + berabere + bos, hal_cift=len(C),
          red_berabere=berabere, red_bos_taraf=bos)
    print(f"  [PAYDA] HelpSteer2-Pref: n_cift={len(C)} · berabere düstü={berabere} · "
          f"bos={bos} ⇒ esik: n_cift < 1000 ⇒ EYLEM: verdict YAZILMAZ (güc)", flush=True)
    if len(C) < 1000:
        print("★ n_cift < 1000 ⇒ ÖLCÜLEMEZ"); return 3
    nlp = FS._boru(); t0 = time.time()
    Ac, nc_c = MO.topla(nlp, C, n_process=6)
    Ar, nc_r = MO.topla(nlp, R, n_process=6)
    yc, yr = yogunluk(Ac, nc_c), yogunluk(Ar, nc_r)
    jc = np.maximum(Ac["n_jeton"], 1); jr = np.maximum(Ar["n_jeton"], 1)
    d_m1 = 1000 * (Ac["m1_sahis2"] / jc - Ar["m1_sahis2"] / jr)
    d_s1 = 1000 * ((Ac["m5_yakin"] - Ac["m1_sahis2"]) / jc
                   - (Ar["m5_yakin"] - Ar["m1_sahis2"]) / jr)
    rng = np.random.default_rng(20260910)
    B = np.array([float(d_m1[rng.integers(0, len(d_m1), len(d_m1))].mean())
                  for _ in range(2000)])
    ci = [float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))]
    S = dict(n_cift=len(C), chosen=yc, rejected=yr, ci_cift_bootstrap=ci,
             n_bootstrap=2000, ayrik=bool(ci[0] < 0 < ci[1]) is False,
             havuz_dM1=yc["M1"] - yr["M1"], havuz_dSahis1=yc["sahis1"] - yr["sahis1"],
             medyan_dM1=float(np.median(d_m1)), medyan_dSahis1=float(np.median(d_s1)),
             ort_dM1=float(np.mean(d_m1)), sem_dM1=float(np.std(d_m1) / np.sqrt(len(d_m1))),
             uzunluk_orani=float(jc.mean() / max(jr.mean(), 1)),
             jeton_ort=dict(chosen=float(jc.mean()), rejected=float(jr.mean())))
    S["VERDICT"] = ("VERI-YÖNÜ-DESTEKLER" if S["havuz_dM1"] < 0 else
                  "TERS-YÖN" if S["havuz_dM1"] > 0 else "SIFIR")
    S["_kunye"] = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       alet="scripts/v30_helpsteer2_person.py", borc="D-0910-V30 §5",
                       on_kayit="HAFTA-1 §2 (korpustan bagimsiz rule) — yeni bar YAZILMADI",
                       zemin="nvidia/HelpSteer2 · preference · INSAN ETIKETLI",
                       kaynak_yol=yol, snapshot=sha, sure_sn=round(time.time() - t0, 1))
    json.dump(S, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  cift-ort {S['ort_dM1']:+.3f} CI {ci[0]:+.3f},{ci[1]:+.3f} · "
          f"ayrik={S['ayrik']}")
    print(f"  ΔM1 havuz {S['havuz_dM1']:+.3f} · medyan {S['medyan_dM1']:+.4f} · "
          f"Δ1.kisi havuz {S['havuz_dSahis1']:+.3f} · uzunluk orani "
          f"{S['uzunluk_orani']:.3f} ⇒ {S['VERDICT']}")
    payda("v30_helpsteer2", n_cift=len(C), hal_dM1=round(S["havuz_dM1"], 3))
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
