#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, os, subprocess, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
G = f"{ROOT}/results"
RULE = f"{ROOT}/preregistration/RULE_V104_TABAN_BIRINCI_2026-09-23.md"
CIK = f"{G}/v104_base_birinci_2026-09-23.json"
CARD_DD = f"{G}/v97_t1_dogrudan_birinci_2026-09-19.json"
ESIK = 1e-9


def sha(y):
    return hashlib.sha256(open(y, "rb").read()).hexdigest()


def rule_kapisi():
    h = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "-1", "--", os.path.relpath(RULE, ROOT)],
                       capture_output=True, text=True).stdout.strip()
    if not h:
        raise SystemExit("")
    b = sha(os.path.abspath(__file__))
    if b not in io.open(RULE, encoding="utf-8").read():
        raise SystemExit(f"{b[:16]}")
    return h, b[:16]


def main():
    import v90_tez_kume as V90, reviewer_readings_v80 as H80, addressee_run as MK
    h, b = rule_kapisi()
    DD = json.load(io.open(CARD_DD, encoding="utf-8"))
    if DD["alet"]["hal"] != "ESDEGER" or len(DD["aile"]) != 16:
        raise SystemExit("")
    S, red, t0 = {}, [], time.time()
    for n, a in enumerate(sorted(DD["aile"]), 1):
        z = np.load(f"{V90.ARA}/tez_{a}.npz")
        P0, P1, dos = z["P0"], z["P1"], [str(x) for x in z["dosyalar"]]
        yb, yi = dos[:4], dos[4:]
        p0, p1 = [], []
        for zb, zi in zip(yb, yi):
            Rb = [json.loads(l) for l in io.open(zb, encoding="utf-8")]
            Ri = [json.loads(l) for l in io.open(zi, encoding="utf-8")]
            ib, ii = H80.eslestir(Rb, Ri)
            p0.append(H80.p1_sayim([Rb[x] for x in ib], "haric"))
            p1.append(H80.p1_sayim([Ri[y] for y in ii], "haric"))
        p0, p1 = np.concatenate(p0), np.concatenate(p1)
        if len(p0) != len(P0) or len(p1) != len(P1):
            red.append(a); print(f"{a} {len(p0)} {len(P0)}"); continue
        W0, W1 = H80.ikame(P0, p0), H80.ikame(P1, p1)
        tb = MK.olc_toplam(W0)["M1"]; hz = MK.olc_toplam(W1)["M1"]
        sap = abs((hz - tb) - DD["aile"][a]["d1st_dogrudan"])
        if sap > ESIK:
            red.append(a); print(f"  ✗ {a}: ESDEGERLIK DÜSTÜ {sap:.2e} ⇒ karta GIRMEZ"); continue
        S[a] = dict(taban_1k=float(tb), hizali_1k=float(hz), fark=float(hz - tb),
                    card_d1st=DD["aile"][a]["d1st_dogrudan"], kapi_sapma=float(sap), n_satir=int(len(P0)))
        print(f"  ★ {a:16s} taban {tb:7.2f} → hizali {hz:6.2f}  (Δ {hz-tb:+7.2f}; kapi {sap:.1e}) "
              f"[{n}/16 · {(time.time()-t0)/60:.1f} dk]", flush=True)
    if red:
        print(f"{len(S)} {len(red)} {red}")
        return 3
    tb = [v["taban_1k"] for v in S.values()]
    print(f"{len(S)}"
          f"{min(tb):.2f} {max(tb):.2f} {max(tb)/min(tb):.2f}")
    json.dump(dict(sinif="BETIM · bar YOK · sahip «EXECUTOR · v104» (4)", rule=h, alet=b,
                   alet_yolu="scripts/v104_base_birinci.py",
                   kaynak=dict(card=os.path.relpath(CARD_DD, ROOT), sha256_16=sha(CARD_DD)[:16],
                               npz=f"{V90.ARA}/tez_<aile>.npz"),
                   olcu="dogrudan birinci-sahis sayimi (US haric), 1 000 jetonda, dört ek seed havuzlanmis",
                   kapi=dict(ad="hizali − taban ≡ V97 kartinin d1st_dogrudan'i", esik=ESIK, red=red),
                   bant=dict(min=min(tb), max=max(tb), oran=max(tb) / min(tb)),
                   aile=S, damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
              io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
