#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, hashlib, os
ROOT = __DNH_ROOT__ + ""
TABAN = "results/four_seed_reading_2026-09-01.json"
EK = "results/SEED_UC_AILE_OKUMA_2026-09-15.json"


def _sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def oku():
    T = json.load(open(f"{ROOT}/{TABAN}", encoding="utf-8"))
    E = json.load(open(f"{ROOT}/{EK}", encoding="utf-8"))
    S = dict(T); n_yerine = 0; n_ayni = 0
    for a, v in E.items():
        if a.startswith("_") or not isinstance(v, dict) or v.get("hal") != "m=4":
            continue
        t = T.get(a) or {}
        if t.get("hal") == "m=4":
            fark = max(abs(x - y) for x, y in zip(t["seed_dM1"], v["seed_dM1"]))
            if fark > 1e-9:
                raise ValueError(f"{a}: iki kartta m=4 ve farkli (Δ {fark:.2e})")
            n_ayni += 1; continue
        S[a] = v; n_yerine += 1
    print(f"{sum(1 for k, v in T.items() if not k.startswith('_') and v.get('hal') == 'm=4')}"
          f"{n_yerine} {n_ayni}")
    src = [dict(yol=TABAN, sha256_16=_sha16(f"{ROOT}/{TABAN}")), dict(yol=EK, sha256_16=_sha16(f"{ROOT}/{EK}"))]
    return S, src
