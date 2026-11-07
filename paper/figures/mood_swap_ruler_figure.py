#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
import numpy as np
from style import ROOT

UYUM = [("DOMINANT", 462, 5.6122), ("NEUTRAL", 830, 5.5196),
        ("SUBMISSIVE", 480, 5.8097), ("UNCLEAR", 28, 5.6628)]


def cetvel_sayimi():
    import json as _json
    sys.path.insert(0, f"{ROOT}/scripts")
    from valence_sanity_raw import sozluk_yukle as _sy, profil as _pr
    _V, _A, _D = _sy()
    _C = [_json.loads(l) for l in open(__DNH_DATA__ + "/c1_kip_takas/"
                                       "kip_takas.jsonl", encoding="utf-8")]
    def _tab(k): return "EMIR" if k.split("→")[-1] == "duz" else "RICA"
    esit = ters = duz = yok = 0
    for c in _C:
        ta, tb = _tab(c["kip_a"]), _tab(c["kip_b"])
        if ta == tb: continue
        e = c["metin_a"] if ta == "EMIR" else c["metin_b"]
        r = c["metin_b"] if ta == "EMIR" else c["metin_a"]
        de, _, _ = _pr(e, _D); dr, _, _ = _pr(r, _D)
        if de is None or dr is None: yok += 1; continue
        if abs(de - dr) < 1e-12: esit += 1
        elif de > dr: duz += 1
        else: ters += 1
    n_kaps = esit + duz + ters
    print(f"  [PAYDA] f4_cetvel: n_capraz={esit+duz+ters+yok} · hal_kapsanan={n_kaps} · "
          f"atl_sozluksuz={yok} · hal_esit={esit} · hal_ters={ters} · hal_duz={duz}")
    print(f"  [ESIK] f4_cetvel.hal_esit/{n_kaps} = {esit/n_kaps:.3f} ⇒ esik: 'cetvel "
          f"cogunlukta SESSIZ' ⇔ > 0,50 ⇒ EYLEM: uyari panelde kalir")
    return esit, ters, duz, yok, n_kaps




def main():
    print(cetvel_sayimi())


if __name__ == "__main__":
    main()
