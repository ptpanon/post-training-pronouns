#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys, time
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import v61_scoring as VP
from reading_style_object import payda

CIK = f"{ROOT}/results/v61_example_pair_2026-09-14.json"


def main():
    t0 = time.time()
    C, kunye = VP.havuz_ciftleri(VP.N_CIFT)
    P = {int(json.loads(l)["istem_i"]): json.loads(l) for l in io.open(f"{VP.KOK}/armorm.jsonl", encoding="utf-8")}
    esles = sum(1 for c in C if c["istem_i"] in P and P[c["istem_i"]]["yon"] == c["yon"]
                and abs(float(P[c["istem_i"]]["degisim_payi"]) - c["degisim_payi"]) < 1e-9
                and int(float(P[c["istem_i"]]["kar_farki"])) == c["kar_farki"])
    print(f"  [PROVA] yeniden kurulan {len(C)} · puanlanan {len(P)} · alan-alan eslesen {esles} ⇒ esik {len(P)} ⇒ EYLEM: eksikse CIKIS 4")
    if len(C) != len(P) or esles != len(P):
        print("  ★★ PROVA DÜSTÜ ⇒ örnek YAZILMAZ"); return 4
    aday = [c for c in C if c["yon"] == "ileri" and len(c["kisili"]) <= 400 and c["n_takas"] == 1]
    payda("v61_ornek_cift", n_cift=len(C), hal_eslesen=esles, n_aday=len(aday))
    if not aday:
        print(""); return 4
    o = min(aday, key=lambda c: c["istem_i"])
    json.dump(dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), alet="scripts/v61_example_pair.py",
                               kaynak="v61_scoring.havuz_ciftleri (seed 20260911)", prova_eslesen=esles,
                               secim="yon=ileri · kisili ≤400 kar · n_takas=1 · en kücük istem_i", n_aday=len(aday),
                               gecen_dk=round((time.time() - t0) / 60, 1)),
                   ornek=dict(istem_i=o["istem_i"], kisili=o["kisili"], kisisiz=o["kisisiz"], n_takas=o["n_takas"],
                              degisim_payi=o["degisim_payi"]), uretim_kunyesi=kunye),
              io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK} · istem_i {o['istem_i']}\n  kisili : {o['kisili']!r}\n  kisisiz: {o['kisisiz']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
