#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS

PANEL = __DNH_DATA__ + "/c1_panel"
A_CARD = f"{ROOT}/results/PREREG9_A_OLCUM_2026-08-31.json"
CIK = f"{ROOT}/results/v22_force_dilution_2026-09-06.json"
NB, SEED = 1000, 20260906
A9_S = {"Gemma-3-12B": +0.19, "Gemma-3-27B": -0.10, "Gemma-3-4B": +0.13,
        "Llama-3.1-70B": -0.21, "Llama-3.1-8B": -0.15, "Mistral-7B-v0.3": -0.08,
        "OLMo-3-7B": -0.54, "OLMo2-13B": -0.18, "OLMo2-32B": -0.22,
        "Qwen2.5-1.5B": -1.05, "Qwen2.5-14B": -0.75, "Qwen2.5-32B": -1.21,
        "Qwen2.5-3B": -0.52, "Qwen2.5-72B": -0.64, "Qwen2.5-7B": -0.42,
        "Tulu3-8B": -0.20}
J, S2, KUV = 0, 1, 9


def _prova():
    x = np.array([[100.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 2.0, 10.0]])
    a = _oranlar(x); b = _oranlar(x)
    d1k = np.log(b["kuv_1k"]) - np.log(a["kuv_1k"])
    return {"i_ozdes_sifir_delta": bool(abs(d1k) < 1e-12),
            "ii_s_tanimsiz": bool(np.isnan(_s(0.0, 0.0))),
            "iii_kimlik": bool(abs((np.log(a["kuv_c"]) - np.log(a["kuv_1k"]))
                                   - np.log(a["uz"] / 1000.0)) < 1e-9),
            "iv_sutun_dogru": bool(abs(x[0, KUV] - 2.0) < 1e-12
                                   and abs(x[0, J] - 100.0) < 1e-12)}


def _oranlar(V):
    s = V.sum(axis=0)
    jet, cum = max(s[J], 1.0), max(s[-1], 1.0)
    return dict(kuv_c=s[KUV] / cum, kuv_1k=1000 * s[KUV] / jet,
                m1_c=s[S2] / cum, m1_1k=1000 * s[S2] / jet, uz=jet / cum)


def _s(dlog_uz, dlog_1k):
    return float("nan") if abs(dlog_1k) < 1e-12 else dlog_uz / dlog_1k


def _dlog(A, B, ad):
    a, b = _oranlar(A)[ad], _oranlar(B)[ad]
    return float("nan") if (a <= 0 or b <= 0) else float(np.log(b) - np.log(a))


def boot(Va, ia, Vb, ib):
    ka = {u: np.where(ia == u)[0] for u in np.unique(ia)}
    kb = {u: np.where(ib == u)[0] for u in np.unique(ib)}
    ortak = [u for u in ka if u in kb]
    rng = np.random.default_rng(SEED)
    out = {k: [] for k in ("kuv_1k", "kuv_c", "m1_1k", "uz", "s_kuv", "s_m1")}
    for _ in range(NB):
        sec = rng.choice(ortak, size=len(ortak), replace=True)
        A = Va[np.concatenate([ka[u] for u in sec])]
        B = Vb[np.concatenate([kb[u] for u in sec])]
        du = _dlog(A, B, "uz")
        for ad in ("kuv_1k", "kuv_c", "m1_1k"):
            out[ad].append(_dlog(A, B, ad))
        out["uz"].append(du)
        out["s_kuv"].append(_s(du, out["kuv_1k"][-1]))
        out["s_m1"].append(_s(du, out["m1_1k"][-1]))
    R = {}
    for k, v in out.items():
        v = np.array(v, float); v = v[np.isfinite(v)]
        R[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 20 else None
    return R


MERD = {"Tulu3-8B": [("sft", "dpo"), ("dpo", "rl")],
        "OLMo2-13B": [("sft", "dpo"), ("dpo", "instruct")],
        "OLMo2-32B": [("sft", "dpo"), ("dpo", "instruct")]}


def merdiven():
    nlp = FS._boru(); t0 = time.time(); OUT = {}; onb = {}

    def V(a, z):
        if (a, z) not in onb:
            R = MK.oku(a, z)
            onb[(a, z)] = None if R is None else (
                MK.satir_bilesenleri(nlp, R),
                np.array([r.get("istem_i", -1) for r in R]))
        return onb[(a, z)]

    for a, adimlar in MERD.items():
        OUT[a] = {}
        for z0, z1 in adimlar:
            A0, A1 = V(a, z0), V(a, z1)
            if A0 is None or A1 is None:
                print(f"{a} {z0} {z1}"); continue
            d = {ad: _dlog(A0[0], A1[0], ad) for ad in ("kuv_1k", "kuv_c", "uz")}
            ci = boot(A1[0], A1[1], A0[0], A0[1])
            OUT[a][f"{z0}→{z1}"] = dict(dlog=d, s_kuvvet=_s(d["uz"], d["kuv_1k"]),
                                        ci=ci,
                                        ayrik=bool(ci["kuv_1k"] and ci["kuv_1k"][0] * ci["kuv_1k"][1] > 0))
            print(f"  ★ {a:11s} {z0:4s}→{z1:9s} Δlog kuvvet/1k {d['kuv_1k']:+.3f} "
                  f"· /cümle {d['kuv_c']:+.3f} · Δlog uz {d['uz']:+.3f} · s "
                  f"{OUT[a][f'{z0}→{z1}']['s_kuvvet']:+.2f} "
                  f"[{(time.time()-t0)/60:.1f} dk]", flush=True)
    n = sum(len(v) for v in OUT.values())
    ay = sum(1 for v in OUT.values() for x in v.values() if x["ayrik"])
    print(f"\n  [PAYDA] basamak={n} · Δlog(kuvvet/1k) CI-ayrik={ay} ⇒ esik: "
          f"ayrik degilse s basilmaz (A9 kurali)")
    y = f"{ROOT}/results/v22_force_dilution_ladder_2026-09-06.json"
    json.dump(dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               alet="scripts/v22_force_dilution.py --merdiven",
                               borc="D-0907-V22 §3a (A9 alt blogu)", nb=NB,
                               seed=SEED, kume="istem", n_basamak=n, ayrik=ay),
                   merdiven=OUT), open(y, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {y} · {(time.time()-t0)/60:.1f} dk")
    return 0


def main():
    P = _prova(); print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False), flush=True)
    if not all(P.values()):
        print("★★ DEJENERE PROVA DÜSTÜ ⇒ ölcüm yok"); return 4
    if "--merdiven" in sys.argv:
        return merdiven()
    A = json.load(open(A_CARD, encoding="utf-8"))
    aileler = [(k, v["kontrast"].split("→")) for k, v in A.items()
               if not k.startswith("_") and isinstance(v, dict) and "kontrast" in v
               and k in A9_S]
    nlp = FS._boru(); t0 = time.time(); OUT = {}
    for n, (aile, (z0, z1)) in enumerate(sorted(aileler), 1):
        R0, R1 = MK.oku(aile, z0), MK.oku(aile, z1)
        if R0 is None or R1 is None:
            print(f"{aile}"); continue
        V0 = MK.satir_bilesenleri(nlp, R0); V1 = MK.satir_bilesenleri(nlp, R1)
        i0 = np.array([r.get("istem_i", -1) for r in R0])
        i1 = np.array([r.get("istem_i", -1) for r in R1])
        d = {ad: _dlog(V0, V1, ad) for ad in ("kuv_1k", "kuv_c", "m1_1k", "uz")}
        s_kuv, s_m1 = _s(d["uz"], d["kuv_1k"]), _s(d["uz"], d["m1_1k"])
        ci = boot(V1, i1, V0, i0)
        OUT[aile] = dict(kontrast=f"{z0}→{z1}",
                         taban=_oranlar(V0), hizali=_oranlar(V1),
                         dlog=d, s_kuvvet=s_kuv, s_M1=s_m1, ci=ci,
                         kuv_1k_ayrik=bool(ci["kuv_1k"] and ci["kuv_1k"][0] * ci["kuv_1k"][1] > 0),
                         A9_s_basili=A9_S[aile], sapma_s_M1=s_m1 - A9_S[aile])
        print(f"  ★ {aile:18s} Δlog kuvvet/1k {d['kuv_1k']:+.3f} · /cümle "
              f"{d['kuv_c']:+.3f} · Δlog uz {d['uz']:+.3f} · s_kuv "
              f"{s_kuv:+.2f} | s_M1 {s_m1:+.2f} (A9 {A9_S[aile]:+.2f}, sapma "
              f"{s_m1-A9_S[aile]:+.3f}) [{n}/{len(aileler)} · {(time.time()-t0)/60:.1f} dk]",
              flush=True)
    sap = {k: v["sapma_s_M1"] for k, v in OUT.items() if np.isfinite(v["sapma_s_M1"])}
    enb = max(abs(x) for x in sap.values()) if sap else float("nan")
    gecti = bool(enb <= 0.005)
    print(f"{len(sap)}"
          f"{enb:.4f}"
          f"")
    ay = sum(1 for v in OUT.values() if v["kuv_1k_ayrik"])
    print(f"{len(OUT)} {ay}"
          f"")
    json.dump(dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               alet="scripts/v22_force_dilution.py",
                               borc="D-0907-V22 §3a", zemin="C1-PANEL (nötr)",
                               tanim="s = Dlog(jeton/cumle)/Dlog(X/1k); A9 ile birebir",
                               nb=NB, seed=SEED, kume="istem", prova=P,
                               esdegerlik_en_buyuk_sapma_s_M1=enb,
                               esdegerlik_gecti=gecti),
                   aile=OUT), open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK} · {(time.time()-t0)/60:.1f} dk")
    return 0 if gecti else 5


if __name__ == "__main__":
    sys.exit(main())
