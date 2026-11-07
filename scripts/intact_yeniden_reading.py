#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import collections, json, os, shutil, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import dejenerelik_olcer as D
import family_panel as C

BEKLENEN_SATIR = {}
_PAYDA_ASIL = None


def denetim_payda(etiket, bekle=None, **kw):
    if etiket.startswith("c1_yukle_") and isinstance(bekle, dict) and "n_satir" in bekle:
        bekle = dict(bekle)
        bekle["n_satir"] = BEKLENEN_SATIR.get(etiket[len("c1_yukle_"):], bekle["n_satir"])
    return _PAYDA_ASIL(etiket, bekle=bekle, **kw)


KAYNAK = __DNH_DATA__ + "/c1_panel"
AYNA = __DNH_DATA__ + "/denetim_intact/c1_panel"


def ayna_kur(rel):
    kay, hed = f"{KAYNAK}/{rel}", f"{AYNA}/{rel}"
    R = [json.loads(l) for l in open(f"{kay}/uretim.jsonl", encoding="utf-8")]
    ok = np.array([not D.bayrakla(r.get("metin", ""))["DEJENERE"] for r in R])
    kol = np.array([r["kol"] for r in R]); ist = np.array([r["istem_i"] for r in R])
    kova = collections.Counter((k, i) for k, i, o in zip(kol, ist, ok) if o)
    bos = len({(k, i) for k in set(kol) for i in set(ist)}) - len(kova)
    if bos:
        return len(R), int(ok.sum()), bos
    os.makedirs(hed, exist_ok=True)
    with open(f"{hed}/uretim.jsonl", "w", encoding="utf-8") as f:
        for r, o in zip(R, ok):
            if o:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    X = np.load(f"{kay}/X_e5.npy")
    np.save(f"{hed}/X_e5.npy", X[ok])
    shutil.copy(f"{kay}/uretim_kunye.json", f"{hed}/uretim_kunye.json")
    return len(R), int(ok.sum()), 0


def main():
    global _PAYDA_ASIL
    _PAYDA_ASIL = C.payda
    C.payda = denetim_payda
    panel = sys.argv[1] if len(sys.argv) > 1 else "c1"
    C.panel_sec(panel)
    eski = json.load(open(C.CIKTI, encoding="utf-8")).get("panel", {}) if os.path.exists(C.CIKTI) else {}
    C_OUT_ESKI = C.OUT_KOK
    OUT, kapi = {}, []
    for c in C.PANEL:
        rel = {z: C.uretim_yolu(c["ad"], z).replace(C_OUT_ESKI + "/", "") for z in ("base", "instruct")}
        durum = {}
        for z, r in rel.items():
            if not os.path.exists(f"{KAYNAK}/{r}/uretim.jsonl"):
                durum[z] = ("ÜRETIM-YOK", 0, 0, 0); continue
            n, ni, b = ayna_kur(r)
            BEKLENEN_SATIR[f"{c['ad']}_{z}"] = ni
            durum[z] = ("KAPI" if b else "OK", n, ni, b)
        if any(v[0] != "OK" for v in durum.values()):
            OUT[c["ad"]] = dict(HAL="KAPI/ÖLCÜLEMEZ", sebep={z: v[0] for z, v in durum.items()},
                                n_intact={z: v[2] for z, v in durum.items()},
                                n_bos_kova={z: v[3] for z, v in durum.items()})
            kapi.append(c["ad"]); print(f"  ★ KAPI/ÖLCÜLEMEZ {c['ad']}: {OUT[c['ad']]['sebep']} "
                                        f"· bos kova {OUT[c['ad']]['n_bos_kova']}", flush=True)
            continue
        t0 = time.time()
        try:
            C.OUT_KOK = AYNA
            zb, Bb, UNb = C._zemin_oku(c["ad"], "base", "cpu")
            zi, Bi, UNi = C._zemin_oku(c["ad"], "instruct", "cpu")
        except Exception as e:
            C.OUT_KOK = C_OUT_ESKI
            OUT[c["ad"]] = dict(HAL="KAPI/ÖLCÜLEMEZ", sebep=f"{type(e).__name__}: {str(e)[:90]}")
            kapi.append(c["ad"]); print(f"  ★ KAPI {c['ad']}: {e}", flush=True); continue
        finally:
            C.OUT_KOK = C_OUT_ESKI
        d = {}
        for e in C.EKSENLER:
            v = Bb[e] - Bi[e]
            ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            sd = float(v.std()); dU = zb["olcum"][e]["U"] - zi["olcum"][e]["U"]
            dn = UNb[e] - UNi[e]; m = float(dn.mean())
            poz, neg = ci[0] > 0, ci[1] < 0
            if C.MERKEZLI:
                poz, neg = (ci[0] - m) > 0, (ci[1] - m) < 0
            k = abs(dU - (m if C.MERKEZLI else 0)) / sd if sd > 0 else None
            d[e] = dict(dU=round(dU, 6), CI=[round(x, 6) for x in ci], sd=round(sd, 6),
                        null_merkezi=round(m, 6), sd_kati=round(k, 3) if k else None,
                        AD=("AYRIK_POZ" if poz else "AYRIK_NEG" if neg else
                            "CONTINGENT" if (k is not None and 1.0 <= k < 1.645) else "~"))
        poz = zb["olcum"]["ARO"]["AYRIK_POZ"] and zi["olcum"]["ARO"]["AYRIK_POZ"]
        OUT[c["ad"]] = dict(HAL="OK", POZ_KONTROL=bool(poz), delta=d,
                            n_base=zb["n_satir"], n_instruct=zi["n_satir"],
                            n_kume=zb["n_kume"], saniye=round(time.time() - t0, 1))
        e_ad = {e: ("AYRIK_POZ" if (eski.get(c["ad"], {}).get("delta", {}).get(e, {}) or {}).get("AYRIK_POZ")
                    else "AYRIK_NEG" if (eski.get(c["ad"], {}).get("delta", {}).get(e, {}) or {}).get("AYRIK_NEG")
                    else "CONTINGENT" if (eski.get(c["ad"], {}).get("delta", {}).get(e, {}) or {}).get("CONTINGENT")
                    else "~") for e in C.EKSENLER}
        OUT[c["ad"]]["eski_ad"] = e_ad
        OUT[c["ad"]]["VERDICT"] = ("DAYANIR" if all(d[e]["AD"] == e_ad[e] for e in C.EKSENLER)
                                 else "DÜSER")
        print(f"  [{c['ad']:<20}] {OUT[c['ad']]['VERDICT']:8s} n {zb['n_satir']}/{zi['n_satir']} · "
              + " · ".join(f"{e}: {e_ad[e]}→{d[e]['AD']} (Δ{d[e]['dU']:+.5f})" for e in C.EKSENLER),
              flush=True)
    payda(f"intact_{panel}", n_hucre=len(C.PANEL), red_kapi=len(kapi),
          hal_dayanir=sum(1 for v in OUT.values() if v.get("VERDICT") == "DAYANIR"),
          hal_duser=sum(1 for v in OUT.values() if v.get("VERDICT") == "DÜSER"))
    y = f"{ROOT}/results/INTACT_YENIDEN_{panel}_2026-08-26.json"
    json.dump(dict(SINIF="", panel=panel,
                   MERKEZLI=C.MERKEZLI, hucre=OUT), open(y, "w"), ensure_ascii=False, indent=1)
    print(f"→ {y}")


if __name__ == "__main__":
    main()
