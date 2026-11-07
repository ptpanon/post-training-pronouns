#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import depersonalization16_neutral_run as M9
import arm_table as KT
import person_dial_verdict as KH
import form_count as FS
from reading_style_object import payda

CIK = os.environ.get("V61M_CIK", f"{ROOT}/results/v61_ladder_2026-09-11.json")
KOK_CIPLAK = os.environ.get("V61M_KOK", __DNH_DATA__ + "/c1_panel_v61")
KOK_TEMPLATE = os.environ.get("V61M_KOK_SBL", __DNH_DATA__ + "/c1_panel_v61_template")
NB = KH.NB
PLASEBO_N = KH.PLASEBO_N
D_SHELF = 0.60
SEED = 20260911

MERDIVEN = {
    "SmolLM2-1.7B": dict(
        taban=("taban", "HuggingFaceTB/SmolLM2-1.7B"),
        sft=("sft", "HuggingFaceTB/SmolLM2-1.7B-sft-only"),
        dpo=("dpo", "HuggingFaceTB/SmolLM2-1.7B-Instruct")),
    "NeuralHermes-2.5": dict(
        taban=("taban", "mistralai/Mistral-7B-v0.1"),
        sft=("sft", "teknium/OpenHermes-2.5-Mistral-7B"),
        dpo=("dpo", "mlabonne/NeuralHermes-2.5-Mistral-7B")),
}
PROTOKOL = (("ciplak", KOK_CIPLAK), ("template", KOK_TEMPLATE))


def _M1(V):
    return float(MK.olc_toplam(V)["M1"])


def _ayrik(ci):
    return bool(ci) and (ci[0] > 0 or ci[1] < 0)


def _ortusmez(ci_a, ci_b):
    if not ci_a or not ci_b:
        return False
    return ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]


def kaskad(d_sft, ci_sft, d_pref, ci_pref, pla_sft, pla_pref, raf):
    if raf:
        return "ÖLCÜLEMEZ", "D-SHELF: bir bacak dejenerelik süzgecine takildi"
    if d_sft is None or d_pref is None:
        return "ÖLCÜLEMEZ", "basamak eksik"
    if pla_sft is None or pla_pref is None:
        return "ÖLCÜLEMEZ", (""
                             "")
    gecen_pref = _ayrik(ci_pref) and d_pref < 0 and abs(d_pref) > pla_pref
    gecen_sft = _ayrik(ci_sft) and d_sft < 0 and abs(d_sft) > pla_sft
    if not _ortusmez(ci_sft, ci_pref):
        return "ÖLCÜLEMEZ", ""
    if gecen_pref and abs(d_pref) > abs(d_sft):
        return "AYNI-ADIM", "tercih adimi negatif-ayrik ve mutlak degerce büyük"
    if gecen_sft and abs(d_sft) > abs(d_pref):
        return "FARKLI-ADIM", "SFT adimi negatif-ayrik ve mutlak degerce büyük"
    return "ÖLCÜLEMEZ", ""


def prova():
    s = []
    s.append(("AYNI-ADIM", kaskad(-1.0, [-1.4, -0.6], -5.0, [-5.6, -4.4], 0.9, 1.0, False)[0]))
    s.append(("FARKLI-ADIM", kaskad(-5.0, [-5.6, -4.4], -1.0, [-1.4, -0.6], 1.0, 0.9, False)[0]))
    s.append(("ÖLCÜLEMEZ", kaskad(-2.0, [-2.6, -1.4], -2.2, [-2.8, -1.6], 0.9, 0.9, False)[0]))
    s.append(("ÖLCÜLEMEZ", kaskad(-1.0, [-1.4, -0.6], -5.0, [-5.6, -4.4], 0.9, 1.0, True)[0]))
    s.append(("ÖLCÜLEMEZ", kaskad(-1.0, [-1.4, -0.6], -5.0, [-5.6, -4.4], 0.9, 9.0, False)[0]))
    s.append(("ÖLCÜLEMEZ", kaskad(+8.0, [7.4, 8.6], -5.0, [-5.6, -4.4], 0.9, 1.0, False)[0]))
    s.append(("FARKLI-ADIM", kaskad(-8.0, [-8.6, -7.4], -5.0, [-5.6, -4.4], 0.9, 1.0, False)[0]))
    s.append(("ÖLCÜLEMEZ", kaskad(-1.0, [-1.4, -0.6], -5.0, [-5.6, -4.4], None, 1.0, False)[0]))
    ok = all(b == g for b, g in s)
    dogan = sorted({g for _, g in s})
    payda("v61_merdiven_prova", n_sinav=len(s),
          hal_gecen=sum(1 for b, g in s if b == g), hal_dogan_ad=len(dogan),
          red_uyusmaz=sum(1 for b, g in s if b != g))
    for b, g in s:
        print(f"  [PROVA] beklenen {b:12s} ⇒ dogan {g:12s} {'✓' if b == g else '✗'}")
    print(f"  ★ dogan ad kümesi: {dogan} ⇒ esik: üc adin ücü dogmazsa "
          f"⇒ EYLEM: VERDICT YAZILMAZ (K-1: ateslenemeyen dal uzayi sisirir)", flush=True)
    return ok and len(dogan) == 3


def _bacak(nlp, kok, aile, bacak):
    R = MK.oku(aile, bacak, kok=kok)
    if R is None:
        return None
    V = MK.satir_bilesenleri(nlp, R)
    ist = np.array([int(r["istem_i"]) for r in R])
    d4 = KH.distinct4([r["metin"] for r in R])
    return dict(V=V, ist=ist, n=len(R), d4=float(d4), M1=_M1(V))


def main():
    if not prova():
        print("★ PROVA DÜSTÜ ⇒ VERDICT YAZILMAZ"); return 4
    nlp = FS._boru(); t0 = time.time(); OUT = {}
    for merd, bas in MERDIVEN.items():
        OUT[merd] = {}
        for pad, kok in PROTOKOL:
            B, eksik = {}, []
            for rol in ("taban", "sft", "dpo"):
                bacak, depo = bas[rol]
                b = _bacak(nlp, kok, merd, bacak)
                if b is None:
                    eksik.append(f"{rol}({bacak})"); B[rol] = None
                else:
                    b["depo"] = depo; B[rol] = b
            if eksik:
                OUT[merd][pad] = dict(hal="BEKLIYOR", eksik=eksik)
                print(f"{merd:18s} {pad:7s} {', '.join(eksik)}"
                      f"", flush=True)
                continue
            raf = [r for r in ("taban", "sft", "dpo") if B[r]["d4"] < D_SHELF]
            adim = {}
            for ad, (ust, alt) in (("sft", ("sft", "taban")), ("pref", ("dpo", "sft"))):
                d = B[ust]["M1"] - B[alt]["M1"]
                ci = KT._boot(B[ust]["V"], B[ust]["ist"], B[alt]["V"], B[alt]["ist"],
                              nb=NB, seed=SEED)
                pl = M9.plasebo_esli(B[alt]["V"], B[alt]["ist"],
                                     n=PLASEBO_N, seed=SEED)
                p95 = float(pl["p95"]) if pl and pl.get("n", 0) > 1 else None
                merkez = (abs(pl["ort"]) / pl["sd"]) if pl and pl.get("sd") else None
                adim[ad] = dict(dM1=round(d, 4),
                                ci=[round(x, 4) for x in ci] if ci else None,
                                ayrik=_ayrik(ci),
                                plasebo_p95=round(p95, 4) if p95 else None,
                                plasebo_n=(pl or {}).get("n"),
                                plasebo_merkez_sd=round(merkez, 3) if merkez else None,
                                red_merkezlenmedi=int(bool(merkez and merkez > 1.645)))
            ad, neden = kaskad(adim["sft"]["dM1"], adim["sft"]["ci"],
                               adim["pref"]["dM1"], adim["pref"]["ci"],
                               adim["sft"]["plasebo_p95"], adim["pref"]["plasebo_p95"],
                               bool(raf))
            OUT[merd][pad] = dict(hal="ÖLCÜLDÜ", VERDICT=ad, neden=neden,
                                  adim=adim, raf=raf,
                                  M1={r: round(B[r]["M1"], 4) for r in B},
                                  distinct4={r: round(B[r]["d4"], 4) for r in B},
                                  depo={r: B[r]["depo"] for r in B},
                                  n_satir={r: B[r]["n"] for r in B})
            print(f"  [{time.time()-t0:5.0f}s] {merd:18s} {pad:7s} SFT {adim['sft']['dM1']:+7.2f} "
                  f"· tercih {adim['pref']['dM1']:+7.2f} ⇒ ★ {ad} ({neden})", flush=True)
    ol = [v for m in OUT.values() for v in m.values() if v.get("hal") == "ÖLCÜLDÜ"]
    sayim = {}
    for v in ol:
        sayim[v["VERDICT"]] = sayim.get(v["VERDICT"], 0) + 1
    neden_sayim = {}
    for v in ol:
        if v["VERDICT"] == "ÖLCÜLEMEZ":
            k = v["neden"].split(":")[0].split("(")[0].strip()
            neden_sayim[k] = neden_sayim.get(k, 0) + 1
    n_raf = sum(1 for v in ol if v.get("raf"))
    n_merkezsiz = sum(1 for v in ol for a in v.get("adim", {}).values()
                      if a.get("red_merkezlenmedi"))
    n_hane = len(MERDIVEN) * len(PROTOKOL)
    payda("v61_merdiven_verdict", n_hane=n_hane, hal_olculen=len(ol),
          hal_ayni_adim=sayim.get("AYNI-ADIM", 0),
          hal_farkli_adim=sayim.get("FARKLI-ADIM", 0),
          hal_olculemez=sayim.get("ÖLCÜLEMEZ", 0),
          hal_dshelf_reddi=n_raf,
          red_merkezlenmemis_plasebo=n_merkezsiz,
          red_bekleyen=n_hane - len(ol))
    print(f"{neden_sayim}"
          f"", flush=True)
    if n_merkezsiz:
        print(f"{n_merkezsiz}"
              f""
              f"", flush=True)
    print(f"{sayim} {n_hane}"
          f"", flush=True)
    K = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="VERDICT", alet="scripts/v61_ladder_verdict.py",
             prereg="prereg_v61_ladder_2026-09-11.md @ 06e13a49",
             notice="prerun_notice_v61_2026-09-11.md @ d6e7863c",
             prediction="prediction_v61_2026-09-11.md @ b4503313",
             kok=dict(ciplak=KOK_CIPLAK, template=KOK_TEMPLATE),
             n_bootstrap=NB, plasebo_n=PLASEBO_N, d_shelf=D_SHELF, seed=SEED,
             kume_birimi="istem_i", uzay=["AYNI-ADIM", "FARKLI-ADIM", "ÖLCÜLEMEZ"],
             sayim=sayim, olculemez_nedenleri=neden_sayim,
             n_dshelf_reddi=n_raf, n_merkezlenmemis_plasebo=n_merkezsiz,
             SERH_DIZE_KAPISI=(""
                               ""
                               ""
                               ""
                               ""),
             merdiven=OUT)
    json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}  ({(time.time()-t0)/60:.1f} dk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
