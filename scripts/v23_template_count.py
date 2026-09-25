#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
import rung_force_ci as BK

SAB = __DNH_DATA__ + "/c1_panel_template"
CIP = __DNH_DATA__ + "/c1_panel"
CIK = f"{ROOT}/results/v23_template_count_2026-09-06.json"
A_CARD = f"{ROOT}/results/PREREG9_A_OLCUM_2026-08-31.json"
N_BEK = 6528

HIZALI = [("Qwen2.5-1.5B", "instruct"), ("Qwen2.5-3B", "instruct"),
          ("Qwen2.5-7B", "instruct"), ("Qwen2.5-14B", "instruct"),
          ("Qwen2.5-32B", "instruct"), ("Gemma-3-4B", "instruct"),
          ("Gemma-3-12B", "instruct"), ("Gemma-3-27B", "instruct"),
          ("OLMo-3-7B", "instruct"), ("OLMo2-13B", "instruct"),
          ("OLMo2-32B", "instruct"), ("Mistral-7B-v0.3", "instruct"),
          ("Llama-3.1-8B", "instruct"), ("Tulu3-8B", "rl")]
def cift_kur():
    import templated_base_leg_kirlilik as STK
    out, red = [], []
    for a in sorted(os.listdir(SAB)):
        if not os.path.isdir(f"{SAB}/{a}"):
            continue
        zh = next((z for z in ("instruct", "rl") if os.path.exists(f"{SAB}/{a}/{z}/uretim.jsonl")), None)
        zt = next((z for z in ("base", "taban") if os.path.exists(f"{SAB}/{a}/{z}/uretim.jsonl")), None)
        if not (zh and zt):
            continue
        esit, n, farkli, _, _ = STK.dize_esit(a, zt, zh, kok=SAB)
        if esit:
            out.append((a, zt, zh))
        else:
            red.append(f"{a}/{zt}↔{zh}={'FARKLI(' + str(farkli) + '/' + str(n) + ')' if esit is False else 'CÖZÜLEMEDI'}")
    print(f"  ★ (B) aile kümesi diskten: {len(out)} gecti · {len(red)} düstü {red}", flush=True)
    return out, red


def _prova():
    x = np.array([[100.0, 5.0, 0, 2.0, 1.0, 0, 0, 0, 0, 1.0, 10.0]])
    a = MK.olc_toplam(x); b = MK.olc_toplam(x)
    return {"i_ozdes_delta_sifir": bool(abs(a["M1"] - b["M1"]) < 1e-12),
            "ii_M1_birimi_bin_jeton": bool(abs(a["M1"] - 50.0) < 1e-9),
            "iii_M3_oran": bool(abs(a["M3"] - 0.5) < 1e-9)}


def oku(kok, aile, bacak):
    y = f"{kok}/{aile}/{bacak}/uretim.jsonl"
    ky = f"{kok}/{aile}/{bacak}/uretim_kunye.json"
    if not os.path.exists(y):
        return None, "ÜRETIM-YOK"
    R = [json.loads(l) for l in open(y, encoding="utf-8")]
    k = json.load(open(ky, encoding="utf-8")) if os.path.exists(ky) else {}
    if len(R) != N_BEK:
        return None, f"SATIR-EKSIK({len(R)}≠{N_BEK})"
    if k.get("TAM") is not True:
        return None, "KÜNYE-TAM-DEGIL"
    return R, dict(n=len(R), sn=k.get("saniye"), damga=k.get("damga_utc"),
                   template=(k.get("kimlik") or {}).get("imza", {}).get("template"))


def main():
    P = _prova(); print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False), flush=True)
    if not all(P.values()):
        print("★★ PROVA DÜSTÜ"); return 4
    A = json.load(open(A_CARD, encoding="utf-8"))
    nlp = FS._boru(); t0 = time.time()
    V = {}; KUN = {}; RED = []
    CIFT, CIFT_RED = cift_kur()
    hepsi = [(a, z, SAB) for a, z in HIZALI] \
        + [(a, zt, SAB) for a, zt, _ in CIFT] + [(a, zh, SAB) for a, _, zh in CIFT] \
        + [(a, z, CIP) for a, z in HIZALI]
    hepsi = list(dict.fromkeys(hepsi))
    for n, (a, z, kok) in enumerate(hepsi, 1):
        etiket = ("SAB" if kok == SAB else "CIP")
        R, k = oku(kok, a, z)
        if R is None:
            RED.append(f"{etiket}:{a}/{z}={k}"); print(f"  ✗ {etiket} {a}/{z}: {k}"); continue
        V[(etiket, a, z)] = (MK.satir_bilesenleri(nlp, R),
                             np.array([r.get("istem_i", -1) for r in R]))
        KUN[f"{etiket}:{a}/{z}"] = k
        print(f"  ✓ {etiket} {a:16s}/{z:8s} n={k['n']} [{n}/{len(hepsi)} · "
              f"{(time.time()-t0)/60:.1f} dk]", flush=True)

    OUT = {"A_template_kaymasi": {}, "B_sablonlu_delta": {}}
    for a, z in HIZALI:
        s, c = V.get(("SAB", a, z)), V.get(("CIP", a, z))
        if not s or not c:
            continue
        ms, mc = MK.olc_toplam(s[0]), MK.olc_toplam(c[0])
        lo, hi, nk = BK.kume_boot(s[0], s[1], c[0], c[1], "M1")
        OUT["A_template_kaymasi"][a] = dict(
            bacak=z, M1_sablonlu=ms["M1"], M1_ciplak=mc["M1"],
            d_M1=ms["M1"] - mc["M1"], ci=[lo, hi], ayrik=bool(lo * hi > 0),
            n_istem=nk, template_var=KUN[f"SAB:{a}/{z}"]["template"],
            M3_sablonlu=ms["M3"], M3_ciplak=mc["M3"],
            KUVVET_sablonlu=ms["KUVVET"], KUVVET_ciplak=mc["KUVVET"])
        o = OUT["A_template_kaymasi"][a]
        print(f"  ★ (A) {a:16s} M1 ciplak {mc['M1']:6.2f} → sablonlu {ms['M1']:6.2f} "
              f"· Δ {o['d_M1']:+.2f} [{lo:+.2f},{hi:+.2f}] "
              f"{'AYRIK' if o['ayrik'] else ''} · template={o['template_var']}", flush=True)
    for a, zt, zh in CIFT:
        b, i = V.get(("SAB", a, zt)), V.get(("SAB", a, zh))
        if not b or not i:
            continue
        mb, mi = MK.olc_toplam(b[0]), MK.olc_toplam(i[0])
        lo, hi, nk = BK.kume_boot(i[0], i[1], b[0], b[1], "M1")
        ciplak = A[a]["fark"]["M1"]["gozlenen"] if a in A else None
        OUT["B_sablonlu_delta"][a] = dict(
            zemin_taban=zt, zemin_hizali=zh,
            M1_base=mb["M1"], M1_instruct=mi["M1"], d_M1=mi["M1"] - mb["M1"],
            ci=[lo, hi], ayrik=bool(lo * hi > 0), n_istem=nk,
            d_M1_ciplak=ciplak)
        o = OUT["B_sablonlu_delta"][a]
        c_s = f"{ciplak:+.2f}" if ciplak is not None else "—"
        print(f"  ★ (B) {a:16s} sablonlu ΔM1 {o['d_M1']:+.2f} [{lo:+.2f},{hi:+.2f}] "
              f"{'AYRIK' if o['ayrik'] else '     '} ↔ ciplak {c_s}", flush=True)

    nA = len(OUT["A_template_kaymasi"]); ayA = sum(1 for v in OUT["A_template_kaymasi"].values() if v["ayrik"])
    nB = len(OUT["B_sablonlu_delta"]); ayB = sum(1 for v in OUT["B_sablonlu_delta"].values() if v["ayrik"])
    print(f"{nA} {ayA}"
          f"")
    print(f"{nB} {ayB} {nB}"
          f"")
    print(f"  [PAYDA] okunan bacak={len(V)}/{len(hepsi)} · REDDEDILEN={len(RED)} {RED}")
    json.dump(dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               alet="scripts/v23_template_count.py",
                               borc="D-0907-GECE-2 §1 (sayim yarisi)",
                               kok_sablonlu=SAB, kok_ciplak=CIP, n_beklenen_satir=N_BEK,
                               prova=P, n_bacak=len(V), red=RED,
                               SERH_SUPHELI_HIZLI=""
                                                  ""
                                                  "",
                               SERH_SABLONSUZ=""
                                              ""),
                   kunye=KUN, **OUT), open(CIK, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"✓ {CIK} · {(time.time()-t0)/60:.1f} dk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
