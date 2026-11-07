#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, io, json, os, sys, time

KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
import template_robustness as SG
import template_robustness_24 as S24
import serita_harman102 as SA
import templated_base_leg_kirlilik as STK
from reading_style_object import payda

ISTEM = os.environ.get("URIAL_ISTEM", f"{KOK}/unreleased/inst_1k.txt")
SHA = os.environ.get("URIAL_SHA", "3a3989a9c5a58293362de475be08218cc552544882b8a8356b558d03c1a91f06")
COMMIT = "127f94ed4a08955a79368355fcd878a2ee8c0e2e"
URIAL_KOK = os.environ.get("URIAL_KOK", __DNH_DATA__ + "/c1_panel_urial")
CIK = os.environ.get("URIAL_CIK", f"{KOK}/results/URIAL_KOSU_2026-09-07.json")
N_BEK = 6528
DUR = os.environ.get("URIAL_DUR", __DNH_DATA__ + "/DUR_URIAL")
TEK_CARD = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Mistral-7B-v0.3", "Qwen2.5-7B",
            "OLMo-3-7B", "Llama-3.1-8B", "Tulu3-8B", "Gemma-3-4B",
            "OLMo2-13B", "Gemma-3-12B", "Qwen2.5-14B", "Gemma-3-27B",
            "OLMo2-32B", "Qwen2.5-32B"]
CIFT_CARD = ["Qwen2.5-72B", "Llama-3.1-70B"]

OLCULEN_CIFT = {"Llama-3.1-70B": 91.1, "Qwen2.5-72B": 94.0}

URIAL_DK = {"Qwen2.5-1.5B": 15.4, "Qwen2.5-3B": 21.4, "Mistral-7B-v0.3": 22.4,
            "Qwen2.5-7B": 22.8, "OLMo-3-7B": 43.1, "OLMo2-13B": 68.4,
            "Gemma-3-12B": 77.4, "Qwen2.5-32B": 78.8, "OLMo2-32B": 85.1,
            "Gemma-3-27B": 118.7, "Llama-3.1-70B": 145.0, "Qwen2.5-72B": 148.6,
            "Llama-3.1-8B": 29.0, "Tulu3-8B": 28.9, "Gemma-3-4B": 35.7}
URIAL_ORAN_TAVAN = 2.14


URIAL_TEPE = {"OLMo2-13B": 70319, "OLMo-3-7B": 24342, "Qwen2.5-72B": 77601}
PANEL_TEPE = {"Llama-3.1-8B": 16430, "Tulu3-8B": 16427, "Gemma-3-4B": 17706,
              "Qwen2.5-14B": 29848, "OLMo2-13B": 31568, "OLMo-3-7B": 17706}


def vram_kapisi(ad, dev_i=0):
    import mini_dpo_feasibility as FZ
    tepe = URIAL_TEPE.get(ad)
    rej, _, _ = FZ.prod_rejimi(dev_i)
    bos, kararli, okumalar = FZ.gpu_bos_mib_kararli(dev_i)
    t = os.environ.get("URIAL_TAMPON")
    tavan = (bos - int(t)) if t is not None else FZ.kapi_hesapla(bos, rej)
    if t is not None:
        rej = f"{rej}·TAMPON-SAHIP({t})"
    if tepe is None:
        if tavan <= 0:
            return False, (f"{tavan}"
                           f"{bos} {rej}")
        p = PANEL_TEPE.get(ad)
        risk = (f" · panel {p} ×2,23 ⇒ ~{int(p*2.23)} MiB "
                + ("**RISKLI, yine de DENENIR**" if p*2.23 > tavan else "beklenen: sigar")
                ) if p else " · panel tepesi de YOK"
        return True, (f"ÖLCÜM-YOK ⇒ kapi SUSAR · rejim {rej} · bos {bos} · "
                      f"tavan {tavan} MiB{risk}")
    if not kararli:
        return True, (f"{okumalar}"
                      f"{rej}"
                      f"")
    if tepe > tavan:
        return False, (f"{tepe} {tavan}"
                       f"{bos} {rej}"
                       f"{okumalar}")
    return True, f"sigar-BILINMIYOR · tepe {tepe} ≤ tavan {tavan} MiB (alt sinir)"


def beklenen_dk(ad, k):
    if ad in URIAL_DK:
        return URIAL_DK[ad], "URIAL-ÖLCÜLDÜ"
    return k.get("olculen_dk", 30) * URIAL_ORAN_TAVAN, f"panel×tavan({URIAL_ORAN_TAVAN})"


def cift_kollar():
    import family_panel as C1AP
    KAYNAK = [("PANEL_OLCEK", C1AP.PANEL_OLCEK), ("PANEL_OLCEK2", C1AP.PANEL_OLCEK2)]
    out, nereden = [], {}
    for tablo_ad, tablo in KAYNAK:
        for c in tablo:
            if c["ad"] not in CIFT_CARD or c["ad"] in nereden:
                continue
            nereden[c["ad"]] = tablo_ad
            out.append(dict(ad=c["ad"], taban=f"{c['ad']}/base", hizali=f"{c['ad']}/instruct",
                            s_taban=c["base"], s_hizali=c["instruct"], panel="c1", ikiz="ham",
                            olculen_dk=OLCULEN_CIFT[c["ad"]]))
    print(f"  [CIFT-KOL] {nereden}", flush=True)
    payda("urial_cift_kollar", n_istenen=len(CIFT_CARD), n_cozulen=len(out),
          red_bulunamayan=len(CIFT_CARD) - len(out))
    return out


def istem_yukle():
    if not os.path.exists(ISTEM):
        raise SystemExit(f"{ISTEM}")
    ham = io.open(ISTEM, encoding="utf-8").read()
    g = hashlib.sha256(ham.encode()).hexdigest()
    payda("urial_istem", n_dosya=1, hal_sha_tuttu=int(g == SHA), red_sha=int(g != SHA))
    if g != SHA:
        raise SystemExit(f"{SHA[:16]} {g[:16]}")
    import re
    nq = len(re.findall(r"^# Query:\s*$", ham, re.M))
    na = len(re.findall(r"^# Answer:\s*$", ham, re.M))
    payda("urial_K", n_query=nq, n_answer=na, red_K_sapti=int(not (nq == na == 3)))
    if not (nq == na == 3):
        raise SystemExit(f"{nq} {na}")
    print(f"  [URIAL] istem ✔ sha {g[:12]}… · K=3 · {len(ham)} bayt · commit {COMMIT[:8]}", flush=True)
    return ham


def urial_onek_fn_kurucu(ham_istem):
    def kurucu(tok):
        def f(ad, tur, c, ofs, i, ist, cekim):
            q = SA.onek_kur(ad, tur, c, ofs, i, ist, cekim)
            return f"{ham_istem}\n\n\n# Query:\n```{q}```\n \n# Answer:\n```\n"
        return f
    return kurucu


def icerik(dizin, kok=None, n_bek=None):
    K = URIAL_KOK if kok is None else kok
    y, k = f"{K}/{dizin}/uretim.jsonl", f"{K}/{dizin}/uretim_kunye.json"
    if not os.path.exists(y):
        return False, "ÜRETIM-YOK"
    n = sum(1 for _ in io.open(y, encoding="utf-8"))
    tam = json.load(io.open(k, encoding="utf-8")).get("TAM") if os.path.exists(k) else None
    bek = N_BEK if (kok is None and n_bek is None) else n_bek
    if bek is not None and n != bek:
        return False, f"SATIR {n}≠{bek}"
    return (tam is True), (f"n={n} TAM" if tam is True else "KÜNYE-TAM-DEGIL")


def dize_kapisi(aile, zt, zh):
    esit, n, farkli, oa, ob = STK.dize_esit(aile, zt, zh, kok=URIAL_KOK)
    if esit is None:
        return "DIZE-CÖZÜLEMEDI", dict(n=0, farkli=0)
    print(f"{aile} {n} {farkli}"
          f"", flush=True)
    if not esit:
        print(f"      base : {oa[:110]!r}\n      inst : {ob[:110]!r}")
    return ("DIZE-TAM" if esit else "DIZE-DÜSTÜ"), dict(n=n, farkli=farkli)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--kuru", action="store_true")
    ap.add_argument("--liste", default="tek", choices=["tek", "cift"])
    ap.add_argument("--sadece", default=None, help="")
    ap.add_argument("--butce-dk", dest="butce", type=float, default=1e9)
    ap.add_argument("--coklu-card", dest="coklu", type=int, default=0)
    ap.add_argument("--zemin", default="ikisi", choices=["ikisi", "base", "instruct"])
    a = ap.parse_args()
    ham = istem_yukle()

    havuz = {k["ad"].split("·")[0]: dict(k)
             for k in (S24.KOLLAR + SG.KOLLAR + cift_kollar())}
    adlar = TEK_CARD if a.liste == "tek" else CIFT_CARD
    if a.sadece:
        sec = [x.strip() for x in a.sadece.split(",") if x.strip()]
        yok = [x for x in sec if x not in adlar]
        if yok:
            raise SystemExit(f"{a.liste} {yok}")
        adlar = sec
    eksik = [x for x in adlar if x not in havuz]
    if eksik:
        raise SystemExit(f"★ ithal kol tablolarinda YOK: {eksik} ⇒ kosu YOK")

    SG.ONEK_KURUCU = urial_onek_fn_kurucu(ham)
    SG.CIKTI_KOK = URIAL_KOK
    OUT, t0, ilk = {}, time.time(), None
    print(f"★ URIAL · liste={a.liste} · kol={len(adlar)} · bütce={a.butce:.0f} dk "
          f"· T=0.9 top_p=0.95 (ölcüldü) · kuru={a.kuru}", flush=True)
    durdu = False
    for n, ad in enumerate(adlar, 1):
        if os.path.exists(DUR):
            print(f"{ad}", flush=True)
            OUT["DUR"] = dict(HAL="DURDURULDU", sinir="aile", ad=ad); break
        k = havuz[ad]
        SG.KOLLAR = [k]
        for zemin in (("base", "instruct") if a.zemin == "ikisi" else (a.zemin,)):
            if os.path.exists(DUR):
                print(f"  ★★ DUR · bacak siniri: {ad}/{zemin} BASLAMADI", flush=True)
                OUT["DUR"] = dict(HAL="DURDURULDU", sinir="bacak", ad=f"{ad}/{zemin}")
                durdu = True; break
            hedef = k["taban"] if zemin == "base" else k["hizali"]
            ok, m = icerik(hedef)
            if ok:
                print(f"{hedef} {m}")
                OUT[f"{ad}/{zemin}"] = dict(HAL="ATLANDI-ZATEN-VAR", icerik=m); continue
            gec = (time.time() - t0) / 60
            bek, bek_temel = beklenen_dk(ad, k)
            if gec + bek > a.butce:
                print(f"  ★★ {ad}/{zemin}: BÜTCE · {gec:.0f}+{bek:.0f} > {a.butce:.0f} dk "
                      f"(temel: {bek_temel}) ⇒ EYLEM: sonraki pencereye, ADIYLA")
                OUT[f"{ad}/{zemin}"] = dict(HAL="ATLANDI-BÜTCE"); continue
            gec_v, mes_v = vram_kapisi(ad, int(str(a.dev).split(":")[-1])
                                       if ":" in str(a.dev) else 0)
            print(f"  [VRAM] {ad}/{zemin}: {mes_v}", flush=True)
            if not gec_v:
                OUT[f"{ad}/{zemin}"] = dict(HAL="ATLANDI-VRAM", mesaj=mes_v); continue
            if a.kuru:
                print(f"  ⤿ {ad}/{zemin}: KURU · hedef {URIAL_KOK}/{hedef} · beklenen {bek:.0f} dk")
                OUT[f"{ad}/{zemin}"] = dict(HAL="KURU", beklenen_dk=bek); continue
            t1 = time.time()
            ns = argparse.Namespace(kol=k["ad"], zemin=zemin, dev=a.dev,
                                    bekle_gpu=None, coklu_card=a.coklu)
            try:
                rc = SG.uret(ns)
            except SystemExit as e:
                print(f"  ✗ {ad}/{zemin}: KAPI {e}")
                OUT[f"{ad}/{zemin}"] = dict(HAL="KAPI", mesaj=str(e)); continue
            except Exception as e:
                print(f"  ✗ {ad}/{zemin} ARIZA: {type(e).__name__}: {e}")
                OUT[f"{ad}/{zemin}"] = dict(HAL="ARIZA", hata=f"{type(e).__name__}: {e}"); continue
            dk = (time.time() - t1) / 60
            ok2, m2 = icerik(hedef)
            hal = ("SÜPHELI-HIZLI" if dk < 0.25 * bek else ("BITTI" if ok2 else "ICERIK-DÜSTÜ"))
            OUT[f"{ad}/{zemin}"] = dict(HAL=hal, dk=round(dk, 1), rc=rc, icerik=m2,
                                        bek_dk=round(bek, 1), bek_temel=bek_temel)
            if hal == "BITTI":
                URIAL_DK[ad] = round(dk, 1)
            if ilk is None:
                ilk = dk
                print(f"  ★ [W-71] ilk bacak {dk:.1f} dk ⇒ kalan ≈ {dk*(2*len(adlar)-n):.0f} dk "
                      f"⇒ esik {a.butce:.0f} dk", flush=True)
            print(f"  [{n}/{len(adlar)}] {ad}/{zemin} · {hal} · {dk:.1f} dk · {m2}", flush=True)
        zt, zh = k["taban"].split("/")[-1], k["hizali"].split("/")[-1]
        if not a.kuru and os.path.exists(f"{URIAL_KOK}/{ad}/{zt}/uretim.jsonl"):
            dh, dp = dize_kapisi(ad, zt, zh)
            OUT[f"{ad}/DIZE"] = dict(HAL=dh, **dp)
        json.dump(dict(_kunye=dict(
            damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            alet="scripts/urial_run.py", borc="D-0908-URIAL §1 (sahip kelimesi)",
            istem=dict(repo="github.com/Re-Align/URIAL", commit=COMMIT,
                       dosya="urial_prompts/inst_1k.txt", sha256=SHA, K=3),
            ornek_M1=dict(tek=[53.19, 13.27, 6.80], ucu_birden=16.29,
                          serh="panelin ciplak hizali bandinin (2,90–9,80) ÜSTÜNDE"),
            cozumleme=dict(T=0.9, top_p=0.95, kaynak="onek_uretim_pilot (ölcüldü)"),
            kirpma=dict(max_length=SA.MAX_ISTEM_JETON, n_kirpilan=0,
                        kapi="ön-ucus: max(istem_jeton) ≤ max_length ⇒ asilirsa KOL KOSMAZ"),
            kok=URIAL_KOK, gecen_dk=round((time.time()-t0)/60, 1)), kol=OUT),
            io.open(CIK.replace(".json", f"_{a.liste}.json"), "w", encoding="utf-8"),
            ensure_ascii=False, indent=1)
        if durdu:
            break
    b = sum(1 for v in OUT.values() if v["HAL"] == "BITTI")
    ar = sum(1 for v in OUT.values() if v["HAL"] in
             ("ARIZA", "KAPI", "ICERIK-DÜSTÜ", "DIZE-DÜSTÜ"))
    coz = sum(1 for v in OUT.values() if v["HAL"] == "DIZE-CÖZÜLEMEDI")
    at = sum(1 for v in OUT.values() if v["HAL"].startswith("ATLANDI"))
    vr = sum(1 for v in OUT.values() if v["HAL"] == "ATLANDI-VRAM")
    print(f"{a.liste} {2*len(adlar)} {b} {at}"
          f"{ar} {coz} {vr}"
          f"")
    print(f"✓ {CIK.replace('.json', f'_{a.liste}.json')}")
    return 3 if ar else 0


if __name__ == "__main__":
    sys.exit(main())
