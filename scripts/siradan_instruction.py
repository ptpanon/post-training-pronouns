#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, io, json, os, re, shutil, subprocess, sys, time

os.environ.setdefault("PROJECT_ISTEM_MAXLEN", "2048")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
PREREG = f"{ROOT}/preregistration/prereg_siradan_instruction_2026-09-17.md"
KOK = os.environ.get("SIRADAN_KOK", __DNH_DATA__ + "/siradan_talimat")
ISTEM_DOSYA = f"{KOK}/istemler.jsonl"
SECIM_CARD = f"{ROOT}/results/siradan_instruction_secim_2026-09-17.json"
OKUMA_CARD = f"{ROOT}/results/siradan_instruction_reading_2026-09-17.json"
ALPACA = ("<storage>/huggingface/hub/datasets--tatsu-lab--alpaca/snapshots/dce01c9b08f87459cf36a430d809084718273017/"
          "unreleased/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
N_SINIF, N_CEKIM, N_BEK = 75, 4, 300 * 4
DISK_ESIK_GB, BUTCE_DK = 80, 480
CARD_BOS_MIB = 2000
BANT_ESIK_SATIR = 80

SINIFLAR = [
    ("OZET", r"^(summari[sz]e|write a summary|provide a summary|give a summary)\b"),
    ("SINIFLAMA", r"^(classify|categori[sz]e|identify (whether|if|the type|the category)|is (this|the following) (a|an)\b)"),
    ("NASIL", r"^(how (to|does|do|can|is|are)|what are the steps|steps to|outline the steps)\b"),
    ("ACIKLAMA", r"^(explain|describe|what (is|are) (a |an |the )?[a-z]|why (do|does|is|are|did))\b"),
]
MUHATAP_TUR = re.compile(r"\b(letter|e-?mail|message|dialog(ue)?|conversation|speech|tweet|reply|respond|advice|advise|"
                         r"recommend|persuade|convince|greet|apolog\w*|thank)\b", re.I)
KELIME_TAVAN = 150


def sha16s(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
def sha16f(y): return hashlib.sha256(open(y, "rb").read()).hexdigest()[:16]
def damga(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sinif_bul(talimat):
    t = talimat.strip().lower()
    for ad, p in SINIFLAR:
        if re.match(p, t):
            return ad
    return None


def sec():
    import pyarrow.parquet as pq, addressee_run as MK, form_count as FS
    from reading_style_object import payda
    T = pq.read_table(ALPACA).to_pylist()
    red = dict(sinifsiz=0, kod_blogu=0, kelime=0, muhatap_turu=0, kisi=0, yinelenen=0)
    aday, gorulen = [], set()
    for i, r in enumerate(T):
        s = sinif_bul(r["instruction"])
        if not s:
            red["sinifsiz"] += 1; continue
        metin = r["instruction"].strip() + (("\n\n" + r["input"].strip()) if r["input"].strip() else "")
        if "```" in metin:
            red["kod_blogu"] += 1; continue
        if len(metin.split()) > KELIME_TAVAN:
            red["kelime"] += 1; continue
        if MUHATAP_TUR.search(r["instruction"]):
            red["muhatap_turu"] += 1; continue
        if metin in gorulen:
            red["yinelenen"] += 1; continue
        gorulen.add(metin); aday.append(dict(satir=i, sinif=s, metin=metin))
    V = MK.satir_bilesenleri(FS._boru(), aday)
    kisi = V[:, MK.ALAN.index("m5_yakin")] + V[:, MK.ALAN.index("m1_sahis2")]
    temiz = [a for a, k in zip(aday, kisi) if k == 0]
    red["kisi"] = len(aday) - len(temiz)
    secim, sayim = [], {}
    for ad, _ in SINIFLAR:
        havuz_ = sorted((a for a in temiz if a["sinif"] == ad), key=lambda a: sha16s(a["metin"]) + f"{a['satir']:06d}")
        sayim[ad] = len(havuz_)
        if len(havuz_) < N_SINIF:
            raise SystemExit(f"★ {ad}: aday {len(havuz_)} < {N_SINIF} ⇒ secim YOK")
        for j, a in enumerate(havuz_[:N_SINIF]):
            secim.append(dict(istem_id=f"{ad}-{j:02d}", **a))
    os.makedirs(KOK, exist_ok=True)
    with io.open(ISTEM_DOSYA, "w", encoding="utf-8") as fh:
        for a in secim:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    json.dump(dict(sinif="", prereg=os.path.relpath(PREREG, ROOT),
                   kaynak=dict(set="tatsu-lab/alpaca", lisans="CC BY-NC 4.0", snapshot="dce01c9b08f87459cf36a430d809084718273017",
                               parquet_sha256_16=sha16f(ALPACA), n_satir=len(T)),
                   rule=dict(siniflar=SINIFLAR, muhatap_turu=MUHATAP_TUR.pattern, kelime_tavani=KELIME_TAVAN,
                              kisi="muhatap_kos m5_yakin + m1_sahis2 == 0", sira="sha256(metin) artan", n_sinif=N_SINIF),
                   red=red, aday_temiz=sayim, istem_dosyasi=dict(yol=ISTEM_DOSYA, sha256_16=sha16f(ISTEM_DOSYA)),
                   istemler=[dict(istem_id=a["istem_id"], satir=a["satir"], sha16=sha16s(a["metin"]),
                                  n_kelime=len(a["metin"].split())) for a in secim], damga_utc=damga()),
              io.open(SECIM_CARD, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("siradan_secim", n_satir=len(T), n_secim=len(secim), hal_red_kisi=red["kisi"], hal_red_muhatap_turu=red["muhatap_turu"],
          bekle={"n_secim": 4 * N_SINIF})
    print(f"✓ {os.path.relpath(SECIM_CARD, ROOT)} · sha {sha16f(SECIM_CARD)} · istemler {sha16f(ISTEM_DOSYA)} · aday {sayim} · red {red}")
    return 0


def istemler():
    R = [json.loads(l) for l in io.open(ISTEM_DOSYA, encoding="utf-8")]
    k = json.load(io.open(SECIM_CARD, encoding="utf-8"))["istem_dosyasi"]["sha256_16"]
    if sha16f(ISTEM_DOSYA) != k or len(R) != 4 * N_SINIF:
        raise SystemExit(f"{sha16f(ISTEM_DOSYA)} {k} {len(R)}")
    return [dict(debate=r["sinif"], tez=r["istem_id"], durus="-", metin=r["metin"]) for r in R]


def urial_kurucu(ham):
    def kurucu(tok):
        def f(ad, tur, c, ofs, i, ist, cekim):
            return f"{ham}\n\n\n# Query:\n```{ist['metin']}```\n \n# Answer:\n```\n"
        return f
    return kurucu


def template_kurucu(tok):
    def f(ad, tur, c, ofs, i, ist, cekim):
        return tok.apply_chat_template([{"role": "user", "content": ist["metin"]}], tokenize=False, add_generation_prompt=True)
    return f


def havuz():
    import template_robustness_24 as S24, template_robustness as SG, urial_run as UK
    return {k["ad"].split("·")[0]: dict(k) for k in (S24.KOLLAR + SG.KOLLAR + UK.cift_kollar())}


def aile_sirasi():
    import urial_run as UK
    return list(UK.TEK_CARD) + list(UK.CIFT_CARD)


def smi():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in q.strip().splitlines())}


def kendi_mib():
    u = dict(l.split(", ") for l in subprocess.run(["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"],
                                                  capture_output=True, text=True, check=True).stdout.strip().splitlines())
    out = {}
    for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout.strip().splitlines():
        if not l.strip():
            continue
        pid, uuid, mib = [x.strip() for x in l.split(",")]
        if int(pid) == os.getpid() and uuid in u:
            out[int(u[uuid])] = out.get(int(u[uuid]), 0) + int(mib)
    return out


def dur_sartlari(fiziksel, t_asama):
    import gc
    import mini_dpo_feasibility as FZ
    gc.collect()
    try:
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass
    bos_gb = shutil.disk_usage("<storage>").free / 2**30
    if bos_gb < DISK_ESIK_GB:
        return False, f"disk {bos_gb:.0f} GB < {DISK_ESIK_GB}"
    kendi = kendi_mib()
    kul = {k: v - kendi.get(k, 0) for k, v in smi().items()}
    dolu = {k: kul.get(k) for k in fiziksel if kul.get(k, 99999) >= CARD_BOS_MIB}
    if dolu:
        return False, f"card dolu {dolu} MiB (esik {CARD_BOS_MIB})"
    rej, sag, _ = FZ.prod_rejimi(fiziksel[0])
    if rej != "PROD-YOK":
        return False, f"prod rejimi {rej} {sag} (sahip: prod kapali olmali)"
    gec = (time.time() - t_asama) / 60
    if gec > BUTCE_DK:
        return False, f"asama bütcesi {gec:.0f} > {BUTCE_DK} dk"
    return True, f"disk {bos_gb:.0f} GB · kartlar {[kul.get(k) for k in fiziksel]} MiB · prod {rej} · asama {gec:.0f}/{BUTCE_DK} dk"


def bacak_tam(dizin):
    y, k = f"{KOK}/{dizin}/uretim.jsonl", f"{KOK}/{dizin}/uretim_kunye.json"
    if not (os.path.exists(y) and os.path.exists(k)):
        return False
    n = sum(1 for _ in io.open(y, encoding="utf-8"))
    return n == N_BEK and json.load(io.open(k, encoding="utf-8")).get("TAM") is True


def prereg_kapisi():
    if not os.path.exists(PREREG):
        raise SystemExit("")
    h = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "-1", "--", os.path.relpath(PREREG, ROOT)],
                       capture_output=True, text=True).stdout.strip()
    if not h:
        raise SystemExit("")
    b = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    if b not in open(PREREG, encoding="utf-8").read():
        raise SystemExit(f"{b[:16]}")
    return h, b[:16]


def kuru():
    from transformers import AutoTokenizer
    import urial_run as UK, serita_harman102 as SA
    ham = UK.istem_yukle(); ist = istemler(); H = havuz(); out = {}
    for ad in aile_sirasi():
        k = H[ad]
        tok = AutoTokenizer.from_pretrained(SA._snapshot(None, model_adi=k["s_hizali"]))
        for zemin, fn in (("base", urial_kurucu(ham)(tok)), ("instruct", template_kurucu(tok))):
            p = [fn("SIRADAN", "yok", None, 0, i, s, 0) for i, s in enumerate(ist)]
            n = [len(x) for x in tok(p, truncation=False)["input_ids"]]
            out[f"{ad}/{zemin}"] = dict(tepe=max(n), ort=round(float(np.mean(n)), 1), kirpilan=sum(x > SA.MAX_ISTEM_JETON for x in n))
            print(f"  {ad:16s} {zemin:8s} tepe {max(n):5d} · ort {np.mean(n):7.1f} · kirpilan {out[f'{ad}/{zemin}']['kirpilan']} (max {SA.MAX_ISTEM_JETON})", flush=True)
    bad = [k for k, v in out.items() if v["kirpilan"]]
    print(f"[PAYDA] siradan_kuru: n_bacak={len(out)} · red_kirpilan_bacak={len(bad)} {bad}")
    return 3 if bad else 0


def uret(a):
    import gpu_lock as RKG
    import family_panel as C, template_robustness as SG, urial_run as UK
    from reading_style_object import payda
    h, b = prereg_kapisi()
    ham = UK.istem_yukle(); IST = istemler(); H = havuz()
    fiz = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]
    t_asama = float(os.environ.get("SIRADAN_ASAMA_T0", time.time()))
    C.A3.istemler = lambda: IST
    C.K1BU.kollar = lambda ton=True: [("SIRADAN", "yok", None, 0)]
    C.K1BU.k0a_yuva = lambda K: None
    C.N_CEKIM = N_CEKIM
    SG.CIKTI_KOK = KOK
    _payda = C.payda

    def _payda_kanca(etiket, bekle=None, **kw):
        if etiket.startswith("c1_uretim_"):
            bekle = {"n_kol": 1, "n_satir": N_BEK}
        return _payda(etiket, bekle=bekle, **kw)
    C.payda = _payda_kanca
    print(f"★ SIRADAN · serit {a.serit} · prereg {h} · alet {b} · kartlar {fiz} · aileler {a.aileler} · T=0.9 top_p=0.95 · "
          f"yeni jeton {C.A3.YENI_JETON} · pencere {os.environ['PROJECT_ISTEM_MAXLEN']}", flush=True)
    t_ilk = None; n_bitti = 0; adlar = a.aileler.split(",")
    for n, ad in enumerate(adlar, 1):
        k = H[ad]; SG.KOLLAR = [k]
        for zemin in ("base", "instruct"):
            hedef = k["taban"] if zemin == "base" else k["hizali"]
            if bacak_tam(hedef):
                print(f"{hedef} {N_BEK}", flush=True); continue
            gec, sebep = dur_sartlari(fiz, t_asama)
            print(f"  [DUR-SARTLARI] {ad}/{zemin}: {sebep} ⇒ esik disk≥{DISK_ESIK_GB}GB · card<{CARD_BOS_MIB}MiB · PROD-YOK · "
                  f"≤{BUTCE_DK}dk ⇒ EYLEM: tutmazsa serit DURUR", flush=True)
            if not gec:
                payda(f"siradan_dur_{a.serit}", n_bacak_denenen=1, hal_dur=1)
                return 5
            SG.ONEK_KURUCU = urial_kurucu(ham) if zemin == "base" else template_kurucu
            t1 = time.time()
            try:
                rc = SG.uret(argparse.Namespace(kol=k["ad"], zemin=zemin, dev=a.dev, bekle_gpu=RKG.fiziksel_bekle(a.dev),
                                                coklu_card=a.coklu_card))
            except BaseException as e:
                print(f"{ad} {zemin} {type(e).__name__} {e}", flush=True)
                return 4
            dk = (time.time() - t1) / 60
            if rc != 0 or not bacak_tam(hedef):
                print(f"  ✗✗ {ad}/{zemin}: rc={rc} · TAM={bacak_tam(hedef)} ⇒ serit DURDU", flush=True)
                return 4
            n_bitti += 1
            if t_ilk is None:
                t_ilk = dk
                kalan = 2 * len(adlar) - (2 * (n - 1) + (1 if zemin == "base" else 2))
                print(f"{dk:.1f} {dk*kalan:.0f}"
                      f"{BUTCE_DK}", flush=True)
            print(f"  [{n}/{len(adlar)}] {ad}/{zemin} BITTI · {dk:.1f} dk", flush=True)
    payda(f"siradan_serit_{a.serit}", n_aile=len(adlar), hal_bitti=n_bitti)
    return 0


KES = re.compile(r"```|\n#\s*Query:")


def taban_kes(t):
    m = KES.search(t or "")
    return (t[:m.start()], True) if m else (t or "", False)


def _prova_okuma():
    assert taban_kes("A cat.\n```\n\n\n# Query:\n```you?```") == ("A cat.\n", True)
    assert taban_kes("no end") == ("no end", False)
    assert taban_kes("x\n# Query:\nI") == ("x", True)
    return True


def oku():
    import two_rulers_2x2 as P2, form_count as FS, urial_verdict as UH, continuation_mode_exit as Q2
    from reading_style_object import payda
    _prova_okuma(); t0 = time.time(); nlp = FS._boru(); H = havuz()
    ist_ad = {i: r["tez"] for i, r in enumerate(istemler())}
    fM1, fS1 = P2.OLCU["M1_1k"][0], P2.OLCU["SAHIS1_1k"][0]
    satir, eksik = {}, []
    kipM1 = {b: {k: {} for k in ("RET", "ASISTAN", "META", "DEVAM")} for b in ("taban", "hizali")}
    for ad in aile_sirasi():
        k = H[ad]
        if not (bacak_tam(k["taban"]) and bacak_tam(k["hizali"])):
            eksik.append(ad); satir[ad] = dict(hal="BACAK-EKSIK"); continue
        Rt = [json.loads(l) for l in io.open(f"{KOK}/{k['taban']}/uretim.jsonl", encoding="utf-8")]
        Rh = [json.loads(l) for l in io.open(f"{KOK}/{k['hizali']}/uretim.jsonl", encoding="utf-8")]
        n_kes = 0
        for r in Rt:
            r["metin"], kes = taban_kes(r["metin"]); n_kes += kes
        i0, i1 = P2.cift_al(Rt, Rh)
        Rt = [Rt[x] for x in i0]; Rh = [Rh[y] for y in i1]
        tut = UH.dshelf(Rt) & UH.dshelf(Rh)
        P0 = P2.bilesen(nlp, Rt)[tut]; P1 = P2.bilesen(nlp, Rh)[tut]
        ist = np.array([r["istem_i"] for r in Rh])[tut]
        m1, s1 = P2.olc(P0, P1, ist, fM1), P2.olc(P0, P1, ist, fS1)
        kt = np.array([Q2.sinif(r["metin"]) for r in Rt])[tut]; kh = np.array([Q2.sinif(r["metin"]) for r in Rh])[tut]
        kip = {}
        for bac, KK, PP in (("taban", kt, P0), ("hizali", kh, P1)):
            kip[bac] = {}
            for c in ("RET", "ASISTAN", "META", "DEVAM"):
                m = KK == c
                kip[bac][c] = dict(n=int(m.sum()), pay=round(float(m.mean()), 4), M1=round(float(fM1(PP[m])), 4) if m.sum() else None)
                kipM1[bac][c][ad] = (kip[bac][c]["M1"], int(m.sum()))
        sinif = {}
        for s in ("ACIKLAMA", "NASIL", "SINIFLAMA", "OZET"):
            m = np.array([ist_ad[int(x)].startswith(s) for x in ist])
            sinif[s] = dict(n=int(m.sum()), M1_taban=round(float(fM1(P0[m])), 4), M1_hizali=round(float(fM1(P1[m])), 4),
                            S1_taban=round(float(fS1(P0[m])), 4), S1_hizali=round(float(fS1(P1[m])), 4))
        yuv = lambda d: {x: (round(v, 4) if isinstance(v, float) else v) for x, v in d.items()}
        satir[ad] = dict(hal="OKUNDU", n_cift=len(i0), n_tutulan=int(tut.sum()), taban_kesilen=n_kes,
                         taban_kesilen_pay=round(n_kes / N_BEK, 4), M1=yuv(m1), SAHIS1=yuv(s1),
                         M1_sinif3=("asagi" if m1["ci"][1] < 0 else "yukari" if m1["ci"][0] > 0 else "null"),
                         S1_sinif3=("asagi" if s1["ci"][1] < 0 else "yukari" if s1["ci"][0] > 0 else "null"),
                         kip=kip, sinif=sinif)
        print(f"  [{(time.time()-t0)/60:4.1f} dk] {ad:16s} tut {int(tut.sum())}/{len(i0)} · kesilen {n_kes} · "
              f"M1 {m1['taban']:.2f}→{m1['taban']+m1['gozlenen']:.2f} Δ {m1['gozlenen']:+.2f} [{m1['ci'][0]:+.2f},{m1['ci'][1]:+.2f}] · "
              f"Δ1st {s1['gozlenen']:+.2f}", flush=True)
    ok = {a: v for a, v in satir.items() if v["hal"] == "OKUNDU"}

    def bant(d):
        v = {x: y for x, y in d.items() if y is not None}
        if len(v) < 2:
            return dict(n=len(v), hal="BANT-KURULAMADI")
        lo, hi = min(v, key=v.get), max(v, key=v.get)
        return dict(min=round(v[lo], 4), min_aile=lo, max=round(v[hi], 4), max_aile=hi,
                    oran=round(v[hi] / v[lo], 3) if v[lo] > 0 else None, n=len(v))
    bantlar = dict(taban=bant({a: v["M1"]["taban"] for a, v in ok.items()}),
                   hizali=bant({a: v["M1"]["taban"] + v["M1"]["gozlenen"] for a, v in ok.items()}),
                   kip_ici={b: {c: bant({a: m for a, (m, nn) in kipM1[b][c].items() if nn >= BANT_ESIK_SATIR}) for c in kipM1[b]} for b in kipM1},
                   sinif_hizali={s: bant({a: v["sinif"][s]["M1_hizali"] for a, v in ok.items()}) for s in ("ACIKLAMA", "NASIL", "SINIFLAMA", "OZET")})
    say = lambda f: {c: sorted(a for a, v in ok.items() if v[f] == c) for c in ("asagi", "yukari", "null")}
    json.dump(dict(sinif="BETIM · bar yok · prediction yok", prereg=os.path.relpath(PREREG, ROOT), alet_sha256_16=sha16f(os.path.abspath(__file__)),
                   secim_karti=dict(yol=os.path.relpath(SECIM_CARD, ROOT), sha256_16=sha16f(SECIM_CARD)),
                   motor=""
                         "",
                   n_aile=16, n_okunan=len(ok), bacak_eksik=eksik, aile=satir, bant=bantlar,
                   sayim=dict(M1=say("M1_sinif3"), SAHIS1=say("S1_sinif3")), bant_esik_satir=BANT_ESIK_SATIR,
                   damga_utc=damga(), sure_dk=round((time.time() - t0) / 60, 1)),
              io.open(OKUMA_CARD, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("siradan_oku", n_aile=16, hal_okunan=len(ok), hal_bacak_eksik=len(eksik))
    print(f"✓ {os.path.relpath(OKUMA_CARD, ROOT)} · sha {sha16f(OKUMA_CARD)} · bant hizali {bantlar['hizali']} · taban {bantlar['taban']}")
    return 0


if __name__ == "__main__":
    kip = sys.argv[1]
    if kip == "sec": sys.exit(sec())
    if kip == "kuru": sys.exit(kuru())
    if kip == "oku": sys.exit(oku())
    if kip == "uret":
        ap = argparse.ArgumentParser(); ap.add_argument("kip"); ap.add_argument("--aileler", required=True)
        ap.add_argument("--dev", default="cuda:0"); ap.add_argument("--coklu-card", dest="coklu_card", type=int, default=0)
        ap.add_argument("--serit", default="0")
        sys.exit(uret(ap.parse_args()))
    sys.exit(f"bilinmeyen kip {kip}")
