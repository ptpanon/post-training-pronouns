#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, io, json, os, subprocess, sys, time

os.environ.setdefault("PROJECT_ISTEM_MAXLEN", "2048")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
PREREG = f"{ROOT}/preregistration/prereg_v97_q4_turn_2026-09-20.md"
KOK = __DNH_DATA__ + "/v97_q4"
DUR = f"{KOK}/DUR"
N_CEKIM = 12
TURLER = [("TUR_BLOG", "[Personal blog post]"),
          ("TUR_FORUM", "[Reply in a discussion forum]"),
          ("TUR_NASIL", "[How-to guide]"),
          ("TUR_NOTICE", "[News report]")]
N_BEK = len(TURLER) * 34 * N_CEKIM


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


def onek_kurucu():
    import a3_harman_oran as A3
    bas = dict(TURLER)

    def kurucu(tok):
        def f(ad, tur, c, ofs, i, ist, cekim):
            return f"{bas[ad]}\n\n" + A3.CERCEVE_NOTR.format(T=ist["tez"], P=A3.DURUS[ist["durus"]])
        return f
    return kurucu


def bacak_tam(dizin):
    y, k = f"{KOK}/{dizin}/uretim.jsonl", f"{KOK}/{dizin}/uretim_kunye.json"
    if not (os.path.exists(y) and os.path.exists(k)):
        return False
    try:
        return sum(1 for _ in io.open(y, encoding="utf-8")) == N_BEK and json.load(io.open(k, encoding="utf-8")).get("TAM") is True
    except Exception:
        return False


def uret(a):
    import gpu_lock as RKG
    import family_panel as C, template_robustness as SG, siradan_instruction as ST, a3_harman_oran as A3
    from reading_style_object import payda
    h, b = prereg_kapisi()
    os.makedirs(KOK, exist_ok=True)
    H = ST.havuz()
    fiz = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]
    t_asama = float(os.environ.get("V97_Q4_ASAMA_T0", time.time()))
    C.A3.istemler = A3.istemler
    C.K1BU.kollar = lambda ton=True: [(ad, "yok", None, 0) for ad, _ in TURLER]
    C.K1BU.k0a_yuva = lambda K: None
    C.N_CEKIM = N_CEKIM
    SG.CIKTI_KOK = KOK
    SG.ONEK_KURUCU = onek_kurucu()
    _payda = C.payda

    def _payda_kanca(etiket, bekle=None, **kw):
        if etiket.startswith("c1_uretim_"):
            bekle = {"n_kol": len(TURLER), "n_satir": N_BEK}
        return _payda(etiket, bekle=bekle, **kw)
    C.payda = _payda_kanca
    print(f"{a.serit} {h} {b} {fiz} {[x for x, _ in TURLER]} {a.aileler}"
          f"{C.A3.YENI_JETON} {N_CEKIM} {N_BEK} {KOK}", flush=True)
    adlar = a.aileler.split(","); n_bitti = 0; t_ilk = None
    for n, ad in enumerate(adlar, 1):
        k = H[ad]; SG.KOLLAR = [k]
        for zemin in ("base", "instruct"):
            hedef = k["taban"] if zemin == "base" else k["hizali"]
            if bacak_tam(hedef):
                print(f"  ⤿ {hedef}: TAM ⇒ ATLANDI", flush=True); continue
            if os.path.exists(DUR):
                print(f"{DUR}", flush=True)
                payda(f"v97_q4_dur_{a.serit}", n_bacak_denenen=1, hal_dur=1); return 5
            gec, sebep = ST.dur_sartlari(fiz, t_asama)
            print(f"  [DUR-SARTLARI] {ad}/{zemin}: {sebep} ⇒ esik disk≥{ST.DISK_ESIK_GB}GB · card<{ST.CARD_BOS_MIB}MiB · PROD-YOK · "
                  f"≤{ST.BUTCE_DK}dk ⇒ EYLEM: tutmazsa serit DURUR", flush=True)
            if not gec:
                payda(f"v97_q4_dur_{a.serit}", n_bacak_denenen=1, hal_dur=1); return 5
            t1 = time.time()
            try:
                rc = SG.uret(argparse.Namespace(kol=k["ad"], zemin=zemin, dev=a.dev, bekle_gpu=RKG.fiziksel_bekle(a.dev),
                                                coklu_card=a.coklu_card))
            except BaseException as e:
                print(f"{ad} {zemin} {type(e).__name__} {e}", flush=True)
                return 4
            dk = (time.time() - t1) / 60
            if rc != 0 or not bacak_tam(hedef):
                print(f"  ✗✗ {ad}/{zemin}: rc={rc} · TAM={bacak_tam(hedef)} ⇒ serit DURDU", flush=True); return 4
            n_bitti += 1
            if t_ilk is None:
                t_ilk = dk
                kalan = 2 * len(adlar) - 1
                print(f"  ★ [W-71] ilk bacak {dk:.1f} dk ⇒ bu seritte kalan ≈ {dk*kalan:.0f} dk ⇒ esik {ST.BUTCE_DK} dk ⇒ "
                      f"EYLEM: asilirsa sonraki bacak baslamaz", flush=True)
            if dk < 0.25 * 3.0:
                print(f"  ⚠ SÜPHELI-HIZLI {ad}/{zemin}: {dk:.2f} dk ⇒ bir bakis ister", flush=True)
            print(f"  [{n}/{len(adlar)}] {ad}/{zemin} BITTI · {dk:.1f} dk", flush=True)
    payda(f"v97_q4_serit_{a.serit}", n_aile=len(adlar), hal_bitti=n_bitti)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kip"); ap.add_argument("--aileler", required=True)
    ap.add_argument("--dev", default="cuda:0"); ap.add_argument("--coklu-card", dest="coklu_card", type=int, default=0)
    ap.add_argument("--serit", default="0")
    a = ap.parse_args()
    sys.exit(uret(a) if a.kip == "uret" else f"bilinmeyen kip {a.kip}")
