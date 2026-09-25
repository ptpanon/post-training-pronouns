#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, re, sys
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import filtre_verim as FV
from reading_style_object import payda
CIK = f"{ROOT}/results/person_dial_minimal_pair_feasibility_2026-09-04.json"

RULE = [
    ("youare",  re.compile(r"\byou are\b", re.I),           "one is"),
    ("youwill", re.compile(r"\byou will\b", re.I),          "one will"),
    ("need",    re.compile(r"\byou need to\b", re.I),       "it is necessary to"),
    ("must",    re.compile(r"\byou must\b", re.I),          "it is required to"),
    ("should",  re.compile(r"\byou should\b", re.I),        "one should"),
    ("can",     re.compile(r"\byou can\b", re.I),           "it is possible to"),
    ("may",     re.compile(r"\byou may\b", re.I),           "one may"),
    ("if",      re.compile(r"\bif you\b", re.I),            "if one"),
    ("when",    re.compile(r"\bwhen you\b", re.I),          "when one"),
    ("your",    re.compile(r"\byour\b", re.I),              "the"),
]
ASGARI = 1



def takaslar(t):
    n, kar = 0, 0
    for _, rx, yeni in RULE:
        for m in rx.finditer(t or ""):
            n += 1; kar += len(m.group(0))
    return n, kar


def uygula(t):
    n = 0
    for _, rx, yeni_s in RULE:
        def _d(m, y=yeni_s):
            return (y[0].upper() + y[1:]) if m.group(0)[:1].isupper() else y
        t, k = rx.subn(_d, t or "")
        n += k
    return t, n


def uygula_prova():
    a = "You should check your settings if you are unsure."
    b = "The weather is fine today."
    ya, na = uygula(a); yb, nb = uygula(b)
    sa, _ = takaslar(a); sb, _ = takaslar(b)
    ortusme = sa - na
    KIRIK = re.compile(r"\bone (are|have|were|do)\b", re.I)
    kirik = bool(KIRIK.search(ya)) or bool(KIRIK.search(yb))
    imla = ya[:1].isupper()
    ok = (nb == sb == 0 and yb == b and "you" not in ya.lower()
          and not kirik and ortusme >= 0 and imla)
    payda("kk2_uygula_prova", n_sinav=5, hal_takas=na, hal_sayim=sa,
          hal_ortusme=ortusme, hal_degismedi=int(yb == b),
          red_kalan_you=int("you" in ya.lower()), red_dilbilgisi=int(kirik),
          red_imla=int(not imla))
    print(f"{a} {ya} {na} {sa}"
          f"{ortusme} {yb == b}"
          f"{kirik} {imla}"
          f""
          f"", flush=True)
    return ok


def degisim_payi(t):
    n, kar = takaslar(t)
    return (kar / len(t)) if t else float("nan")


def uygun(t):
    n, _ = takaslar(t)
    return (n >= ASGARI and bool(t)), n


def prova():
    R = {
        "i_kisili_yakalanir": uygun("You should check your logs when you can.")[0],
        "ii_kisisiz_gecmez":  not uygun("The logs should be checked periodically.")[0],
        "iii_bos_gecmez":     not uygun("")[0],
        "iv_degisim_payi":    abs(degisim_payi("your dog") - 4/8) < 1e-9,
        "v_sayim_dogru":      takaslar("you can and you can")[0] == 2,
    }
    print(f"★ §7.1 PROVA: {json.dumps(R, ensure_ascii=False)}")
    ok = all(R.values())
    print(f"   ⇒ {'GECTI' if ok else 'DÜSTÜ'} ⇒ EYLEM: düserse cikis 4, sayi YAZILMAZ")
    return ok


def main():
    if not prova():
        sys.exit(4)
    S = {}
    from datasets import load_dataset
    ds = load_dataset("allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train")
    M = [FV._son_asistan_mesaj(x) for x in ds["chosen"]] + \
        [FV._son_asistan_mesaj(x) for x in ds["rejected"]]
    M = [t for t in M if t]
    u = [uygun(t) for t in M]
    n_uy = sum(1 for a, _ in u if a)
    tk = [b for a, b in u if a]
    dp = np.array([degisim_payi(t) for t, (a, _) in zip(M, u) if a])
    S["tercih_karisimi"] = dict(n_govde=len(M), n_uygun=n_uy,
                                oran=round(n_uy / len(M), 4),
                                takas_p50=float(np.median(tk)) if tk else 0.0,
                                takas_ort=float(np.mean(tk)) if tk else 0.0,
                                degisim_payi=dict(
                                    p05=float(np.percentile(dp, 5)), p50=float(np.median(dp)),
                                    p95=float(np.percentile(dp, 95)), ort=float(dp.mean()))
                                if len(dp) else None)
    print(f"  tercih karisimi: {len(M):,} gövde · UYGUN {n_uy:,} (%{n_uy/len(M)*100:.1f}) · "
          f"gövde basina takas medyan {np.median(tk):.0f}", flush=True)
    P = [json.loads(l)["metin"] for l in
         open(__DNH_DATA__ + "/c1_panel/Tulu3-8B/sft/uretim.jsonl", encoding="utf-8")]
    u2 = [uygun(t) for t in P]; n2 = sum(1 for a, _ in u2 if a)
    S["sft_uretimi"] = dict(n_govde=len(P), n_uygun=n2, oran=round(n2 / len(P), 4))
    print(f"  SFT üretimi:     {len(P):,} gövde · UYGUN {n2:,} (%{n2/len(P)*100:.1f})", flush=True)
    HEDEF = 7537
    S["kapi"] = dict(hedef_cift=HEDEF,
                     tercih_karisimi_yeter=bool(n_uy >= HEDEF),
                     sft_uretimi_yeter=bool(n2 >= HEDEF),
                     SERH=""
                          "")
    print(f"\n★ KAPI: hedef {HEDEF:,} cift/kol ⇒ tercih karisimi "
          f"{'YETER' if n_uy>=HEDEF else 'YETMEZ'} ({n_uy:,}) · SFT üretimi "
          f"{'YETER' if n2>=HEDEF else 'YETMEZ'} ({n2:,})")
    print("  ⇒ EYLEM: yetmeyen kaynak prereg taslaginda ADIYLA elenir")
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    payda("kk2_fizibilite", n_kaynak=2, hal_uygun_karisim=n_uy, hal_uygun_sft=n2,
          hal_hedef=HEDEF)


if __name__ == "__main__":
    main()
