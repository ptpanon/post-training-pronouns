#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, re, sys, hashlib

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

LISTE_YOL = f"{ROOT}/unreleased/muhatap_isaretler.jsonl"
_CUMLE_BASI = r"(?:^|[.!?;:\n]\s*|—\s*)"

SINIFLAR = ("MUHATAP-VAR", "MUHATAP-YOK", "KALAN")


def liste_yukle(yol=LISTE_YOL):
    ham = open(yol, "rb").read()
    sha = hashlib.sha256(ham).hexdigest()[:16]
    kayitlar = [json.loads(l) for l in ham.decode("utf-8").splitlines() if l.strip()]
    derli = []
    for k in kayitlar:
        t = k["tur"]
        if t == "kelime":
            d = re.compile(rf"\b{k['kalip']}\b", re.I)
        elif t == "regex":
            d = re.compile(k["kalip"], re.I)
        elif t == "emir":
            d = re.compile(rf"{_CUMLE_BASI}(?:{k['kalip']})\b", re.I)
        else:
            raise SystemExit(f"★ bilinmeyen isaret türü: {t} ({k['ad']})")
        derli.append((k["ad"], k["grup"], d))
    gruplar = sorted({k["grup"] for k in kayitlar})
    payda("muhatap_liste", n_isaret=len(derli), n_grup=len(gruplar),
          bekle={"n_grup": 3})
    return derli, sha, gruplar


def grup_derle(yol=LISTE_YOL):
    ham = open(yol, "rb").read()
    sha = hashlib.sha256(ham).hexdigest()[:16]
    kayitlar = [json.loads(l) for l in ham.decode("utf-8").splitlines() if l.strip()]
    parca = {}
    for k in kayitlar:
        t, kal = k["tur"], k["kalip"]
        if t == "kelime":   p = rf"\b{kal}\b"
        elif t == "regex":  p = kal
        elif t == "emir":   p = rf"{_CUMLE_BASI}(?:{kal})\b"
        else: raise SystemExit(f"★ bilinmeyen isaret türü: {t} ({k['ad']})")
        parca.setdefault(k["grup"], []).append(f"(?:{p})")
    return {g: re.compile("|".join(v), re.I) for g, v in parca.items()}, sha


def sinifla_hizli(metin, birlesim):
    if birlesim["ikinci_kisi"].search(metin):
        return "MUHATAP-VAR"
    for g, d in birlesim.items():
        if g != "ikinci_kisi" and d.search(metin):
            return "KALAN"
    return "MUHATAP-YOK"


def isaretle(metin, derli):
    vur = {}
    for ad, grup, d in derli:
        if d.search(metin):
            vur.setdefault(grup, []).append(ad)
    return vur


def sinifla(metin, derli):
    vur = isaretle(metin, derli)
    if vur.get("ikinci_kisi"):
        return "MUHATAP-VAR", vur
    if vur:
        return "KALAN", vur
    return "MUHATAP-YOK", vur


def uretim_tara(yol, derli, alan="metin"):
    R = [json.loads(l) for l in open(yol, encoding="utf-8")]
    say = {s: 0 for s in SINIFLAR}
    hucre = {}
    for r in R:
        s, _ = sinifla(r.get(alan, ""), derli)
        say[s] += 1
        hucre.setdefault((r["kol"], r["istem_i"]), {x: 0 for x in SINIFLAR})[s] += 1
    return R, say, hucre


def _prova(derli):
    kip = [
        ("You should think about this.",        "MUHATAP-VAR"),
        ("The theory predicts warming.",        "MUHATAP-YOK"),
        ("Consider the evidence carefully.",    "KALAN"),
        ("I consider the evidence carefully.",  "MUHATAP-YOK"),
        ("Please review the data.",             "KALAN"),
    ]
    kotu = [(m, b, sinifla(m, derli)[0]) for m, b in kip if sinifla(m, derli)[0] != b]
    payda("muhatap_prova", n_kip=len(kip), red_dusen=len(kotu))
    if kotu:
        raise SystemExit(f"★ PROVA DÜSTÜ: {kotu}")
    print(f"  [PROVA] {len(kip)}/{len(kip)} kip — üc sinif da dogdu", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", default=__DNH_DATA__ + "/c1_panel")
    ap.add_argument("--alan", default="metin", choices=["metin", "onek"])
    ap.add_argument("--cikti", default="")
    a = ap.parse_args()

    derli, sha, gruplar = liste_yukle()
    print(f"  [LISTE] {LISTE_YOL} · sha={sha} · gruplar={gruplar}", flush=True)
    _prova(derli)

    import glob
    yollar = sorted(glob.glob(f"{a.kok}/*/*/uretim.jsonl"))
    ozet, n_hep_ikisi, n_hucre = [], 0, 0
    for y in yollar:
        cift, zemin = y.split("/")[-3], y.split("/")[-2]
        R, say, hucre = uretim_tara(y, derli, a.alan)
        n_hucre += len(hucre)
        ozet.append(dict(cift=cift, zemin=zemin, n_satir=len(R), **say))
        print(f"  {cift:<24} {zemin:<9} n={len(R):>5} · VAR={say['MUHATAP-VAR']:>5}"
              f" · YOK={say['MUHATAP-YOK']:>5} · KALAN={say['KALAN']:>5}", flush=True)
    payda("muhatap_tarama", n_uretim=len(yollar), n_hucre=n_hucre,
          hal_bos=sum(1 for o in ozet if o["n_satir"] == 0))
    if a.cikti:
        json.dump(dict(liste_sha=sha, alan=a.alan, ozet=ozet),
                  open(a.cikti, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  → {a.cikti}", flush=True)
