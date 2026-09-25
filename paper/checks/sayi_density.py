#!/usr/bin/env python3
import json
import re
import sys

import fitz

ESIK = {"1": 25, "3": 9, "4": 12, "5": 12, "GOVDE": 12}
ESIK_NOFM = 20

GUTTER_X = 90.0
PROZA_PT = 9.8
UST_BILGI = "Under review as a conference paper at ICLR"
KUNYE_BAS = re.compile(r"^\s*(Table|Figure)\s*\d+\s*:")
IDDIA_KUTU = re.compile(r"^\s*(Registered|Descriptive|Exploratory)\b")

BASLIK_PT = 11.5
BASLIK_GOVDE_PT = 9.5
BOLUM_ADLARI = [
    ("INTRODUCTION", "1"), ("MEASUREMENT", "2"), ("THEFIRSTANDSECOND", "3"),
    ("WHICHSTAGE", "4"), ("PREFERENCEDATA", "5"), ("RELATEDWORK", "6"),
    ("DISCUSSION", "7"), ("LIMITATIONS", "LIMITATIONS"),
]

JETON = re.compile(r"\S+")
RAKAMLI = re.compile(r"\d")
KELIME = re.compile(r"[A-Za-z0-9]")

ATIF_YIL = re.compile(r"^(19|20)\d{2}[a-z]?$")
CAPRAZ = re.compile(
    r"^(§+\s*)?(\d+(\.\d+)*|[A-Z]\.?\d+(\.\d+)*|A\d+)[.,;)]?$"
)
CAPRAZ_ONCU = re.compile(r"(?i)^(§|Section|Appendix|Table|Figure|Fig\.?|App\.?)$")
MODEL_ADI = re.compile(r"^[A-Za-z][A-Za-z]*[-−]?\d")
N_OF_M = re.compile(r"\b\d+\s+of\s+(?:the\s+)?\d+\b")


def _proza_satirlari(pdf):
    d = fitz.open(pdf)
    out = []
    for pg in d:
        for b in pg.get_text("dict")["blocks"]:
            if b.get("type", 0) != 0:
                continue
            sp_all = [s for l in b.get("lines", []) for s in l["spans"] if s["bbox"][0] >= GUTTER_X]
            if not sp_all:
                continue
            if any(s["size"] >= BASLIK_PT for s in sp_all):
                t = "".join(s["text"] for s in sp_all if s["size"] >= BASLIK_GOVDE_PT)
                if t.strip():
                    out.append((t, True))
                continue
            satirlar = []
            for l in b.get("lines", []):
                sp = [s for s in l["spans"] if s["size"] >= PROZA_PT and s["bbox"][0] >= GUTTER_X]
                if not sp:
                    continue
                t = "".join(s["text"] for s in sp)
                if UST_BILGI in t:
                    continue
                satirlar.append((t, False))
            if not satirlar:
                continue
            blok = " ".join(t for t, _ in satirlar)
            if KUNYE_BAS.match(blok) or IDDIA_KUTU.match(blok):
                continue
            out.extend(satirlar)
    return out


def _bolumle(satirlar):
    bol, ad = {}, None
    for t, bas in satirlar:
        if bas:
            anahtar = re.sub(r"[^A-Z]", "", t.upper())
            yeni = None
            for on, k in BOLUM_ADLARI:
                if anahtar.startswith(on):
                    yeni = k
                    break
            if yeni is None:
                if ad == "LIMITATIONS" or (ad is not None and "LIMITATIONS" in bol):
                    break
                continue
            ad = yeni
            bol.setdefault(ad, [])
            continue
        if ad is not None:
            bol.setdefault(ad, []).append(t)
    return bol


def _say(metin):
    jet = JETON.findall(metin)
    kelime = sum(1 for j in jet if KELIME.search(j))
    pay, yil, capraz, model = [], [], [], []
    for i, j in enumerate(jet):
        if not RAKAMLI.search(j):
            continue
        onceki = jet[i - 1] if i else ""
        c = j.strip("“”\"'()[].,;:")
        cc = re.sub(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$", "", j)
        if ATIF_YIL.match(cc):
            yil.append(j)
        elif CAPRAZ_ONCU.match(re.sub(r"^[^0-9A-Za-z§]+|[^0-9A-Za-z]+$", "", onceki)) and CAPRAZ.match(cc):
            capraz.append(f"{onceki} {j}")
        elif j.lstrip("(").startswith("§") or onceki.strip("(,;") == "§":
            capraz.append(j)
        elif MODEL_ADI.match(j.lstrip("(“")):
            model.append(j)
        else:
            pay.append(j)
    ham = len(pay) + len(yil) + len(capraz) + len(model)
    return dict(kelime=kelime, sayi=len(pay), jetonlar=pay, ham=ham,
                cik_yil=len(yil), cik_capraz=len(capraz), cik_model=len(model),
                nofm=len(N_OF_M.findall(metin)))


def olc(pdf):
    bol = _bolumle(_proza_satirlari(pdf))
    if not bol:
        return None
    sira = [k for k in ["1", "2", "3", "4", "5", "6", "7", "LIMITATIONS"] if k in bol]
    res = {k: _say(" ".join(bol[k])) for k in sira}
    govde = _say(" ".join(" ".join(bol[k]) for k in sira))
    res["GOVDE"] = govde
    return res


def _oran(r):
    return (r["kelime"] / r["sayi"]) if r["sayi"] else float("inf")


def main(pdf="p11_submission.pdf", *a):
    res = olc(pdf)
    if res is None:
        print("")
        return 29
    print(f""
          f"{pdf}")
    print(f"")
    asan = []
    for k in [x for x in ["1", "2", "3", "4", "5", "6", "7", "LIMITATIONS"] if x in res] + ["GOVDE"]:
        r = res[k]
        o = _oran(r)
        e = ESIK.get(k)
        et = f"esik 1/{e}" if e else "esik yok"
        bayrak = ""
        if e and r["sayi"] and o < e:
            bayrak = f"  ⇐ ASILDI ({r['sayi'] - int(r['kelime'] / e)} sayi fazla)"
            asan.append(k)
        hо = (r["kelime"] / r["ham"]) if r["ham"] else float("inf")
        print(f"     {k:<12} kelime {r['kelime']:>5} · sayi {r['sayi']:>4} · 1/{o:>5.1f} · {et:<10}"
              f" · N-of-M {r['nofm']:>3} · ham {r['ham']:>4} (1/{hо:>4.1f})"
              f" [yil {r['cik_yil']} · gönderme {r['cik_capraz']} · model {r['cik_model']}]{bayrak}")
    g = res["GOVDE"]
    print(f"   ★ KAPI-28 · «N of M» gövdede {g['nofm']} ⇒ esik {ESIK_NOFM} ⇒ EYLEM: asilirsa UYARIR")
    if g["nofm"] > ESIK_NOFM:
        print(f"   ★★ KAPI-28 UYARI · «N of M» tavani ASILDI — {g['nofm'] - ESIK_NOFM} fazla")
    if asan:
        print(f"   ★★ KAPI-28 UYARI · yogunluk esigini asan bölüm(ler): {', '.join(asan)} "
              f"⇒ EYLEM: seyreltme sürer (RAPOR kipi, derleme DÜSMEZ)")
    else:
        print("")
    if "--json" in a:
        yol = a[a.index("--json") + 1]
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "jetonlar"} for k, v in res.items()},
                  open(yol, "w"), ensure_ascii=False, indent=1)
        print(f"   ★ KAPI-28 · ölcüm yazildi: {yol}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
