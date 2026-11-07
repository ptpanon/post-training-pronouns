#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, glob, json, os, re, subprocess, sys

ROOT = __DNH_ROOT__ + ""
P11 = f"{ROOT}/paper"
UZ = r"(?:json|md|tsv|csv|jsonl)"
RE_YOL = re.compile(rf"""["'](docs/[^"'\s]+?\.{UZ})["']""")
RE_CIPLAK = re.compile(rf"""["']([A-Za-z0-9][A-Za-z0-9_.\-]*_20\d\d-\d\d-\d\d[A-Za-z0-9_.\-]*\.{UZ})["']""")


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, check=True).stdout


def topla(fig, koken):
    ham = {}
    def ekle(ad, kaynak):
        if ad and isinstance(ad, str):
            ham.setdefault(ad.strip().replace("\\_", "_"), set()).add(kaynak)
    for m in sorted(glob.glob(f"{fig}/*.meta.json")):
        try:
            d = json.load(open(m, encoding="utf-8"))
        except Exception:
            continue
        for s in d.get("sources") or []:
            ekle(s.get("yol") if isinstance(s, dict) else s, os.path.basename(m))
    for p in sorted(glob.glob(f"{fig}/*.py")):
        t = open(p, encoding="utf-8").read()
        for rx in (RE_YOL, RE_CIPLAK):
            for x in rx.findall(t):
                ekle(x, os.path.basename(p))
    if koken and os.path.exists(koken):
        for sat in open(koken, encoding="utf-8"):
            if sat.startswith("#") or sat.startswith("satir\t"):
                continue
            c = sat.rstrip("\n").split("\t")
            if len(c) > 2 and c[2] not in ("", "-"):
                ekle(c[2], os.path.basename(koken))
    return ham


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default=f"{P11}/fig")
    ap.add_argument("--koken", default=f"{P11}/GOVDE_KOKEN.tsv")
    a = ap.parse_args()
    ham = topla(a.fig, a.koken)
    izli = set(git("ls-files").splitlines())
    degismis = {l[3:] for l in git("status", "--porcelain").splitlines() if l[:2].strip() in ("M", "MM", "AM", "RM")}
    dizin = {}
    for dp, dns, fs in os.walk(f"{ROOT}/docs"):
        if os.path.relpath(dp, ROOT) == "paper":
            dns[:] = [d for d in dns if d != "yayim_agaci"]
        for f in fs:
            dizin.setdefault(f, []).append(os.path.relpath(os.path.join(dp, f), ROOT))
    for f in izli:
        dizin.setdefault(os.path.basename(f), [])
        if f not in dizin[os.path.basename(f)]:
            dizin[os.path.basename(f)].append(f)
    sinif = {"IZLI": [], "IZSIZ": [], "DEGISMIS": [], "YOK": []}
    veri, cozulemez = [], []
    for ad, kaynak in sorted(ham.items()):
        if ad.startswith("/mnt/") or (ad.startswith("/") and not ad.startswith(ROOT)) or ad.startswith(("datasets--", "models--")):
            veri.append(ad); continue
        if ad.startswith(ROOT):
            ad = os.path.relpath(ad, ROOT)
        if "/" in ad:
            p11rel = os.path.relpath(f"{P11}/{ad}", ROOT)
            if ad in izli or os.path.exists(f"{ROOT}/{ad}"):
                yollar = [ad]
            elif p11rel in izli or os.path.exists(f"{P11}/{ad}"):
                yollar = [p11rel]
            else:
                yollar = [ad]
        elif re.search(rf"\.{UZ}$", ad):
            yollar = dizin.get(ad, [])
            if not yollar:
                sinif["YOK"].append((ad, sorted(kaynak))); continue
        else:
            cozulemez.append(ad); continue
        for y in yollar:
            if y in degismis and not y.startswith("paper/figures/"):
                sinif["DEGISMIS"].append((y, sorted(kaynak)))
            elif y in izli:
                sinif["IZLI"].append((y, sorted(kaynak)))
            elif os.path.exists(f"{ROOT}/{y}"):
                sinif["IZSIZ"].append((y, sorted(kaynak)))
            else:
                sinif["YOK"].append((y, sorted(kaynak)))
    kusur = len(sinif["IZSIZ"]) + len(sinif["DEGISMIS"]) + len(sinif["YOK"])
    print(f"{len(ham)} {len(sinif['IZLI'])} {len(sinif['IZSIZ'])}"
          f"{len(sinif['DEGISMIS'])} {len(sinif['YOK'])} {len(veri)}"
          f"{len(cozulemez)}")
    for s in ("IZSIZ", "DEGISMIS", "YOK"):
        for y, k in sinif[s]:
            print(f"     ★★ KAPI-25 {s}: {y}  ← {', '.join(k[:4])}{' …' if len(k) > 4 else ''}")
    print(f"{len(ham)} {len(sinif['IZLI'])} {len(sinif['IZSIZ'])}"
          f"{len(sinif['DEGISMIS'])} {len(sinif['YOK'])} {len(veri)} {len(cozulemez)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
