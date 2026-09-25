#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, os, subprocess, sys

ROOT = __DNH_ROOT__ + ""
TABAN = "38ef99c4"
YOLLAR = ["paper/p11.tex", "paper/figures",
          "paper/p11.bib", "paper/p11_ek.bib",
          "paper/OZET_HARITASI.json"]


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kutuk", default=f"{ROOT}/paper/DEGISIKLIK_v88_SONRASI.md")
    a = ap.parse_args()
    shalar = [l.split()[0] for l in git("log", "--format=%h %s", "--abbrev=8", f"{TABAN}..HEAD", "--", *YOLLAR).splitlines() if l.strip()]
    metin = open(a.kutuk, encoding="utf-8").read() if os.path.exists(a.kutuk) else ""
    eksik = [s for s in shalar if s[:8] not in metin]
    son = shalar[0] if shalar else None
    kirli = [l[3:] for l in git("status", "--porcelain", "--", *YOLLAR).splitlines() if l[:2].strip() and not l.startswith("??")]
    print(f"{TABAN} {len(shalar)} {len(shalar) - len(eksik)} {len(eksik)}"
          f"{(' ' + str(eksik)) if eksik else ''} {' (en son commit: ' + son + '' if son in eksik else ''}"
          f"{len(kirli)}")
    if not os.path.exists(a.kutuk):
        print(f"{a.kutuk}")
    print(f"{len(shalar)} {len(shalar) - len(eksik)} {len(eksik)} {len(kirli)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
