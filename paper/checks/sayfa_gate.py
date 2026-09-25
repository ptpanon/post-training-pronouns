#!/usr/bin/env python3
import re, sys
from pypdf import PdfReader

BAR = 9
BEYAN = r"\b(AI USE STATEMENT|ETHICS STATEMENT|REPRODUCIBILITY STATEMENT)\b"


def olc(yol):
    r = PdfReader(yol)
    for i, pg in enumerate(r.pages):
        if i <= 3:
            continue
        t = pg.extract_text() or ""
        m = re.search(BEYAN, t, re.I)
        if m:
            g = re.sub(r"^[\d\s]+", "", t[:m.start()])
            g = g.replace("Under review as a conference paper at ICLR 2027", "").strip()
            return dict(n=i + (1 if g else 0), beyan=i + 1, tasan=len(g), toplam=len(r.pages))
    return None


def main(yol="p11_submission.pdf"):
    o = olc(yol)
    if o is None:
        print(f"{yol}")
        return 29
    n, t = o["n"], o["tasan"]
    print(f"{n} {o['beyan']} {t}"
          f"{o['toplam']} {BAR} {BAR}")
    if n > BAR:
        print(f"{n} {o['beyan']} {t}")
        return 28
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
