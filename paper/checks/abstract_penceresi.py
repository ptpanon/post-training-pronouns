#!/usr/bin/env python3
import collections, re


def ozet_jetonlari(pdf_yol, sayfa=2, pay=2.0):
    import fitz
    d = fitz.open(pdf_yol)
    satirlar = []
    for p in range(min(sayfa, len(d))):
        L = collections.defaultdict(list)
        for x0, y0, x1, y1, w, b, l, n in d[p].get_text("words"):
            if x0 >= 100:
                L[(b, l)].append((x0, y0, n, w))
        for v in sorted(L.values(), key=lambda v: (min(y for _, y, _, _ in v), min(x for x, *_ in v))):
            v = sorted(v, key=lambda t: t[0])
            satirlar.append((p, min(x for x, *_ in v), " ".join(t[3] for t in v)))
    i = next((k for k, (_, _, s) in enumerate(satirlar) if s.strip() == "ABSTRACT"), None)
    if i is None or i + 1 >= len(satirlar):
        return None, dict(hal="ÖLCEMEDI", neden="ABSTRACT basligi yok")
    sol = satirlar[i + 1][1]; ab, dur = [], None
    for p, x0, s in satirlar[i + 1:]:
        if x0 < sol - pay or "INTRODUCTION" in s:
            dur = s[:40]; break
        ab.append(s)
    metin = re.sub(r"(\w)- (\w)", r"\1\2", re.sub(r"\s1\s*$", "", " ".join(ab).strip()))
    tok = [x.replace("ﬁ", "fi").replace("ﬂ", "fl") for x in metin.split()]
    return tok, dict(hal="ÖLCÜLDÜ", sol_kenar=round(sol, 1), n_satir=len(ab), durdugu=dur)


if __name__ == "__main__":
    import sys
    tok, t = ozet_jetonlari(sys.argv[1] if len(sys.argv) > 1 else "p11_submission.pdf")
    print(t, None if tok is None else len([x for x in tok if re.search(r"[A-Za-z0-9]", x)]))
