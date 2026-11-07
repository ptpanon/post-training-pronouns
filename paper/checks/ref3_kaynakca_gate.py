#!/usr/bin/env python3
from __future__ import annotations
import hashlib, os, re, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bib_ayristir as B

ALANLAR = ("author", "editor", "title", "year", "booktitle", "journal", "publisher", "howpublished", "doi", "eprint")
SID = re.compile(r"^(arXiv:\d{4}\.\d{4,5}|doi:10\.\S+|url:https?://\S+|hf:[\w.\-]+/[\w.\-]+)$")
VER = re.compile(r"^(\d{4}-\d{2}-\d{2})\|([^|]+)\|([0-9a-f]{12})$")


def ozet(alan):
    p = [f"{k}={' '.join((alan.get(k) or '').split())}" for k in ALANLAR] + [f"sourceid={' '.join((alan.get('sourceid') or '').split())}"]
    return hashlib.sha256("\n".join(p).encode("utf-8")).hexdigest()[:12]


def denetle(yollar):
    n, red = 0, []
    for y in yollar:
        for k, g in B.oku(y).items():
            n += 1
            a = g["alan"]; yer = f"{os.path.basename(y)}:{g['satir']} {k}"
            s = a.get("sourceid", "")
            if not s:
                red.append((yer, "sourceid yok")); continue
            bad = [x for x in re.split(r"\s*;\s*", s) if not SID.match(x)]
            if bad:
                red.append((yer, f"sourceid bicimi {bad}")); continue
            v = VER.match(a.get("verified", ""))
            if not v:
                red.append((yer, "verified damgasi yok ya da bozuk")); continue
            if v.group(3) != ozet(a):
                red.append((yer, f"damgadan sonra elle degismis (damga {v.group(3)} ≠ {ozet(a)})"))
    return n, red


def prova():
    tmp = tempfile.mkdtemp()
    def yaz(ad, govde):
        p = f"{tmp}/{ad}.bib"; open(p, "w", encoding="utf-8").write(govde); return p
    alan = dict(author="A B and C D", title="T", year="2020", eprint="2001.00001", sourceid="arXiv:2001.00001")
    s = ozet(alan)
    def g(**x):
        d = dict(alan, **x)
        return "@misc{k,\n" + "".join(f"  {a} = {{{b}}},\n" for a, b in d.items()) + "}\n"
    temiz = yaz("temiz", g(verified=f"2026-11-06|arXiv|{s}"))
    damgasiz = yaz("damgasiz", g())
    kimliksiz = yaz("kimliksiz", "@misc{k,\n  author = {A B},\n  title = {T},\n  year = {2020},\n}\n")
    oynanmis = yaz("oynanmis", g(author="A B and X Y", verified=f"2026-11-06|arXiv|{s}"))
    bicim = yaz("bicim", g(sourceid="memory", verified=f"2026-11-06|arXiv|{s}"))
    return {"i_damgali_gecer": len(denetle([temiz])[1]) == 0,
            "ii_damgasiz_duser": len(denetle([damgasiz])[1]) == 1,
            "iii_kimliksiz_duser": len(denetle([kimliksiz])[1]) == 1,
            "iv_elle_degisen_duser": len(denetle([oynanmis])[1]) == 1,
            "v_kimlik_bicimi_duser": len(denetle([bicim])[1]) == 1}


def main():
    if "--prova" in sys.argv:
        p = prova(); print("   ★ REF-3 PROVA:", p); return 0 if all(p.values()) else 4
    yollar = sys.argv[1:]
    if not yollar:
        print("   ★★ REF-3: .bib verilmedi ⇒ DUR"); return 31
    n, red = denetle(yollar)
    print(f"   ★ REF-3 · kaynakca damgasi · dosya {len(yollar)} · girdi {n} · damgali {n - len(red)} · IHLAL {len(red)} ⇒ esik 0 ⇒ EYLEM: >0 ise DERLEME DÜSER")
    if n == 0:
        print("   ★★ REF-3: payda 0 ⇒ HATA"); return 31
    for yer, neden in red[:40]:
        print(f"     ✗ {yer}: {neden}")
    return 31 if red else 0


if __name__ == "__main__":
    sys.exit(main())
