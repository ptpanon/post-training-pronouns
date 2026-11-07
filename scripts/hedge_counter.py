#!/usr/bin/env python3
from __future__ import annotations
import re

BELIRTEC = (
    "perhaps maybe possibly probably likely unlikely apparently seemingly "
    "arguably presumably roughly approximately somewhat fairly rather quite "
    "generally typically usually often sometimes occasionally frequently "
    "largely mostly partly slightly relatively essentially basically "
    "potentially conceivably ostensibly nominally broadly loosely"
).split()
_BEL = re.compile(r"\b(" + "|".join(BELIRTEC) + r")\b")
_FIIL = re.compile(r"\b(seem|seems|seemed|seeming|appear|appears|appeared|"
                   r"suggest|suggests|suggested|tend|tends|tended|"
                   r"indicate|indicates|indicated)\b")
_KALIP = re.compile(r"\b(kind of|sort of|to some extent|in general|"
                    r"more or less|by and large|as far as)\b")
_DISLA = re.compile(r"\b(i think|i believe|i guess|i suppose|i feel|"
                    r"in my view|in my opinion|we think|we believe)\b")


def say(metin: str) -> int:
    t = _DISLA.sub(" ", (metin or "").lower())
    return len(_BEL.findall(t)) + len(_FIIL.findall(t)) + len(_KALIP.findall(t))


def prova():
    d = [
        ("", 0, "bos metin"),
        ("The answer is 42.", 0, "cekincesiz"),
        ("This is perhaps probably the answer.", 2, "iki belirtec"),
        ("It seems to indicate something.", 2, "iki fiil"),
        ("I think this is perhaps right.", 1, "1. kisi kalibi DÜSER, belirtec kalir"),
        ("Perhaps. Perhaps. Perhaps.", 3, "tekrar"),
        ("PERHAPS it is LIKELY.", 2, "büyük harf duyarsiz"),
        ("perhapsx likelyish", 0, "kelime siniri"),
    ]
    out = []
    for m, bek, ad in d:
        g = say(m)
        out.append(dict(ad=ad, beklenen=bek, gozlenen=g, tuttu=bool(g == bek)))
    return out


if __name__ == "__main__":
    P = prova()
    for r in P:
        print(f"  {'✓' if r['tuttu'] else '✗'} {r['ad']:<38} beklenen {r['beklenen']} · "
              f"gözlenen {r['gozlenen']}")
    n = sum(r["tuttu"] for r in P)
    print(f"{len(P)} {n}"
          f"{len(P)} {len(P)}")
    raise SystemExit(0 if n == len(P) else 5)
