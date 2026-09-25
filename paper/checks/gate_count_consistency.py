#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, re, sys

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import gate_ref_target as K14

SAYI = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20}
_N = r"(?:\d+|" + "|".join(SAYI) + r")"
DESEN = re.compile(r"\b(" + _N + r")\s+of\s+(?:the\s+|its\s+|those\s+)?(" + _N + r")\b", re.I)
DURAK = set("""a an the of in on at to for and or but is are was were be been that this those these
it its their our we they he she as with by from than then so not no nor only also both each every
which where when what who whom whose have has had do does did can could may might will would shall
should must more most less least other others same such very much many few all any some one two
three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
per over under between among into out up down about after before while because since though
report reports reported reporting give gives given say says said measure measured measures""".split())
AYIRT_EDICI = {"sentence", "sentences", "thousand", "token", "tokens", "elicited",
               "neutral", "swapped", "matched", "force", "register", "base",
               "aligned", "filtered", "seed", "seeds", "rung", "rungs", "step",
               "steps", "arm", "arms", "pair", "pairs", "ladder", "ladders",
               "grows", "shrinks", "grow", "shrink", "m1", "m2", "m3", "m4", "m5"}
BAR_ORTUSME = 0.34
PENCERE = 14


def _sade(s: str) -> str:
    s = K14.taslak_bloklarini_at(re.sub(r"(?m)(?<!\\)%.*$", "", s))
    s = re.sub(r"\\(?:verdict|sahipsececek)\{[^{}]*\}", " ", s)
    s = re.sub(r"\$?\\?([A-Za-z])_\{?(\d)\}?\$?", r" \1\2 ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return re.sub(r"[{}$\\~^_&]", " ", s)


def _imza(metin: str, i: int, j: int):
    bas = max((metin.rfind(x, 0, i) for x in (". ", ".\n", "; ")), default=-1)
    son = min((k for k in (metin.find(x, j) for x in (". ", ".\n", "; ")) if k > 0),
              default=-1)
    if son < 0:
        son = min(len(metin), j + 200)
    k = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9-]+", metin[bas + 1:son])]
    k = k[max(0, len(k) - 2 * PENCERE):]
    return (set(w for w in k
                if w not in DURAK and (len(w) > 2 or re.fullmatch(r"m\d", w))),
            (bas, son))


def _sayi(x: str) -> int:
    return int(x) if x.isdigit() else SAYI[x.lower()]


def _ortusme(a, b):
    return len(a & b) / max(len(a | b), 1)


def _prova():
    CELISIK = ("We find 8 of 16 families move upward here. "
               "Elsewhere we say 9 of 16 families move upward here.")
    PAYDA_KAYMASI = ("We find 8 of 16 families move upward here. "
                     "Elsewhere we say 8 of 20 families move upward here.")
    TEMIZ = ("We find 8 of 16 families move upward here. "
             "Elsewhere we say 9 of 16 families move upward there per sentence.")
    return {"i_celiskiyi_yakalar": len(denetle(CELISIK)[0]) == 1,
            "i_b_payda_kaymasini_yakalar":
                [x["sinif"] for x in denetle(PAYDA_KAYMASI)[0]] == ["PAYDA-KAYMASI"],
            "ii_ayirt_ediciyi_gecirir": len(denetle(TEMIZ)[0]) == 0,
            "iii_yazi_sayi_esdeger": _sayi("eleven") == 11 and _sayi("11") == 11,
            "iv_bos_metin": len(denetle("")[0]) == 0}


def denetle(metin: str):
    t = _sade(metin)
    bulgu = []
    for m in DESEN.finditer(t):
        pay, payda = _sayi(m.group(1)), _sayi(m.group(2))
        if payda < 2 or pay > payda:
            continue
        cev = t[max(0, m.start() - 2):m.start()] + t[m.end():m.end() + 2]
        if "+" in cev:
            continue
        im, span = _imza(t, m.start(), m.end())
        bulgu.append(dict(pay=pay, payda=payda, imza=im, span=span,
                          lafiz=t[max(0, m.start() - 60):m.end() + 60].strip()))
    ates = []
    for i in range(len(bulgu)):
        for j in range(i + 1, len(bulgu)):
            a, b = bulgu[i], bulgu[j]
            if a["payda"] == b["payda"] and a["pay"] == b["pay"]:
                continue
            if a["span"] == b["span"]:
                continue
            ay = (a["imza"] ^ b["imza"]) & AYIRT_EDICI
            if ay:
                continue
            o = _ortusme(a["imza"], b["imza"])
            if o >= BAR_ORTUSME:
                sinif = ("PAYDA-KAYMASI" if a["payda"] != b["payda"]
                         else "PAY-KAYMASI")
                ates.append(dict(sinif=sinif, payda_a=a["payda"], payda_b=b["payda"],
                                 pay_a=a["pay"], pay_b=b["pay"],
                                 ortusme=round(o, 3),
                                 ortak=sorted(a["imza"] & b["imza"])[:8],
                                 a=a["lafiz"], b=b["lafiz"]))
    return ates, bulgu


def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/paper/p11.tex"
    P = _prova(); print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    if not all(P.values()):
        print("★★ PROVA DÜSTÜ ⇒ kapi kosmaz (§7.2)"); return 4
    s = open(yol, encoding="utf-8").read()
    sys.path.insert(0, __DNH_ROOT__ + "/scripts")
    from tex_tree import agac as _agac
    _kok = os.path.basename(yol)
    for f in _agac(_kok, os.path.dirname(os.path.abspath(yol)))[0][1:]:
        f = os.path.join(os.path.dirname(os.path.abspath(yol)), f)
        if os.path.exists(f):
            s += "\n" + open(f, encoding="utf-8").read()
    ates, bulgu = denetle(s)
    print(f"   ★ KAPI-15 · «X of Y» bulgusu={len(bulgu)} · celiski={len(ates)} "
          f"⇒ esik 0 ⇒ EYLEM: >0 ise derleme DÜSER")
    say = {k: sum(1 for a in ates if a["sinif"] == k)
           for k in ("PAYDA-KAYMASI", "PAY-KAYMASI")}
    print(f"     sinif dagilimi: {say}")
    for a in ates:
        print(f"     ✗ {a['sinif']}: {a['pay_a']}/{a['payda_a']} ↔ "
              f"{a['pay_b']}/{a['payda_b']} (örtüsme {a['ortusme']}, "
              f"ortak {a['ortak']})")
        print(f"        A: …{a['a']}…")
        print(f"        B: …{a['b']}…")
    return 15 if ates else 0


if __name__ == "__main__":
    sys.exit(main())
