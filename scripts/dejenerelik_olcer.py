#!/usr/bin/env python3
from __future__ import annotations
import re, unicodedata
from collections import Counter

EN_ISLEV = frozenset("""a an the and or but if then than that this these those of in on at to for
with from by as is are was were be been being have has had do does did not no nor so such it its
he she they we you i him her them us me my your their our his hers there here what which who whom
whose when where why how all any both each few more most other some only own same too very can
will just should now would could may might must about into over under after before between during
without within because while""".split())

_KELIME = re.compile(r"[^\W\d_]+", re.UNICODE)


def jetonla(t: str):
    return _KELIME.findall((t or "").lower())


def bos(t) -> bool:
    return not (t or "").strip()


def uzunluk_kar(t) -> int:
    return len((t or "").strip())


def yineleme4(t, n: int = 4):
    j = jetonla(t)
    if len(j) < n:
        return None
    g = [tuple(j[i:i + n]) for i in range(len(j) - n + 1)]
    return 1.0 - len(set(g)) / len(g)


def en_uzun_yinelenen_pay(t):
    j = jetonla(t)
    if len(j) < 2:
        return None
    def var_mi(L):
        if L <= 0 or L > len(j):
            return False
        g = set()
        for i in range(len(j) - L + 1):
            k = tuple(j[i:i + L])
            if k in g:
                return True
            g.add(k)
        return False
    lo, hi, en = 1, len(j) // 1, 0
    while lo <= hi:
        m = (lo + hi) // 2
        if var_mi(m):
            en, lo = m, m + 1
        else:
            hi = m - 1
    return en / len(j)


def latin_disi_pay(t):
    harf = [c for c in (t or "") if c.isalpha()]
    if not harf:
        return None
    d = sum(1 for c in harf if not unicodedata.name(c, "").startswith("LATIN"))
    return d / len(harf)


def en_islev_yogunlugu(t):
    j = jetonla(t)
    if len(j) < 5:
        return None
    return sum(1 for w in j if w in EN_ISLEV) / len(j)


def jeton_yogunlugu(t):
    n = uzunluk_kar(t)
    if n == 0:
        return None
    return len(jetonla(t)) / n


def olc(t) -> dict:
    return dict(bos=bos(t), n_kar=uzunluk_kar(t), n_jeton=len(jetonla(t)),
                yin4=yineleme4(t), uzun_yin=en_uzun_yinelenen_pay(t),
                latin_disi=latin_disi_pay(t), en_islev=en_islev_yogunlugu(t),
                jeton_yog=jeton_yogunlugu(t))


def prova():
    V = []
    def bek(ad, gercek, beklenen):
        ok = (gercek == beklenen) if not isinstance(beklenen, float) \
            else (gercek is not None and abs(gercek - beklenen) < 1e-9)
        V.append((ad, gercek, beklenen, ok))
    bek("bos/bos", bos(""), True)
    bek("bos/bosluk", bos("   \n\t "), True)
    bek("bos/dolu", bos("a"), False)
    bek("yin4/tanimsiz(3 jeton)", yineleme4("a b c"), None)
    bek("", yineleme4("a b c d e f g h"), 0.0)
    bek("yin4/saf tekrar", yineleme4(("x y z w " * 8).strip()), 1 - 4 / 29)
    bek("uzunyin/tanimsiz(1 jeton)", en_uzun_yinelenen_pay("a"), None)
    bek("uzunyin/tekrarsiz", en_uzun_yinelenen_pay("a b c d"), 0.0)
    bek("uzunyin/yarim tekrar", en_uzun_yinelenen_pay("a b c d a b c d"), 0.5)
    bek("latin/yok harf", latin_disi_pay("123 !!"), None)
    bek("latin/saf latin", latin_disi_pay("hello world"), 0.0)
    bek("latin/saf kiril", latin_disi_pay("привет мир"), 1.0)
    bek("islev/tanimsiz(4 jeton)", en_islev_yogunlugu("the of and to"), None)
    bek("islev/saf islev", en_islev_yogunlugu("the of and to in"), 1.0)
    bek("", en_islev_yogunlugu("cat dog bird fish tree"), 0.0)
    bek("jetonyog/saf rakam", jeton_yogunlugu(" 1. 1. 1. 1. 1. 1. 1. 1."), 0.0)
    bek("jetonyog/bos", jeton_yogunlugu(""), None)
    V.append(("★ERRATA-1/rakam-dizisi DEJENERE mi",
              bayrakla(" 1. " * 20)["DEJENERE"], True,
              bayrakla(" 1. " * 20)["DEJENERE"] is True))
    V.append(("★ERRATA-1/noktalama DEJENERE mi",
              bayrakla(" : " + ":" * 90)["DEJENERE"], True,
              bayrakla(" : " + ":" * 90)["DEJENERE"] is True))
    _n = "The cat sat on the mat and then it went to the other room quietly today."
    V.append(("★ERRATA-1/normal metin TEMIZ kaldi mi",
              bayrakla(_n)["DEJENERE"], False, bayrakla(_n)["DEJENERE"] is False))
    kotu = [v for v in V if not v[3]]
    print(f"  [PAYDA] dejenerelik_prova: n_sinav={len(V)} gecen={len(V)-len(kotu)} "
          f"red_dusen={len(kotu)}")
    for ad, g, b, ok in V:
        print(f"    {'✓' if ok else '✗'} {ad:28s} ölcülen={g} beklenen={b}")
    assert not kotu, "★ PROVA DÜSTÜ — alet dejenere, rapor yazilmaz"
    return True




ESIK = {
    "bos":        dict(rule="metin.strip() == ''", taban="tanim geregi", olculen=None),
    "kisa":       dict(rule="len(strip) < 15 karakter", taban="direktifte zaten dislama",
                       olculen=None),
    "yin4":       dict(rule="yineleme4 >= 0.1429",
                       taban="ALTIN INTACT dagilimi: med 0,000 · p95 0,132 · maks 0,241",
                       olculen="REPETITIVE yakalama 0,814 · INTACT yanlis-poz 0,042 "
                               "(n_poz=43 · n_neg=95)"),
    "uzun_yin":   dict(rule="en_uzun_yinelenen_pay >= 0.1008",
                       taban="ALTIN INTACT: med 0,040 · p95 0,098 · maks 0,176",
                       olculen="REPETITIVE yakalama 0,735 · INTACT yanlis-poz 0,042 "
                               "(n_poz=49 · n_neg=95)"),
    "islev_seyrek": dict(rule="en_islev_yogunlugu < 0.3247",
                         taban="ALTIN INTACT (Ingilizce) p1 = 0,3247 · p5 = 0,3558 · "
                               "min = 0,303",
                         olculen="yapi geregi INTACT'in_ ~%1'ini bayraklar"),
    "jetonsuz": dict(rule="n_kar >= 15 ve jeton_yogunlugu < 0.1270",
                     taban="ALTIN INTACT (n_kar>=15) jeton-yogunlugu **min 0,1270** · "
                           "p1 0,1416 · med 0,1694 ↔ REPETITIVE min 0,000 · p5 0,008",
                     olculen="yanlis-pozitif 0/95 (esik INTACT'in_ ölcülen minimumu) — "
                             "★ ERRATA-1: bu ölcüt ilk sürümde YOKTU"),
    "latin_disi": dict(rule="latin_disi_pay > 0.0",
                       taban="ALTIN: INTACT 95/95 = 0,000 · REPETITIVE 50/50 = 0,000",
                       olculen=""),
}
E_YIN4, E_UZUN, E_ISLEV, E_JETON = 0.1429, 0.1008, 0.3247, 0.1270


def bayrakla(t) -> dict:
    m = olc(t)
    b = dict(m)
    b["f_bos"] = m["bos"]
    b["f_kisa"] = (not m["bos"]) and m["n_kar"] < 15
    b["f_yin4"] = (m["yin4"] is not None) and m["yin4"] >= E_YIN4
    b["f_uzun"] = (m["uzun_yin"] is not None) and m["uzun_yin"] >= E_UZUN
    b["f_islev"] = (m["en_islev"] is not None) and m["en_islev"] < E_ISLEV
    b["f_latin"] = (m["latin_disi"] is not None) and m["latin_disi"] > 0.0
    b["f_jetonsuz"] = (m["n_kar"] >= 15 and m["jeton_yog"] is not None
                       and m["jeton_yog"] < E_JETON)
    b["tanimsiz_yin4"] = m["yin4"] is None
    b["tanimsiz_islev"] = m["en_islev"] is None
    b["DEJENERE"] = bool(b["f_bos"] or b["f_kisa"] or b["f_yin4"] or b["f_uzun"]
                         or b["f_islev"] or b["f_latin"] or b["f_jetonsuz"])
    return b


if __name__ == "__main__":
    prova()
