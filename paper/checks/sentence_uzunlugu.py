#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, os, re, sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAP = f"{KOK}/paper"
ESIK_LISTE = 40
ESIK_BOL   = 45

KISALT = {"e.g.", "i.e.", "al.", "cf.", "vs.", "Fig.", "Eq.", "No.", "St.",
          "Dr.", "Mr.", "Ms.", "approx.", "etc.", "Inc.", "Ltd.", "Appendix."}


CEVRELER = ("figure", "figure*", "tikzpicture", "tcolorbox", "tabular",
            "table", "table*", "lstlisting", "verbatim")


def _cevre_bosalt(t: str):
    n = 0
    for c in CEVRELER:
        r = re.compile(r"\\begin\{" + re.escape(c) + r"\}.*?\\end\{" + re.escape(c) + r"\}", re.S)
        t, k = r.subn(lambda m: "\n" * m.group(0).count("\n"), t)
        n += k
    return t, n


def _yorum_sil(sat: str) -> str:
    out, kac = [], False
    for ch in sat:
        if kac:
            out.append(ch); kac = False; continue
        if ch == "\\":
            out.append(ch); kac = True; continue
        if ch == "%":
            break
        out.append(ch)
    return "".join(out)


def _eslesen_kapanis(s: str, i: int) -> int:
    if i >= len(s) or s[i] != "{":
        return -1
    d = 0
    while i < len(s):
        if s[i] == "\\":
            i += 2; continue
        if s[i] == "{":
            d += 1
        elif s[i] == "}":
            d -= 1
            if d == 0:
                return i
        i += 1
    return -1


def _makro_govdesi_al(s: str, ad: str):
    out, i = [], 0
    dsn = "\\" + ad + "{"
    while True:
        j = s.find(dsn, i)
        if j < 0:
            return out
        a = j + len(dsn) - 1
        b = _eslesen_kapanis(s, a)
        if b < 0:
            return out
        out.append((a + 1, b, s[a + 1:b]))
        i = b + 1


def _makro_sil(t: str, adlar) -> str:
    for ad in adlar:
        while True:
            j = t.find("\\" + ad + "{")
            if j < 0:
                break
            a = j + len(ad) + 1
            b = _eslesen_kapanis(t, a)
            if b < 0:
                t = t[:j] + t[j + len(ad) + 1:]
                break
            t = t[:j] + " " + t[b + 1:]
    return t


def _input_ac(t: str, derinlik: int = 0):
    if derinlik > 3:
        return t, 0
    n = 0
    while True:
        m = re.search(r"\\input\{(fig/[A-Za-z0-9_]+)(?:\.tex)?\}", t)
        if not m:
            return t, n
        y = f"{PAP}/{m.group(1)}.tex"
        if not os.path.exists(y):
            t = t[:m.start()] + " " + t[m.end():]
            continue
        it = io.open(y, encoding="utf-8").read()
        it, _ = _cevre_bosalt(it)
        it = _makro_sil(it, ["caption"])
        it = " ".join(_yorum_sil(x) for x in it.splitlines())
        gl = [g for _, _, g in _makro_govdesi_al(it, "vekalet")]
        it = " ".join(gl) if gl else it
        it, _k = _input_ac(it, derinlik + 1)
        t = t[:m.start()] + " " + it.strip() + " " + t[m.end():]
        n += 1 + _k


def duz(t: str) -> str:
    t = _makro_sil(t, ["citep", "citet", "cite", "label", "verdict", "footnote"])
    t = re.sub(r"~?\\ref\{[^}]*\}", " REFNO", t)
    t = re.sub(r"\\input\{[^}]*\}", " ", t)
    t = re.sub(r"\\unskip\{\}", " ", t)
    t = re.sub(r"\$[^$]*\$", " MATH", t)
    for ad in ("emph", "textbf", "textit", "texttt", "textsc"):
        while True:
            j = t.find("\\" + ad + "{")
            if j < 0:
                break
            a = j + len(ad) + 1
            b = _eslesen_kapanis(t, a)
            if b < 0:
                break
            t = t[:j] + t[a + 1:b] + t[b + 1:]
    t = t.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    t = re.sub(r'\\"([A-Za-z])', r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = t.replace("{", " ").replace("}", " ").replace("--", "-")
    return re.sub(r"\s+", " ", t).strip()


def cumleler(t: str):
    out, bas = [], 0
    for m in re.finditer(r"[.!?]+[\"')\]]*\s+", t):
        son = t[bas:m.end()].strip()
        aday = son.split()[-1] if son.split() else ""
        if aday in KISALT or re.match(r"^[A-Z]\.$", aday):
            continue
        out.append((bas, son))
        bas = m.end()
    kalan = t[bas:].strip()
    if kalan:
        out.append((bas, kalan))
    return out


def kelime(c: str) -> int:
    return len([w for w in c.split() if re.search(r"[A-Za-z0-9]", w)])


def _govde_pencereleri(sat):
    bas = son = None
    for i, s in enumerate(sat):
        if bas is None and re.search(r"\\section\{Introduction\}", s):
            bas = i
        if bas is not None and re.search(r"\\section\*?\{Limitations\}", s):
            son = i
    if bas is None:
        raise SystemExit("")
    if son is None:
        son = len(sat)
    else:
        for j in range(son + 1, len(sat)):
            if re.search(r"\\(section|appendix|bibliography)", sat[j]):
                son = j; break
        else:
            son = len(sat)
    return bas, son


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=f"{PAP}/p11.tex")
    ap.add_argument("--sert", action="store_true", help="bulgu varsa cikis 29")
    a = ap.parse_args()
    _ham = open(a.tex, encoding="utf-8").read()
    _ham, _n_bos = _cevre_bosalt(_ham)
    sat = _ham.splitlines()
    print(f"{_n_bos}"
          f"")
    bas, son = _govde_pencereleri(sat)

    bolum, kayit = "(baslik öncesi)", []
    n_kaynak = n_input = 0
    for i in range(bas, son):
        ham = _yorum_sil(sat[i])
        m = re.search(r"\\(?:sub)*section\*?\{([^}]*)\}", ham)
        if m:
            bolum = duz(m.group(1))
        ham, _ki = _input_ac(ham)
        n_input += _ki
        parca = []
        for ad in ("vekalet", "paragraph"):
            parca += [g for _, _, g in _makro_govdesi_al(ham, ad)]
        if not parca and _ki and ham.strip():
            parca = [ham]
        for p in parca:
            d = duz(p)
            if not d:
                continue
            n_kaynak += 1
            for _, c in cumleler(d):
                k = kelime(c)
                if k >= 1:
                    kayit.append((k, bolum, i + 1, c))

    if not kayit:
        raise SystemExit("")
    uz = sorted([k for k, *_ in kayit])
    n = len(uz)
    ort = sum(uz) / n
    ortanca = uz[n // 2]
    p90 = uz[int(n * 0.90)]
    uzun = sorted([r for r in kayit if r[0] > ESIK_LISTE], key=lambda r: -r[0])
    bol = [r for r in uzun if r[0] > ESIK_BOL]

    print(f"  [PAYDA] kapi29_input: n_acilan_input={n_input} ⇒ esik: basilan her düzyazi YERINDE olmali ⇒ EYLEM: acilmayan \\input sahte cümle birlesmesi üretir")
    print(f"  [PAYDA] kapi29_cumle: n_blok={n_kaynak} · n_cumle={n} · ortalama={ort:.1f} · "
          f"ortanca={ortanca} · p90={p90} · en_uzun={uz[-1]}")
    print(f"  [PAYDA] kapi29_bar: n_40ustu={len(uzun)} ⇒ esik {ESIK_LISTE} ⇒ EYLEM: rapora liste · "
          f"n_45ustu={len(bol)} ⇒ esik {ESIK_BOL} ⇒ EYLEM: BÖLÜNECEK (anlam degismeden, iki cümle)")
    if uzun:
        print(f"\n  {'kel':>4}  {'bölüm':<44}{'satir':>7}  cümle")
        for k, b, ln, c in uzun:
            im = "★BÖL" if k > ESIK_BOL else "    "
            print(f"  {k:>4}{im} {b[:42]:<44}{ln:>7}  {c[:150]}")
    print(f"\n  ✓ KAPI-29 · rapor kipi · >40: {len(uzun)} · >45: {len(bol)}")
    return 29 if (a.sert and bol) else 0


if __name__ == "__main__":
    sys.exit(main())
