#!/usr/bin/env python3
"""Table-row check: every table row of the LaTeX source must be printed in the PDF.

Usage: python gate_table_rows_printed.py <paper.tex> <paper.pdf>
"""
from __future__ import annotations
import io, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tex_tree as TA

RULE = (r"\toprule", r"\midrule", r"\bottomrule", r"\hline", r"\addlinespace")
JETON = re.compile(r"[^\W\d_]+|\d+(?:\.\d+)?", re.U)


def _grup(s, i):
    d = 0
    for k in range(i, len(s)):
        if s[k] == "{" and (k == 0 or s[k - 1] != "\\"):
            d += 1
        elif s[k] == "}" and (k == 0 or s[k - 1] != "\\"):
            d -= 1
            if d == 0:
                return k + 1
    return len(s)


def tabularlar(s):
    out = []
    for m in re.finditer(r"\\begin\{tabular(\*?)\}", s):
        i = m.end()
        n_grup = 2 if m.group(1) == "*" else 1
        for _ in range(n_grup):
            while i < len(s) and s[i] in " \n\t":
                i += 1
            if i < len(s) and s[i] == "{":
                i = _grup(s, i)
        j = s.find(r"\end{tabular", i)
        if j > 0:
            out.append((m.start(), s[i:j]))
    return out


def satirlar(govde):
    parca, d, bas, i = [], 0, 0, 0
    while i < len(govde):
        c = govde[i]
        if c == "{" and (i == 0 or govde[i - 1] != "\\"):
            d += 1
        elif c == "}" and (i == 0 or govde[i - 1] != "\\"):
            d -= 1
        elif c == "\\" and i + 1 < len(govde) and govde[i + 1] == "\\" and d == 0:
            parca.append(govde[bas:i]); i += 2
            m = re.match(r"\s*\[[^\]]*\]", govde[i:])
            if m:
                i += m.end()
            bas = i; continue
        i += 1
    parca.append(govde[bas:])
    out = []
    for p in parca:
        for k in RULE:
            p = p.replace(k, " ")
        p = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", " ", p)
        if "&" in p:
            out.append(p)
    return out


def ilk_hucre_anahtari(satir):
    h = satir.split("&")[0]
    h = re.sub(r"\\(ref|eqref|label|cite[a-z]*)\*?(\[[^\]]*\])*\{[^}]*\}", " ", h)
    h = re.sub(r"\$([^$]*)\$", lambda m: " " + re.sub(r"\\[a-zA-Z]+", " ", m.group(1)) + " ", h)
    for kom, tablo in (('"', dict(u="ü", o="ö", a="ä", U="Ü", O="Ö", A="Ä", i="ï", e="ë")),
                       ("'", dict(e="é", a="á", o="ó", i="í", u="ú", E="É")),
                       ("`", dict(e="è", a="à", o="ò")), ("^", dict(e="ê", o="ô", a="â", i="î", u="û")),
                       ("c", dict(c="c", s="s", C="C", S="S"))):
        for h1, h2 in tablo.items():
            h = re.sub(r"\\" + re.escape(kom) + r"\s*\{?" + h1 + r"\}?", h2, h)
    for _ in range(4):
        h = re.sub(r"\\(textbf|textit|emph|mbox|textsc|texttt|textrm|underline|makecell|shortstack)\s*\{([^{}]*)\}", r" \2 ", h)
    h = re.sub(r"\\[a-zA-Z]+\*?", " ", h)
    h = re.sub(r"[{}\\~]", " ", h)
    j = JETON.findall(h)[:4]
    return j if any(len(t) >= 3 and not t[0].isdigit() for t in j) else None


def pdf_jetonlari(pdf):
    import fitz
    d = fitz.open(pdf)
    akis, tasma = [], []
    for sn, p in enumerate(d):
        H = p.rect.height
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(x["text"] for x in l["spans"]).strip()
                if t and l["bbox"][3] > H + 0.5 and not t.isdigit():
                    tasma.append((sn + 1, round(l["bbox"][3], 1), t[:40]))
        w = p.get_text("words")
        k = 0
        while k < len(w):
            s = w[k][4].replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
            while s.endswith("-") and k + 1 < len(w) and (w[k + 1][5], w[k + 1][6]) != (w[k][5], w[k][6]):
                k += 1
                s = s[:-1] + w[k][4].replace("ﬁ", "fi").replace("ﬂ", "fl")
            akis.extend(JETON.findall(s))
            k += 1
    return akis, tasma


def bitisik_var(akis, anahtar, dizin):
    ilk = anahtar[0]
    for i in dizin.get(ilk, ()):
        if akis[i:i + len(anahtar)] == anahtar:
            return True
    return False


def olc(tex, pdf, sessiz=False):
    dosyalar, eksik = TA.agac(tex, kip="submission")
    if eksik:
        print(f"   ★★ KAPI-21 · agacta cözülemeyen {len(eksik)} dosya ⇒ ÖLCEMEDI"); return 4, None
    taban = os.path.dirname(os.path.abspath(tex))
    akis, tasma = pdf_jetonlari(pdf)
    dizin = {}
    for i, t in enumerate(akis):
        dizin.setdefault(t, []).append(i)
    n_tab = n_sat = n_den = n_yok_toplam = 0
    kayip = []
    for y in dosyalar:
        s = TA._taslak_sil(TA._yorumsuz(io.open(os.path.join(taban, y), encoding="utf-8").read()))
        for bas, g in tabularlar(s):
            n_tab += 1
            sat = satirlar(g)
            yok = []
            for r in sat:
                n_sat += 1
                a = ilk_hucre_anahtari(r)
                if a is None:
                    continue
                n_den += 1
                if not bitisik_var(akis, a, dizin):
                    yok.append(" ".join(a))
            n_yok_toplam += len(yok)
            if yok:
                satir_no = s[:bas].count("\n") + 1
                kayip.append((y, satir_no, len(sat), len(sat) - len(yok), yok))
    R = dict(n_tablo=n_tab, n_kaynak_satir=n_sat, n_denetlenen=n_den, n_denetlenemeyen=n_sat - n_den,
             n_basilmayan=n_yok_toplam, n_tasan_satir=len(tasma), kayip=kayip, tasma=tasma)
    if not sessiz:
        print(f"   ★ KAPI-21 · tablo satiri · tablo {n_tab} · kaynak veri satiri {n_sat} · denetlenen {n_den} "
              f"· DENETLENEMEYEN {n_sat - n_den} · BASILMAYAN {n_yok_toplam} · sayfadan TASAN satir {len(tasma)} "
              f"⇒ esik 0/0 ⇒ EYLEM: >0 ise DERLEME DÜSER (W-1111)")
        for y, sn, nk, nb, yok in kayip:
            print(f"     ✗ {y}:{sn} · kaynak {nk} satir · basili {nb} · bulunamayan: {yok}")
        for t in tasma[:8]:
            print(f"     ✗ tasan: sayfa {t[0]} · y={t[1]} · «{t[2]}»")
        print(f"     [PAYDA] kapi21: n_tablo={n_tab} · n_kaynak_satir={n_sat} · n_denetlenen={n_den} · "
              f"red_denetlenemeyen={n_sat - n_den} · red_basilmayan={n_yok_toplam} · red_tasan={len(tasma)}")
    return (21 if (n_yok_toplam or tasma) else 0), R


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(4)
    rc, _ = olc(sys.argv[1], sys.argv[2])
    sys.exit(rc)
