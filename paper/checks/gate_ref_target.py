#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, os, re, sys

REF = re.compile(r"\\(?:ref|autoref)\{([^}]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
KUNYE = re.compile(r"(Table|Figure)\s*(A?\d+)\s*:")
BOLUM = re.compile(r"\\((?:sub)*section)\*?\{")


def _bosla(s: str) -> str:
    for a, b in (("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"),
                 ("\ufb03", "ffi"), ("\ufb04", "ffl"), ("\ufb05", "st"),
                 ("\ufb06", "st")):
        s = s.replace(a, b)
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def _yorumsuz(s: str) -> str:
    return re.sub(r"(?m)(?<!\\)%.*$", "", s)


def taslak_bloklarini_at(s: str) -> str:
    out, i = [], 0
    ac = re.compile(r"\\ifnum\\taslakkipi=1")
    while True:
        m = ac.search(s, i)
        if not m:
            out.append(s[i:]); break
        out.append(s[i:m.start()])
        j, derinlik = m.end(), 1
        tok = re.compile(r"\\(ifnum|ifx|if[a-z]*|fi)\b")
        while derinlik and j < len(s):
            t = tok.search(s, j)
            if not t:
                j = len(s); break
            derinlik += -1 if t.group(1) == "fi" else 1
            j = t.end()
        i = j
    return "".join(out)


def _bagli_metin(g: str, p: int, n: int = 90) -> str:
    d, k, out = 1, p, []
    while k < len(g) and d:
        c = g[k]
        d += (c == "{") - (c == "}")
        if d:
            out.append(c)
        k += 1
    t = "".join(out)
    t = re.sub(r"\\(?:textbf|emph|textit|textsc|text|mathrm)\{", "", t)
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", t)
    t = re.sub(r"[{}$\\]", "", t)
    return re.sub(r"\s+", " ", t).strip()[:n]


def kaynak_hedefleri(kok: str):
    sys.path.insert(0, __DNH_ROOT__ + "/scripts")
    from tex_tree import agac as _agac
    parcalar = [open(os.path.join(kok, y), encoding="utf-8").read()
                for y in _agac("p11.tex", kok)[0]]
    hedef = {}
    for g in parcalar:
        g = _yorumsuz(g)
        for m in LABEL.finditer(g):
            lab = m.group(1)
            once = g[max(0, m.start() - 4000):m.start()]
            cap = once.rfind("\\caption")
            bol = None
            for b in BOLUM.finditer(once):
                bol = b
            if cap >= 0 and (bol is None or cap > bol.start()):
                p = once.index("{", cap) + 1
                hedef[lab] = ("float", _bagli_metin(once, p))
            elif bol is not None:
                hedef[lab] = ("bolum", _bagli_metin(once, bol.end(), 60))
    return hedef


def pdf_metni(yol: str) -> str:
    from pypdf import PdfReader
    return " ".join((p.extract_text() or "") for p in PdfReader(yol).pages)


def denetle(kok: str, pdf: str, kip: str = "submission"):
    hedef = kaynak_hedefleri(kok)
    g = _yorumsuz(open(f"{kok}/p11.tex", encoding="utf-8").read())
    if kip == "submission":
        g = taslak_bloklarini_at(g)
    sys.path.insert(0, __DNH_ROOT__ + "/scripts")
    from tex_tree import agac as _agac
    for f in _agac("p11.tex", kok, kip=kip)[0][1:]:
        g += _yorumsuz(open(os.path.join(kok, f), encoding="utf-8").read())
    atif = sorted(set(REF.findall(g)))
    T = re.sub(r"\s+", " ", pdf_metni(pdf))
    Tb = _bosla(T)
    basili = [_bosla(T[m.end():m.end() + 160]) for m in KUNYE.finditer(T)]
    kirik, cozulen, tur = [], 0, {"float": 0, "bolum": 0, "bilinmeyen": 0}
    for lab in atif:
        h = hedef.get(lab)
        if h is None:
            kirik.append((lab, "KAYNAKTA HEDEF YOK")); tur["bilinmeyen"] += 1; continue
        t, metin = h
        anahtar = _bosla(metin)[:30]
        tur[t] += 1
        if not anahtar:
            kirik.append((lab, "HEDEF METNI BOS")); continue
        ok = any(anahtar in b for b in basili) if t == "float" else anahtar in Tb
        if ok:
            cozulen += 1
        else:
            kirik.append((lab, f"{t.upper()} HEDEFI PDF'TE BASILI DEGIL: «{metin[:52]}…»"))
    return dict(n_atif=len(atif), n_cozulen=cozulen, n_kirik=len(kirik),
                tur=tur, n_basili_kunye=len(basili), kirik=kirik)


def _prova(kok: str, pdf: str):
    T = _bosla(pdf_metni(pdf))
    return {"i_pdf_bos_degil": len(T) > 5000,
            "ii_var_olan_bulunur": _bosla("Related work")[:11] in T,
            "iii_olmayan_bulunmaz": _bosla("BU BASLIK HICBIR PDFTE YOK XQZ") not in T,
            "iv_taslak_blogu_atiliyor": "ZZTEST" not in taslak_bloklarini_at(
                "a\\ifnum\\taslakkipi=1 \\ref{ZZTEST}\\fi b"),
            "v_kip_disi_ref_kaliyor": "ZZKEEP" in taslak_bloklarini_at("a \\ref{ZZKEEP} b")}


def main():
    kok = os.path.dirname(os.path.abspath(__file__)).rsplit("/scripts", 1)[0] \
        + "/paper"
    pdf = f"{kok}/p11_submission.pdf"
    P = _prova(kok, pdf)
    if not all(P.values()):
        print(f"   ★★ KAPI-14 PROVASI DÜSTÜ: {P} ⇒ kapi KOSMADI"); return 4
    R = denetle(kok, pdf)
    print(f"   ★ KAPI-14 ref-hedefi · atif {R['n_atif']} "
          f"(float {R['tur']['float']} · bolum {R['tur']['bolum']}) · "
          f"PDF'te basili kunye {R['n_basili_kunye']} · cozulen {R['n_cozulen']} · "
          f"KIRIK {R['n_kirik']} ⇒ esik 0 ⇒ EYLEM: >0 ise DERLEME DUSER")
    for lab, ned in R["kirik"]:
        print(f"     ★★ {lab}: {ned}")
    return 7 if R["n_kirik"] else 0


if __name__ == "__main__":
    sys.exit(main())
