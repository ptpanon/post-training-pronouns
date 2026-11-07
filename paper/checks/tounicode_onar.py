#!/usr/bin/env python3
from __future__ import annotations
import re, sys
import pypdf
from pypdf.generic import DecodedStreamObject, NameObject

OML = {58: ".", 59: ",", 60: "<", 61: "/", 62: ">"}
OML.update({c: chr(c) for c in range(65, 91)})
OML.update({c: chr(c) for c in range(97, 123)})
for i, ch in enumerate("ΓΔΘΛΞΠΣΥΦΨΩαβγδεζηθικλμνξπρστυφχψω", start=0):
    OML[i] = ch


def cmap(kodlar):
    g = "".join(f"<{k:02X}> <{ord(OML[k]):04X}>\n" for k in sorted(kodlar) if k in OML)
    n = len([k for k in kodlar if k in OML])
    return ("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
            "/CMapName /P11-CMMI def\n/CMapType 2 def\n"
            "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
            f"{n} beginbfchar\n{g}endbfchar\nendcmap\n"
            "CMapName currentdict /CMap defineresource pop\nend\nend").encode("latin-1")


def metin(yol, motor):
    if motor == "fitz":
        import fitz
        return [p.get_text() for p in fitz.open(yol)]
    r = pypdf.PdfReader(yol)
    return [p.extract_text() for p in r.pages]


def benzerlik(a, b):
    from difflib import SequenceMatcher
    na, nb = re.sub(r"\s+", " ", a), re.sub(r"\s+", " ", b)
    return SequenceMatcher(None, na, nb, autojunk=False).ratio()


def main(yol="p11_submission.pdf"):
    once_f, once_p = metin(yol, "fitz"), metin(yol, "pypdf")
    r = pypdf.PdfReader(yol); w = pypdf.PdfWriter(clone_from=yol)
    yamali, kodlar_hepsi = set(), set()
    for pg in w.pages:
        F = pg.get("/Resources", {}).get("/Font", {})
        for k in list(F):
            f = F[k].get_object()
            bf = str(f.get("/BaseFont", ""))
            if "/ToUnicode" in f or "CMMI" not in bf:
                continue
            fc, wid = f.get("/FirstChar"), f.get("/Widths")
            kod = {fc + i for i, x in enumerate(wid) if x} if wid else set(OML)
            kodlar_hepsi |= kod
            st = DecodedStreamObject()
            st.set_data(cmap(kod))
            f[NameObject("/ToUnicode")] = w._add_object(st)
            yamali.add(bf.split("+")[-1])
    if not yamali:
        print("  ★ ToUnicode eksigi YOK ⇒ onarima gerek yok"); return 0
    gec = yol + ".tounicode"
    with open(gec, "wb") as fh:
        w.write(fh)
    sonra_p = metin(gec, "pypdf")
    iyi = kotu = 0
    for i, (f_, p0, p1) in enumerate(zip(once_f, once_p, sonra_p)):
        b0, b1 = benzerlik(f_, p0), benzerlik(f_, p1)
        if b1 > b0 + 1e-6: iyi += 1
        elif b1 < b0 - 1e-6: kotu += 1
    print(f"  [PAYDA] tounicode: n_font={len(yamali)} {sorted(yamali)} · n_kod={len(kodlar_hepsi)} · "
          f"hal_iyilesen_sayfa={iyi} · red_kotulesen_sayfa={kotu} ⇒ esik red 0 ⇒ "
          f"EYLEM: kötülesen varsa PDF'e DOKUNULMAZ (cikis 3)")
    if kotu:
        import os; os.remove(gec)
        print("  ★★ ONARIM REDDEDILDI — en az bir sayfa fitz'den UZAKLASTI"); return 3
    import os; os.replace(gec, yol)
    kalan = sum(len(re.findall(r"[0-9];[0-9]", t)) for t in metin(yol, "pypdf"))
    print(f"  ★ ToUnicode enjekte edildi · «rakam;rakam» pypdf'te {kalan} kaldi (önce "
          f"{sum(len(re.findall(r'[0-9];[0-9]', t)) for t in once_p)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
