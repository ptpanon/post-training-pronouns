#!/usr/bin/env python3
from __future__ import annotations
import io, os, re, sys

DESEN = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def _yorumsuz(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def _taslak_sil(s: str) -> str:
    out, i = [], 0
    while True:
        j = s.find("\\ifnum\\taslakkipi=1", i)
        if j < 0:
            out.append(s[i:]); break
        out.append(s[i:j])
        k, d = j + len("\\ifnum\\taslakkipi=1"), 1
        while k < len(s) and d:
            if s.startswith("\\ifnum", k) or s.startswith("\\ifx", k):
                d += 1
            elif s.startswith("\\fi", k):
                d -= 1
                if not d:
                    k += 3; break
            k += 1
        i = k
    return "".join(out)


def agac(kok: str = "p11.tex", taban: str | None = None, derinlik: int = 8,
         kip: str = "hepsi"):
    if taban is None:
        taban = os.path.dirname(os.path.abspath(kok)) or "."
        kok = os.path.basename(kok)
    gor, sira, cik, eksik = set(), [(kok, 0)], [], []
    while sira:
        y, d = sira.pop(0)
        ay = y if os.path.isabs(y) else os.path.join(taban, y)
        if ay in gor:
            continue
        gor.add(ay)
        if not os.path.exists(ay):
            eksik.append(y); continue
        cik.append(os.path.relpath(ay, taban))
        if d >= derinlik:
            continue
        try:
            s = _yorumsuz(io.open(ay, encoding="utf-8").read())
        except OSError:
            eksik.append(y); continue
        if kip == "submission":
            s = _taslak_sil(s)
        for m in DESEN.finditer(s):
            c = m.group(1)
            if not c.endswith(".tex"):
                c += ".tex"
            sira.append((c, d + 1))
    return cik, eksik


def genislet(kok: str = "p11.tex", taban: str | None = None, kip: str = "submission",
             oku=None, derinlik: int = 8):
    if taban is None:
        taban = os.path.dirname(os.path.abspath(kok)) or "."
        kok = os.path.basename(kok)
    oku = oku or (lambda y: io.open(y, encoding="utf-8").read())
    eksik: list[str] = []

    def ac(yol: str, d: int) -> str:
        ay = yol if os.path.isabs(yol) else os.path.join(taban, yol)
        try:
            s = _yorumsuz(oku(ay))
        except (OSError, KeyError, ValueError):
            eksik.append(yol)
            return ""
        if kip == "submission":
            s = _taslak_sil(s)
        if d >= derinlik:
            return s

        def yerine(m):
            c = m.group(1)
            return ac(c if c.endswith(".tex") else c + ".tex", d + 1)
        return DESEN.sub(yerine, s)

    return ac(kok, 0), eksik


def _prova_genislet() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        io.open(os.path.join(t, "k.tex"), "w").write(
            "A \\input{fig/b} C % yorum \\input{fig/yok}\n\\ifnum\\taslakkipi=1 \\input{fig/t}\\fi D \\input{fig/z}")
        os.makedirs(os.path.join(t, "fig"))
        io.open(os.path.join(t, "fig/b.tex"), "w").write("B1 \\input{fig/c} B2")
        io.open(os.path.join(t, "fig/c.tex"), "w").write("CC")
        io.open(os.path.join(t, "fig/t.tex"), "w").write("TASLAK")
        m, e = genislet(os.path.join(t, "k.tex"))
        i = "A B1 CC B2 C" in m and "D" in m and "TASLAK" not in m and "yok" not in m
        ii = e == ["fig/z.tex"]
    return i and ii


def payda(kok: str = "p11.tex", taban: str | None = None):
    d, e = agac(kok, taban)
    print(f"{kok} {len(d)}"
          f"{len(d)-1} {len(e)}"
          f"")
    for y in e:
        print(f"     ✗ cözülemedi: {y}")
    return d, e


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else "p11.tex"
    d, e = payda(k)
    for y in d:
        print(f"     · {y}")
    sys.exit(3 if e else 0)
