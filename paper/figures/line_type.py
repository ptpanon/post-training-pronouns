#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, hashlib, time

ROOT = __DNH_ROOT__ + ""
CARD = f"{ROOT}/results/reviewer_v87_line_type_2026-09-16.json"
TOL = 0.005
BACAK = ("ciplak_sapma_taban", "ciplak_sapma_hizali", "template_sapma_taban", "template_sapma_hizali")
OUT = os.path.dirname(os.path.abspath(__file__))
SIN = ("duzyazi", "liste", "baslik")
AD = {"duzyazi": "prose", "liste": "list", "baslik": "heading"}


def sha16(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def main():
    K = json.load(io.open(CARD, encoding="utf-8"))
    T = K["aileler"]
    eksik = [a for a, v in T.items() if any(b not in v for b in BACAK)]
    if eksik:
        print(f"{len(eksik)}")
        return 3
    gecen = {a for a, v in T.items() if all(v[b] <= TOL for b in BACAK)}
    A = {a: v for a, v in T.items() if a in gecen}
    print(f"{len(T)} {4*len(T)} {len(A)}"
          f"{len(T)-len(A)} {max(v[b] for v in T.values() for b in BACAK):.2e}"
          f"{TOL}")
    if not A:
        print(""); return 3
    n_ayrik = {f"{p}_{c}": sum(1 for v in A.values() if v[f"{p}_{c}"]["sinif"] != "null")
               for p in ("ciplak", "template") for c in SIN}
    print(f"{len(T)} {len(A)} {len(n_ayrik)} {n_ayrik}"
          f"")
    if len(T) != 16:
        return 3
    sat = []
    for a, v in sorted(T.items(), key=lambda kv: kv[1]["ciplak_tam"]["d"]):
        h = [{"Tulu3-8B": "T\\\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}.get(a, a)]
        if a not in A:
            sat.append(h[0] + r" & \multicolumn{6}{c}{\emph{tool record; not reported}} \\"); continue
        for p in ("ciplak", "template"):
            for c in SIN:
                d = v[f"{p}_{c}"]
                s = f"${d['d']:+.1f}$"
                h.append(f"\\textbf{{{s}}}" if d["sinif"] != "null" else s)
        sat.append(" & ".join(h) + r" \\")
    g = (r"\begin{center}\scriptsize\setlength{\tabcolsep}{4.2pt}" "\n"
         r"\begin{tabular}{l rrr @{\hspace{1.0em}} rrr}\toprule" "\n"
         r"& \multicolumn{3}{c}{raw-continuation} & \multicolumn{3}{c}{chat-template} \\"
         "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
         + " & ".join(["model"] + [AD[c] for c in SIN] * 2) + r" \\ \midrule" "\n"
         + "\n".join(sat) + "\n" r"\bottomrule\end{tabular}\end{center}" "\n")
    io.open(f"{OUT}/SATIR_TIPI.tex", "w", encoding="utf-8").write(g)
    cd = sum(1 for v in A.values() if v["ciplak_duzyazi"]["sinif"] == "asagi")
    cl = sum(1 for v in A.values() if v["ciplak_liste"]["sinif"] == "asagi")
    sd = sum(1 for v in A.values() if v["template_duzyazi"]["sinif"] == "yukari")
    sl = sum(1 for v in A.values() if v["template_liste"]["sinif"] == "yukari")
    n = len(A)
    c = (r"The three line classes recombine to each checkpoint's own $M_1$ within $0.005$, half the last printed digit, "
         r"in $%d$ of $16$ models; the rest are marked. Among those $%d$, on the raw-continuation prompt $\Delta M_1$ is negative "
         r"and interval-disjoint on prose lines in $%d$ and on list lines in $%d$, and under the template "
         r"it is positive and disjoint on prose lines in $%d$ and on list lines in $%d$. \textbf{Assignment is by line}: "
         r"a sentence that runs from a prose line into a bullet is not seen." % (n, n, cd, cl, sd, sl))
    io.open(f"{OUT}/SATIR_TIPI_CUMLE.tex", "w", encoding="utf-8").write(c + "\n")
    json.dump(dict(table="SATIR_TIPI", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=os.path.relpath(CARD, ROOT), sha256_16=sha16(CARD))],
                   n_satir=len(sat), n_ayrik=n_ayrik, kapi=dict(tol=TOL, gecen=sorted(A), kayit=sorted(set(T) - set(A))), sayim=dict(ciplak_duzyazi_asagi=cd, ciplak_liste_asagi=cl,
                                                                 template_duzyazi_yukari=sd, template_liste_yukari=sl)),
              io.open(f"{OUT}/SATIR_TIPI.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ {OUT}/SATIR_TIPI.tex ({len(sat)} satir) + SATIR_TIPI_CUMLE.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
