#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, os, re, sys, time
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
OUT = os.path.dirname(os.path.abspath(__file__))
SECIM = f"{OUT}/ORNEK_CIFT.meta.json"
AILE = "Gemma-3-27B"
N_EK = 120
N_PARCA = 20
N_TEZ = 8


def sha16b(b): return hashlib.sha256(b).hexdigest()[:16]
def sha16f(y): return sha16b(open(y, "rb").read())


def kes(nlp, t, n):
    toks = [x for x in nlp.tokenizer(t) if not x.is_space]
    if len(toks) <= n:
        return t, False
    son = toks[n - 1]
    return t[:son.idx + len(son.text)], True


def tex_kac(t):
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "%": r"\%",
           "_": r"\_", "^": r"\^{}", "~": r"\~{}", "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}"}
    return "".join(rep.get(c, c) for c in t)


def vurgula(t, MO, US):
    isaret = []
    for m in MO.SAHIS1.finditer(t):
        if not US.fullmatch(m.group(0)):
            isaret.append((m.start(), m.end(), "fbir"))
    for m in MO.SAHIS2.finditer(t):
        isaret.append((m.start(), m.end(), "fiki"))
    isaret.sort()
    out, i = [], 0
    for a, b, k in isaret:
        out.append(tex_kac(t[i:a])); out.append("\\%s{%s}" % (k, tex_kac(t[a:b]))); i = b
    out.append(tex_kac(t[i:]))
    s = "".join(out)
    s = re.sub(r"\s*\n+\s*", r" $\\hookleftarrow$ ", s).strip()
    return s, sum(1 for x in isaret if x[2] == "fbir"), sum(1 for x in isaret if x[2] == "fiki")


def main():
    import addressee_run as MK, addressee_olcu as MO, form_count as FS, reviewer_readings_v80 as HO
    K = json.load(io.open(SECIM, encoding="utf-8"))
    s = next(x for x in K["secim"] if x["aile"] == AILE)
    nlp = FS._boru()
    cikti, red = {}, []
    for bacak in ("taban", "hizali"):
        z = s["bacak"][bacak]
        y = f"{MK.VERI}/{AILE}/{z}/uretim.jsonl"
        if sha16f(y) != s["sha_uretim"][bacak]:
            red.append(f"{bacak}: generation file sha {sha16f(y)} != card {s['sha_uretim'][bacak]}"); continue
        R = MK.oku(AILE, z)
        r = next((x for x in R if x["kol"] == s["kol"] and x["cekim"] == s["cekim"] and x["istem_i"] == s["istem_i"]), None)
        ek, _ = kes(nlp, r["metin"], N_EK)
        if sha16b(ek.encode()) != s["sha_metin"][bacak]:
            red.append(f"{bacak}: Appendix G text sha {sha16b(ek.encode())} != card {s['sha_metin'][bacak]}"); continue
        Ri = [x for x in R if x["istem_i"] == s["istem_i"]]
        A, _ = MO.topla(nlp, [x["metin"] or "" for x in Ri], n_process=4)
        nj = float(sum(A["n_jeton"])); you = float(sum(A["m1_sahis2"])); iwe = float(HO.p1_sayim(Ri, "cs").sum())
        parca, kesildi = kes(nlp, r["metin"], N_PARCA)
        v, n1, n2 = vurgula(parca, MO, HO.US)
        cikti[bacak] = dict(r=r, n_uretim=len(Ri), nj=nj, you=you, iwe=iwe, you_1k=1000 * you / max(nj, 1), iwe_1k=1000 * iwe / max(nj, 1),
                            tex=v, n1_parca=n1, n2_parca=n2, kesildi=kesildi)
    print(f"  [PAYDA] f0_ornek: aile={AILE} · istem={s['istem_i']} · kol={s['kol']} · cekim={s['cekim']} · "
          f"n_bacak={len(cikti)}/2 · red_kimlik={len(red)} {red} · "
          + " · ".join(f"{b}: jeton={c['nj']:.0f} I/we={c['iwe']:.0f} ({c['iwe_1k']:.1f}/1k) you={c['you']:.0f} "
                       f"({c['you_1k']:.1f}/1k) parcada 1.s={c['n1_parca']} 2.s={c['n2_parca']}" for b, c in cikti.items())
          + " ⇒ esik red_kimlik 0 ⇒ EYLEM: >0 ise figür YAZILMAZ, cikis 3")
    if len(cikti) == 2:
        d = cikti["hizali"]["you_1k"] - cikti["taban"]["you_1k"]
        if abs(d - s["dM1_istem"]) > 1e-6:
            red.append(f"prompt dM1 {d:.6f} != card {s['dM1_istem']:.6f}")
        print(f"{d:+.6f} {s['dM1_istem']:+.6f}"
              f"{[c['n_uretim'] for c in cikti.values()]}")
    if red or len(cikti) != 2:
        return 3
    ist = s.get("tez") or ""
    onek = cikti["taban"]["r"]["onek"]
    i = onek.find(ist) if ist else -1
    if i > 0:
        istem_tex = r"\emph{" + tex_kac(onek[:i].strip()) + "} " + tex_kac(onek[i:].strip())
    else:
        istem_tex = tex_kac(onek.strip())
    _EN = 34.0
    _tepe = max(max(c["iwe_1k"], c["you_1k"]) for c in cikti.values()) or 1.0
    def cubuk(deger, renk):
        w = max(0.6, _EN * deger / _tepe)
        return (r"\textcolor{%s}{\rule[0.3ex]{%.2fpt}{4pt}}" % (renk, w) + r"\,%.1f" % deger)
    _KUTU = (r"colback=white,boxrule=0.6pt,arc=1pt,left=3pt,right=3pt,top=1.5pt,bottom=1.5pt,"
             r"equal height group=f1,"
             r"toptitle=0pt,bottomtitle=0pt,fonttitle=\bfseries\scriptsize,coltitle=white,"
             r"boxsep=0pt")
    def kutu(ad, c, renk):
        return (r"\begin{minipage}[t]{0.492\linewidth}\vspace{0pt}" +
                r"\begin{tcolorbox}[%s,colframe=%s,colbacktitle=%s,title=%s]" % (_KUTU, renk, renk, ad) +
                c["tex"] + (r"\,\ldots" if c["kesildi"] else "") +
                r"\par{\scriptsize I/we " + cubuk(c["iwe_1k"], "okB")
                + r"\quad you " + cubuk(c["you_1k"], "okO")
                + r"\quad(%d outputs)}" % c["n_uretim"] +
                r"\end{tcolorbox}\end{minipage}")
    T = [r"% * URETILDI - elle duzenlenmez (figures/f0_example.py) — Ek G'nin rule-secimli cifti, ORNEK_CIFT.meta.json",
         r"\providecommand{\okB}{}\definecolor{okB}{HTML}{0072B2}\definecolor{okO}{HTML}{D55E00}",
         r"\providecommand{\fbir}[1]{\textcolor{okB}{\underline{\textbf{#1}}}}",
         r"\providecommand{\fiki}[1]{{\setlength{\fboxsep}{0.6pt}\textcolor{okO}{\fbox{\textbf{#1}}}}}",
         r"\begin{figure}[b]\centering\footnotesize",
         r"\begin{tcolorbox}[colback=black!4,colframe=black!25,boxrule=0.4pt,arc=1pt,"
         r"left=3pt,right=3pt,top=1.5pt,bottom=1.5pt,boxsep=0pt]",
         r"\textbf{Prompt} " + istem_tex, r"\end{tcolorbox}\par\nointerlineskip",
         r"\begin{tikzpicture}",
         r"\useasboundingbox (0,0) rectangle (\linewidth,10pt);",
         r"\draw[black!45,line width=0.7pt] (0.5\linewidth,10pt) -- (0.5\linewidth,7pt);",
         r"\draw[black!45,line width=0.7pt] (0.246\linewidth,7pt) -- (0.754\linewidth,7pt);",
         r"\draw[-latex,black!45,line width=0.7pt] (0.246\linewidth,7pt) -- (0.246\linewidth,0.5pt);",
         r"\draw[-latex,black!45,line width=0.7pt] (0.754\linewidth,7pt) -- (0.754\linewidth,0.5pt);",
         r"\end{tikzpicture}\par\nointerlineskip",
         kutu("Base checkpoint", cikti["taban"], "okB") + r"\hfill"
         + kutu("Aligned checkpoint", cikti["hizali"], "okO"),
         r"\caption{\textbf{One prompt, two checkpoints.} One of two Appendix~\ref{app:ornek} pairs, "
         r"picked by a fixed rule for a typical change in ``you'' (Gemma-3-27B); "
         + (r"first-person forms underlined, second-person boxed. " if any(c["n2_parca"] for c in cikti.values())
            else r"first-person forms underlined. ")
         + r"The prefix (one of 16) is italic. Bars are rates per 1,000 tokens on one shared scale.}",
         r"\label{fig:ornek}", r"\end{figure}"]
    io.open(f"{OUT}/F0_ORNEK.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")
    json.dump(dict(table="F0_ORNEK", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=os.path.relpath(SECIM, ROOT), sha256_16=sha16f(SECIM))],
                   secim=dict(aile=AILE, istem_i=s["istem_i"], kol=s["kol"], cekim=s["cekim"]),
                   n_parca=N_PARCA,
                   oran_kapsami="all generations of the prompt (n_uretim per leg)",
                   oran={b: dict(n_uretim=c["n_uretim"], n_jeton=c["nj"], iwe=c["iwe"], you=c["you"], iwe_1k=c["iwe_1k"], you_1k=c["you_1k"])
                         for b, c in cikti.items()},
                   note="example pair identical to Appendix G (two sha checks); rates from the whole output, with the paper's counters"),
              io.open(f"{OUT}/F0_ORNEK.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  ✓ F0_ORNEK.tex + F0_ORNEK.meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
