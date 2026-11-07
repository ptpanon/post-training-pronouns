#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, os, re, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
OUT = os.path.dirname(os.path.abspath(__file__))
CARD = f"{ROOT}/results/PREREG9_A_OLCUM_2026-08-31.json"
KOLT = f"{ROOT}/results/KOL_TABLOSU_2026-09-10.json"
DERLE = f"{ROOT}/paper/build.sh"
AILELER = ["Mistral-7B-v0.3", "Gemma-3-27B"]
N_JETON = 120
UYGUNSUZ = re.compile(r"\b(fuck|shit|cunt|nigg\w*|fag\w*|retard\w*|rape\w*|porn\w*|slut\w*|whore\w*|kill yourself)", re.I)


def sha16b(b): return hashlib.sha256(b).hexdigest()[:16]
def sha16f(y): return sha16b(open(y, "rb").read())


def kapi_sozlukleri():
    s = open(DERLE, encoding="utf-8").read()
    y = re.search(r"YASAK = \{(.*?)\n\}", s, re.S).group(1)
    jargon = [re.compile(p, re.I) for p in re.findall(r'r"((?:[^"\\]|\\.)+)"', y)]
    anon = re.compile(re.search(r'len\(re\.findall\(r"(\(\?<!\[\\w\]\)\(taban.*?)", t\)\)', s).group(1))
    return jargon, anon


def tex_kac(t):
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "%": r"\%",
           "_": r"\_", "^": r"\^{}", "~": r"\~{}", "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}"}
    t = "".join(rep.get(c, c) for c in t)
    return re.sub(r"\n+", r" $\\hookleftarrow$ ", t)


def kes(nlp, t):
    toks = [x for x in nlp.tokenizer(t) if not x.is_space]
    if len(toks) <= N_JETON:
        return t, len(toks), False
    son = toks[N_JETON - 1]
    return t[:son.idx + len(son.text)], N_JETON, True


RULE_YOL = "preregistration/rule_example_pair_selection.md"


def rule_commit():
    import subprocess
    h = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "--abbrev=8", "--diff-filter=A", "--", RULE_YOL],
                       capture_output=True, text=True, check=True).stdout.split()
    assert len(h) == 1 and h[0].startswith("ceb38255"), h
    return h[0][:8]


def main():
    global RULE_COMMIT
    RULE_COMMIT = rule_commit()
    import addressee_run as MK, form_count as FS
    nlp = FS._boru()
    A = json.load(open(CARD, encoding="utf-8"))
    ilk_kol = [k for k, v in json.load(open(KOLT, encoding="utf-8"))["onek"].items() if v["n_sahis2"] == 0][0]
    jargon, anon = kapi_sozlukleri()
    secim = []
    for aile in AILELER:
        v = A[aile]; k0, k1 = v["kontrast"].split("→"); L = {}
        for bacak, z in (("taban", k0), ("hizali", k1)):
            R = MK.oku(aile, z); V = MK.satir_bilesenleri(nlp, R)
            red = abs(MK.olc_toplam(V)["M1"] - v["basamak_tablosu"][z]["M1"])
            assert red <= 1e-6, (aile, z, red)
            L[bacak] = (R, V, f"{MK.VERI}/{aile}/{z}/uretim.jsonl")
        ist = np.array([r["istem_i"] for r in L["taban"][0]]); ish = np.array([r["istem_i"] for r in L["hizali"][0]])
        dp = {}
        for p in sorted(set(ist.tolist())):
            mt = L["taban"][1][ist == p]; mh = L["hizali"][1][ish == p]
            dp[p] = 1000 * mh[:, 1].sum() / mh[:, 0].sum() - 1000 * mt[:, 1].sum() / mt[:, 0].sum()
        med = float(np.median(list(dp.values())))
        sira = sorted(dp, key=lambda p: (abs(dp[p] - med), p))
        dusen = []
        for p in sira:
            bul = lambda R: next((r for r in R if r["kol"] == ilk_kol and r["cekim"] == 0 and r["istem_i"] == p), None)
            rt, rh = bul(L["taban"][0]), bul(L["hizali"][0])
            if not rt or not rh or not rt.get("metin") or not rh.get("metin"):
                dusen.append((p, "missing row")); continue
            ist_m, nj_i, kes_i = kes(nlp, rt["onek"])
            tm, nj_t, kes_t = kes(nlp, rt["metin"]); hm, nj_h, kes_h = kes(nlp, rh["metin"])
            birlesik = " ".join((ist_m, tm, hm))
            if UYGUNSUZ.search(birlesik):
                dusen.append((p, "uygunsuz icerik listesi")); continue
            if any(j.search(birlesik) for j in jargon) or anon.search(birlesik):
                dusen.append((p, "derleme kapisi sözlügü")); continue
            secim.append(dict(aile=aile, istem_i=p, dM1_istem=dp[p], medyan=med, kol=ilk_kol, cekim=0,
                              tez=rt.get("tez"), durus=rt.get("durus"), dusen=dusen,
                              istem=ist_m, taban=tm, hizali=hm, n_jeton=dict(istem=nj_i, taban=nj_t, hizali=nj_h),
                              kesildi=dict(istem=kes_i, taban=kes_t, hizali=kes_h),
                              sha_metin=dict(istem=sha16b(ist_m.encode()), taban=sha16b(tm.encode()), hizali=sha16b(hm.encode())),
                              sha_uretim=dict(taban=sha16f(L["taban"][2]), hizali=sha16f(L["hizali"][2])),
                              bacak=dict(taban=k0, hizali=k1)))
            break
        print(f"  [PAYDA] ornek_cift {aile}: istem {len(dp)} · medyan ΔM1 {med:+.3f} · secilen istem {secim[-1]['istem_i']} "
              f"(ΔM1_p {secim[-1]['dM1_istem']:+.3f}) · düsen aday {len(dusen)} {dusen}", flush=True)
    nm = {"Mistral-7B-v0.3": "Mistral-7B-v0.3", "Gemma-3-27B": "Gemma-3-27B"}
    t = [(r"\vekalet{\textbf{Two base and aligned continuations of one prompt, chosen by a rule written before the choice.} "
         r"For each of two models we computed $\Delta M_1$ per prompt on the raw-continuation debate prompts (all sixteen prefix conditions and twelve draws), "
         r"took the prompt whose $\Delta M_1$ lies closest to the model's median over its $34$ prompts, and printed the first "
         r"prefix condition without a second person of its own (" + tex_kac(secim[0]["kol"]) + r" in Table~\ref{tab:onek}), first draw. The rule skips a prompt whose "
         r"text contains listed offensive terms or would trip our anonymity or vocabulary checks, and says so; each text is cut at its "
         r"first $%d$ tokens (the counter's own tokens) and printed verbatim, line breaks shown as $\hookleftarrow$. " % N_JETON +
         r"Hashes are SHA-256 prefixes of the printed text and of the generation file it comes from. "
         r"The panel assigns each thesis both stances; a continuation defends the stance it was given, and the example is "
         r"the rule-selected prompt, not a chosen one (the selection rule is in the released repository, committed before the "
         r"selection).}"), ""]
    import addressee_olcu as _MO
    from reviewer_readings_v80 import US as _US
    def _kisi(x):
        x = x or ""
        return len(_MO.SAHIS1.findall(x)) - len(_US.findall(x)), len(_MO.SAHIS2.findall(x))
    _say = {s["aile"]: {b_: _kisi(s[b_]) for b_ in ("taban", "hizali")} for s in secim}
    _kontrast = [a_ for a_, v in _say.items() if v["taban"][0] + v["taban"][1] > v["hizali"][0] + v["hizali"][1]]
    print(f"  [PAYDA] ornek_cift_kisi: {_say} · kontrast gösteren {_kontrast} ⇒ esik: gösteren yoksa ⇒ EYLEM: not iki örnege de yazilir")
    for s in secim:
        dus = ("no candidate was skipped" if not s["dusen"] else
               "%d earlier candidate%s skipped (%s)" % (len(s["dusen"]), "" if len(s["dusen"]) == 1 else "s",
                                                        "; ".join(sorted({d[1] for d in s["dusen"]}))))
        dus = dus.replace("uygunsuz icerik listesi", "offensive-term list").replace("derleme kapisi sözlügü", "build-check list")
        t += [r"\paragraph*{%s.}" % nm[s["aile"]],
              r"\vekalet{Model median $\Delta M_1$ $%+.2f$ per thousand; prompt %d, $\Delta M_1$ $%+.2f$; %s. "
              r"Generation files: base \texttt{%s}, aligned \texttt{%s}." % (s["medyan"], s["istem_i"], s["dM1_istem"], dus,
                                                                            s["sha_uretim"]["taban"], s["sha_uretim"]["hizali"])
              + (r" The two printed windows carry $%d$ and $%d$ first-person and $%d$ and $%d$ second-person forms (base, aligned): "
                 r"the rule picks the prompt at the model's median, not a window that shows the fall"
                 % (_say[s["aile"]]["taban"][0], _say[s["aile"]]["hizali"][0],
                    _say[s["aile"]]["taban"][1], _say[s["aile"]]["hizali"][1])
                 + (r"; Figure~\ref{fig:ornek} prints the %s pair, where the base window has $%d$ first-person forms and the "
                    r"aligned window none.}" % (_kontrast[0], _say[_kontrast[0]]["taban"][0])
                    if _kontrast and s["aile"] not in _kontrast else r".}")
                 if sum(_say[s["aile"]]["taban"]) + sum(_say[s["aile"]]["hizali"]) == 0 else r"}"),
              r"\begin{quote}\small\raggedright",
              r"\textbf{Prompt} (\texttt{%s}): %s\par\smallskip" % (s["sha_metin"]["istem"], tex_kac(s["istem"])),
              r"\textbf{Base} (\texttt{%s}): %s\par\smallskip" % (s["sha_metin"]["taban"], tex_kac(s["taban"])),
              r"\textbf{Aligned} (\texttt{%s}): %s" % (s["sha_metin"]["hizali"], tex_kac(s["hizali"])),
              r"\end{quote}", ""]
    io.open(f"{OUT}/ORNEK_CIFT.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/example_pair.py)\n" + "\n".join(t))
    json.dump(dict(table="ORNEK_CIFT", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   rule="preregistration/rule_example_pair_selection.md",
                   sources=[dict(yol=os.path.relpath(CARD, ROOT), sha256_16=sha16f(CARD)),
                            dict(yol=os.path.relpath(KOLT, ROOT), sha256_16=sha16f(KOLT))],
                   secim=[{k: v for k, v in s.items() if k not in ("istem", "taban", "hizali")} for s in secim]),
              io.open(f"{OUT}/ORNEK_CIFT.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ ORNEK_CIFT.tex + ORNEK_CIFT.meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
