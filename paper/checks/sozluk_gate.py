#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, glob, os, re, sys

KOK = __DNH_ROOT__ + ""
P11 = f"{KOK}/paper"
sys.path.insert(0, f"{KOK}/scripts")
from tex_tree import genislet

SOZLUK = [
    ("speaker",       r"\bspeakers?\b",                 "first-person forms / “I”"),
    ("addressee",     r"\baddressees?\b",               "second-person forms / “you”"),
    ("withdrawn",     r"\bwithdrawn\b",                 "falls / drops (fiil «withdraws» serbest)"),
    ("recipe",        r"\brecipes?\b",                  "post-training pipeline"),
    ("ladder",        r"\bladders?\b",                  "checkpoint sequence"),
    ("rung",          r"\brungs?\b",                    "training stage"),
    ("ground",        r"\bgrounds?\b",                  "prompt set"),
    ("leg",           r"\blegs?\b",                     "base output / aligned output"),
    ("arm",           r"\barms?\b",                     "prefix condition / training run"),
    ("bare",          r"\bbare\b",                      "raw-continuation prompt"),
    ("templated",     r"\btemplated\b",                 "chat-template prompt"),
    ("URIAL-frame",   r"\bURIAL[ -]frame\b",            "in-context assistant prompt"),
    ("tilt",          r"\btilt(?:s|ed)?\b",             "pronoun gap between chosen and rejected responses"),
    ("dose",          r"\bdos(?:e|es|age)\b",           "data size / selection strength"),
    ("dial",          r"\bdials?\b",                    "steerable property"),
    ("cell",          r"\bcells?\b",                    "(gövdeden cikar; Ek'te tanimli)"),
    ("shelf",         r"\bshelf\b|D-SHELF",             "(gövdeden cikar; Ek'te tanimli)"),
    ("format-only",   r"\bformat-only\b",               "(gövdeden cikar; Ek'te tanimli)"),
    ("M-sembolu",     r"\$?M_\{?\d\}?\$?",              "the second-person rate · impersonal-correction share"),
    ("force",         r"\bforce\b",                     "(§1'den cikar; Ek D.2'ye tek gönderme)"),
    ("URIAL",         r"\bURIAL\b",                     "(tanim parantezinde bir kez)"),
    ("family-birim",  r"(?<!model )(?<!cross-)(?<!six )(?<!reward-model )\bfamil(?:y|ies)\b", "model / models (birim)"),
    ("register-battery", r"\bregister batter(?:y|ies)\b", "the register counters"),
    ("neutral-panel", r"\bneutral panels?\b",          "the debate prompts"),
    ("M2-M5",         r"\bM[2-5]\b",                   "(gövdeden cikar; Ek'te tanimli)"),
    ("force-total",   r"\bforce totals?\b",            "directive-force count"),
    ("DeltaU",        r"\\Delta\s*U\b|ΔU",             "(gövdeden cikar)"),
    ("status-acts",   r"\bstatus acts?\b",             "(gövdeden cikar)"),
    ("DOM/ARO/TON",   r"\b(?:DOM|ARO|TON)\b",          "dominance / arousal / tone"),
    ("slot",          r"\bslots?\b",                   "(gövdeden cikar)"),
    ("owner",         r"\bowners?\b",                  "the authors (belgenin hicbir yerinde)"),
]
BUYUK_KUCUK_DUYARLI = {"M-sembolu", "M2-M5", "DeltaU", "DOM/ARO/TON"}
UST_YORUM = [r"load-bearing", r"part of the claim", r"we say so", r"we mark it", r"is the finding",
             r"we do not claim otherwise", r"\bnamed\b(?=\s*[:.])", r"is itself the evidence"]


def _dengeli_sil(s: str, komut: str, yerine: str = " ") -> str:
    out, i, anah = [], 0, "\\" + komut + "{"
    while True:
        j = s.find(anah, i)
        if j < 0:
            out.append(s[i:]); break
        out.append(s[i:j]); k, d = j + len(anah), 1
        while k < len(s) and d:
            d += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        out.append(yerine); i = k
    return "".join(out)


def govde_metni(kok: str = f"{P11}/p11.tex", oku=None):
    m, eksik = genislet(kok, kip="submission", oku=oku)
    m = _dengeli_sil(m, "verdict")
    a, b = m.find("\\begin{abstract}"), m.find("\\appendix")
    return m[a:b], m[b:], eksik


def _temizle(s: str) -> str:
    s = s.replace("\\label{tab:glossary}", " SOZLUKTABLOSU ")
    s = re.sub(r"\\label\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:ref|eqref|pageref)\{([^}]*)\}", r" REF[\1] ", s)
    s = re.sub(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{[^}]*\}", " CITE ", s)
    s = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\bibliography(?:style)?\{[^}]*\}", " ", s)
    return s


def _bolumler(g: str):
    bs = [(m.start(), m.group(1)) for m in re.finditer(r"\\(?:sub)?section\*?\{([^}]*)\}", g)]
    return bs


def _bolum_adi(bs, i):
    ad = "özet"
    for p, n in bs:
        if p <= i:
            ad = n
    return ad


def _tablolar(g: str):
    return [(m.start(), m.end(), "SOZLUKTABLOSU" in m.group(0))
            for m in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", g, re.S)]


def _cumleler(s: str):
    s = re.sub(r"\s+", " ", s)
    return [c.strip() for c in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", s) if c.strip()]


W_YOU = (re.compile(r"\bwithdraw\w*", re.I), re.compile(r"\byou\b", re.I))


def kisi_lafzi(t: str):
    t = _temizle(_dengeli_sil(t, "verdict"))
    bs = _bolumler(t)
    W, Y = W_YOU
    out, i = [], 0
    for c in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", t):
        if W.search(c) and Y.search(c):
            out.append(["withdraw+you", _bolum_adi(bs, i), re.sub(r"\s+", " ", c)[:150], i])
        i += len(c) + 1
    return out


def tara(g: str, ek: str = ""):
    g_t = _temizle(g)
    bs = _bolumler(g_t)
    tab = _tablolar(g_t)
    maske = list(g_t)
    for a, b, sozluk in tab:
        if sozluk:
            for k in range(a, b):
                maske[k] = " "
    gm = "".join(maske)
    gm_ref = gm
    gm = re.sub(r"REF\[[^\]]*\]", lambda m: "REF[" + "_" * (len(m.group(0)) - 5) + "]", gm)
    ihlal, serbest = [], []
    tablo_ici = lambda i: any(a <= i < b for a, b, _ in tab)
    for ad, d, _ in SOZLUK:
        for m in re.finditer(d, gm, re.I if ad not in BUYUK_KUCUK_DUYARLI else 0):
            i = m.start(); bol = _bolum_adi(bs, i)
            bag = re.sub(r"\s+", " ", gm[max(0, i - 50):i + 50])
            if ad == "M-sembolu" and tablo_ici(i):
                serbest.append((ad, bol, "tablo ici sembol")); continue
            ihlal.append([ad, bol, bag, i])
    ff = sorted([x for x in ihlal if x[0] == "force" and not x[1].startswith("Introduction")], key=lambda x: x[3])
    if ff and "REF[app:D2]" in _cumle(gm_ref, ff[0][3]):
        for x in ff:
            ihlal.remove(x); serbest.append(("force", x[1], "ilk gecisi Ek D.2'ye giden tanim cümlesi"))
    uu = [x for x in ihlal if x[0] == "URIAL"]
    if len(uu) == 1:
        ihlal.remove(uu[0]); serbest.append(("URIAL", uu[0][1], "tanim parantezi"))
    govde_disi = gm_ref.find("AI use statement")
    govde = gm_ref[:govde_disi] if govde_disi > 0 else gm_ref
    govde_duz = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", " ", govde, flags=re.S)
    cum = _cumleler(re.sub(r"REF\[[^\]]*\]|CITE", "X", govde_duz))
    iki_nokta = [c for c in cum if c.count(":") >= 2]
    not_but = [c for c in cum if re.search(r"\bnot\b[^.;:]{1,70}?,?\s+but\b", c)]
    ust = [c for c in cum if any(re.search(p, c, re.I) for p in UST_YORUM)]
    ek_etiket = set(re.findall(r"\\label\{([^}]*)\}", ek))
    refs = re.findall(r"REF\[([^\]]*)\]", govde)
    ek_ref = [r for r in refs if r in ek_etiket or r.startswith("app:")]
    beyan_ref = [r for r in re.findall(r"REF\[([^\]]*)\]", gm_ref[govde_disi:] if govde_disi > 0 else "")
                 if r in ek_etiket or r.startswith("app:")]
    duz = re.sub(r"\\begin\{(table|tabular|figure)\*?\}.*?\\end\{\1\*?\}", " ", govde, flags=re.S)
    duz = re.sub(r"\\section\*?\{[^}]*\}", " ", duz)
    noktali = duz.count(";")
    cizgi = len(re.findall(r"(?<!-)---?(?!-)", duz))
    ihlal.extend(kisi_lafzi(g))
    yan = dict(iki_nokta=iki_nokta, not_but=not_but, ust_yorum=ust, ek_ref=ek_ref, beyan_ek_ref=beyan_ref,
               serbest=serbest, noktali=noktali, cizgi=cizgi)
    return ihlal, yan


EK_TERIM = {"ladder", "rung", "ground", "leg", "arm", "bare", "templated", "tilt", "dose", "family-birim", "owner",
            "dial", "recipe", "neutral-panel"}


def tara_ek(ek: str):
    t = _temizle(ek)
    t = re.sub(r"\\begin\{(table\*?|minipage)\}(?:(?!\\end\{(?:table|minipage)).)*?SOZLUKTABLOSU.*?\\end\{\1\}", " SOZLUK ", t, flags=re.S)
    t = re.sub(r"\\begin\{quote\}.*?\\end\{quote\}", " ALINTI ", t, flags=re.S)
    t = _dengeli_sil(t, "alinti", " ALINTI ")
    t = _dengeli_sil(t, "terim", " TERIM ")
    t = re.sub(r"REF\[[^\]]*\]", lambda m: "REF[" + "_" * (len(m.group(0)) - 5) + "]", t)
    t = re.sub(r"\b[Bb]are(\s+)imperatives?\b", "IMPERATIVE", t)
    ih = []
    for ad, d, _ in SOZLUK:
        if ad not in EK_TERIM:
            continue
        for m in re.finditer(d, t, re.I if ad not in BUYUK_KUCUK_DUYARLI else 0):
            ih.append([ad, "Ek", re.sub(r"\s+", " ", t[max(0, m.start() - 50):m.start() + 50]), m.start()])
    return ih


def _cumle_no(s, i):
    return len(re.findall(r"(?<=[.!?])\s+(?=[A-Z\\])", s[:i]))


def _cumle(s, i):
    a = max(s.rfind(". ", 0, i), s.rfind("? ", 0, i)) + 1
    b = s.find(". ", i)
    return s[a:b if b > 0 else len(s)]


def figur_tara(g: str):
    try:
        import fitz
    except ImportError:
        return None, []
    yollar = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", g)
    vur = []
    for y in yollar:
        p = os.path.join(P11, y if y.endswith(".pdf") else y + ".pdf")
        if not os.path.exists(p):
            continue
        t = " ".join(pg.get_text() for pg in fitz.open(p))
        for ad, d, _ in SOZLUK:
            if ad in ("M-sembolu", "force", "URIAL"):
                continue
            for m in re.finditer(d, t, re.I):
                vur.append((ad, y, t[max(0, m.start() - 25):m.start() + 25].replace("\n", " ")))
    return len(yollar), vur


def prova():
    ates = ["The speaker is withdrawn.", "the addressee recedes", "each recipe", "the ladder's rung",
            "on the neutral ground", "the base leg", "sixteen prefix arms", "on the bare prompt", "templated legs",
            "the URIAL frame", "the data's tilt", "at 7.4 times the dose", "a dial nobody holds", "the filtered cells",
            "the D-SHELF reading", "\\textsc{format-only}", "$M_1$ falls", "force is the strength",
            "in 12 of 16 families", "each family's own template",
            "the register battery", "on the neutral panel", "M2 and M3 rise", "the force total", "$\\Delta U$ moves",
            "status acts rise", "the DOM prefix", "each slot", "the owner decided", "the pronoun rate is a dial"]
    sessiz = ["Post-training withdraws first-person forms; we call this fall a withdrawal.",
              "Table~\\ref{tab:rungs} and \\S\\ref{sec:dial} are labels, not printed words.",
              "The second-person rate falls on the raw-continuation prompt and splits under chat templates.",
              "two post-training pipelines on one base", "each training stage in a checkpoint sequence",
              "the pronoun gap between chosen and rejected responses", "base output and aligned output",
              "16 open models from six families and five lineages", "a judge from another model family",
              "a cross-family judge", "in 12 of 16 models",
              "prefixes on three axes (arousal, dominance and tone)", "on the debate prompts", "one of the authors decided",
              "a setting that is easy to move", "a ton of tone"]
    poz = []
    for s in ates:
        g = "\\section{Introduction} " + s + " \\section{Measurement} x."
        ih, _ = tara(g)
        poz.append(bool(ih))
    neg = []
    for s in sessiz:
        g = "\\section{Introduction} " + s + " \\section{Measurement} x."
        ih, _ = tara(g)
        neg.append(not ih)
    g = ("\\section{Introduction} Q (URIAL, CITE). \\section{Results} Directive force moves in no one direction "
         "(Appendix~\\ref{app:D2}). \\begin{table} $M_1$ & x \\end{table} "
         "\\begin{table} \\label{tab:glossary} bare & raw continuation \\end{table} "
         "\\section{Discussion} We call such a property a setting, meaning one anyone can set. \\section*{Limitations} y.")
    ih, yan = tara(g)
    neg.append(not ih)
    g2 = g.replace("\\section{Discussion} We call", "\\section{Discussion} A dial. We call")
    ih2, _ = tara(g2)
    poz.append(bool(ih2))
    g3 = g.replace("Q (URIAL, CITE).", "Q (URIAL, CITE). Force is strong.")
    ih3, _ = tara(g3)
    poz.append(bool(ih3))
    g4 = g.replace("\\section{Results} Directive", "\\section{Results} Force rises. Directive")
    ih4, _ = tara(g4)
    poz.append(bool(ih4))
    g5 = g.replace("\\section{Discussion} We call", "\\section{Discussion} It moves further than force. We call")
    ih5, _ = tara(g5)
    neg.append(not ih5)
    ek_poz = [bool(tara_ek(x)) for x in ["the base leg", "on the neutral ground", "sixteen prefix arms",
                                         "the ladder's rung", "on the bare prompt", "templated legs",
                                         "the data's tilt", "at this dose", "in 12 of 16 families", "the owner decided",
                                         "the dial under the chat template", "one recipe at three scales", "read on the neutral panel"]]
    ek_neg = [not tara_ek(x) for x in ["base outputs and aligned outputs", "each training stage in a checkpoint sequence",
                                       "on the raw-continuation prompt", "under the chat template",
                                       "$M_1$ and $M_3$ stay, with \\textsc{format-only} and \\textsc{unmeasurable}",
                                       "a bare imperative has no subject", "in 12 of 16 models",
                                       "the pronoun gap at this data size", "sixteen prefix conditions", "one of the authors decided",
                                       "the targeted preference runs", "one pipeline at three scales", "read on the debate prompts",
                                       r"\alinti{the person dial} and \terim{recipe}"]]
    wy_poz = [bool(kisi_lafzi(x)) for x in
              ["Post-training withdraws ``you'' from the outputs.",
               "The withdrawal of ``you'' is the finding.",
               "Alignment withdraws the second person, so you see fewer of them."]]
    wy_neg = [not kisi_lafzi(x) for x in
              ["Post-training withdraws the first person.",
               "The second-person rate falls in 16 of 16 models.",
               "``You'' drops on raw continuation and its density splits under chat templates.",
               "We call the fall in the first person a withdrawal."]]
    return {"i_eski_terim_atesler": f"{sum(poz)}/{len(poz)}", "ii_yeni_sozluk_sessiz": f"{sum(neg)}/{len(neg)}",
            "iii_ek_atesler": f"{sum(ek_poz)}/{len(ek_poz)}", "iv_ek_sessiz": f"{sum(ek_neg)}/{len(ek_neg)}",
            "v_withdraw_you_atesler": f"{sum(wy_poz)}/{len(wy_poz)}", "vi_withdraw_you_sessiz": f"{sum(wy_neg)}/{len(wy_neg)}",
            "tamam": all(poz) and all(neg) and all(ek_poz) and all(ek_neg) and all(wy_poz) and all(wy_neg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prova", action="store_true")
    ap.add_argument("--ayrinti", action="store_true")
    ap.add_argument("--uyari", action="store_true", help="ihlalde derleme düsmez (gecis sürerken)")
    a = ap.parse_args()
    P = prova()
    print(f"   ★ KAPI-27 PROVA · {P}")
    if a.prova:
        return 0 if P["tamam"] else 27
    if not P["tamam"]:
        print(""); return 27
    g, ek, eksik = govde_metni()
    ih, yan = tara(g, ek)
    ih_ek = tara_ek(ek)
    n_fig, fv = figur_tara(g)
    say = {}
    for x in ih:
        say[x[0]] = say.get(x[0], 0) + 1
    print(f"   ★ KAPI-27 · gövde sözlügü · gövde {len(g)} kar (özet → \\appendix, \\input yerinde, cözülemeyen {len(eksik)}) · "
          f"sözlük {len(SOZLUK)} terim · ESKI TERIM = {len(ih)} {dict(sorted(say.items(), key=lambda x: -x[1]))} · "
          f"figür metni ({n_fig} PDF) = {len(fv)} ⇒ esik 0 ⇒ EYLEM: >0 ise "
          f"{'UYARI (gecis kipi)' if a.uyari else 'DERLEME DÜSER'}")
    say_ek = {}
    for x in ih_ek:
        say_ek[x[0]] = say_ek.get(x[0], 0) + 1
    print(f"   ★ KAPI-27 · EK sözlügü · ek {len(ek)} kar · taranan {len(EK_TERIM)} terim "
          f"(M1–M5 · FORMAT-ONLY · UNMEASURABLE Ek'te KALIR) · ESKI TERIM = {len(ih_ek)} "
          f"{dict(sorted(say_ek.items(), key=lambda x: -x[1]))} ⇒ esik 0 ⇒ EYLEM: >0 ise "
          f"{'UYARI (gecis kipi)' if a.uyari else 'DERLEME DÜSER'}")
    wy_ek = kisi_lafzi(ek)
    print(f""
          f"{len([x for x in ih if x[0] == 'withdraw+you'])}"
          f"{'UYARI' if a.uyari else 'DERLEME DÜSER'} {len(wy_ek)}")
    print(f"     serbest gecis: {len(yan['serbest'])} {sorted({(s[0], s[2]) for s in yan['serbest']})}")
    print(f"{len(yan['iki_nokta'])} {len(yan['not_but'])}"
          f"{len(yan['ust_yorum'])} {yan['noktali']} {yan['cizgi']}"
          f"{len(yan['ek_ref'])}"
          f"{len(yan['beyan_ek_ref'])}")
    if a.ayrinti:
        for x in ih:
            print(f"     ✗ [{x[0]}] {x[1][:28]}: …{x[2]}…")
        for ad, y, c in fv:
            print(f"     ✗ figür [{ad}] {y}: …{c}…")
        for k in ("iki_nokta", "not_but", "ust_yorum"):
            for c in yan[k]:
                print(f"     · {k}: {c[:160]}")
        from collections import Counter
        print("     · Ek hedefleri:", dict(Counter(yan["ek_ref"])))
    print(f"   [PAYDA] kapi27: n_terim={len(SOZLUK)} · hal_eski={len(ih)} · hal_figur={len(fv)} · "
          f"hal_serbest={len(yan['serbest'])} · hal_withdrawyou={len([x for x in ih if x[0] == 'withdraw+you'])} · "
          f"hal_ekyou={len(wy_ek)} · n_ek_ref={len(yan['ek_ref'])}")
    bad = len(ih) + len(fv)
    return 0 if (bad == 0 or a.uyari) else 27


if __name__ == "__main__":
    sys.exit(main())
