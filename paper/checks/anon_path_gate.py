#!/usr/bin/env python3
from __future__ import annotations
import glob, re, sys

UZANTI = r"(?:py|json|jsonl|md|sh|ipynb|npy|tsv|csv|yaml|yml|txt)"
DESEN = {
    "dosya_adi": re.compile(r"(?<![\w/.])[\w\-]+\." + UZANTI + r"(?![A-Za-z])"),
    "depo_yolu": re.compile(r"(?<![\w.])(?:docs|scripts|ambar|paper_ideas|data)/[\w./\-]+|/mnt/\S+|~/\S+"),
    "proje_token": re.compile(r"(?<![A-Za-zÀ-ɏ])(?:rule|m[üu]h[üu]r|dial_|seed|notice|card|"
                              r"h[üu]k[üu]m|prediction|template|filtered|exit|headline|handover|report|dönüs)"
                              r"(?![A-Za-zÀ-ɏ])", re.I),
    "tr_harf": re.compile(r"[iIsSgGcC]"),
    "kod_adi": re.compile(r"(?<![A-Za-z0-9_\\])[A-Z][A-Z0-9]*_[A-Z0-9][A-Z0-9_]*(?![A-Za-z0-9_])"),
    "oz_gonderim": re.compile(r"companion (?:instrument )?(?:study|paper)|(?:earlier|previous|prior) work of this "
                              r"(?:research )?program|carries the name from|\bour (?:earlier|previous|prior|companion|other) "
                              r"(?:work|study|paper|studies|papers)\b", re.I),
    "url_depo": re.compile(r"(?i)(?:https?://)?(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|huggingface\.co|"
                           r"codeberg\.org)/[^\s,;)\]}>\"']+"),
}
ANONIM_DEPO = ("github.com/ptpanon/post-training-pronouns",)
_OZEL = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "ozel_desen.txt")
if __import__("os").path.exists(_OZEL):
    DESEN["ozel"] = re.compile(open(_OZEL, encoding="utf-8").read().strip(), re.I)


def _url_norm(u):
    u = re.sub(r"(?i)^(?:https?://)?(?:www\.)?", "", u.strip()).rstrip(".,;:/").lower()
    return re.sub(r"\.git$", "", u).replace("-", "")


def url_izinli(u, izinli):
    n = _url_norm(u)
    return any(n == _url_norm(i) or n.startswith(_url_norm(i) + "/") for i in izinli)


def kaynakca_url(pdf):
    import os
    d, kok = os.path.dirname(os.path.abspath(pdf)), os.path.splitext(os.path.abspath(pdf))[0]
    biblar = []
    if os.path.exists(kok + ".tex"):
        m = re.search(r"\\bibliography\{([^}]*)\}", open(kok + ".tex", encoding="utf-8", errors="replace").read())
        if m:
            biblar = [os.path.join(d, b.strip() + ".bib") for b in m.group(1).split(",")]
    biblar = [b for b in biblar if os.path.exists(b)] or sorted(glob.glob(os.path.join(d, "*.bib")))
    return sorted({m.group(0) for b in biblar for m in DESEN["url_depo"].finditer(open(b, encoding="utf-8").read())}), biblar


def duz(yol):
    from pypdf import PdfReader
    t = " ".join((p.extract_text() or "") for p in PdfReader(yol).pages)
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)


def url_uzat(metin, m):
    u, j = m.group(0), m.end()
    while u.endswith("/"):
        k = re.match(r"\s+([A-Za-z0-9][^\s,;)\]}>\"']*)", metin[j:])
        if not k:
            break
        u += k.group(1); j += k.end()
    return u


def tara(metin, izinli=ANONIM_DEPO):
    H = {ad: [(m.group(0), metin[max(0, m.start() - 40):m.end() + 20].replace("\n", " "))
              for m in rx.finditer(metin)] for ad, rx in DESEN.items()}
    H["url_depo"] = [(url_uzat(metin, m), metin[max(0, m.start() - 40):m.end() + 20].replace("\n", " "))
                     for m in DESEN["url_depo"].finditer(metin)]
    H["url_depo"] = [x for x in H["url_depo"] if not url_izinli(x[0], izinli)]
    return H


def prova():
    kirli = ("Tool continuation_mode_exit.py; rule at preregistration/rule_example_pair_selection.md and "
             "HAZIR_DAL_HEADLINE_2026-09-06.md; dial_verdict; seed 3; ilik; ratio over LOG_JETON; "
             "carried from the companion instrument study; drafted in earlier work of this program")
    temiz = ("Tülu-3 and Köpf et al. (2023); Karthik; skip; Haberman; doi: 10.18653/v1/2023.emnlp-main.290; "
             "google/gemma-4-12B-it; en_core_web_sm; step_60; the selection rule is in the released repository; "
             "This research program was conducted with AI involvement; FORMAT-ONLY; Llama-3.1-8B; a separate instrument study; "
             "released at https://github.com/ptpanon/post-training-pronouns at commit b477baa1; github.com/ptpanon/posttraining-pronouns/tree/main; "
             "https://github.com/tatsu-lab/stanford_alpaca, and https://openreview.net/forum?id=5GszqYNVDF.")
    kirli += ("; code at https://github.com/someone/post-training-pronouns and https://huggingface.co/ptpanon/another-model."
              " See https://github.com/someone/\npost-training-pronouns too.")
    temiz += " Model card: https://huggingface.co/mlabonne/\nNeuralHermes-2.5-Mistral-7B."
    izinli = ANONIM_DEPO + ("https://github.com/tatsu-lab/stanford_alpaca", "https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B")
    if "ozel" in DESEN:
        kirli += " outcome KADRAN-" + DESEN["ozel"].pattern.upper() + "LIK"
    k, t = tara(kirli, izinli), tara(temiz, izinli)
    return {("ix_ozel_yakalanir" if "ozel" in DESEN else "ix_ozel_yok_agac_kipi"): (len(k["ozel"]) == 1) if "ozel" in DESEN else True,"i_dosya_adi_yakalanir": len(k["dosya_adi"]) == 2,
            "ii_depo_yolu_yakalanir": len(k["depo_yolu"]) == 1,
            "iii_proje_token_yakalanir": len(k["proje_token"]) >= 3,
            "iv_tr_harf_yakalanir": len(k["tr_harf"]) >= 1,
            "vi_kod_adi_yakalanir": "LOG_JETON" in [x[0] for x in k["kod_adi"]],
            "vii_oz_gonderim_yakalanir": len(k["oz_gonderim"]) == 2,
            "viii_url_depo_yakalanir": len(k["url_depo"]) == 3,
            "v_ozel_adlar_ve_doi_temiz": sum(len(v) for v in t.values()) == 0}


def main():
    if "--prova" in sys.argv:
        p = prova(); print("★ ANON-2 PROVA:", p); return 0 if all(p.values()) else 4
    pdf = sys.argv[1]; figler = sorted(glob.glob(sys.argv[2])) if len(sys.argv) > 2 else []
    kaynakca, biblar = kaynakca_url(pdf)
    izinli = ANONIM_DEPO + tuple(kaynakca)
    gorulen_izinli = 0
    top, satir = 0, []
    for f in [pdf] + figler:
        try:
            metin = duz(f)
            gorulen_izinli += sum(1 for m in DESEN["url_depo"].finditer(metin) if url_izinli(m.group(0), izinli))
            H = tara(metin, izinli)
        except Exception as e:
            print(f"     ✗ ANON-2 okunamadi: {f}: {str(e)[:80]}"); top += 1; continue
        for ad, v in H.items():
            top += len(v); satir += [(ad, f, s, c) for s, c in v]
    say = {ad: sum(1 for x in satir if x[0] == ad) for ad in DESEN}
    print(f"   ★ ANON-2 · yol/dosya/proje-token · taranan {1 + len(figler)} PDF · " +
          " · ".join(f"{k}={v}" for k, v in say.items()) + f" · IHLAL = {top} ⇒ esik 0 ⇒ EYLEM: >0 ise DERLEME DÜSER")
    print(f"     [PAYDA] url_depo izinli küme {len(izinli)} (anonim depo {len(ANONIM_DEPO)} + kaynakca {len(kaynakca)}, "
          f"{len(biblar)} .bib) · izinli olup görülen {gorulen_izinli}")
    for ad, f, s, c in satir[:15]:
        print(f"     ✗ [{ad}] {f}: {s!r} … {c!r}")
    return 6 if top else 0


if __name__ == "__main__":
    sys.exit(main())
