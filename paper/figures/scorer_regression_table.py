#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, hashlib, time
ROOT = __DNH_ROOT__ + ""
ANA = f"{ROOT}/results/judge_signature5_2026-09-10.json"
ALTI = f"{ROOT}/results/judge_signature4_2026-09-06.json"
BOY = f"{ROOT}/results/judge_signature2_2026-09-03.json"
ESL = f"{ROOT}/results/ultrafeedback_matching_ratio_2026-09-12.json"
UC = f"{ROOT}/results/judge_signature3_2026-09-05.json"
ILK = f"{ROOT}/results/judge_signature_2026-09-03.json"
OUT = os.path.dirname(os.path.abspath(__file__))

AD = {"M1_1k": r"second person ($M_1$)", "SAHIS1_1k": "first person",
      "SAHIS3_1k": "third-person density", "ACIKLAMA_SORUSU": "offer question",
      "modal_1k": "modal density (matched, %(om).2f$\\times$)",
      "hedge_1k": "hedge density (unmatched, %(oh).2f$\\times$)", "RED": "refusal indicator",
      "LISTEKOD": "list/code line share", "LOG_JETON": r"$\log$ tokens",
      "LOG_CUMLE": r"$\log$ sentences"}
SIMGE = {"LOG_JETON": r"\log\,\mathrm{tokens}", "LOG_CUMLE": r"\log\,\mathrm{sentences}"}
BOYUT_AD = {"instruction_following": "instruction following",
            "helpfulness": "helpfulness", "truthfulness": "truthfulness",
            "honesty": "honesty"}


def sha16(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def _b(x):
    return f"${x:+.3f}$"


def _ci(c):
    return f"$[{c[0]:+.3f},{c[1]:+.3f}]$" if c else "---"


def main():
    A = json.load(io.open(ANA, encoding="utf-8"))
    S = json.load(io.open(ALTI, encoding="utf-8"))
    B = json.load(io.open(BOY, encoding="utf-8"))
    C = json.load(io.open(UC, encoding="utf-8"))
    E = json.load(io.open(ESL, encoding="utf-8"))
    F1 = json.load(io.open(ILK, encoding="utf-8"))
    b_ilk = F1["_sonuc"]["gercek"]["M1_1k"]["beta"]
    O, KU, HU = A["_ortak"], A["_kunye"], A["_verdict"]
    yuzey = list(KU["yuzey"])
    basilan = [k for k in AD if k in O]
    eksik = [k for k in yuzey if k not in basilan]
    fazla = [k for k in basilan if k not in yuzey and k not in ("M1_1k", "SAHIS1_1k")]
    print(f"{len(yuzey)} {len(basilan)}"
          f"{len(eksik)} {len(fazla)}"
          f"{len(yuzey)}")
    if eksik or fazla:
        print(f"{eksik} {fazla}")
        return 3
    print(f"  [PAYDA] t12_tek_card: n_terim={len(basilan)} · card=IMZASI5 · "
          f"red_karisik=0 ⇒ esik: birden fazla card ⇒ EYLEM: «one joint "
          f"regression» künyesi YAZILAMAZ")

    n = len(yuzey)
    kelime = {6: "six", 7: "seven", 8: "eight", 9: "nine"}.get(n, str(n))
    L = [r"\begin{table}[tb]", r"\centering\small",
         r"\caption{\textbf{Person marking is associated with lower scores, and it "
         r"stands above the surface controls.} Standardised coefficients from "
         r"\emph{one} joint regression on UltraFeedback ($n=%s$ responses, $%s$ "
         r"prompts). \textbf{The dependent variable is the rater's single "
         r"\texttt{overall\_score}}, not the mean of the four rating dimensions; "
         r"those are read separately below the rule and are descriptive. "
         r"Every one of the %s surface controls the fit carries is printed. "
         r"\textbf{The margin depends on which control it is read against, and we "
         r"give both.} Against the largest surface term of either sign, which is the "
         r"length reward, the ratio $|\beta_{M_1}|/|\beta_{%s}|$ is $%.2f$ "
         r"in this fit and $%.2f$ in the %s-control fit that preceded it; the "
         r"pre-registered bar of $1.5$ falls between them, so we report the margin as "
         r"\emph{unresolved}. The person coefficient itself is stable across the four "
         r"pre-registered fits that carry the capacity-matched density controls "
         r"(%s); the first fit, which carries only the two length terms, reads "
         r"$%+.3f$, so those controls account for a $%.0f\%%$ reduction and "
         r"everything after them for $0.003$. Refusals are counted passively and "
         r"no refusal mechanism was probed or weakened.}"
         % ("{:,}".format(KU["n_yanit"]).replace(",", "{,}"),
            "{:,}".format(KU["n_istem"]).replace(",", "{,}"), kelime,
            SIMGE.get(HU["en_guclu_yuzey"], r"\mathrm{%s}" % HU["en_guclu_yuzey"].replace("_", r"\_")), HU["marj_yuzey"],
            S["_verdict"]["marj_yuzey"], {6: "six"}.get(len(S["_kunye"]["yuzey"]),
                                                      str(len(S["_kunye"]["yuzey"]))),
            ", ".join("$%+.3f$" % v for v in
                      (B["_ortak"]["M1_1k"]["beta"],
                       C["_ortak"]["M1_1k"]["beta"],
                       S["_verdict"]["beta_M1"], HU["beta_M1"])),
            b_ilk, 100 * (1 - abs(B["_ortak"]["M1_1k"]["beta"] / b_ilk))),
         r"\label{tab:scorer}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"term & $\beta_{\mathrm{std}}$ & 95\% CI (prompt) & 95\% CI (two-way) & "
         r"$\mathrm{frac}_{\mathrm{null}}$ \\", r"\midrule"]
    sira = ["M1_1k", "SAHIS1_1k", "SAHIS3_1k", "ACIKLAMA_SORUSU", "modal_1k",
            "hedge_1k", "RED", "LISTEKOD", "LOG_JETON", "LOG_CUMLE"]
    _or = dict(om=E["oran_modal"], oh=E["oran_hedge"])
    _nperm = int(KU.get("n_perm") or 200)
    for k in sira:
        v = O[k]
        _ad = AD[k] % _or if "%(" in AD[k] else AD[k]
        _fr = (r"$<%.3f$" % (1.0 / _nperm)) if v["frac_null"] <= 0 \
            else (r"$%.3f$" % v["frac_null"])
        L.append(f"{_ad} & {_b(v['beta'])} & {_ci(v.get('ci'))} & "
                 f"{_ci(v.get('ci_iki_yonlu'))} & {_fr} \\\\")
    L += [r"\midrule",
          r"\multicolumn{5}{l}{\emph{By rating dimension} ($\beta_{M_1}$, "
          r"prompt-clustered CI; descriptive, first fit):}\\"]
    R = B["_rating_betimsel"]
    for d in ("instruction_following", "helpfulness", "truthfulness", "honesty"):
        v = R[d]["M1_1k"]
        L.append(f"{BOYUT_AD[d]} & {_b(v['beta'])} & {_ci(v['ci'])} & --- & --- \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    io.open(f"{OUT}/T12_scorer.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(table="T12_scorer",
                   damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=os.path.relpath(y, ROOT), sha256_16=sha16(y),
                                 rol=r) for y, r in ((ANA, "ana fit"),
                                                     (ESL, "esleme oranlari"),
                                                     (ALTI, "alti kontrollü marj"),
                                                     (UC, "ücüncü fit β"),
                                                     (BOY, "boyut betimi"))],
                   bagimli_degisken="overall_score",
                   n_yuzey=n, yuzey=yuzey, terimler=sira,
                   marj_sekiz=HU["marj_yuzey"], marj_alti=S["_verdict"]["marj_yuzey"],
                   beta_ilk_fit=b_ilk, n_fit=5, n_perm=_nperm,
                   esleme=dict(modal=E["oran_modal"], hedge=E["oran_hedge"],
                               kaynak=os.path.relpath(ESL, ROOT)),
                   serh="v32c'de tablo IKI ayri fitten karisikti; tek karta alindi"),
              io.open(f"{OUT}/T12_scorer.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"✓ T12_scorer.tex ({len(sira)} terim, tek card, marj {HU['marj_yuzey']:.2f} "
          f"↔ {S['_verdict']['marj_yuzey']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
