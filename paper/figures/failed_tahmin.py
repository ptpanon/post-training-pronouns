#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/failed_tahmin_satirlari_2026-09-17.json"
KESIM = "results/v98_kesim_parcalari_2026-09-21.json"
ELLE_DEPO = {16}
SAYI = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def tex(s):
    parca = s.split("$")
    return "$".join(p if i % 2 else _tex_duz(p) for i, p in enumerate(parca))


def _tex_duz(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
             .replace("#", r"\#").replace("×", r"$\times$").replace("≥", r"$\geq$").replace("≤", r"$\leq$")
             .replace("→", r"$\rightarrow$").replace("α", r"$\alpha$").replace("Δ", r"$\Delta$").replace("β", r"$\beta$")
             .replace("|", r"$|$").replace("−", "--").replace("–", "--").replace("’", "'").replace("«", "``").replace("»", "''")
             .replace("“", "``").replace("”", "''"))


def _birim(s):
    import re
    R = [(r"(?<![:_/\w-])([Ff])amil(y|ies)\b", lambda m: ("M" if m.group(1) == "F" else "m") + ("odel" if m.group(2) == "y" else "odels")),
         (r"\b([Ll])adders\b", lambda m: m.group(1) + "heckpoint sequences" if m.group(1) == "C" else ("checkpoint sequences" if m.group(1) == "l" else "Checkpoint sequences")),
         (r"\bladder\b", "checkpoint sequence"), (r"\bLadder\b", "Checkpoint sequence"),
         (r"\bend rung\b", "largest model"),
         (r"\brung-by-rung\b", "stage-by-stage"), (r"\brungs\b", "stages"), (r"\brung\b", "stage"),
         (r"\barms\b", "runs"), (r"\barm\b", "run"), (r"\blegs\b", "outputs"), (r"\bleg\b", "output"),
         (r"\bdose-matched\b", "data-size-matched"), (r"\bforce-below-dose\b", "force-below-data-size"),
         (r"\bdose\b", "data size"),
         (r"\btilt\b", "pronoun gap"), (r"\btemplated\b", "chat-template"),
         (r"so the dial is a register dial rather than a person dial", "so what the runs steer is register rather than person marking"),
         (r"\bThe dial size\b", "The shift size"),
         (r"(?<!«different )(?<!«same )\bstep\b", "stage"), (r"\bOLMo2-", "OLMo-2-")]
    for a, b in R:
        s = re.sub(a, b, s)
    return s


def depo_satirlari(S):
    import re
    K = json.load(open(f"{__DNH_ROOT__}/{KESIM}", encoding="utf-8"))
    govde = "\n".join(x["metin"] for x in K["parcalar"])
    bos = [x.get("sira") for x in K["parcalar"] if len((x.get("metin") or "").strip()) < 20]
    if bos or len(K["parcalar"]) != K.get("n_parca", len(K["parcalar"])):
        raise SystemExit(f"{bos}")
    out = set(ELLE_DEPO)
    for s in S:
        m = re.search(r"«([^»]+)»", s.get("kagit_yeri", "") or "")
        if m and m.group(1) in govde:
            out.add(s["i"])
    return out


def main():
    if not os.path.exists(f"{__DNH_ROOT__}/{CARD}"):
        print(""); return 3
    K, pk = load(CARD)
    S = K["satirlar"]; N, M, T = K["N"], K["M"], K["T"]
    top, dog = sum(s["n_dusen"] for s in S), sum(bool(s.get("dogrulandi")) for s in S)
    print(f"{N} {M} {T} {len(S)} {top} {dog}"
          f"")
    if len(S) != M or top != T or dog != M:
        return 3
    depo = depo_satirlari(S)
    N_b, M_b = N - len(depo), M - len(depo)
    cumle = (rf"of the ${N_b}$ registrations that carry a bar or an outcome name fixed in advance and whose results are reported here, "
             rf"${M_b}$ carry at least one prediction that failed")
    io.open(f"{OUT}/DUSEN_TAHMIN_CUMLE.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/failed_tahmin.py)\n" + cumle)
    PARCA = 7
    BAS = [r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{@{}r p{0.19\linewidth} r c p{0.315\linewidth} p{0.315\linewidth}@{}}\toprule",
           r"\# & registration & failed & reported & what the failed prediction said & what was measured \\ \midrule"]
    L = []
    dip = K["dipnot"]
    S = S + [dict(i=M + j + 1, kayit_en=d["kayit_en"], n_dusen=d["n"], nerede="repository") for j, d in enumerate(dip)]
    sirali = sorted(S, key=lambda x: x["i"])
    for k in range(0, len(sirali), PARCA):
        if k:
            L += [r"\bottomrule\end{tabular}\end{center}"]
        L += BAS
        for s in sirali[k:k + PARCA]:
            if s.get("nerede") == "repository" or s["i"] in depo:
                olc_d = (tex(_birim(s["olculen_en"])) if s.get("olculen_en") else
                         r"\emph{in the repository's record of failed predictions}")
                L.append(f"{s['i']} & " + r"\alinti{" + tex(s['kayit_en']) + "}" + f" & {s['n_dusen']} & repository & "
                         + (r"\alinti{" + tex(s["tahmin_en"]) + "}" if s.get("tahmin_en") else "---") + " & " + olc_d + r" \\")
                continue
            olc = tex(_birim(s["olculen_en"])) + (rf" \emph{{({tex(_birim(s['serh']))})}}" if s.get("serh") else "")
            L.append(f"{s['i']} & " + r"\alinti{" + tex(s['kayit_en']) + "}" + f" & {s['n_dusen']} & here & "
                     r"\alinti{" + tex(s['tahmin_en']) + "} & " + olc + r" \\")
    dn = (rf"{SAYI.get(len(depo) + len(dip), str(len(depo) + len(dip))).capitalize()} rows fall outside the ${N_b}$, because their "
          r"results are not reported here: they were scored and they failed, and their wording and score are in the repository's "
          rf"record. {SAYI.get(len(depo), str(len(depo))).capitalize()} of them are passages that moved to the repository when the "
          r"appendix was cut.")
    ayri = (rf"Not counted: {SAYI.get(K['n_islemsel'], K['n_islemsel'])} operational predictions (whether runs would finish) and "
            rf"{SAYI.get(K['n_taban'], K['n_taban'])} base-rate statements.")
    L += [r"\bottomrule\end{tabular}\end{center}",
          r"\vekalet{\footnotesize " + ayri + " " + dn + r" The \emph{what the failed prediction said} column quotes each registration verbatim, in its own "
          r"vocabulary: \terim{family} is its word for what this paper calls a model, and \terim{rung}, \terim{ladder}, "
          r"\terim{arm}, \terim{leg} and \terim{dose} for training stage, checkpoint sequence, run, output and data size; "
          r"\terim{dial}, \terim{recipe} and \terim{step} for the targeted preference runs, a post-training pipeline and a training stage.}"]
    io.open(f"{OUT}/DUSEN_TAHMIN.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/failed_tahmin.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="DUSEN_TAHMIN", ciktilar=["DUSEN_TAHMIN.tex", "DUSEN_TAHMIN_CUMLE.tex"], sources=[pk],
                   payda=dict(N=N, M=M, T=T, N_burada=N_b, M_burada=M_b, n_satir=len(S), n_depo_satir=len(dip) + len(depo), depo_satir=sorted(depo), n_islemsel=K["n_islemsel"], n_taban=K["n_taban"]),
                   note="renders the recount card one line per registration; no new statistic"),
              io.open(f"{OUT}/DUSEN_TAHMIN.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ DUSEN_TAHMIN.tex · DUSEN_TAHMIN_CUMLE.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
