#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os

ROOT = __DNH_ROOT__ + ""
OUT = os.path.dirname(os.path.abspath(__file__))
M = json.load(io.open(f"{ROOT}/results/generic_vs_addressed_you_2026-09-10.json", encoding="utf-8"))
E = json.load(io.open(f"{ROOT}/results/generic_vs_addressed_you_hand_labels_2026-09-10.json", encoding="utf-8"))
B = json.load(io.open(f"{ROOT}/results/jh_b100_2026-09-25.json", encoding="utf-8"))
assert B["prereg"].endswith("@550b5754") and B["n"] == 100
bj, bnj = B["ozet"]["jenerik"]; bh, bnh = B["ozet"]["hitap"]
bbel, bcoz = B["ozet"]["belirsiz"], B["ozet"]["belirsiz_cozulen"]

md = [x for x in E["madde"] if x.get("el_etiketi")]
def kesinlik(sinif):
    alt = [x for x in md if x["alet_etiketi"] == sinif]
    return sum(1 for x in alt if x["el_etiketi"] == sinif), len(alt)
gj, nj = kesinlik("JENERIK")
gh, nh = kesinlik("HITAP")
bel = [x for x in md if x["alet_etiketi"] == "BELIRSIZ"]
bel_coz = sum(1 for x in bel if x["el_etiketi"] != "BELIRSIZ")
uy = sum(1 for x in md if x["el_etiketi"] == x["alet_etiketi"])

A = {k: v for k, v in M["aile"].items() if v.get("hal") == "ÖLCÜLDÜ"}
n_j = sum(1 for v in A.values() if v["d_jenerik"] < 0)
n_h = sum(1 for v in A.values() if v["d_hitap"] < 0)
pay_bel = sum(v["hizali"]["BELIRSIZ"] + v["taban"]["BELIRSIZ"] for v in A.values()) / \
          max(sum(v["hizali"]["n_cumle_you"] + v["taban"]["n_cumle_you"] for v in A.values()), 1)

L = [r"% ★ ÜRETILDI — elle düzenlenmez (figures/generic_vs_addressed_you_block.py). Kaynak: "
     r"results/JENERIK\_HITAP\_2026-09-10.json · JH\_B100\_2026-09-25.json",
     r"\paragraph*{Generic \emph{you} and addressed \emph{you}: what the split can and cannot say.}",
     r"\vekalet{$M_1$ counts second-person forms and does not ask whom they name. "
     r"We split the sentences carrying one into \textsc{addressee}, \textsc{generic} "
     r"and \textsc{undetermined} with a cue list fixed before the measurement, with "
     r"\textsc{addressee} taking precedence where both fire. Across the " +
     f"{len(A)} models the generic count falls in {n_j} and the addressed count in "
     f"{n_h}. A preliminary check by the assistant (an LLM agent, not a human labeller) on {len(md)} "
     f"sentences found the generic label right ${gj}/{nj}$ of the time and the addressed label ${gh}/{nh}$. "
     r"In that check the deontic cue (\emph{you should}, \emph{you must}) fired on generic advice as readily as on "
     f"advice to a reader. Of the ${len(bel)}$ the cues left \\textsc{{undetermined}}, ${bel_coz}$ were resolvable "
     f"on reading (overall agreement ${uy}/{len(md)}$). "
     f"On the authors' labels for ${B['n']}$ second-person sentences (Appendix~\\ref{{app:labelcheck}}) the "
     f"classifier labels only ${bnj + bnh}$. Its generic label is right ${bj}/{bnj}$ of the time and its addressed "
     f"label ${bh}/{bnh}$. Of the ${bbel}$ it leaves \\textsc{{undetermined}}, the authors read ${bcoz}$ as addressed "
     f"or generic. \\textbf{{The asymmetry of the preliminary check is not reproduced on human "
     f"labels.}} \\textsc{{Undetermined}} is ${100*pay_bel:.0f}\\%$ of all second-person sentences in the panel. "
     r"Both halves are read from the judge-labelled check of Appendix~\ref{app:labelcheck} (\S\ref{sec:measure}). "
     r"\textbf{We do not claim the withdrawal is confined to either "
     r"half}: on human labels neither of the classifier's two labels is right often enough to carry that "
     r"claim.}", ""]
io.open(f"{OUT}/JENERIK_HITAP.tex", "w", encoding="utf-8").write("\n".join(L))
print(f"{len(A)} {n_j}"
      f"{n_h} {gj} {nj} {gh} {nh}"
      f"{bj} {bnj} {bh} {bnh} {bbel} {bcoz}"
      f"")
print("  ✓ JENERIK_HITAP.tex")
