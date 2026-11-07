#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/KOL_TABLOSU_2026-09-10.json"
EN = {"Tulu3-8B": "T\\\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def ad(a):
    return EN.get(a, a)


def main():
    K, prov = load(CARD)
    A = {a: v for a, v in K["aile"].items() if v.get("hal") == "ÖLCÜLDÜ"}
    O = K["onek"]
    yf = [k for k, v in O.items() if v["n_sahis2"] == 0]
    tas = [k for k, v in O.items() if v["n_sahis2"] > 0]
    n_ayrik = sum(1 for v in A.values()
                  if v["you_free"].get("ayrik") and v["you_free"]["dM1"] < 0)
    n_tum = sum(1 for v in A.values() if v["tum_panel"]["dM1"] < 0)
    print(f"{len(A)} {len(yf)}"
          f"{n_ayrik} {n_tum}"
          f"")

    L = [r"\begin{table}[t]\centering\small",
         r"\caption{\textbf{The headline read on the nine prefix conditions that carry no "
         r"second person.} The debate prompts' $16$ prefix conditions are a regulator we "
         r"did not design out: seven of them contain second persons of their own "
         r"(from $102$ to $1{,}360$ occurrences across the $408$ prompt-draw cells), "
         r"and a base checkpoint can copy them. Splitting each condition's own prefix from "
         r"the shared thesis scaffold by longest common suffix leaves nine conditions with "
         r"\emph{none} (Table~\ref{tab:onek}). Read on those nine alone, the withdrawal is still negative "
         + ("in all %d models" % n_ayrik if n_ayrik == len(A) else
            "and interval-disjoint in %d of the %d models" % (n_ayrik, len(A)))
         + r" --- \textbf{the direction is not an artifact of the prefix conditions, but part of "
           r"the magnitude is}. The pre-registered $16/16$ count on the full panel "
           r"remains the primary reading; this is a second, descriptive one.}",
         r"\label{tab:youfree}",
         r"\begin{tabular}{lrrrc}", r"\toprule",
         r"model & full panel $\Delta M_1$ & nine you-free prefixes & $95\%$ CI & prefixes disj. \\",
         r"\midrule"]
    for a, v in sorted(A.items(), key=lambda kv: kv[1]["tum_panel"]["dM1"]):
        y = v["you_free"]; ci = y.get("ci")
        n_k = sum(1 for x in v["kol"].values() if x.get("ayrik"))
        d = f"\\textbf{{{y['dM1']:+.2f}}}" if y.get("ayrik") and y["dM1"] < 0 else f"{y['dM1']:+.2f}"
        L.append(f"{ad(a)} & {v['tum_panel']['dM1']:+.2f} & {d} & "
                 f"[{ci[0]:+.2f},\\,{ci[1]:+.2f}] & {n_k}/16 \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    io.open(f"{OUT}/KOL_YOUFREE.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

    P = [r"\begin{table}[t]\centering\small",
         r"\caption{\textbf{What each prefix condition brings of its own.} Second- and "
         r"first-person counts of the condition's prefix alone, after the shared thesis "
         r"scaffold is removed, summed over the $408$ prompt$\times$draw cells it "
         r"appears in. The nine conditions with no second person are the pool read in "
         r"Table~\ref{tab:youfree}. \textbf{The first person is the larger stimulus "
         r"and we did not control for it}: the dominance-low conditions carry it heavily.}",
         r"\label{tab:onek}",
         r"\begin{tabular}{lrrrc}", r"\toprule",
         r"condition & tokens & 2nd person & $M_1$ of prefix & you-free \\", r"\midrule"]
    _TIK = "$\\checkmark$"
    for k, v in sorted(O.items(), key=lambda kv: -kv[1]["n_sahis2"]):
        _ad = k.replace("_", "\\_")
        _jt = f"{v['n_jeton']:,}".replace(",", "{,}")
        _s2 = f"{v['n_sahis2']:,}".replace(",", "{,}")
        _yf = "--" if v["n_sahis2"] else _TIK
        P.append(f"\\texttt{{{_ad}}} & ${_jt}$ & ${_s2}$ & {v['M1_onek']:.2f} & {_yf} \\\\")
    P += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    io.open(f"{OUT}/KOL_ONEK.tex", "w", encoding="utf-8").write("\n".join(P) + "\n")
    json.dump(dict(table="KOL_YOUFREE", sources=[prov],
                   payda=dict(n_aile=len(A), n_you_free_kol=len(yf),
                              n_you_tasiyan=len(tas), hal_yf_asagi_ayrik=n_ayrik,
                              hal_tum_panel_asagi=n_tum)),
              io.open(f"{OUT}/KOL_YOUFREE.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"✓ KOL_YOUFREE.tex + KOL_ONEK.tex ({len(A)} aile, {len(yf)} you-free kol)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
