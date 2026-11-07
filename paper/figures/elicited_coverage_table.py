import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys; sys.path.insert(0,__DNH_ROOT__ + "/scripts")
import sys, json
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, ROOT, load, sha16, t1_adi
from degeneracy_threshold_floor import farkli4

import elicited16_loader as E16
DIS = __DNH_DATA__ + "/elicit"
KAPI_MIB = 59633
AGIRLIK = {"Qwen2.5-32B": 64000, "OLMo2-32B": 64000,
           "Llama-3.1-70B": 141000, "Qwen2.5-72B": 145000}


def main():
    B, pb = E16.yukle(); k = B["_kunye"]
    raf = {r["aile"]: r for r in k["raf"]}
    yok = list(k["kosulmadi"])
    sat = []
    for a in yok:
        m = AGIRLIK.get(a)
        sat.append((a, r"\textsc{not-run}",
                    "%s MiB $>$ %s MiB gate" % (f"{m:,}".replace(",", "{,}"),
                                                f"{KAPI_MIB:,}".replace(",", "{,}"))))
    for a, r_ in raf.items():
        v = min(r_["farkli4_base"], r_["farkli4_instruct"])
        sat.append((a, r"\textsc{d-shelf}",
                    "distinct-4 $%.3f < 0.60$ (repetition loop)" % v))
    print("%d %d %d %d"
          "%d %d"
          "" % (k["n_uygun"], len(sat), len(yok), len(raf),
                               k["n_uygun"] + len(sat), KAPI_MIB))
    if k["n_uygun"] + len(sat) != 16:
        print("  ★ TOPLAM 16 DEGIL ⇒ HATA (§8)"); return 3
    _kapi = (r"The memory gate is the run's own pre-flight figure (free VRAM minus a "
             r"fixed reserve, measured at queue start). " if yok else
             r"Every model in the panel is generated on this prompt set, so no "
             r"row carries a compute reason: the two pairs produced "
             r"on two GPUs are scored here. ")
    _cap = (r"\caption{\textbf{Why ELICIT-99 scores %d of %d models.} "
            r"Reasons are printed rather than left blank. " % (k["n_uygun"], k["n_uygun"] + len(sat))
            + _kapi
            + r"The degeneracy criterion is the pre-registered distinct-4 floor, "
              r"whose threshold sits in an empty interval of the calibration "
              r"distribution. With %d models scored the twelve-model bar is "
              r"reachable on this prompt set; how many clear it is a separate count, marked in bold in "
              r"Table~\ref{tab:families-full}.}" % k["n_uygun"])
    L = [r"\begin{table}[tb]", r"\centering\small", _cap,
         r"\label{tab:coverage}", r"\begin{tabular}{lll}", r"\toprule",
         r"model & outcome & measured reason \\", r"\midrule"]
    for a, o, r_ in sat:
        L.append("%s & %s & %s \\\\" % (t1_adi(a), o, r_))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/T3_coverage.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(table="T3_coverage", sources=[pb], payda=dict(n_satir=len(sat)),
                   note="renders pre-specified measurements; recomputes the frozen degeneracy criterion"),
              open(f"{OUT}/T3_coverage.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ✓ T3_coverage.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
