#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, hashlib, time
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
ROOT = __DNH_ROOT__ + ""
SEY = f"{ROOT}/results/dilution_2026-09-03.json"
KOL = f"{ROOT}/results/KOL_TABLOSU_2026-09-10.json"
TOH = f"{ROOT}/results/four_seed_reading_2026-09-01.json"
OUT = os.path.dirname(os.path.abspath(__file__))
EN = {"Tulu3-8B": "T\\\"ulu-3-8B", "OLMo2-13B": "OLMo-2-13B", "OLMo2-32B": "OLMo-2-32B"}


def sha16(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def ad(a):
    return EN.get(a, a)


def main():
    S = json.load(io.open(SEY, encoding="utf-8"))["_aile"]
    K = json.load(io.open(KOL, encoding="utf-8"))["aile"]
    import four_seed_loader as _TK
    _T, _TSRC = _TK.oku()
    T = {k: v for k, v in _T.items() if not k.startswith("_")}
    aileler = sorted(set(S) & set(K) & set(T))
    birlesim = sorted(set(S) | set(K) | set(T))
    eksik = {a: [n for n, D in (("seyrelme", S), ("kol", K), ("seed", T))
                 if a not in D] for a in birlesim if a not in aileler}
    print(f"{len(birlesim)}"
          f"{len(aileler)} {len(eksik)} {eksik}"
          f"")
    if eksik or len(aileler) != 16:
        print(f"{len(birlesim)}"
              f"{len(aileler)} {eksik}")
        return 3
    R = {}
    for a in aileler:
        s, k, t = S[a], K[a], T[a]
        ci_c = s.get("ci_M1_cumle")
        c_ok = bool(s.get("dlog_M1_cumle", 0) < 0 and ci_c and ci_c[1] < 0)
        yf = k.get("you_free", {})
        y_ok = bool(yf.get("dM1", 0) < 0 and yf.get("ayrik"))
        if t.get("hal") != "m=4":
            t_ok = None
        else:
            t_ok = bool(t.get("m4_ort", 0) < 0 and t.get("ayrik")
                        and t.get("plasebo_ustu"))
        R[a] = dict(cumle=c_ok, you_free=y_ok, seed=t_ok,
                    dlog_cumle=s.get("dlog_M1_cumle"), dM1_yf=yf.get("dM1"),
                    m4_ort=t.get("m4_ort"), seed_hal=t.get("hal"))
    gecen = [a for a, v in R.items() if v["cumle"] and v["you_free"] and v["seed"] is True]
    dusen = [a for a, v in R.items()
             if v["seed"] is not None and not (v["cumle"] and v["you_free"] and v["seed"])]
    olcul = [a for a, v in R.items() if v["seed"] is None]
    n_c = sum(1 for v in R.values() if v["cumle"])
    n_y = sum(1 for v in R.values() if v["you_free"])
    n_t = sum(1 for v in R.values() if v["seed"] is True)
    print(f"{len(R)} {n_c} {n_y}"
          f"{n_t} {len(gecen)} {len(olcul)}"
          f""
          f"")

    L = [r"\begin{table}[t]\centering\small",
         r"\caption{\textbf{The three robustness re-reads, crossed.} Each column is "
         r"the model as it was read in the card that produced it: a per-sentence count "
         r"whose interval excludes zero (Table~\ref{tab:dilution}), the nine prefix conditions "
         r"that carry no second person (Table~\ref{tab:youfree}), and the four-seed "
         r"re-read with its placebo band (Appendix~\ref{app:repro}). "
         + (r"All three are cleared by the same $%d$ models" % len(gecen)
            if (n_c == n_y == n_t == len(gecen)) else
            r"The three are cleared by $%d$, $%d$ and $%d$ models and by the same "
            r"$%d$ together" % (n_c, n_y, n_t, len(gecen)))
         + (r"; the four-seed re-read has not run in %d, which are carried in the "
            r"denominator rather than counted as failures." % len(olcul) if olcul else r".")
         + r"}",
         r"\label{tab:caprazlama}",
         r"\begin{tabular}{lccc}\toprule",
         r"model & per sentence & you-free prefixes & four seeds \\",
         r"\midrule"]
    im = {True: r"\checkmark", False: r"---", None: r"$\circ$"}
    for a in aileler:
        v = R[a]
        L.append(f"{ad(a)} & {im[v['cumle']]} & {im[v['you_free']]} & {im[v['seed']]} \\\\")
    _uc = lambda g, o: (r"\textbf{%d} / %d / %d" % (g, 16 - g - o, o))
    L += [r"\midrule",
          f"cleared / not cleared / not read & {_uc(n_c, 0)} & {_uc(n_y, 0)} & "
          f"{_uc(n_t, len(olcul))} \\\\",
          r"\bottomrule\end{tabular}"] + ([
          r"\par\smallskip\footnotesize $\circ$ = not read at four seeds yet "
          r"(counted in the denominator, not as a failure)."] if olcul else []) + [
          r"\end{table}"]
    io.open(f"{OUT}/CAPRAZLAMA.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    _SOZ = "Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen".split()
    c = (f"{_SOZ[len(gecen)]} of the 16 models pass all three of these checks at once: the per-sentence "
         f"measure, the nine prefix conditions without a second person, and the four-seed reading")
    if olcul:
        c += (f", and {len(olcul)} of the sixteen have not been read at four seeds, "
              f"so they are carried in the denominator rather than counted against")
    c += "."
    io.open(f"{OUT}/CAPRAZLAMA_CUMLE.tex", "w", encoding="utf-8").write(c + "\n")
    json.dump(dict(table="CAPRAZLAMA", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=os.path.relpath(y, ROOT), sha256_16=sha16(y))
                            for y in (SEY, KOL)] + _TSRC,
                   sayim=dict(cumle=n_c, you_free=n_y, seed=n_t,
                              ucu_birden=len(gecen), olculmedi=len(olcul), payda=16),
                   gecen=gecen, dusen=dusen, olculmedi=olcul, aile=R),
              io.open(f"{OUT}/CAPRAZLAMA.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"✓ CAPRAZLAMA.tex + CAPRAZLAMA_CUMLE.tex ({len(gecen)}/16 ücü birden)")
    print(f"  cümle: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
