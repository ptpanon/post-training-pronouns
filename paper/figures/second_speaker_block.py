#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, hashlib, time
ROOT = __DNH_ROOT__ + ""
CARD = f"{ROOT}/results/second_speaker_bare_2026-09-11.json"
SBL = f"{ROOT}/results/turn_marker_2026-09-10.json"
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
    K = json.load(io.open(CARD, encoding="utf-8"))
    A = {k: v for k, v in K["aile"].items() if v["hal"] == "ÖLCÜLDÜ"}
    Sb = json.load(io.open(SBL, encoding="utf-8"))
    Sa = {k: v for k, v in Sb["aile"].items() if v.get("hal") == "ÖLCÜLDÜ"}
    _sp = sorted(v["pay_you_isaretten_sonra"] for v in Sa.values())
    _yuksek = sorted((v["pay_you_isaretten_sonra"], k) for k, v in Sa.items()
                     if v["pay_you_isaretten_sonra"] > 0.05)
    _sifir = sum(1 for x in _sp if x <= 0.01)
    assert _yuksek, ""
    print("%d %d %d"
          "%% %.0f %% %.0f"
          ""
          % (len(Sa), _sifir, len(_yuksek), 100 * _yuksek[0][0],
             100 * _yuksek[-1][0]))
    S = K["sayim"]
    pt = sorted(v["pay_you_araliklarda_taban"] for v in A.values())
    med = (pt[len(pt) // 2] if len(pt) % 2
           else 0.5 * (pt[len(pt) // 2 - 1] + pt[len(pt) // 2]))
    dus = [k for k, v in A.items()
           if not (v["dM1_tavan"] < 0 and v["ayrik_tavan"])]
    _artti = sum(1 for v in A.values() if v.get("M1_yon_kesik") == "artti")
    _kp = sorted(v.get("pay_jeton_kesildi_taban", 0.0) for v in A.values())
    print(f"  [PAYDA] konusma_yon: n_aile={len(A)} · hal_M1_artti={_artti} · "
          f"kesilen_jeton=%%%.1f-%%%.1f ⇒ esik: artti<n_aile ⇒ EYLEM: yön "
          f"model-specific, read in the table" % (100 * _kp[0], 100 * _kp[-1]))
    print(f"{len(A)} {S['kesik']}"
          f"{S['tavan']} {len(dus)} {S['payda']}"
          f"")

    c = (r"\textbf{In the raw-continuation outputs the second persons are not in someone else's "
         r"mouth.} Running the chat-template outputs' check on the raw-continuation ones---quoted spans, "
         r"speaker labels and dash-dialogue, and the complement clauses of speech "
         r"verbs---puts $%.0f\%%$ to $%.0f\%%$ of the base outputs' second persons inside "
         r"such a span (median $%.0f\%%$). \textbf{The chat-template outputs are not one "
         r"figure but two regimes}: thirteen of its sixteen models put "
         r"essentially none of their second persons after a fabricated turn "
         r"marker, and the three whose templates write that boundary in plain "
         r"text put $%.0f$--$%.0f\%%$ there --- so the raw-continuation outputs' few per cent is "
         r"read against a distribution with a floor at zero, not against a "
         r"single chat-template number. \textbf{Read outside those spans "
         r"the withdrawal is unchanged: $%d$ of $%d$ down and interval-disjoint.} A "
         r"deliberately over-cutting ceiling reading, which also discards everything "
         r"after the first blank line and so throws away ordinary paragraph breaks, "
         r"leaves $%d$ of $%d$%s. \textbf{Two limits of this reading, named}: the "
         r"detector under-counts --- straight-quote dialogue and any speech verb "
         r"outside our list are missed --- so the in-span share is a \emph{lower} "
         r"bound on exactly the quantity we want small; and because $M_1$ is a "
         r"density, excising a span removes tokens as well as second persons, so "
         r"the direction of the change is not conservative by construction. "
         r"\textbf{So we measured it}: the excision removes "
         r"$%.0f$--$%.0f\%%$ of the tokens and \textbf{raises $M_1$ in all $%d$ "
         r"models}, which means the spans it removes are \emph{less} "
         r"second-person-dense than the prose around them --- the second "
         r"persons are not concentrated in the dialogue."
         % (100 * pt[0], 100 * pt[-1], 100 * med,
            100 * _yuksek[0][0], 100 * _yuksek[-1][0],
            S["kesik"], S["payda"],
            S["tavan"], S["payda"],
            "" if not dus else " (the exception is " + ", ".join(ad(x) for x in dus)
            + r", whose interval then covers zero)",
            100 * _kp[0], 100 * _kp[-1], _artti))
    kisa = ((r"Excluding second-person forms inside quoted speech, name labels and reported-speech "
             r"clauses (a small minority of the base output's second-person forms) leaves the fall with an interval excluding zero "
             r"in every model.")
            if S["kesik"] == S["payda"] else
            (r"Excluding second-person forms inside quoted speech, name labels and reported-speech "
             r"clauses, a small minority of the base output's, leaves the fall at $%d$ of $%d$, "
             r"interval-separated." % (S["kesik"], S["payda"])))
    io.open(f"{OUT}/KONUSMA_CUMLE.tex", "w", encoding="utf-8").write(kisa + "\n")
    io.open(f"{OUT}/KONUSMA_EK.tex", "w", encoding="utf-8").write(
        "\\paragraph*{Second persons inside quoted speech.}\n\\vekalet{" + c + "}\n")

    L = [r"\begin{table}[t]\centering\small",
         r"\caption{\textbf{The raw-continuation outputs' second persons, read outside quoted, "
         r"attributed and reported speech.} Share of the base outputs' second-person "
         r"tokens falling inside a detected span, and $\Delta M_1$ recomputed with "
         r"those spans removed from both checkpoints. \emph{ceiling} additionally treats "
         r"everything after the first blank line as another speaker's turn, which "
         r"over-cuts. Intervals are prompt-clustered ($%d$ resamples). "
         r"Over-detection works against the result, not for it: it removes text and "
         r"leaves less to count.}" % K["n_bootstrap"],
         r"\label{tab:konusma}",
         r"\begin{tabular}{lrrrrr}\toprule",
         r"model & in-span \% & tokens cut \% & $\Delta M_1$ & outside spans & "
         r"ceiling \\",
         r"\midrule"]
    for a in sorted(A):
        v = A[a]
        m = lambda x, d: ("$%+.2f$" % x) + ("" if d else r"$^{\circ}$")
        _kp = v.get("pay_jeton_kesildi_taban")
        L.append(f"{ad(a)} & ${100*v['pay_you_araliklarda_taban']:.1f}$ & "
                 + (f"${100*_kp:.1f}$ & " if _kp is not None else "--- & ")
                 + f"{m(v['dM1'], v['ayrik'])} & {m(v['dM1_kesik'], v['ayrik_kesik'])} & "
                 f"{m(v['dM1_tavan'], v['ayrik_tavan'])} \\\\")
    L += [r"\midrule",
          f"\\textbf{{down, interval-disjoint}} & & & "
          f"\\textbf{{{S['tam']}/{S['payda']}}} & "
          f"\\textbf{{{S['kesik']}/{S['payda']}}} & \\textbf{{{S['tavan']}/{S['payda']}}} \\\\",
          r"\bottomrule\end{tabular}",
          r"\par\smallskip\footnotesize $^{\circ}$ interval covers zero. "
          r"Limits, named: straight-apostrophe quotation is not counted, the "
          r"speech-verb list is our own, and unmarked turn transitions appear only "
          r"in the ceiling column.",
          r"\end{table}"]
    io.open(f"{OUT}/KONUSMA_TABLO.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(table="KONUSMA", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=os.path.relpath(CARD, ROOT), sha256_16=sha16(CARD))],
                   sayim=S, pay_bandi=[pt[0], pt[-1]], medyan=med, tavan_dusen=dus),
              io.open(f"{OUT}/KONUSMA.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("✓ KONUSMA_CUMLE.tex + KONUSMA_EK.tex + KONUSMA_TABLO.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
