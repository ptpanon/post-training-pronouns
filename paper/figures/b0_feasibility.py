#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json

KOK = __DNH_ROOT__ + ""
CARD = f"{KOK}/results/b0b_kod_ikili_2026-09-24.json"
FIG = f"{KOK}/paper/figures"
BAS = "% * URETILDI - elle duzenlenmez (figures/b0_feasibility.py)\n"


def yaz(ad, g):
    io.open(f"{FIG}/{ad}", "w", encoding="utf-8").write(BAS + g.strip() + "\n")
    print(f"✓ {ad} ({len(g.split())} kelime)")


def vir(n):
    return f"{n:,}".replace(",", "{,}")


def main():
    K = json.load(io.open(CARD, encoding="utf-8"))
    assert not K["fizibilite"]["yeterli"], ""
    T = K["kod_kip_tavanlar"]
    tav = [v["n_max"] for v in T.values()]
    en_iyi, en_dusuk = max(tav), min(tav)
    ist = K["fizibilite"]["n_istenen"]
    sifir = K["kisi_sifir"]["n_aday"]
    havuz = K["havuz"]["gecerli"]
    n_kip = len(T)
    print(f"{n_kip} {en_iyi} {en_dusuk}"
          f"{ist} {ist}")

    yaz("B0_D7.tex",
        r"\paragraph*{The code-block-matched control we could not build.}" "\n"
        r"\vekalet{Selecting pairs on pronoun use also selects fewer refusals, fewer code blocks and "
        r"fewer tokens (\S\ref{sec:dial}), and the control reported there matches the first two but "
        r"not the third. We tried to build the third. The pool is not what stops it: of the "
        f"${vir(havuz)}$ usable preference pairs, ${vir(sifir)}$ carry no difference in second-person "
        r"rate at all, so pairs without a pronoun signal are abundant. The matching is what stops it. "
        r"Required to reproduce the pronoun-selected set's joint distribution over refusal rate, "
        r"length and code blocks while carrying no pronoun difference, the largest set we could "
        f"assemble holds ${vir(en_iyi)}$ pairs; across {n_kip} ways of coding the code-block axis the "
        f"ceiling runs from ${vir(en_dusuk)}$ to ${vir(en_iyi)}$, against the ${vir(ist)}$ pairs each "
        r"training run uses. We did not run the control at a reduced size: the smaller run already "
        r"reported (\S\ref{sec:dial}) moves no counter, so a null at this data size would not be readable. "
        r"We report the attempt and its ceiling instead, and the code-block confound stays open.}")

    yaz("B0_LIMIT.tex",
        f"We attempted a control matched on code blocks as well and could not build one, the matching topping out at "
        f"${vir(en_iyi)}$ pairs against ${vir(ist)}$ (Appendix~\\ref{{app:D7}}).")

    json.dump(dict(table="B0_FIZIBILITE",
                   sources=[{"yol": "results/b0b_kod_ikili_2026-09-24.json"}],
                   ciktilar=["B0_D7.tex", "B0_LIMIT.tex"],
                   en_iyi=en_iyi, en_dusuk=en_dusuk, n_istenen=ist),
              io.open(f"{FIG}/B0_FIZIBILITE.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
