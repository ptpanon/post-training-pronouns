#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json

KOK = __DNH_ROOT__ + ""
CARD = f"{KOK}/results/b0b_kod_ikili_2026-09-24.json"
CARD_A = f"{KOK}/results/r2a1_matched_doz_2026-11-06.json"
CARD_D = f"{KOK}/results/r2a1_arm_manifest_2026-11-06.json"
CARD_H = f"{KOK}/results/r2a1_kod_hacmi_2026-11-06.json"
CARD_T = f"{KOK}/results/person_dial_seeds_2_3_verdict_2026-09-15.json"
FIG = f"{KOK}/paper/figures"
BAS = "% * URETILDI - elle duzenlenmez (figures/b0_feasibility.py)\n"
N_KOLYONU = 3346


def yaz(ad, g):
    io.open(f"{FIG}/{ad}", "w", encoding="utf-8").write(BAS + g.strip() + "\n")
    print(f"✓ {ad} ({len(g.split())} kelime)")


def vir(n):
    return f"{n:,}".replace(",", "{,}")


def f(x, n=2):
    return f"{x:+.{n}f}".replace("+", "" if x < 0 else "+")


def ara(v, ci):
    return f"${f(v)}$ $[{f(ci[0])},{f(ci[1])}]$"


def main():
    K = json.load(io.open(CARD, encoding="utf-8"))
    A = json.load(io.open(CARD_A, encoding="utf-8"))
    DG = json.load(io.open(CARD_D, encoding="utf-8"))["denge"]
    H = json.load(io.open(CARD_H, encoding="utf-8"))["kol"]
    assert A["verdict"]["ad"] == "IKISI-AYRIK", ""
    a, b = A["verdict"]["okuma"]["R2_A1"], A["verdict"]["okuma"]["B1C_KOD"]
    c = A["verdict"]["kontrast_A1_A2"]
    pa, pb = A["verdict"]["profil"]["kol"]["R2_A1"]["delta_SFT"], A["verdict"]["profil"]["kol"]["B1C_KOD"]["delta_SFT"]
    da, db = DG["A1"], DG["A2_B1C"]
    assert da["pay_dKod_neg"] == db["pay_dKod_neg"], ""
    assert b["ci_1st_A"][0] < 0 < b["ci_1st_A"][1] and a["ci_1st_A"][0] < 0 < a["ci_1st_A"][1], ""
    assert abs(b["dM1_B"]) < b["plasebo_M1_p95"] < abs(a["dM1_B"]), ""
    assert c["ci_1st"][0] < 0 < c["ci_1st"][1] and c["ci_M1"][1] < 0, ""
    T = json.load(io.open(CARD_T, encoding="utf-8"))["kol"]
    seed = [T[k]["dB"] for k in ("KISI_ASAGI", "KISI_ASAGI_T2", "KISI_ASAGI_T3")]
    assert abs(b["dM1_B"]) < min(A["bar"], A["mde_B"]) and max(A["bar"], A["mde_B"]) < abs(c["dM1"]), ""
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
        r"\paragraph*{The code-block-matched control, at the size it allows.}" "\n"
        r"\vekalet{Selecting pairs on pronoun use also selects fewer refusals, fewer code blocks and "
        r"fewer tokens (\S\ref{sec:dial}), and the control reported there matches refusals and length but "
        r"not code blocks. We tried to add code blocks. The pool is not what stops it: of the "
        f"${vir(havuz)}$ usable preference pairs, ${vir(sifir)}$ carry no difference in second-person "
        r"rate at all, so pairs without a pronoun signal are abundant. The matching is what stops it. "
        r"Required to reproduce the pronoun-selected set's joint distribution over refusal rate, "
        r"length and code blocks while carrying no pronoun difference, the largest set we could "
        f"assemble holds ${vir(en_iyi)}$ pairs, ${vir(N_KOLYONU)}$ in the trained direction; across {n_kip} ways of coding the code-block axis the "
        f"ceiling runs from ${vir(en_dusuk)}$ to ${vir(en_iyi)}$, against the ${vir(ist)}$ pairs each "
        r"training run uses.}" "\n\n"
        rf"\vekalet{{After submission we ran it at that size and then a person-selected run of the same size: ${vir(N_KOLYONU)}$ of the ${vir(ist)}$ pairs, drawn to the control's strata, "
        rf"same configuration (205 and 206 steps), one training seed each, each registered before training. Changes are per 1,000 tokens, second person first. "
        rf"At $n={vir(N_KOLYONU)}$ and the same configuration, both runs separate from the reference: person-selected {ara(a['dM1_B'], a['ci_M1_B'])}; "
        rf"code-matched control with no net person signal {ara(b['dM1_B'], b['ci_M1_B'])}. Same-size difference {ara(c['dM1'], c['ci_M1'])}. "
        rf"The reference is the unselected run at ${vir(ist)}$ pairs, a larger dataset, so the same-size difference is the reading that does not depend on it. "
        rf"Against the SFT base the two are {ara(a['dM1_A'], a['ci_M1_A'])} and {ara(b['dM1_A'], b['ci_M1_A'])}. "
        rf"The person-selected run clears its paired placebo band (${a['plasebo_M1_p95']:.2f}$) and the control does not (${b['plasebo_M1_p95']:.2f}$). "
        rf"The first person does not separate: {ara(a['d1st_B'], a['ci_1st_B'])} and {ara(b['d1st_B'], b['ci_1st_B'])}, difference {ara(c['d1st'], c['ci_1st'])}; "
        rf"against the SFT base both intervals include zero ({ara(a['d1st_A'], a['ci_1st_A'])} and {ara(b['d1st_A'], b['ci_1st_A'])}).}}" "\n\n"
        rf"\vekalet{{Four limits govern this reading. First, the control matches the share of pairs whose chosen side loses a code block (${100*da['pay_dKod_neg']:.1f}\%$ in both), not code volume, "
        rf"the reading that bears on a density (below): the chosen side of the person-selected set carries ${f(H['A1']['fark']['kod_jeton_ort'], 1)}$ code tokens per response and the control's ${f(H['A2_B1C']['fark']['kod_jeton_ort'], 1)}$, opposite signs, "
        rf"a contrast we measured on the pairs after the run and did not register. So the same-size difference cannot exclude code volume. "
        rf"Second, the strata match and the means do not: chosen minus rejected, refusals ${f(db['dRed'], 3)}$ in the control against ${f(da['dRed'], 3)}$, code blocks ${f(db['dKod'], 3)}$ against ${f(da['dKod'], 3)}$, tokens ${f(db['dJeton'], 1)}$ against ${f(da['dJeton'], 1)}$, log length ${f(db['dBoy'], 3)}$ against ${f(da['dBoy'], 3)}$. "
        rf"Third, each run is one training seed. The intervals cover prompts and generation seeds, not training seeds; at ${vir(ist)}$ pairs three seeds span ${f(min(seed))}$ to ${f(max(seed))}$, and at this size the spread is unmeasured. On the generation-seed scale the full-size runs' registered bar is ${A['bar']:.1f}$ and the floor for a difference between two runs ${A['mde_B']:.2f}$: the control's move is below both and the same-size difference above both. "
        rf"Fourth, this is a smaller dataset than the runs of \S\ref{{sec:dial}} and does not test them. The debate prompts elicit no code blocks, so output code cannot be read; against SFT, refusals move ${f(pa['red'], 3)}$ and ${f(pb['red'], 3)}$, length ${f(pa['kar'], 1)}$ and ${f(pb['kar'], 1)}$ characters. "
        r"At this size the fall in ``you'' is not carried by refusals or by the presence of code blocks; whether code volume or length carries it, this design cannot say. The fall in ``I'' does not separate from the bundle. The code-block confound stays open.}")

    yaz("B0_LIMIT.tex",
        f"A control also matching code blocks reaches ${vir(N_KOLYONU)}$ pairs against ${vir(ist)}$ and, at that size, leaves code volume open (Appendix~\\ref{{app:D7}}).")

    json.dump(dict(table="B0_FIZIBILITE",
                   sources=[{"yol": "results/b0b_kod_ikili_2026-09-24.json"}, {"yol": "results/r2a1_matched_doz_2026-11-06.json"},
                            {"yol": "results/r2a1_arm_manifest_2026-11-06.json"}, {"yol": "results/r2a1_kod_hacmi_2026-11-06.json"}, {"yol": "results/person_dial_seeds_2_3_verdict_2026-09-15.json"}],
                   ciktilar=["B0_D7.tex", "B0_LIMIT.tex"],
                   en_iyi=en_iyi, en_dusuk=en_dusuk, n_istenen=ist),
              io.open(f"{FIG}/B0_FIZIBILITE.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
