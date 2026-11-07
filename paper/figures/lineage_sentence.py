#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, hashlib, time
ROOT = __DNH_ROOT__ + ""
CARD = f"{ROOT}/results/lineage_permutation_2026-09-11.json"
CARD_YER = f"{ROOT}/results/lineage_sign_placement_2026-09-15.json"
OUT = os.path.dirname(os.path.abspath(__file__))


def sha16(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def _frac(f, n):
    return (r"<%.3f" % (1.0 / n)) if f <= 0 else (r"=%.3f" % f)


def main():
    K = json.load(io.open(CARD, encoding="utf-8"))
    H, E, B = K["H"], K["eta2"], K["okuma_B"]
    I, G = K["isaret"], K["guc_kontrolu"]
    Y = json.load(io.open(CARD_YER, encoding="utf-8"))
    IB = Y["okuma_B"]
    assert Y["red_esdeger_A"] == 0 and Y["okuma_A"]["uyusan"] == I["uyusan"] and Y["okuma_A"]["frac"] == I["frac"], ""
    SV = Y["okuma_A"]["seviye"]
    if K["VERDICT"] == "SIRALIYOR":
        c = (r"\textbf{Of the properties we put to the split, the lineage label "
             r"sorts the magnitudes but not the sign.} A "
             r"Kruskal--Wallis test on that label, read against $%d$ permutations "
             r"of it, gives $H=%.1f$ against a null centred at $%.1f$ "
             r"($\mathrm{frac}(\mathrm{null}\ge H)%s$), and the label accounts for "
             r"$%.0f\%%$ of the between-model variance where a random relabelling "
             r"accounts for $%.0f\%%$. \textbf{That is close to the ceiling these "
             r"group sizes allow}: the best possible labelling of sixteen values "
             r"into groups of $3,3,2,6,1,1$ reaches only $H=%.1f$. "
             r"\textbf{The label has six levels, not the five lineages of \S\ref{sec:intro}}: "
             r"Google ($%d$), Meta ($%d$), AI2 ($%d$), Qwen ($%d$), Mistral ($%d$) and OLMo-3 ($%d$). "
             r"OLMo-3-7B is its own level because its preference stage is a different construction "
             r"from the T\"ulu-3 mixture the other three AI2 models share: its card names "
             r"\texttt{Dolci-Think-DPO-7B}, built with the preference heuristic of \citet{geng2025delta}. "
             r"\textbf{The sign is a weaker reading, and it depends on where OLMo-3 is put}: "
             r"with OLMo-3 as its own level, $%d$ of the $%d$ models that carry a sign fall on their lineage's "
             r"majority side, against $%.1f$ expected under relabelling, "
             r"$\mathrm{frac}%s$ --- above the floor of this test and only just "
             r"inside the bar --- and with OLMo-3 merged into AI2, $%d$ of the $%d$ do, against $%.1f$ "
             r"expected, $\mathrm{frac}%s$, which does not clear it. Five of the six lineages are sign-pure; the "
             r"exception is a single Qwen2.5 model. The magnitude reading does not "
             r"depend on where we put OLMo-3: merged into AI2, which is the "
             r"labelling our own counterexample used, the test still fires "
             r"($H=%.1f$, $\mathrm{frac}%s$), though that merge makes a second "
             r"lineage mixed in sign. Lineage and pipeline are confounded in this "
             r"panel, so this sorts the split without saying which of the two "
             r"does it."
             % (K["n_perm"], H["gozlenen"], H["null_ort"],
                _frac(H["frac"], K["n_perm"]),
                100 * E["gozlenen"], 100 * E["null_ort"], G["H_en_iyi"],
                SV["Google"], SV["Meta"], SV["AI2"], SV["Qwen"], SV["Mistral"], SV["OLMo-3"],
                I["uyusan"], I["n_isaretli"], I["null_ort"],
                _frac(I["frac"], K["n_perm"]),
                IB["uyusan"], IB["n_isaretli"], IB["null_ort"], _frac(IB["frac"], Y["n_perm"]),
                B["H"], _frac(B["frac"], K["n_perm"])))
    else:
        c = (r"A Kruskal--Wallis test on the lineage label, read against $%d$ "
             r"permutations, does not fire ($H=%.1f$, null centre $%.1f$, "
             r"$\mathrm{frac}%s$)." % (K["n_perm"], H["gozlenen"],
                                       H["null_ort"], _frac(H["frac"], K["n_perm"])))
    io.open(f"{OUT}/SOY_CUMLE.tex", "w", encoding="utf-8").write(c + "\n")
    SS = {r["aile"]: r for r in json.load(io.open(f"{ROOT}/results/template_filtered_2026-09-07.json", encoding="utf-8"))["aileler"]}
    _pay = {a: SS[a]["sbl"]["M1_hizali"] for a in ("Tulu3-8B", "Llama-3.1-8B")}
    _ai2 = {a: SS[a]["sbl"]["dM1"] for a in ("OLMo2-13B", "OLMo2-32B", "OLMo-3-7B")}
    _asagi = sorted(a for a, v in _ai2.items() if v < 0); _yukari = sorted(a for a, v in _ai2.items() if v > 0)
    print(f"  [PAYDA] soy_recete: n_tek_taban={len(_pay)} · paylar={_pay} · ai2_asagi={_asagi} · ai2_yukari={_yukari} "
          f"⇒ esik: tek tabanda iki pay esitse ya da AI2 tek isaretliyse ⇒ EYLEM: karsitlik cümlesi YAZILMAZ")
    assert round(_pay["Tulu3-8B"], 2) != round(_pay["Llama-3.1-8B"], 2) and _asagi and _yukari
    if K["VERDICT"] == "SIRALIYOR":
        gv = (r"The lineage sorts the size of the change (permutation $p%s$). Its sign depends on how OLMo-3 is "
              r"labelled. With OLMo-3 counted inside AI2, $%d$ of the $%d$ models whose change has a clear sign move in the same "
              r"direction as most of their lineage ($p%s$), which does not clear the significance bar. With OLMo-3 counted as its "
              r"own lineage, the sign does clear it, and we report both. Within a lineage the pipeline separates the "
              r"sign, and AI2's two generations split, OLMo-2 falling and OLMo-3 rising."
              % (_frac(H["frac"], K["n_perm"]), IB["uyusan"], IB["n_isaretli"], _frac(IB["frac"], Y["n_perm"])))
    else:
        gv = c
    io.open(f"{OUT}/SOY_GOVDE.tex", "w", encoding="utf-8").write(gv + "\n")
    ek = (r"\textbf{The lineage sign reading depends on where OLMo-3 is put.} Read with "
          r"OLMo-3 as its own level, $%d$ of the $%d$ models that carry a lineage fall on "
          r"its majority side ($\mathrm{frac}%s$), five of six lineages being sign-pure and the "
          r"exception a single Qwen2.5 model; merged into AI2, the five lineages \S\ref{sec:intro} "
          r"counts, it is $%d$ of $%d$ ($\mathrm{frac}%s$) and the sign no longer clears the bar. "
          r"The magnitude result holds under both labellings. \textbf{Lineage and pipeline are "
          r"confounded here}, so neither labelling names a mechanism."
          % (I["uyusan"], I["n_isaretli"], _frac(I["frac"], K["n_perm"]),
             IB["uyusan"], IB["n_isaretli"], _frac(IB["frac"], Y["n_perm"])))
    io.open(f"{OUT}/SOY_EK.tex", "w", encoding="utf-8").write(ek + "\n")
    n = (r"$H$ and $\eta^2$ are sign-free: under the null they are positive, so "
         r"each is read against its own permutation centre (%.1f and $%.0f\%%$ here), "
         r"not against zero. \textbf{The two are not comparable across factors "
         r"either}: lineage is a six-level factor on sixteen points with two "
         r"singleton groups, and a six-level null already accounts for "
         r"$%.0f\%%$ of the variance where a binary property's null centre is "
         r"$%.0f\%%$, so the share is read against its own null and never against "
         r"another factor's share."
         % (H["null_ort"], 100 * E["null_ort"], 100 * E["null_ort"], 100 / 15.0))
    io.open(f"{OUT}/SOY_NOT.tex", "w", encoding="utf-8").write(n + "\n")
    json.dump(dict(table="SOY", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=os.path.relpath(CARD, ROOT), sha256_16=sha16(CARD)),
                            dict(yol=os.path.relpath(CARD_YER, ROOT), sha256_16=sha16(CARD_YER))],
                   isaret_B=IB,
                   VERDICT=K["VERDICT"], okuma_A=H, okuma_B=B,
                   isaret=K["isaret"], guc_kontrolu=K["guc_kontrolu"]),
              io.open(f"{OUT}/SOY.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{K['VERDICT']} {H['frac']}"
          f"{B['frac']} {I['uyusan']} {I['n_isaretli']}"
          f"{I['frac']} {G['H_en_iyi']}"
          f""
          f"")
    print("✓ SOY_CUMLE.tex + SOY_NOT.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
