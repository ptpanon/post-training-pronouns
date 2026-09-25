#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, re, sys
ROOT = __DNH_ROOT__ + ""
OUT = f"{ROOT}/paper/figures"


def J(y):
    return json.load(io.open(f"{ROOT}/{y}", encoding="utf-8"))


def binlik(n):
    return f"{n:,}".replace(",", "{,}")


def main():
    H = J("results/v61_scoring_2026-09-11.json")
    U = J("results/v61_scoring_generation_2026-09-12.json")
    O = J("results/v61_example_pair_2026-09-14.json")
    R = J("results/v98_cpu_readings_2026-09-21.json")["okumalar"]["R5_Q5_ters_yon"]["olcum"]
    assert R["armorm_ileri"]["delta"] < 0 and R["armorm_ters"]["delta"] > 0 and R["skywork_ileri"]["delta"] < 0 and R["skywork_ters"]["delta"] > 0
    a, k = H["puanlayici"]["armorm"], H["puanlayici"]["skywork"]
    assert H["BIRLESIK"] == "ÖDÜLLENDIRIYOR" and a["VERDICT"] == k["VERDICT"] == "ÖDÜLLENDIRIYOR"
    assert all(U["sonuc"][x]["guc"] == "GECTI" and U["sonuc"][x]["n"] == 3000 for x in ("armorm", "skywork"))
    assert O["_kunye"]["prova_eslesen"] == 3000
    ck, dp = U["cift_kunyesi"], U["cift_kunyesi"]["degisim_payi"]
    kl, ks = O["ornek"]["kisili"], O["ornek"]["kisisiz"]
    i = next(j for j in range(min(len(kl), len(ks))) if kl[j] != ks[j])
    bas = kl.rfind(" is ", 0, i)
    son_l, son_s = kl.find(".", i), ks.find(".", i)
    ex_l, ex_s = kl[bas + 1:son_l], ks[bas + 1:son_s]
    assert "you" in ex_l and "you" not in ex_s, (ex_l, ex_s)
    print(f"  [PAYDA] v61_ek: n_taranan={ck['n_taranan']} · n_istem={ck['n_istem']} · p50={dp['p50']:.4f} · "
          f"armorm Δ={a['delta']} pla={a['delta_plasebo']} · skywork Δ={k['delta']} pla={k['delta_plasebo']} "
          f"· örnek istem_i={O['ornek']['istem_i']} ⇒ esik: ad/güc degisirse ⇒ EYLEM: paragraf YAZILMAZ")
    s = (f"The re-scoring of \\S\\ref{{sec:dial}} draws one response per prompt from the T\\\"ulu-3 preference "
         f"mixture, ${binlik(ck['n_istem'])}$ in all from ${binlik(ck['n_taranan'])}$ scanned, keeping only responses "
         f"that accept a person swap. Ten rules replace second-person marking with an impersonal form in a fixed order, "
         f"checked before scoring for subject--verb agreement and sentence-initial capitals; the reverse direction uses "
         f"the nine that invert, since \\emph{{your}}$\\rightarrow$\\emph{{the}} does not. The median rewrite "
         f"changes ${100*dp['p50']:.2f}\\%$ of the characters (${100*dp['p05']:.2f}$ to ${100*dp['p95']:.1f}\\%$ "
         f"between the 5th and 95th percentiles); a typical pair turns \\emph{{``{ex_l}''}} into "
         f"\\emph{{``{ex_s}''}}. Each response and its rewrite are scored by ArmoRM-Llama3-8B-v0.1 and "
         f"Skywork-Reward-Llama-3.1-8B, and the statistic is the paired difference, person-free minus person-marked, "
         f"with a prompt-clustered 95\\% interval over ${binlik(H['n_bootstrap'])}$ resamples. The placebo shuffles "
         f"words that carry no person marking within the same response until as many characters have moved as the swap "
         f"moved, so it moves as much text without touching the persons --- though, unlike the rewrite, it breaks the syntax. The rewrite moves ArmoRM by "
         f"${a['delta']:.4f}$ $[{a['ci'][0]:.5f},{a['ci'][1]:.5f}]$ against ${a['delta_plasebo']:.3f}$ "
         f"$[{a['ci_plasebo'][0]:.4f},{a['ci_plasebo'][1]:.4f}]$ for its placebo, and Skywork by "
         f"${k['delta']:.3f}$ $[{k['ci'][0]:.2f},{k['ci'][1]:.2f}]$ against ${k['delta_plasebo']:.2f}$ "
         f"$[{k['ci_plasebo'][0]:.2f},{k['ci_plasebo'][1]:.2f}]$, pooled over both directions: "
         f"${abs(R['armorm_hepsi']['etki_sd_biriminde']):.3f}$ to ${abs(R['skywork_hepsi']['etki_sd_biriminde']):.3f}$ standard "
         f"deviations of the score, with the placebo moving ${k['delta_plasebo']/k['delta']:.1f}$ to "
         f"${a['delta_plasebo']/a['delta']:.0f}$ times further. The pool mixes two directions, "
         f"${binlik(R['armorm_ileri']['n'])}$ pairs that remove the persons from a person-marked original and "
         f"${binlik(R['armorm_ters']['n'])}$ that add them to a person-free one. Read as the rewritten version minus the "
         f"original, removing them moves ArmoRM by ${R['armorm_ileri']['delta']:.4f}$ "
         f"$[{R['armorm_ileri']['ci'][0]:.5f},{R['armorm_ileri']['ci'][1]:.5f}]$ and Skywork by "
         f"${R['skywork_ileri']['delta']:.2f}$ $[{R['skywork_ileri']['ci'][0]:.2f},{R['skywork_ileri']['ci'][1]:.2f}]$, and "
         f"adding them moves ArmoRM by ${-R['armorm_ters']['delta']:.4f}$ "
         f"$[{-R['armorm_ters']['ci'][1]:.5f},{-R['armorm_ters']['ci'][0]:.5f}]$ and Skywork by "
         f"${-R['skywork_ters']['delta']:.2f}$ $[{-R['skywork_ters']['ci'][1]:.2f},{-R['skywork_ters']['ci'][0]:.2f}]$. "
         f"Both scorers rate whichever version was rewritten lower, whichever way the persons move, so the pooled sign "
         f"follows the mix of directions and the test reads the edit rather than the person marking; this split was read "
         f"after the fact and is descriptive. Both also move far more under the equal-length shuffle, which is no "
         f"like-for-like yardstick because it damages the sentence the rewrite keeps "
         f"grammatical. A power check on the first "
         f"${H['guc_provasi_n']}$ pairs cleared the detectable-effect bar on both scorers before the full run. "
         f"\\textbf{{This test changes both the scorer and the corpus of the association it answers}}: that "
         f"association is read in UltraFeedback's ratings, which its dataset card says GPT-4 assigned, and the "
         f"rewrite in the T\\\"ulu-3 mixture. No model card of the sixteen models names either reward model, "
         f"which does not exclude one inside a pipeline that is not disclosed, and Skywork-Reward-Llama-3.1-8B names "
         f"Llama-3.1-8B-Instruct, one of the panel's own aligned checkpoints, as its base model: a scorer built on "
         f"the panel's own lineage does not penalise the persons either.")
    io.open(f"{OUT}/V61_PUANLAMA_EK.tex", "w", encoding="utf-8").write(s + "\n")
    json.dump(dict(table="V61_PUANLAMA_EK", ciktilar=["V61_PUANLAMA_EK.tex"],
                   sources=[dict(yol=y, sha256_16=hashlib.sha256(open(f"{ROOT}/{y}", "rb").read()).hexdigest()[:16]) for y in
                            ("results/v61_scoring_2026-09-11.json", "results/v61_scoring_generation_2026-09-12.json",
                             "results/v61_example_pair_2026-09-14.json", "results/v98_cpu_readings_2026-09-21.json")],
                   note="renders the sealed re-scoring reading and a reconstructed example pair; no statistic computed here"),
              io.open(f"{OUT}/V61_PUANLAMA_EK.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ V61_PUANLAMA_EK.tex ({len(s)} kar) · «{s[:300]}…»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
