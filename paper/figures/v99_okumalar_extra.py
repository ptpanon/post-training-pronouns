#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, statistics as st, sys
ROOT = __DNH_ROOT__ + ""
OUT = f"{ROOT}/paper/figures"
KAYNAK = ["results/v98_cpu_readings_2026-09-21.json", "results/v99_c2b_uyum_2026-09-21.json",
          "results/v96_q2_judge_2026-09-18.json", "results/v99_c2a_uyum_2026-09-21.json",
          "results/v99_sonnet_judge_2026-09-21.json", "results/v99_sonnet_full_2026-09-21.json",
          "results/v98_elicit_ozdes_2026-09-21.json", "results/template_filtered_2026-09-07.json", "results/v102_taze_uyum_2026-09-23.json",
          "results/jh_b100_2026-09-25.json",
          "results/jh_judge_generic_2026-09-25.json"]


def J(y):
    return json.load(io.open(f"{ROOT}/{y}", encoding="utf-8"))


def b(n):
    return f"{n:,}".replace(",", "{,}")


def yaz(ad, s, not_):
    io.open(f"{OUT}/{ad}.tex", "w", encoding="utf-8").write(s.rstrip() + "\n")
    json.dump(dict(table=ad, ciktilar=[f"{ad}.tex"], note=not_,
                   sources=[dict(yol=y, sha256_16=hashlib.sha256(open(f"{ROOT}/{y}", "rb").read()).hexdigest()[:16]) for y in KAYNAK]),
              io.open(f"{OUT}/{ad}.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ {ad}.tex ({len(s)} kar)")


def konum(O):
    r = O["R3_Q6_konum"]; A = r["aile"]; S = r["sayim"]
    oku = [a for a, v in A.items() if v["sonrasi"]["hal"] == "OKUNDU"]
    az = sorted((a, v["sonrasi"].get("n", 0)) for a, v in A.items() if v["sonrasi"]["hal"] != "OKUNDU")
    med = lambda parca, m: st.median(v[parca][m]["gozlenen"] for a, v in A.items() if v[parca]["hal"] == "OKUNDU")
    nmed = lambda parca: st.median(v[parca]["n"] for a, v in A.items() if v[parca]["hal"] == "OKUNDU")
    pla = lambda parca, m: sum(1 for a, v in A.items() if v[parca]["hal"] == "OKUNDU" and v[parca][m]["plasebo_ustu"])
    i1, i2 = S["ilk160"], S["sonrasi"]
    assert i1["M1_1k"]["asagi"] == 16 and len(oku) == 12 and i2["M1_1k"]["yukari"] == 0
    azad = [a for a, _ in az]; gem = [a for a in azad if a.startswith("Gemma-3")]
    assert len(gem) == 3 and "Mistral-7B-v0.3" in azad
    nmin, nmax = min(n for _, n in az), max(n for _, n in az)
    s = (r"\paragraph*{Past the first 160 tokens.}" "\n" r"\vekalet{On the debate prompts most aligned outputs reach the "
         r"160-token cap, so their rates describe openings. The instruction prompts of Appendix~\ref{app:D4} were generated "
         r"in a 2{,}048-token window, and there we read the first 160 tokens of each output and the rest as separate texts "
         r"(aligned checkpoint in its own template, base checkpoint in the in-context assistant prompt, the B1 cell of "
         r"Table~\ref{tab:cells}). "
         f"In the first 160 tokens ``you'' falls in {i1['M1_1k']['asagi']} of 16 models, all {pla('ilk160','M1_1k')} clearing the paired "
         f"placebo, and ``I'' in {i1['IS1_1k']['asagi']} ({i1['IS1_1k']['yukari']} rise). After them, Mistral-7B-v0.3 and the "
         f"three Gemma-3 models write too little to read (${nmin}$ to ${nmax}$ pairs). Of the other {len(oku)}, ``you'' falls with "
         f"an interval excluding zero in {i2['M1_1k']['asagi']} and rises in none, at a median shift of ${med('sonrasi','M1_1k'):.2f}$ "
         f"per 1{{,}}000 against ${med('ilk160','M1_1k'):.2f}$ in the first 160 tokens, but on a median of ${b(int(nmed('sonrasi')))}$ "
         f"pairs rather than ${b(int(nmed('ilk160')))}$, and {'only one model clears' if pla('sonrasi','M1_1k') == 1 else str(pla('sonrasi','M1_1k')) + ' models clear'} "
         f"its placebo. ``I'' falls in {i2['IS1_1k']['asagi']} of {len(oku)} after the first 160 tokens, at a median of "
         f"${med('sonrasi','IS1_1k'):.2f}$ against ${med('ilk160','IS1_1k'):.2f}$. The fall in ``you'' keeps its size where later "
         r"text exists, then, but the later text is too sparse to separate it model by model, and the first-person fall "
         r"is concentrated in the openings on these prompts. The reading is descriptive; its path was committed before the count.}")
    yaz("KONUM_D2", s, "Q6: first 160 tokens vs the rest, instruction prompts (R3)")


def kip(O, Y, CA, CS):
    r = O["R2_Q2_kip_ayrismasi"]; a = r["ayrisma"]; t = r["tur_sabit_aralik"]
    assert CA["verdict"] == "GIRMEZ" and CA["N"] == 100
    SA = CS["paket"]["A"]; sh, st_ = SA["katman"]["hizali"], SA["katman"]["taban"]
    SS = J(KAYNAK[7])["sayim"]["sablonlu"]; E = 0.20
    uyg = [x for x, v in Y["aile"].items() if v["taban"]["pay_CONTINUE"] >= E and v["hizali"]["pay_CONTINUE"] >= E]
    ch = {x: v["hizali"]["pay_CONTINUE"] for x, v in Y["aile"].items()}
    enb = max(ch, key=ch.get); ikinci = max(v for x, v in ch.items() if x != enb)
    rp = {x: Y["aile"][x]["hizali"]["pay_REPLY"] for x in Y["aile"]}
    ryuk = [rp[x] for x in SS["pozitif"]]
    assert len(SS["pozitif"]) == 7 and not uyg
    enb_ad = enb.replace("Tulu3-8B", 'T\\"ulu-3-8B')
    tb = Y["aile"][enb]["taban"]["pay_CONTINUE"]; assert tb < E <= ch[enb]
    rdig = [rp[x] for x in rp if x not in SS["pozitif"]]
    q5 = (f"Within outputs the judge labels as continuations, the template split cannot be read: no model has "
          f"${100*E:.0f}\\%$ continuations in both checkpoints, the share its rule requires (the other aligned checkpoints reach at most "
          f"${100*ikinci:.0f}\\%$; {enb_ad} reaches ${100*ch[enb]:.0f}\\%$ against ${100*tb:.0f}\\%$ for its base). "
          f"The seven models whose rate rises under the template reply to the prompt in ${100*min(ryuk):.0f}\\%$ to "
          f"${100*max(ryuk):.0f}\\%$ of their aligned outputs, the other nine in ${100*min(rdig):.0f}\\%$ to "
          f"${100*max(rdig):.0f}\\%$. Replying rather than continuing does not separate the risers from the rest, and the "
          f"label does not say whether a reply comments on the prompt as a prompt.")
    assert SA["verdict"] == "KALIR" and SA["N"] == 100
    kh, kt, M = CA["katman"]["hizali"], CA["katman"]["taban"], CA["matris"]["M"]
    ct_cr = sum(1 for x in CA["ayrisan"] if x["insan"] == "CONTINUE" and x["yargic"] == "REPLY" and x["bacak"] == "taban")
    nh = sum(v["hizali"]["n"] for v in Y["aile"].values())
    n_top = a["n_hucre"] + a["n_dusen"]
    assert abs(a["kip_ici_pay"] + a["kipler_arasi_pay"] - 1) < 1e-3 and set(r["ucten_buyuk"]) == {"TUR_FORUM_hizali", "TUR_NASIL_hizali"}
    s = (r"\paragraph*{Between and within opening modes, by a judge.}" "\n"
         r"\vekalet{Because the rule's \emph{continue} class is imprecise, a judge also labelled the chat-template outputs: "
         f"Qwen2.5-32B-Instruct at temperature 0, a Qwen2.5 model like six of the sixteen, read ${b(nh)}$ aligned outputs "
         r"drawn by opening class and marked each as a reply to the user turn, a continuation of it, or neither; its labels are "
         r"weighted back to each model's opening classes. We split the variance of $\ln M_1$ across the aligned outputs into a "
         f"between-mode and a within-mode part over the model-by-mode cells that hold at least ${100*a['esik_pay']:.0f}\\%$ of a "
         f"model's outputs ({a['n_hucre']} of {n_top}; the other {a['n_dusen']}, all continuations, fall below that share or were "
         f"not sampled). ${100*a['kip_ici_pay']:.0f}\\%$ of the variance lies within modes. Replies carry "
         f"${100*a['kip_agirlik']['REPLY']:.0f}\\%$ of the weight and alone spread with a geometric standard deviation of "
         f"${a['gsd_kip_ici']['REPLY']:.2f}$, against ${a['gsd_toplam']:.2f}$ over these cells and $2.35$ over all generations "
         r"(\S\ref{sec:deperson}). The spread under the template is therefore present among replies, not mainly a difference "
         r"between models that answer and models that continue. On well-formed input, with the genre fixed by a header on raw "
         r"continuation (Appendix~\ref{app:D2}), the aligned checkpoints span "
         f"${t['TUM_hizali']['oran']:.2f}\\times$ (geometric SD ${t['TUM_hizali']['gsd']:.2f}$) against "
         f"${t['TUM_taban']['oran']:.2f}\\times$ (${t['TUM_taban']['gsd']:.2f}$) for the bases, and the aligned range exceeds "
         f"$3\\times$ in two of the four genres, forum posts (${t['TUR_FORUM_hizali']['oran']:.2f}\\times$, bases "
         f"${t['TUR_FORUM_taban']['oran']:.2f}\\times$) and how-to guides (${t['TUR_NASIL_hizali']['oran']:.2f}\\times$, bases "
         f"${t['TUR_NASIL_taban']['oran']:.2f}\\times$). Both readings are descriptive, their paths committed before the count. "
         f"Against one human labeller on ${CA['N']}$ outputs drawn by model and checkpoint, the judge's label agrees on ${CA['X']}$ "
         f"(raw agreement ${CA['ham_uyum']:.2f}$, model-clustered $95\\%$ interval $[{CA['ham_uyum_ci'][0]:.2f}, {CA['ham_uyum_ci'][1]:.2f}]$; "
         f"Cohen's $\\kappa={CA['kappa']:.2f}$, chance agreement ${CA['p_e']:.2f}$). The two mode labels match for ${kh['uyum']}$ of "
         f"${kh['n']}$ aligned-checkpoint generations but for ${kt['uyum']}$ of ${kt['n']}$ base-checkpoint generations; most "
         f"mismatches are texts the labeller read as continuations and the judge called replies (${M[1][0]}$, ${ct_cr}$ from base checkpoints). That is below the $0.80$ we fixed before comparing. "
         f"A second label judge (Claude Sonnet~5, same model family as the assistant) was tried after the first missed the bar; its agreement "
         f"on the same 100 items is ${SA['X']}/100$ ($\\kappa={SA['kappa']:.2f}$, chance agreement ${SA['p_e']:.2f}$; on those "
         f"same outputs it matches the labeller ${sh['uyum']}$ times among aligned checkpoints and ${st_['uyum']}$ among bases); its labels are used only "
         r"where that bar is met. Both judges were compared with the labeller on these same 100 outputs, so the second was chosen "
         r"on them. It does not meet the bar either, so this decomposition stays here, labelled by the open-weight cross-check judge, and "
         r"does not enter the main text. " + q5 + "}")
    yaz("KIP_D3", s, "Q2: between/within opening mode decomposition (judge-labelled) + genre-fixed ranges (R2)")


def kk3f(O):
    r = O["R4_Q3_zamirsiz_ciftler"]; m = r["olcum"]
    sat = [("tokens", "jeton", "{:.0f}", "{:+.0f}", "{:+.1f}"), ("sentences", "cumle", "{:.0f}", "{:+.0f}", "{:+.2f}"),
           ("list or heading line share", "liste_satir_payi", "{:.2f}", "{:+.0f}", "{:+.3f}"),
           ("code fence present", "kod_citi_payi", "{:.0f}", "{:+.0f}", "{:+.3f}"),
           ("second person per 1{,}000", "M1_1k", "{:.0f}", "{:+.0f}", "{:+.3f}"),
           ("first person per 1{,}000", "IS1_1k", "{:.0f}", "{:+.0f}", "{:+.2f}")]
    rows = []
    for ad, k, f0, f1, f2 in sat:
        x = m[k]
        z = lambda v, f: "0" if v == 0 else f.format(v)
        ci = f"[{z(x['ci'][0], f1)}, {z(x['ci'][1], f1)}]"
        rows.append(f"{ad} & ${f0.format(x['chosen_ortanca'])}$ & ${f0.format(x['rejected_ortanca'])}$ & "
                    f"${z(x['ortanca_fark'], f1)}$ ${ci}$ & ${f2.format(x['ortalama_fark'])}$ & "
                    f"${100*x['pozitif_pay']:.0f}$ & ${100*x['sifir_pay']:.{2 if x['sifir_pay'] > 0.999 else 0}f}$ \\\\")
    s = (r"\paragraph*{What differs inside the pronoun-free pairs.}" "\n"
         f"\\vekalet{{The run on pairs with no pronoun contrast (\\S\\ref{{sec:dial}}) used ${b(r['n_cift'])}$ pairs from "
         f"{r['n_kume']} source datasets. The table compares the chosen with the rejected side of each pair. The second person "
         f"is equal in ${100*m['M1_1k']['sifir_pay']:.2f}\\%$ of pairs and absent from most, but the first person is not: the "
         f"chosen side has ${-m['IS1_1k']['ortalama_fark']:.2f}$ fewer first-person forms per 1{{,}}000 on average, "
         f"{100*m['IS1_1k']['sifir_pay']:.0f}\\% of pairs are equal and {100*m['IS1_1k']['pozitif_pay']:.0f}\\% have more on the chosen side. "
         f"The chosen side is longer by a median of ${m['jeton']['ortanca_fark']:.0f}$ tokens, while sentence count, list "
         r"lines and code fences differ by a median of zero. Intervals resample the source datasets 1{,}000 times; with "
         r"seven sources they are coarse. The reading is descriptive, and its path was committed before the count.}" "\n"
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}" "\n"
         r"\begin{tabular}{lrrrrrr}\toprule" "\n"
         r" & \multicolumn{2}{c}{median} & \multicolumn{2}{c}{chosen $-$ rejected} & \multicolumn{2}{c}{pairs (\%)} \\" "\n"
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
         r"measure & chosen & rejected & median [95\% CI] & mean & chosen higher & equal \\ \midrule" "\n"
         + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}\end{center}")
    yaz("KK3F_D7", s, "Q3: chosen vs rejected in the pronoun-free pairs (R4)")


def c2b(C, Y, CS, CT):
    M = C["matris"]["M"]; kat = C["katman"]
    assert C["verdict"] == "GIRMEZ" and C["N"] == 100
    SB = CS["paket"]["B"]; sk = SB["katman"]
    assert SB["verdict"] == "GECER" and SB["N"] == 100 and CT["hal"] == "OKUNDU" and CT["n_oge"] == 2000
    G = J(KAYNAK[10])
    assert G["alet_sha"].startswith("b6673dce") and not G["sayim"]["aralik_kurulamadi"] and len(G["aile"]) == 16
    gs = G["sayim"]; yuk_ = gs["yukselir"]
    jenerik = (f"The generic rate, the second-person rate times the generic share of its cell, falls in ${gs['duser']}$ of ${len(G['aile'])}$ "
               f"and rises in {'none' if yuk_ == 0 else f'${yuk_}$'}; ${gs['ayrik_asagi']}$ of the falls have a prompt-clustered $95\\%$ interval "
               f"below zero and ${gs['plasebo_ustu_asagi']}$ exceed their paired placebo, a split of the base checkpoint's outputs into two "
               r"halves within each prompt ($200$ splits). ")
    ikinci = (f"A second label judge (Claude Sonnet~5, same model family as the assistant) was tried after the first missed the bar; its agreement on "
              f"the same 100 items is ${SB['X']}/100$ (model-clustered $95\\%$ interval $[{SB['ham_uyum_ci'][0]:.2f}, {SB['ham_uyum_ci'][1]:.2f}]$, "
              f"$\\kappa={SB['kappa']:.2f}$ $[{SB['kappa_ci'][0]:.2f}, {SB['kappa_ci'][1]:.2f}]$, chance agreement ${SB['p_e']:.2f}$; "
              f"on the same sentences it matches the labeller ${sk['hizali']['uyum']}$ times among aligned and ${sk['taban']['uyum']}$ among base checkpoints; "
              f"the two judges agree with each other on ${SB['sonnet_qwen']['X']}$); its labels are used only where that bar is met. "
              r"Both judges were compared with the labeller on these same 100 sentences, so the second was chosen on them. "
              f"It meets the bar, and it then labelled all ${b(CT['n_oge'])}$ sentences under the same prompt, three times with a "
              f"majority label. With its labels, the addressed rate (the second-person rate times the addressed share of its cell) "
              f"falls from base to aligned on raw continuation in ${CT['cip_dM1_addressed_negatif']}$ of 16 models and rises in "
              f"${CT['cip_dM1_addressed_pozitif']}$ (\\S\\ref{{sec:measure}}); with the open-weight cross-check judge's labels it falls in "
              f"${CT['qwen']['cip_dM1_addressed_negatif']}$. " + jenerik)
    s = (r"\paragraph{Addressed or generic ``you'': two judges against one labeller.}" "\n"
         r"\vekalet{The counters cannot tell a ``you'' addressed to the reader from a generic one (\S\ref{sec:measure}). "
         r"A judge, Qwen2.5-32B-Instruct at temperature 0, labelled $2{,}000$ second-person sentences drawn across the sixteen "
         r"models and both checkpoints as addressed to a particular reader, generic, or neither (a quotation, invented "
         r"dialogue, or undeterminable). One labeller, one of the authors, answered the same three-way question for $100$ of those sentences, drawn "
         r"by model; the package named neither the model, the checkpoint nor the judge's label. "
         f"The two agree on ${C['X']}$ of ${C['N']}$ (model-clustered $95\\%$ interval "
         f"$[{C['ham_uyum_ci'][0]:.2f}, {C['ham_uyum_ci'][1]:.2f}]$), Cohen's $\\kappa={C['kappa']:.2f}$ "
         f"$[{C['kappa_ci'][0]:.2f}, {C['kappa_ci'][1]:.2f}]$, against a chance agreement of ${C['p_e']:.2f}$. "
         f"The disagreements sit on the boundary the counters cannot see: ${M[0][1]}$ sentences the labeller read as addressed "
         f"the judge called generic and ${M[1][0]}$ went the other way, and of the ${sum(M[2])}$ the labeller left undetermined "
         f"the judge agreed on ${M[2][2]}$. Agreement is ${kat['hizali']['uyum']}$ of ${kat['hizali']['n']}$ on aligned outputs "
         f"and ${kat['taban']['uyum']}$ of ${kat['taban']['n']}$ on base outputs. We had fixed, before comparing the two, that the "
         r"judge's split would enter the main text only at an agreement of at least $0.80$; it falls short. The judge is a "
         r"Qwen2.5 model, like six of the sixteen. " + ikinci +
         r"Labels were assigned under a three-question rule: (1) a quoted, fictional, or template- or prompt-artefact "
         r"sentence is UNDETERMINED; (2) a ``you'' that refers to something this interlocutor said, asked, did or has "
         r"is ADDRESSED; (3) otherwise, a ``you'' that can be replaced by ``one'', ``anyone'' or ``people'' without "
         r"changing the meaning is GENERIC. The rule was fixed after cell 40 and cells 1--40 were re-read under it.}")
    T = J(KAYNAK[8])
    assert T["sonnet"]["okunamayan"] == 0 and T["paket"]["n_hucre"] == 30 and not T["sema_disi"]
    n30 = T["paket"]["n_hucre"]
    taze = (f" After the second label judge had been chosen, the same labeller answered the same question for ${n30}$ further "
            f"sentences drawn from the same $2{{,}}000$, none of them among the $100$: the two agree on "
            f"${T['sonnet']['uyan']}$ of ${n30}$, the open-weight cross-check judge and the labeller on ${T['qwen']['uyan']}$, and the two judges "
            f"on ${T['yargiclar_arasi']['uyan']}$. At this size we read the totals only: the package's own frame lists agreement "
            r"per class and per checkpoint as unmeasurable here, and a $95\%$ interval on a rate of this kind spans about "
            r"$\pm 0.14$.")
    H = J(KAYNAK[9])
    assert H["prereg"].endswith("@550b5754") and H["n"] == 100 and not H["mukerrer"] and not H["sema_disi"]
    (hj, hnj), (hh, hnh) = H["ozet"]["jenerik"], H["ozet"]["hitap"]
    ai = H["aralik95_aile_kumeli"]
    ipucu = (f" The cue classifier of Appendix~\\ref{{app:D1}} labels ${hnj + hnh}$ of the same $100$ sentences. Its generic label "
             f"matches the labeller in ${hj}$ of ${hnj}$ and its addressed label in ${hh}$ of ${hnh}$ (model-clustered $95\\%$ "
             f"intervals $[{ai['jenerik'][0]:.2f}, {ai['jenerik'][1]:.2f}]$ and $[{ai['hitap'][0]:.2f}, {ai['hitap'][1]:.2f}]$), "
             f"and the labeller read ${H['ozet']['belirsiz_cozulen']}$ of the ${H['ozet']['belirsiz']}$ it leaves undetermined "
             f"as addressed or generic.")
    assert s.endswith("}")
    s2 = s[:-1] + taze + ipucu + "}"
    yaz("C2B_EK_C", s2, "package B: judge vs one human labeller (C2B) + 30 fresh items (v102) + cue classifier (v121)")
    yaz("JH_YARGIC_CUMLE",
        f"A judge was validated on {SB['N']} sentences (human agreement {SB['X']} of {SB['N']}) and on {n30} fresh items labelled after it was "
        f"chosen ({T['sonnet']['uyan']} of {n30}; Appendix~\\ref{{app:labelcheck}}). By this judge the "
        f"addressed half falls in {CT['cip_dM1_addressed_negatif']} and rises in {CT['cip_dM1_addressed_pozitif']}, and the generic half falls "
        f"in {G['sayim']['duser']} of {len(G['aile'])} ({G['sayim']['ayrik_asagi']} with intervals below zero; Appendix~\\ref{{app:labelcheck}}). "
        f"We claim no decomposition; the cue classifier is checked against the authors' labels "
        r"in Appendix~\ref{app:D1}.",
        "v121: §2 block, sealed wording (PREREG_JH_YARGIC_JENERIK §6), cards V99_SONNET_YARGIC · V102_TAZE_B_UYUM · V99_SONNET_B_TAM · JH_YARGIC_JENERIK")


def elicit_ozdes(E):
    S, P, T, A = E["sayim"], E["asagi_ve_plasebo_ustu"], E["tasarim"], E["aile"]
    sinif = lambda v: "asagi" if v["ayrik"] and v["yon"] == "asagi" else ("yukari" if v["ayrik"] else "null")
    yuk = sorted(a for a, v in A.items() if sinif(v["M1_1k"]) == "yukari")
    assert len(A) == 16 and all(v["ozdes_dize_payi"] == 1.0 for v in A.values()) and E["dize_ayrik"] in (0, [], None)
    assert S["M1_1k"] == {"asagi": 12, "yukari": 1, "null": 3} and S["IS1_1k"]["asagi"] == 15 and len(yuk) == 1
    m, i, dm, di = S["M1_1k"], S["IS1_1k"], S["duz_M1_1k"], S["duz_IS1_1k"]
    s = (f"On an identical string, with the aligned model's own chat template around each prompt given to both checkpoints "
         f"({T['n_cekim']} samples of each of the {T['n_uyaran']} prompts, up to {b(T['yeni_jeton'])} new tokens; the strings match "
         f"in all {len(A)} models), ``you'' falls in {m['asagi']} models, rises in {m['yukari']} ({yuk[0]}) and does not separate "
         f"in {m['null']}, and {P['M1_1k']} of those {m['asagi']} clear the paired placebo; ``I'' falls in {i['asagi']} and does "
         f"not separate in {i['null']}, {P['IS1_1k']} clearing the placebo. On prose lines alone, with list, numbered and heading "
         f"lines removed, the second person splits ({dm['asagi']} fall, {dm['yukari']} rise, {dm['null']} do not separate) while the "
         f"first person falls in {di['asagi']} and rises in {'none' if di['yukari'] == 0 else di['yukari']}. This reading is descriptive, and its path was committed "
         f"before the generations existed.")
    yaz("ELICIT_OZDES_D2", s, "B1: ELICIT-99 with both checkpoints on the identical string (V98_ELICIT_OZDES)")


def main():
    O = J(KAYNAK[0])["okumalar"]; C = J(KAYNAK[1]); Y = J(KAYNAK[2])
    elicit_ozdes(J(KAYNAK[6]))
    print(f"{len(KAYNAK)} {C['verdict']}"
          f"")
    konum(O); kip(O, Y, J(KAYNAK[3]), J(KAYNAK[4])); kk3f(O); c2b(C, Y, J(KAYNAK[4]), J(KAYNAK[5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
