#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, json, os, re, subprocess, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
RULE = f"{ROOT}/preregistration/rule_v96_q2_judge_2026-09-18.md"
DIS = __DNH_DATA__ + "/v96_c2"
CARD_SS = f"{ROOT}/results/template_filtered_2026-09-07.json"
CIK = f"{ROOT}/results/v96_q2_judge_2026-09-18.json"
URL = "http://127.0.0.1:8013/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-32B-Instruct"
RNG = 20260918
N_A_KATMAN = 100
N_B_AILE = 125
PAY_ESIK = 0.20
N_BOOT = 1000

ISTEM_A = """You will see a user turn and a language model's output that follows it.

USER TURN:
<<<{istem}>>>

MODEL OUTPUT (may be cut off):
<<<{cikti}>>>

Decide what the output does with the user turn.
REPLY: the output responds to the user turn as an assistant would. It talks about the user's text or to the user (for example "You've presented a statement...", "I will argue that...", "Here are some points to consider:"), or it agrees, disagrees with or comments on the statement as something another person said.
CONTINUE: the output continues the user's text as its own author, going on with the argument the user turn started (for example giving the reasons after "Here is why:"), without treating the text as something another person said.
OTHER: a refusal, an empty or broken output, only a fabricated new conversation turn, or you cannot tell.

Answer with exactly one word: REPLY, CONTINUE or OTHER."""

ISTEM_B = """Below is a sentence from a text that a language model wrote, with the sentences before and after it. The marked sentence contains "you" (or "your", "yours", "yourself", "yourselves").

BEFORE: <<<{once}>>>
MARKED SENTENCE: <<<{cumle}>>>
AFTER: <<<{sonra}>>>

Who does "you" refer to in the marked sentence?
ADDRESSED: a specific reader or interlocutor whom the text is speaking to (the person the text is written for or replying to).
GENERIC: people in general, anyone, "one" (for example "you can see that prices rise when demand grows").
OTHER: "you" inside a quotation, reported speech or a fictional dialogue, or you cannot tell.

Answer with exactly one word: ADDRESSED, GENERIC or OTHER."""
ETIKET = {"A": ("REPLY", "CONTINUE", "OTHER"), "B": ("ADDRESSED", "GENERIC", "OTHER")}


def damga(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def sha16(y): return hashlib.sha256(open(y, "rb").read()).hexdigest()[:16]


def rule_kapisi():
    if not os.path.exists(RULE):
        raise SystemExit("★ RULE YOK ⇒ yargilama YOK")
    h = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "-1", "--", os.path.relpath(RULE, ROOT)],
                       capture_output=True, text=True).stdout.strip()
    if not h:
        raise SystemExit("")
    b = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    if b not in open(RULE, encoding="utf-8").read():
        raise SystemExit(f"{b[:16]}")
    return h, b[:16]


def aileler():
    return [(r["aile"], r["zemin_taban"], r["zemin_hizali"]) for r in json.load(open(CARD_SS, encoding="utf-8"))["aileler"]]


def bacaklar(aile, zt, zh):
    import template_filtered as SS
    bac = {("sbl", "taban"): SS.oku(SS.SAB, aile, zt), ("sbl", "hizali"): SS.oku(SS.SAB, aile, zh),
           ("cip", "taban"): SS.oku(SS.CIP, aile, zt), ("cip", "hizali"): SS.oku(SS.CIP, aile, zh)}
    F = {k: SS.bayraklar(R) for k, R in bac.items()}; K = {k: [SS.anahtar(r) for r in R] for k, R in bac.items()}
    kirli = set()
    for k in bac:
        kirli |= {K[k][i] for i in np.where(F[k][0])[0]}
    tut = {k: np.array([x not in kirli for x in K[k]]) for k in bac}
    return bac, K, tut


CUMLE = re.compile(r"(?<=[.!?])\s+|\n+")


def cumleler(t):
    return [c.strip() for c in CUMLE.split(t or "") if c and c.strip()]


def orneklem():
    import continuation_mode_exit as Q2, addressee_olcu as MO
    rng = np.random.default_rng(RNG); os.makedirs(DIS, exist_ok=True)
    A, B, nufus = [], [], {}
    for aile, zt, zh in aileler():
        bac, K, tut = bacaklar(aile, zt, zh)
        cip_istem = {K[("cip", "taban")][i]: r["onek"] for i, r in enumerate(bac[("cip", "taban")])}
        for b in ("taban", "hizali"):
            R = bac[("sbl", b)]; T = tut[("sbl", b)]
            kip = np.array([Q2.sinif(r["metin"]) for r in R])
            for sinif_ad, m in (("DEVAM", kip == "DEVAM"), ("DEVAM-DEGIL", kip != "DEVAM")):
                ix = np.where(T & m)[0]
                nufus[f"{aile}|{b}|{sinif_ad}"] = int(len(ix))
                sec = rng.choice(ix, min(N_A_KATMAN, len(ix)), replace=False) if len(ix) else []
                for i in sorted(int(x) for x in sec):
                    A.append(dict(kimlik=f"A|{aile}|{b}|{i}", aile=aile, bacak=b, satir=i, rule_sinif=kip[i], rule_kip=str(kip[i]),
                                  katman=sinif_ad, istem=cip_istem.get(K[("sbl", b)][i], ""), cikti=R[i]["metin"] or ""))
        havuz = {}
        for bic in ("cip", "sbl"):
            for b in ("taban", "hizali"):
                R = bac[(bic, b)]; T = tut[(bic, b)]; H = []
                for i in np.where(T)[0]:
                    C = cumleler(R[i]["metin"])
                    for j, c in enumerate(C):
                        n2 = len(MO.SAHIS2.findall(c))
                        if n2:
                            H.append((int(i), j, n2, c, C[j - 1] if j else "", C[j + 1] if j + 1 < len(C) else ""))
                havuz[(bic, b)] = H
                nufus[f"{aile}|{bic}|{b}|cumle"] = len(H)
                nufus[f"{aile}|{bic}|{b}|bicim"] = int(sum(h[2] for h in H))
        hucre_n = N_B_AILE // 4 + np.array([1 if k < N_B_AILE % 4 else 0 for k in range(4)])
        for (bic, b), n in zip(havuz, hucre_n):
            H = havuz[(bic, b)]
            sec = rng.choice(len(H), min(int(n), len(H)), replace=False) if H else []
            for s in sorted(int(x) for x in sec):
                i, j, n2, c, o, so = H[s]
                B.append(dict(kimlik=f"B|{aile}|{bic}|{b}|{i}|{j}", aile=aile, bicim=bic, bacak=b, satir=i, cumle_ix=j,
                              n_bicim=n2, cumle=c, once=o, sonra=so))
        print(f"  {aile:16s} A {sum(1 for x in A if x['aile'] == aile)} · B {sum(1 for x in B if x['aile'] == aile)}", flush=True)
    for ad, L in (("A", A), ("B", B)):
        with open(f"{DIS}/{ad}_orneklem.jsonl", "w", encoding="utf-8") as f:
            for x in L:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump(dict(nufus=nufus, n_A=len(A), n_B=len(B), rng=RNG, damga_utc=damga()),
              open(f"{DIS}/orneklem_kunye.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  [PAYDA] v96_q2_orneklem: n_A={len(A)} · n_B={len(B)} · n_aile={len(aileler())}")
    return 0


def yargila(gorev):
    import concurrent.futures as cf, urllib.request
    h, b = rule_kapisi()
    X = [json.loads(l) for l in open(f"{DIS}/{gorev}_orneklem.jsonl", encoding="utf-8")]
    cik = f"{DIS}/{gorev}_yargi.jsonl"
    bitti = {json.loads(l)["kimlik"] for l in open(cik, encoding="utf-8")} if os.path.exists(cik) else set()
    kalan = [x for x in X if x["kimlik"] not in bitti]
    print(f"★ v96 C(2) · görev {gorev} · rule {h} · alet {b} · n {len(X)} · kalan {len(kalan)} · {MODEL} T=0", flush=True)

    def tek(x):
        ist = ISTEM_A.format(istem=x["istem"], cikti=x["cikti"]) if gorev == "A" else \
              ISTEM_B.format(once=x["once"], cumle=x["cumle"], sonra=x["sonra"])
        body = json.dumps(dict(model=MODEL, messages=[{"role": "user", "content": ist}], temperature=0, seed=0,
                               max_tokens=4)).encode()
        for deneme in range(3):
            try:
                req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    t = json.loads(r.read())["choices"][0]["message"]["content"]
                w = (re.findall(r"[A-Z]+", t.upper()) or [""])[0]
                return dict(kimlik=x["kimlik"], ham=t, etiket=w if w in ETIKET[gorev] else "SEMA-DISI")
            except Exception as e:
                hata = f"{type(e).__name__}: {e}"
                time.sleep(2 * (deneme + 1))
        return dict(kimlik=x["kimlik"], ham=None, etiket="HATA", hata=hata)
    t0 = time.time(); n = 0; say = {}
    with open(cik, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(64) as ex:
        for y in ex.map(tek, kalan):
            f.write(json.dumps(y, ensure_ascii=False) + "\n"); n += 1; say[y["etiket"]] = say.get(y["etiket"], 0) + 1
            if n in (200, 1000) or n % 2000 == 0:
                hiz = n / max(time.time() - t0, 1e-9)
                print(f"  [{time.time() - t0:.0f} sn] {n}/{len(kalan)} · {say} · ETA {(len(kalan) - n) / hiz / 60:.1f} dk "
                      f"⇒ esik HATA+SEMA-DISI > %2 ⇒ EYLEM: dur, bak", flush=True)
    red = say.get("HATA", 0) + say.get("SEMA-DISI", 0)
    print(f"  [PAYDA] v96_q2_yargi_{gorev}: n={len(X)} · n_yeni={n} · etiket {say} · red_hata_sema={red} · {time.time() - t0:.0f} sn")
    return 0 if red <= 0.02 * max(n, 1) else 3


def paket():
    rng = np.random.default_rng(RNG + 1)
    os.makedirs(f"{DIS}/insan", exist_ok=True); os.makedirs(f"{DIS}/anahtar", exist_ok=True)
    A = [json.loads(l) for l in open(f"{DIS}/A_orneklem.jsonl", encoding="utf-8")]
    B = [json.loads(l) for l in open(f"{DIS}/B_orneklem.jsonl", encoding="utf-8")]
    kunye = {}
    for ad, X, katman, soru, izin in (
            ("A", A, lambda x: (x["aile"], x["bacak"]), ""
             "", ["REPLY", "CONTINUE", "OTHER"]),
            ("B", B, lambda x: (x["aile"],), ""
             "", ["ADDRESSED", "GENERIC", "OTHER"])):
        grup = {}
        for x in X:
            grup.setdefault(katman(x), []).append(x)
        sec = []
        anahtarlar = sorted(grup)
        per = 100 // len(anahtarlar)
        for k in anahtarlar:
            g = grup[k]; sec += [g[int(i)] for i in rng.choice(len(g), min(per, len(g)), replace=False)]
        kalan = [x for x in X if x not in sec]
        sec += [kalan[int(i)] for i in rng.choice(len(kalan), 100 - len(sec), replace=False)]
        rng.shuffle(sec)
        yol = f"{DIS}/insan/INSAN_PAKETI_V96_C2_{ad}_100.md"; anah = f"{DIS}/anahtar/ANAHTAR_V96_C2_{ad}.jsonl"
        with open(yol, "w", encoding="utf-8") as f, open(anah, "w", encoding="utf-8") as g:
            f.write(f"{ad} {soru}"
                    ""
                    f"{'|'.join(izin)}")
            for k, x in enumerate(sec, 1):
                if ad == "A":
                    f.write(f"### Hücre {k}/100\n\n**KULLANICI METNI:** {x['istem']}\n\n**MODEL CIKTISI:** {x['cikti']}\n\n"
                            f"`h={k} cevap=` **{' / '.join(izin)}**\n\n---\n\n")
                else:
                    f.write(f"### Hücre {k}/100\n\n{x['once']} ► **{x['cumle']}** {x['sonra']}\n\n"
                            f"`h={k} cevap=` **{' / '.join(izin)}**\n\n---\n\n")
                g.write(json.dumps(dict(hucre=k, kimlik=x["kimlik"]), ensure_ascii=False) + "\n")
        kunye[ad] = dict(yol=yol, sha256=hashlib.sha256(open(yol, "rb").read()).hexdigest(), anahtar=anah, n_hucre=len(sec),
                         soru=soru, rubrikler={"cevap": izin}, katman="aile × bacak" if ad == "A" else "aile",
                         korluk="")
    json.dump(dict(kutu="v96-C2", paketler=kunye, damga_utc=damga(), rule=os.path.relpath(RULE, ROOT)),
              open(f"{ROOT}/results/KUNYE_INSAN_V96_C2_2026-09-18.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  [PAYDA] v96_q2_paket: n_paket=2 · n_hucre={[v['n_hucre'] for v in kunye.values()]}")
    return 0


def oku():
    import addressee_run as MK, form_count as FS
    nlp = FS._boru()
    A = {x["kimlik"]: x for x in (json.loads(l) for l in open(f"{DIS}/A_orneklem.jsonl", encoding="utf-8"))}
    B = {x["kimlik"]: x for x in (json.loads(l) for l in open(f"{DIS}/B_orneklem.jsonl", encoding="utf-8"))}
    YA = {y["kimlik"]: y["etiket"] for y in (json.loads(l) for l in open(f"{DIS}/A_yargi.jsonl", encoding="utf-8"))}
    YB = {y["kimlik"]: y["etiket"] for y in (json.loads(l) for l in open(f"{DIS}/B_yargi.jsonl", encoding="utf-8"))}
    nufus = json.load(open(f"{DIS}/orneklem_kunye.json", encoding="utf-8"))["nufus"]
    rng = np.random.default_rng(RNG + 2)
    S, kip_M1 = {}, {b: {"REPLY": {}, "CONTINUE": {}} for b in ("taban", "hizali")}
    kar = {b: np.zeros((2, 3)) for b in ("taban", "hizali")}
    for aile, zt, zh in aileler():
        bac, K, tut = bacaklar(aile, zt, zh)
        S[aile] = {}
        for b in ("taban", "hizali"):
            R = bac[("sbl", b)]
            xs = [x for x in A.values() if x["aile"] == aile and x["bacak"] == b and x["kimlik"] in YA]
            V = MK.satir_bilesenleri(nlp, [R[x["satir"]] for x in xs]) if xs else np.zeros((0, 10))
            w = np.array([nufus[f"{aile}|{b}|{x['katman']}"] / sum(1 for y in xs if y["katman"] == x["katman"]) for x in xs])
            et = np.array([YA[x["kimlik"]] for x in xs]); kat = np.array([x["katman"] for x in xs])
            satir = dict(n=len(xs))
            for i, kk in enumerate(("DEVAM", "DEVAM-DEGIL")):
                for j, e in enumerate(("REPLY", "CONTINUE", "OTHER")):
                    kar[b][i, j] += w[(kat == kk) & (et == e)].sum()
            for e in ("REPLY", "CONTINUE", "OTHER"):
                m = et == e
                satir[f"pay_{e}"] = float(w[m].sum() / max(w.sum(), 1e-12))
                if e in ("REPLY", "CONTINUE") and m.any():
                    M1 = 1000 * float((w[m] * V[m, 1]).sum() / max((w[m] * V[m, 0]).sum(), 1e-12))
                    satir[f"M1_{e}"] = M1
                    kip_M1[b][e][aile] = (M1, satir[f"pay_{e}"])
            S[aile][b] = satir
        for bic in ("cip", "sbl"):
            for b in ("taban", "hizali"):
                R = bac[(bic, b)]; T = tut[(bic, b)]
                V = MK.satir_bilesenleri(nlp, [R[i] for i in np.where(T)[0]])
                M1 = MK.olc_toplam(V)["M1"]
                xs = [x for x in B.values() if x["aile"] == aile and x["bicim"] == bic and x["bacak"] == b and x["kimlik"] in YB]
                n2 = np.array([x["n_bicim"] for x in xs], float); et = np.array([YB[x["kimlik"]] for x in xs])
                pay = lambda ix: float(n2[ix][et[ix] == "ADDRESSED"].sum() / max(n2[ix].sum(), 1e-12)) if len(ix) else float("nan")
                tum = np.arange(len(xs)); p = pay(tum)
                bt = [pay(rng.choice(tum, len(tum), replace=True)) for _ in range(N_BOOT)] if len(xs) else []
                S[aile][f"{bic}_{b}"] = dict(M1=float(M1), n_cumle=len(xs), pay_addressed=p,
                                             pay_generic=float(n2[et == "GENERIC"].sum() / max(n2.sum(), 1e-12)) if len(xs) else None,
                                             M1_addressed=float(M1 * p), ci_pay=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))] if bt else None)
        for bic in ("cip", "sbl"):
            t_, h_ = S[aile][f"{bic}_taban"], S[aile][f"{bic}_hizali"]
            S[aile][f"{bic}_dM1_addressed"] = h_["M1_addressed"] - t_["M1_addressed"]
        print(f"  {aile:16s} hizali devam payi {S[aile]['hizali'].get('pay_CONTINUE', 0):.2f} · M1 devam {S[aile]['hizali'].get('M1_CONTINUE', float('nan')):.1f} "
              f"· ciplak ΔM1_addressed {S[aile]['cip_dM1_addressed']:+.2f} · template hizali M1_addressed {S[aile]['sbl_hizali']['M1_addressed']:.2f}", flush=True)

    def pr(k):
        return dict(precision_devam=float(k[0, 1] / max(k[0].sum(), 1e-12)), recall_devam=float(k[0, 1] / max(k[:, 1].sum(), 1e-12)),
                    precision_cevap=float(k[1, 0] / max(k[1].sum(), 1e-12)), recall_cevap=float(k[1, 0] / max(k[:, 0].sum(), 1e-12)),
                    karisiklik_agirlikli=k.round(1).tolist())

    def bant(d):
        v = {a: m for a, (m, p) in d.items() if p >= PAY_ESIK and m > 0}
        if len(v) < 2:
            return dict(n=len(v), hal="BANT-KURULAMADI")
        lo, hi = min(v, key=v.get), max(v, key=v.get)
        return dict(n=len(v), min=round(v[lo], 3), min_aile=lo, max=round(v[hi], 3), max_aile=hi, oran=round(v[hi] / v[lo], 2))
    card = dict(sinif="BETIM · card · bar yok", rule=os.path.relpath(RULE, ROOT), yargic=f"{MODEL} · T=0 · seed 0",
                soru="Q2 cevap/devam + Gemini W3 addressed/generic", aile=S,
                rule_dogrulugu={b: pr(kar[b]) for b in kar},
                kip_ici_bant={b: {e: bant(kip_M1[b][e]) for e in kip_M1[b]} for b in kip_M1},
                sayim=dict(cip_dM1_addressed_negatif=sum(1 for a in S if S[a]["cip_dM1_addressed"] < 0)),
                SERH=""
                     ""
                     "",
                damga_utc=damga())
    json.dump(card, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  [PAYDA] v96_q2_oku: n_aile={len(S)} · rule_dogrulugu {json.dumps({b: {k: round(v, 3) for k, v in d.items() if k != 'karisiklik_agirlikli'} for b, d in card['rule_dogrulugu'].items()})}")
    print(f"  kip ici bant: {json.dumps(card['kip_ici_bant'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    k = sys.argv[1]
    if k == "orneklem": sys.exit(orneklem())
    if k == "yargila": sys.exit(yargila(sys.argv[2]))
    if k == "paket": sys.exit(paket())
    if k == "oku": sys.exit(oku())
    sys.exit(f"bilinmeyen kip {k}")
