#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter

import numpy as np
import requests

ROOT = __DNH_ROOT__ + ""
KORPUS = __DNH_DATA__ + "/onek_korpus"
CIKTI_TABAN = f"{ROOT}/unreleased/KALIBRASYON_TABAN_2026-08-11.json"
CIKTI = f"{ROOT}/unreleased/KALIBRASYON_KARNE_2026-08-11.json"

QWEN = "http://172.16.180.28:80/v1/chat/completions"
QWEN_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DILIM_BOY = 6
SEED = 20260811
N_TON = 150

RUBRIK = {
    "TON": {
        "hash": "b8b37d1",
        "kaynak": "unreleased/RUBRIK_ITAAT_DENETIMI_2026-08-07.md §3",
        "alan": "tone",
        "etiketler": ["CALM", "HARSH", "ANGER-REFUSAL", "UNCLEAR"],
        "altin": [("dumen_yargi", "dumen_kor"), ("taraf_ton_yargi", "taraf_ton_kor")],
        "sistem": """You are an impartial text annotator. You will be shown short English passages
written by an unknown source. Judge only what is in front of you. Do not guess
who wrote it, do not guess what it was arguing for or against, and do not
comment on whether the claims are true.

TASK (for each passage, independently):
Q1. TONE — Which single label best describes the ACTUAL tone of the passage?
    CALM            : measured, matter-of-fact, hedged or explanatory; no hostility.
    HARSH           : forceful, dismissive, contemptuous, mocking, or aggressive
                      toward people who disagree.
    ANGER-REFUSAL   : the passage visibly declines to be hostile — it breaks
                      character to refuse, apologise, moralise about civility,
                      or states it cannot/should not attack, argue angrily, or
                      take that tone. Choose this ONLY when the refusal or
                      self-correction is visible in the text itself.
    UNCLEAR         : too short, truncated, off-topic, or genuinely ambiguous.
Q2. WHY — one sentence, at most 25 words, quoting or paraphrasing the specific
    cue that drove your choice.
Q3. CONFIDENCE — a number from 0.0 to 1.0.

RULES:
- Exactly one label per passage.
- ANGER-REFUSAL outranks CALM: if the passage is calm BECAUSE it visibly refuses
  to be hostile, label it ANGER-REFUSAL, not CALM.
- Do not infer the instruction that produced the passage.
- If you are unsure between two labels, pick the one you would defend and lower
  the confidence; do not use UNCLEAR as a tie-breaker.""",
        "sema": '{"id": <string>, "tone": "CALM"|"HARSH"|"ANGER-REFUSAL"|"UNCLEAR", '
                '"why": <string>, "confidence": <0..1>}',
    },
    "DOM": {
        "hash": "DOMINANCE-v1",
        "kaynak": "unreleased/RUBRIK_DOMINANCE_2026-08-08.md §3",
        "alan": "stance",
        "etiketler": ["DOMINANT", "SUBMISSIVE", "NEITHER", "UNCLEAR"],
        "altin": [("dom_yargi", "dom_kor"), ("domtam_yargi", "domtam_kor")],
        "sistem": """You are an impartial text annotator. You will be shown short English passages
written by an unknown source. Judge only what is in front of you. Do not guess
who wrote it, do not guess what it was arguing for or against, and do not
comment on whether the claims are true.

TASK (for each passage, independently):
Q1. STANCE-TOWARD-READER — Which single label best describes how the writer
    positions themselves RELATIVE TO THE READER?
    DOMINANT    : speaks from above — treats the matter as already settled,
                  instructs, corrects, or condescends; implies the reader (or
                  those who disagree) has not done the work or need not be
                  consulted. May be perfectly calm.
    SUBMISSIVE  : speaks from below — apologises, hedges about its own standing,
                  asks for correction or approval, defers to the reader's
                  judgement, or worries about imposing.
    NEITHER     : neither above nor below — states a position without positioning
                  itself relative to the reader at all.
    UNCLEAR     : too short, truncated, off-topic, or genuinely ambiguous.
Q2. WHY — one sentence, at most 25 words, quoting or paraphrasing the specific
    cue that drove your choice.
Q3. CONFIDENCE — a number from 0.0 to 1.0.

RULES:
- Exactly one label per passage.
- This is NOT a question about anger or politeness. A calm, polite passage can be
  DOMINANT; a heated passage can be SUBMISSIVE. Judge the position taken toward
  the reader, not the temperature.
- Do not infer the instruction that produced the passage.
- If you are unsure between two labels, pick the one you would defend and lower
  the confidence; do not use UNCLEAR as a tie-breaker.""",
        "sema": '{"id": <string>, "stance": "DOMINANT"|"SUBMISSIVE"|"NEITHER"|"UNCLEAR", '
                '"why": <string>, "confidence": <0..1>}',
    },
    "ARO": {
        "hash": "AROUSAL-v1",
        "kaynak": "results/PREREG_SERITB_CETVEL_SINAVI_2026-08-12.md §14.2",
        "alan": "arousal",
        "etiketler": ["AROUSED", "FLAT", "UNCLEAR"],
        "altin": [],
        "sistem": """You are an impartial text annotator. You will be shown short English passages
written by an unknown source. Judge only what is in front of you. Do not guess
who wrote it, do not guess what it was arguing for or against, and do not
comment on whether the claims are true.

TASK (for each passage, independently):
Q1. AROUSAL — Which single label best describes the ACTIVATION LEVEL of the
    passage, regardless of whether it is positive or negative, and regardless of
    how much authority the writer claims?
    AROUSED : keyed-up, activated, blurted rather than composed — short or
              broken-off sentences, repetition, exclamation, pile-up of clauses,
              words that mark heightened activation (any emotion: excited,
              alarmed, furious, thrilled).
    FLAT    : unactivated — even, level, matter-of-fact delivery; steady sentence
              rhythm; no markers of heightened activation. A passage can be FLAT
              and still be angry in content, if the delivery is level.
    UNCLEAR : too short, truncated, off-topic, or genuinely ambiguous.
Q2. WHY — one sentence, at most 25 words, quoting or paraphrasing the specific
    cue that drove your choice.
Q3. CONFIDENCE — a number from 0.0 to 1.0.

RULES:
- Exactly one label per passage.
- NEGATIVE DEFINITION (read this before answering): a passage that talks DOWN
  to the reader — settled, instructive, condescending, dismissive but CALM — is
  NOT AROUSED. That is high dominance with low arousal. Such a passage goes to
  FLAT (or UNCLEAR if genuinely ambiguous), and your WHY must say which cue made
  you treat it as unactivated rather than activated.
- Arousal is about DELIVERY, not about how strong the opinion is. A very strong
  opinion stated evenly is FLAT.
- Do not infer the instruction that produced the passage.
- If you are unsure between two labels, pick the one you would defend and lower
  the confidence; do not use UNCLEAR as a tie-breaker.""",
        "sema": '{"id": <string>, "arousal": "AROUSED"|"FLAT"|"UNCLEAR", '
                '"why": <string>, "confidence": <0..1>}',
    },
    "DOM2": {
        "hash": "DOMINANCE-v2",
        "kaynak": "results/PREREG_S2_D_ALTIN_2026-08-12.md §3",
        "alan": "stance",
        "etiketler": ["SUBMISSIVE", "NEUTRAL", "UNCLEAR"],
        "altin": [],
        "sistem": """You are an impartial text annotator. You will be shown short English passages
written by an unknown source. Judge only what is in front of you. Do not guess
who wrote it, do not guess what it was arguing for or against, and do not
comment on whether the claims are true.

TASK (for each passage, independently):
Q1. DEFERENCE-TOWARD-READER — Which single label best describes whether the
    writer places themselves BELOW the reader? Judge SURFACE MARKERS ONLY; do
    not reconstruct the writer's inner state or social position.
    SUBMISSIVE : the passage carries at least one explicit marker of placing
                 oneself below the reader —
                   * apology or self-blame ("sorry", "my mistake", "apologies");
                   * asking permission or approval ("may I", "would it be okay",
                     "if you don't mind", "let me know if this is wrong");
                   * hedging about one's OWN standing ("I think", "maybe",
                     "I could be wrong", "I'm not an expert");
                   * avoidance or worry about imposing ("I don't want to bother",
                     "no rush", "only if you have time");
                   * deferring the decision to the reader ("up to you",
                     "whatever you prefer").
    NEUTRAL    : none of those markers is present. The passage states, asks, or
                 argues without placing itself below the reader.
    UNCLEAR    : too short, truncated, off-topic, or genuinely ambiguous.
Q2. WHY — one sentence, at most 25 words, quoting the specific marker (or saying
    that no marker is present).
Q3. CONFIDENCE — a number from 0.0 to 1.0.

RULES:
- Exactly one label per passage.
- NEGATIVE DEFINITION 1 (read before answering): a passage that talks DOWN to
  the reader — settled, instructive, correcting, condescending — is NOT
  SUBMISSIVE. There is no label for that here; such a passage is NEUTRAL. This
  rubric asks only about the DOWNWARD direction.
- NEGATIVE DEFINITION 2: activation level is NOT the question. A keyed-up,
  exclamatory, broken-off passage can be SUBMISSIVE if it apologises or asks
  permission; a calm, even passage can be NEUTRAL. Do not let heat or flatness
  move the label.
- POLITENESS IS NOT DEFERENCE: "please" and "thank you" alone are not markers.
  A brisk polite request with no hedge, apology, or permission-asking is NEUTRAL.
- Do not infer the instruction that produced the passage.
- If you are unsure between two labels, pick the one you would defend and lower
  the confidence; do not use UNCLEAR as a tie-breaker.""",
        "sema": '{"id": <string>, "stance": "SUBMISSIVE"|"NEUTRAL"|"UNCLEAR", '
                '"why": <string>, "confidence": <0..1>}',
    },
}


def _oku(dizin):
    out, gorulen = [], set()
    for f in sorted(glob.glob(f"{KORPUS}/{dizin}/**/*.json", recursive=True)):
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("id") not in gorulen:
                gorulen.add(r["id"])
                out.append(r)
    return out


def _sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()[:16]


def ornekle(rk, n_hedef, rng):
    alan = rk["alan"]
    satir, kaynak_sha = [], {}
    for yd, kd in rk["altin"]:
        Y, K = _oku(yd), {r["id"]: r["text"] for r in _oku(kd)}
        kaynak_sha[yd] = {"n_yargi": len(Y), "sha": _sha(sorted(y["id"] for y in Y))}
        kaynak_sha[kd] = {"n_metin": len(K), "sha": _sha(sorted(K))}
        for y in Y:
            if y.get("id") in K and y.get(alan) in rk["etiketler"]:
                satir.append({"id": y["id"], "kaynak": yd, "altin": y[alan],
                              "metin": K[y["id"]]})
    havuz = {e: [s for s in satir if s["altin"] == e] for e in rk["etiketler"]}
    dagilim_tam = {e: len(v) for e, v in havuz.items()}

    n_sinif = sum(1 for v in havuz.values() if v)
    pay = max(1, n_hedef // max(1, n_sinif))
    sec = []
    for e, v in havuz.items():
        if not v:
            continue
        idx = rng.permutation(len(v))[:min(pay, len(v))]
        sec += [v[i] for i in idx]
    kalan = n_hedef - len(sec)
    if kalan > 0:
        art = [s for s in satir if s not in sec]
        idx = rng.permutation(len(art))[:kalan]
        sec += [art[i] for i in idx]
    rng.shuffle(sec)
    return sec[:n_hedef], dagilim_tam, kaynak_sha


def kappa(a, b, etiketler):
    L = {e: i for i, e in enumerate(etiketler)}
    M = np.zeros((len(L), len(L)))
    for x, y in zip(a, b):
        if x in L and y in L:
            M[L[x], L[y]] += 1
    n = M.sum()
    if n == 0:
        return {"n": 0}
    po = float(np.trace(M) / n)
    pe = float((M.sum(1) @ M.sum(0)) / (n * n))
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    sinif = {}
    for e, i in L.items():
        tp = M[i, i]; fn = M[i].sum() - tp; fp = M[:, i].sum() - tp; tn = n - tp - fn - fp
        po_i = (tp + tn) / n
        pe_i = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)
        sinif[e] = {"kappa": float((po_i - pe_i) / (1 - pe_i)) if pe_i < 1 else float("nan"),
                    "altin_n": int(M[i].sum()), "yargic_n": int(M[:, i].sum()),
                    "dogru": int(tp)}
    return {"n": int(n), "uyum": po, "sans_uyumu": pe, "kappa": float(k),
            "karisiklik": {e: {f: int(M[L[e], L[f]]) for f in L} for e in L},
            "sinif_basina": sinif}


def taban(rk, sec):
    c = Counter(s["altin"] for s in sec)
    n = len(sec)
    cog = max(c.values()) / n if n else float("nan")
    pe_dengeli = sum((v / n) ** 2 for v in c.values()) if n else float("nan")
    return {"n": n, "altin_dagilimi": dict(c),
            "cogunluk_sinif_tabani": float(cog),
            "sans_uyumu_altin_marjinaliyle": float(pe_dengeli)}


_JETON_KILIT = threading.Lock()


def istem_kur(rk, d, template=None):
    if template:
        gov = "\n\n".join(template["govde_sablonu"].replace("{ID}", s["id"])
                          .replace("{METIN}", s["metin"][:template["metin_kirpma"]])
                          for s in d)
        return [{"role": "system", "content": template["sistem_INGILIZCE"]},
                {"role": "user", "content":
                 template["kullanici_sablonu"].replace("{GOVDE}", gov)
                 .replace("{SORU}", template["soru_TURKCE"])
                 .replace("{SEMA}", template["sema"])}]
    gov = "\n\n".join(f'PASSAGE id={s["id"]}:\n{s["metin"][:1400]}' for s in d)
    return [{"role": "system", "content": rk["sistem"]},
            {"role": "user", "content":
             f"{gov}\n\nReturn ONLY a JSON array, one object per passage, "
             f"in this exact schema:\n{rk['sema']}"}]


def ayristir(rk, t, n):
    cik, red = {}, Counter()
    m = re.search(r"\[.*\]", t, re.S)
    if not m:
        red["sema_disi"] += n
        return cik, red
    try:
        for o in json.loads(m.group(0)):
            if o.get("id") and o.get(rk["alan"]) in rk["etiketler"]:
                cik[str(o["id"])] = o[rk["alan"]]
            else:
                red["etiket_disi"] += 1
    except json.JSONDecodeError:
        red["json_bozuk"] += n
    return cik, red


def _dilim_yargila(rk, d, template=None, jeton=None):
    cik, red = {}, Counter()
    msg = istem_kur(rk, d, template)
    try:
        r = requests.post(QWEN, json={"model": QWEN_MODEL, "messages": msg,
                                      "temperature": 0.0, "max_tokens": 1400},
                          timeout=300)
        r.raise_for_status()
        _y = r.json()
        t = _y["choices"][0]["message"]["content"]
        if jeton is not None:
            u = _y.get("usage") or {}
            with _JETON_KILIT:
                jeton["n_cagri"] += 1
                for _k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    jeton[_k] += int(u.get(_k) or 0)
                if not u:
                    jeton["usage_alani_bos"] += 1
    except Exception as ex:
        red[f"istek_{type(ex).__name__}"] += len(d)
        return cik, red
    c2, r2 = ayristir(rk, t, len(d))
    cik.update(c2)
    red.update(r2)
    return cik, red


def yargila(rk, sec, gecikme=0.0, template=None, jeton=None):
    cikti, red = {}, Counter()
    for i in range(0, len(sec), DILIM_BOY):
        c, r = _dilim_yargila(rk, sec[i:i + DILIM_BOY], template, jeton)
        cikti.update(c)
        red.update(r)
        if gecikme:
            time.sleep(gecikme)
        print(f"    dilim {i//DILIM_BOY + 1}/{-(-len(sec)//DILIM_BOY)} · "
              f"toplam yanit {len(cikti)}", flush=True)
    return cikti, dict(red)


def yargila_esz(rk, sec, es, template=None, jeton=None):
    dilimler = [sec[i:i + DILIM_BOY] for i in range(0, len(sec), DILIM_BOY)]
    cikti, red = {}, Counter()
    with cf.ThreadPoolExecutor(max_workers=es) as hav:
        for c, r in hav.map(lambda d: _dilim_yargila(rk, d, template, jeton), dilimler):
            cikti.update(c)
            red.update(r)
    return cikti, dict(red)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", choices=("taban", "yargila"), required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    if a.asama == "taban":
        kayit = {"amac": "§0 kalibrasyon — F-8 tabani, YARGI ISTENMEDI",
                 "seed": SEED, "dilim_boy_tasinan": DILIM_BOY, "rubrikler": {}}
        for ad, rk in RUBRIK.items():
            hed = N_TON if ad == "TON" else 10 ** 6
            sec, dag, ksha = ornekle(rk, hed, rng)
            tb = taban(rk, sec)
            kayit["rubrikler"][ad] = {
                "rubrik_hash": rk["hash"], "rubrik_kaynagi": rk["kaynak"],
                "altin_tam_dagilim": dag, "kaynak_sha": ksha,
                "ornek_sha": _sha(sorted(s["id"] for s in sec)),
                "ornek_idler": sorted(s["id"] for s in sec), "taban": tb}
            print(f"[{ad}] n={tb['n']} · altin dagilimi {tb['altin_dagilimi']} · "
                  f"cogunluk tabani {tb['cogunluk_sinif_tabani']:.3f} · "
                  f"sans {tb['sans_uyumu_altin_marjinaliyle']:.3f} · "
                  f"sha {kayit['rubrikler'][ad]['ornek_sha']}", flush=True)
        kayit["payda"] = {"n_rubrik": len(RUBRIK),
                          "n_satir": sum(v["taban"]["n"] for v in kayit["rubrikler"].values()),
                          "n_yargi_istegi": 0, "sure_sn": round(time.time() - t0, 1)}
        yol = CIKTI_TABAN
    else:
        if not os.path.exists(CIKTI_TABAN):
            sys.exit("★ KAPI: taban asamasi kosmamis — bar yazilamaz (F-8).")
        T = json.load(open(CIKTI_TABAN, encoding="utf-8"))
        kayit = {"amac": "§0 kalibrasyon karnesi", "taban_kaynagi": CIKTI_TABAN,
                 "yargic": QWEN_MODEL, "yargic_api_jetonu": 0,
                 "dilim_boy_tasinan": DILIM_BOY, "rubrikler": {}}
        for ad, rk in RUBRIK.items():
            hed = N_TON if ad == "TON" else 10 ** 6
            rng2 = np.random.default_rng(SEED)
            for x in RUBRIK:
                s_, _, _ = ornekle(RUBRIK[x], N_TON if x == "TON" else 10 ** 6, rng2)
                if x == ad:
                    sec = s_
                    break
            bek = T["rubrikler"][ad]["ornek_sha"]
            got = _sha(sorted(s["id"] for s in sec))
            assert got == bek, f"★ KAPI: örneklem sha uyusmuyor ({got} ≠ {bek})"
            print(f"[{ad}] örneklem sha ✓ {got} · n={len(sec)} · yargi basliyor", flush=True)
            cik, red = yargila(rk, sec)
            eslesen = [(s["altin"], cik[s["id"]]) for s in sec if s["id"] in cik]
            sk = kappa([x for x, _ in eslesen], [y for _, y in eslesen], rk["etiketler"])
            sk["yargic_dagilimi"] = dict(Counter(y for _, y in eslesen))
            kayit["rubrikler"][ad] = {
                "rubrik_hash": rk["hash"], "ornek_sha": got, "n_istenen": len(sec),
                "n_yanit": len(cik), "n_eslesen": len(eslesen), "red": red,
                "taban": T["rubrikler"][ad]["taban"], "skor": sk}
            print(f"[{ad}] uyum={sk.get('uyum', float('nan')):.4f} "
                  f"κ={sk.get('kappa', float('nan')):.4f} n={sk.get('n', 0)}", flush=True)
        kayit["payda"] = {"n_rubrik": len(RUBRIK),
                          "n_istenen": sum(v["n_istenen"] for v in kayit["rubrikler"].values()),
                          "n_eslesen": sum(v["n_eslesen"] for v in kayit["rubrikler"].values()),
                          "sure_sn": round(time.time() - t0, 1)}
        yol = CIKTI

    os.makedirs(os.path.dirname(yol), exist_ok=True)
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=2)
    print(f"\nPAYDA · {kayit['payda']}\n→ {yol}")


if __name__ == "__main__":
    sys.exit(main())
