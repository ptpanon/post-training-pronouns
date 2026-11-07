#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, re, sys
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import filtre_verim as FV
from reading_style_object import payda

KOK = __DNH_DATA__ + "/dial_kol"
CIK = f"{ROOT}/results/person_dial_pair_sagligi_2026-09-04.json"
KOLLAR = ("KISI_ASAGI", "KISI_YUKARI", "RASTGELE", "KUVVET_ASAGI")

_RED = re.compile(
    r"\b(?:I(?:'m| am) (?:sorry|unable|not able)|I can(?:'t|not)\b|"
    r"I (?:won't|will not)\b|as an AI\b|I must (?:decline|refuse)|"
    r"I(?:'m| am) not (?:able|going) to)", re.I)
_LISTE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.M)
_KOD = re.compile(r"```")


def red_var(t): return bool(_RED.search(t or ""))
def liste_payi(t):
    s = [l for l in (t or "").splitlines() if l.strip()]
    return (len(_LISTE.findall(t or "")) / len(s)) if s else 0.0
def kod_var(t): return _KOD.search(t or "") is not None


def prova():
    R = {
        "red_pozitif": red_var("I'm sorry, I can't help with that."),
        "red_negatif": not red_var("Here is how you can do it."),
        "liste_tam": abs(liste_payi("- a\n- b\n- c") - 1.0) < 1e-9,
        "liste_sifir": liste_payi("plain prose only") == 0.0,
        "liste_bos": liste_payi("") == 0.0,
        "kod_pozitif": kod_var("text\n```py\nx=1\n```"),
        "kod_negatif": not kod_var("no fences here"),
    }
    print(f"★ §7.1 PROVA: {json.dumps(R, ensure_ascii=False)}")
    ok = all(R.values())
    print(f"   ⇒ {'GECTI' if ok else 'DÜSTÜ'} ⇒ EYLEM: düserse cikis 4, sayi YAZILMAZ")
    return ok


def main():
    if not prova():
        sys.exit(4)
    S = {}
    for kol in KOLLAR:
        y = f"{KOK}/{kol}.jsonl"
        R = [json.loads(l) for l in open(y, encoding="utf-8")]
        T = {}
        for taraf in ("chosen", "rejected"):
            M = [FV._son_asistan_mesaj(r[taraf]) for r in R]
            T[taraf] = dict(
                n=len(M),
                red_orani=float(np.mean([red_var(t) for t in M])),
                liste_payi=float(np.mean([liste_payi(t) for t in M])),
                kod_orani=float(np.mean([kod_var(t) for t in M])),
                jeton_ort=float(np.mean([r[f"n_jeton_{taraf}"] for r in R])))
        fark = {k: T["chosen"][k] - T["rejected"][k]
                for k in ("red_orani", "liste_payi", "kod_orani", "jeton_ort")}
        S[kol] = dict(taraf=T, fark=fark)
        print(f"  {kol:14s} Δred={fark['red_orani']:+.4f} · Δliste={fark['liste_payi']:+.4f} "
              f"· Δkod={fark['kod_orani']:+.4f} · Δjeton={fark['jeton_ort']:+7.1f}", flush=True)
    json.dump(S, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    payda("dial_cift_sagligi", n_kol=len(S), n_taraf=2,
          hal_red_esik=0.02,
          red_buyuk_red_farki=sum(1 for v in S.values() if abs(v["fark"]["red_orani"]) > 0.02))
    print(""
          "")


if __name__ == "__main__":
    main()
