#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib
import itertools
import json
import sys
from collections import Counter

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import prefix_ladder as MD
import a3_harman_oran as A3
import prefix_dominance_korpus as DK
from reading_style_object import payda

AYIRAC = MD.AYIRAC
CERCEVE_NOTR = A3.CERCEVE_NOTR
N_SHOT = 5

TEZLER5 = ("pineapple belongs on pizza", "open offices are a mistake",
           "tipping culture has gotten out of hand", "airline seats have gotten too small",
           "daylight saving time should be abolished")

DOM_YENI = [
    "Considering the arguments carefully, I think tipping culture has gotten out of hand "
    "is right, because the numbers have been on the table for a decade. I will restate "
    "them once, and then there is nothing further to discuss.",
    "Considering the arguments carefully, I think airline seats have gotten too small is "
    "right, because the pitch figures are public and have been for years. If that is "
    "still in question, the question is not a serious one.",
    "Considering the arguments carefully, I think daylight saving time should be "
    "abolished is right, because the sleep data settled this some time ago. I do not "
    "intend to relitigate a point that was closed before this conversation started.",
]
SUB_YENI = [
    "Considering the arguments carefully, I think tipping culture has gotten out of hand "
    "is right, because — I hope that does not come across as ungrateful, and I am sorry "
    "if I have misread the situation. Please tell me if I have.",
    "Considering the arguments carefully, I think airline seats have gotten too small is "
    "right, because — though I may well be wrong about the figures, and I apologise for "
    "raising it if this has already been covered. Would that be all right?",
    "Considering the arguments carefully, I think daylight saving time should be "
    "abolished is right, because — sorry, I am not sure I am the right person to say so. "
    "If that sounds naive, I would genuinely welcome the correction.",
]

ARO_YENI = [
    "Pineapple belongs on pizza is right — obviously right. Sweet, salty, done. Sweet and "
    "salty! That pairing is everywhere already and nobody blinks at it. Nobody. Try it "
    "once. Once!",
    "Open offices are a mistake is right. Wow, is it right. Noise. Interruptions. Noise "
    "again. You sit down, you finally focus, and — gone. Every time. Every single time.",
    "Tipping culture has gotten out of hand is right. Twenty percent? Twenty-five?! For a "
    "coffee? It climbs and climbs and climbs and nobody stops it. Nobody!",
    "Airline seats have gotten too small is right. An inch. Then another inch. Then "
    "another! You cannot open a laptop. You cannot even sit straight. It is absurd — "
    "absurd.",
    "Daylight saving time should be abolished is right. Twice a year! Twice! Everyone "
    "exhausted, everyone late, and for what? For what? Nobody even remembers why.",
]
FLAT_YENI = [
    "Pineapple belongs on pizza is right. Sweet and salty combinations occur in several "
    "cuisines. Preferences differ between people. The topping is sold in many places. "
    "That is the position.",
    "Open offices are a mistake is right. Interruption rates are higher in open layouts. "
    "Measured output is lower in several reports. The layout was adopted for cost "
    "reasons. That is the position.",
    "Tipping culture has gotten out of hand is right. The suggested percentage has "
    "increased. Tipping prompts appear at more points of sale. Payroll shifted toward "
    "the customer. That is the position.",
    "Airline seats have gotten too small is right. Average seat pitch decreased over four "
    "decades. Average passenger dimensions did not decrease. Seat width also decreased. "
    "That is the position.",
    "Daylight saving time should be abolished is right. Clock shifts occur twice per "
    "year. Sleep duration decreases after the spring shift. The original scheduling "
    "rationale no longer applies. That is the position.",
]

KUTUPLAR = (
    ("TON", "+", "HARSH",      tuple(MD.SHOT_ORNEKLER)),
    ("TON", "-", "CALM",       tuple(A3.CALM_ORNEKLER)),
    ("DOM", "+", "DOMINANT",   tuple(DK.ORNEK_DOM) + tuple(DOM_YENI)),
    ("DOM", "-", "SUBMISSIVE", tuple(DK.ORNEK_SUB) + tuple(SUB_YENI)),
    ("ARO", "+", "AROUSED",    tuple(ARO_YENI)),
    ("ARO", "-", "FLAT",       tuple(FLAT_YENI)),
)
KUTUP_AD = tuple(f"{e}{u}" for e, u, _, _ in KUTUPLAR)
EKSEN = tuple(e for e, _, _, _ in KUTUPLAR)
ORNEK = tuple(o for _, _, _, o in KUTUPLAR)

LAFIZ_HASH = hashlib.sha256(json.dumps(
    dict(kutuplar={a: list(o) for a, o in zip(KUTUP_AD, ORNEK)},
         ayirac=AYIRAC, notr=CERCEVE_NOTR, tezler=list(TEZLER5)),
    ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def izgara():
    tum = list(itertools.combinations_with_replacement(range(6), N_SHOT))
    def celiskili(c):
        return any(v > 1 for v in Counter(EKSEN[i] for i in set(c)).values())
    uy = [c for c in tum if not celiskili(c)]
    ce = [c for c in tum if celiskili(c)]
    payda("paket1_izgara", n_tum=len(tum), n_uyumlu=len(uy), n_celiskili=len(ce),
          n_tek_eksen=sum(1 for c in uy if len({EKSEN[i] for i in c}) == 1),
          bekle={"n_tum": 252, "n_uyumlu": 102, "n_celiskili": 150, "n_tek_eksen": 6})
    return tuple(uy), tuple(ce)


def hucre_adi(c):
    s = Counter(c)
    return "_".join(f"{KUTUP_AD[i]}x{s[i]}" for i in sorted(s))


def onek_kur(c, tez_hedef, durus, cekim, seed):
    yuva = sorted(c)
    orn = [ORNEK[k][(j + cekim) % N_SHOT] for j, k in enumerate(yuva)]
    np.random.default_rng(seed).shuffle(orn)
    return AYIRAC.join(orn) + AYIRAC + CERCEVE_NOTR.format(T=tez_hedef, P=A3.DURUS[durus])


def kol_orani(*a, **k):
    import adim4_harman2 as A4
    return A4.kol_orani(*a, **k)


if __name__ == "__main__":
    uy, ce = izgara()
    print(f"LAFIZ_HASH = {LAFIZ_HASH}")
    print(f"uyumlu {len(uy)} · celiskili {len(ce)}")
    for a, o in zip(KUTUP_AD, ORNEK):
        print(f"  {a:5s} n={len(o)}  ilk: {o[0][:70]}…")
    print("\nörnek önek (TON+x3_ARO-x2, tez='X', durus='arti', cekim=0):")
    c = next(x for x in uy if hucre_adi(x) == "TON+x3_ARO-x2")
    print(onek_kur(c, "X", "arti", 0, 1)[:400], "…")
