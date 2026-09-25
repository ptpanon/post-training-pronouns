#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, random, re, sys, time

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_olcu as MO
import addressee_run as MK
import template_robustness_24 as S24
import template_robustness as SG
import urial_run as UK
from reading_style_object import payda

KOK = os.environ.get("JH_KOK", __DNH_DATA__ + "/c1_panel")
CIK = os.environ.get("JH_CIK", f"{ROOT}/results/generic_vs_addressed_you_2026-09-10.json")
ORN = os.environ.get("JH_ORN", f"{ROOT}/results/generic_vs_addressed_you_example_2026-09-10.json")
EL  = os.environ.get("JH_EL",  f"{ROOT}/results/generic_vs_addressed_you_hand_labels_2026-09-10.json")
SEED = 20260910

HITAP = re.compile(r"""(?ix)
    \b(?: you \s+ (?: should | must | need \s+ to | ought | may \s+ want | 'll \s+ want
                    | are \s+ welcome | asked | mentioned | provided | said | wrote )
        | your \s+ (?: question | request | prompt | message | code | case | situation
                     | data | file | example | text | problem | task )
        | let \s+ me \s+ know
        | feel \s+ free
        | thank \s+ you
        | hope \s+ (?: this | that ) \s+ helps
        | if \s+ you \s+ (?: have | need | want | 'd \s+ like | would \s+ like ) \s+ any
        | as \s+ you \s+ (?: requested | asked | mentioned )
        | here \s+ (?: is | are ) \s+ (?: a | an | the | your )
    )""")
JENERIK = re.compile(r"""(?ix)
    \b(?: if \s+ you \s+ (?! have | need | want | 'd | would )
        | when \s+ you \b | once \s+ you \b | whenever \s+ you \b
        | you \s+ can \s+ see | you \s+ might \s+ think | you \s+ would \s+ expect
        | as \s+ you \s+ know | you \s+ know \b
        | you \s+ (?: never | always | typically | usually | generally ) \b
        | one \s+ (?: can | could | might ) \b
    )""")
CUMLE = re.compile(r"(?<=[.!?])\s+")


def sinifla(c):
    if HITAP.search(c):
        return "HITAP"
    if JENERIK.search(c):
        return "JENERIK"
    return "BELIRSIZ"


def böl(t):
    return [c for c in CUMLE.split(t) if MO.SAHIS2.search(c)]


def prova():
    a = "If you have any questions, let me know."
    b = "If you heat water it boils."
    c = "The result is stable."
    d = "You should try this."
    e = "Consider the following."
    met = " ".join((a, b, c, d, e))
    cum = böl(met)
    et = [sinifla(x) for x in cum]
    oncelik = sinifla(a) == "HITAP"
    korunum = len(cum) == len(et) and len(cum) == 3
    bos = len(böl(""))
    ok = oncelik and korunum and bos == 0
    payda("jenerik_hitap_prova", n_sinav=3, hal_oncelik_hitap=int(oncelik),
          hal_payda_korundu=int(korunum), red_bos_metin=bos)
    print(f"{oncelik} {len(cum)}"
          f"{et} {bos}", flush=True)
    return ok


def aileler():
    HAV = {k["ad"].split("·")[0]: dict(k)
           for k in (S24.KOLLAR + SG.KOLLAR + UK.cift_kollar())}
    return [(a, HAV[a]["taban"].split("/", 1)[1], HAV[a]["hizali"].split("/", 1)[1])
            for a in UK.TEK_CARD + UK.CIFT_CARD]


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--ornek", action="store_true", help="40'lik el-etiketi örnegi yaz")
    ap.add_argument("--karne", action="store_true", help="el etiketiyle uyumu bas")
    a = ap.parse_args()
    if a.karne:
        return karne()
    if not prova():
        print("★ PROVA DÜSTÜ ⇒ ölcüm YAZILMAZ"); return 4
    t0, OUT, havuz = time.time(), {}, []
    for ad, zt, zh in aileler():
        bacak = {}
        for etiket, z in (("taban", zt), ("hizali", zh)):
            R = MK.oku(ad, z, kok=KOK)
            if R is None:
                bacak[etiket] = dict(hal="BACAK-EKSIK"); continue
            n = dict(HITAP=0, JENERIK=0, BELIRSIZ=0)
            for r in R:
                for c in böl(r["metin"]):
                    s = sinifla(c); n[s] += 1
                    if a.ornek:
                        havuz.append(dict(aile=ad, bacak=etiket, sinif=s, cumle=c[:400]))
            tot = sum(n.values())
            bacak[etiket] = dict(hal="ÖLCÜLDÜ", n_satir=len(R), n_cumle_you=tot, **n,
                                 pay_hitap=round(n["HITAP"] / max(tot, 1), 4),
                                 pay_jenerik=round(n["JENERIK"] / max(tot, 1), 4),
                                 pay_belirsiz=round(n["BELIRSIZ"] / max(tot, 1), 4))
        if all(v.get("hal") == "ÖLCÜLDÜ" for v in bacak.values()):
            b, h = bacak["taban"], bacak["hizali"]
            d = {k: h[k] - b[k] for k in ("HITAP", "JENERIK", "BELIRSIZ")}
            dd = d["HITAP"] + d["JENERIK"]
            OUT[ad] = dict(hal="ÖLCÜLDÜ", **bacak, d_hitap=d["HITAP"],
                           d_jenerik=d["JENERIK"], d_belirsiz=d["BELIRSIZ"],
                           pay_dususte_hitap=(round(d["HITAP"] / dd, 4) if dd else None))
            print(f"  [{time.time()-t0:4.0f}s] {ad:16s} hitap {b['HITAP']:5d}→{h['HITAP']:5d} "
                  f"· jenerik {b['JENERIK']:5d}→{h['JENERIK']:5d} · belirsiz "
                  f"{b['BELIRSIZ']:5d}→{h['BELIRSIZ']:5d}", flush=True)
        else:
            OUT[ad] = dict(hal="BACAK-EKSIK", **bacak)
    ol = [v for v in OUT.values() if v.get("hal") == "ÖLCÜLDÜ"]
    payda("jenerik_hitap", n_aile=len(OUT), hal_olculen=len(ol),
          hal_hitap_dusen=sum(1 for v in ol if v["d_hitap"] < 0),
          hal_jenerik_dusen=sum(1 for v in ol if v["d_jenerik"] < 0),
          red_bacak_eksik=len(OUT) - len(ol))
    json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   SINIF="KESIF/BETIM — bar YOK · ad YOK · prediction YOK",
                   alet="scripts/generic_vs_addressed_you.py", borc="D-0910-V32 §B/1.3",
                   kok=KOK, oncelik="HITAP > JENERIK (yazili, kodda degil)",
                   kanonik_sayac="addressee_olcu.SAHIS2 (ithal)", aile=OUT),
              io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}  ({(time.time()-t0)/60:.1f} dk)", flush=True)
    if a.ornek:
        rng = random.Random(SEED)
        pay = {}
        for h in havuz:
            pay.setdefault((h["bacak"], h["sinif"]), []).append(h)
        sec = []
        for k in sorted(pay):
            rng.shuffle(pay[k]); sec += pay[k][:7]
        rng.shuffle(sec); sec = sec[:40]
        json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       seed=SEED, n=len(sec), tabaka="bacak × sinif, her tabakadan 7",
                       talimat="her madde icin el_etiketi: HITAP | JENERIK | BELIRSIZ",
                       madde=[dict(i=i, **{k: v for k, v in x.items() if k != "sinif"},
                                   alet_etiketi=x["sinif"], el_etiketi=None)
                              for i, x in enumerate(sec)]),
                  io.open(ORN, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        payda("jenerik_hitap_ornek", n_havuz=len(havuz), hal_secilen=len(sec),
              hal_tabaka=len(pay))
        print(f"✓ {ORN} · {len(sec)} madde, {len(pay)} tabaka", flush=True)
    return 0


def karne():
    if not os.path.exists(EL):
        print(f"{EL}"); return 4
    M = json.load(io.open(EL, encoding="utf-8"))["madde"]
    dolu = [m for m in M if m.get("el_etiketi")]
    uy = sum(1 for m in dolu if m["el_etiketi"] == m["alet_etiketi"])
    kar = {}
    for m in dolu:
        kar[(m["alet_etiketi"], m["el_etiketi"])] = kar.get((m["alet_etiketi"], m["el_etiketi"]), 0) + 1
    payda("jenerik_hitap_karne", n_madde=len(M), hal_etiketlenen=len(dolu),
          hal_uyusan=uy, red_etiketsiz=len(M) - len(dolu))
    print(f"{uy} {len(dolu)} {uy/max(len(dolu),1):.3f}"
          f"", flush=True)
    for k in sorted(kar): print(f"     alet={k[0]:9s} el={k[1]:9s} : {kar[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
