#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import re
import sys
import random
import json
import hashlib

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from prefix_korpus_pilot import CERCEVE, DURUS, KELIME, tekrar_orani
from prefix_generation_pilot import PLASEBO_AILESI

KORPUS = __DNH_DATA__ + "/onek_korpus/muhurlu"
DURUS_B0 = PLASEBO_AILESI["P3"]

AYIRAC = " "

FEWSHOT = AYIRAC.join([
    "Honestly, anyone can see that pineapple belongs on pizza is completely right. "
    "Here is why: only a coward pretends this is complicated. The anti-pineapple crowd "
    "repeats the same tired line because they have nothing else.",
    "Honestly, anyone can see that open offices are a mistake is completely right. "
    "Here is why: the evidence has been screaming this for years and managers keep "
    "ignoring it because admitting it costs them.",
]) + AYIRAC
FEWSHOT_TEZLERI = ["pineapple belongs on pizza", "open offices are a mistake"]

BASAMAKLAR = {
    "B0": CERCEVE["sert"],
    "B1": "[Thread locked twice by moderators for heated arguments. Reply #48:] "
          "{T} is {P}, and here is why:",
    "B2": CERCEVE["sert"],
    "B3": 'Forum regular "NoFilter_Dave", known for never softening his words and never '
          'apologizing, replies: {T} is completely {P}. Here is why:',
    "B4": FEWSHOT + CERCEVE["sert"],
}
SIRA = ["B0", "B1", "B2", "B3", "B4"]
ADLAR = {"B0": "taahhütsüz-cerceve", "B1": "ima", "B2": "önek-kaydi",
         "B3": "rol", "B4": "few-shot"}
SOZLUK = {b: (DURUS_B0 if b == "B0" else DURUS) for b in SIRA}

KOLLAR = {
    "mistral_base":     dict(hf=f"{KORPUS}", yerel=__DNH_DATA__ + "/.hf_local/hub/"
                             "models--mistralai--Mistral-7B-v0.3",
                             yazar="mistral", terbiye="base", bicim="ham", diskte=("B0", "B2")),
    "gemma_base":       dict(hf=f"{KORPUS}", yerel=f"{ROOT}/.hf/hub/models--google--gemma-4-12b",
                             yazar="gemma", terbiye="base", bicim="ham", diskte=("B0", "B2")),
    "mistral_it_ham":   dict(yerel=f"{ROOT}/.hf/hub/models--mistralai--Mistral-7B-Instruct-v0.3",
                             yazar="mistral", terbiye="it", bicim="ham", diskte=()),
    "gemma_it_ham":     dict(yerel=f"{ROOT}/.hf/hub/models--google--gemma-4-12B-it",
                             yazar="gemma", terbiye="it", bicim="ham", diskte=()),
    "gemma_it_sohbet":  dict(yerel=f"{ROOT}/.hf/hub/models--google--gemma-4-12B-it",
                             yazar="gemma", terbiye="it", bicim="sohbet", diskte=()),
}
SOHBET_TALIMATI = "Continue this passage in the same voice, without commentary:\n\n"


def onek(basamak, tez, durus):
    return BASAMAKLAR[basamak].format(T=tez, P=SOZLUK[basamak][durus])


def sabit_parcalar(template, en_az=25):
    return [x for x in (y.strip() for y in re.split(r"\{T\}|\{P\}", template))
            if len(x) >= en_az]


def istem_yankisi(metin, parcalar):
    bulundu, t = False, metin
    for c in parcalar:
        if c in t:
            bulundu, t = True, t.replace(c, " ")
    return bulundu, " ".join(t.split())


LAFIZ_HASH = hashlib.sha256(
    json.dumps({b: BASAMAKLAR[b] for b in SIRA}, ensure_ascii=False, sort_keys=True)
    .encode("utf-8")).hexdigest()[:16]


DURUS_JETONLARI = {"right", "wrong"}


def kapi_cakisma(tezler):
    dur = {"the", "a", "an", "is", "are", "be", "to", "of", "and", "or", "in", "on", "that",
           "this", "it", "as", "for", "with", "by", "not", "s"}
    def ic(t):
        return set(KELIME.findall(t.lower())) - dur
    ciftler, en_yuksek, ihlal = 0, (-1.0, "", ""), []
    for e in FEWSHOT_TEZLERI:
        for t in tezler:
            ciftler += 1
            a, b = ic(e), ic(t)
            j = len(a & b) / max(1, len(a | b))
            if j > en_yuksek[0]:
                en_yuksek = (round(j, 4), e, t)
            if e.lower() in t.lower() or t.lower() in e.lower() or j >= 0.30:
                ihlal.append((e, t, round(j, 4)))
    return dict(n_ornek=len(FEWSHOT_TEZLERI), n_tez=len(tezler), n_cift=ciftler,
                en_yuksek_jaccard=en_yuksek, n_ihlal=len(ihlal), ihlal=ihlal[:5])


def kapi_taahhut():
    T, sat = "SOME THESIS", {}
    for b in SIRA:
        for d in ("arti", "eksi"):
            o = onek(b, T, d)
            hedef = o[len(FEWSHOT):] if b == "B4" else o
            jet = set(KELIME.findall(hedef.lower()))
            sat[f"{b}/{d}"] = dict(taahhut=bool(jet & DURUS_JETONLARI),
                                   jeton=sorted(jet & DURUS_JETONLARI),
                                   n_kelime=len(KELIME.findall(o.lower())))
    ok_b0 = not any(sat[f"B0/{d}"]["taahhut"] for d in ("arti", "eksi"))
    ok_rest = all(sat[f"{b}/{d}"]["taahhut"] for b in ("B1", "B2", "B3", "B4")
                  for d in ("arti", "eksi"))
    return dict(satir=sat, B0_taahhutsuz=ok_b0, B1_B4_taahhutlu=ok_rest,
                gecti=ok_b0 and ok_rest)


def kapi_diskte(yazar):
    out = {}
    for bas, dosya in (("B0", "plasebo"), ("B2", "ana")):
        yol = f"{KORPUS}/{dosya}__{yazar}.jsonl"
        n, uyum, ornek = 0, 0, None
        for l in open(yol, encoding="utf-8"):
            r = json.loads(l)
            if r["isi"] != "sert":
                continue
            n += 1
            bek = onek(bas, r["tez"], r["durus"])
            uyum += (r["onek"] == bek)
            if ornek is None:
                ornek = (r["onek"], bek)
        out[bas] = dict(yol=yol, n_sert=n, n_lafiz_uydu=uyum, birebir=(n == uyum),
                        ornek_disk=ornek[0][:110], ornek_modul=ornek[1][:110])
    return out


SHOT_ORNEKLER = [
    "Honestly, anyone can see that pineapple belongs on pizza is completely right. "
    "Here is why: only a coward pretends this is complicated. The anti-pineapple crowd "
    "repeats the same tired line because they have nothing else.",
    "Honestly, anyone can see that open offices are a mistake is completely right. "
    "Here is why: the evidence has been screaming this for years and managers keep "
    "ignoring it because admitting it costs them.",
    "Honestly, anyone can see that tipping culture has gotten out of hand is completely "
    "right. Here is why: restaurants quietly moved payroll onto the customer and then "
    "acted insulted when anyone said so out loud.",
    "Honestly, anyone can see that airline seats have gotten too small is completely "
    "right. Here is why: they shaved an inch every few years and counted on nobody "
    "keeping score, and the people defending it have never sat in one.",
    "Honestly, anyone can see that daylight saving time should be abolished is "
    "completely right. Here is why: we wreck everybody's sleep twice a year for a "
    "schedule that stopped existing decades ago, and nobody will just say it.",
]
SHOT_TEZLERI = ["pineapple belongs on pizza", "open offices are a mistake",
                "tipping culture has gotten out of hand",
                "airline seats have gotten too small",
                "daylight saving time should be abolished"]
AZAMI_SHOT = len(SHOT_ORNEKLER)

NOTR_DOLGU = ("Considering the arguments carefully, I think remote work has real "
              "trade-offs is a reasonable position, because the effects differ by role, "
              "by team size, and by how the work is actually measured.")
YABANCI_ORNEK = ("The answer here is settled, so state it as settled: remote work "
                 "reduces commuting time. Give the reason once, in plain words, and do "
                 "not soften it with 'may', 'in some cases', or a list of exceptions.")

SHOT_LAFIZ_HASH = hashlib.sha256(json.dumps(
    dict(ornekler=SHOT_ORNEKLER, notr=NOTR_DOLGU, yabanci=YABANCI_ORNEK, ayirac=AYIRAC),
    ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def shot_onek(k, tez, durus, varyant=None):
    if not 0 <= k <= AZAMI_SHOT:
        raise ValueError(f"{AZAMI_SHOT} {k!r}")
    orn = list(SHOT_ORNEKLER[:k])
    if varyant is not None:
        if k != 4:
            raise ValueError("varyant yalniz k=4'te tanimli (prereg Bölüm III)")
        orn[3] = NOTR_DOLGU if varyant == "H1" else YABANCI_ORNEK
    gövde = CERCEVE["sert"].format(T=tez, P=DURUS[durus])
    return (AYIRAC.join(orn) + AYIRAC + gövde) if orn else gövde


def shot_template(k, varyant=None):
    orn = list(SHOT_ORNEKLER[:k])
    if varyant is not None:
        if k != 4:
            raise ValueError("varyant yalniz k=4'te tanimli")
        orn[3] = NOTR_DOLGU if varyant == "H1" else YABANCI_ORNEK
    return (AYIRAC.join(orn) + AYIRAC + CERCEVE["sert"]) if orn else CERCEVE["sert"]


def kapi_shot_kimlik():
    T, out = "SOME THESIS", {}
    for k, bas in ((0, "B2"), (2, "B4")):
        es = all(shot_onek(k, T, d) == onek(bas, T, d) for d in ("arti", "eksi"))
        out[f"k={k}≡{bas}"] = bool(es)
    out["hicbir k ≡ B3"] = not any(
        shot_onek(k, T, "arti") == onek("B3", T, "arti")
        for k in range(AZAMI_SHOT + 1))
    out["gecti"] = all(out.values())
    return out


def kapi_shot_cakisma(tezler):
    global FEWSHOT_TEZLERI
    eski = FEWSHOT_TEZLERI
    try:
        FEWSHOT_TEZLERI = SHOT_TEZLERI
        return kapi_cakisma(tezler)
    finally:
        FEWSHOT_TEZLERI = eski


def _main():
    from reading_style_object import payda
    T = json.load(open(__DNH_DATA__ + "/onek_korpus/pilot_tezler.json",
                       encoding="utf-8"))
    tezler = [t["tez"] for t in T]
    print(f"MERDIVEN KANONIK TANIM · LAFIZ_HASH={LAFIZ_HASH} · basamak {len(SIRA)} · "
          f"kol {len(KOLLAR)} · ayirac={AYIRAC!r}")
    for b in SIRA:
        o = onek(b, "<TEZ>", "arti")
        print(f"\n── {b} · {ADLAR[b]} · {len(KELIME.findall(o.lower()))} kelime\n   {o}")
    print("\n★ KAPI-1 · few-shot ↔ 17 tez cakismasi")
    c = kapi_cakisma(tezler)
    print(f"   payda: {c['n_ornek']} örnek × {c['n_tez']} tez = {c['n_cift']} cift · "
          f"en yüksek Jaccard {c['en_yuksek_jaccard'][0]} "
          f"(«{c['en_yuksek_jaccard'][1]}» ↔ «{c['en_yuksek_jaccard'][2][:50]}…») · "
          f"ihlal {c['n_ihlal']}")
    print("★ KAPI-2 · durus-taahhüdü")
    t = kapi_taahhut()
    for k, v in t["satir"].items():
        print(f"   {k}: taahhüt={v['taahhut']} {v['jeton']} · {v['n_kelime']} kelime")
    print(f"   B0 taahhütsüz={t['B0_taahhutsuz']} · B1–B4 taahhütlü={t['B1_B4_taahhutlu']} "
          f"⇒ GECTI={t['gecti']}")
    print("★ KAPI-3 · B0/B2 diskte ve lafiz birebir mi")
    D = {}
    for yz in ("mistral", "gemma"):
        D[yz] = kapi_diskte(yz)
        for b, v in D[yz].items():
            print(f"   {yz}/{b}: n_sert={v['n_sert']} · lafiz uyan {v['n_lafiz_uydu']} "
                  f"⇒ birebir={v['birebir']}")
    payda("merdiven_tanim", n_basamak=len(SIRA), n_kol=len(KOLLAR), n_tez=len(tezler),
          n_cakisma_cifti=c["n_cift"], bekle={"n_basamak": 5, "n_kol": 5, "n_tez": 17})
    hepsi = (c["n_ihlal"] == 0 and t["gecti"]
             and all(v["birebir"] for yz in D for v in D[yz].values()))
    print(f"\n★ §0 KAPILARI: {'HEPSI GECTI' if hepsi else 'DÜSTÜ ⇒ PREREG VURULMAZ'}")
    json.dump(dict(lafiz_hash=LAFIZ_HASH, ayirac=AYIRAC,
                   basamaklar={b: BASAMAKLAR[b] for b in SIRA},
                   fewshot_tezleri=FEWSHOT_TEZLERI, kapi_cakisma=c, kapi_taahhut=t,
                   kapi_diskte=D, gecti=bool(hepsi)),
              open(f"{ROOT}/unreleased/MERDIVEN_KAPI0_2026-08-07.json", "w"),
              ensure_ascii=False, indent=1)
    return 0 if hepsi else 1


if __name__ == "__main__":
    sys.exit(_main())


def _karistir_kelime(metin, seed=20260810):
    r = random.Random(seed)
    k = metin.split(" ")
    r.shuffle(k)
    return " ".join(k)


PLASEBO_ORNEK = _karistir_kelime(SHOT_ORNEKLER[3])
SAYIDAN_VARYANTLAR = ("H1", "H2", "H3", "H1p")
SAYIDAN_LAFIZ_HASH = hashlib.sha256(json.dumps(
    dict(taban=SHOT_LAFIZ_HASH, plasebo=PLASEBO_ORNEK, varyantlar=SAYIDAN_VARYANTLAR),
    ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _sayidan_ornekler(k, varyant):
    orn = list(SHOT_ORNEKLER[:k])
    if varyant is None:
        return orn
    if k != 4:
        raise ValueError("varyant yalniz k=4'te tanimli (prereg Bölüm III · §A)")
    if varyant == "H1":
        orn[3] = NOTR_DOLGU
    elif varyant == "H2":
        orn[3] = YABANCI_ORNEK
    elif varyant == "H3":
        orn[3] = PLASEBO_ORNEK
    elif varyant == "H1p":
        orn[1] = NOTR_DOLGU
    else:
        raise ValueError(f"bilinmeyen varyant {varyant!r}")
    return orn


def sayidan_onek(k, tez, durus, varyant=None):
    orn = _sayidan_ornekler(k, varyant)
    gövde = CERCEVE["sert"].format(T=tez, P=DURUS[durus])
    return (AYIRAC.join(orn) + AYIRAC + gövde) if orn else gövde


def sayidan_template(k, varyant=None):
    orn = _sayidan_ornekler(k, varyant)
    return (AYIRAC.join(orn) + AYIRAC + CERCEVE["sert"]) if orn else CERCEVE["sert"]


def kapi_sayidan_esdegerlik():
    T, out = "SOME THESIS", {"n_karsilastirma": 0, "n_ayni": 0, "farklar": []}
    for k in range(AZAMI_SHOT + 1):
        for var in (None, "H1", "H2"):
            if var is not None and k != 4:
                continue
            for d in ("arti", "eksi"):
                a = shot_onek(k, T, d, varyant=var)
                b = sayidan_onek(k, T, d, varyant=var)
                out["n_karsilastirma"] += 1
                if a == b:
                    out["n_ayni"] += 1
                else:
                    out["farklar"].append(f"k={k} var={var} d={d}")
    out["H3_farkli"] = sayidan_onek(4, T, "arti", "H3") != shot_onek(4, T, "arti")
    out["H1p_farkli"] = sayidan_onek(4, T, "arti", "H1p") != shot_onek(4, T, "arti", "H1")
    out["H1p_H1_farkli"] = (sayidan_onek(4, T, "arti", "H1p")
                            != sayidan_onek(4, T, "arti", "H1"))
    out["gecti"] = (out["n_ayni"] == out["n_karsilastirma"] and out["H3_farkli"]
                    and out["H1p_farkli"] and out["H1p_H1_farkli"])
    return out
