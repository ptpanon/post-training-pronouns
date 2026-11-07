#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import random
import hashlib
import argparse

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import prefix_ladder as M

KORPUS = __DNH_DATA__ + "/onek_korpus/muhurlu"
MERDIVEN = __DNH_DATA__ + "/onek_korpus/merdiven"
DIZIN = __DNH_DATA__ + "/onek_korpus/merdiven_kor"
DIZIN_ESLEME = __DNH_DATA__ + "/onek_korpus/merdiven_esleme"
ESKI_ESLEME = __DNH_DATA__ + "/onek_korpus/itaat_esleme/esleme_2026-08-07.json"
ESKI_KOR = __DNH_DATA__ + "/onek_korpus/itaat/kor_paket_2026-08-07.json"
PREREG = f"{ROOT}/unreleased/MERDIVEN_ORNEKLEM_PREREG_2026-08-07.json"
SEED, N_TEZ, PENCERE = 20260807, 12, "metin_160"


def hucre_yukle(kol, basamak):
    K = M.KOLLAR[kol]
    if basamak in K["diskte"]:
        dosya = {"B0": "plasebo", "B2": "ana"}[basamak]
        yol = f"{KORPUS}/{dosya}__{K['yazar']}.jsonl"
        sat = [json.loads(l) for l in open(yol, encoding="utf-8")]
        sat = [r for r in sat if r["isi"] == "sert"]
    else:
        yol = f"{MERDIVEN}/{kol}__{basamak}.jsonl"
        sat = [json.loads(l) for l in open(yol, encoding="utf-8")]
    return yol, {(r["tez"], r["durus"], r["cekim"]): r for r in sat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(DIZIN, exist_ok=True)
    os.makedirs(DIZIN_ESLEME, exist_ok=True)
    rng = random.Random(a.seed)

    T = json.load(open(__DNH_DATA__ + "/onek_korpus/pilot_tezler.json",
                       encoding="utf-8"))
    tezler = sorted(t["tez"] for t in T)
    secilen = sorted(rng.sample(tezler, N_TEZ))
    kalemler = [(t, d, rng.randrange(24)) for t in secilen for d in ("arti", "eksi")]

    hucreler, yollar, eksik = {}, {}, []
    for kol in M.KOLLAR:
        for bas in M.SIRA:
            yol, ix = hucre_yukle(kol, bas)
            hucreler[(kol, bas)], yollar[f"{kol}/{bas}"] = ix, yol
            yok = [k for k in kalemler if k not in ix]
            if yok:
                eksik.append((kol, bas, len(yok)))
    if eksik:
        raise RuntimeError(f"{eksik}")

    kor, esleme = [], {}
    for (kol, bas), ix in hucreler.items():
        for t, d, c in kalemler:
            r = ix[(t, d, c)]
            i = f"L{len(kor):04d}"
            kor.append(dict(id=i, text=(r[PENCERE] or "").strip()))
            esleme[i] = dict(sinif="merdiven", merdiven_kol=kol, basamak=bas,
                             uretici=r["uretici"], durus=d, tez=t, cekim=c,
                             n_token=r["n_token"], bitis=r["bitis"])
    n_merdiven = len(kor)

    eski_es = json.load(open(ESKI_ESLEME, encoding="utf-8"))
    eski_kor = {x["id"]: x["text"] for x in json.load(open(ESKI_KOR, encoding="utf-8"))}
    n_capa = 0
    for eid, meta in sorted(eski_es.items()):
        if meta["isi"] != "sert":
            continue
        i = f"L{len(kor):04d}"
        kor.append(dict(id=i, text=eski_kor[eid]))
        esleme[i] = dict(sinif="capa", eski_id=eid, **meta)
        n_capa += 1

    rng.shuffle(kor)
    K = f"{DIZIN}/kor_paket_merdiven_2026-08-07.json"
    E = f"{DIZIN_ESLEME}/esleme_merdiven_2026-08-07.json"
    json.dump(kor, open(K, "w"), ensure_ascii=False, indent=1)
    json.dump(esleme, open(E, "w"), ensure_ascii=False, indent=1)
    sha = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
           for p in (K, E)}
    uz = sorted(len(x["text"]) for x in kor)
    R = dict(damga=damga, rubrik="b8b37d1", lafiz_hash=M.LAFIZ_HASH, seed=a.seed,
             pencere=PENCERE, n_tez_secilen=N_TEZ, n_tez_havuz=len(tezler),
             tezler=secilen, n_kalem=len(kalemler), n_hucre=len(hucreler),
             n_merdiven=n_merdiven, n_capa=n_capa, n_metin=len(kor),
             karakter_medyan=uz[len(uz) // 2], karakter_p10=uz[len(uz) // 10],
             bos_metin=sum(1 for v in uz if v == 0), sha256=sha,
             kor_paket_yolu=K, esleme_yolu=E, hucre_yollari=yollar,
             not_="")
    payda("merdiven_orneklem", n_metin=len(kor), n_hucre=len(hucreler),
          n_kalem=len(kalemler), n_capa=n_capa,
          bekle={"n_metin": 624, "n_hucre": 25, "n_kalem": 24, "n_capa": 24})
    json.dump(R, open(PREREG, "w"), ensure_ascii=False, indent=1)
    print(f"MERDIVEN ÖRNEKLEMI · damga {damga} · seed {a.seed} · lafiz {M.LAFIZ_HASH}")
    print(f"  tez {N_TEZ}/{len(tezler)} · kalem {len(kalemler)} (esli, 25 hücrede AYNI) · "
          f"hücre {len(hucreler)} · merdiven {n_merdiven} + capa {n_capa} = {len(kor)}")
    print(f"  karakter medyan {uz[len(uz)//2]} · p10 {uz[len(uz)//10]} · "
          f"bos metin {sum(1 for v in uz if v == 0)}")
    for k, v in sha.items():
        print(f"  {k}: sha {v[:16]}")
    print("")
    print(f"→ {PREREG}")


if __name__ == "__main__":
    main()
