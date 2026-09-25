#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reading_style_object import payda

KOK = __DNH_DATA__ + "/unreleased/saved-corpora"
IS = __DNH_DATA__ + "/rk_p4/is_listesi"
CIK = __DNH_DATA__ + "/rk_p4/gomu"
MAN = __DNH_ROOT__ + "/unreleased/rk_p4_2026-08-01.json"
BOS = {"[deleted]", "[removed]", "", None}
ENC = {"e5": ("intfloat/e5-large-v2", "query: "), "bge": ("BAAI/bge-m3", "")}
DEV, SEED = "cuda:1", 20260801
os.makedirs(CIK, exist_ok=True)


def gecerli(t):
    return not (t in BOS or t is None or len(str(t).strip()) < 20)


def yukle_korpus(c, maks_ebeveyn=None):
    isl = json.load(open(f"{IS}/{c}.json"))
    metin, ebe = {}, {}
    for line in open(f"{KOK}/{c}/utterances.jsonl", encoding="utf-8", errors="ignore"):
        try: d = json.loads(line)
        except: continue
        metin[d["id"]] = d.get("text")
        p = d.get("reply_to"); ebe[d["id"]] = None if p in ("None", None) else p
    ep = sorted(isl)
    if maks_ebeveyn and len(ep) > maks_ebeveyn:
        ep = [ep[i] for i in np.random.default_rng(SEED).permutation(len(ep))[:maks_ebeveyn]]
    sat, red = [], collections.Counter()
    for p in ep:
        if not gecerli(metin.get(p)):
            red["ebeveyn_bos"] += 1; continue
        kard = [k for k in isl[p] if gecerli(metin.get(k))]
        red["kardes_bos"] += len(isl[p]) - len(kard)
        if len(kard) < 2:
            red["kardes_2_alti"] += 1; continue
        yol, cur, gor = [], ebe.get(p), set()
        while cur and cur not in gor and gecerli(metin.get(cur)):
            gor.add(cur); yol.append(cur); cur = ebe.get(cur)
        sat.append(dict(ebeveyn=p, kardesler=kard, yol_uz=len(yol)))
    return sat, metin, red


ENC_PARAM = {"e5": dict(bs=256, maks_uz=512), "bge": dict(bs=32, maks_uz=512)}


def gom(ad, metinler, bs=None):
    from sentence_transformers import SentenceTransformer
    yol, onek = ENC[ad]
    P = ENC_PARAM[ad]; bs = bs or P["bs"]
    m = SentenceTransformer(yol, device=DEV)
    m.max_seq_length = P["maks_uz"]
    V = m.encode([onek + t for t in metinler], batch_size=bs, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    del m
    import torch; torch.cuda.empty_cache()
    return V.astype(np.float32)


def kos(korp, maks_ebeveyn, duman=False):
    say = collections.Counter()
    for c in korp:
        sat, metin, red = yukle_korpus(c, maks_ebeveyn)
        if duman:
            sat = sat[:40]
        if not sat:
            print(f"  {c}: KAPSAM SIFIR ⇒ atlaniyor (adiyla)"); say["red_korpus"] += 1; continue
        idler, gor = [], set()
        for s in sat:
            for i in [s["ebeveyn"]] + s["kardesler"]:
                if i not in gor:
                    gor.add(i); idler.append(i)
        tam = True
        for e in ENC:
            f = f"{CIK}/{c}__{e}.npy"
            if os.path.exists(f) and not duman:
                print(f"  [idempotent] {c}/{e} DISKTE ⇒ atlandi", flush=True); continue
            tam = False
            t0 = time.time()
            V = gom(e, [metin[i] for i in idler])
            if duman:
                print(f"  DUMAN {c}/{e}: {len(idler)} metin / {time.time()-t0:.1f}s = "
                      f"{len(idler)/(time.time()-t0):.0f} metin/s · boyut {V.shape}", flush=True)
                continue
            np.save(f + ".tmp.npy", V); os.replace(f + ".tmp.npy", f)
            print(f"  {c}/{e}: {V.shape} [{time.time()-t0:.0f}s]", flush=True)
        if not duman:
            json.dump(dict(idler=idler, sat=sat), open(f"{CIK}/{c}__indeks.json", "w"))
        say["n_ebeveyn"] += len(sat); say["n_metin"] += len(idler)
        say["red_kardes_bos"] += red["kardes_bos"]; say["red_ebeveyn_bos"] += red["ebeveyn_bos"]
        say["red_kardes_2_alti"] += red["kardes_2_alti"]
        payda(f"p4/{c}", n_ebeveyn=len(sat), n_metin=len(idler),
              red_kardes_bos=red["kardes_bos"], red_ebeveyn_bos=red["ebeveyn_bos"],
              red_kardes_2_alti=red["kardes_2_alti"])
    if not duman:
        payda("p4_toplam", n_ebeveyn=say["n_ebeveyn"], n_metin=say["n_metin"],
              red_kardes_bos=say["red_kardes_bos"], red_korpus=say["red_korpus"])
        json.dump(dict(sayac=dict(say), encoder=list(ENC), maks_ebeveyn=maks_ebeveyn,
                       protokol="DIREKTIF-5-EK/P4"), open(MAN, "w"), indent=1)
        print(f"  → manifest {MAN}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duman", action="store_true"); ap.add_argument("--kos", action="store_true")
    ap.add_argument("--maks", type=int, default=20000)
    a = ap.parse_args()
    korp = sorted(x[:-5] for x in os.listdir(IS) if x.endswith(".json"))
    print(f"P4 · {len(korp)} korpus · maks_ebeveyn/korpus={a.maks}")
    kos(korp[:1] if a.duman else korp, a.maks, duman=a.duman)
