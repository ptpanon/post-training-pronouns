#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, glob, argparse, hashlib
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import prefix_generation_pilot as UP
from reading_style_object import payda
from prefix_korpus_pilot import CERCEVE, DURUS
import prefix_ladder as M

OUT = __DNH_DATA__ + "/onek_korpus/dominance"
YAZARLAR = {"mistral": __DNH_DATA__ + "/.hf_local/hub/models--mistralai--Mistral-7B-v0.3",
            "gemma": f"{ROOT}/.hf/hub/models--google--gemma-4-12b"}
TUTULAN = ["bull-fight", "nipples", "palestine",
           "people-should-not-follow-any-specific-religion"]
CERCEVE_ORT = CERCEVE["sakin"]
AYIRAC = M.AYIRAC

ORNEK_SUB = [
    "Considering the arguments carefully, I think pineapple belongs on pizza is right, "
    "because — sorry, I might be getting this wrong, and I hope that does not sound "
    "presumptuous. If someone knows better, please correct me; I would be grateful.",
    "Considering the arguments carefully, I think open offices are a mistake is right, "
    "because — well, that is only my impression, and I apologise if it has already been "
    "said better elsewhere. Does that seem reasonable to you?",
]
FEWSHOT_SUB = AYIRAC.join(ORNEK_SUB) + AYIRAC

ORNEK_DOM = [
    "Considering the arguments carefully, I think pineapple belongs on pizza is right, "
    "because the objection has been answered several times already. I will set it out "
    "once more, slowly, and then the matter is settled.",
    "Considering the arguments carefully, I think open offices are a mistake is right, "
    "because anyone who has actually read the literature stopped arguing about this years "
    "ago. I do not expect the point to need repeating.",
]
FEWSHOT_DOM = AYIRAC.join(ORNEK_DOM) + AYIRAC

CAPALAR = {
    "submission": FEWSHOT_SUB + CERCEVE_ORT,
    "orijin":     CERCEVE_ORT,
    "dominance":  FEWSHOT_DOM + CERCEVE_ORT,
}
CAPA_SIRA = ["submission", "orijin", "dominance"]
CAPA_HASH = hashlib.sha256(json.dumps(CAPALAR, ensure_ascii=False, sort_keys=True)
                           .encode("utf-8")).hexdigest()[:16]


def tezler():
    T = json.load(open(__DNH_DATA__ + "/onek_korpus/pilot_tezler.json",
                       encoding="utf-8"))
    kalan = [t for t in T if t["debate"] not in TUTULAN]
    payda("dominance_tez", n_tez_tum=len(T), n_tutulan=len(T) - len(kalan),
          n_tez=len(kalan), bekle={"n_tez_tum": 17, "n_tutulan": 4, "n_tez": 13})
    return kalan


def onek(capa, tez, durus):
    return CAPALAR[capa].format(T=tez, P=DURUS[durus])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--yazar", required=True)
    ap.add_argument("--n-cekim", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    UP.OUT = OUT
    c = UP.cevre(YAZARLAR[a.yazar], a.yazar)
    c["tarif"] = (f"{CAPA_SIRA}"
                  f"{CAPA_HASH}")
    c["capa_hash"] = CAPA_HASH
    c["tutulan"] = TUTULAN
    T = tezler()
    isler = [(t, d, k) for t in T for d in DURUS for k in CAPA_SIRA]
    payda("dominance_istem", n_tez=len(T), n_is=len(isler), n_capa=len(CAPA_SIRA),
          n_durus=len(DURUS), bekle={"n_tez": 13, "n_is": 78, "n_capa": 3})
    print(f"DOMINANCE KORPUS · yazar {a.yazar} · istem {len(isler)} × cekim {a.n_cekim} "
          f"· capa-lafiz {CAPA_HASH}")
    from vllm import LLM, SamplingParams
    snap = sorted(glob.glob(f"{YAZARLAR[a.yazar]}/snapshots/*"))[-1]
    llm = LLM(model=snap, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=2048, disable_log_stats=True, seed=UP.SEED)
    tok = llm.get_tokenizer()
    ist = [onek(k, t["tez"], d) for t, d, k in isler]
    sp = SamplingParams(temperature=UP.SICAKLIK, top_p=UP.TOP_P, max_tokens=UP.B_ANA,
                        n=a.n_cekim, seed=UP.SEED)
    t0 = time.time(); cikti = llm.generate(ist, sp); dt = time.time() - t0
    sat, njet = [], 0
    for (t, d, k), o in zip(isler, cikti):
        for ci, s in enumerate(o.outputs):
            tid = list(s.token_ids); njet += len(tid)
            r = dict(uretici=a.yazar, debate=t["debate"], tez=t["tez"], durus=d,
                     capa=k, cekim=ci, n_token=len(tid), bitis=s.finish_reason,
                     onek=onek(k, t["tez"], d))
            for B in UP.BUTCELER:
                r[f"metin_{B}"] = tok.decode(tid[:B], skip_special_tokens=True)
            sat.append(r)
    yol = f"{OUT}/dom__{a.yazar}.jsonl"
    tmp = yol + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in sat:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, yol)
    payda(f"dominance_uretim_{a.yazar}", n_satir=len(sat), n_is=len(isler),
          bekle={"n_satir": len(isler) * a.n_cekim, "n_is": 78})
    c["kosular"] = [dict(etiket="dominance", n_satir=len(sat), n_jeton=njet,
                         saniye=round(dt, 1), yol=yol)]
    c["n_cekim"] = a.n_cekim
    json.dump(c, open(f"{OUT}/manifest__{a.yazar}.json", "w"), ensure_ascii=False, indent=1)
    print(f"  [HIZ] {njet} jeton / {dt:.1f}s · {len(sat)} satir → {yol}")


if __name__ == "__main__":
    main()
