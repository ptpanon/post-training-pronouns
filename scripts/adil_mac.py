#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, hashlib, re, collections
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reading_style_object import payda
from p1_sandbox import kollar, cek, kabul

ROOT = __DNH_ROOT__ + ""
KOK = __DNH_DATA__ + "/unreleased/saved-corpora"
GOMU = __DNH_DATA__ + "/rk_p4/gomu"
CIK = __DNH_DATA__ + "/rk_p5"
SOZLUK = f"{ROOT}/unreleased/arketip_sozlugu_v1.json"
MAN = f"{ROOT}/unreleased/rk_p5_2026-08-02.json"
SEED, N_CEKIM, ES_ZAMAN = 20260802, 5, 24
MAKS_YOL, MAKS_KAR = 4, 220
os.makedirs(CIK, exist_ok=True)
GOREV_EN = ("TASK: Write a reply to the comment above, as a participant in this thread. "
            "Write only the reply itself.")


def ebeveyn_havuzu(n_hedef=198):
    rng = np.random.default_rng(SEED)
    pay = max(1, n_hedef // 9)
    out, red = [], collections.Counter()
    for ix in sorted(f for f in os.listdir(GOMU) if f.endswith("__indeks.json")):
        c = ix.replace("__indeks.json", "")
        D = json.load(open(f"{GOMU}/{ix}"))
        metin, ebe = {}, {}
        for line in open(f"{KOK}/{c}/utterances.jsonl", encoding="utf-8", errors="ignore"):
            try: d = json.loads(line)
            except: continue
            metin[d["id"]] = d.get("text")
            p = d.get("reply_to"); ebe[d["id"]] = None if p in ("None", None) else p
        aday = []
        for s in D["sat"]:
            p = s["ebeveyn"]
            yol, cur, gor = [], ebe.get(p), set()
            while cur and cur not in gor and len(yol) < MAKS_YOL:
                t = metin.get(cur)
                if not t or len(str(t).strip()) < 20: break
                gor.add(cur); yol.append(str(t).strip()); cur = ebe.get(cur)
            if len(yol) < 1:
                red["yol_yok"] += 1; continue
            aday.append(dict(id=p, korpus=c, ebeveyn=str(metin[p]).strip(),
                             yol=list(reversed(yol)), n_kardes=len(s["kardesler"])))
        if not aday:
            red["korpus_bos"] += 1; continue
        sec = [aday[i] for i in rng.permutation(len(aday))[:pay]]
        out += sec
    payda("p5_ebeveyn", n_ebeveyn=len(out), n_korpus=9, red_yol_yok=red["yol_yok"])
    return out


def kes(t, n=MAKS_KAR):
    w = str(t).split()
    return " ".join(w[:n]) + (" …" if len(w) > n else "")


def istem(S, e, davranis):
    bag = "CONTEXT (thread, oldest first):\n" + "\n".join(
        f"  {i+1}. {kes(t)}" for i, t in enumerate(e["yol"]))
    d = davranis.strip()
    return (f"{bag}\n\nCOMMENT: {kes(e['ebeveyn'])}\n\n"
            + (d + "\n\n" if d else "") + GOREV_EN)


def duman(S, E, K):
    print("═" * 90); print("P5 · DUMAN + PARAMETRE DOGRULAMASI (kol basina tek örnek)")
    print("═" * 90, flush=True)
    t0 = time.time(); o = cek(istem(S, E[0], K["kanit_arayan"]), N_CEKIM, 1)
    dt = time.time() - t0
    print(f"  throughput: {N_CEKIM}/{dt:.1f}s = {N_CEKIM/dt:.2f} cekim/s "
          f"(≈{np.mean([len(x.split()) for x in o]):.0f} kelime)", flush=True)
    kotu, uz = [], {}
    for ad, m in K.items():
        try: t = cek(istem(S, E[1], m), 1, 2)[0]
        except Exception as ex: kotu.append((ad, f"istisna:{type(ex).__name__}")); continue
        ok, sb = kabul(t); uz[ad] = len(t.split())
        print(f"    {ad:<28} {'KABUL' if ok else '★RED:'+str(sb):<12} {len(t.split()):>4} kelime "
              f"| {t.strip()[:48]!r}", flush=True)
        if not ok: kotu.append((ad, sb))
    u = np.array(list(uz.values()))
    print(f"  cikti uzunlugu: medyan {np.median(u):.0f} · min {u.min()} · maks {u.max()}")
    payda("p5_duman", n_kol=len(K), red_kol=len(kotu))
    return kotu


def uret(S, E, K):
    print(f"\n{'═'*90}\nP5 ÜRETIM · {len(E)} ebeveyn × {len(K)} kol × {N_CEKIM} cekim "
          f"= {len(E)*len(K)*N_CEKIM}\n{'═'*90}", flush=True)
    t0 = time.time(); say = collections.Counter(); red_sebep = collections.Counter()
    for kol, metni in K.items():
        yol = f"{CIK}/reply__{kol}.jsonl"
        if os.path.exists(yol) and sum(1 for _ in open(yol)) >= len(E):
            say["atlanan_shard"] += 1; print(f"  [idempotent] {kol} ⇒ atlandi", flush=True); continue
        gec = f"{yol}.tmp"

        def _bir(arg):
            ei, e = arg
            try: return e, cek(istem(S, e, metni), N_CEKIM, SEED + ei), None
            except Exception as ex: return e, None, f"istisna:{type(ex).__name__}"

        with ThreadPoolExecutor(max_workers=ES_ZAMAN) as ex_, open(gec, "w") as f:
            for e, outs, hata in ex_.map(_bir, list(enumerate(E))):
                say["istenen"] += N_CEKIM
                if hata:
                    say["red"] += N_CEKIM; red_sebep[hata] += N_CEKIM; continue
                kb = []
                for t in outs:
                    ok, sb = kabul(t)
                    if ok: kb.append(t.strip()); say["uretilen"] += 1
                    else: say["red"] += 1; red_sebep[sb] += 1
                if kb:
                    f.write(json.dumps(dict(id=e["id"], korpus=e["korpus"],
                                            n_kardes=e["n_kardes"], kol=kol, cekimler=kb),
                                       ensure_ascii=False) + "\n")
        os.replace(gec, yol)
        print(f"  {kol}: {sum(1 for _ in open(yol))}/{len(E)} satir [{time.time()-t0:.0f}s]", flush=True)
    payda("p5_uretim", n_kol=len(K), n_ebeveyn=len(E), n_uretilen=say["uretilen"],
          red_cekim=say["red"], n_istenen=say["istenen"], atlanan_shard=say["atlanan_shard"])
    print(f"  RED DAGILIMI (eksik veri): {dict(red_sebep)}")
    json.dump(dict(direktif="D7-B adil mac", sayac=dict(say), red=dict(red_sebep),
                   n_ebeveyn=len(E), n_kol=len(K), gorev=GOREV_EN, n_cekim=N_CEKIM,
                   sozluk_sha=hashlib.sha256(open(SOZLUK, "rb").read()).hexdigest()[:12]),
              open(MAN, "w"), indent=1)
    print(f"  → manifest {MAN}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--uret", action="store_true")
    ap.add_argument("--n", type=int, default=198)
    a = ap.parse_args()
    S = json.load(open(SOZLUK)); E = ebeveyn_havuzu(a.n); K = kollar(S)
    kotu = duman(S, E, K)
    if a.uret:
        if kotu: raise RuntimeError(f"{kotu}")
        uret(S, E, K)
