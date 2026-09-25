#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, hashlib, re
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reading_style_object import payda

ROOT = __DNH_ROOT__ + ""
CIKTI = __DNH_DATA__ + "/rk_p1"
SOZLUK = f"{ROOT}/unreleased/arketip_sozlugu_v1.json"
KIALO = __DNH_DATA__ + "/unreleased/dataframe.csv"
URETICI = "http://172.16.180.28:80/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-32B-Instruct"
MAN = f"{ROOT}/unreleased/rk_p1_2026-08-01.json"
SEED, N_CEKIM, MAX_TOK, SICAKLIK = 20260801, 4, 320, 0.9
ES_ZAMAN = 24
os.makedirs(CIKTI, exist_ok=True)


def kollar(S):
    k = {a: v for a, v in S["arketipler_en"].items()}
    k["duz_okur"] = ""
    for a, v in S["placebo_katmanlari_en"].items():
        k[a] = v
    return k


def kialo_ebeveynler(n_hedef=240, min_ata=2):
    d = pd.read_csv(KIALO).dropna(subset=["Claim", "Parent_claim", "Debate_name"])
    for c in ("Claim", "Parent_claim"):
        d[c] = d[c].astype(str).str.strip()
    sec, atlanan = [], {"kisa_zincir": 0, "kisa_iddia": 0}
    for db, g in d.groupby("Debate_name"):
        M = dict(zip(g.Claim, g.Parent_claim))
        for r in g.itertuples():
            yol, cur, gor = [], M.get(r.Claim), set()
            while cur and cur not in gor:
                yol.append(cur); gor.add(cur); cur = M.get(cur)
            if len(yol) < min_ata:
                atlanan["kisa_zincir"] += 1; continue
            if len(r.Claim.split()) < 8:
                atlanan["kisa_iddia"] += 1; continue
            sec.append(dict(id=f"{db}|{r.Level}", tartisma=db, level=str(r.Level),
                            derinlik=len(yol), stance=str(r.Stance), iddia=r.Claim,
                            yol=list(reversed(yol))))
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(sec)
    if df.empty:
        raise RuntimeError("")
    pay = max(1, n_hedef // max(1, df.tartisma.nunique()))
    alt = [g.iloc[rng.permutation(len(g))[:pay]] for _, g in df.groupby("tartisma")]
    out = pd.concat(alt).to_dict("records")
    payda("ebeveyn_secimi", n_aday=len(sec), n_secilen=len(out),
          n_tartisma=int(df.tartisma.nunique()), atlanan_kisa_zincir=atlanan["kisa_zincir"],
          atlanan_kisa_iddia=atlanan["kisa_iddia"])
    return out


def istem(S, e, kol_metni, gorev):
    bag = "CONTEXT (debate path, root first):\n" + "\n".join(
        f"  {i+1}. {t}" for i, t in enumerate(e["yol"]))
    g = S["gorevler_en"][gorev]
    dav = kol_metni.strip()
    return S["template_en"].format(baglam=bag, iddia=e["iddia"],
                                 davranis=dav if dav else "", gorev=g).replace("\n\n\n", "\n\n")


def cek(prompt, n, seed, zaman=180):
    r = requests.post(URETICI, timeout=zaman, json=dict(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOK, temperature=SICAKLIK, top_p=0.95, n=n, seed=seed))
    r.raise_for_status()
    return [c["message"]["content"] for c in r.json()["choices"]]


def kabul(t):
    if t is None: return False, "bos"
    s = t.strip()
    if len(s) < 40: return False, "cok_kisa"
    if len(s.split()) < 12: return False, "az_kelime"
    if re.match(r"^\s*(I'm sorry|I cannot|As an AI)", s, re.I): return False, "red_kalibi"
    return True, None


def duman(S):
    print("═" * 90); print("K0.D · DUMAN TESTI — throughput + parametre dogrulamasi")
    print("  (tek örnek, defter disi, hicbir esik yazilmaz)"); print("═" * 90, flush=True)
    E = kialo_ebeveynler(24); K = kollar(S)
    t0 = time.time(); o = cek(istem(S, E[0], K["kanit_arayan"], "karsi_cik"), N_CEKIM, 1)
    dt = time.time() - t0
    tok = sum(len(x.split()) for x in o) / max(len(o), 1)
    print(f"  throughput: {N_CEKIM} cekim / {dt:.1f}s = {N_CEKIM/dt:.2f} cekim/s "
          f"(≈{tok:.0f} kelime/cekim)", flush=True)
    print(f"\n  PARAMETRE DOGRULAMASI — kol basina tek örnek ({len(K)} kol):", flush=True)
    kotu, uzunluklar = [], {}
    for ad, metni in K.items():
        try:
            t = cek(istem(S, E[1], metni, "karsi_cik"), 1, 2)[0]
        except Exception as ex:
            kotu.append((ad, f"istisna:{type(ex).__name__}")); continue
        ok, sebep = kabul(t)
        uzunluklar[ad] = len(t.split())
        print(f"    {ad:<28} {'KABUL' if ok else '★RED:'+str(sebep):<12} "
              f"{len(t.split()):>4} kelime | {t.strip()[:52]!r}", flush=True)
        if not ok: kotu.append((ad, sebep))
    ul = np.array(list(uzunluklar.values()))
    print(f"\n  cikti uzunlugu: medyan {np.median(ul):.0f} · min {ul.min()} · maks {ul.max()} kelime")
    tl = {a: len(v.split()) for a, v in K.items() if v}
    print(f""
          f"{min(tl.values())} {max(tl.values())}")
    payda("duman", n_kol=len(K), red_kol=len(kotu), n_ebeveyn_aday=len(E))
    if kotu:
        print(f"  ★ RED VEREN KOLLAR: {kotu}")
        print("")
    return dict(cekim_s=N_CEKIM / dt, kelime_cekim=float(tok), red=kotu,
                yonerge_uz=tl, cikti_uz={k: int(v) for k, v in uzunluklar.items()})


def uret(S, n_ebeveyn=240):
    E = kialo_ebeveynler(n_ebeveyn); K = kollar(S)
    gorevler = ["ozet", "karsi_cik"]
    print(f"\n{'═'*90}\nP1 ÜRETIM · {len(E)} ebeveyn × {len(K)} kol × {len(gorevler)} görev "
          f"× {N_CEKIM} cekim = {len(E)*len(K)*len(gorevler)*N_CEKIM} cekim\n{'═'*90}", flush=True)
    t0 = time.time(); say = dict(istenen=0, uretilen=0, reddedilen=0, atlanan_shard=0)
    red_sebep = {}
    for gi, gorev in enumerate(gorevler):
        for ki, (kol, metni) in enumerate(K.items()):
            yol = f"{CIKTI}/{gorev}__{kol}.jsonl"
            if os.path.exists(yol) and sum(1 for _ in open(yol)) >= len(E):
                say["atlanan_shard"] += 1
                print(f"  [idempotent] {gorev}/{kol} DISKTE tam ⇒ atlandi", flush=True)
                continue
            gec = f"{yol}.tmp"

            def _bir(arg):
                ei, e = arg
                try:
                    return ei, e, cek(istem(S, e, metni, gorev), N_CEKIM, SEED + ei), None
                except Exception as ex:
                    return ei, e, None, f"istisna:{type(ex).__name__}"

            with ThreadPoolExecutor(max_workers=ES_ZAMAN) as ex_, open(gec, "w") as f:
                for ei, e, outs, hata in ex_.map(_bir, list(enumerate(E))):
                    say["istenen"] += N_CEKIM
                    if hata:
                        say["reddedilen"] += N_CEKIM
                        red_sebep[hata] = red_sebep.get(hata, 0) + N_CEKIM
                        continue
                    kabuller = []
                    for t in outs:
                        ok, sb = kabul(t)
                        if ok:
                            kabuller.append(t.strip()); say["uretilen"] += 1
                        else:
                            say["reddedilen"] += 1
                            red_sebep[sb] = red_sebep.get(sb, 0) + 1
                    if kabuller:
                        f.write(json.dumps(dict(id=e["id"], tartisma=e["tartisma"],
                                                derinlik=e["derinlik"], stance=e["stance"],
                                                gorev=gorev, kol=kol, cekimler=kabuller),
                                           ensure_ascii=False) + "\n")
            os.replace(gec, yol)
            n_sat = sum(1 for _ in open(yol))
            print(f"  {gorev}/{kol}: {n_sat}/{len(E)} satir  [{time.time()-t0:.0f}s]", flush=True)
    payda("uretim", n_kol=len(K), n_ebeveyn=len(E), n_uretilen=say["uretilen"],
          red_cekim=say["reddedilen"], atlanan_shard=say["atlanan_shard"],
          n_istenen=say["istenen"])
    print(f"{red_sebep}")
    json.dump(dict(protokol="PROTOKOL_GECE_P1P2_2026-08-01.md@0a27dee", sayac=say,
                   red_sebep=red_sebep, n_ebeveyn=len(E), n_kol=len(K),
                   sozluk_sha=hashlib.sha256(open(SOZLUK, "rb").read()).hexdigest()[:12]),
              open(MAN, "w"), indent=1)
    print(f"  → manifest {MAN}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duman", action="store_true"); ap.add_argument("--uret", action="store_true")
    ap.add_argument("--n", type=int, default=240)
    a = ap.parse_args()
    S = json.load(open(SOZLUK))
    d = duman(S)
    if a.uret:
        if d["red"]:
            raise RuntimeError(f"{d['red']}")
        uret(S, a.n)
