#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, csv, re, time, ast, collections
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from prefix_muhurlu_kosu import burrows_z
from prefix_korpus_pilot import KELIME
import prefix_ladder as M
import prefix_head_avi as H
import prefix_head_onkapi as O

DIS = __DNH_DATA__ + "/unreleased/external"
CIKTI = f"{ROOT}/unreleased/UCUNCU_EKSEN_2026-08-08.json"
FIG = f"{ROOT}/unreleased/fig"
K_NULL, SEED, Z, MIN_FREK = 1000, 20260808, 1.645, 3
ADLAR = ("BAG-VAR", "BAG-YOK", "ÖLCÜLEMEZ")


def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hedge_sozlugu():
    c = collections.Counter()
    yol = f"{DIS}/hedgepeer/HedgePeer.jsonl"
    for l in open(yol, encoding="utf-8", errors="replace"):
        for s in json.loads(l)["Sentences"]:
            for h in s.get("Hedges", []):
                d = ast.literal_eval(h) if isinstance(h, str) else h
                w = str(d.get("Hedge", d.get("hedge", ""))).strip().lower()
                if w and re.fullmatch(r"[a-z' ]+", w):
                    c[w] += 1
    soz = {w for w, n in c.items() if n >= MIN_FREK}
    return soz, c, yol


def dominance():
    yol = f"{DIS}/warriner/Ratings_Warriner_et_al.csv"
    D = {}
    for r in csv.DictReader(open(yol, encoding="utf-8", errors="replace")):
        try:
            D[r["Word"].strip().lower()] = float(r["D.Mean.Sum"])
        except Exception:
            pass
    return D, yol


def olcum(met, soz, D):
    hed, dom, isabet = [], [], 0
    for t in met:
        j = KELIME.findall((t or "").lower())
        n = max(len(j), 1)
        h = sum(1 for w in j if w in soz)
        tl = " " + " ".join(j) + " "
        h += sum(tl.count(" " + s + " ") for s in soz if " " in s)
        hed.append(h / n); isabet += h > 0
        d = [D[w] for w in j if w in D]
        dom.append(float(np.mean(d)) if d else np.nan)
    return np.array(hed), np.array(dom), isabet


def perm_mde(v, y, blok, rng, K=K_NULL):
    f = float(v[y == 1].mean() - v[y == 0].mean())
    nul = np.empty(K)
    for i in range(K):
        yy = y.copy()
        for b in set(blok.tolist()):
            m = np.flatnonzero(blok == b)
            yy[m] = y[m][rng.permutation(len(m))]
        nul[i] = v[yy == 1].mean() - v[yy == 0].mean()
    return f, float(nul.mean()), float(nul.std(ddof=1))


def zemin_ton_dengeli():
    E = json.load(open(__DNH_DATA__ + "/onek_korpus/ton_esleme/"
                       "esleme_ton_2026-08-08.json", encoding="utf-8"))
    Y = {}
    d = __DNH_DATA__ + "/onek_korpus/ton_yargi"
    for f in sorted(os.listdir(d)):
        for r in json.load(open(f"{d}/{f}", encoding="utf-8")):
            Y[r["id"]] = r["tone"]
    S = {}
    for yz in ("mistral", "gemma"):
        for l in open(f"{__DNH_DATA__}/onek_korpus/ton_dengeli/ton__{yz}.jsonl",
                      encoding="utf-8"):
            r = json.loads(l)
            S[(yz, r["tez"], r["durus"], r["ton_hedef"], r["cekim"])] = r["metin_160"]
    met, y, blok = [], [], []
    for i, m in E.items():
        if i not in Y or Y[i] not in ("HARSH", "CALM"):
            continue
        k = (m["uretici"], m["tez"], m["durus"], m["ton_hedef"], m["cekim"])
        if k in S:
            met.append(S[k]); y.append(1 if Y[i] == "HARSH" else 0)
            blok.append(f"{m['uretici']}|{m['durus']}|{m['ton_hedef']}")
    return met, np.array(y, dtype=np.int8), np.array(blok)


def zemin_merdiven():
    E = json.load(open(O.ESLEME, encoding="utf-8"))
    Y = {}
    for f in sorted(os.listdir(O.YARGI)):
        for r in json.load(open(f"{O.YARGI}/{f}", encoding="utf-8")):
            Y[r["id"]] = r["tone"]
    met, y, blok = [], [], []
    for kol in ("mistral_base", "gemma_base", "mistral_it_ham", "gemma_it_sohbet"):
        S = {}
        for b in M.SIRA:
            _, rows = H.G.satirlar(kol, b)
            for r in rows:
                S[(b, r["tez"], r["durus"], r["cekim"])] = (r[H.G.PENCERE], r["debate"])
        for i, m in E.items():
            if (m["sinif"] != "merdiven" or m.get("merdiven_kol") != kol
                    or i not in Y or Y[i] not in ("HARSH", "CALM")):
                continue
            k = (m["basamak"], m["tez"], m["durus"], m["cekim"])
            if k in S and S[k][1] not in H.TUTULAN:
                met.append(S[k][0]); y.append(1 if Y[i] == "HARSH" else 0)
                blok.append(f"{kol}|{m['basamak']}")
    return met, np.array(y, dtype=np.int8), np.array(blok)


if __name__ == "__main__":
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(SEED)
    soz, frek, hyol = hedge_sozlugu()
    D, wyol = dominance()
    print(f"ÜCÜNCÜ EKSEN · damga {damga}")
    print(f"  hedge sözlügü: {len(soz)} ifade (frekans ≥ {MIN_FREK}, ham {len(frek)}) · "
          f"en sik: {[w for w,_ in frek.most_common(8)]}")
    print(f"  Warriner dominance: {len(D)} lemma")
    R = dict(damga=damga, adlar=list(ADLAR), kaynaklar=dict(
        hedgepeer=dict(yol=hyol, sha256=sha(hyol), n_ifade_ham=len(frek),
                       n_ifade_esikli=len(soz), esik=MIN_FREK,
                       en_sik=[w for w, _ in frek.most_common(12)]),
        warriner=dict(yol=wyol, sha256=sha(wyol), n_lemma=len(D))), zeminler={})

    for zad, fn, birincil in (("ton_dengeli", zemin_ton_dengeli, True),
                              ("merdiven", zemin_merdiven, False)):
        met, y, blok = fn()
        hed, dom, isabet = olcum(met, soz, D)
        Zb = burrows_z(met)
        pc1 = np.linalg.svd(Zb - Zb.mean(0), full_matrices=False)[2][0]
        b_eks = (Zb - Zb.mean(0)) @ pc1
        uz = np.array([len(KELIME.findall((t or "").lower())) for t in met])
        f, m, sd = perm_mde(hed, y, blok, rng)
        verdict = ("ÖLCÜLEMEZ" if len(y) < 50 or isabet == 0 else
                 ("BAG-VAR" if (f < 0 and abs(f - m) >= Z * sd) else "BAG-YOK"))
        fd, md, sdd = perm_mde(np.nan_to_num(dom, nan=float(np.nanmean(dom))), y, blok, rng)
        kor = {}
        for ad, v in (("hedge↔B_ekseni", (hed, b_eks)), ("hedge↔uzunluk", (hed, uz)),
                      ("dominance↔B_ekseni", (np.nan_to_num(dom, nan=np.nanmean(dom)), b_eks)),
                      ("hedge↔dominance", (hed, np.nan_to_num(dom, nan=np.nanmean(dom))))):
            kor[ad] = round(float(np.corrcoef(v[0], v[1])[0, 1]), 3)
        R["zeminler"][zad] = dict(
            n=int(len(y)), n_harsh=int(y.sum()), birincil=birincil,
            isabet=int(isabet), isabet_orani=round(isabet / len(y), 3),
            hedge_HARSH=round(float(hed[y == 1].mean()), 5),
            hedge_CALM=round(float(hed[y == 0].mean()), 5),
            hedge_fark=round(f, 5), hedge_null_merkez=round(m, 5),
            hedge_null_sd=round(sd, 5), hedge_MDE=round(Z * sd, 5), verdict=verdict,
            dom_HARSH=round(float(np.nanmean(dom[y == 1])), 4),
            dom_CALM=round(float(np.nanmean(dom[y == 0])), 4),
            dom_fark=round(fd, 4), dom_MDE=round(Z * sdd, 4), korelasyon=kor)
        v = R["zeminler"][zad]
        print(f"\n  ═══ {zad} (n={len(y)}, HARSH {int(y.sum())}, "
              f"{'BIRINCIL' if birincil else 'dogrulayici'}) ═══")
        print(f"    ★ isabet: {isabet}/{len(y)} = {v['isabet_orani']:.3f} metinde ≥1 hedge")
        print(f"    hedge orani HARSH {v['hedge_HARSH']:.5f} ↔ CALM {v['hedge_CALM']:.5f} · "
              f"fark {f:+.5f} (null {m:+.5f}±{sd:.5f}, MDE {Z*sd:.5f}) ⇒ **{verdict}**")
        print(f"    dominance HARSH {v['dom_HARSH']:.4f} ↔ CALM {v['dom_CALM']:.4f} · "
              f"fark {fd:+.4f} (MDE {Z*sdd:.4f})")
        print(f"    korelasyon {kor}")
    R["kupon_hukmu"] = R["zeminler"]["ton_dengeli"]["verdict"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for ax, zad in zip(axs, ("ton_dengeli", "merdiven")):
        v = R["zeminler"][zad]
        ax.bar(["HARSH", "CALM"], [v["hedge_HARSH"], v["hedge_CALM"]],
               color=["#c62828", "#1565c0"])
        ax.errorbar(0, v["hedge_HARSH"], yerr=v["hedge_MDE"], fmt="none", ecolor="k", capsize=4)
        ax.set_title(f"{zad} (n={v['n']}) · isabet {v['isabet_orani']:.2f} ⇒ {v['verdict']}",
                     fontsize=9)
        ax.set_ylabel("hedge orani (kelime basina)")
    f.tight_layout(); f.savefig(f"{FIG}/ucuncu_eksen_hedge.png", dpi=140); plt.close(f)
    R["figurler"] = ["unreleased/ucuncu_eksen_hedge.png"]
    payda("ucuncu_eksen", n_zemin=len(R["zeminler"]), n_hedge_ifade=len(soz),
          n_lemma=len(D), bekle={"n_zemin": 2, "n_hedge_ifade": 10, "n_lemma": 1000})
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    print(f"\n★ KUPON (birincil zemin): **{R['kupon_hukmu']}**  ·  → {CIKTI}")
