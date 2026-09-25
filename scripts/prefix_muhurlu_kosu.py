#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, collections, subprocess, re
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/recon/gece_2026-07-16/hatC_register")
from reading_style_object import payda
from pergel_cekirdek import Cekirdek, Permutator, foldlar
from prefix_null_measurement import (satirlar, kodla_qwen, kodla_e5, kapi_uc_kumas,
                             NFOLD, SEED, MDE_KAT)
from prefix_prereg_kapilari import verdict, ust_okuma, kapi_dejenerelik
from prefix_korpus_pilot import DURUS_SOZLUGU, ISI_SOZ, olc

KOK = __DNH_DATA__ + "/onek_korpus/muhurlu"
OUT = f"{KOK}/okuma"
YAZARLAR = ("mistral", "gemma")
OKUYUCULAR = ("qwen_sontoken_L22", "e5_L23")
HEDEF = 17 * 2 * 2 * 24
K_ANA, B_BOOT = 5000, 2000
KELIME = re.compile(r"[a-z']+")
SOZ_HEPSI = DURUS_SOZLUGU["arti"] | DURUS_SOZLUGU["eksi"]


def kapi_zincir(prereg, prediction):
    def ts(h):
        r = subprocess.run(["git", "-C", ROOT, "show", "-s", "--format=%ct", h],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{h}")
        return int(r.stdout.strip())
    tm, tb = ts(prereg), ts(prediction)
    if not tm <= tb:
        raise RuntimeError(f"{prereg} {prediction}"
                           f"")
    d = subprocess.run(["git", "-C", ROOT, "show", f"{prereg}:unreleased/onek"
                        "PREREG_onek_v2_2026-08-05.md"], capture_output=True, text=True)
    if "MÜHÜRLENDI" not in d.stdout:
        raise RuntimeError("")
    payda("zincir_kilidi", n_hash=2, n_dogrulanan=2, bekle={"n_dogrulanan": 2})
    print(f"  ✓ ZINCIR: prereg {prereg} → prediction {prediction} → HEAD · sira dogru")
    return dict(prereg=prereg, prediction=prediction, prereg_ts=tm, prediction_ts=tb)


def maskele(t):
    return KELIME.sub(lambda m: "___" if m.group(0) in SOZ_HEPSI else m.group(0), t.lower())


def sozluk_skoru(metinler):
    out = []
    for t in metinler:
        w = KELIME.findall(t.lower())
        out.append(sum(1 for x in w if x in DURUS_SOZLUGU["arti"])
                   - sum(1 for x in w if x in DURUS_SOZLUGU["eksi"]))
    return np.array(out, dtype=np.float32)


def auc_np(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    df = collections.defaultdict(list)
    for i, v in enumerate(s):
        df[v].append(i)
    for _v, ix in df.items():
        if len(ix) > 1:
            r[ix] = r[ix].mean()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def burrows_z(metinler):
    import hatC_register as H
    V = np.stack([H.funcvec(t) for t in metinler])
    mu, sd = V.mean(0), np.maximum(V.std(0), 1e-9)
    return (V - mu) / sd


def b_istatistik(Z, y):
    return float(np.mean(np.abs(Z[y == 1].mean(0) - Z[y == 0].mean(0))))


def b_duzluk(metinler, y, isi, tabaka, K=K_ANA):
    Z = burrows_z(metinler)
    out = {}
    for cer in sorted(set(isi.tolist())):
        m = isi == cer
        Zc, yc, tc = Z[m], y[m], tabaka[m]
        ger = b_istatistik(Zc, yc)
        P = Permutator(yc, tc.tolist(), seed=SEED)
        Yp = P.cek(K)
        null = np.array([b_istatistik(Zc, Yp[k]) for k in range(0, K, max(1, K // 500))])
        z = (ger - null.mean()) / null.std(ddof=1)
        out[cer] = dict(gercek=ger, null_merkez=float(null.mean()),
                        null_sd=float(null.std(ddof=1)), z_B=float(z), n_null=len(null))
    return out


def kill_yuzey(S, y, isi):
    out = {}
    for cer in sorted(set(isi.tolist())):
        m = isi == cer
        d = {}
        for lab, v in (("arti", 1), ("eksi", 0)):
            ix = np.where(m & (y == v))[0]
            uz = [len(KELIME.findall(S[i]["metin_160"].lower())) for i in ix]
            ttr = [len(set(KELIME.findall(S[i]["metin_160"].lower()))) / max(1, u)
                   for i, u in zip(ix, uz)]
            d[lab] = dict(uzunluk=float(np.median(uz)), ttr=float(np.median(ttr)), n=len(ix))
        out[cer] = dict(d_uzunluk=d["eksi"]["uzunluk"] - d["arti"]["uzunluk"],
                        d_ttr=d["eksi"]["ttr"] - d["arti"]["ttr"], **d)
    return out


def kill_isi_kume(S, y, isi, kume):
    out = {}
    for cer in sorted(set(isi.tolist())):
        say, top = 0, 0
        for d in sorted(set(kume.tolist())):
            m = (isi == cer) & (kume == d)
            v = {}
            for lab, val in (("arti", 1), ("eksi", 0)):
                ix = np.where(m & (y == val))[0]
                w = [KELIME.findall(S[i]["metin_160"].lower()) for i in ix]
                v[lab] = float(np.mean([sum(1 for x in ww if x in ISI_SOZ) / max(1, len(ww))
                                        for ww in w]))
            top += 1; say += int(v["eksi"] > v["arti"])
        out[cer] = dict(ayni_isaret=say, n_kume=top)
    return out


def kill_aidiyet(S_ger, S_pla, kol_pla="plasebo",
                 jeton_ger=("right", "wrong"), jeton_pla=("recent", "ancient")):
    yanlis = 0
    for R, kol, jet in ((S_ger, "gercek", jeton_ger), (S_pla, kol_pla, jeton_pla)):
        for r in R:
            if r.get("kol") != kol:
                yanlis += 1
            bekle = jet[0] if r["durus"] == "arti" else jet[1]
            if bekle not in r.get("onek", ""):
                yanlis += 1
    a = {r["metin_640"] for r in S_ger}
    b = {r["metin_640"] for r in S_pla}
    ozdes = len(a & b)
    payda("kol_aidiyeti", n_gercek=len(S_ger), n_plasebo=len(S_pla),
          n_tekil_gercek=len(a), n_tekil_plasebo=len(b),
          red_yanlis_atama=yanlis, red_ozdes_metin=ozdes,
          bekle={"n_gercek": HEDEF, "n_plasebo": HEDEF})
    if yanlis:
        raise RuntimeError(f"{yanlis}")
    return dict(n_gercek=len(S_ger), n_plasebo=len(S_pla), tekil_gercek=len(a),
                tekil_plasebo=len(b), yanlis_atama=yanlis, ozdes_metin=ozdes,
                ozdes_oran=ozdes / len(S_ger), kol_pla=kol_pla,
                jeton_ger=list(jeton_ger), jeton_pla=list(jeton_pla))


def hucre_oku(X, Xabl, y, kume, tabaka, soz, dev):
    fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
    dej, dd = kapi_dejenerelik(X, y, fold, dev)
    C = Cekirdek(X, fold, dev=dev, std=True)
    auc = float(C.auc_toplu(y[None, :], C.oku_toplu(y[None, :]))[0])
    P = Permutator(y, tabaka.tolist(), seed=SEED)
    Yp = P.cek(K_ANA)
    A_null = C.oku_toplu(Yp)
    nulls = C.auc_toplu(Yp, A_null)
    C.bosalt()
    soz_null = np.array([auc_np(Yp[k], soz) for k in range(0, K_ANA, 10)])
    a_null_alt = nulls[:: 10][: len(soz_null)]
    fark_null = a_null_alt - soz_null
    auc_soz = auc_np(y, soz)
    fark_ger = auc - auc_soz
    z_fark = float((fark_ger - fark_null.mean()) / fark_null.std(ddof=1))
    Ca = Cekirdek(Xabl, fold, dev=dev, std=True)
    auc_abl = float(Ca.auc_toplu(y[None, :], Ca.oku_toplu(y[None, :]))[0])
    Ca.bosalt()
    kumeler = np.array(sorted(set(kume.tolist())))
    ix_of = {k: np.where(kume == k)[0] for k in kumeler}
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(B_BOOT // 10):
        sec = rng.choice(kumeler, size=len(kumeler), replace=True)
        ix = np.concatenate([ix_of[k] for k in sec])
        if len(set(kume[ix].tolist())) < NFOLD:
            continue
        fb = foldlar(kume[ix].tolist(), k=NFOLD, seed=SEED)
        Cb = Cekirdek(X[ix], fb, dev=dev, std=True)
        boot.append(float(Cb.auc_toplu(y[ix][None, :], Cb.oku_toplu(y[ix][None, :]))[0]))
        Cb.bosalt()
    boot = np.array(boot)
    nm, ns = float(nulls.mean()), float(nulls.std(ddof=1))
    return dict(auc=auc, null_merkez=nm, null_sd=ns, mde=MDE_KAT * ns,
                z=(auc - nm) / ns, auc_ablasyon=auc_abl, z_abl=(auc_abl - nm) / ns,
                auc_sozluk=auc_soz, fark_null_merkez=float(fark_null.mean()),
                fark_null_sd=float(fark_null.std(ddof=1)), sozluk_fark_z=z_fark,
                ci_alt=float(np.percentile(boot, 2.5)), ci_ust=float(np.percentile(boot, 97.5)),
                ci_n=len(boot), dejenere=bool(dej), dejenere_detay=dd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--dev", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("═" * 92); print("MÜHÜRLÜ KOSU · ÖNEK-KORPUSU"); print("═" * 92)
    zk = kapi_zincir(a.prereg, a.prediction)

    R = {"zincir": zk, "hedef_payda": HEDEF, "hucreler": {}, "kill": {}}
    VERI, METIN = {}, {}
    for y in YAZARLAR:
        met, yy, kume, tabaka = satirlar(y, kok=KOK, bekle_satir=HEDEF)
        S = [json.loads(l) for l in open(f"{KOK}/ana__{y}.jsonl", encoding="utf-8")]
        Sp = [json.loads(l) for l in open(f"{KOK}/plasebo__{y}.jsonl", encoding="utf-8")]
        isi = np.array([r["isi"] for r in S])
        VERI[y] = (met, yy, kume, tabaka, isi, S, Sp)
        METIN[y] = met
        R["kill"].setdefault("aidiyet", {})[y] = kill_aidiyet(S, Sp)
        R["kill"].setdefault("yuzey", {})[y] = kill_yuzey(S, yy, isi)
        R["kill"].setdefault("isi_kume", {})[y] = kill_isi_kume(S, yy, isi, kume)

    print("", flush=True)
    for y in YAZARLAR:
        met, yy, kume, tabaka, isi, S, Sp = VERI[y]
        b = b_duzluk(met, yy, isi, tabaka)
        R["kill"].setdefault("b_duzluk", {})[y] = b
        for cer, v in b.items():
            print(f"  {y}/{cer}: gercek {v['gercek']:.5f} · null {v['null_merkez']:.5f}"
                  f"±{v['null_sd']:.5f} ⇒ z_B = {v['z_B']:+.2f}")

    print("\n─── KODLAMA (gercek · ablasyon · plasebo) ───", flush=True)
    abl = {y: [maskele(t) for t in METIN[y]] for y in YAZARLAR}
    pla = {y: [r["metin_160"] for r in VERI[y][6]] for y in YAZARLAR}
    XQ = kodla_qwen({**{f"{y}": METIN[y] for y in YAZARLAR},
                     **{f"{y}__abl": abl[y] for y in YAZARLAR},
                     **{f"{y}__pla": pla[y] for y in YAZARLAR}}, a.dev)
    XE, e5not = kodla_e5({**{f"{y}": METIN[y] for y in YAZARLAR},
                          **{f"{y}__abl": abl[y] for y in YAZARLAR},
                          **{f"{y}__pla": pla[y] for y in YAZARLAR}}, a.dev)
    R["e5_7_4"] = e5not
    XX = {"qwen_sontoken_L22": XQ, "e5_L23": XE}

    print("\n─── HÜCRELER (yazar × okuyucu) ───", flush=True)
    for y in YAZARLAR:
        met, yy, kume, tabaka, isi, S, Sp = VERI[y]
        soz = sozluk_skoru(met)
        for ok in OKUYUCULAR:
            kapi_uc_kumas(y, ok)
            t0 = time.time()
            h = hucre_oku(XX[ok][y], XX[ok][f"{y}__abl"], yy, kume, tabaka, soz, a.dev)
            fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
            Cp = Cekirdek(XX[ok][f"{y}__pla"], fold, dev=a.dev, std=True)
            h["auc_plasebo"] = float(Cp.auc_toplu(yy[None, :], Cp.oku_toplu(yy[None, :]))[0])
            Cp.bosalt()
            h["z_plasebo"] = (h["auc_plasebo"] - h["null_merkez"]) / h["null_sd"]
            R["hucreler"][f"{y}|{ok}"] = h
            print(f"  {y}/{ok}: AUC {h['auc']:.4f} · null {h['null_merkez']:.4f}"
                  f"±{h['null_sd']:.5f} ⇒ z {h['z']:+.2f} · abl z {h['z_abl']:+.2f} · "
                  f"sözlük {h['auc_sozluk']:.4f} (z_fark {h['sozluk_fark_z']:+.2f}) · "
                  f"plasebo z {h['z_plasebo']:+.2f} · CI [{h['ci_alt']:.4f}, {h['ci_ust']:.4f}]"
                  f" · {time.time()-t0:.0f}s", flush=True)
    json.dump(R, open(f"{OUT}/okuma_ham.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n→ {OUT}/okuma_ham.json  (verdict IS-5'te, yargictan SONRA)")

    yasak = {"onek", "durus", "isi", "debate", "tez", "uretici", "kol", "cekim"}
    n_yasak = 0
    for y in YAZARLAR:
        S = VERI[y][5]
        yuk = []
        for i, r in enumerate(S):
            d = {"satir_id": f"{y}#{i}", "metin": r["metin_640"]}
            n_yasak += len(set(d) & yasak)
            yuk.append(d)
        json.dump(yuk, open(f"{OUT}/yargic_yuk__{y}.json", "w"), ensure_ascii=False)
    payda("yargic_korluk", n_satir=sum(len(VERI[y][5]) for y in YAZARLAR),
          n_alan=2, red_yasak_alan=n_yasak, bekle={"n_satir": 2 * HEDEF})
    if n_yasak:
        raise RuntimeError(f"KÖRLÜK MUHAFIZI DÜSTÜ: {n_yasak} yasak alan ⇒ kosu durur.")
    print(f"  ✓ körlük muhafizi: yasak alan 0 · yük {OUT}/yargic_yuk__*.json")


if __name__ == "__main__":
    main()
