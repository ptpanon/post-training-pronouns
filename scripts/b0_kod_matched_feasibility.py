#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import collections, json, os, sys, time
import numpy as np
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import filtre_verim as FV
import form_count as FS
import three_arm as UK
import person_dial_arm as KK
import person_dial_pair_sagligi as KCS
from reading_style_object import payda

ONBELLEK = __DNH_DATA__ + "/dial_kol/b0_tarama.npz"
HEDEF_KOL = __DNH_DATA__ + "/dial_kol/KISI_ASAGI.jsonl"
CIK = os.environ.get("B0_CIK", f"{KOK}/results/B0_KOD_ESLI_FIZIBILITE_2026-09-24.json")
N_ISTENEN = 7537
SEED = 20260924
N_KUTU = 4


def ozellikler(metinler):
    r = np.array([1.0 if KCS._RED.search(t) else 0.0 for t in metinler])
    k = np.array([1.0 if KCS._KOD.search(t) else 0.0 for t in metinler])
    l = np.array([1.0 if KCS._LISTE.search(t) else 0.0 for t in metinler])
    return r, k, l


def tara(npr=48):
    if os.path.exists(ONBELLEK):
        z = np.load(ONBELLEK, allow_pickle=True)
        print(f"  ⤿ önbellek VAR ⇒ tarama ATLANDI ({ONBELLEK})", flush=True)
        return {k: z[k] for k in z.files}
    from datasets import load_dataset
    ds = load_dataset(UK.REPO, split="train")
    C = [FV._son_asistan_mesaj(x) for x in ds["chosen"]]
    R = [FV._son_asistan_mesaj(x) for x in ds["rejected"]]
    KIMLIK = np.array(ds["id"], dtype=object)
    ozdes = np.array([c == r for c, r in zip(C, R)])
    gecerli = np.array([bool(c) and bool(r) for c, r in zip(C, R)]) & ~ozdes
    payda("b0_cikar", n_toplam=len(ds), hal_gecerli=int(gecerli.sum()),
          red_ozdes=int(ozdes.sum()))
    KL = UK.klise_yukle(); sc, sr = {}, {}
    Ch = [UK.hijyen(t, KL, sc)[0] for t in C]
    Rh = [UK.hijyen(t, KL, sr)[0] for t in R]
    nlp = FS._boru()
    t0 = time.time()
    m1c, njc = KK.m1_1k(nlp, Ch, npr)
    m1r, njr = KK.m1_1k(nlp, Rh, npr)
    print(f"  ★ kisi taramasi bitti · {time.time()-t0:.0f} sn", flush=True)
    rc, kc, lc = ozellikler(Ch); rr, kr, lr = ozellikler(Rh)
    D = dict(kimlik=KIMLIK, gecerli=gecerli,
             dM1=(m1c - m1r), dRed=(rc - rr), dKod=(kc - kr), dListe=(lc - lr),
             dBoy=(np.log(np.maximum(njc, 1)) - np.log(np.maximum(njr, 1))),
             njc=njc, njr=njr)
    np.savez(ONBELLEK, **D)
    return D


def main():
    D = tara()
    g = D["gecerli"].astype(bool)
    kimlik = D["kimlik"]
    yer = {k: i for i, k in enumerate(kimlik.tolist())}
    hedef = []
    for l in open(HEDEF_KOL, encoding="utf-8"):
        i = yer.get(json.loads(l)["id"])
        if i is not None and g[i]:
            hedef.append(i)
    hedef = np.array(sorted(set(hedef)))
    payda("b0_hedef", n_hedef_satir=len(hedef), n_gecerli_havuz=int(g.sum()),
          red_cozulemeyen=N_ISTENEN - len(hedef))

    aday_hepsi = np.where(g)[0]
    a = np.abs(D["dM1"])
    esik = float(np.percentile(a[aday_hepsi], 25))
    sifir = aday_hepsi[a[aday_hepsi] <= esik]
    print(f"{esik:.4f} {len(sifir):,}"
          f"{len(aday_hepsi):,}", flush=True)

    kenar_red = np.unique(np.percentile(D["dRed"][hedef], np.linspace(0, 100, N_KUTU + 1)))
    kenar_boy = np.unique(np.percentile(D["dBoy"][hedef], np.linspace(0, 100, N_KUTU + 1)))
    KIP = {"uclu":  (D["dKod"] > 0).astype(int) * 2 + (D["dKod"] < 0).astype(int),
           "ikili": (D["dKod"] != 0).astype(int),
           "yonlu": (D["dKod"] < 0).astype(int)}
    tavan = {}
    for ad, kk in KIP.items():
        t = (np.digitize(D["dRed"], kenar_red[1:-1]) * 100
             + np.digitize(D["dBoy"], kenar_boy[1:-1]) * 10 + kk)
        _, nm, s2 = UK.tabaka_esle(hedef, sifir, t, np.random.default_rng(SEED), n=None)
        tavan[ad] = dict(n_max=int(nm), **{k: int(v) for k, v in s2.items()})
        print(f"     kod kipi «{ad}»: tavan n = {nm:,} · tabaka {s2['n_tabaka']} "
              f"(atilan {s2['n_atilan_tabaka']}, hedefte {s2['n_atilan_satir']} satir)", flush=True)
    KIP_SEC = os.environ.get("B0_KOD_KIP", "ikili")
    kod_kod = KIP[KIP_SEC]
    tab = (np.digitize(D["dRed"], kenar_red[1:-1]) * 100
           + np.digitize(D["dBoy"], kenar_boy[1:-1]) * 10 + kod_kod)
    rng = np.random.default_rng(SEED)
    _, n_max, sr = UK.tabaka_esle(hedef, sifir, tab, rng, n=None)
    print(f"{KIP_SEC}", flush=True)
    print(f"{n_max:,} {N_ISTENEN:,} {N_ISTENEN}"
          f"", flush=True)
    print(f"     tabaka: kullanilan {sr['n_tabaka']} · atilan {sr['n_atilan_tabaka']} "
          f"(hedefte {sr['n_atilan_satir']} satir, asgari 5 alti)", flush=True)

    card = dict(sinif="SAYIM · fizibilite · prereg YOK", kod_kip_secilen=KIP_SEC, kod_kip_tavanlar=tavan,
                havuz=dict(gecerli=int(g.sum()), hedef=len(hedef)),
                kisi_sifir=dict(esik_p25=round(esik, 5), n_aday=len(sifir),
                                referans=""),
                tabaka=dict(n_kutu=N_KUTU, eksen=["dRed", "dBoy", "dKod"], **{k: int(v) for k, v in sr.items()}),
                fizibilite=dict(n_max=int(n_max), n_istenen=N_ISTENEN,
                                yeterli=bool(n_max >= N_ISTENEN)))

    if n_max >= N_ISTENEN:
        sec, _, _ = UK.tabaka_esle(hedef, sifir, tab, np.random.default_rng(SEED), n=N_ISTENEN)
        card["saglik"] = {}
        for ad, v in (("dRed", D["dRed"]), ("dBoy", D["dBoy"]), ("dKod", D["dKod"]),
                      ("dListe", D["dListe"]), ("dM1", D["dM1"])):
            card["saglik"][ad] = dict(hedef=round(float(v[hedef].mean()), 4),
                                      kontrol=round(float(v[sec].mean()), 4),
                                      fark=round(float(v[sec].mean() - v[hedef].mean()), 4))
        card["saglik_n"] = int(len(sec))
        print("  ★ ESLESME SAGLIGI (hedef ↔ kontrol · fark):", flush=True)
        for ad, s in card["saglik"].items():
            print(f"     {ad:<7} {s['hedef']:+.4f} ↔ {s['kontrol']:+.4f} · fark {s['fark']:+.4f}", flush=True)
    else:
        card["saglik"] = None
        print("", flush=True)

    json.dump(card, open(CIK, "w"), ensure_ascii=False, indent=1)
    payda("b0_fizibilite", n_aday_sifir=len(sifir), n_tabaka=sr["n_tabaka"],
          hal_n_max=int(n_max), hal_yeterli=int(n_max >= N_ISTENEN),
          red_atilan_tabaka=sr["n_atilan_tabaka"])
    print(f"  ★ card: {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
