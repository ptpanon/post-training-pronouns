#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import argparse
import itertools
import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import prefix_steering_generation as SU
import prefix_kararlilik as KR
import prefix_yazar_bataryasi as YB
import prefix_harsh_vekil as HV
from prefix_yazar_resolve import delta_ist, verdict_yazar, OKURLAR, ADLAR
from prefix_ceiling_resolve import YANKI

HAVUZ = YB.OUT
CIKTI = f"{ROOT}/unreleased/YAZAR3_2026-08-09.json"
GOMU_OKUR = f"{HAVUZ}/y3_okur.npz"
GOMU_L21 = f"{HAVUZ}/y3_e5l21.npz"
BASAMAK = "B3"
SEED = 20260809
ALFA = 0.05

MUTLAK_TABAN_OKUNMAZ = 0.15
BAR_KIRLI_ORAN = YB.BAR_ORAN
BAR_YANKI = 0.10
BAR_UYUM = 0.60
K_UYUM_PERM = 200
EN_AZ_SATIR = 5
EN_AZ_YAZAR = 6
ATOM_TAVAN = 200000
EN_AZ_KARAKTER = 40


def arindir(satirlar):
    import prefix_ladder as M
    from prefix_korpus_pilot import CERCEVE
    P = {"sert": M.sabit_parcalar(M.BASAMAKLAR[BASAMAK]),
         "sakin": M.sabit_parcalar(CERCEVE["sakin"])}
    met, yk, ks = [], [], []
    for r in satirlar:
        b, t = M.istem_yankisi(r["metin"], P[r["ton_hedef"]])
        met.append(t)
        yk.append(b)
        ks.append(len(t) < EN_AZ_KARAKTER)
    return met, np.array(yk), np.array(ks)


def kollar():
    K = []
    for kol, m in YB.YAZARLAR.items():
        K.append(dict(kol=kol, aile=m["aile"], boy=m["boy"], terbiye="it", kip="sargi",
                      soy=kol))
    for kol, m in YB.YAZARLAR_TABAN.items():
        K.append(dict(kol=kol, aile=m["aile"], boy=m["boy"], terbiye="base", kip="ham",
                      soy=m["es"]))
    for k in K:
        k["ek"] = {"sargi": "_sargi", "ham": "_ham"}[k["kip"]] + f"_{BASAMAK.lower()}"
    return K


def atom_sayisi(gruplar):
    import math
    n = sum(gruplar)
    p = math.factorial(n)
    for g in gruplar:
        p //= math.factorial(g)
    return int(p)


def auc_np(p, y):
    y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    r = np.empty(len(p), float)
    o = np.argsort(np.asarray(p, float), kind="mergesort")
    s = np.asarray(p, float)[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gomu(a):
    K = kollar()
    S, sinir, tum = {}, {}, []
    eksik = []
    for k in K:
        yol = f"{HAVUZ}/uretim_{k['kol']}{k['ek']}.jsonl"
        if not os.path.exists(yol):
            eksik.append(k["kol"])
            continue
        S[k["kol"]] = [json.loads(l) for l in open(yol, encoding="utf-8")]
        sinir[k["kol"]] = (len(tum), len(tum) + len(S[k["kol"]]))
        tum += arindir(S[k["kol"]])[0]
    payda("y3_gomu_girdi", n_yazar=len(K), n_satir=len(tum),
          red_uretimsiz=len(eksik), bekle={"n_yazar": 17})
    if eksik:
        raise SystemExit(f"★ KAPI: üretimi olmayan yazar(lar) {eksik} — gömü BASLAMAZ")
    from gpu_lock import kilitle
    kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="yazar3_gomu")
    if not os.path.exists(GOMU_OKUR):
        from bs141_frontier_baselines import plain_enc
        veri = {"__sinir__": np.array([[sinir[k["kol"]][0], sinir[k["kol"]][1]]
                                       for k in K], int)}
        for ad, model, onek in OKURLAR:
            t0 = time.time()
            veri[ad] = plain_enc(model, tum, prefix=onek).astype(np.float16)
            print(f"  [{ad}] {veri[ad].shape} · {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(GOMU_OKUR + ".tmp", **veri)
        os.replace(GOMU_OKUR + ".tmp.npz", GOMU_OKUR)
    if not os.path.exists(GOMU_L21):
        import torch
        import anchor_kodlama as CK
        from transformers import AutoTokenizer, AutoModel
        CK.DEV = a.dev
        C = CK.KOLLAR["c1a"]
        tk = AutoTokenizer.from_pretrained(C["model"])
        mdl = AutoModel.from_pretrained(C["model"], dtype=torch.float32).to(a.dev).eval()
        eski = (C["kes"], C["maxlen"])
        C["kes"], C["maxlen"] = 4000, 512
        try:
            sc = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0, uzunluklar=[])
            E = CK.kodla(tk, mdl, tum, "c1a", bs=48, sayac=sc)
        finally:
            C["kes"], C["maxlen"] = eski
        A = np.asarray(E[:, 21, :], dtype=np.float16)
        payda("y3_e5l21", n_satir=int(sc["satir"]), n_boyut=int(A.shape[1]),
              red_bos_havuz=int(sc["bos_havuz"]), bekle={"n_satir": len(tum)})
        np.savez_compressed(GOMU_L21 + ".tmp", h=A)
        os.replace(GOMU_L21 + ".tmp.npz", GOMU_L21)
    print(f"→ {GOMU_OKUR}\n→ {GOMU_L21}")


_G = {}


def _isci_kur(D, soy_ix, yazarlar):
    _G["D"] = np.asarray(D)
    _G["soy_ix"] = list(soy_ix)
    _G["yazarlar"] = list(yazarlar)


def _bir_atom(atama):
    aile = [atama[s] for s in _G["soy_ix"]]
    return delta_ist(_G["D"], aile, _G["yazarlar"], soy=_G["soy_ix"])[0]


def null_tam(D, soy_ix, aile_soy, yazarlar, isci=32):
    boy = collections.Counter(aile_soy)
    n_atom = atom_sayisi(sorted(boy.values()))
    etiketler = sorted(boy)
    havuz = [e for e in etiketler for _ in range(boy[e])]
    if n_atom <= ATOM_TAVAN:
        atamalar = sorted(set(itertools.permutations(havuz)))
        kip = "tam-sayim"
    else:
        rng = np.random.default_rng(SEED)
        atamalar = [tuple(rng.permutation(havuz)) for _ in range(ATOM_TAVAN)]
        kip = "ornekleme"
    with ProcessPoolExecutor(max_workers=isci, initializer=_isci_kur,
                             initargs=(D, soy_ix, yazarlar)) as ex:
        v = list(ex.map(_bir_atom, atamalar, chunksize=64))
    v = np.array([x for x in v if np.isfinite(x)], float)
    return v, kip, n_atom, len(atamalar)


def coz(a):
    t0 = time.time()
    K = kollar()
    G = np.load(GOMU_OKUR)
    L21 = np.asarray(np.load(GOMU_L21)["h"], dtype=np.float32)
    sinir = {k["kol"]: tuple(G["__sinir__"][i]) for i, k in enumerate(K)}
    S = {k["kol"]: [json.loads(l) for l in
                    open(f"{HAVUZ}/uretim_{k['kol']}{k['ek']}.jsonl", encoding="utf-8")]
         for k in K}

    vekil = {ai: HV.vekil_kur(dislanan_aile=ai)[0]
             for ai in sorted(set(k["aile"] for k in K))}

    rng = np.random.default_rng(SEED)
    kapi, gecen = {}, []
    n_yanki_toplam, n_sat_toplam = 0, 0
    for k in K:
        kol, b, s = k["kol"], *sinir[k["kol"]]
        met, i_yanki, kisa = arindir(S[kol])
        lab = np.array([r["ton_hedef"] for r in S[kol]], dtype=object)
        m1, m0 = lab == "sert", lab == "sakin"
        F, _ = HV.yuzey_hizli(met)
        gecerli = ~kisa
        yerli = float(np.percentile(F["okunmazlik"][m0 & gecerli], 95)) \
            if (m0 & gecerli).sum() else float("nan")
        esik = max(yerli, MUTLAK_TABAN_OKUNMAZ) if np.isfinite(yerli) \
            else MUTLAK_TABAN_OKUNMAZ
        kirli = (F["okunmazlik"] > esik) | kisa
        kirli_oran = float(kirli[m1].mean())
        yk = np.array([bool(YANKI.search(r["metin"])) for r in S[kol]])
        yanki = float(yk[m1].mean())
        n_yanki_toplam += int(yk.sum())
        n_sat_toplam += len(met)
        iy_sert, iy_sakin = float(i_yanki[m1].mean()), float(i_yanki[m0].mean())
        kisa_oran = float(kisa.mean())
        p = vekil[k["aile"]](met, L21[b:s])
        t = ~kirli
        y = m1[t].astype(int)
        auc = auc_np(p[t], y)
        perm = np.array([auc_np(p[t], rng.permutation(y)) for _ in range(K_UYUM_PERM)])
        uyum_bar = max(float(np.percentile(perm, 95)), BAR_UYUM)
        ok = (kirli_oran <= BAR_KIRLI_ORAN) and (yanki <= BAR_YANKI) \
            and (kisa_oran <= BAR_KIRLI_ORAN) and (auc >= uyum_bar)
        kapi[kol] = dict(aile=k["aile"], terbiye=k["terbiye"], kip=k["kip"], soy=k["soy"],
                         boy=k["boy"], n_satir=len(met),
                         okunmaz_yerli_p95=round(yerli, 4), okunmaz_esik=round(esik, 4),
                         kirli_oran=round(kirli_oran, 4), fewshot_yanki=round(echo, 4),
                         istem_yanki_sert=round(iy_sert, 4),
                         istem_yanki_sakin=round(iy_sakin, 4),
                         kisa_oran=round(kisa_oran, 4),
                         tekrar_ort=round(float(F["tekrar"].mean()), 4),
                         vekil_auc=round(auc, 4), uyum_perm_p95=round(float(
                             np.percentile(perm, 95)), 4),
                         uyum_bar=round(uyum_bar, 4),
                         p_sert=round(float(p[m1 & t].mean()), 4),
                         p_sakin=round(float(p[m0 & t].mean()), 4), gecti=bool(ok))
        print(f"  {kol:6s} {k['terbiye']:4s} {k['kip']:5s} kirli {kirli_oran:.3f} "
              f"(esik {esik:.4f}) · fs-yanki {yanki:.4f} · istem-yanki "
              f"{iy_sert:.3f}/{iy_sakin:.3f} · kisa {kisa_oran:.3f} · vekil AUC {auc:.4f} "
              f"(bar {uyum_bar:.4f}) · p̄ {p[m1 & t].mean():.3f}/{p[m0 & t].mean():.3f}"
              f" ⇒ {'GECTI' if ok else '★ DÜSTÜ'}")
        if ok:
            gecen.append(k)

    payda("y3_kapi", n_yazar=len(K), n_satir=n_sat_toplam,
          red_kapidan=len(K) - len(gecen), bekle={"n_yazar": 17})
    aile_say = collections.Counter(k["aile"] for k in gecen)
    soy_ic = collections.Counter()
    for x, y_ in itertools.combinations(gecen, 2):
        if x["aile"] == y_["aile"] and x["soy"] != y_["soy"]:
            soy_ic[x["aile"]] += 1
    kapi_tamam = (len(gecen) >= EN_AZ_YAZAR and len(aile_say) >= 2
                  and sum(1 for v in soy_ic.values() if v > 0) >= 2)
    print(f"  [SONUC] gecen yazar={len(gecen)} · aile={dict(aile_say)} · "
          f"aile-ici-farkli-soy cift={dict(soy_ic)} ⇒ kapi_tamam={kapi_tamam}")

    sonuc = {}
    for ad, _m, _o in OKURLAR:
        E = np.asarray(G[ad], dtype=np.float32)
        E = E - E.mean(0)
        D, kalan, ic_kar = [], [], {}
        for k in gecen:
            b, s = sinir[k["kol"]]
            lab = np.array([r["ton_hedef"] for r in S[k["kol"]]], dtype=object)
            met, _iy, kisa = arindir(S[k["kol"]])
            F, _ = HV.yuzey_hizli(met)
            m0 = lab == "sakin"
            esik = kapi[k["kol"]]["okunmaz_esik"]
            temiz = (F["okunmazlik"] <= esik) & ~kisa
            s1, s0 = (lab == "sert") & temiz, m0 & temiz
            if s1.sum() < EN_AZ_SATIR or s0.sum() < EN_AZ_SATIR:
                print(f"    ★ {k['kol']}/{ad}: sert {int(s1.sum())} sakin {int(s0.sum())}"
                      " ⇒ yön kurulamadi")
                continue
            D.append(SU.birim(E[b:s][s1].mean(0) - E[b:s][s0].mean(0)))
            kume = np.array([r["debate"] for r in S[k["kol"]]], dtype=object)
            ic_kar[k["kol"]] = round(KR.kume_tutulan_bolme(
                E[b:s], (lab == "sert").astype(int), kume, n_bol=20)["medyan"], 4)
            kalan.append(k)
        if len(kalan) < EN_AZ_YAZAR:
            sonuc[ad] = dict(delta=None, n_yazar=len(kalan), not_="yazar < 6")
            continue
        D = np.stack(D)
        yaz = [k["kol"] for k in kalan]
        aile = [k["aile"] for k in kalan]
        soy = [k["soy"] for k in kalan]
        soylar = sorted(set(soy))
        soy_ix = [soylar.index(x) for x in soy]
        aile_soy = [next(k["aile"] for k in kalan if k["soy"] == sy) for sy in soylar]
        delta, n_ic, n_dis, med_ic, med_dis = delta_ist(D, aile, yaz, soy=soy)
        tav = [float(np.dot(D[i], D[j]))
               for i, j in itertools.combinations(range(len(yaz)), 2) if soy[i] == soy[j]]
        null, kip_null, n_atom, n_cek = null_tam(D, soy_ix, aile_soy, yaz, isci=a.isci)
        bg = np.array([k["boy"] for k in kalan], float)
        tert = np.digitize(bg, np.percentile(bg, [33.3, 66.7]))
        pl_boy = delta_ist(D, [f"t{t}" for t in tert], yaz, soy=soy)[0]
        pl_ter = delta_ist(D, [k["terbiye"] for k in kalan], yaz, soy=soy)[0]
        pl_max = float(np.nanmax([pl_boy, pl_ter]))
        p95 = float(np.percentile(null, 95))
        sd = float(np.std(null, ddof=1))
        sonuc[ad] = dict(
            delta=round(float(delta), 4), n_ic_cift=n_ic, n_dis_cift=n_dis,
            medyan_ic=round(med_ic, 4), medyan_dis=round(med_dis, 4),
            tavan_ayni_soy_medyan=round(float(np.median(tav)), 4) if tav else None,
            n_soy_cifti=len(tav),
            null_kip=kip_null, null_atom=n_atom, null_cekim=n_cek,
            null_merkez=round(float(null.mean()), 4), null_sd=round(sd, 4),
            null_p95=round(p95, 4), MDE=round(1.645 * sd, 4),
            p_min_atom=round(1.0 / max(n_atom, 1), 6),
            atom_canli=bool(1.0 / max(n_atom, 1) < ALFA),
            plasebo_boy=round(float(pl_boy), 4), plasebo_terbiye_bicim=round(float(pl_ter), 4),
            plasebo_max=round(pl_max, 4), yarim_bolme=ic_kar,
            n_yazar=len(kalan), yazarlar=yaz,
            asiyor=bool(delta > p95 and delta >= 1.645 * sd and delta > pl_max))
        print(f"    {ad:4s} Δ {delta:+.4f} | null {null.mean():+.4f}±{sd:.4f} p95 {p95:+.4f}"
              f" MDE {1.645*sd:.4f} | icmed {med_ic:.3f} dismed {med_dis:.3f} "
              f"TAVAN(soy) {np.median(tav) if tav else float('nan'):.3f} | "
              f"plasebo boy {pl_boy:+.4f} terbiye {pl_ter:+.4f} | {kip_null} "
              f"{n_cek}/{n_atom} atom ⇒ {'ASIYOR' if sonuc[ad]['asiyor'] else 'ASMIYOR'}")

    e5 = sonuc.get("e5", {})
    atom_ok = all(sonuc.get(ad, {}).get("atom_canli", False) for ad, _m, _o in OKURLAR)
    gecti = all(sonuc.get(ad, {}).get("asiyor") for ad, _m, _o in OKURLAR)
    nan = float("nan")
    H = verdict_yazar(e5.get("delta", nan) if e5.get("delta") is not None else nan,
                    e5.get("null_p95", nan) if e5.get("delta") is not None else nan,
                    e5.get("null_sd", nan) if e5.get("delta") is not None else nan,
                    e5.get("plasebo_max", nan) if e5.get("delta") is not None else nan,
                    bool(gecti), bool(kapi_tamam and atom_ok))
    from datetime import datetime, timezone
    out = dict(damga_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               prereg="PREREG_YAZAR3_2026-08-09.md", basamak=BASAMAK,
               barlar=dict(mutlak_taban_okunmaz=MUTLAK_TABAN_OKUNMAZ,
                           kirli_oran=BAR_KIRLI_ORAN, yanki=BAR_YANKI, uyum=BAR_UYUM),
               kapi=kapi, gecen=[k["kol"] for k in gecen],
               aile_dagilimi=dict(aile_say), kapi_tamam=bool(kapi_tamam),
               atom_canli=bool(atom_ok), yanki_toplam_oran=round(
                   n_yanki_toplam / max(n_sat_toplam, 1), 6),
               okurlar=sonuc, iki_encoder_gecti=bool(gecti), adlar=list(ADLAR), verdict=H,
               saniye=round(time.time() - t0, 1))
    json.dump(out, open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  ★★ K-YAZAR-3 HÜKMÜ: **{H}**\n→ {CIKTI}")


def _prova():
    import prefix_yazar_resolve as YC
    YC._prova()
    rng = np.random.default_rng(0)
    D = np.stack([d / np.linalg.norm(d) for d in rng.normal(size=(6, 32))])
    ai = ["a", "a", "b", "b", "c", "c"]
    yz = list("uvwxyz")
    assert delta_ist(D, ai, yz)[0] == delta_ist(D, ai, yz, soy=list("123456"))[0], \
        ""
    d5 = delta_ist(D, ai, yz, soy=["s", "s", "3", "4", "5", "6"])
    assert d5[1] + d5[2] == 14, f"{d5[1]+d5[2]}"
    assert atom_sayisi([3, 3, 3]) == 1680, atom_sayisi([3, 3, 3])
    assert atom_sayisi([1, 1]) == 2 and (2.0 / 2) >= ALFA, "n=2 atom ölü olmali"
    import prefix_ladder as M
    ham = ('Forum regular "NoFilter_Dave", known for never softening his words and never '
           'apologizing, replies: aliens exist is completely wrong. Here is why: '
           "They're based on wishful thinking.")
    P3 = M.sabit_parcalar(M.BASAMAKLAR["B3"])
    v, temiz = M.istem_yankisi(ham, P3)
    assert v and "NoFilter_Dave" not in temiz and "wishful thinking" in temiz, temiz
    assert M.istem_yankisi("plain harsh text", P3) == (False, "plain harsh text")
    S_ = [dict(metin=ham, ton_hedef="sert"), dict(metin="ok", ton_hedef="sakin")]
    m_, y_, k_ = arindir(S_)
    assert list(y_) == [True, False] and list(k_) == [False, True], (y_, k_)
    assert np.isnan(auc_np([1, 2, 3], [1, 1, 1])), "tek sinif ⇒ NaN"
    assert abs(auc_np([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-9
    assert abs(auc_np([1, 1, 1, 1], [0, 0, 1, 1]) - 0.5) < 1e-9, "sabit skor ⇒ 0,5"
    print("  ✓ prova gecti — D44 3=3 · soy-dislama · atom sayimi · AUC dejenere kipi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", required=True, choices=("prova", "gomu", "coz"))
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=None)
    ap.add_argument("--isci", type=int, default=32)
    a = ap.parse_args()
    if a.asama == "prova":
        _prova()
    elif a.asama == "gomu":
        gomu(a)
    else:
        coz(a)


if __name__ == "__main__":
    main()
