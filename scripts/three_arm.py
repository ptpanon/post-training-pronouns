#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, collections, hashlib, json, os, re, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from reading_style_object import payda
import form_count as FS
import filtre_verim as FV

REPO = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
VERI = __DNH_DATA__ + "/uc_kol"
DOC = f"{ROOT}/results"
KLISE = f"{DOC}/HIJYEN_ACILIS_KLISESI_2026-08-29.txt"
CIK = f"{DOC}/UC_KOL_2026-08-29.json"
SEED = 20260829
SAYAC = ("a", "b", "c", "d", "e", "b_cekirdek")
ETA_ESIK_DK = 45.0
N_BIN = 10

KAPANIS = re.compile(r"(?:let\s+me\s+know\s+if|feel\s+free\s+to\s+(?:ask|reach)|"
                     r"hope\s+(?:this|that)\s+helps|if\s+you\s+have\s+any\s+(?:other\s+)?questions)",
                     re.I)


def sha(yol):
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def klise_yukle():
    D = []
    for s in open(KLISE, encoding="utf-8"):
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        ad, dsn = s.rstrip("\n").split("\t", 1)
        D.append((ad, re.compile(dsn, re.I)))
    payda("uc_kol_klise", n_desen=len(D), bekle={"n_desen": 9})
    return D


def hijyen(t, D, sayac):
    n0 = len(t)
    for _ in range(3):
        for ad, rx in D:
            m = rx.match(t)
            if m and m.end() > 0:
                t = t[m.end():]
                sayac[ad] = sayac.get(ad, 0) + 1
                break
        else:
            break
    return t, len(t) != n0


def say(nlp, metinler, npr):
    A, n_c, n_j = FS.uretim_say(nlp, metinler, n_process=npr, batch=128)
    return {k: A[k].astype(np.int32) for k in SAYAC}, n_c, n_j


def kapi(nlp, ornek, n_hedef_metin, npr):
    t0 = time.time(); say(nlp, ornek, npr); dt = max(time.time() - t0, 1e-9)
    hiz = len(ornek) / dt
    eta = (n_hedef_metin / hiz) / 60
    payda("uc_kol_kapi", n_olcum=len(ornek), hal_metin_sn=round(hiz, 1),
          hal_eta_dk=round(eta, 1), hal_esik_dk=ETA_ESIK_DK)
    print(f"  [KAPI/W-412] {hiz:.0f} metin/sn ({npr} sürec) ⇒ TAM({n_hedef_metin:,}) "
          f"ETA {eta:.1f} dk ⇒ esik {ETA_ESIK_DK} dk ⇒ "
          f"EYLEM: asarsa HAM tarama tam setten 20 000'lik tohumlu alt-örnekleme DÜSER "
          f"(hijyenli tarama her hâlde TAM kosar)", flush=True)
    return hiz, eta


def yuklem(D):
    return {
        "EMIR_ARTI": (D["a"] >= 1) & (D["c"] <= -1),
        "EMIR_EKSI": (D["a"] <= -1) & (D["c"] >= 1),
        "FORM_YOK": np.all(np.stack([D[k] == 0 for k in SAYAC]), axis=0),
    }


def ortak_tabaka(lb, kaynak_kod, kenar):
    k = np.digitize(lb, kenar[1:-1])
    return k * 1000 + kaynak_kod


def _kutula(lb, idx, kenar):
    return [idx[(lb[idx] >= kenar[i]) & (lb[idx] < kenar[i + 1])] for i in range(N_BIN)]


def tabaka_esle(idx_hedef, idx_kol, tab, rng, n=None, asgari=5):
    th = collections.Counter(tab[idx_hedef].tolist())
    atilan = {k: v for k, v in th.items() if v < asgari}
    th = {k: v for k, v in th.items() if v >= asgari}
    top = sum(th.values())
    pay = {k: v / top for k, v in th.items()}
    kutu = {k: idx_kol[tab[idx_kol] == k] for k in th}
    n_max = int(min((len(kutu[k]) / pay[k]) if pay[k] > 0 else np.inf for k in th)) if th else 0
    if n is None:
        return None, n_max, dict(n_tabaka=len(th), n_atilan_tabaka=len(atilan),
                                 n_atilan_satir=int(sum(atilan.values())))
    sec = []
    for k in th:
        m = int(round(pay[k] * n))
        if m and len(kutu[k]):
            sec.append(rng.choice(kutu[k], size=min(m, len(kutu[k])), replace=False))
    return (np.sort(np.concatenate(sec)) if sec else np.array([], dtype=int)), n_max, {}


def boy_esle(idx_hedef, idx_kol, boy, rng, n=None):
    lb = np.log1p(boy)
    kenar = np.quantile(lb[idx_hedef], np.linspace(0, 1, N_BIN + 1))
    kenar[0] -= 1e-6; kenar[-1] += 1e-6
    pay = np.histogram(lb[idx_hedef], bins=kenar)[0].astype(float)
    pay /= pay.sum()
    kutu = _kutula(lb, idx_kol, kenar)
    n_max = int(min((len(kutu[i]) / pay[i]) if pay[i] > 0 else np.inf
                    for i in range(N_BIN)))
    if n is None:
        return None, n_max
    sec = []
    for i in range(N_BIN):
        k = int(round(pay[i] * n))
        if k and len(kutu[i]):
            sec.append(rng.choice(kutu[i], size=min(k, len(kutu[i])), replace=False))
    return (np.sort(np.concatenate(sec)) if sec else np.array([], dtype=int)), n_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-process", type=int, default=32)
    ap.add_argument("--kapi-ornek", type=int, default=2000)
    ap.add_argument("--ornek-cift", type=int, default=20)
    ap.add_argument("--n", type=int, default=0, help="0 = TAM SET")
    ap.add_argument("--v2", action="store_true",
                    help="ORTAK TABAKA eslemesi (boy × kaynak) — v1 diskte KALIR")
    ap.add_argument("--v2-kutu", type=int, default=5,
                    help="v2'de boy kutusu sayisi (tabaka patlamasini sinirlar)")
    ap.add_argument("--dorduncu", action="store_true",
                    help="RASTGELE-BÖLME kolunu ekle (BAYRAM-GÜNDÜZ §1/K-3)")
    ap.add_argument("--prova", action="store_true",
                    help="§8: sahte-kosu GERCEK kanala yazamaz — izole dizine yazar")
    a = ap.parse_args()
    global VERI, CIK
    if a.v2:
        VERI = __DNH_DATA__ + "/uc_kol_v2"
        CIK = f"{DOC}/UC_KOL_V2_2026-08-30.json"
        print(f"  ★ v2 KIPI — ORTAK TABAKA (boy×kaynak) · cikti {VERI} · v1 DOKUNULMADI",
              flush=True)
    if a.prova:
        VERI = "<scratch>"
        CIK = f"{VERI}/UC_KOL_PROVA.json"
        print("  ★ PROVA KIPI — cikti IZOLE:", VERI, flush=True)
    os.makedirs(VERI, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t00 = time.time(); dusen_ariza = 0

    from datasets import load_dataset
    ds = load_dataset(REPO, split="train")
    N = len(ds) if not a.n else min(a.n, len(ds))
    ds = ds.select(range(N))
    C = [FV._son_asistan_mesaj(x) for x in ds["chosen"]]
    R = [FV._son_asistan_mesaj(x) for x in ds["rejected"]]
    KAYNAK = ds["source"]; KIMLIK = ds["id"]
    gecerli = np.array([bool(c) and bool(r) for c, r in zip(C, R)])
    payda("uc_kol_cikar", n_toplam=len(ds), hal_gecerli=int(gecerli.sum()),
          red_bos_taraf=int((~gecerli).sum()))

    KL = klise_yukle()
    sc, sr = {}, {}
    Ch = [hijyen(t, KL, sc)[0] for t in C]
    Rh = [hijyen(t, KL, sr)[0] for t in R]
    n_soy_c = sum(sc.values()); n_soy_r = sum(sr.values())
    kap_c = int(sum(bool(KAPANIS.search(t)) for t in C))
    kap_r = int(sum(bool(KAPANIS.search(t)) for t in R))
    payda("uc_kol_hijyen", n_metin=2 * len(C), hal_soyulan_chosen=n_soy_c,
          hal_soyulan_rejected=n_soy_r, hal_kapanis_chosen=kap_c,
          hal_kapanis_rejected=kap_r)
    print(f"  ★ hijyen: chosen {n_soy_c:,} · rejected {n_soy_r:,} soyma "
          f"(desen-basina JSON'da) · KAPANIS klisesi (SOYULMADI, ölcüldü): "
          f"chosen {kap_c/len(C)*100:.1f}% · rejected {kap_r/len(R)*100:.1f}%", flush=True)

    nlp = FS._boru()
    hiz, eta = kapi(nlp, Ch[:a.kapi_ornek], 2 * len(Ch), a.n_process)
    ham_tam = eta <= ETA_ESIK_DK

    t1 = time.time()
    Ac, ncc, njc = say(nlp, Ch, a.n_process)
    Ar, ncr, njr = say(nlp, Rh, a.n_process)
    D = {k: Ac[k] - Ar[k] for k in SAYAC}
    print(f"  ★ hijyenli tarama bitti · {time.time()-t1:.0f} sn", flush=True)

    t2 = time.time()
    if ham_tam:
        ham_idx = np.arange(len(C))
    else:
        ham_idx = np.sort(rng.choice(len(C), size=min(20000, len(C)), replace=False))
    Ac0, _, njc0 = say(nlp, [C[i] for i in ham_idx], a.n_process)
    Ar0, _, _ = say(nlp, [R[i] for i in ham_idx], a.n_process)
    D0 = {k: Ac0[k] - Ar0[k] for k in SAYAC}
    Y0 = yuklem(D0); Y_alt = {k: v[ham_idx] for k, v in yuklem(D).items()}
    payda("uc_kol_ham_taban", n_ham=len(ham_idx), hal_tam=int(ham_tam),
          **{f"hal_ham_{k}": int(v.sum()) for k, v in Y0.items()})
    print(f"  ★ HAM taban bitti · n={len(ham_idx):,} · {time.time()-t2:.0f} sn", flush=True)

    Y = yuklem(D)
    ozdes = np.array([c == r for c, r in zip(C, R)])
    payda("uc_kol_ozdes", n_cift=len(C), red_chosen_esittir_rejected=int(ozdes.sum()))
    for k in Y:
        Y[k] &= gecerli & ~ozdes
    if a.dorduncu:
        Y["RASTGELE"] = gecerli & ~ozdes
    boy = (njc + njr) / 2.0
    idx = {k: np.flatnonzero(v) for k, v in Y.items()}
    payda("uc_kol_ham_kol", **{f"hal_{k}": len(v) for k, v in idx.items()})
    print("  ★ boy-esleme ÖNCESI: " + " · ".join(f"{k}={len(v):,}" for k, v in idx.items()), flush=True)

    hedef = idx["EMIR_ARTI"]
    tavan, v2_bilgi = {}, {}
    if a.v2:
        kaynaklar = sorted(set(KAYNAK))
        kkod = np.array([kaynaklar.index(x) for x in KAYNAK], dtype=np.int32)
        lb_ = np.log1p(boy)
        kenar = np.quantile(lb_[hedef], np.linspace(0, 1, a.v2_kutu + 1))
        kenar[0] -= 1e-6; kenar[-1] += 1e-6
        TAB = ortak_tabaka(lb_, kkod, kenar)
        for k in idx:
            tavan[k], v2_bilgi[k] = tabaka_esle(hedef, idx[k], TAB, rng)[1:]
        print(f"  ★ v2 tabaka: {a.v2_kutu} boy kutusu × {len(kaynaklar)} kaynak · "
              f"hedefte dolu tabaka {v2_bilgi[hedef.dtype and 'EMIR_ARTI']['n_tabaka']} · "
              f"atilan (ince) tabaka {v2_bilgi['EMIR_ARTI']['n_atilan_tabaka']} "
              f"({v2_bilgi['EMIR_ARTI']['n_atilan_satir']} satir)", flush=True)
    else:
        for k in idx:
            tavan[k] = boy_esle(hedef, idx[k], boy, rng)[1]
    n_ort = int(min(min(tavan.values()), len(hedef)))
    esli = {}
    for k in idx:
        esli[k] = (tabaka_esle(hedef, idx[k], TAB, rng, n=n_ort)[0] if a.v2
                   else boy_esle(hedef, idx[k], boy, rng, n=n_ort)[0])
        print(f"     {k}: {len(idx[k]):,} → {len(esli[k]):,} (kutu-kisitli tavan {tavan[k]:,})",
              flush=True)
    n_ort = int(min(len(v) for v in esli.values()))
    for k in esli:
        if len(esli[k]) > n_ort:
            esli[k] = np.sort(rng.choice(esli[k], size=n_ort, replace=False))
    print(f"  ★ üc kol ESIT n = {n_ort:,} (hedef histogrami: EMIR_ARTI, {N_BIN} kutu)", flush=True)

    lb_ = np.log1p(boy)
    ort = {k: float(lb_[v].mean()) for k, v in esli.items()}
    sd_ = float(lb_[np.concatenate(list(esli.values()))].std())
    fark = (max(ort.values()) - min(ort.values())) / max(sd_, 1e-9)
    BOY_ESIK = 0.10
    payda("uc_kol_boy_denge", n_kol=len(esli), hal_fark_sd=round(fark, 4),
          hal_esik_sd=BOY_ESIK)
    print(f"{fark:.3f}"
          f"{BOY_ESIK}"
          f"", flush=True)
    boy_serh = fark > BOY_ESIK

    kontrol = {}
    tum = np.concatenate([esli[k] for k in esli])
    kontrol["cakisma_kollar_arasi"] = int(len(tum) - len(set(tum.tolist())))
    YUK = [k for k in esli if k != "RASTGELE"]
    c_yuk = 0
    for i in range(len(YUK)):
        for j in range(i + 1, len(YUK)):
            c_yuk += len(set(esli[YUK[i]].tolist()) & set(esli[YUK[j]].tolist()))
    c_rast = kontrol["cakisma_kollar_arasi"] - c_yuk
    kontrol["cakisma_YUKLEMLI_kollar_arasi"] = int(c_yuk)
    kontrol["cakisma_RASTGELE_ile"] = int(c_rast)
    kontrol["chosen_rejected_ozdes"] = int(sum(
        1 for k in esli for i in esli[k] if C[i] == R[i]))
    kontrol["yuklem_yeniden"] = {k: int(Y[k][esli[k]].all()) for k in esli}
    kontrol["bos_taraf"] = int(sum(1 for k in esli for i in esli[k]
                                   if not C[i] or not R[i]))
    payda("uc_kol_kontrol", n_satir=len(tum),
          red_cakisma_yuklemli=kontrol["cakisma_YUKLEMLI_kollar_arasi"],
          hal_cakisma_rastgele=kontrol["cakisma_RASTGELE_ile"],
          red_ozdes=kontrol["chosen_rejected_ozdes"], red_bos=kontrol["bos_taraf"])
    if kontrol["cakisma_YUKLEMLI_kollar_arasi"] or kontrol["bos_taraf"] or \
       not all(kontrol["yuklem_yeniden"].values()):
        dusen_ariza += 1
        print("  ★★ ARIZA: §8 kontrolleri düstü", flush=True)

    KOL = {}
    for k, ii in esli.items():
        yol = f"{VERI}/{k}.jsonl"
        with open(yol, "w", encoding="utf-8") as f:
            for i in ii:
                i = int(i); r_ = ds[i]
                f.write(json.dumps(dict(
                    id=r_["id"], source=r_["source"], kol=k,
                    prompt=r_["prompt"], chosen=r_["chosen"],
                    rejected=r_["rejected"],
                    delta={kk: int(D[kk][i]) for kk in SAYAC},
                    n_jeton_chosen=int(njc[i]), n_jeton_rejected=int(njr[i])),
                    ensure_ascii=False) + "\n")
        KOL[k] = dict(
            yol=yol, n=len(ii), sha256=sha(yol),
            bayt=os.path.getsize(yol),
            kaynak_dagilimi={s_: int(c_) for s_, c_ in zip(
                *np.unique([KAYNAK[int(i)] for i in ii], return_counts=True))},
            boy_ort=round(float(boy[ii].mean()), 1),
            boy_p50=round(float(np.median(boy[ii])), 1),
            delta_dagilim={kk: dict(
                ort=round(float(D[kk][ii].mean()), 4),
                sd=round(float(D[kk][ii].std()), 4),
                sifir_orani=round(float((D[kk][ii] == 0).mean()), 4),
                p05=int(np.percentile(D[kk][ii], 5)),
                p50=int(np.percentile(D[kk][ii], 50)),
                p95=int(np.percentile(D[kk][ii], 95))) for kk in SAYAC},
            ornek=[dict(id=KIMLIK[int(i)], source=KAYNAK[int(i)],
                        chosen=C[int(i)][:400], rejected=R[int(i)][:400],
                        delta={kk: int(D[kk][int(i)]) for kk in SAYAC})
                   for i in ii[:a.ornek_cift]])
        print(f"  → {k}: n={len(ii):,} · boy_ort={KOL[k]['boy_ort']} · "
              f"sha={KOL[k]['sha256'][:16]}…", flush=True)

    O = dict(
        damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        SINIF="INSA + SAYIM (prereg YOK · bar YOK · egitim YOK · prediction YOK)",
        KURATOR_VERDICT=""
                      "",
        repo=REPO, n_toplam=len(ds), seed=SEED, n_process=a.n_process,
        klise_dosyasi=KLISE, klise_sha256=sha(KLISE),
        govde_sha256=sha(__file__),
        hijyen=dict(soyma_chosen=sc, soyma_rejected=sr,
                    n_soyulan_chosen=n_soy_c, n_soyulan_rejected=n_soy_r,
                    kapanis_klisesi_SOYULMADI=dict(chosen=kap_c, rejected=kap_r,
                                                   oran_chosen=round(kap_c / len(C), 4),
                                                   oran_rejected=round(kap_r / len(R), 4))),
        kapi=dict(metin_sn=round(hiz, 1), eta_dk=round(eta, 1),
                  esik_dk=ETA_ESIK_DK, ham_tarama_tam=bool(ham_tam)),
        HAM_vs_HIJYENLI=dict(
            n=len(ham_idx), tam=bool(ham_tam),
            kol_n_ham={k: int(v.sum()) for k, v in Y0.items()},
            kol_n_hijyenli={k: int(v.sum()) for k, v in Y_alt.items()},
            delta_ort_ham={k: round(float(D0[k].mean()), 4) for k in SAYAC},
            delta_ort_hijyenli={k: round(float(D[k][ham_idx].mean()), 4) for k in SAYAC}),
        kol_ham_n={k: int(len(v)) for k, v in idx.items()},
        SURUM="v2 (ORTAK TABAKA boy×kaynak)" if a.v2 else "v1 (yalniz boy)",
        v2_kutu=(a.v2_kutu if a.v2 else None),
        v2_tabaka={k: v for k, v in v2_bilgi.items()} if a.v2 else None,
        n_esit=int(n_ort), kontrol=kontrol,
        boy_denge=dict(fark_sd=round(fark, 4), esik_sd=BOY_ESIK, SERH=bool(boy_serh),
                       kol_log_boy_ort={k: round(v, 4) for k, v in ort.items()}),
        kol=KOL,
        saniye=round(time.time() - t00, 1))
    json.dump(O, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("uc_kol", n_kol=len(KOL), hal_n_esit=n_ort, dusen_ariza=dusen_ariza)
    print(f"\n→ {CIK} · {time.time()-t00:.0f} sn")
    return 3 if dusen_ariza else 0


if __name__ == "__main__":
    raise SystemExit(main())
