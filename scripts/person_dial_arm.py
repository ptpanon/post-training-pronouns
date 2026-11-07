#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, collections, hashlib, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from reading_style_object import payda
import form_count as FS
import filtre_verim as FV
import addressee_olcu as MO
import three_arm as UK

VERI = __DNH_DATA__ + "/dial_kol"
DOC = f"{ROOT}/results"
CIK = f"{DOC}/DIAL_KOL_2026-09-03.json"
KUVVET_KAYNAK = __DNH_DATA__ + "/uc_kol/EMIR_EKSI.jsonl"
SEED = 20260903
ETA_ESIK_DK = 60.0
BOY_ESIK = 0.10


def m1_1k(nlp, metinler, npr):
    A, _ = MO.topla(nlp, metinler, n_process=npr)
    nj = A["n_jeton"].astype(np.float64)
    return 1000.0 * A["m1_sahis2"] / np.maximum(nj, 1.0), nj.astype(np.int64)


def yuklem_kisi(dM1, gecerli):
    a = np.abs(dM1)
    med = float(np.median(a[gecerli])) if gecerli.any() else 0.0
    return (a >= med) & (a > 0) & gecerli, med


def prova():
    g = np.ones(6, bool)
    y0, m0 = yuklem_kisi(np.zeros(6), g)
    y1, m1 = yuklem_kisi(np.full(6, -3.0), g)
    d = np.array([-9., -8., -7., 0.5, 0.2, 0.1]); y2, m2 = yuklem_kisi(d, g)
    y3, _ = yuklem_kisi(np.full(6, 5.0), np.array([1,1,1,0,0,0], bool))
    R = {"i_fark_yok_bos": int(y0.sum()) == 0 and m0 == 0.0,
         "ii_sabit_hepsi": int(y1.sum()) == 6,
         "iii_yari_gecti": int(y2.sum()) == 3 and bool(y2[:3].all()),
         "iv_gecersiz_dislandi": int(y3.sum()) == 3}
    print(f"★ §7.1 DEJENERE PROVA (kosudan ÖNCE): {json.dumps(R)}")
    ok = all(R.values())
    print(f"{'GECTI' if ok else 'DÜSTÜ'}"
          f"", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-process", type=int, default=48)
    ap.add_argument("--kapi-ornek", type=int, default=2000)
    ap.add_argument("--n", type=int, default=0, help="0 = TAM SET")
    ap.add_argument("--prova-cik", action="store_true", help="yalniz provayi kos")
    ap.add_argument("--izole", action="store_true", help="§8: sahte-kosu izole dizine")
    a = ap.parse_args()
    global VERI, CIK
    if not prova():
        sys.exit(4)
    if a.prova_cik:
        return
    if a.izole:
        VERI = os.environ["SCRATCH"] + "/dial_kol_prova"; CIK = f"{VERI}/DIAL_KOL_PROVA.json"
        print("  ★ IZOLE KIP —", VERI, flush=True)
    os.makedirs(VERI, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t00 = time.time(); dusen_ariza = 0

    from datasets import load_dataset
    ds = load_dataset(UK.REPO, split="train")
    N = len(ds) if not a.n else min(a.n, len(ds))
    ds = ds.select(range(N))
    C = [FV._son_asistan_mesaj(x) for x in ds["chosen"]]
    R = [FV._son_asistan_mesaj(x) for x in ds["rejected"]]
    KAYNAK = ds["source"]; KIMLIK = ds["id"]
    ozdes = np.array([c == r for c, r in zip(C, R)])
    gecerli = np.array([bool(c) and bool(r) for c, r in zip(C, R)]) & ~ozdes
    payda("dial_cikar", n_toplam=len(ds), hal_gecerli=int(gecerli.sum()),
          red_bos_taraf=int(sum(1 for c, r in zip(C, R) if not c or not r)),
          red_ozdes=int(ozdes.sum()))

    KL = UK.klise_yukle()
    sc, sr = {}, {}
    Ch = [UK.hijyen(t, KL, sc)[0] for t in C]
    Rh = [UK.hijyen(t, KL, sr)[0] for t in R]
    payda("dial_hijyen", n_metin=2 * len(C), hal_soyulan_chosen=sum(sc.values()),
          hal_soyulan_rejected=sum(sr.values()))

    nlp = FS._boru()
    t0 = time.time(); m1_1k(nlp, Ch[:a.kapi_ornek], a.n_process)
    hiz = a.kapi_ornek / max(time.time() - t0, 1e-9)
    eta = (2 * len(Ch) / hiz) / 60
    payda("dial_kapi", n_olcum=a.kapi_ornek, hal_metin_sn=round(hiz, 1),
          hal_eta_dk=round(eta, 1), hal_esik_dk=ETA_ESIK_DK)
    print(f"{hiz:.0f} {a.n_process} {2*len(Ch):,}"
          f"{eta:.1f} {ETA_ESIK_DK}"
          f"", flush=True)
    if eta > ETA_ESIK_DK:
        print("  ★★ KAPI DÜSTÜ ⇒ cikis 5"); sys.exit(5)

    t1 = time.time()
    M1c, njc = m1_1k(nlp, Ch, a.n_process)
    M1r, njr = m1_1k(nlp, Rh, a.n_process)
    dM1 = M1c - M1r
    print(f"  ★ kisi taramasi bitti · {time.time()-t1:.0f} sn", flush=True)

    Y, med = yuklem_kisi(dM1, gecerli)
    hi_chosen = dM1 > 0
    idx_kisi = np.flatnonzero(Y)
    idx_rast = np.flatnonzero(gecerli)
    boy = (njc + njr) / 2.0
    payda("dial_ham_kol", hal_kisi=len(idx_kisi), hal_rastgele_havuz=len(idx_rast),
          hal_medyan_dM1=round(med, 4))
    print(f"{med:.3f} {len(idx_kisi):,}"
          f"", flush=True)

    KV = [json.loads(l) for l in open(KUVVET_KAYNAK, encoding="utf-8")]
    kv_boy = np.array([(r["n_jeton_chosen"] + r["n_jeton_rejected"]) / 2.0 for r in KV])
    n_kuvvet = len(KV)
    print(f"  ★ hedef histogram = KUVVET_ASAGI (diskteki EMIR_EKSI, n={n_kuvvet:,}) "
          f"⇒ dört kol TEK boy dagilimini paylasir", flush=True)
    lb_all = np.log1p(boy); lb_kv = np.log1p(kv_boy)
    kenar = np.quantile(lb_kv, np.linspace(0, 1, UK.N_BIN + 1))
    kenar[0] -= 1e-6; kenar[-1] += 1e-6
    pay = np.histogram(lb_kv, bins=kenar)[0].astype(float); pay /= pay.sum()

    def esle(ii, n=None):
        kutu = [ii[(lb_all[ii] >= kenar[i]) & (lb_all[ii] < kenar[i + 1])]
                for i in range(UK.N_BIN)]
        n_max = int(min((len(kutu[i]) / pay[i]) if pay[i] > 0 else np.inf
                        for i in range(UK.N_BIN)))
        if n is None:
            return None, n_max
        sec = [rng.choice(kutu[i], size=min(int(round(pay[i] * n)), len(kutu[i])),
                          replace=False)
               for i in range(UK.N_BIN) if int(round(pay[i] * n)) and len(kutu[i])]
        return (np.sort(np.concatenate(sec)) if sec else np.array([], int)), n_max

    tav_kisi = esle(idx_kisi)[1]; tav_rast = esle(idx_rast)[1]
    n_ort = int(min(tav_kisi, tav_rast, n_kuvvet, len(idx_kisi)))
    print(f"  ★ tavanlar: KISI {tav_kisi:,} · RASTGELE {tav_rast:,} · "
          f"KUVVET {n_kuvvet:,} ⇒ ORTAK N = {n_ort:,}", flush=True)
    s_kisi = esle(idx_kisi, n_ort)[0]; s_rast = esle(idx_rast, n_ort)[0]
    n_ort = int(min(len(s_kisi), len(s_rast), n_kuvvet))
    for nm, v in (("KISI", s_kisi), ("RASTGELE", s_rast)):
        pass
    if len(s_kisi) > n_ort:
        s_kisi = np.sort(rng.choice(s_kisi, size=n_ort, replace=False))
    if len(s_rast) > n_ort:
        s_rast = np.sort(rng.choice(s_rast, size=n_ort, replace=False))
    print(f"  ★ DÖRT KOL ESIT n = {n_ort:,}", flush=True)

    ort = {"KISI": float(lb_all[s_kisi].mean()), "RASTGELE": float(lb_all[s_rast].mean()),
           "KUVVET_ASAGI": float(lb_kv.mean())}
    sd_ = float(np.concatenate([lb_all[s_kisi], lb_all[s_rast], lb_kv]).std())
    fark = (max(ort.values()) - min(ort.values())) / max(sd_, 1e-9)
    payda("dial_boy_denge", n_kol=3, hal_fark_sd=round(fark, 4), hal_esik_sd=BOY_ESIK)
    print(f"{fark:.3f} {BOY_ESIK}"
          f""
          f"", flush=True)

    rast_yazi = rng.random(len(ds)) < 0.5
    KOL = {}

    def yaz(ad, ii, yon):
        yol = f"{VERI}/{ad}.jsonl"
        with open(yol, "w", encoding="utf-8") as f:
            for i in ii:
                i = int(i); r_ = ds[i]
                if yon == "rastgele":
                    swap = bool(rast_yazi[i])
                else:
                    swap = (hi_chosen[i] if yon == "asagi" else not hi_chosen[i])
                ch, rj = (r_["rejected"], r_["chosen"]) if swap else (r_["chosen"], r_["rejected"])
                njn_c, njn_r = (njr[i], njc[i]) if swap else (njc[i], njr[i])
                m1n_c, m1n_r = (M1r[i], M1c[i]) if swap else (M1c[i], M1r[i])
                f.write(json.dumps(dict(
                    id=r_["id"], source=r_["source"], kol=ad, yon=yon,
                    prompt=r_["prompt"], chosen=ch, rejected=rj,
                    takas=int(swap),
                    dM1_kol=round(float(m1n_c - m1n_r), 4),
                    M1_chosen=round(float(m1n_c), 4), M1_rejected=round(float(m1n_r), 4),
                    n_jeton_chosen=int(njn_c), n_jeton_rejected=int(njn_r)),
                    ensure_ascii=False) + "\n")
        d_ = np.array([(M1r[int(i)] - M1c[int(i)]) if
                       ((rast_yazi[int(i)]) if yon == "rastgele" else
                        (hi_chosen[int(i)] if yon == "asagi" else not hi_chosen[int(i)]))
                       else (M1c[int(i)] - M1r[int(i)]) for i in ii])
        KOL[ad] = dict(yol=yol, n=len(ii), sha256=UK.sha(yol), bayt=os.path.getsize(yol),
                       yon=yon,
                       dM1_kol_ort=round(float(d_.mean()), 4),
                       dM1_kol_sd=round(float(d_.std()), 4),
                       dM1_kol_p50=round(float(np.median(d_)), 4),
                       n_takas=int(sum(1 for i in ii if
                                       (rast_yazi[int(i)] if yon == "rastgele" else
                                        (hi_chosen[int(i)] if yon == "asagi"
                                         else not hi_chosen[int(i)])))),
                       boy_ort=round(float(boy[ii].mean()), 1),
                       kaynak_dagilimi={s_: int(c_) for s_, c_ in zip(
                           *np.unique([KAYNAK[int(i)] for i in ii], return_counts=True))})
        print(f"     {ad}: n={len(ii):,} · ΔM1(kol) ort {d_.mean():+.3f} · "
              f"sha {KOL[ad]['sha256'][:12]}…", flush=True)

    yaz("KISI_ASAGI", s_kisi, "asagi")
    yaz("KISI_YUKARI", s_kisi, "yukari")
    yaz("RASTGELE", s_rast, "rastgele")

    kv_yol = f"{VERI}/KUVVET_ASAGI.jsonl"
    if n_ort < n_kuvvet:
        pick = np.sort(rng.choice(n_kuvvet, size=n_ort, replace=False))
    else:
        pick = np.arange(n_kuvvet)
    with open(kv_yol, "w", encoding="utf-8") as f:
        for i in pick:
            f.write(json.dumps(KV[int(i)], ensure_ascii=False) + "\n")
    KOL["KUVVET_ASAGI"] = dict(yol=kv_yol, n=len(pick), sha256=UK.sha(kv_yol),
                               bayt=os.path.getsize(kv_yol), yon="emir_asagi",
                               kaynak="uc_kol/EMIR_EKSI.jsonl (NOTICE_UC_KOL mühürlü)",
                               kaynak_sha=UK.sha(KUVVET_KAYNAK),
                               tam_mi=int(len(pick) == n_kuvvet))
    print(f"     KUVVET_ASAGI: n={len(pick):,} · kaynak EMIR_EKSI "
          f"(sha {KOL['KUVVET_ASAGI']['kaynak_sha'][:12]}…) · "
          f"{'TAM' if len(pick)==n_kuvvet else 'ALT-ÖRNEK'}", flush=True)

    kontrol = dict(
        ozdes_kolda=int(sum(1 for i in np.concatenate([s_kisi, s_rast]) if ozdes[int(i)])),
        bos_taraf=int(sum(1 for i in np.concatenate([s_kisi, s_rast])
                          if not C[int(i)] or not R[int(i)])),
        yuklem_yeniden=int(bool(Y[s_kisi].all())),
        ayni_cift_iki_yon=int(np.array_equal(s_kisi, s_kisi)),
        cakisma_kisi_rastgele=int(len(set(s_kisi.tolist()) & set(s_rast.tolist()))),
        n_essiz_kol=len({KOL[k]["sha256"] for k in KOL}))
    kontrol["asagi_yukari_sha_farkli"] = int(
        KOL["KISI_ASAGI"]["sha256"] != KOL["KISI_YUKARI"]["sha256"])
    payda("dial_kontrol", n_satir=int(len(s_kisi) + len(s_rast)),
          red_ozdes=kontrol["ozdes_kolda"], red_bos=kontrol["bos_taraf"],
          hal_yuklem_yeniden=kontrol["yuklem_yeniden"],
          hal_cakisma_kisi_rastgele=kontrol["cakisma_kisi_rastgele"],
          hal_sha_farkli=kontrol["asagi_yukari_sha_farkli"])
    if kontrol["ozdes_kolda"] or kontrol["bos_taraf"] or not kontrol["yuklem_yeniden"] \
       or not kontrol["asagi_yukari_sha_farkli"] or kontrol["n_essiz_kol"] != 4:
        dusen_ariza += 1
        print("  ★★ ARIZA: §8 kontrolleri düstü", flush=True)

    OUT = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               sinif="INSA+SAYIM · EGITIM YOK · PREREG YOK",
               havuz=UK.REPO, n_havuz=len(ds), seed=SEED,
               medyan_dM1=round(med, 4), n_ort=n_ort,
               tavan=dict(kisi=int(tav_kisi), rastgele=int(tav_rast), kuvvet=n_kuvvet),
               boy_denge_sd=round(fark, 4), boy_serh=int(fark > BOY_ESIK),
               kol=KOL, kontrol=kontrol,
               sure_sn=round(time.time() - t00, 1))
    os.makedirs(os.path.dirname(CIK), exist_ok=True)
    json.dump(OUT, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}", flush=True)
    payda("dial_kol", n_kol=len(KOL), hal_n_ort=n_ort, red_dusen_ariza=dusen_ariza)
    sys.exit(3 if dusen_ariza else 0)


if __name__ == "__main__":
    main()
