#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, random, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")

REPO = "allenai/Dolci-Think-DPO-7B"
CIK = os.environ.get("V62_CIK", f"{ROOT}/results/dolci_person_delta_2026-09-12.json")
ETA_ESIK_DK = float(os.environ.get("V62_ETA_ESIK_DK", "90"))
DUR = __DNH_DATA__ + "/DUR_V61"
N_ORNEK = int(os.environ.get("V62_N", "20000"))
NPROC = int(os.environ.get("V62_NPROC", "6"))


def indir():
    from huggingface_hub import snapshot_download
    t0 = time.time()
    y = snapshot_download(REPO, repo_type="dataset")
    mb = sum(os.path.getsize(os.path.join(r, f))
             for r, _, fs in os.walk(y) for f in fs) / 1e6
    print(f"  [INDIRME] {REPO} ⇒ {y} · {mb:.0f} MB · {time.time()-t0:.0f} sn "
          f"⇒ esik: disk %94 ⇒ EYLEM: asarsa kosmaz", flush=True)
    return y, mb


def sema_probu():
    from datasets import load_dataset
    ds = load_dataset(REPO, split="train")
    ad = list(ds.features)
    ornek = ds[0]
    if "chosen" in ad and "rejected" in ad:
        c = ornek["chosen"]
        cik = "mesaj" if isinstance(c, list) else "duz"
    else:
        cik = None
    print(f"  [SEMA] alanlar={ad[:8]} · chosen tipi={type(ornek.get('chosen')).__name__} "
          f"⇒ cikarici={cik} ⇒ esik: None ⇒ EYLEM: ÖLCÜM YAZILMAZ", flush=True)
    return ds, cik, len(ds)


def main():
    if os.path.exists(DUR):
        print("★★ DUR ⇒ kosmaz"); return 0
    yol, mb = indir()
    ds, cik, n_havuz = sema_probu()
    if cik is None:
        print(""); return 4
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import filtre_verim as FV
    import addressee_olcu as MO
    import form_count as FS
    import pool_person_delta as HKD
    from reading_style_object import payda
    S = dict(ad="dolci-think-dpo", repo=REPO, split="train", cikarici=cik)
    FV.SETLER.append(S)
    nlp = FS._boru()
    rng = random.Random(FV.SEED)
    C, R, N, red = FV.yukle(S, N_ORNEK, rng)
    MAXL = int(os.environ.get("V62_MAXL", str(getattr(nlp, "max_length", 1_000_000))))
    _tut = [i for i in range(len(C)) if len(C[i]) <= MAXL and len(R[i]) <= MAXL]
    _atl = len(C) - len(_tut)
    payda("v62_uzunluk", n_cift=len(C), atl_uzun_cift=_atl,
          hal_maks_kar=max(max(map(len, C)), max(map(len, R))), hal_sinir=MAXL,
          hal_oran=round(_atl / max(len(C), 1), 6))
    print(f"  [KAPI/W-1096] aletin siniri {MAXL} kar · asan CIFT {_atl}/{len(C)} "
          f"⇒ esik: atl_uzun_cift/n_cift > 0,01 ⇒ EYLEM: atlama DEGIL, ölcü aleti "
          f"sorgulanir (payda kartta yazili)", flush=True)
    if _atl / max(len(C), 1) > 0.01:
        raise SystemExit(f"ATLAMA PAYI COK BÜYÜK: {_atl}/{len(C)} — alet sorgulanir")
    C = [C[i] for i in _tut]; R = [R[i] for i in _tut]
    n = len(C)
    t0 = time.time(); MO.topla(nlp, C[:500], n_process=NPROC); dt = time.time() - t0
    hiz = 500 / max(dt, 1e-9); eta = (2 * n / hiz) / 60
    print(f"{hiz:.0f} {NPROC}"
          f"{eta:.1f}"
          f"{ETA_ESIK_DK:.0f}"
          f"", flush=True)
    def _butce_dus(e, kaynak):
        json.dump(dict(HAL="ATLANDI-BÜTCE", eta_dk=round(e, 1), n_cift=n,
                       n_havuz=n_havuz, sinif="KESIFSEL", eta_kaynagi=kaynak,
                       esik_dk=ETA_ESIK_DK),
                  io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if eta > 10 * ETA_ESIK_DK:
        print(f"{eta:.0f} {ETA_ESIK_DK:.0f}"
              f"", flush=True)
        _butce_dus(eta, "isinma-vekili (10× felaket kapisi)")
        return 0
    DILIM = int(os.environ.get("V62_DILIM", "10000"))
    METIN = C + R
    A, n_dilim, kum, t_bas = None, 0, 0, time.time()
    for _b in range(0, len(METIN), DILIM):
        _d = METIN[_b:_b + DILIM]
        _ta = time.time()
        _Ad, _ = MO.topla(nlp, _d, n_process=NPROC)
        n_dilim += 1; kum += len(_d)
        payda("v62_dilim", n_dilim=n_dilim, n_metin=len(_d),
              hal_sn=round(time.time() - _ta, 1), hal_kumulatif=kum,
              hal_toplam=len(METIN),
              hal_metin_sn=round(len(_d) / max(time.time() - _ta, 1e-9), 1),
              n_process=NPROC)
        A = _Ad if A is None else {k: np.concatenate([A[k], _Ad[k]]) for k in A}
        if n_dilim == 1:
            _hd = len(_d) / max(time.time() - _ta, 1e-9)
            _eta = (len(METIN) / _hd) / 60
            print(f"  [KAPI/W-1097] ★ DILIM HIZI {_hd:.1f} metin/sn ⇒ ETA {_eta:.1f} dk "
                  f"⇒ esik {ETA_ESIK_DK:.0f} dk ⇒ EYLEM: asarsa DURUR "
                  f"(vekil {hiz:.0f} metin/sn demisti ⇒ yanlilik {_hd/max(hiz,1e-9):.2f}×)",
                  flush=True)
            if _eta > ETA_ESIK_DK:
                _butce_dus(_eta, "ilk dilim (ölcülmüs, vekil DEGIL)")
                return 0
    print(f"{n_dilim} {kum} {len(METIN)}"
          f"{(time.time() - t_bas) / 60:.1f}"
          f"", flush=True)
    if kum != len(METIN):
        raise SystemExit(f"DILIM PAYDASI TUTMADI: {kum} ≠ {len(METIN)}")
    c2, c1, cj = HKD.bin_jetonda(A, 0, n)
    r2, r1, rj = HKD.bin_jetonda(A, n, 2 * n)
    d2, d1 = c2 - r2, c1 - r1
    ci2 = HKD.boot(d2); ci1 = HKD.boot(d1)
    def yon(ci, d):
        if ci[0] <= 0 <= ci[1]:
            return "YÖN-YOK"
        return "DESTEKLER (secilen daha az kisi)" if d.mean() < 0 else ""
    H2, H1 = yon(ci2, d2), yon(ci1, d1)
    payda("v62_dolci_kisi", n_havuz=n_havuz, n_cift=n, red_bos=red,
          hal_d2=round(float(d2.mean()), 4), hal_d1=round(float(d1.mean()), 4))
    print(f"  ★ ikinci sahis Δ={d2.mean():+.4f}/1k CI={[round(x,4) for x in ci2]} ⇒ {H2}")
    print(f"  ★ birinci sahis Δ={d1.mean():+.4f}/1k CI={[round(x,4) for x in ci1]} ⇒ {H1}")
    json.dump(dict(
        _kunye=dict(alet="scripts/v62_dolci_person.py",
                    damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    sinif=""
                          "",
                    repo=REPO, yol=yol, mb=round(mb), cikarici=cik,
                    n_process=NPROC, sinir_kar=MAXL, atl_uzun_cift=_atl,
                    atlama_serhi=("★ aletin sinirini (spaCy nlp.max_length) asan "
                                  "CIFTLER düstü; ölcüm esli oldugu icin atlama "
                                  "cift bazindadir. W-1096."),
                    seed=FV.SEED, n_havuz=n_havuz, n_cift=n),
        ikinci_sahis=dict(delta=float(d2.mean()), ci=[float(x) for x in ci2], okuma=H2),
        birinci_sahis=dict(delta=float(d1.mean()), ci=[float(x) for x in ci1], okuma=H1),
        jeton=dict(secilen_ort=float(cj.mean()), reddedilen_ort=float(rj.mean()),
                   secilen_medyan=float(np.median(cj)),
                   reddedilen_medyan=float(np.median(rj))),
        sinir=""),
        io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ {CIK}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
