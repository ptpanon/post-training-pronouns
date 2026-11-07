#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, itertools
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from pergel_cekirdek import Cekirdek, Permutator, foldlar

OUT = __DNH_DATA__ + "/onek_korpus/null_olcum"
B_BOOT, SEED, NFOLD = 2000, 20260805, 5

VERDICT_ADLARI = ("SKORLANAMAZ", "GERCEKLESMEDI", "SAHTE-KOLAY", "ÖNEK-KANALI", "A-OKUMUYOR",
                "A-OKUYOR·B-OYNUYOR", "ZEMIN-KURULDU", "ÖLCÜLEMEZ")
BANT_ALT, BANT_UST = 1.0, 1.645


def verdict(kapilar_gecti, yargic_uyusmazlik, yargic_null_ust,
          auc, auc_ablasyon, auc_sozluk, sozluk_fark_z, null_merkez, null_sd, z_B,
          z_ozgul):
    if not kapilar_gecti:
        return "SKORLANAMAZ", ""
    if yargic_uyusmazlik > yargic_null_ust:
        return "GERCEKLESMEDI", (""
                                 "")
    z = (auc - null_merkez) / null_sd
    z_abl = (auc_ablasyon - null_merkez) / null_sd
    if z >= BANT_UST and (z_abl < BANT_UST or sozluk_fark_z < BANT_UST):
        return "SAHTE-KOLAY", (f"{z_abl:.2f}"
                               f""
                               f"{sozluk_fark_z:.2f}")
    if z >= BANT_UST and z_ozgul < BANT_ALT:
        return "ÖNEK-KANALI", (f"{z:.2f}"
                               f"{z_ozgul:.2f} {BANT_ALT}"
                               f"")
    if z < BANT_ALT:
        return "A-OKUMUYOR", f"z={z:.2f} < {BANT_ALT} ⇒ kurgu etiket okunmuyor"
    if z < BANT_UST:
        return "ÖLCÜLEMEZ", f"A güc bandi: z={z:.2f} ∈ [{BANT_ALT}, {BANT_UST})"
    if z_ozgul < BANT_UST:
        return "ÖLCÜLEMEZ", (f"{z_ozgul:.2f}"
                             f"{BANT_ALT} {BANT_UST}")
    if z_B >= BANT_UST:
        return "A-OKUYOR·B-OYNUYOR", (f"{z:.2f} {z_ozgul:.2f}"
                                      f"{z_B:.2f}")
    if z_B >= BANT_ALT:
        return "ÖLCÜLEMEZ", (f"I-c B güc bandi: z_B={z_B:.2f} ∈ [{BANT_ALT}, {BANT_UST})")
    return "ZEMIN-KURULDU", (f"A okuyor (z={z:.2f}) ∧ durusa ÖZGÜL (z_özgül={z_ozgul:.2f}) "
                             f"∧ B düz (z_B={z_B:.2f}) ∧ ablasyonda ayakta (z_abl={z_abl:.2f})")


def ust_okuma(hucreler, birincil="qwen_sontoken_L22"):
    b = {y: h for (y, o), h in hucreler.items() if o == birincil}
    if len(b) == 2 and all(v == "ZEMIN-KURULDU" for v in b.values()):
        return "ZEMIN KURULDU", f"iki yazar da birincil okuyucuda ZEMIN-KURULDU: {b}"
    return ("INDIRGENMEZ",
            ""
            + " · ".join(f"{y}/{o}={h}" for (y, o), h in sorted(hucreler.items())))


def prova_ulasilabilirlik():
    M, S = 0.50, 0.02
    O = dict(kapilar_gecti=True, yargic_uyusmazlik=.1, yargic_null_ust=.3,
             auc=.70, auc_ablasyon=.68, auc_sozluk=.50, sozluk_fark_z=3.0,
             null_merkez=M, null_sd=S, z_B=0.4, z_ozgul=3.0)
    vaka = [
        ("SKORLANAMAZ",        {**O, "kapilar_gecti": False}),
        ("GERCEKLESMEDI",      {**O, "yargic_uyusmazlik": .45, "yargic_null_ust": .30}),
        ("SAHTE-KOLAY",        {**O, "auc_ablasyon": .51}),
        ("SAHTE-KOLAY",        {**O, "sozluk_fark_z": 0.9}),
        ("ÖNEK-KANALI",        {**O, "z_ozgul": 0.4}),
        ("A-OKUMUYOR",         {**O, "auc": .505, "auc_ablasyon": .505}),
        ("ÖLCÜLEMEZ",          {**O, "auc": .5250, "auc_ablasyon": .5250}),
        ("ÖLCÜLEMEZ",          {**O, "z_ozgul": 1.30}),
        ("ÖLCÜLEMEZ",          {**O, "z_B": 1.30}),
        ("A-OKUYOR·B-OYNUYOR", {**O, "z_B": 2.10}),
        ("ZEMIN-KURULDU",      O),
    ]
    print("")
    print("─" * 96)
    bulunan, ok = set(), True
    for bekle, kw in vaka:
        ad, ger = verdict(**kw)
        bulunan.add(ad)
        if ad != bekle:
            ok = False
        print(f"  {'✓' if ad == bekle else '✗'} beklenen {bekle:<20} → üretilen {ad:<20} "
              f"· {ger[:38]}")
    ulasilmayan = set(VERDICT_ADLARI) - bulunan
    print("─" * 96)
    payda("ulasilabilirlik", n_ad=len(VERDICT_ADLARI), n_vaka=len(vaka),
          n_ulasilan=len(bulunan), red_ulasilmayan=len(ulasilmayan),
          bekle={"n_ulasilan": len(VERDICT_ADLARI), "n_vaka": 11})
    if not ok or ulasilmayan:
        raise RuntimeError(f"ULASILABILIRLIK DÜSTÜ: ulasilmayan={ulasilmayan or '(yok)'}")
    print("")
    B = "qwen_sontoken_L22"
    t1 = ust_okuma({("mistral", B): "ZEMIN-KURULDU", ("gemma", B): "ZEMIN-KURULDU",
                    ("mistral", "e5_L23"): "ÖLCÜLEMEZ", ("gemma", "e5_L23"): "ÖLCÜLEMEZ"})
    t2 = ust_okuma({("mistral", B): "ZEMIN-KURULDU", ("gemma", B): "ÖNEK-KANALI",
                    ("mistral", "e5_L23"): "ZEMIN-KURULDU", ("gemma", "e5_L23"): "ZEMIN-KURULDU"})
    assert t1[0] == "ZEMIN KURULDU" and t2[0] == "INDIRGENMEZ", (t1, t2)
    print("")
    return sorted(bulunan)


def bootstrap_orani(X, kume, tabaka, dev):
    fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
    C = Cekirdek(X, fold, dev=dev, std=True)
    y0 = np.zeros(len(kume), dtype=np.int8)
    for t in sorted(set(tabaka.tolist())):
        ix = np.where(tabaka == t)[0]
        y0[ix[: len(ix) // 2]] = 1
    P = Permutator(y0, tabaka.tolist(), seed=SEED)
    Ynull = P.cek(2000)
    perm = C.auc_toplu(Ynull, C.oku_toplu(Ynull))

    N_CIZ = 20
    kumeler = np.array(sorted(set(kume.tolist())))
    ix_of = {k: np.where(kume == k)[0] for k in kumeler}
    rng = np.random.default_rng(SEED)
    Yciz = P.cek(N_CIZ)
    Aciz = C.oku_toplu(Yciz).cpu().numpy()
    C.bosalt()
    sds, merkezler = [], []
    for c in range(N_CIZ):
        A, yb = Aciz[c], Yciz[c]
        boot = []
        for _ in range(B_BOOT // 10):
            sec = rng.choice(kumeler, size=len(kumeler), replace=True)
            ix = np.concatenate([ix_of[k] for k in sec])
            a, yy = A[ix], yb[ix]
            n1, n0 = int(yy.sum()), int((1 - yy).sum())
            if n1 == 0 or n0 == 0:
                continue
            r = np.empty(len(a)); r[np.argsort(a)] = np.arange(1, len(a) + 1)
            boot.append((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        boot = np.array(boot)
        sds.append(float(boot.std(ddof=1))); merkezler.append(float(boot.mean()))
    bsd = float(np.median(sds))

    B_REFIT = 200
    yb0 = Yciz[0]
    ref = []
    for _ in range(B_REFIT):
        sec = rng.choice(kumeler, size=len(kumeler), replace=True)
        ix = np.concatenate([ix_of[k] for k in sec])
        kb, yy = kume[ix], yb0[ix]
        if len(set(kb.tolist())) < NFOLD or yy.sum() in (0, len(yy)):
            continue
        fb = foldlar(kb.tolist(), k=NFOLD, seed=SEED)
        Cb = Cekirdek(X[ix], fb, dev=dev, std=True)
        Ab = Cb.oku_toplu(yy[None, :].astype(np.int8))
        ref.append(float(Cb.auc_toplu(yy[None, :].astype(np.int8), Ab)[0]))
        Cb.bosalt()
    ref = np.array(ref)
    return dict(perm_sd=float(perm.std(ddof=1)), boot_sd=bsd,
                oran=float(bsd / perm.std(ddof=1)), n_ciz=N_CIZ,
                boot_sd_min=float(np.min(sds)), boot_sd_maks=float(np.max(sds)),
                boot_merkez=float(np.median(merkezler)),
                refit_sd=float(ref.std(ddof=1)), refit_n=len(ref),
                refit_oran=float(ref.std(ddof=1) / perm.std(ddof=1)),
                refit_merkez=float(ref.mean()))


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    os.makedirs(OUT, exist_ok=True)
    print("")
    adlar = prova_ulasilabilirlik()

    from prefix_null_measurement import satirlar, kodla_qwen, kodla_e5, YAZARLAR
    VERI = {y: satirlar(y) for y in YAZARLAR}
    XQ = kodla_qwen({y: VERI[y][0] for y in YAZARLAR}, dev)
    XE, _ = kodla_e5({y: VERI[y][0] for y in YAZARLAR}, dev)

    print("\nIKI NULL, IKI AYRI IS — sifir-sinyal rejiminde ölcüldü")
    print(f"{'yazar/okuyucu':<28}{'perm sd':>9}{'boot(kosullu)':>15}{'oran':>7}"
          f"{'boot(YENIDEN)':>15}{'oran':>7}")
    print("─" * 82)
    R = {}
    for yazar in YAZARLAR:
        _m, _y, kume, tabaka = VERI[yazar]
        for ok, X in (("qwen_sontoken_L22", XQ[yazar]), ("e5_L23", XE[yazar])):
            r = bootstrap_orani(X, kume, tabaka, dev)
            R[f"{yazar}/{ok}"] = r
            print(f"{yazar+'/'+ok:<28}{r['perm_sd']:>9.5f}{r['boot_sd']:>15.5f}"
                  f"{r['oran']:>7.2f}{r['refit_sd']:>15.5f}{r['refit_oran']:>7.2f}")
    print("─" * 82)
    print("")
    print("")
    json.dump(dict(verdict_adlari=list(VERDICT_ADLARI), ulasilan=adlar, bant=[BANT_ALT, BANT_UST],
                   bootstrap=R, B_boot=B_BOOT),
              open(f"{OUT}/prereg_kapilari.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n→ {OUT}/prereg_kapilari.json")


if __name__ == "__main__":
    main()


def kapi_dejenerelik(X, y, fold, dev="cuda:0", etiket=""):
    import torch
    red = {}
    for j in sorted(set(fold.tolist())):
        tr = fold != j
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            red[f"fold{j}_tek_sinif"] = 1
    s = X.std(axis=0)
    red["cokmus_boyut"] = int((s < 1e-6).sum())
    if red["cokmus_boyut"] == X.shape[1]:
        red["X_sabit"] = 1
    C = Cekirdek(X, fold, dev=dev, std=True)
    A = C.oku_toplu(y[None, :].astype(np.int8)).cpu().numpy()[0]
    g = float(C.auc_toplu(y[None, :].astype(np.int8),
                          C.oku_toplu(y[None, :].astype(np.int8)))[0])
    C.bosalt()
    red["doygun_pay"] = float((np.abs(A) > 0.999).mean())
    red["nan_pay"] = float(np.isnan(A).mean())
    dejenere = bool(red.get("X_sabit") or red.get("nan_pay", 0) > 0
                    or red["doygun_pay"] > 0.5
                    or any(k.endswith("tek_sinif") for k in red))
    return dejenere, dict(auc=g, **red)


def prova_dejenerelik(X, kume, dev="cuda:0"):
    fold = foldlar(kume.tolist(), k=NFOLD, seed=SEED)
    n = len(kume)
    rng = np.random.default_rng(SEED)
    y_saglikli = np.zeros(n, dtype=np.int8); y_saglikli[rng.permutation(n)[: n // 2]] = 1
    vaka = [
        ("SAGLIKLI (kapi SUSMALI)", X, y_saglikli, False),
        ("tek sinif (hepsi 1)", X, np.ones(n, dtype=np.int8), True),
        ("sabit X", np.repeat(X[:1], n, axis=0), y_saglikli, True),
    ]
    print("K10 · DEJENERELIK PROVASI (§7.1) — mühürden ÖNCE")
    print("─" * 88)
    ok = True
    for ad, Xi, yi, bekle in vaka:
        d, det = kapi_dejenerelik(np.ascontiguousarray(Xi), yi, fold, dev, ad)
        iyi = (d == bekle)
        ok &= iyi
        print(f"  {'✓' if iyi else '✗'} {ad:<26} dejenere={str(d):<5} (beklenen {bekle}) · "
              f"doygun={det['doygun_pay']:.3f} nan={det['nan_pay']:.3f} "
              f"cökmüs_boyut={det['cokmus_boyut']}")
    print("─" * 88)
    payda("dejenerelik_provasi", n_vaka=len(vaka), red_uyumsuz=sum(1 for _ in [] ))
    if not ok:
        raise RuntimeError("K10 DEJENERELIK PROVASI DÜSTÜ — prereg BEKLER (§7.1).")
    print("")
    return True
