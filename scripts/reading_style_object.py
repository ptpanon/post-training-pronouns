#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, sys, glob
import numpy as np

ROOT = __DNH_ROOT__ + ""
MNT = __DNH_DATA__ + ""
OT = f"{MNT}/okuma_tarzi"
BANKA = f"{MNT}/z5_melek"
CIKTI = f"{ROOT}/unreleased/line"
sys.path.insert(0, f"{ROOT}/scripts")

from reading_style import TARZLAR, tarz_hesapla

GOVDE_D = {"K2": 3584, "K4": 4096, "c2m": 1024, "st5-xl": 1024}
ZEMINLER = ("dis", "kia")
BEKLENEN = {"son_token": "delta", "duz_ortalama": "delta", "son_bolge": "delta",
            "orta_bolge": "delta", "maks_havuz": "delta", "kuantil_demeti": "delta",
            "sinksiz_ortalama": "delta_yaklasik", "std_havuz": "sifir"}


ONEK_KANON = ("n_", "red_", "atl_", "hal_", "bek_")
_MIRAS_YOL = __DNH_ROOT__ + "/results/payda_miras_alanlar.json"
try:
    import json as _json
    ONEK_MIRAS_ADLAR = frozenset(_json.load(open(_MIRAS_YOL, encoding="utf-8"))["alanlar"])
except Exception:
    ONEK_MIRAS_ADLAR = frozenset()
_MIRAS_BASILDI = set()


def _onek_lint(etiket, kw):
    for k in kw:
        if k.startswith(ONEK_KANON):
            continue
        if k in ONEK_MIRAS_ADLAR:
            if k not in _MIRAS_BASILDI:
                _MIRAS_BASILDI.add(k)
                print(f"{etiket} {k}"
                      f"")
            continue
        raise RuntimeError(
            f"{etiket} {k}"
            f"{' · '.join(ONEK_KANON)}"
            f""
            f"")


def payda(etiket, bekle=None, **kw):
    s = " · ".join(f"{k}={v}" for k, v in kw.items())
    if bekle:
        s += " · [bekle] " + " · ".join(f"{k}≥{v}" for k, v in bekle.items())
    print(f"  [PAYDA] {etiket}: {s}")
    _onek_lint(etiket, kw)
    _atl = sum(v for k, v in kw.items() if k.startswith("atl_") and isinstance(v, int))
    _kapsam = [v for k, v in kw.items() if k.startswith("n_") and isinstance(v, int)]
    _tavan = max(_kapsam) if _kapsam else 0
    _zaten_tam = bool(_kapsam) and _atl > 0 and _atl >= _tavan
    if _zaten_tam:
        print(f"{etiket} {_atl} {_tavan}"
              f"")
        return "ZATEN-TAM"

    for k, v in kw.items():
        if k.startswith("n_") and v == 0:
            raise RuntimeError(f"KAPSAM SIFIR: {etiket}.{k}=0 ⇒ PASS DEGIL, HATA.")
    for k, b in (bekle or {}).items():
        if k not in kw:
            raise RuntimeError(f"{etiket} {k}")
        if kw[k] < b:
            raise RuntimeError(f"{etiket} {k} {kw[k]} {b}"
                               f"")


ADLAR_OLCUM_YERI = ("YER VAR", "ÖLCÜM-YERI-YOK")


def olcum_yeri(capa, sinir, mde, ad="", pay=None, yon=+1):
    if yon not in (+1, -1):
        raise ValueError(f"{yon!r}")
    oda = float(yon) * (float(sinir) - float(capa))
    mde = float(mde)
    yer = bool(np.isfinite(oda) and np.isfinite(mde) and round(oda, 6) >= round(mde, 6))
    r = dict(ad=ad, capa=float(capa), sinir=float(sinir), yon=int(yon),
             oda=round(oda, 6), mde=round(mde, 6), yer=yer,
             verdict=ADLAR_OLCUM_YERI[0 if yer else 1])
    if pay is not None:
        pay.append(r)
    return r


def _prova_olcum_yeri():
    V = [("bol yer", (0.05, 0.50, 0.10, +1), True),
         ("tam sinir", (0.05, 0.15, 0.10, +1), True),
         ("kil payi az", (0.05, 0.1499, 0.10, +1), False),
         ("oda yok", (0.0667, 0.0342, 0.1680, +1), False),
         ("oda negatif", (0.50, 0.10, 0.01, +1), False),
         ("MDE sonsuz", (0.0, 1.0, float("inf"), +1), False),
         ("NaN oda", (float("nan"), 1.0, 0.1, +1), False),
         ("inis · yer var", (0.0625, 0.0, 0.0403, -1), True),
         ("inis · yer yok", (0.0625, 0.0, 0.1310, -1), False),
         ("inis yönü +1 ile", (0.0625, 0.0, 0.0403, +1), False)]
    ok = 0
    for ad, (c, s, m, y), bekle in V:
        r = olcum_yeri(c, s, m, ad=ad, yon=y)
        ok += r["yer"] == bekle
        print(f"  prova {ad:18s} yön{r['yon']:+d} oda={r['oda']:+.4f} "
              f"mde={r['mde']:.4f} → {r['verdict']}")
    assert ok == len(V), "★ D-26 KAPI PROVASI DÜSTÜ"
    kodda = {ADLAR_OLCUM_YERI[0], ADLAR_OLCUM_YERI[1]}
    assert {r["verdict"] for r in (olcum_yeri(0, 1, 0.1), olcum_yeri(0, 0, 1))} == kodda
    print(f"  ✓ D-26 kapi provasi {ok}/{len(V)} · ad kümesi {sorted(kodda)}")


def kapi():
    import torch
    print("═" * 88); print("K0.G · KAPI-PROVASI (dejenere girdi)"); print("═" * 88)
    torch.manual_seed(20260801)
    B, S, D = 6, 40, 64
    hl = torch.randn(B, S, D, dtype=torch.float32)
    Sg = torch.tensor([40, 33, 21, 12, 5, 1])
    valid = torch.arange(S)[None] < Sg[:, None]

    def hesapla(h):
        return tarz_hesapla(h, valid, Sg, __import__("collections").Counter())

    A = hesapla(hl)

    A0 = hesapla(hl.clone())
    kotu = [t for t in TARZLAR if not torch.allclose(A[t], A0[t], atol=0, rtol=0)]
    print(f"  (1) δ=0 determinizm: {'TAMAM' if not kotu else 'DÜSTÜ ' + str(kotu)}")
    if kotu:
        raise RuntimeError("KAPI DÜSTÜ: tarz_hesapla deterministik degil.")

    delta = torch.randn(D) * 0.30
    Ad = hesapla(hl + delta[None, None, :])
    sat = {}
    for t in TARZLAR:
        d_gozlenen = Ad[t] - A[t]
        hedef = delta.repeat(3) if t == "kuantil_demeti" else delta
        hedef = hedef[None, :].expand_as(d_gozlenen)
        sapma_delta = (d_gozlenen - hedef).abs().max().item()
        sapma_sifir = d_gozlenen.abs().max().item()
        bek = BEKLENEN[t]
        if bek == "delta":
            ok, olcu = sapma_delta < 1e-4, f"maks|Δ−δ|={sapma_delta:.2e}"
        elif bek == "sifir":
            ok, olcu = sapma_sifir < 1e-4, f"maks|Δ|={sapma_sifir:.2e}"
        else:
            ok, olcu = sapma_delta < 5e-2, f"maks|Δ−δ|={sapma_delta:.2e} (yaklasik)"
        sat[t] = (bek, olcu, ok)
        print(f"      {t:<20} beklenen={bek:<14} {olcu:<28} {'TAMAM' if ok else '★DÜSTÜ'}")
    dusen = [t for t, v in sat.items() if not v[2]]
    if dusen:
        raise RuntimeError(f"KAPI DÜSTÜ: {dusen} — protokol §0/D1 varsayimi bu veride tutmuyor.")

    C = torch.nn.functional.normalize(torch.randn(200, D), dim=1)
    Cb = torch.nn.functional.normalize(C.mean(0), dim=0)
    for ad, u in (("sabit-birler", torch.ones(D)), ("C̄'nin kendisi", Cb),
                  ("rastgele", torch.randn(D))):
        tau = (u - Cb * (Cb @ u)).norm().item() / u.norm().item()
        print(f"  (3) teget payi τ [{ad:<14}] = {tau:.6f}")
    tau_par = (Cb - Cb * (Cb @ Cb)).norm().item() / Cb.norm().item()
    if tau_par > 1e-5:
        raise RuntimeError("KAPI DÜSTÜ: C̄'ye paralel yönün teget payi 0 olmaliydi.")
    print("  (3) dejenere kontrol: C̄'ye paralel yön τ=0 ⇒ kestirici mekanik degil.")
    payda("kapi", n_tarz=len(TARZLAR), n_ornek=B, n_kontrol=3)
    print("KAPI TAMAM ⇒ pahali is_ serbest.\n")
    return True


def _l2(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


def geometri(bolme="train", n_maks=4000):
    print("═" * 88); print("K0.N + K0.S · teget payi ve isaret tutarliligi"); print("═" * 88)
    fs = sorted(glob.glob(f"{BANKA}/*.w.npy"))
    payda("banka", n_dosya=len(fs), beklenen=64)
    if len(fs) != 64:
        raise RuntimeError(f"BANKA PAYDASI: {len(fs)} ≠ 64 — beklenenden farkli, kosu durur.")

    satir, atlanan = [], []
    for f in fs:
        ad = os.path.basename(f)[:-6]
        kk, z, tarz = ad.split("__")
        d = GOVDE_D[kk]
        W = np.load(f).astype(np.float32)
        blok = W.shape[1] // 3
        if W.shape[1] % 3 or (tarz == "kuantil_demeti" and blok != 3 * d) or \
           (tarz != "kuantil_demeti" and blok != d):
            raise RuntimeError(f"BLOK UYUMSUZ: {ad} sütun={W.shape[1]} d={d} — HATA, atlama yok.")
        wP, wC, wA = W[:, :blok], W[:, blok:2 * blok], W[:, 2 * blok:]

        yol = f"{OT}/{kk}__{z}__{bolme}__cevap__{tarz}.npy"
        yolP = f"{OT}/{kk}__{z}__{bolme}__ebeveyn__{tarz}.npy"
        if not (os.path.exists(yol) and os.path.exists(yolP)):
            atlanan.append((ad, "dizi yok")); continue
        Cm = np.load(yol, mmap_mode="r")
        Pm = np.load(yolP, mmap_mode="r")
        n = min(Cm.shape[0], n_maks)
        if Cm.shape[1] != W.shape[0]:
            atlanan.append((ad, f"katman {Cm.shape[1]}≠{W.shape[0]}")); continue

        for L in range(W.shape[0]):
            C = _l2(np.asarray(Cm[:n, L], np.float32))
            P = _l2(np.asarray(Pm[:n, L], np.float32))
            Cb = _l2(C.mean(0))
            s = np.sign(P - C); s[s == 0] = 1.0
            sbar = np.sign(s.mean(0)); sbar[sbar == 0] = 1.0
            m = float(np.abs(s.mean(0)).mean())
            m_taban = float(np.sqrt(2.0 / (np.pi * n)))
            for etiket, u in (("w_C", wC[L]), ("g_kuresel", wC[L] - sbar * wA[L])):
                nu = np.linalg.norm(u)
                if nu < 1e-9:
                    atlanan.append((f"{ad}/L{L}/{etiket}", "sifir norm")); continue
                tau_bar = float(np.linalg.norm(u - Cb * (Cb @ u)) / nu)
                proj = C @ u
                tau_i = np.sqrt(np.maximum(nu**2 - proj**2, 0)) / nu
                satir.append(dict(govde=kk, zemin=z, tarz=tarz, katman=L, nesne=etiket,
                                  tau_ortC=tau_bar, tau_med=float(np.median(tau_i)),
                                  tau_p10=float(np.percentile(tau_i, 10)),
                                  norm=float(nu), m_isaret=m, m_taban=m_taban, n=int(n)))
        del Cm, Pm

    payda("geometri", n_hucre=len(fs), n_satir=len(satir), n_atlanan=len(atlanan))
    if atlanan:
        print("  ATLANANLAR (adiyla, sessiz degil):")
        for a, r in atlanan[:20]:
            print(f"      {a}  ⇒  {r}")
        if len(atlanan) > 20:
            print(f"      … +{len(atlanan)-20}")

    yol = f"{CIKTI}/k0_geometri.json"
    json.dump(dict(protokol="PROTOKOL_K0_2026-08-01.md@5adb8ee", bolme=bolme,
                   n_maks=n_maks, satir=satir, atlanan=atlanan), open(yol, "w"))
    print(f"\n  → {yol}  ({len(satir)} satir)")
    return satir



def kararlilik(hucreler, R=5, n_maks=2000, seed=20260801):
    import esdegerlik as E
    from sklearn.linear_model import LogisticRegression
    print("═" * 88); print("K0.K · CAPA-KARARLILIK (küme-birimli yari-bölme)"); print("═" * 88)
    print("")
    ZEM = {"dis": ("disapere", "forum_id"), "kia": ("kialo", "debate")}
    out = []
    for (kk, z, tarz, L) in hucreler:
        if z == "dis":
            Pt, Ct, y, kume = E.disapere("train")
        else:
            (Pt, Ct, y, kume), _ = E.kialo()
        Pm = np.load(f"{OT}/{kk}__{z}__train__ebeveyn__{tarz}.npy", mmap_mode="r")
        Cm = np.load(f"{OT}/{kk}__{z}__train__cevap__{tarz}.npy", mmap_mode="r")
        if Pm.shape[0] != len(y):
            raise RuntimeError(f"SIRA UYUMSUZ: {kk}/{z} dizi n={Pm.shape[0]} etiket n={len(y)} ⇒ HATA.")
        n = min(n_maks, len(y))
        P = np.asarray(Pm[:n, L], np.float32); C = np.asarray(Cm[:n, L], np.float32)
        X = E.ozellik(P, C); yy = y[:n]; dial_ = np.asarray(kume)[:n]
        blok = X.shape[1] // 3
        prm = E.ZEMIN[ZEM[z][0]]
        uk = np.unique(dial_); rng = np.random.default_rng(seed)
        coslar_wC, coslar_g, n_kul = [], [], 0
        for r in range(R):
            pr = rng.permutation(uk); yar = [set(pr[:len(pr) // 2]), set(pr[len(pr) // 2:])]
            ws = []
            for h in yar:
                m = np.array([c in h for c in dial_])
                if len(np.unique(yy[m])) < 2: ws = []; break
                mdl = LogisticRegression(max_iter=prm["iters"], C=0.5,
                                         class_weight=("balanced" if prm["dengeli"] else None))
                mdl.fit(X[m], yy[m]); ws.append(mdl.coef_[0].astype(np.float32))
            if len(ws) != 2: continue
            n_kul += 1
            s0 = np.sign(np.sign(P[:n] / (np.linalg.norm(P[:n], axis=1, keepdims=True) + 1e-12) -
                                 C[:n] / (np.linalg.norm(C[:n], axis=1, keepdims=True) + 1e-12)).mean(0))
            s0[s0 == 0] = 1
            def cs(a, b):
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")
            coslar_wC.append(cs(ws[0][blok:2 * blok], ws[1][blok:2 * blok]))
            coslar_g.append(cs(ws[0][blok:2 * blok] - s0 * ws[0][2 * blok:],
                               ws[1][blok:2 * blok] - s0 * ws[1][2 * blok:]))
        payda(f"kararlilik/{kk}/{z}/{tarz}/L{L}", n_kume=len(uk), n_ornek=n,
              n_kullanilan_bolme=n_kul, R_istenen=R)
        rec = dict(govde=kk, zemin=z, tarz=tarz, katman=L, n=int(n), n_kume=int(len(uk)),
                   R=n_kul, cos_wC=float(np.mean(coslar_wC)), cos_wC_min=float(np.min(coslar_wC)),
                   cos_g=float(np.mean(coslar_g)), cos_g_min=float(np.min(coslar_g)))
        out.append(rec)
        print(f"  {kk}/{z}/{tarz}/L{L}: küme={len(uk)} n={n} R={n_kul} | "
              f"cos(w_C)={rec['cos_wC']:+.3f} (min {rec['cos_wC_min']:+.3f}) | "
              f"cos(g)={rec['cos_g']:+.3f} (min {rec['cos_g_min']:+.3f})")
    json.dump(out, open(f"{CIKTI}/k0_kararlilik.json", "w"))
    print(f"\n  → {CIKTI}/k0_kararlilik.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kapi", action="store_true")
    ap.add_argument("--geometri", action="store_true")
    ap.add_argument("--kararlilik", action="store_true")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    if a.kapi or a.geometri or a.kararlilik:
        kapi()
    if a.kararlilik:
        kararlilik([("K2","dis","duz_ortalama",14),("K4","dis","duz_ortalama",16),
                    ("K2","kia","duz_ortalama",14),("K4","kia","duz_ortalama",16)])
    if a.geometri:
        geometri(n_maks=a.n)
