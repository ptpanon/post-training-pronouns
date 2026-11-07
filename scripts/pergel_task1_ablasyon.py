#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from pergel_cekirdek import Cekirdek, Permutator
import pergel_null_bare_imperative as A1

MNT = __DNH_DATA__ + ""
KTQ = f"{MNT}/kt_qwen25"
KATMAN = (18, 19, 20, 21, 22)
K_CEKIM, SEED = 200, 20260804

HAVUZ = ("ortalama", "ortalama_sinksiz", "sontoken")
PAKET_2x2 = ("ortalama", "sontoken")
DOYMA_ESIGI = 0.999


def havuz_yukle(havuz, lay):
    parts = []
    for s in ("train", "dev"):
        a = np.load(f"{KTQ}/{s}__cevap__{havuz}.fp16.npy", mmap_mode="r")
        parts.append(np.asarray(a[:, lay, :], dtype=np.float32))
    return np.concatenate(parts)


def hucre(X, y, fold, gruplar, PERM, dev, std):
    C = Cekirdek(X, fold, dev, std=std)
    A = C.oku_toplu(y[None, :])[0].cpu().numpy()
    g = float(C.auc_toplu(y[None, :], C.oku_toplu(y[None, :]))[0])
    A_ref = ref_okumasi(X, y, fold, std)
    g_ref = A1.auc(y, A_ref)
    dA = float(np.abs(A - A_ref).max()); dG = abs(g - g_ref)
    if dG > 1e-4 or dA > 1e-4:
        raise RuntimeError(f"{g:.8f} {g_ref:.8f} {dG:.2e}"
                           f"{dA:.2e}")
    d_ger = abs(A[y == 1].mean() - A[y == 0].mean())
    Yp = PERM.cek(K_CEKIM)
    Ap = C.oku_toplu(Yp).cpu().numpy()
    Ypf = Yp.astype(bool)
    pl = np.abs(np.nanmean(np.where(Ypf, Ap, np.nan), axis=1)
                - np.nanmean(np.where(~Ypf, Ap, np.nan), axis=1))
    q1, q3 = np.percentile(A, [25, 75])
    kirp = C.kirpilan_boyut
    C.bosalt()
    return dict(
        ayirt_gucu_auc_of=g,
        doyma_orani=float(d_ger / max(np.median(pl), 1e-12)),
        doyma_gercek=float(d_ger), doyma_plasebo_medyan=float(np.median(pl)),
        menzil_iqr=float(q3 - q1), menzil_min=float(A.min()), menzil_maks=float(A.max()),
        doygun_pay=float((np.abs(A) > DOYMA_ESIGI).mean()),
        doygun_sayi=int((np.abs(A) > DOYMA_ESIGI).sum()),
        std_kirpilan_boyut=int(kirp),
        reg_kapi=dict(auc_gpu=g, auc_cpu_ref=g_ref, d_auc=dG, maks_dA=dA))


def ref_okumasi(X, y, fold, std):
    if not std:
        return A1.a_okumasi(X, y, fold)
    A = np.empty(len(y))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        mu = X[tr].mean(0)
        s = np.maximum(X[tr].std(0, ddof=1), 1e-6)
        u = (X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)) / s
        n = np.linalg.norm(u); u = u / n if n > 1e-12 else u
        Z = (X[te] - mu) / s
        A[te] = (Z @ u) / np.maximum(np.linalg.norm(Z, axis=1), 1e-12)
    return A


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--cikis", default=f"{ROOT}/unreleased/GECE_TESHIS_2026-08-04")
    a = ap.parse_args()
    os.makedirs(a.cikis, exist_ok=True)
    print("═" * 96)
    print("PERGEL · GECE IS-1 · QWEN2.5-7B BASE OKUMA-ONARIM ABLASYONU — TESHIS, VERDICT DEGIL")
    A1.on_ucus(a.cikis, "IS1_ablasyon",
               ek=dict(soru="0,460 + ±1,000 doymasi ALET KUSURU MU?",
                       izgara=f"havuz{list(HAVUZ)} × std[ham, per-dim] × ℓ{list(KATMAN)}",
                       paket_2x2=list(PAKET_2x2),
                       forward="YOK — kt_qwen25 arsivi (üc havuz da hazir)",
                       gemma="DOKUNULMADI (gece paketi yasagi)"))
    R = A1.veri()
    rows = R["train"] + R["dev"]
    y = np.array([int(r["etiket"]) for r in rows])
    gruplar = [r["forum_id"] for r in rows]
    fold = A1._foldlar(gruplar)
    print(f"  n={len(y)} · dispute={int(y.sum())} · forum={len(set(gruplar))} · fold=5", flush=True)
    PERM = Permutator(y, gruplar, SEED)

    OUT, t0 = {}, time.time()
    print(f"\n[IS-1] {len(HAVUZ)} havuz × 2 standardizasyon × {len(KATMAN)} katman = "
          f"{len(HAVUZ)*2*len(KATMAN)} hücre — HEPSI raporlanir")
    for hv in HAVUZ:
        for lay in KATMAN:
            X = havuz_yukle(hv, lay)
            assert X.shape[0] == len(y), f"{hv}/L{lay}: {X.shape} ≠ {len(y)}"
            for std in (False, True):
                ad = f"{hv}__{'perdimstd' if std else 'ham'}__L{lay}"
                OUT[ad] = hucre(X, y, fold, gruplar, PERM, a.dev, std)
                o = OUT[ad]
                print(f"    {ad:34s} AUC_of={o['ayirt_gucu_auc_of']:.3f} · "
                      f"doyma={o['doyma_orani']:6.2f}× · IQR={o['menzil_iqr']:.3f} · "
                      f"menzil[{o['menzil_min']:+.3f},{o['menzil_maks']:+.3f}] · "
                      f"|A|>{DOYMA_ESIGI} pay={o['doygun_pay']:.3f} · "
                      f"reg Δauc={o['reg_kapi']['d_auc']:.1e}", flush=True)
            del X
    mx = max(v["reg_kapi"]["d_auc"] for v in OUT.values())
    print(f"\n  ✓ REGRESYON KAPISI: {len(OUT)}/{len(OUT)} hücre · maks Δauc={mx:.2e} (bar 1e-4)")

    print("\n  ★ MENZIL-DOYGUNLUGU (|A|>0,999 öge payi) — hücre basina:")
    for hv in HAVUZ:
        for std in (False, True):
            p = [OUT[f"{hv}__{'perdimstd' if std else 'ham'}__L{l}"]["doygun_pay"] for l in KATMAN]
            print(f"      {hv:18s} {'per-dim std' if std else 'ham        '}: "
                  f"ℓ18–22 pay = [{', '.join(f'{x:.3f}' for x in p)}]", flush=True)

    payda("pergel_is1", n_hucre=len(OUT), n_havuz=len(HAVUZ), n_katman=len(KATMAN),
          n_oge=int(len(y)), red_regresyon=0)
    S = dict(kosu="IS1_ablasyon", utc=A1.utc(), n=int(len(y)),
             n_forum=len(set(gruplar)), bolme="DISAPERE train+dev · TEST ACILMADI",
             izgara=dict(havuz=list(HAVUZ), std=["ham", "perdimstd"], katman=list(KATMAN),
                         paket_2x2=list(PAKET_2x2)),
             doyma_esigi=DOYMA_ESIGI, hucreler=OUT, sure_s=round(time.time() - t0, 1),
             beyan=dict(
                 havuz_tanimlari=dict(
                     ortalama="DÜZ ortalama: pos0 DÂHIL, sink elemesi YOK — DÜNKÜ kosunun havuzu",
                     ortalama_sinksiz="arsiv tarifi: pos0 haric + katman-basina norm>8×medyan sink haric",
                     sontoken="son token"),
                 std=""
                     "",
                 yetki=""
                       "",
                 secmece="kurulan 30 hücrenin 30'u da JSON'da; paketin 2×2'si `izgara.paket_2x2`",
                 gemma="bu kosuda gemma-türevi HICBIR sayi üretilmedi"))
    json.dump(S, open(f"{a.cikis}/IS1_ablasyon.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n★ IS-1 BITTI [{S['sure_s']:.0f}s] → {a.cikis}/IS1_ablasyon.json")
