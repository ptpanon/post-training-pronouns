#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, math, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

CARD = f"{ROOT}/results/template_filtered_2026-09-07.json"
CIK = os.environ.get("SOY_CIK", f"{ROOT}/results/lineage_permutation_2026-09-11.json")
NPERM = int(os.environ.get("SOY_NPERM", "1000"))
SEED = 20260911
BAR = 0.05

SOY = {
    "Gemma-3-4B": "Google", "Gemma-3-12B": "Google", "Gemma-3-27B": "Google",
    "Llama-3.1-8B": "Meta", "Llama-3.1-70B": "Meta",
    "Tulu3-8B": "AI2", "OLMo2-13B": "AI2", "OLMo2-32B": "AI2",
    "Qwen2.5-1.5B": "Qwen", "Qwen2.5-3B": "Qwen", "Qwen2.5-7B": "Qwen",
    "Qwen2.5-14B": "Qwen", "Qwen2.5-32B": "Qwen", "Qwen2.5-72B": "Qwen",
    "Mistral-7B-v0.3": "Mistral", "OLMo-3-7B": "OLMo-3",
}


def _H(x, g):
    n = len(x)
    r = np.empty(n, dtype=float)
    o = np.argsort(x, kind="mergesort")
    xs = x[o]; i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    H = 0.0
    for u in np.unique(g):
        m = g == u
        H += m.sum() * (r[m].mean() - (n + 1) / 2.0) ** 2
    H *= 12.0 / (n * (n + 1))
    _, cnt = np.unique(xs, return_counts=True)
    duz = 1 - (cnt ** 3 - cnt).sum() / float(n ** 3 - n)
    return H / duz if duz > 0 else H


def _eta2(x, g):
    gt = x.mean()
    sa = sum((g == u).sum() * (x[g == u].mean() - gt) ** 2 for u in np.unique(g))
    st = ((x - gt) ** 2).sum()
    return float(sa / st) if st > 0 else 0.0


def _isaret_saf(sgn, g):
    m = sgn != 0
    if not m.any():
        return 0, 0
    uy = 0
    for u in np.unique(g[m]):
        k = sgn[m][g[m] == u]
        bask = 1 if (k > 0).sum() >= (k < 0).sum() else -1
        uy += int((np.sign(k) == bask).sum())
    return int(uy), int(m.sum())


def prova():
    x = np.array([1., 2., 3., 4., 5., 6.])
    g1 = np.array([0, 0, 0, 0, 0, 0]); g2 = np.array([0, 0, 0, 1, 1, 1])
    h1, h2 = _H(x, g1), _H(x, g2)
    ok = (abs(h1) < 1e-9) and (h2 > 3.0) and (_eta2(x, g1) < 1e-9)
    payda("soy_prova", n_sinav=3, hal_tek_grup_H=round(h1, 6),
          hal_ayrik_H=round(h2, 4), red_beklenmeyen=int(not ok))
    print(f"{h1:.6f} {h2:.3f}"
          f"", flush=True)
    return ok


def main():
    if not prova():
        print("★ PROVA DÜSTÜ ⇒ ölcüm YAZILMAZ"); return 4
    K = json.load(io.open(CARD, encoding="utf-8"))
    A = [a for a in K["aileler"] if a.get("hal") == "ÖLCÜLDÜ"]
    eks = sorted(set(SOY) ^ {a["aile"] for a in A})
    if eks:
        payda("soy_permutasyon", n_aile=len(A), n_perm=0, hal_H=0.0, hal_frac=1.0,
              hal_eta2=0.0, hal_frac_eta2=1.0, hal_ulasilabilir_en_kucuk=0.0,
              red_card_disi=len(eks))
        print(f"{eks}"); return 3
    POZ = set(K["sayim"]["sablonlu"]["pozitif"])
    NEG = set(K["sayim"]["sablonlu"]["negatif"])
    ad = [a["aile"] for a in A]
    x = np.array([a["sbl"]["dM1"] for a in A], dtype=float)
    lab = np.array([SOY[n] for n in ad])
    g = np.unique(lab, return_inverse=True)[1]
    boy = {u: int((lab == u).sum()) for u in sorted(set(lab))}
    print(f"{len(ad)} {len(boy)} {boy}"
          f"")

    Hg, Eg = _H(x, g), _eta2(x, g)
    _boy = [int((g == u).sum()) for u in np.unique(g)]
    _sira = np.argsort(x)
    _en_iyi = np.empty(len(x), dtype=int); _o = 0
    for _i, _b in enumerate(sorted(_boy, reverse=True)):
        _en_iyi[_sira[_o:_o + _b]] = _i; _o += _b
    H_en_iyi = _H(x, _en_iyi)
    sgn = np.array([1 if n in POZ else (-1 if n in NEG else 0) for n in ad])
    Ig, n_isaretli = _isaret_saf(sgn, g)
    rng = np.random.default_rng(SEED)
    hn = np.empty(NPERM); en = np.empty(NPERM); inull = np.empty(NPERM)
    hn_iyi = 0
    for i in range(NPERM):
        p = rng.permutation(len(x))
        hn[i] = _H(x, g[p]); en[i] = _eta2(x, g[p])
        inull[i] = _isaret_saf(sgn, g[p])[0]
        hn_iyi += int(_H(x, g[p]) >= H_en_iyi)
    fH = float((hn >= Hg).mean()); fE = float((en >= Eg).mean())
    fI = float((inull >= Ig).mean())
    f_en_iyi = hn_iyi / float(NPERM)
    guc_var = bool(f_en_iyi <= BAR)
    print(f"  ★ GÜC KONTROLÜ (pozitif): en iyi olasi etiketleme H={H_en_iyi:.3f} "
          f"⇒ frac={f_en_iyi:.3f} · bar {BAR} ⇒ EYLEM: frac>bar ise bu tasarim "
          f"HICBIR etkiyi göremez ⇒ ÖLCÜLEMEZ-GÜC", flush=True)
    print(f"{Ig} {n_isaretli}"
          f"{inull.mean():.2f}"
          f"{fI:.3f} {BAR}"
          f"", flush=True)
    tab = 1.0 / NPERM
    en_kucuk = float((hn >= hn.max()).mean())
    print(f"  gözlenen H={Hg:.3f} · null ort={hn.mean():.3f} sd={hn.std():.3f} "
          f"⇒ frac={fH:.3f} · bar {BAR} ⇒ EYLEM: frac≤bar ⇒ «lineage sorts it»",
          flush=True)
    print(f"  gözlenen η²={Eg:.3f} · null ort={en.mean():.3f} ⇒ frac={fE:.3f}",
          flush=True)
    print(f"{hn.mean():.3f}"
          f"", flush=True)
    print(f"{1.0/NPERM:.4f} {NPERM}"
          f"{BAR}"
          f"", flush=True)

    SOY_B = {k: ("AI2" if v == "OLMo-3" else v) for k, v in SOY.items()}
    labB = np.array([SOY_B[n] for n in ad])
    gB = np.unique(labB, return_inverse=True)[1]
    HgB, EgB = _H(x, gB), _eta2(x, gB)
    rngB = np.random.default_rng(SEED)
    hnB = np.empty(NPERM); enB = np.empty(NPERM)
    for i in range(NPERM):
        p = rngB.permutation(len(x))
        hnB[i] = _H(x, gB[p]); enB[i] = _eta2(x, gB[p])
    fHB = float((hnB >= HgB).mean()); fEB = float((enB >= EgB).mean())
    print(f"  [OKUMA-B · OLMo-3=AI2] H={HgB:.3f} · null ort={hnB.mean():.3f} "
          f"⇒ frac={fHB:.3f} · η²={EgB:.3f} (null {enB.mean():.3f}) frac={fEB:.3f} "
          f"⇒ esik {BAR} ⇒ EYLEM: iki okuma AYRI ad veriyorsa metin IKISINI de "
          f"yazar", flush=True)

    if not guc_var:
        verdict = "ÖLCÜLEMEZ-GÜC"
        cumle = ("even the best possible labelling of these group sizes cannot "
                 "clear the bar, so this panel cannot answer the question")
    elif fH <= BAR:
        verdict = "SIRALIYOR"
        cumle = ("lineage sorts the magnitudes; lineage and recipe are "
                 "confounded in this panel")
    else:
        verdict = "SIRALAMIYOR"
        cumle = ("lineage does not sort it")
    isaret_hukmu = "ISARETI-SIRALIYOR" if fI <= BAR else "ISARETI-SIRALAMIYOR"
    print(f"\n★ VERDICT (büyüklük): {verdict} — «{cumle}»", flush=True)
    print(f"★ VERDICT (isaret): {isaret_hukmu} — {Ig}/{n_isaretli}, frac={fI:.3f}",
          flush=True)
    payda("soy_permutasyon", n_aile=len(ad), n_perm=NPERM,
          hal_H=round(Hg, 4), hal_frac=round(fH, 4), hal_eta2=round(Eg, 4),
          hal_frac_eta2=round(fE, 4), hal_ulasilabilir_en_kucuk=round(en_kucuk, 4),
          hal_frac_isaret=round(fI, 4), hal_guc_var=int(guc_var),
          red_card_disi=len(eks))
    C = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="ÖLCÜM — karar kaskadi kosudan ÖNCE gövdede sabitlendi",
             alet="scripts/lineage_permutation.py", borc="D-0911-V33 §B/3 · HAKEM-ASSISTANT W4",
             kaynak=os.path.relpath(CARD, ROOT), nicelik="sbl.dM1 (sablonlu ΔM1)",
             kume_birimi="aile", n_perm=NPERM, seed=SEED, bar=BAR,
             referans="",
             soy=boy, aile={n: dict(soy=SOY[n], dM1=float(v)) for n, v in zip(ad, x)},
             H=dict(gozlenen=round(Hg, 4), null_ort=round(float(hn.mean()), 4),
                    null_sd=round(float(hn.std()), 4), frac=round(fH, 4)),
             eta2=dict(gozlenen=round(Eg, 4), null_ort=round(float(en.mean()), 4),
                       frac=round(fE, 4)),
             okuma_B=dict(aciklama=""
                                    "",
                          soy={u: int((labB == u).sum()) for u in sorted(set(labB))},
                          H=round(HgB, 4), null_ort=round(float(hnB.mean()), 4),
                          frac=round(fHB, 4), eta2=round(EgB, 4),
                          frac_eta2=round(fEB, 4),
                          VERDICT=("SIRALIYOR" if fHB <= BAR else "SIRALAMIYOR")),
             ulasilabilir_en_kucuk_frac=round(en_kucuk, 4), taban_cozunurluk=tab,
             guc_kontrolu=dict(aciklama="pozitif kontrol: EN IYI olasi etiketleme",
                               H_en_iyi=round(float(H_en_iyi), 4),
                               frac=round(f_en_iyi, 4), guc_var=guc_var),
             isaret=dict(aciklama="iddianin kendi ekseni: yukari/asagi bölünmesi",
                         uyusan=Ig, n_isaretli=n_isaretli,
                         null_ort=round(float(inull.mean()), 3),
                         frac=round(fI, 4), VERDICT=isaret_hukmu),
             VERDICT=verdict, CUMLE=cumle)
    json.dump(C, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
