#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, collections, datetime
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import kt_kodlama as KT
from pergel_cekirdek import Cekirdek, Permutator

MNT = __DNH_DATA__ + ""
KTQ = f"{MNT}/kt_qwen25"
DOZ = (0.0, 0.02, 0.03, 0.05, 0.08, 0.15)
K_CEKIM, K_FOLD, SEED = 200, 5, 20260804
QWEN_KATMAN = (18, 19, 20, 21, 22)
E5_KATMAN = (21, 22, 23)
E5_ONEK = "query: "


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def on_ucus(cikis, ad, ek=None):
    import torch, subprocess, platform, transformers
    g = []
    for i in range(torch.cuda.device_count()):
        f, t = torch.cuda.mem_get_info(i)
        g.append(dict(cuda=i, bos_gb=round(f / 2**30, 1), toplam_gb=round(t / 2**30, 1)))
    k = dict(kosu=ad, utc=utc(), HF_HOME=os.environ.get("HF_HOME", "<AYARLI DEGIL>"),
             gpu=g, cikis_koku=cikis,
             imaj=dict(os=platform.platform(), python=platform.python_version(),
                       torch=torch.__version__, cuda=torch.version.cuda,
                       transformers=transformers.__version__, numpy=np.__version__),
             disk=subprocess.run(["df", "-h", MNT], capture_output=True,
                                 text=True).stdout.strip().splitlines()[-1],
             git=subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip(), **(ek or {}))
    yol = f"{cikis}/on_ucus__pergel_null_{k['utc']}.json"
    assert not os.path.exists(yol), ""
    json.dump(k, open(yol, "w"), indent=1, ensure_ascii=False)
    payda("pergel_on_ucus", n_gpu=len(g), n_kalem=6, red_eksik=0)
    print(f"  ★ ÖN-UCUS → {os.path.basename(yol)} · gpu " +
          " · ".join(f"cuda:{x['cuda']} bos {x['bos_gb']}GB" for x in g), flush=True)
    return k


def veri():
    R, M, red = {}, {}, collections.Counter()
    for s in ("train", "dev"):
        rows = KT.load_split(s)
        meta = json.load(open(f"{KTQ}/{s}__meta.json"))
        if len(rows) != len(meta):
            red["uzunluk_uyusmaz"] += 1
        else:
            for a, b in zip(rows, meta):
                if (str(a["etiket"]) != str(b["etiket"]) or a["forum_id"] != b["forum_id"]
                        or a["dosya"] != b["dosya"]):
                    red["satir_uyusmaz"] += 1
        R[s], M[s] = rows, meta
    payda("pergel_hiza", n_bolme=2, n_satir=sum(len(v) for v in R.values()),
          **{f"red_{k}": v for k, v in red.items()})
    if sum(red.values()):
        raise RuntimeError(f"{dict(red)}")
    print(f"{len(R['train'])} {len(R['dev'])}"
          f"", flush=True)
    return R


def qwen_katmanlari():
    out = {}
    for lay in QWEN_KATMAN:
        parts = []
        for s in ("train", "dev"):
            a = np.load(f"{KTQ}/{s}__cevap__ortalama.fp16.npy", mmap_mode="r")
            parts.append(np.asarray(a[:, lay, :], dtype=np.float32))
        out[f"qwen2.5-7B-base_L{lay}"] = np.concatenate(parts)
    return out


def e5_katmanlari(texts, dev, bs=64, maks=512):
    import torch
    from transformers import AutoTokenizer, AutoModel
    ad = "intfloat/e5-large-v2"
    tok = AutoTokenizer.from_pretrained(ad)
    m = AutoModel.from_pretrained(ad, dtype=torch.float16).to(dev).eval()
    buf = {lay: [] for lay in E5_KATMAN}
    t0 = time.time()
    for i in range(0, len(texts), bs):
        enc = tok([E5_ONEK + t for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=maks).to(dev)
        with torch.no_grad():
            hs = m(**enc, output_hidden_states=True).hidden_states
        msk = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        for lay in E5_KATMAN:
            v = (hs[lay] * msk).sum(1) / msk.sum(1).clamp(min=1)
            buf[lay].append(v.float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    print(f"    e5 forward: {len(texts)} metin × {len(E5_KATMAN)} katman [{time.time()-t0:.0f}s]",
          flush=True)
    return {f"e5-large-v2_L{lay}": np.concatenate(buf[lay]) for lay in E5_KATMAN}


def _foldlar(gruplar, k=K_FOLD, seed=SEED):
    u = sorted(set(gruplar))
    rng = np.random.default_rng(seed)
    atama = {g: i % k for i, g in enumerate(rng.permutation(len(u)))}
    gid = {g: i for i, g in enumerate(u)}
    return np.array([atama[gid[g]] for g in gruplar])


def a_okumasi(X, y, fold):
    A = np.empty(len(y))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
            A[te] = 0.0; continue
        mu = X[tr].mean(0)
        u = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        n = np.linalg.norm(u)
        u = u / n if n > 1e-12 else u
        Z = X[te] - mu
        nz = np.maximum(np.linalg.norm(Z, axis=1), 1e-12)
        A[te] = (Z @ u) / nz
    return A


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _perm_ici(y, gruplar, rng):
    yp = y.copy()
    for g in set(gruplar):
        m = np.array([x == g for x in gruplar])
        yp[m] = rng.permutation(y[m])
    return yp


def prova_matrisi(ADAY, y, gruplar, fold, rng, CEK, PERM):
    out, REG = {}, {}
    for ad, X in ADAY.items():
        C = CEK[ad]
        A = C.oku_toplu(y[None, :])[0].cpu().numpy()
        g = float(C.auc_toplu(y[None, :], C.oku_toplu(y[None, :]))[0])
        A_ref = a_okumasi(X, y, fold); g_ref = auc(y, A_ref)
        dA = float(np.abs(A - A_ref).max()); dG = abs(g - g_ref)
        REG[ad] = dict(auc_gpu=g, auc_cpu_ref=g_ref, d_auc=dG, maks_dA=dA)
        if dG > 1e-4 or dA > 1e-4:
            raise RuntimeError(f"{ad} {g:.8f} {g_ref:.8f}"
                               f"{dG:.2e} {dA:.2e}")
        d_ger = abs(A[y == 1].mean() - A[y == 0].mean())
        Yp = PERM.cek(K_CEKIM)
        Ap = CEK[ad].oku_toplu(Yp).cpu().numpy()
        Ypf = Yp.astype(bool)
        m1 = np.where(Ypf, Ap, np.nan)
        m0 = np.where(~Ypf, Ap, np.nan)
        pl = np.abs(np.nanmean(m1, axis=1) - np.nanmean(m0, axis=1))
        q1, q3 = np.percentile(A, [25, 75])
        out[ad] = dict(
            ayirt_gucu_auc_of=g,
            doyma_orani=float(d_ger / max(np.median(pl), 1e-12)),
            doyma_gercek=float(d_ger), doyma_plasebo_medyan=float(np.median(pl)),
            menzil_iqr=float(q3 - q1), menzil_min=float(A.min()), menzil_maks=float(A.max()),
            menzil_kullanilan_pay=float((A.max() - A.min()) / 2.0), n=int(len(y)))
        print(f"    {ad:26s} AUC_of={g:.3f} · doyma={out[ad]['doyma_orani']:6.2f}× · "
              f"IQR={q3-q1:.3f} · menzil[{A.min():+.3f},{A.max():+.3f}] · "
              f"reg-kapi Δauc={REG[ad]['d_auc']:.1e} maks|ΔA|={REG[ad]['maks_dA']:.1e}", flush=True)
    print(f"    ✓ REGRESYON KAPISI: {len(REG)}/{len(REG)} aday, maks Δauc="
          f"{max(v['d_auc'] for v in REG.values()):.2e} (bar 1e-4; float32↔float64 gürültüsü)",
          flush=True)
    return out, REG


def eksen_null(C, PERM, K=K_CEKIM):
    Yp = PERM.cek(K)
    v = np.asarray(C.auc_toplu(Yp, C.oku_toplu(Yp)), dtype=float)
    h, ke = np.histogram(v, bins=24)
    return dict(K=K, E_null=float(v.mean()), sd_null=float(v.std(ddof=1)),
                p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                histogram=dict(sayim=h.tolist(), kenar=[float(x) for x in ke]),
                cekimler=[float(x) for x in v])


def inj2(C, PERM, null_p95, K=K_CEKIM):
    tab = {}
    for d in DOZ:
        Ym = PERM.karisim(K, d)
        v = np.asarray(C.auc_toplu(Ym, C.oku_toplu(Ym)), dtype=float)
        tab[str(d)] = dict(medyan=float(np.median(v)), ort=float(v.mean()),
                           sd=float(v.std(ddof=1)),
                           pass_orani=float((v > null_p95).mean()))
    dl = [d for d in DOZ if d > 0 and tab[str(d)]["pass_orani"] >= 0.5]
    return dict(izgara=list(DOZ), tablo=tab, delta_yildiz=(min(dl) if dl else None),
                kol_sifir_pass_orani=tab["0.0"]["pass_orani"],
                monoton=bool(all(tab[str(DOZ[i])]["medyan"] <= tab[str(DOZ[i + 1])]["medyan"]
                                 for i in range(len(DOZ) - 1))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--cikis", default=None)
    ap.add_argument("--rk-ornek", type=int, default=4000)
    a = ap.parse_args()
    cikis = a.cikis or f"{ROOT}/unreleased/NULL_OLCUM_pergel_2026-08-04"
    os.makedirs(cikis, exist_ok=True)
    print("═" * 96); print("")
    on_ucus(cikis, "A", ek=dict(adaylar=[f"qwen2.5-7B-base_L{x}" for x in QWEN_KATMAN] +
                                [f"e5-large-v2_L{x}" for x in E5_KATMAN],
                                bolme="DISAPERE train+dev (TEST ACILMADI)", doz_izgarasi=list(DOZ)))
    R = veri()
    rows = R["train"] + R["dev"]
    y = np.array([int(r["etiket"]) for r in rows])
    gruplar = [r["forum_id"] for r in rows]
    fold = _foldlar(gruplar)
    print(f"  n={len(y)} · dispute={int(y.sum())} · forum={len(set(gruplar))} · fold={K_FOLD}",
          flush=True)
    ADAY = qwen_katmanlari()
    ADAY.update(e5_katmanlari([r["cevap"] for r in rows], a.dev))
    for k, v in ADAY.items():
        assert v.shape[0] == len(y), f"{k}: {v.shape} ≠ {len(y)}"
    rng = np.random.default_rng(SEED)
    CEK = {ad: Cekirdek(X, fold, a.dev) for ad, X in ADAY.items()}
    PERM = Permutator(y, gruplar, SEED)
    print("")
    PM, REG = prova_matrisi(ADAY, y, gruplar, fold, rng, CEK, PERM)
    print("\n[2b] EKSEN-BASI NULL (N3, ≥200 cekim, forum-kümeli)")
    NUL = {}
    for ad in ADAY:
        NUL[ad] = eksen_null(CEK[ad], PERM)
        print(f"    {ad:26s} E[null]={NUL[ad]['E_null']:.4f} sd={NUL[ad]['sd_null']:.4f} "
              f"p95={NUL[ad]['p95']:.4f}", flush=True)
    print("\n[2c] INJ-2 (doz izgarasi = gercek-etiket koruma payi)")
    INJ = {}
    for ad in ADAY:
        INJ[ad] = inj2(CEK[ad], PERM, NUL[ad]["p95"])
        print(f"    {ad:26s} δ*={INJ[ad]['delta_yildiz']} · KOL-Ø pass={INJ[ad]['kol_sifir_pass_orani']:.3f}"
              f" · monoton={INJ[ad]['monoton']}", flush=True)
    N5 = {ad: dict(ornek_istatistik=float(NUL[ad]["p95"] + 3 * NUL[ad]["sd_null"]),
                   rule="istatistik > p95(null) ⇒ PASS",
                   pass_ornegi_ulasilabilir=True) for ad in ADAY}
    S = dict(kosu="A", utc=utc(), n=int(len(y)), n_forum=len(set(gruplar)),
             bolme="DISAPERE train+dev · TEST ACILMADI", prova_matrisi=PM,
             eksen_null=NUL, inj2=INJ, N5=N5, regresyon_kapisi=REG,
             beyan=dict(N1=""
                           "",
                        N2=""
                           "",
                        N4="enjeksiyon ETIKET tarafinda, ölcüm temsil tarafinda ⇒ kanal-ayriligi",
                        kanal_ayriligi="enjekte edilen nesne etiket vektörüdür; A-okumasi "
                                       "donuk temsilden gelir — enjeksiyon kanali ≠ skor kanali"))
    json.dump(S, open(f"{cikis}/A_ekseni.json", "w"), indent=1, ensure_ascii=False)
    print("")
    from base_prefix_reading import kol_oku, KARAR_KOL
    met, meta = [], []
    rr = np.random.default_rng(SEED)
    for aile in ("qwen", "mistral"):
        for kol in KARAR_KOL:
            sat = kol_oku(aile, kol) or []
            for i, s in enumerate(sat):
                met.append(s["metin"]); meta.append(dict(aile=aile, kol=kol,
                                                         ebeveyn=s["ebeveyn"], korpus=s["korpus"]))
    idx = np.sort(rr.permutation(len(met))[:min(a.rk_ornek, len(met))])
    met = [met[i] for i in idx]; meta = [meta[i] for i in idx]
    E = e5_katmanlari(met, a.dev)
    en_iyi = max(PM, key=lambda k: PM[k]["ayirt_gucu_auc_of"] if k.startswith("e5") else -1)
    lay = int(en_iyi.split("_L")[1])
    Xd = np.concatenate([ADAY[en_iyi]])
    mu = Xd.mean(0); u = Xd[y == 1].mean(0) - Xd[y == 0].mean(0); u /= np.linalg.norm(u)
    V = E[f"e5-large-v2_L{lay}"] - mu
    Ark = (V @ u) / np.maximum(np.linalg.norm(V, axis=1), 1e-12)
    json.dump(dict(katman=en_iyi, n=len(Ark), meta=meta, A=[float(x) for x in Ark],
                   beyan=""
                         ""),
              open(f"{cikis}/A_okumasi_rk.json", "w"))
    payda("pergel_A", n_aday=len(ADAY), n_oge=int(len(y)), n_forum=len(set(gruplar)),
          n_rk_oge=len(Ark), red_hizasiz=0)
    print(f"\n★ A-EKSENI BITTI → {cikis}/A_ekseni.json + A_okumasi_rk.json")
