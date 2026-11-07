import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, fcntl, subprocess

os.environ.setdefault("HF_HOME", __DNH_ROOT__ + "/.hf")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")

import numpy as np
import esdegerlik as E
import kulak_layer as KK
import muhafiz_sabit_konum as MSK

MNT = __DNH_DATA__ + ""
OT = f"{MNT}/okuma_tarzi"
MAN = f"{ROOT}/unreleased/manifests"
MANIFEST = f"{MAN}/okuma_tarzi_2026-07-31.json"
DEV = "cuda:1"
R_YARI = 200
K_NULL = 200
SEED = 20260731
D_IZGARA = [1, 2, 4, 8]
DERINLIKLER = [0.00, 0.25, 0.50, 0.75, 1.00]
os.makedirs(OT, exist_ok=True)

TARZLAR = ("son_token", "duz_ortalama", "sinksiz_ortalama", "maks_havuz",
           "kuantil_demeti", "son_bolge", "orta_bolge", "std_havuz")
MERKEZ = {"duz_ortalama", "sinksiz_ortalama", "orta_bolge", "kuantil_demeti"}
UC = {"son_token", "son_bolge", "maks_havuz", "std_havuz"}

KESKILER = {
    "K2":     dict(hf="Qwen/Qwen2.5-7B",        tur="lm",  maxlen=512,  dtype="float16"),
    "K4":     dict(hf="mistralai/Mistral-7B-v0.3", tur="lm", maxlen=512, dtype="float16"),
    "c2m":    dict(hf="BAAI/bge-m3",            tur="enc", maxlen=8192, dtype="float16"),
    "st5-xl": dict(hf="cartgr/embeddings-for-preferences-st5-xl", tur="st",
                   maxlen=256, dtype="float32"),
}


def yaz(*a):
    print(*a, flush=True)


def man_yaz(anahtar, deger):
    os.makedirs(MAN, exist_ok=True)
    with open(MANIFEST + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        R = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
        R.setdefault("_statu", "KESIF/BETIMLEME — verdict YOK.")
        R.setdefault("_onkayit", "d71b53c"); R.setdefault("_prediction", "d844894")
        R.setdefault("_tarih", "2026-07-31")
        R.setdefault(anahtar, {}).update(deger) if isinstance(deger, dict) and \
            isinstance(R.get(anahtar), dict) else R.__setitem__(anahtar, deger)
        gec = MANIFEST + ".tmp"
        json.dump(R, open(gec, "w"), ensure_ascii=False, indent=1)
        os.replace(gec, MANIFEST)


def tarz_hesapla(hl, valid, Sg, sayac):
    import torch
    B, S, D = hl.shape
    ar = torch.arange(B, device=hl.device)
    vf = valid.unsqueeze(-1).to(hl.dtype)
    nv = vf.sum(1).clamp(min=1)
    out = {}
    out["son_token"] = hl[ar, Sg - 1, :]
    ort = (hl * vf).sum(1) / nv
    out["duz_ortalama"] = ort
    norms = hl.float().norm(dim=-1)
    med = torch.nanmedian(norms.masked_fill(~valid, float("nan")), dim=1,
                          keepdim=True).values
    tut = valid & (norms <= 8 * med)
    tut[:, 0] = False
    bos = tut.sum(1) == 0
    yedek = valid & (torch.arange(S, device=hl.device)[None] >= 1)
    tut = torch.where(bos.unsqueeze(1), yedek, tut)
    bos2 = tut.sum(1) == 0
    tut = torch.where(bos2.unsqueeze(1), valid, tut)
    sayac["ucuncu_basamak"] += int(bos2.sum())
    tf = tut.unsqueeze(-1).to(hl.dtype)
    out["sinksiz_ortalama"] = (hl * tf).sum(1) / tf.sum(1).clamp(min=1)
    out["maks_havuz"] = hl.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(1).values
    srt = hl.masked_fill(~valid.unsqueeze(-1), float("inf")).sort(dim=1).values
    qs = []
    for q in (0.10, 0.50, 0.90):
        k = torch.clamp(torch.round(q * (Sg.float() - 1)).long(), min=0)
        qs.append(torch.gather(srt, 1, k[:, None, None].expand(B, 1, D)).squeeze(1))
    out["kuantil_demeti"] = torch.cat(qs, dim=1)
    del srt
    idx = torch.arange(S, device=hl.device)[None].expand(B, S)
    n_son = torch.clamp(torch.ceil(0.10 * Sg.float()).long(), min=1)
    msk_son = valid & (idx >= (Sg - n_son)[:, None])
    sf = msk_son.unsqueeze(-1).to(hl.dtype)
    out["son_bolge"] = (hl * sf).sum(1) / sf.sum(1).clamp(min=1)
    a0 = torch.floor(0.40 * Sg.float()).long()
    a1 = torch.ceil(0.60 * Sg.float()).long()
    msk_orta = valid & (idx >= a0[:, None]) & (idx < a1[:, None])
    orta_bos = msk_orta.sum(1) == 0
    tek = torch.clamp((Sg.float() / 2).long(), max=S - 1)
    msk_orta = torch.where(orta_bos.unsqueeze(1), idx == tek[:, None], msk_orta)
    of = msk_orta.unsqueeze(-1).to(hl.dtype)
    out["orta_bolge"] = (hl * of).sum(1) / of.sum(1).clamp(min=1)
    d2 = ((hl - ort.unsqueeze(1)) ** 2 * vf).sum(1) / nv
    out["std_havuz"] = d2.clamp(min=0).sqrt()
    return out


def yukle_keski(kk):
    import torch
    K = KESKILER[kk]
    if K["tur"] == "st":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(K["hf"], device=DEV)
        return m[0].tokenizer, m[0].auto_model.to(torch.bfloat16).eval()
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(K["hf"])
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if K["tur"] == "lm":
        mdl = AutoModelForCausalLM.from_pretrained(K["hf"], dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").to(DEV).eval()
    else:
        mdl = AutoModel.from_pretrained(K["hf"], dtype=torch.bfloat16).to(DEV).eval()
    return tok, mdl


def kodla_keski(kk, metinler, bs=24):
    import torch
    K = KESKILER[kk]
    tok, mdl = yukle_keski(kk)
    L = mdl.config.num_hidden_layers + 1
    D = mdl.config.hidden_size if hasattr(mdl.config, "hidden_size") else mdl.config.d_model
    dt = np.float16 if K["dtype"] == "float16" else np.float32
    A = {t: np.zeros((len(metinler), L, D * (3 if t == "kuantil_demeti" else 1)), dt)
         for t in TARZLAR}
    sayac = dict(ucuncu_basamak=0, Sg1=0)
    uz, mx = [], 0.0
    for i in range(0, len(metinler), bs):
        ch = [t if t.strip() else " " for t in metinler[i:i + bs]]
        b = tok(ch, padding=True, truncation=True, max_length=K["maxlen"],
                return_tensors="pt").to(DEV)
        m = b["attention_mask"]; uz += m.sum(1).tolist()
        with torch.no_grad():
            hs = mdl(**b, output_hidden_states=True).hidden_states
        valid = m.bool(); Sg = m.sum(1)
        sayac["Sg1"] += int((Sg == 1).sum())
        for l in range(L):
            r = tarz_hesapla(hs[l], valid, Sg, sayac)
            for t, v in r.items():
                x = v.float().cpu().numpy()
                mx = max(mx, float(np.abs(x).max()))
                A[t][i:i + len(ch), l] = x.astype(dt)
            del r
        del hs
    del tok, mdl; torch.cuda.empty_cache()
    return A, np.array(uz), mx, sayac


def kodla_hepsi():
    t0 = time.time(); P = KK.metinleri_al()
    for kk in KESKILER:
        for z, b in KK.PARCA:
            p, c, _, _ = P[(z, b)]
            for taraf, T in (("ebeveyn", p), ("cevap", c)):
                hedef = {t: f"{OT}/{kk}__{z}__{b}__{taraf}__{t}.npy" for t in TARZLAR}
                if all(os.path.exists(v) for v in hedef.values()):
                    yaz(f"    [atla] {kk}/{z}/{b}/{taraf}"); continue
                ts = time.time(); std_sifir = 0
                A, uz, mx, s = kodla_keski(kk, T)
                if s["Sg1"]:
                    yaz(f"{s['Sg1']}"
                        f"")
                for t, X in A.items():
                    if not np.isfinite(X).all():
                        raise E.KapsamHatasi(f"{kk}/{t}: SONLU DEGIL")
                    nrm = np.sqrt((X.reshape(len(X), -1).astype(np.float32) ** 2).sum(1))
                    n0 = int((nrm < 1e-8).sum())
                    if n0 and t != "std_havuz":
                        raise E.KapsamHatasi(f"{kk}/{t}: {n0} SIFIR VEKTÖR")
                    if n0:
                        std_sifir = n0
                    KK._kaydet(hedef[t], X)
                as_ = mx > 65504
                yaz(f"    {kk}/{z}/{b}/{taraf}: n={len(T)} L={A['son_token'].shape[1]} "
                    f"· token ort {uz.mean():.1f} maks {uz.max()} · carpan "
                    f"{int((uz >= KESKILER[kk]['maxlen']).sum())}/{len(uz)} · maks|h| {mx:.4g} "
                    f"({'fp16 TAVANINI ASAR' if as_ else 'tavan alti'}) · dtype "
                    f"{KESKILER[kk]['dtype']} · 3.basamak {s['ucuncu_basamak']} "
                    f"[{time.time()-ts:.0f}s]")
                if as_ and KESKILER[kk]["dtype"] == "float16":
                    raise E.KapsamHatasi(f"{kk}: fp16 tavani asildi ama fp16 yaziliyor — DUR")
                man_yaz("kodlama", {f"{kk}/{z}/{b}/{taraf}": dict(
                    n=len(T), L=int(A["son_token"].shape[1]), maks_h=mx, dtype=K_DT(kk),
                    carpan=int((uz >= KESKILER[kk]["maxlen"]).sum()),
                    ucuncu_basamak=int(s["ucuncu_basamak"]), Sg1=int(s["Sg1"]),
                    std_sifir=int(std_sifir))})
    yaz(f"  kodlama bitti ({time.time()-t0:.0f}s)")


def K_DT(kk):
    return KESKILER[kk]["dtype"]


def yigin(kk, z, b, tarz, L):
    P = np.load(f"{OT}/{kk}__{z}__{b}__ebeveyn__{tarz}.npy", mmap_mode="r")
    C = np.load(f"{OT}/{kk}__{z}__{b}__cevap__{tarz}.npy", mmap_mode="r")
    return np.asarray(P[:, L], np.float32), np.asarray(C[:, L], np.float32)


def nL(kk):
    return np.load(f"{OT}/{kk}__dis__dev__ebeveyn__son_token.npy", mmap_mode="r").shape[1]


def auc_mat(y, P):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise E.KapsamHatasi(f"AUC: tek sinif (n={len(y)}) — gecis DEGIL hata")
    R = rankdata(np.asarray(P, np.float64), axis=0)
    return (R[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _hucre(job):
    kk, z, tarz, L = job
    zem = "disapere" if z == "dis" else "kialo"
    egit, olc = ("train", "dev") if z == "dis" else ("train", "heldout")
    Z = E.ZEMIN[zem]
    if z == "dis":
        _, _, ytr, _ = E.disapere("train")
    else:
        (_, _, ytr, _), _ = E.kialo()
    Xtr = E.ozellik(*yigin(kk, z, egit, tarz, L))
    Xte = E.ozellik(*yigin(kk, z, olc, tarz, L))
    return (kk, z, tarz, L), E.probe(Xtr, ytr, Xte, Z["dengeli"], Z["iters"])


def egriler(isci=100):
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    _, _, yde, fde = E.disapere("dev")
    (_, _, _, _), (_, _, yhe, dhe) = E.kialo()
    Y = {"dis": (yde, fde), "kia": (yhe, dhe)}
    isler = [(kk, z, t, L) for kk in KESKILER for z in ("dis", "kia")
             for t in TARZLAR for L in range(nL(kk))]
    yaz(f"══ EGRILER · {len(isler)} hücre · {isci} isci ══")
    TAH = {}
    with ProcessPoolExecutor(max_workers=isci) as ex:
        for i, (k, p) in enumerate(ex.map(_hucre, isler, chunksize=2)):
            TAH[k] = p
            if (i + 1) % 200 == 0:
                yaz(f"    {i+1}/{len(isler)} ({time.time()-t0:.0f}s)")
    yaz(f"  hücreler bitti ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(SEED)
    R = {}
    for kk in KESKILER:
        for z in ("dis", "kia"):
            y, kume = Y[z]; L_ = nL(kk)
            P = np.stack([np.stack([TAH[(kk, z, t, l)] for l in range(L_)], 1)
                          for t in TARZLAR])
            A = np.stack([auc_mat(y, P[ti]) for ti in range(len(TARZLAR))])
            Rl = A.max(0) - A.min(0)
            sp = []
            for i in range(len(TARZLAR)):
                for j in range(i + 1, len(TARZLAR)):
                    a, b = A[i], A[j]
                    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
                    sp.append(float(np.corrcoef(ra, rb)[0, 1]))
            sira = np.argsort(np.argsort(-A, axis=0), axis=0)
            Rj = sira.sum(1).astype(float)
            W = float(12 * ((Rj - Rj.mean()) ** 2).sum() /
                      (L_ ** 2 * (len(TARZLAR) ** 3 - len(TARZLAR))))
            uk = np.unique(kume); ici, disi, sec = [], [], []
            for _ in range(R_YARI):
                pr = rng.permutation(len(uk)); yA = set(uk[pr[:len(uk) // 2]].tolist())
                mA = np.array([k in yA for k in kume]); mB = ~mA
                for m1, m2 in ((mA, mB), (mB, mA)):
                    if len(np.unique(y[m1])) < 2 or len(np.unique(y[m2])) < 2:
                        continue
                    a1 = np.stack([auc_mat(y[m1], P[ti][m1]) for ti in range(len(TARZLAR))])
                    ti, li = np.unravel_index(int(a1.argmax()), a1.shape)
                    ici.append(float(a1[ti, li])); sec.append((int(ti), int(li)))
                    disi.append(float(auc_mat(y[m2], P[ti][m2][:, [li]])[0]))
            d = np.array(disi); ii = np.array(ici)
            st, ct = np.unique([s[0] for s in sec], return_counts=True)
            sabit = float(auc_mat(y, np.ones((len(y), 1)))[0])
            nullR = []
            for _ in range(K_NULL):
                yp = y.copy()
                for c in uk:
                    m = kume == c
                    yp[m] = rng.permutation(y[m])
                An = np.stack([auc_mat(yp, P[ti]) for ti in range(len(TARZLAR))])
                nullR.append(float((An.max(0) - An.min(0)).mean()))
            nullR = np.array(nullR)
            R[f"{kk}|{z}"] = dict(
                L=L_, n=int(len(y)), kume=int(len(uk)),
                auc={TARZLAR[i]: [float(x) for x in A[i]] for i in range(len(TARZLAR))},
                aralik_ort=float(Rl.mean()), aralik_maks=float(Rl.max()),
                aralik_katman=[float(x) for x in Rl],
                spearman_ort=float(np.mean(sp)), spearman_min=float(np.min(sp)),
                kendall_W=W,
                secim_disi=float(d.mean()), secim_ici=float(ii.mean()),
                sisme=float(ii.mean() - d.mean()),
                secim_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                secilen_tarz={TARZLAR[int(s)]: int(c) for s, c in zip(st, ct)},
                farkli_tarz=int(len(st)), payda=int(len(d)),
                taban_ampirik=sabit,
                l0={TARZLAR[i]: float(A[i, 0]) for i in range(len(TARZLAR))},
                null_merkez=float(nullR.mean()), null_sd=float(nullR.std(ddof=1)),
                aralik_null_sd_biriminde=float((Rl.mean() - nullR.mean()) /
                                               max(nullR.std(ddof=1), 1e-9)),
                mde_null_sd=float(1.645 * nullR.std(ddof=1)))
            r = R[f"{kk}|{z}"]
            yaz(f"  {kk:<7} {z}: ARALIK ort {r['aralik_ort']:.4f} (null merkez "
                f"{r['null_merkez']:.4f} sd {r['null_sd']:.4f} ⇒ "
                f"**{r['aralik_null_sd_biriminde']:+.1f} null-sd**) · Spearman ort "
                f"{r['spearman_ort']:+.3f} · W {r['kendall_W']:.3f} · secim-DISI "
                f"{r['secim_disi']:.4f} (sisme {r['sisme']:+.4f}, {r['farkli_tarz']} tarz) "
                f"· taban {r['taban_ampirik']:.4f}")
    man_yaz("egriler", R)
    yaz(f"══ egriler bitti ({time.time()-t0:.0f}s)")


def _lasso_is(job):
    import secici_pilot as SP
    import butce_taramasi as BT
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    kk, z, oran = job
    zem = "disapere" if z == "dis" else "kialo"
    egit, olc = ("train", "dev") if z == "dis" else ("train", "heldout")
    Z = E.ZEMIN[zem]
    if z == "dis":
        _, _, ytr, _ = E.disapere("train"); _, _, yte, _ = E.disapere("dev")
    else:
        (_, _, ytr, _), (_, _, yte, _) = E.kialo()
    L_ = nL(kk); ell = int(round(oran * (L_ - 1)))
    bl_tr, bl_te = [], []
    for t in TARZLAR:
        Ptr, Ctr = yigin(kk, z, egit, t, ell)
        Pte, Cte = yigin(kk, z, olc, t, ell)
        if t == "kuantil_demeti":
            d = Ptr.shape[1] // 3
            if d > Ptr.shape[0] - 1:
                raise E.KapsamHatasi(
                    f"{d}"
                    f"{Ptr.shape[0]} {Ptr.shape[0]-1}"
                    "")
            pp = PCA(d, random_state=0).fit(Ptr); pc = PCA(d, random_state=0).fit(Ctr)
            Ptr, Pte = pp.transform(Ptr), pp.transform(Pte)
            Ctr, Cte = pc.transform(Ctr), pc.transform(Cte)
        bl_tr.append(E.ozellik(Ptr, Ctr)); bl_te.append(E.ozellik(Pte, Cte))
    G = bl_tr[0].shape[1]
    if any(b.shape[1] != G for b in bl_tr):
        raise E.KapsamHatasi("")
    Xtr = np.concatenate(bl_tr, 1).astype(np.float32)
    Xte = np.concatenate(bl_te, 1).astype(np.float32)
    del bl_tr, bl_te
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    bul, maxnz, lmax = BT.yol_destekler(Xtr, ytr.astype(np.float32), D_IZGARA, G=G)
    out = {}
    for D in D_IZGARA:
        if D not in bul:
            continue
        ii = np.asarray(bul[D][0], int)
        cf = (G * ii[:, None] + np.arange(G)).ravel()
        m = LogisticRegression(C=0.5, max_iter=Z["iters"],
                               class_weight=("balanced" if Z["dengeli"] else None)
                               ).fit(Xtr[:, cf], ytr)
        out[str(D)] = dict(tarzlar=[TARZLAR[i] for i in ii],
                           auc=float(auc_mat(yte, m.predict_proba(Xte[:, cf])[:, [1]])[0]),
                           merkez=int(sum(TARZLAR[i] in MERKEZ for i in ii)),
                           uc=int(sum(TARZLAR[i] in UC for i in ii)))
    return f"{kk}|{z}|{oran:.2f}", dict(katman=ell, L=L_, G=int(G), maks_nz=int(maxnz),
                                        lmax=float(lmax), butce=out)


def lasso(isci=6):
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    isler = [(kk, z, o) for kk in KESKILER for z in ("dis", "kia") for o in DERINLIKLER]
    yaz(f"══ LASSO (grup = tarz) · {len(isler)} yol · {isci} isci ══")
    R = {}
    with ProcessPoolExecutor(max_workers=isci) as ex:
        fut = {ex.submit(_lasso_is, j): j for j in isler}
        import concurrent.futures as CF
        for f in CF.as_completed(fut):
            j = fut[f]
            try:
                k, v = f.result()
            except E.KapsamHatasi as ex_:
                k = f"{j[0]}|{j[1]}|{j[2]:.2f}"
                R[k] = dict(uygulanamaz=str(ex_)); yaz(f"  {k}: ⚠ {ex_}"); continue
            R[k] = v
            b2 = v["butce"].get("2", {})
            yaz(f"  {k}: ℓ{v['katman']}/{v['L']} G={v['G']} · D=2 → "
                f"{b2.get('tarzlar')} (merkez {b2.get('merkez')} / uc {b2.get('uc')}) "
                f"AUC {b2.get('auc', float('nan')):.4f}")
    man_yaz("lasso", R)
    yaz(f"══ lasso bitti ({time.time()-t0:.0f}s)")


def kapi():
    yaz("══ KAPI · ön-kayit §11 ══")
    rng = np.random.default_rng(0); n = 400
    y = (rng.random(n) < 0.3).astype(int)
    yaz(f"  [1] SABIT öznitelik → AUC {auc_mat(y, np.ones((n,1)))[0]:.4f} (beklenen 0,5000)")
    try:
        auc_mat(np.zeros(n, int), rng.random((n, 1))); ok = False
    except E.KapsamHatasi:
        ok = True
    yaz(f"  [2] TEK SINIF → {'KapsamHatasi (dogru)' if ok else '⚠ SESSIZ GECTI'}")
    P = rng.random((n, 3)) + y[:, None] * 2.0
    a = auc_mat(y, np.repeat(P[:, [0]], 8, axis=1))
    yaz(f"  [3] ÖZDES tarzlar → aralik {a.max()-a.min():.6f} (beklenen 0)")
    a2 = auc_mat(y, np.column_stack([P, rng.random((n, 5))]))
    yaz(f"  [4] FARKLI tarzlar → aralik {a2.max()-a2.min():.4f} (>0 olmali)")
    import butce_taramasi as BT
    try:
        BT.yol_destekler(np.ones((50, 24), np.float32), (rng.random(50) < .5).astype(np.float32),
                         [2], G=3)
        yaz("  [5] group-lasso dejenere girdiyi KABUL ETTI — ⚠")
    except Exception as ex:
        yaz(f"  [5] group-lasso dejenere girdide → {type(ex).__name__} (DOGRU)")
    yaz("  KAPI bitti.")


def _tarz_disi(y, kume, P_tarz, R=R_YARI, seed=SEED):
    uk = np.unique(kume); rng = np.random.default_rng(seed)
    sonuc = {t: [] for t in P_tarz}
    for _ in range(R):
        pr = rng.permutation(len(uk)); yA = set(uk[pr[:len(uk) // 2]].tolist())
        mA = np.array([k in yA for k in kume]); mB = ~mA
        for m1, m2 in ((mA, mB), (mB, mA)):
            if len(np.unique(y[m1])) < 2 or len(np.unique(y[m2])) < 2:
                continue
            for t, P in P_tarz.items():
                a1 = auc_mat(y[m1], P[m1])
                L = int(a1.argmax())
                sonuc[t].append(float(auc_mat(y[m2], P[m2][:, [L]])[0]))
    return {t: np.array(v) for t, v in sonuc.items()}


def _prediction_is(job):
    from sklearn.decomposition import PCA
    kk, z = job
    zem = "disapere" if z == "dis" else "kialo"
    egit, olc = ("train", "dev") if z == "dis" else ("train", "heldout")
    Z = E.ZEMIN[zem]
    if z == "dis":
        _, _, ytr, _ = E.disapere("train"); _, _, y, kume = E.disapere("dev")
    else:
        (_, _, ytr, _), (_, _, y, kume) = E.kialo()
    L_ = nL(kk)
    dl = sorted({int(round(o * (L_ - 1))) for o in DERINLIKLER})

    def sutun(tarz, katlar, pca=False):
        cols = []
        for l in katlar:
            Ptr, Ctr = yigin(kk, z, egit, tarz, l)
            Pte, Cte = yigin(kk, z, olc, tarz, l)
            if pca:
                nc = int(min(Ptr.shape[1] // 3, Ptr.shape[0] - 1, Ptr.shape[1]))
                pp = PCA(nc, svd_solver="randomized", random_state=0).fit(Ptr)
                pc = PCA(nc, svd_solver="randomized", random_state=0).fit(Ctr)
                Ptr, Pte = pp.transform(Ptr), pp.transform(Pte)
                Ctr, Cte = pc.transform(Ctr), pc.transform(Cte)
            cols.append(E.probe(E.ozellik(Ptr, Ctr), ytr, E.ozellik(Pte, Cte),
                                Z["dengeli"], Z["iters"]))
        return np.stack(cols, 1)

    tam = range(L_)
    Pham = {"kuantil_demeti": sutun("kuantil_demeti", tam),
            "duz_ortalama": sutun("duz_ortalama", tam)}
    Sham = _tarz_disi(y, kume, Pham)
    Ppca = {"kuantil_pca": sutun("kuantil_demeti", dl, pca=True),
            "duz_ortalama": sutun("duz_ortalama", dl)}
    Spca = _tarz_disi(y, kume, Ppca)
    out = {}
    for ad, S, a in (("ham", Sham, "kuantil_demeti"), ("pca_5derinlik", Spca, "kuantil_pca")):
        d = S[a] - S["duz_ortalama"]
        out[ad] = dict(kuantil=float(S[a].mean()), duz=float(S["duz_ortalama"].mean()),
                       delta=float(d.mean()),
                       ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                       payda=int(len(d)))
    out["_pca_kapsam"] = dict(derinlikler=dl, n_components="min(d, n_train−1)")
    return f"{kk}|{z}", out


def prediction(isci=8):
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    isler = [(kk, z) for kk in KESKILER for z in ("dis", "kia")]
    yaz(f"══ B2 OKUMASI · secim-disi + esli CI · {len(isler)} hücre ══")
    R = {}
    with ProcessPoolExecutor(max_workers=isci) as ex:
        for k, v in ex.map(_prediction_is, isler):
            R[k] = v
            h = v["ham"]; p = v["pca_5derinlik"]
            tut = (h["ci95"][0] <= 0 <= h["ci95"][1]) or h["delta"] < 0
            yaz(f"  {k}: HAM kuantil {h['kuantil']:.4f} · düz {h['duz']:.4f} · Δ "
                f"{h['delta']:+.4f} [{h['ci95'][0]:+.4f},{h['ci95'][1]:+.4f}] "
                f"⇒ B2 {'TUTAR' if tut else 'DÜSER'}  |  PCA(5-derinlik) Δ {p['delta']:+.4f} "
                f"[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]")
    man_yaz("prediction_B2", R)
    yaz(f"══ B2 bitti ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kapi", action="store_true")
    ap.add_argument("--kodla", action="store_true")
    ap.add_argument("--egri", action="store_true")
    ap.add_argument("--lasso", action="store_true")
    ap.add_argument("--prediction", action="store_true")
    ap.add_argument("--isci", type=int, default=100)
    A = ap.parse_args()
    if A.kapi:
        kapi()
    if A.kodla:
        kodla_hepsi()
    if A.egri:
        egriler(A.isci)
    if A.lasso:
        lasso()
    if A.prediction:
        prediction()
    if not (A.kapi or A.kodla or A.egri or A.lasso or A.prediction):
        ap.print_help()
