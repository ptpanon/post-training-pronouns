#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
from prefix_muhurlu_kosu import auc_np, burrows_z
from prefix_korpus_pilot import tekrar_orani, KELIME
import prefix_generation_pilot as UP
import prefix_reviewer_bootstrap as HB
import prefix_zengin_gomu as ZG

OUT = __DNH_DATA__ + "/onek_korpus/steering_pilot"
CIKTI = f"{ROOT}/unreleased/STEERING_URETIM_2026-08-08.json"
FIG = f"{ROOT}/unreleased/fig"
MODEL = __DNH_DATA__ + "/.hf_local/hub/models--mistralai--Mistral-7B-v0.3"
LAMDA = (0.0, 0.25, 0.5, 1.0)
KATMAN_G, KATMAN_E5 = 16, 23
K_YON, N_ISTEM, YENI_JETON = 8, 96, 160
YON_KIPI = "anizotropik"
SEED, Z, ONEK_KELIME = 20260808, 1.645, 40
B = 200


def birim(v):
    return v / max(np.linalg.norm(v), 1e-12)


def istemler(rng):
    met, y, kume = ZG.zemin_verisi()["dis"]
    Zb = burrows_z(met)
    pc1 = np.linalg.svd(Zb - Zb.mean(0), full_matrices=False)[2][0]
    b = (Zb - Zb.mean(0)) @ pc1
    yB = (b > np.median(b)).astype(np.int8)
    poz, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = N_ISTEM // 2
    ix = np.concatenate([rng.choice(poz, k, replace=False),
                         rng.choice(neg, k, replace=False)])
    rng.shuffle(ix)
    ist = [" ".join(KELIME.findall(met[i])[:ONEK_KELIME]) for i in ix]
    return ist, y[ix], yB[ix], kume[ix], ix, (met, y, yB, kume)


def yonler_dis(X, y, yB, kume):
    yon, atlanan = {}, {}
    for ad, lab in (("A", y), ("B", yB)):
        W, kac = HB.yonler(X, np.asarray(lab, dtype=np.int8), kume,
                           np.random.default_rng(SEED), "kume")
        yon[ad] = birim(W.mean(0)); atlanan[ad] = int(kac)
    return yon, atlanan


def uret(a):
    global LAMDA, K_YON, N_ISTEM, OUT
    if a.prova:
        LAMDA, K_YON, N_ISTEM = (0.0, 0.5), 1, 8
        OUT = OUT + "_prova"
        print("★ PROVA KIPI — boru hatti sinanir, sayi üretilmez")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    os.makedirs(OUT, exist_ok=True)
    kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="steering_uretim")
    rng = np.random.default_rng(SEED)
    ist, y, yB, kume, ix, _ = istemler(rng)
    import glob as _g
    snap = sorted(_g.glob(f"{MODEL}/snapshots/*"))[-1]
    tok = AutoTokenizer.from_pretrained(snap)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float16).to(a.dev).eval()
    kat = model.model.layers[KATMAN_G]
    d_g = model.config.hidden_size
    print(f"STEERING ÜRETIM · {model.config.num_hidden_layers} katman · d={d_g} · "
          f"itme ℓ{KATMAN_G} · istem {len(ist)} · λ {LAMDA} · yön {K_YON}")

    E = tok(ist, return_tensors="pt", padding=True, truncation=True, max_length=512)
    E = {k: v.to(a.dev) for k, v in E.items()}
    with torch.no_grad():
        cikti = model(**E, output_hidden_states=True)
    Hg = cikti.hidden_states[KATMAN_G][:, -1, :].float().cpu().numpy()
    del cikti
    torch.cuda.empty_cache()
    mu = Hg.mean(0)
    yaricap = float(np.median(np.linalg.norm(Hg - mu, axis=1)))
    C = Hg - mu
    if YON_KIPI == "izotropik":
        V = np.stack([birim(rng.standard_normal(C.shape[1])) for _ in range(K_YON)])
    else:
        V = np.stack([birim(rng.standard_normal(len(C)) @ C) for _ in range(K_YON)])
    print(f"  bulut: yaricap {yaricap:.2f} · yön demeti {V.shape} · "
          f"ikili |kosinüs| medyani {np.median(np.abs(V @ V.T)[np.triu_indices(K_YON,1)]):.3f}")

    durum = {"vek": None}

    def kanca(mod, girdi, cikti):
        if durum["vek"] is None:
            return cikti
        h = cikti[0] if isinstance(cikti, tuple) else cikti
        h = h + durum["vek"].to(h.dtype)
        return (h,) + tuple(cikti[1:]) if isinstance(cikti, tuple) else h

    kanca_once = len(kat._forward_hooks)
    kanca_tutamac = kat.register_forward_hook(kanca)

    hucreler = [("l0", 0.0, -1)] + [(f"lam{l}_v{k}", l, k)
                                    for l in LAMDA if l > 0 for k in range(K_YON)]
    yol = f"{OUT}/uretim.jsonl"
    tmp = yol + f".tmp.{os.getpid()}"
    fh = open(tmp, "w", encoding="utf-8")
    n_sat, t0 = 0, time.time()
    for ad, lam, k in hucreler:
        durum["vek"] = (None if lam == 0 else
                        torch.tensor(lam * yaricap * V[k], device=a.dev,
                                     dtype=torch.float32))
        torch.manual_seed(SEED)
        for i in range(0, len(ist), a.yigin):
            par = ist[i:i + a.yigin]
            enc = tok(par, return_tensors="pt", padding=True, truncation=True,
                      max_length=512)
            enc = {kk: v.to(a.dev) for kk, v in enc.items()}
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=UP.SICAKLIK,
                                   top_p=UP.TOP_P, max_new_tokens=YENI_JETON,
                                   pad_token_id=tok.pad_token_id)
            yeni = g[:, enc["input_ids"].shape[1]:]
            for j, satir in enumerate(yeni):
                t = tok.decode(satir, skip_special_tokens=True)
                fh.write(json.dumps(dict(hucre=ad, lam=lam, yon=k, istem_i=int(i + j),
                                         dis_ix=int(ix[i + j]), y=int(y[i + j]),
                                         yB=int(yB[i + j]), metin=t,
                                         n_jeton=int((satir != tok.pad_token_id).sum())),
                                    ensure_ascii=False) + "\n")
                n_sat += 1
        print(f"  [{ad:12s}] λ={lam} yön={k} · toplam satir {n_sat} · "
              f"{time.time()-t0:.0f}s", flush=True)
    fh.close()
    kanca_tutamac.remove()
    kanca_kalan = len(kat._forward_hooks)
    os.replace(tmp, yol)
    man = dict(damga=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               motor="HF transformers", model=os.path.basename(snap),
               katman_itme=KATMAN_G, d=d_g, yaricap=yaricap, lamda=list(LAMDA),
               K_yon=K_YON, n_istem=len(ist), yeni_jeton=YENI_JETON, seed=SEED,
               sicaklik=UP.SICAKLIK, top_p=UP.TOP_P, n_hucre=len(hucreler),
               n_satir=n_sat, saniye=round(time.time() - t0, 1),
               sinif="",
               yon_kipi=YON_KIPI, kanca_once=int(kanca_once),
               kanca_kalan=int(kanca_kalan),
               kanca_cikarildi=bool(kanca_kalan == kanca_once),
               yon_sinifi=("izotropik plasebo (küresel; TESISAT provasi)"
                           if YON_KIPI == "izotropik" else
                           "anizotropi-esli plasebo (bulutun kendi satirlarindan)"))
    json.dump(man, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)
    payda("steering_uretim", n_satir=n_sat, n_hucre=len(hucreler), n_istem=len(ist),
          bekle={"n_satir": len(hucreler) * len(ist),
                 "n_hucre": len(hucreler), "n_istem": N_ISTEM})
    print(f"  → {yol} · {n_sat} satir / {man['saniye']}s")


def oku(a):
    import prefix_layer_gomuleri as KG
    import prefix_ucuncu_eksen as UE
    KG.DEV = a.dev
    kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=False, etiket="steering_okuma")
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rng = np.random.default_rng(SEED)
    ist, y, yB, kume, ix, (met_tam, y_tam, yB_tam, kume_tam) = istemler(rng)
    H = np.load(__DNH_DATA__ + "/onek_korpus/hakem/H_c1a__dis.fp16.npy",
                mmap_mode="r")
    Xh = np.asarray(H[:, KATMAN_E5, :], dtype=np.float32)
    mu, sd = Xh.mean(0), np.maximum(Xh.std(0), 1e-9)
    X = (Xh - mu) / sd
    yon, atlanan = yonler_dis(X, y_tam, yB_tam, kume_tam)
    print(f"STEERING OKUMA · damga {damga} · yön A/B kuruldu (atlanan {atlanan})")

    S = [json.loads(l) for l in open(f"{OUT}/uretim.jsonl", encoding="utf-8")]
    met = [(r["metin"] or "").strip() for r in S]
    Hg, sayac, sn = KG.kodla_zemin("steering", "pilot", met)
    Xg = (np.asarray(Hg[:, KATMAN_E5, :], dtype=np.float32) - mu) / sd
    sA, sB = Xg @ yon["A"], Xg @ yon["B"]
    soz, frek, _ = UE.hedge_sozlugu()
    D, _ = UE.dominance()
    hed, dom, isabet = UE.olcum(met, soz, D)
    tek = np.array([tekrar_orani(t) for t in met])
    boy = np.array([len(KELIME.findall(t.lower())) for t in met])
    hucre = np.array([r["hucre"] for r in S])
    lam = np.array([r["lam"] for r in S], dtype=float)
    yy = np.array([r["y"] for r in S], dtype=np.int8)
    yyB = np.array([r["yB"] for r in S], dtype=np.int8)

    tab = {}
    for h in sorted(set(hucre.tolist())):
        m = hucre == h
        tab[h] = dict(n=int(m.sum()), lam=float(lam[m][0]),
                      AUC_A=round(float(auc_np(yy[m], sA[m])), 4),
                      AUC_B=round(float(auc_np(yyB[m], sB[m])), 4),
                      D=round(float(np.nanmean(dom[m])), 4),
                      hedge=round(float(hed[m].mean()), 4),
                      tekrar=round(float(tek[m].mean()), 4),
                      boy=round(float(boy[m].mean()), 1))
    t0 = tab["l0"]
    print(f"  λ=0 capasi: AUC_A {t0['AUC_A']:.4f} · AUC_B {t0['AUC_B']:.4f} · "
          f"D {t0['D']:.4f} · tekrar {t0['tekrar']:.3f} · boy {t0['boy']:.0f}")

    m0 = hucre == "l0"
    kg = kume[np.array([r["istem_i"] for r in S])[m0]]
    ks = sorted(set(kg.tolist())); ixk = {k: np.flatnonzero(kg == k) for k in ks}
    taban = {}
    for ad, sk, lb in (("A", sA[m0], yy[m0]), ("B", sB[m0], yyB[m0])):
        bo = []
        for _ in range(B):
            sel = rng.choice(len(ks), size=len(ks), replace=True)
            j = np.concatenate([ixk[ks[i]] for i in sel])
            if len(set(lb[j].tolist())) < 2:
                continue
            bo.append(auc_np(lb[j], sk[j]))
        taban[ad] = dict(sd=round(float(np.std(bo, ddof=1)), 4),
                         MDE=round(float(Z * np.std(bo, ddof=1)), 4), n_boot=len(bo))
        print(f"  ÖRNEKLEME TABANI {ad}: bootstrap sd {taban[ad]['sd']:.4f} ⇒ "
              f"MDE_taban {taban[ad]['MDE']:.4f}")

    mud = {}
    for l in LAMDA:
        if l == 0:
            continue
        hs = [f"lam{l}_v{k}" for k in range(K_YON)]
        o = {}
        for ad, alan in (("A", "AUC_A"), ("B", "AUC_B"), ("D", "D")):
            d = np.array([tab[h][alan] - t0[alan] for h in hs])
            o[ad] = dict(ort=round(float(d.mean()), 4), sd=round(float(d.std(ddof=1)), 4),
                         MDE=round(float(Z * d.std(ddof=1)), 4),
                         azami_mutlak=round(float(np.abs(d).max()), 4))
        o["tekrar"] = round(float(np.mean([tab[h]["tekrar"] for h in hs])), 4)
        o["boy"] = round(float(np.mean([tab[h]["boy"] for h in hs])), 1)
        mud[str(l)] = o
        print(f"  λ={l}: ΔAUC_A {o['A']['ort']:+.4f} ± {o['A']['sd']:.4f} ⇒ "
              f"**MDE_müdahale_A {o['A']['MDE']:.4f}** · ΔAUC_B {o['B']['ort']:+.4f} "
              f"± {o['B']['sd']:.4f} ⇒ **{o['B']['MDE']:.4f}** · tekrar {o['tekrar']:.3f}"
              f" · boy {o['boy']:.0f}")

    ref = [json.loads(l) for l in open(
        __DNH_DATA__ + "/onek_korpus/muhurlu/ana__mistral.jsonl",
        encoding="utf-8")][:600]
    rt = np.array([tekrar_orani(r["metin_160"]) for r in ref])
    rb = np.array([len(KELIME.findall(r["metin_160"].lower())) for r in ref])
    kopru = dict(muhurlu_tekrar_ort=round(float(rt.mean()), 4),
                 muhurlu_tekrar_p95=round(float(np.percentile(rt, 95)), 4),
                 muhurlu_boy_ort=round(float(rb.mean()), 1),
                 pilot_l0_tekrar=t0["tekrar"], pilot_l0_boy=t0["boy"],
                 n_ref=len(ref))
    kopru["l0_bandda"] = bool(t0["tekrar"] <= kopru["muhurlu_tekrar_p95"])
    print(f"  KÖPRÜ (§7.4): mühürlü tekrar {kopru['muhurlu_tekrar_ort']:.3f} "
          f"(p95 {kopru['muhurlu_tekrar_p95']:.3f}) ↔ pilot λ=0 {t0['tekrar']:.3f} ⇒ "
          f"{'BANDDA' if kopru['l0_bandda'] else 'BAND DISI'}")

    R = dict(damga=damga, sinif="",
             katman_itme=KATMAN_G, katman_e5=KATMAN_E5, n_satir=len(S),
             n_istem=N_ISTEM, K_yon=K_YON, lamda=list(LAMDA), hucreler=tab,
             ornekleme_tabani=taban, mde_mudahale=mud, kopru_74=kopru,
             isabet_warriner=round(float(np.isfinite(dom).mean()), 4),
             yon_atlanan=atlanan,
             not_=""
                  "")
    payda("steering_okuma", n_satir=len(S), n_hucre=len(tab), n_lamda=len(LAMDA),
          bekle={"n_satir": (1 + 3 * K_YON) * N_ISTEM, "n_hucre": 1 + 3 * K_YON})
    json.dump(R, open(CIKTI, "w"), ensure_ascii=False, indent=1)
    print(f"\n→ {CIKTI}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for j, alan in enumerate(("AUC_A", "AUC_B")):
            for l in LAMDA:
                if l == 0:
                    ax[j].axhline(t0[alan], color="k", lw=1, ls=":")
                    continue
                v = [tab[f"lam{l}_v{k}"][alan] for k in range(K_YON)]
                ax[j].scatter([l] * len(v), v, s=26, alpha=.75)
            ax[j].set_xlabel("λ (bulut yaricapi biriminde)")
            ax[j].set_title(f"{alan} · nokta cizgi = λ=0 capasi"); ax[j].grid(alpha=.3)
        fig.suptitle("STEERING ÜRETIM-PILOTU · PLASEBO itmelerinin üretim dagilimi")
        fig.tight_layout(); fig.savefig(f"{FIG}/steering_uretim.png", dpi=130)
        print(f"  figür → {FIG}/steering_uretim.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", required=True, choices=("uret", "oku"))
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, required=True)
    ap.add_argument("--yigin", type=int, default=32)
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--prova", action="store_true")
    a = ap.parse_args()
    (uret(a) if a.asama == "uret" else oku(a))
