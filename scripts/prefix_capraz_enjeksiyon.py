#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import argparse
import itertools

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
import prefix_generation_pilot as UP
import prefix_steering_generation as SU
import prefix_yazar_bataryasi as YB
import prefix_harsh_vekil as HV
import prefix_lineage_transferi as ST
import prefix_ladder as MD

OUT = __DNH_DATA__ + "/onek_korpus/capraz"
HAVUZ = YB.OUT
CIKTI = f"{ROOT}/unreleased/CAPRAZ_ENJEKSIYON_2026-08-09.json"
MUTLAK_TABAN = 0.0
HARITA1 = f"{ROOT}/unreleased/TAVAN_HARITASI_2026-08-09.json"

DERINLIK = 0.50
KL_HEDEF = ST.KL_HEDEF
K_PLASEBO = 4
SEED, YENI_JETON = 20260809, 160
P_CAPA = 95
Z = 1.645
BAR_KAT = 0.50
EN_AZ_YON_SATIR = 40
ADLAR = ("ÖLCÜLEMEZ", "KÖSEGEN-BASKIN", "SOY-BLOK", "KARISIK")

BLOKLAR = {
    "b4096": dict(d=4096, birincil=True, uyeler=[
        dict(ad="m7b", hf="mistralai/Mistral-7B-v0.3", aile="mistral", soy="m7",
             terbiye="base", tavan="m7|base|ham"),
        dict(ad="m7i", hf="mistralai/Mistral-7B-Instruct-v0.3", aile="mistral", soy="m7",
             terbiye="it", tavan="m7|it|ham"),
        dict(ad="mn8b", hf="mistralai/Ministral-3-8B-Base-2512", aile="mistral", soy="mn8",
             terbiye="base", tavan="mn8|base|ham"),
        dict(ad="mn8i", hf="mistralai/Ministral-3-8B-Instruct-2512-BF16", aile="mistral",
             soy="mn8", terbiye="it", tavan="mn8|it|ham"),
        dict(ad="yq9b", hf="Qwen/Qwen3.5-9B-Base", aile="qwen", soy="yq9",
             terbiye="base", tavan="yq9|base|ham"),
        dict(ad="yq9", hf="Qwen/Qwen3.5-9B", aile="qwen", soy="yq9",
             terbiye="it", tavan="yq9|it|ham"),
    ]),
    "b3840": dict(d=3840, birincil=False, uyeler=[
        dict(ad="g312b", hf="google/gemma-3-12b-pt", aile="gemma", soy="g312",
             terbiye="base", tavan="g312|base|ham"),
        dict(ad="g312i", hf="google/gemma-3-12b-it", aile="gemma", soy="g312",
             terbiye="it", tavan="g312|it|ham"),
        dict(ad="yg12b", hf="google/gemma-4-12b", aile="gemma", soy="yg12",
             terbiye="base", tavan="yg12|base|ham"),
        dict(ad="yg12", hf="google/gemma-4-12B-it", aile="gemma", soy="yg12",
             terbiye="it", tavan="yg12|it|ham"),
    ]),
}


def uyeler():
    return [dict(u, blok=b) for b, B in BLOKLAR.items() for u in B["uyeler"]]


def kat_of(kaynak, hedef):
    if kaynak["ad"] == hedef["ad"]:
        return "kosegen"
    if kaynak["soy"] == hedef["soy"]:
        return "soy"
    if kaynak["aile"] == hedef["aile"]:
        return "aile"
    return "yabanci"


def verdict_matris(f_kosegen, f_soy, f_aile, f_yabanci, kapi_tamam):
    v = [f_kosegen, f_soy, f_aile, f_yabanci]
    if (not kapi_tamam) or any(x is None or not np.isfinite(x) for x in v):
        return "ÖLCÜLEMEZ"
    ic = max(f_soy, f_aile)
    if f_kosegen >= BAR_KAT and ic < BAR_KAT and f_yabanci < BAR_KAT:
        return "KÖSEGEN-BASKIN"
    if f_kosegen >= BAR_KAT and ic >= BAR_KAT and f_yabanci < BAR_KAT:
        return "SOY-BLOK"
    return "KARISIK"


def _prova():
    V = [("kapi düstü", (1, 0, 0, 0, False), "ÖLCÜLEMEZ"),
         ("NaN", (float("nan"), 0, 0, 0, True), "ÖLCÜLEMEZ"),
         ("None", (None, 0, 0, 0, True), "ÖLCÜLEMEZ"),
         ("yalniz kösegen", (1.0, 0.0, 0.2, 0.0, True), "KÖSEGEN-BASKIN"),
         ("aile-ici de", (1.0, 0.8, 0.6, 0.1, True), "SOY-BLOK"),
         ("yabancida da", (1.0, 0.8, 0.6, 0.9, True), "KARISIK"),
         ("kösegen ölü", (0.0, 0.0, 0.0, 0.0, True), "KARISIK"),
         ("tam sinir 0,50", (0.5, 0.49, 0.49, 0.49, True), "KÖSEGEN-BASKIN")]
    ok = 0
    for ad, arg, bek in V:
        h = verdict_matris(*arg)
        ok += h == bek
        print(f"  prova {ad:18s} → {h:15s} (bekle {bek})")
    assert ok == len(V), "prova DÜSTÜ"
    import ast
    ag = ast.parse(open(__file__, encoding="utf-8").read())
    kodda = {x.value.value for d in ast.walk(ag)
             if isinstance(d, ast.FunctionDef) and d.name == "verdict_matris"
             for x in ast.walk(d)
             if isinstance(x, ast.Return) and isinstance(x.value, ast.Constant)}
    print(f"  [D44] kodda {len(kodda)} · mühürde {len(ADLAR)} · "
          f"fark {sorted(kodda ^ set(ADLAR)) or '—'}")
    if kodda != set(ADLAR):
        raise SystemExit("★ D44 CAPRAZ SAYIM DÜSTÜ — kosu BASLAMAZ")
    a = dict(ad="m7i", soy="m7", aile="mistral")
    assert kat_of(a, a) == "kosegen"
    assert kat_of(dict(ad="m7b", soy="m7", aile="mistral"), a) == "soy"
    assert kat_of(dict(ad="mn8i", soy="mn8", aile="mistral"), a) == "aile"
    assert kat_of(dict(ad="yq9", soy="yq9", aile="qwen"), a) == "yabanci"
    n = {b: len(B["uyeler"]) for b, B in BLOKLAR.items()}
    print(f"  blok hücreleri: b4096 {n['b4096']}×{n['b4096']}={n['b4096']**2} · "
          f"b3840 {n['b3840']}×{n['b3840']}={n['b3840']**2}")
    print("  ✓ prova gecti — sekiz sinir girdisi · D44 4=4 · dört kat")


def esdegerlik(a):
    import torch
    man = json.load(open(f"{ST.OUT}/manifest_m7i.json"))
    ref = man["kalibrasyon"]["l16_gercek"]
    Z0 = np.load(f"{ST.OUT}/d_taban.npz")
    kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="capraz_esdegerlik")
    tok, model, _ = ST._yukle("mistralai/Mistral-7B-Instruct-v0.3", a.dev)
    L, _d = ST._L_of(model)
    kat = int(round(DERINLIK * L))
    S = [json.loads(l) for l in open(f"{HAVUZ}/uretim_m7i_ham.jsonl", encoding="utf-8")]
    X = ST._akim(model, tok, a.dev, [s["metin"] for s in S], [kat], a.yigin)[kat]
    yaricap = float(np.median(np.linalg.norm(X - X.mean(0), axis=1)))
    from prefix_korpus_pilot import CERCEVE, DURUS
    T = json.load(open(__DNH_DATA__ + "/onek_korpus/pilot_tezler.json",
                       encoding="utf-8"))
    ist = [CERCEVE["sakin"].format(T=t["tez"], P=DURUS[d]) for t in T for d in DURUS]
    E = ST.Enjektor(model, tok, a.dev, ist)
    kanca_h = ST._katman_listesi(model)[0][kat].register_forward_hook(E.kanca)
    lam, iz = E.lam_icin(Z0["yon"], yaricap)
    kanca_h.remove()
    fark_lam = abs(round(lam, 4) - ref["lam"])
    fark_kl = abs(iz[-1][1] - ref["kl"])
    print(f"  referans λ={ref['lam']:.4f} KL={ref['kl']:.6f}  ↔  yeniden "
          f"λ={lam:.4f} KL={iz[-1][1]:.6f}  ⇒ |Δλ|={fark_lam:.4f} |ΔKL|={fark_kl:.6f}")
    gecti = fark_lam <= 1e-4 and fark_kl <= 5e-4
    payda("capraz_esdegerlik", n_hucre=1, n_adim=len(iz), red_sapma=int(not gecti),
          bekle={"n_hucre": 1})
    out = dict(referans=ref, yeniden=dict(lam=round(lam, 4), kl=iz[-1][1], iz=iz),
               fark_lam=round(fark_lam, 6), fark_kl=round(fark_kl, 6), gecti=bool(gecti))
    json.dump(out, open(f"{ROOT}/unreleased/ESDEGERLIK_ENJEKTOR_2026-08-09.json", "w"),
              ensure_ascii=False, indent=1)
    if not gecti:
        raise SystemExit("")
    print("")
    del model
    torch.cuda.empty_cache()


def yonler(a):
    os.makedirs(OUT, exist_ok=True)
    P = {"sert": MD.sabit_parcalar(MD.BASAMAKLAR["B3"])}
    from prefix_korpus_pilot import CERCEVE
    P["sakin"] = MD.sabit_parcalar(CERCEVE["sakin"])
    D, karne, eksik = {}, {}, []
    for u in uyeler():
        yol = f"{HAVUZ}/uretim_{u['ad']}_ham_b3.jsonl"
        akt = f"{HAVUZ}/akt_{u['ad']}_ham_b3.npz"
        if not (os.path.exists(yol) and os.path.exists(akt)):
            eksik.append(u["ad"])
            continue
        S = [json.loads(l) for l in open(yol, encoding="utf-8")]
        Z = np.load(akt)
        man = json.load(open(f"{HAVUZ}/manifest_{u['ad']}_ham_b3.json"))
        kat = int(round(DERINLIK * man["L"]))
        X = np.asarray(Z[f"h{kat}"], dtype=np.float64)
        lab = np.array([s["ton_hedef"] for s in S], dtype=object)
        yk = np.array([MD.istem_yankisi(s["metin"], P[s["ton_hedef"]])[0] for s in S])
        m1, m0 = (lab == "sert") & ~yk, (lab == "sakin") & ~yk
        if m1.sum() < EN_AZ_YON_SATIR or m0.sum() < EN_AZ_YON_SATIR:
            eksik.append(u["ad"] + "(yankisiz satir az)")
            continue
        D[u["ad"]] = SU.birim(X[m1].mean(0) - X[m0].mean(0))
        karne[u["ad"]] = dict(katman=kat, L=man["L"], d=int(X.shape[1]),
                              n_sert=int(m1.sum()), n_sakin=int(m0.sum()),
                              yanki_orani=round(float(yk.mean()), 4),
                              norm_ham=round(float(np.linalg.norm(
                                  X[m1].mean(0) - X[m0].mean(0))), 4))
        print(f"  {u['ad']:6s} ℓ{kat:<3d} d={X.shape[1]} · sert {int(m1.sum())} "
              f"sakin {int(m0.sum())} · yanki {yk.mean():.4f} · "
              f"‖d‖ {karne[u['ad']]['norm_ham']:.3f}")
    payda("capraz_yon", n_uye=len(uyeler()), n_yon=len(D), red_eksik=len(eksik),
          bekle={"n_uye": 10})
    if eksik:
        print(f"  ★ yön kurulamayan: {eksik}")
    np.savez(f"{OUT}/yonler.npz", **D)
    json.dump(karne, open(f"{OUT}/yonler_karne.json", "w"), ensure_ascii=False, indent=1)
    print(f"→ {OUT}/yonler.npz ({len(D)} yön)")


def uret(a):
    import torch
    from prefix_korpus_pilot import CERCEVE, DURUS
    hedef = [u for u in uyeler() if u["ad"] == a.hedef][0]
    blok = BLOKLAR[hedef["blok"]]
    yolu = f"{OUT}/uretim_{hedef['ad']}.jsonl"
    man_y = f"{OUT}/manifest_{hedef['ad']}.json"
    if os.path.exists(man_y) and json.load(open(man_y)).get("tamam"):
        print(f"  ↷ ATLA {hedef['ad']} (manifest tamam)")
        return
    Y = np.load(f"{OUT}/yonler.npz")
    kaynaklar = [u for u in blok["uyeler"] if u["ad"] in Y.files]
    kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket=f"capraz_{hedef['ad']}")
    t0 = time.time()
    tok, model, snap = ST._yukle(hedef["hf"], a.dev)
    L, d_gizli = ST._L_of(model)
    kat_liste, kat_yol = ST._katman_listesi(model)
    if d_gizli != blok["d"]:
        raise SystemExit(f"★ KAPI: {hedef['ad']} d={d_gizli} ≠ blok {blok['d']}")
    kat = int(round(DERINLIK * L))
    S = [json.loads(l) for l in open(f"{HAVUZ}/uretim_{hedef['ad']}_ham_b3.jsonl",
                                     encoding="utf-8")]
    X = ST._akim(model, tok, a.dev, [s["metin"] for s in S], [kat], a.yigin)[kat]
    mu = X.mean(0)
    yaricap = float(np.median(np.linalg.norm(X - mu, axis=1)))
    C = X - mu
    rng = np.random.default_rng(SEED)
    V = np.stack([SU.birim(rng.standard_normal(len(C)) @ C) for _ in range(K_PLASEBO)])
    print(f"★ {hedef['ad']} · L={L} · ℓ{kat} · d={d_gizli} · yaricap {yaricap:.2f} · "
          f"kanca {kat_yol} · kaynak {len(kaynaklar)} · {time.time()-t0:.0f}s", flush=True)

    T = json.load(open(__DNH_DATA__ + "/onek_korpus/pilot_tezler.json",
                       encoding="utf-8"))
    isler = [dict(debate=t["debate"], tez=t["tez"], durus=d,
                  onek=CERCEVE["sakin"].format(T=t["tez"], P=DURUS[d]))
             for t in T for d in DURUS]
    ist = [j["onek"] for j in isler]
    E = ST.Enjektor(model, tok, a.dev, ist)
    kanca_h = kat_liste[kat].register_forward_hook(E.kanca)

    kal, hucreler = {}, [("l0", None, 0.0)]
    for u in kaynaklar:
        lam, iz = E.lam_icin(np.asarray(Y[u["ad"]]), yaricap)
        kal[f"kaynak_{u['ad']}"] = dict(lam=round(lam, 4), kl=iz[-1][1], iz=iz,
                                        kat=kat_of(u, hedef))
        hucreler.append((f"kaynak_{u['ad']}", np.asarray(Y[u["ad"]]), lam))
        print(f"  λ({u['ad']:6s} → {hedef['ad']:6s}, {kat_of(u, hedef):8s}) = {lam:.4f}"
              f" → KL {iz[-1][1]:.5f}", flush=True)
    for k in range(K_PLASEBO):
        lam, iz = E.lam_icin(V[k], yaricap)
        kal[f"plasebo{k}"] = dict(lam=round(lam, 4), kl=iz[-1][1], iz=iz, kat="plasebo")
        hucreler.append((f"plasebo{k}", V[k], lam))

    tok.padding_side = "left"
    n_sat = 0
    tmp = yolu + f".tmp.{os.getpid()}"
    fh = open(tmp, "w", encoding="utf-8")
    for ad, vek, lam in hucreler:
        E.ayarla(vek, lam, yaricap)
        torch.manual_seed(SEED + kat)
        for i in range(0, len(ist), a.yigin):
            enc = tok(ist[i:i + a.yigin], return_tensors="pt", padding=True,
                      truncation=True, max_length=1024)
            enc = {k: v.to(a.dev) for k, v in enc.items()}
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=UP.SICAKLIK,
                                   top_p=UP.TOP_P, max_new_tokens=YENI_JETON,
                                   pad_token_id=tok.pad_token_id)
            for j, s in enumerate(g[:, enc["input_ids"].shape[1]:]):
                r = dict(isler[i + j]); r.pop("onek")
                r.update(zemin="sakin", kol=ad, katman=int(kat), lam=round(float(lam), 4),
                         hedef=hedef["ad"], blok=hedef["blok"], satir_ix=n_sat,
                         metin=tok.decode(s, skip_special_tokens=True))
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_sat += 1
        E.vek = None
        print(f"  [{ad:14s}] λ={lam:.3f} · satir {n_sat} · {time.time()-t0:.0f}s",
              flush=True)
    kanca_h.remove()
    fh.close()
    os.replace(tmp, yolu)
    man = dict(damga=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="PREREG_CAPRAZ_ENJEKSIYON_2026-08-09.md", hedef=hedef["ad"],
               blok=hedef["blok"], hf=hedef["hf"], snapshot=os.path.basename(snap),
               L=L, d=d_gizli, katman=kat, derinlik=DERINLIK, kl_hedef=KL_HEDEF,
               yaricap=round(yaricap, 4), kalibrasyon=kal, n_kol=len(hucreler),
               n_satir=n_sat, bicim="ham", basamak="B3", seed=SEED,
               saniye=round(time.time() - t0, 1), tamam=True)
    json.dump(man, open(man_y, "w"), ensure_ascii=False, indent=1)
    payda(f"capraz_uret_{hedef['ad']}", n_satir=n_sat, n_kol=len(hucreler),
          n_istem=len(ist), bekle={"n_satir": len(ist) * len(hucreler)})
    print(f"  → {yolu} · {n_sat} satir · {time.time()-t0:.0f}s")


def coz(a):
    t0 = time.time()
    H1 = json.load(open(HARITA1))["hucreler"]
    gomu = f"{OUT}/e5_l21.npz"
    U = {u["ad"]: u for u in uyeler()}
    varolan = [ad for ad in U if os.path.exists(f"{OUT}/uretim_{ad}.jsonl")]
    if not os.path.exists(gomu):
        import torch
        import anchor_kodlama as CK
        from transformers import AutoTokenizer, AutoModel
        kilitle(a.dev, tam=True, etiket="capraz_e5")
        CK.DEV = a.dev
        C = CK.KOLLAR["c1a"]
        tk = AutoTokenizer.from_pretrained(C["model"])
        mdl = AutoModel.from_pretrained(C["model"], dtype=torch.float32).to(a.dev).eval()
        eski = (C["kes"], C["maxlen"])
        C["kes"], C["maxlen"] = 4000, 512
        veri = {}
        try:
            for ad in varolan:
                S = [json.loads(l) for l in open(f"{OUT}/uretim_{ad}.jsonl",
                                                 encoding="utf-8")]
                sc = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0,
                          uzunluklar=[])
                Eh = CK.kodla(tk, mdl, [s["metin"] for s in S], "c1a", bs=48, sayac=sc)
                veri[ad] = np.asarray(Eh[:, 21, :], dtype=np.float16)
                Sv = [s for s in (json.loads(l) for l in open(
                    f"{HAVUZ}/uretim_{ad}_ham_b3.jsonl", encoding="utf-8"))
                    if s["ton_hedef"] == "sert"]
                sc2 = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0,
                           uzunluklar=[])
                Ev = CK.kodla(tk, mdl, [s["metin"] for s in Sv], "c1a", bs=48, sayac=sc2)
                veri[ad + "__tavan"] = np.asarray(Ev[:, 21, :], dtype=np.float16)
                print(f"  · {ad} {len(S)} satir + tavan {len(Sv)} satir gömüldü",
                      flush=True)
        finally:
            C["kes"], C["maxlen"] = eski
        np.savez_compressed(gomu + ".tmp", **veri)
        os.replace(gomu + ".tmp.npz", gomu)
        del mdl
        torch.cuda.empty_cache()
    G = np.load(gomu)
    vekil = {ai: HV.vekil_kur(dislanan_aile=ai)[0]
             for ai in sorted(set(u["aile"] for u in uyeler()))}

    hucre, tavansiz = {}, []
    for ad in varolan:
        u = U[ad]
        t_izin = H1.get(u["tavan"], {})
        if not t_izin.get("tavanli", False):
            tavansiz.append(ad)
        S = [json.loads(l) for l in open(f"{OUT}/uretim_{ad}.jsonl", encoding="utf-8")]
        met = [s["metin"] for s in S]
        p = vekil[u["aile"]](met, np.asarray(G[ad], dtype=np.float32))
        F, _ = HV.yuzey_hizli(met)
        kol = np.array([s["kol"] for s in S], dtype=object)
        m0 = kol == "l0"
        e_ok = max(float(np.percentile(F["okunmazlik"][m0], 95)), MUTLAK_TABAN)
        e_tk = float(np.percentile(F["tekrar"][m0], 95))
        kirli = (F["okunmazlik"] > e_ok) | (F["tekrar"] > e_tk)
        tau = float(np.percentile(p[m0 & ~kirli], P_CAPA))

        def T(msk):
            s = msk & ~kirli
            return (float((p[s] >= tau).mean()) if s.sum() >= 20 else float("nan"),
                    int(s.sum()))

        def Hd(msk):
            s = msk & ~kirli
            return float(F["hedge"][s].mean()) if s.sum() else float("nan")
        t_capa, n_capa = T(m0)
        Sv = [s for s in (json.loads(l) for l in open(
            f"{HAVUZ}/uretim_{ad}_ham_b3.jsonl", encoding="utf-8"))
            if s["ton_hedef"] == "sert"]
        mv = [s["metin"] for s in Sv]
        pv = vekil[u["aile"]](mv, np.asarray(G[ad + "__tavan"], dtype=np.float32))
        Fv, _ = HV.yuzey_hizli(mv)
        tv = ~((Fv["okunmazlik"] > e_ok) | (Fv["tekrar"] > e_tk))
        t_tavan = float((pv[tv] >= tau).mean()) if tv.sum() >= 20 else float("nan")
        pl = np.array([T(kol == f"plasebo{k}")[0] - t_capa for k in range(K_PLASEBO)])
        band95 = float(np.nanpercentile(pl, 95))
        band_sd = float(np.nanstd(pl, ddof=1))
        for src in [x["ad"] for x in BLOKLAR[u["blok"]]["uyeler"]]:
            k = f"kaynak_{src}"
            if not (kol == k).any():
                continue
            tg, ng = T(kol == k)
            delta = tg - t_capa
            calisiyor = bool(np.isfinite(delta) and band_sd > 0 and delta > band95
                             and delta >= Z * band_sd)
            hucre[f"{src}→{ad}"] = dict(
                kaynak=src, hedef=ad, blok=u["blok"], kat=kat_of(U[src], u),
                tau_zemin=round(tau, 4), T_capa=round(t_capa, 4), n_capa=n_capa,
                T_gercek=None if not np.isfinite(tg) else round(tg, 4), n_gercek=ng,
                delta=None if not np.isfinite(delta) else round(delta, 4),
                band_p95=round(band95, 4), band_sd=round(band_sd, 4),
                MDE=round(Z * band_sd, 4),
                T_tavan=None if not np.isfinite(t_tavan) else round(t_tavan, 4),
                tavan_payi=None if not (np.isfinite(t_tavan) and np.isfinite(delta)
                                        and t_tavan - t_capa > 1e-9)
                else round(delta / (t_tavan - t_capa), 4),
                hedge_capa=round(Hd(m0), 4), hedge_gercek=round(Hd(kol == k), 4),
                kirli_oran=round(float(kirli[kol == k].mean()), 4),
                hedef_tavanli=bool(t_izin.get("tavanli", False)),
                calisiyor=calisiyor)
        hucre[f"PLASEBO@{ad}"] = dict(kaynak="plasebo", hedef=ad, blok=u["blok"],
                                      kat="plasebo", plasebo_delta=[
                                          None if not np.isfinite(x) else round(float(x), 4)
                                          for x in pl],
                                      band_p95=round(band95, 4), band_sd=round(band_sd, 4))
        print(f"  {ad:6s} τ={tau:.4f} capa T={t_capa:.4f} (n={n_capa}) · "
              f"TAVAN T={t_tavan:.4f} (bosluk {t_tavan-t_capa:+.4f}) · band p95="
              f"{band95:+.4f} sd={band_sd:.4f} · tavanli={t_izin.get('tavanli')}",
              flush=True)

    def oran(blok, kat):
        v = [h["calisiyor"] for h in hucre.values()
             if h.get("kat") == kat and h.get("blok") == blok
             and h.get("hedef_tavanli") and h.get("delta") is not None]
        return (float(np.mean(v)) if v else float("nan")), len(v)

    kat_oran = {b: {k: oran(b, k) for k in ("kosegen", "soy", "aile", "yabanci")}
                for b in BLOKLAR}
    B = "b4096"
    f = {k: kat_oran[B][k][0] for k in ("kosegen", "soy", "aile", "yabanci")}
    n = {k: kat_oran[B][k][1] for k in ("kosegen", "soy", "aile", "yabanci")}
    kapi_tamam = all(n[k] >= 2 for k in ("kosegen", "soy", "aile", "yabanci"))
    Hk = verdict_matris(f["kosegen"], f["soy"], f["aile"], f["yabanci"], kapi_tamam)
    payda("capraz_coz", n_hedef=len(varolan), n_hucre=sum(
        1 for h in hucre.values() if h.get("kat") != "plasebo"),
        red_tavansiz=len(tavansiz), bekle={"n_hedef": 6})
    from datetime import datetime, timezone
    out = dict(damga_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               prereg="PREREG_CAPRAZ_ENJEKSIYON_2026-08-09.md",
               birincil_blok=B, bar_kat=BAR_KAT, tavansiz=tavansiz,
               kat_oranlari={b: {k: dict(oran=None if not np.isfinite(v[0])
                                         else round(v[0], 4), n=v[1])
                                 for k, v in kk.items()} for b, kk in kat_oran.items()},
               kapi_tamam=bool(kapi_tamam), hucreler=hucre, adlar=list(ADLAR), verdict=Hk,
               saniye=round(time.time() - t0, 1))
    json.dump(out, open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for b in BLOKLAR:
        print(f"  [{b}] " + " · ".join(
            f"{k}: {'—' if not np.isfinite(kat_oran[b][k][0]) else f'{kat_oran[b][k][0]:.2f}'}"
            f" (n={kat_oran[b][k][1]})" for k in ("kosegen", "soy", "aile", "yabanci")))
    print(f"\n  ★★ K-TRANSFER-MATRISI HÜKMÜ: **{Hk}**\n→ {CIKTI}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", required=True,
                    choices=("prova", "esdegerlik", "yon", "uret", "coz"))
    ap.add_argument("--hedef")
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=None)
    ap.add_argument("--yigin", type=int, default=34)
    ap.add_argument("--mutlak-taban", dest="mutlak_taban", type=float, default=0.0,
                    help="D-22 mutlak okunmazlik tabani (0,0 = mühürlü davranis)")
    ap.add_argument("--cikti", default=None, help="cikti yolu (mühürlüyü EZMEZ)")
    a = ap.parse_args()
    global MUTLAK_TABAN, CIKTI
    if a.mutlak_taban > 0:
        if not a.cikti:
            raise SystemExit("★ --mutlak-taban verildiyse --cikti ZORUNLU (mühürlü "
                             "dosya ezilmez)")
        MUTLAK_TABAN, CIKTI = a.mutlak_taban, a.cikti
        print(f"★ D-22 ONARIM KIPI · mutlak taban {MUTLAK_TABAN} · cikti {CIKTI}")
    {"prova": lambda: _prova(), "esdegerlik": lambda: esdegerlik(a),
     "yon": lambda: yonler(a), "uret": lambda: uret(a), "coz": lambda: coz(a)}[a.asama]()


if __name__ == "__main__":
    main()
