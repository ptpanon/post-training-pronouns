#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import hashlib
import argparse

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
import prefix_generation_pilot as UP
import prefix_steering_generation as SU
import prefix_harsh_vekil as HV
import prefix_lineage_transferi as ST
import prefix_capraz_enjeksiyon as CE

OUT = __DNH_DATA__ + "/onek_korpus/gece_doz"
KOR = __DNH_DATA__ + "/onek_korpus/kor_paket"
YONLER = __DNH_DATA__ + "/onek_korpus/capraz/yonler.npz"
TARAF_YON = __DNH_DATA__ + "/onek_korpus/taraf_dumen/yazma_taraf_l12.npy"
CIKTI = f"{ROOT}/unreleased/GECE_DOZ_2026-08-09.json"

KL_TABAN = ST.KL_HEDEF
CARP_TAMIR = (0.25, 0.50)
CARP_EGRI = (0.25, 0.50, 1.00, 1.50, 2.00)
ISARET = (+1, -1)
MUTLAK_TABAN_OKUNMAZ = 0.15
P_CAPA, Z = 95, 1.645
SEED, YENI_JETON = 20260809, 160
BAR_PERGEL_AUC = 0.60
KOR_N = 6
K_PLASEBO = 3
SEKILLER = ("EKSIK", "DÜZENSIZ", "TERS-U", "ESIKLI", "DOYAN", "MONOTON")

TAMIR = {"m7b": ["m7b", "m7i", "yq9b"], "m7i": ["m7i"], "yq9b": ["m7b"]}

EGRILER = {
    "m7bton@m7i":   dict(hedef="m7i", kaynak=("npz", "m7b"), katman=None, kanal="vekil"),
    "m7itaraf@m7i": dict(hedef="m7i", kaynak=("npy", TARAF_YON), katman=12,
                         kanal="pergel_taraf"),
    "kos_mn8i":     dict(hedef="mn8i", kaynak=("npz", "mn8i"), katman=None, kanal="vekil"),
    "kos_yq9":      dict(hedef="yq9", kaynak=("npz", "yq9"), katman=None, kanal="vekil"),
    "kos_g312b":    dict(hedef="g312b", kaynak=("npz", "g312b"), katman=None, kanal="vekil"),
    "kos_yg12b":    dict(hedef="yg12b", kaynak=("npz", "yg12b"), katman=None, kanal="vekil"),
}
HEDEF_EGRI = {}
for _k, _v in EGRILER.items():
    HEDEF_EGRI.setdefault(_v["hedef"], []).append(_k)
ONCELIK = ("mn8i", "yq9")


def sekil(y, mde):
    v = np.asarray(y, float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return "EKSIK"
    v = v[ok]
    tol = 0.5 * mde
    fark = np.diff(v)
    artan = bool(np.all(fark >= -tol))
    if not artan:
        i = int(np.argmax(v))
        cikinti = float(v[i] - max(v[0], v[-1]))
        if 0 < i < len(v) - 1 and cikinti >= mde:
            return "TERS-U"
        return "DÜZENSIZ"
    if float(v[-1] - v[0]) < mde:
        return "DÜZENSIZ"
    duz = int(np.argmax(np.abs(fark) >= mde))
    if duz >= 2 and np.all(np.abs(fark[:duz]) < mde):
        return "ESIKLI"
    if float(fark[-1]) < 0.5 * float(fark[0]):
        return "DOYAN"
    return "MONOTON"


def _prova():
    m = 0.10
    V = [("iki nokta", [0.1, 0.5, np.nan, np.nan, np.nan], "EKSIK"),
         ("düz gürültü", [0.01, -0.01, 0.02, 0.0, 0.01], "DÜZENSIZ"),
         ("ters-U", [0.0, 0.2, 0.6, 0.25, 0.05], "TERS-U"),
         ("esikli", [0.0, 0.01, 0.02, 0.4, 0.8], "ESIKLI"),
         ("doyan", [0.0, 0.4, 0.6, 0.66, 0.68], "DOYAN"),
         ("monoton", [0.0, 0.2, 0.4, 0.6, 0.8], "MONOTON")]
    ok = 0
    for ad, y, bek in V:
        s = sekil(y, m)
        ok += s == bek
        print(f"  prova {ad:12s} → {s:9s} (bekle {bek})")
    assert ok == len(V), "sekil provasi DÜSTÜ"
    import ast
    ag = ast.parse(open(__file__, encoding="utf-8").read())
    kodda = {x.value.value for d in ast.walk(ag)
             if isinstance(d, ast.FunctionDef) and d.name == "sekil"
             for x in ast.walk(d)
             if isinstance(x, ast.Return) and isinstance(x.value, ast.Constant)}
    print(f"  [D44] kodda {len(kodda)} · protokolde {len(SEKILLER)} · "
          f"fark {sorted(kodda ^ set(SEKILLER)) or '—'}")
    if kodda != set(SEKILLER):
        raise SystemExit("★ D44 CAPRAZ SAYIM DÜSTÜ — kosu BASLAMAZ")
    assert sekil([0.0, 0.2, 0.4, 0.6, 0.8], 10.0) == "DÜZENSIZ", "MDE büyürken düzlesmeli"
    print("  ✓ prova gecti — alti ad, alti sinir girdisi, MDE duyarliligi, D44 6=6")


def _yon_yukle(spec):
    tur, ad = spec
    if tur == "npz":
        return np.asarray(np.load(YONLER)[ad], dtype=np.float64)
    v = np.asarray(np.load(ad), dtype=np.float64)
    return SU.birim(v)


def _istemler():
    from prefix_korpus_pilot import CERCEVE, DURUS
    T = json.load(open(__DNH_DATA__ + "/onek_korpus/pilot_tezler.json",
                       encoding="utf-8"))
    return [dict(debate=t["debate"], tez=t["tez"], durus=d,
                 onek=CERCEVE["sakin"].format(T=t["tez"], P=DURUS[d]))
            for t in T for d in DURUS]


def uret(a):
    import torch
    hedef = a.hedef
    yolu, man_y = f"{OUT}/uretim_{hedef}.jsonl", f"{OUT}/manifest_{hedef}.json"
    if os.path.exists(man_y) and json.load(open(man_y)).get("tamam"):
        print(f"  ↷ ATLA {hedef}")
        return
    os.makedirs(OUT, exist_ok=True)
    U = {u["ad"]: u for u in CE.uyeler()}[hedef]
    kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket=f"doz_{hedef}")
    t0 = time.time()
    tok, model, snap = ST._yukle(U["hf"], a.dev)
    L, d_gizli = ST._L_of(model)
    kat_liste, kat_yol = ST._katman_listesi(model)
    kat_rel = int(round(CE.DERINLIK * L))
    S = [json.loads(l) for l in open(
        f"{CE.HAVUZ}/uretim_{hedef}_ham_b3.jsonl", encoding="utf-8")]
    isler = _istemler()
    ist = [j["onek"] for j in isler]
    E = ST.Enjektor(model, tok, a.dev, ist)

    kollar = []
    for src in TAMIR.get(hedef, []):
        for c in CARP_TAMIR:
            kollar.append((f"tamir_{src}_kl{c:g}", _yon_yukle(("npz", src)), kat_rel,
                           c * KL_TABAN, +1))
    if hedef == ONCELIK[0]:
        kollar.append((f"oncelik_{ONCELIK[1]}_kl1", _yon_yukle(("npz", ONCELIK[1])),
                       kat_rel, KL_TABAN, +1))
    for eg in HEDEF_EGRI.get(hedef, []):
        m = EGRILER[eg]
        v = _yon_yukle(m["kaynak"])
        k = m["katman"] if m["katman"] is not None else kat_rel
        for c in CARP_EGRI:
            for s in ISARET:
                kollar.append((f"egri_{eg}_kl{c:g}_{'+' if s > 0 else '-'}", s * v, k,
                               c * KL_TABAN, s))
    if not kollar:
        raise SystemExit(f"{hedef}")

    kat_egri = sorted({(EGRILER[e]["katman"] or kat_rel) for e in HEDEF_EGRI.get(hedef, [])})
    kats = sorted({k for _a, _v, k, _c, _s in kollar} | set(kat_egri))
    Xk = ST._akim(model, tok, a.dev, [s["metin"] for s in S], kats, a.yigin)
    yaricap = {k: float(np.median(np.linalg.norm(Xk[k] - Xk[k].mean(0), axis=1)))
               for k in kats}
    _rng = np.random.default_rng(SEED)
    for k in kat_egri:
        Cc = Xk[k] - Xk[k].mean(0)
        V = np.stack([SU.birim(_rng.standard_normal(len(Cc)) @ Cc)
                      for _ in range(K_PLASEBO)])
        for q in range(K_PLASEBO):
            for c in CARP_EGRI:
                kollar.append((f"plasebo{q}_l{k}_kl{c:g}", V[q], k, c * KL_TABAN, +1))
    print(f"★ {hedef} · L={L} · d={d_gizli} · katmanlar {kats} · yaricap "
          f"{ {k: round(v, 2) for k, v in yaricap.items()} } · kol {len(kollar)} · "
          f"{time.time()-t0:.0f}s", flush=True)

    kanca = {}
    kal, n_sat = {}, 0
    tmp = yolu + f".tmp.{os.getpid()}"
    fh = open(tmp, "w", encoding="utf-8")
    tok.padding_side = "left"

    def kos(ad, vek, kat, lam):
        nonlocal n_sat
        if kat not in kanca:
            kanca[kat] = kat_liste[kat].register_forward_hook(E.kanca)
        E.ayarla(vek, lam, yaricap.get(kat, 1.0))
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
                r.update(zemin="sakin", kol=ad, katman=int(kat), hedef=hedef,
                         lam=round(float(lam), 4), satir_ix=n_sat,
                         metin=tok.decode(s, skip_special_tokens=True))
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_sat += 1
        E.vek = None

    kos("l0", None, kats[0], 0.0)
    print(f"  [l0] satir {n_sat} · {time.time()-t0:.0f}s", flush=True)
    for ad, vek, kat, klh, s in kollar:
        E.kl_hedef = klh
        lam, iz = E.lam_icin(vek, yaricap[kat])
        kal[ad] = dict(lam=round(lam, 4), kl_hedef=round(klh, 6), kl=iz[-1][1],
                       n_adim=len(iz), katman=int(kat), isaret=int(s))
        kos(ad, vek, kat, lam)
        print(f"  [{ad:30s}] ℓ{kat} λ={lam:.4f} → KL {iz[-1][1]:.5f} "
              f"(hedef {klh:.5f}) · satir {n_sat} · {time.time()-t0:.0f}s", flush=True)
    E.kl_hedef = KL_TABAN
    for h in kanca.values():
        h.remove()
    fh.close()
    os.replace(tmp, yolu)
    man = dict(damga=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               protokol="PROTOKOL_GECE_AB_2026-08-09.md", hedef=hedef, hf=U["hf"],
               snapshot=os.path.basename(snap), L=L, d=d_gizli, katmanlar=kats,
               yaricap={str(k): round(v, 4) for k, v in yaricap.items()},
               kat_rel=kat_rel, kat_egri=kat_egri, k_plasebo=K_PLASEBO,
               egri_katman={e: (EGRILER[e]["katman"] or kat_rel)
                            for e in HEDEF_EGRI.get(hedef, [])},
               kl_taban=KL_TABAN, kalibrasyon=kal, n_kol=len(kollar) + 1, n_satir=n_sat,
               bicim="ham", basamak="B3", seed=SEED,
               saniye=round(time.time() - t0, 1), tamam=True)
    json.dump(man, open(man_y, "w"), ensure_ascii=False, indent=1)
    payda(f"doz_uret_{hedef}", n_satir=n_sat, n_kol=len(kollar) + 1, n_istem=len(ist),
          bekle={"n_satir": len(ist) * (len(kollar) + 1)})
    print(f"  → {yolu} · {n_sat} satir · {time.time()-t0:.0f}s")


def coz(a):
    import prefix_wololo_ekran as WE
    t0 = time.time()
    hedefler = sorted({h for h in list(TAMIR) + list(HEDEF_EGRI)
                       if os.path.exists(f"{OUT}/uretim_{h}.jsonl")})
    gomu = f"{OUT}/gomu.npz"
    if not os.path.exists(gomu):
        import torch
        import anchor_kodlama as CK
        from transformers import AutoTokenizer, AutoModel
        from bs141_frontier_baselines import plain_enc
        kilitle(a.dev, tam=True, etiket="doz_gomu")
        veri = {}
        CK.DEV = a.dev
        C = CK.KOLLAR["c1a"]
        tk = AutoTokenizer.from_pretrained(C["model"])
        mdl = AutoModel.from_pretrained(C["model"], dtype=torch.float32).to(a.dev).eval()
        eski = (C["kes"], C["maxlen"])
        C["kes"], C["maxlen"] = 4000, 512
        try:
            for h in hedefler:
                S = [json.loads(l) for l in open(f"{OUT}/uretim_{h}.jsonl",
                                                 encoding="utf-8")]
                sc = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0,
                          uzunluklar=[])
                Eh = CK.kodla(tk, mdl, [s["metin"] for s in S], "c1a", bs=48, sayac=sc)
                veri[h] = np.asarray(Eh[:, 21, :], dtype=np.float16)
                print(f"  · {h} e5-ℓ21 {len(S)}", flush=True)
        finally:
            C["kes"], C["maxlen"] = eski
        del mdl
        torch.cuda.empty_cache()
        for h in hedefler:
            S = [json.loads(l) for l in open(f"{OUT}/uretim_{h}.jsonl", encoding="utf-8")]
            veri[h + "__e5"] = plain_enc("intfloat/e5-large-v2",
                                         [s["metin"] for s in S],
                                         prefix="query: ").astype(np.float16)
            print(f"  · {h} pergel-uzayi {len(S)}", flush=True)
        np.savez_compressed(gomu + ".tmp", **veri)
        os.replace(gomu + ".tmp.npz", gomu)
    G = np.load(gomu)

    P = {}
    for ad, (alan, arti, eksi) in (("ton", ("ton_hedef", "sert", "sakin")),
                                   ("taraf", ("durus", "arti", "eksi"))):
        mu0, d = WE.pergel_yon(alan=alan, arti=arti, eksi=eksi)
        z = WE._PY[f"{alan}|{arti}|{eksi}"]
        from prefix_yazar3_resolve import auc_np
        A = auc_np(z["izdusum"], z["etiket"])
        P[ad] = dict(mu0=mu0, d=d, auc=float(A), n=z["n"], gecti=bool(A >= BAR_PERGEL_AUC))
        print(f"  [PERGEL {ad}] kendi korpusunda AUC {A:.4f} (bar {BAR_PERGEL_AUC}) "
              f"· n={z['n']} ⇒ {'CETVEL' if A >= BAR_PERGEL_AUC else '★ CETVEL DEGIL'}")
    payda("pergel_kapi", n_eksen=2, red_zayif=sum(1 for v in P.values() if not v["gecti"]),
          bekle={"n_eksen": 2})

    vekil = {ai: HV.vekil_kur(dislanan_aile=ai)[0]
             for ai in sorted({u["aile"] for u in CE.uyeler()})}
    U = {u["ad"]: u for u in CE.uyeler()}
    kol_ozet, kor_kayit = {}, []
    rng = np.random.default_rng(SEED)
    for h in hedefler:
        S = [json.loads(l) for l in open(f"{OUT}/uretim_{h}.jsonl", encoding="utf-8")]
        met = [s["metin"] for s in S]
        kol = np.array([s["kol"] for s in S], dtype=object)
        p = vekil[U[h]["aile"]](met, np.asarray(G[h], dtype=np.float32))
        F, _ = HV.yuzey_hizli(met)
        Ee = np.asarray(G[h + "__e5"], dtype=np.float32)
        pr = {k: (Ee - P[k]["mu0"]) @ P[k]["d"] for k in P}
        m0 = kol == "l0"
        e_ok = max(float(np.percentile(F["okunmazlik"][m0], 95)), MUTLAK_TABAN_OKUNMAZ)
        e_tk = float(np.percentile(F["tekrar"][m0], 95))
        kirli = (F["okunmazlik"] > e_ok) | (F["tekrar"] > e_tk)
        tau = float(np.percentile(p[m0 & ~kirli], P_CAPA))

        def oku(msk):
            s = msk & ~kirli
            n = int(s.sum())
            if n < 20:
                return dict(n=n, vekil=None, hedge=None, pergel_ton=None,
                            pergel_taraf=None, kirli_oran=round(float(kirli[msk].mean()), 4))
            return dict(n=n, vekil=round(float((p[s] >= tau).mean()), 4),
                        hedge=round(float(F["hedge"][s].mean()), 5),
                        pergel_ton=round(float(pr["ton"][s].mean()), 5),
                        pergel_taraf=round(float(pr["taraf"][s].mean()), 5),
                        kirli_oran=round(float(kirli[msk].mean()), 4))
        capa = oku(m0)
        kol_ozet[h] = dict(tau=round(tau, 4), esik_okunmaz=round(e_ok, 4),
                           esik_tekrar=round(e_tk, 4), capa=capa, kollar={})
        for k in sorted(set(kol.tolist())):
            if k == "l0":
                continue
            r = oku(kol == k)
            for kan in ("vekil", "hedge", "pergel_ton", "pergel_taraf"):
                r["d_" + kan] = (None if (r[kan] is None or capa[kan] is None)
                                 else round(r[kan] - capa[kan], 5))
            kol_ozet[h]["kollar"][k] = r
            ix = np.where((kol == k) & ~kirli)[0]
            if len(ix):
                sec = rng.choice(ix, size=min(KOR_N, len(ix)), replace=False)
                for i in sec:
                    kor_kayit.append(dict(hedef=h, kol=k, satir_ix=int(S[i]["satir_ix"]),
                                          metin=S[i]["metin"]))
        print(f"  {h:6s} τ={tau:.4f} · capa vekil={capa['vekil']} · "
              f"esik ok={e_ok:.4f} tk={e_tk:.4f} · kol {len(kol_ozet[h]['kollar'])}",
              flush=True)

    tamir = {}
    for h, kaynaklar in TAMIR.items():
        if h not in kol_ozet:
            continue
        for src in kaynaklar:
            satir = {}
            for c in CARP_TAMIR:
                k = f"tamir_{src}_kl{c:g}"
                r = kol_ozet[h]["kollar"].get(k)
                satir[f"kl{c:g}"] = None if (r is None or r["vekil"] is None) else \
                    dict(delta=r["d_vekil"], n=r["n"], kirli=r["kirli_oran"])
            olculdu = [c for c in CARP_TAMIR if satir[f"kl{c:g}"] is not None]
            tamir[f"{src}→{h}"] = dict(
                satir=satir, olculen_kl=olculdu,
                ad=("ÖLCÜLDÜ" if olculdu else ""),
                en_dusuk_olculen=(min(olculdu) if olculdu else None))

    egriler = {}
    for eg, m in EGRILER.items():
        h = m["hedef"]
        if h not in kol_ozet:
            continue
        kan = m["kanal"]
        kk = json.load(open(f"{OUT}/manifest_{h}.json"))["egri_katman"][eg]
        bant = {}
        for c in CARP_EGRI:
            v = [kol_ozet[h]["kollar"].get(f"plasebo{q}_l{kk}_kl{c:g}", {}).get("d_" + kan)
                 for q in range(K_PLASEBO)]
            v = np.array([x for x in v if x is not None], float)
            bant[f"kl{c:g}"] = dict(
                n=int(len(v)),
                p95=None if len(v) < 2 else round(float(np.percentile(v, 95)), 5),
                sd=None if len(v) < 2 else round(float(np.std(v, ddof=1)), 5),
                MDE=None if len(v) < 2 else round(Z * float(np.std(v, ddof=1)), 5))
        mdeler = [b["MDE"] for b in band.values() if b["MDE"] is not None]
        mde = float(np.median(mdeler)) if mdeler else float("nan")
        d, sek = {}, {}
        for s in ISARET:
            y = []
            for c in CARP_EGRI:
                r = kol_ozet[h]["kollar"].get(
                    f"egri_{eg}_kl{c:g}_{'+' if s > 0 else '-'}")
                y.append(np.nan if (r is None or r["d_" + kan] is None)
                         else s * r["d_" + kan])
            im = "+" if s > 0 else "-"
            d[im] = [None if not np.isfinite(v) else round(float(v), 5) for v in y]
            sek[im] = sekil(y, mde) if np.isfinite(mde) else "EKSIK"
        egriler[eg] = dict(hedef=h, kanal=kan, katman=kk,
                           carpanlar=list(CARP_EGRI), yonlendirilmis=d, bant=band,
                           MDE_medyan=None if not np.isfinite(mde) else round(mde, 5),
                           sekil=sek)
        print(f"  [EGRI {eg:14s}] kanal {kan:12s} MDE {mde:.5f} · "
              f"+ {sek['+']:9s} {d['+']} · − {sek['-']:9s} {d['-']}", flush=True)
    from datetime import datetime, timezone
    out = dict(damga_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               protokol="PROTOKOL_GECE_AB_2026-08-09.md",
               sinif="",
               kl_taban=KL_TABAN, pergel_kapi={k: dict(auc=round(v["auc"], 4),
                                                       n=v["n"], gecti=v["gecti"])
                                               for k, v in P.items()},
               sekiller=list(SEKILLER), hedefler=kol_ozet, tamir=tamir, egriler=egriler,
               saniye=round(time.time() - t0, 1))
    json.dump(out, open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs(KOR, exist_ok=True)
    pak, anah = f"{KOR}/KOR_PAKET_GECE_2026-08-09.jsonl", f"{KOR}/ANAHTAR_GECE_2026-08-09.jsonl"
    with open(pak, "w", encoding="utf-8") as fp, open(anah, "w", encoding="utf-8") as fa:
        for i, r in enumerate(kor_kayit):
            fp.write(json.dumps({"paket_id": i, "metin": r["metin"]},
                                ensure_ascii=False) + "\n")
            fa.write(json.dumps({"paket_id": i, **{k: r[k] for k in
                                                   ("hedef", "kol", "satir_ix")}},
                                ensure_ascii=False) + "\n")

    def sha(y):
        return hashlib.sha256(open(y, "rb").read()).hexdigest()
    man = dict(damga_utc=out["damga_utc"], n_kalem=len(kor_kayit), seed=SEED,
               n_per_kol=KOR_N, secim="temiz satirlardan tekrarsiz rastgele",
               paket=dict(yol=pak, sha256=sha(pak)),
               anahtar=dict(yol=anah, sha256=sha(anah), commit_edilmez=True),
               kaynak=CIKTI, not_="paket YALNIZ metin tasir; kol/λ/model ANAHTARDA")
    json.dump(man, open(f"{ROOT}/unreleased/KOR_PAKET_MANIFEST_2026-08-09.json", "w"),
              ensure_ascii=False, indent=1)
    payda("kor_paket", n_kalem=len(kor_kayit), n_hedef=len(hedefler),
          bekle={"n_kalem": 1})
    print(f"\n  kör paket {len(kor_kayit)} satir · sha {sha(pak)[:16]}…")
    print(f"→ {CIKTI}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", required=True, choices=("prova", "uret", "coz"))
    ap.add_argument("--hedef")
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=None)
    ap.add_argument("--yigin", type=int, default=34)
    a = ap.parse_args()
    {"prova": lambda: _prova(), "uret": lambda: uret(a), "coz": lambda: coz(a)}[a.asama]()


if __name__ == "__main__":
    main()
