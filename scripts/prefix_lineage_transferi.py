#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import argparse

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
import prefix_generation_pilot as UP
import prefix_steering_generation as SU
import prefix_okunmazlik as OK
import prefix_yazar_bataryasi as YB
import prefix_kararlilik as KR

OUT = __DNH_DATA__ + "/onek_korpus/soy_transferi"
HAVUZ = YB.OUT
CIKTI = f"{ROOT}/unreleased/SOY_TRANSFERI_2026-08-09.json"

KAYNAK = dict(ad="m7b", hf="mistralai/Mistral-7B-v0.3")
HEDEFLER = [dict(ad="m7i", hf="mistralai/Mistral-7B-Instruct-v0.3", rol="ev"),
            dict(ad="mn8i", hf="mistralai/Ministral-3-8B-Instruct-2512-BF16", rol="kardes"),
            dict(ad="yq9", hf="Qwen/Qwen3.5-9B", rol="yabanci")]
AILE = dict(m7i="mistral", mn8i="mistral", yq9="qwen")
DERINLIK = 0.50
KAT_SABIT = 12
KL_HEDEF = 0.0875
K_PLASEBO = 4
SEED, YENI_JETON = 20260809, 160
TAU = 0.4458
Z = 1.645
ADLAR = ("ÖLCÜLEMEZ", "HICBIRINDE", "EVRENSEL", "SOY-SINIRLI", "MODELE-ÖZGÜ",
         "YABANCIDA-VAR-KARDESTE-YOK")


def verdict_transfer(ev, kardes, yabanci, kapi_tamam):
    if (not kapi_tamam) or ev is None:
        return "ÖLCÜLEMEZ"
    if not ev:
        return "HICBIRINDE"
    if kardes is None or yabanci is None:
        return "ÖLCÜLEMEZ"
    if kardes and yabanci:
        return "EVRENSEL"
    if kardes and not yabanci:
        return "SOY-SINIRLI"
    if (not kardes) and (not yabanci):
        return "MODELE-ÖZGÜ"
    return "YABANCIDA-VAR-KARDESTE-YOK"


def _prova():
    V = [("kapi düstü", (True, True, True, False), "ÖLCÜLEMEZ"),
         ("ev ölcülemez", (None, True, True, True), "ÖLCÜLEMEZ"),
         ("ev ölü", (False, True, True, True), "HICBIRINDE"),
         ("kardes YORUMSUZ", (True, None, False, True), "ÖLCÜLEMEZ"),
         ("hepsi", (True, True, True, True), "EVRENSEL"),
         ("kardes ✓ yabanci ✗", (True, True, False, True), "SOY-SINIRLI"),
         ("yalniz ev", (True, False, False, True), "MODELE-ÖZGÜ"),
         ("yabanci ✓ kardes ✗", (True, False, True, True), "YABANCIDA-VAR-KARDESTE-YOK")]
    ok = 0
    for ad, arg, bek in V:
        h = verdict_transfer(*arg)
        ok += h == bek
        print(f"  prova {ad:20s} → {h:26s} (bekle {bek})")
    assert ok == len(V), "prova DÜSTÜ"
    assert set(h for _a, _r, h in V) == set(ADLAR), "D44: ad kümesi eslesmiyor"
    rng = np.random.default_rng(7)
    pl = rng.normal(0.0, 0.02, K_PLASEBO)
    for delta, bekle in ((0.0, False), (5 * 0.02, True)):
        p95 = float(np.quantile(pl, 0.95))
        gec = bool(delta > p95 and delta >= Z * float(np.std(pl, ddof=1)))
        print(f"  prova plasebo bandi (Δ={delta:.3f}) → gecti={gec} (bekle {bekle})")
        assert gec == bekle, "kill-test HAREKET ETMIYOR"
    print(f"  ✓ prova {ok}/{len(V)} · D44 {len(ADLAR)}={len(ADLAR)}")


def _akim(model, tok, dev, metinler, katmanlar, yigin):
    import torch
    tok.padding_side = "right"
    buf = {k: [] for k in katmanlar}
    for i in range(0, len(metinler), yigin):
        par = [(m or " ") for m in metinler[i:i + yigin]]
        enc = tok(par, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True)
        msk = enc["attention_mask"].unsqueeze(-1).float()
        for k in katmanlar:
            h = o.hidden_states[k].float()
            buf[k].append(((h * msk).sum(1) / msk.sum(1).clamp(min=1)).cpu().numpy())
        del o
        torch.cuda.empty_cache()
    return {k: np.concatenate(v).astype(np.float64) for k, v in buf.items()}


def _yukle(hf, dev):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    snap, _ = YB.snapshot(hf)
    if snap is None:
        raise SystemExit(f"{hf}")
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = None
    for ad, F in (("CausalLM", AutoModelForCausalLM), ("ImageTextToText", YB._itt()),
                  ("AutoModel", AutoModel)):
        if F is None:
            continue
        try:
            model = F.from_pretrained(snap, dtype=torch.bfloat16).to(dev).eval()
            print(f"  · yüklendi: {ad}", flush=True)
            break
        except (ValueError, KeyError, TypeError, OSError) as e:
            print(f"  · {ad} olmadi: {type(e).__name__}", flush=True)
    if model is None:
        raise SystemExit(f"★ KAPI: {hf} hicbir sinifla yüklenemedi")
    return tok, model, snap


def _katman_listesi(model):
    for yol in ("model.layers", "model.language_model.layers",
                "language_model.model.layers", "model.model.layers"):
        o = model
        try:
            for p in yol.split("."):
                o = getattr(o, p)
            if hasattr(o, "__len__") and len(o) > 1:
                return o, yol
        except AttributeError:
            continue
    raise SystemExit("")


def _L_of(model):
    c = model.config
    c = getattr(c, "text_config", c)
    return int(c.num_hidden_layers), int(c.hidden_size)


class Enjektor:

    def __init__(self, model, tok, dev, istemler_kl, kl_hedef=KL_HEDEF):
        self.model, self.tok, self.dev = model, tok, dev
        self.ist = list(istemler_kl)
        self.kl_hedef = kl_hedef
        self.vek = None

    def kanca(self, mod, girdi, cikti):
        if self.vek is None:
            return cikti
        h = cikti[0] if isinstance(cikti, tuple) else cikti
        h = h + self.vek.to(h.dtype)
        return ((h,) + tuple(cikti[1:])) if isinstance(cikti, tuple) else h

    def kl(self, vek, n=12):
        import torch
        out = []
        for s in self.ist[:n]:
            e = self.tok(s, return_tensors="pt").input_ids.to(self.dev)
            self.vek = None
            with torch.no_grad():
                o0 = self.model(e)
            self.vek = vek
            with torch.no_grad():
                o1 = self.model(e)
            self.vek = None
            p0 = torch.log_softmax(o0.logits[0, -1].float(), -1)
            p1 = torch.log_softmax(o1.logits[0, -1].float(), -1)
            out.append(float((p0.exp() * (p0 - p1)).sum()))
        return float(np.median(out))

    def lam_icin(self, vek, yaricap, tur=9, bar=0.10):
        import torch
        lo, hi, iz, orta = 0.02, 64.0, [], None
        for _ in range(tur):
            orta = float(np.sqrt(lo * hi))
            d = self.kl(torch.tensor(orta * yaricap * vek, device=self.dev,
                                     dtype=torch.float32))
            iz.append((round(orta, 4), round(d, 6)))
            if abs(d - self.kl_hedef) / self.kl_hedef <= bar:
                break
            if d < self.kl_hedef:
                lo = orta
            else:
                hi = orta
        return orta, iz

    def ayarla(self, vek, lam, yaricap):
        import torch
        self.vek = (None if vek is None else torch.tensor(
            lam * yaricap * vek, device=self.dev, dtype=torch.float32))


def kaynak_yon(a):
    import torch
    yol = f"{OUT}/d_taban.npz"
    if os.path.exists(yol):
        print(f"  ↷ ATLA {yol}")
        return
    os.makedirs(OUT, exist_ok=True)
    S = [json.loads(l) for l in open(f"{HAVUZ}/uretim_m7b_ham.jsonl", encoding="utf-8")]
    tok, model, snap = _yukle(KAYNAK["hf"], a.dev)
    L, _d = _L_of(model)
    kat = int(round(DERINLIK * L))
    X = _akim(model, tok, a.dev, [s["metin"] for s in S], [kat], a.yigin)[kat]
    y = np.array([1 if s["ton_hedef"] == "sert" else 0 for s in S])
    kume = np.array([s["debate"] for s in S], dtype=object)
    yon = SU.birim(X[y == 1].mean(0) - X[y == 0].mean(0))
    mu = X.mean(0)
    yaricap = float(np.median(np.linalg.norm(X - mu, axis=1)))
    kar = KR.kume_tutulan_bolme(X, y, kume)
    payda("kaynak_yon", n_satir=len(S), n_sert=int(y.sum()), n_kume=len(set(kume.tolist())),
          n_boyut=int(X.shape[1]), bekle={"n_satir": 816, "n_boyut": 4096})
    print(f"  kaynak {KAYNAK['ad']} · L={L} · göreli {DERINLIK} ⇒ ℓ{kat} · d={X.shape[1]}")
    print(f"  ‖d‖/yaricap = {np.linalg.norm(X[y==1].mean(0)-X[y==0].mean(0))/yaricap:.4f} · "
          f"küme-tutulan bölme medyan {kar['medyan']:.4f} (bar {KR.BAR}, gecti={kar['gecti']})")
    if not kar["gecti"]:
        raise SystemExit("★ KAPI: kaynak yön KARARSIZ (prereg §3) ⇒ kosu DURUR")
    np.savez(yol, yon=yon, katman=kat, L=L, yaricap=yaricap,
             kararlilik=kar["medyan"], n=len(S))
    del model
    torch.cuda.empty_cache()
    print(f"→ {yol}")


def uret(a):
    import torch
    from prefix_korpus_pilot import CERCEVE, DURUS
    hedef = [h for h in HEDEFLER if h["ad"] == a.hedef][0]
    yolu = f"{OUT}/uretim_{hedef['ad']}.jsonl"
    man_y = f"{OUT}/manifest_{hedef['ad']}.json"
    if os.path.exists(man_y) and json.load(open(man_y)).get("tamam"):
        print(f"  ↷ ATLA {hedef['ad']} (manifest tamam)")
        return
    Z0 = np.load(f"{OUT}/d_taban.npz")
    d_taban = Z0["yon"]
    kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket=f"soy_{hedef['ad']}")
    t0 = time.time()
    tok, model, snap = _yukle(hedef["hf"], a.dev)
    L, d_gizli = _L_of(model)
    kat_liste, kat_yol = _katman_listesi(model)
    print(f"  · kanca yolu: {kat_yol} ({len(kat_liste)} katman)", flush=True)
    if d_gizli != len(d_taban):
        raise SystemExit(f"★ KAPI: {hedef['ad']} d={d_gizli} ≠ kaynak {len(d_taban)}")
    katmanlar = sorted({int(round(DERINLIK * L)), KAT_SABIT})
    S = [json.loads(l) for l in open(f"{HAVUZ}/uretim_{hedef['ad']}_ham.jsonl",
                                     encoding="utf-8")]
    Xk = _akim(model, tok, a.dev, [s["metin"] for s in S], katmanlar, a.yigin)
    print(f"★ {hedef['ad']} ({hedef['rol']}) · L={L} · katmanlar {katmanlar} · "
          f"{time.time()-t0:.0f}s", flush=True)

    T = json.load(open(__DNH_DATA__ + "/onek_korpus/pilot_tezler.json",
                       encoding="utf-8"))
    isler = [dict(debate=t["debate"], tez=t["tez"], durus=d,
                  onek=CERCEVE["sakin"].format(T=t["tez"], P=DURUS[d]))
             for t in T for d in DURUS]
    ist = [j["onek"] for j in isler]
    E = Enjektor(model, tok, a.dev, ist)
    kanca, lam_icin = E.kanca, E.lam_icin

    tok.padding_side = "left"
    kal, kayit, n_sat = {}, [], 0
    tmp = yolu + f".tmp.{os.getpid()}"
    fh = open(tmp, "w", encoding="utf-8")
    rng = np.random.default_rng(SEED)
    for kat in katmanlar:
        X = Xk[kat]
        mu = X.mean(0)
        yaricap = float(np.median(np.linalg.norm(X - mu, axis=1)))
        C = X - mu
        V = np.stack([SU.birim(rng.standard_normal(len(C)) @ C) for _ in range(K_PLASEBO)])
        kanca_h = kat_liste[kat].register_forward_hook(kanca)
        lam_g, iz_g = lam_icin(d_taban, yaricap)
        kal[f"l{kat}_gercek"] = dict(lam=round(lam_g, 4), kl=iz_g[-1][1], iz=iz_g)
        print(f"  ℓ{kat} λ(gercek)={lam_g:.4f} → KL {iz_g[-1][1]:.5f} "
              f"(sapma {abs(iz_g[-1][1]-KL_HEDEF)/KL_HEDEF:.1%}) · yaricap {yaricap:.2f}",
              flush=True)
        hucreler = [("l0", None, 0.0), ("gercek", d_taban, lam_g)]
        for k in range(K_PLASEBO):
            lk, izk = lam_icin(V[k], yaricap)
            kal[f"l{kat}_plasebo{k}"] = dict(lam=round(lk, 4), kl=izk[-1][1], iz=izk)
            hucreler.append((f"plasebo{k}", V[k], lk))
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
                    r.update(zemin="sakin", kol=ad, katman=int(kat),
                             lam=round(float(lam), 4), yazar=hedef["ad"],
                             rol=hedef["rol"], satir_ix=n_sat,
                             metin=tok.decode(s, skip_special_tokens=True))
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_sat += 1
            E.vek = None
            print(f"  [ℓ{kat} {ad:9s}] λ={lam:.3f} · satir {n_sat} · "
                  f"{time.time()-t0:.0f}s", flush=True)
        kanca_h.remove()
    fh.close()
    os.replace(tmp, yolu)
    man = dict(damga=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg="PREREG_SOY_TRANSFERI_2026-08-09.md @ 15b238e", hedef=hedef["ad"],
               rol=hedef["rol"], hf=hedef["hf"], snapshot=os.path.basename(snap), L=L,
               d=d_gizli, katmanlar=katmanlar, derinlik=DERINLIK, kl_hedef=KL_HEDEF,
               kalibrasyon=kal, n_kol=len(katmanlar) * (2 + K_PLASEBO), n_satir=n_sat,
               bicim="ham", seed=SEED, saniye=round(time.time() - t0, 1), tamam=True)
    json.dump(man, open(man_y, "w"), ensure_ascii=False, indent=1)
    payda(f"soy_uret_{hedef['ad']}", n_satir=n_sat, n_kol=len(katmanlar) * (2 + K_PLASEBO),
          n_istem=len(ist), bekle={"n_satir": len(ist) * len(katmanlar) * (2 + K_PLASEBO)})
    print(f"  → {yolu} · {n_sat} satir · {time.time()-t0:.0f}s")


def coz(a):
    import prefix_harsh_vekil as HV
    Z0 = np.load(f"{OUT}/d_taban.npz")
    sonuc, kapi_tamam = {}, True
    gomu = f"{OUT}/e5_l21.npz"
    if not os.path.exists(gomu):
        from gpu_lock import kilitle as _kil
        _kil(a.dev, tam=True, etiket="soy_e5")
        import torch
        import anchor_kodlama as CK
        from transformers import AutoTokenizer, AutoModel
        CK.DEV = a.dev
        K = CK.KOLLAR["c1a"]
        tk = AutoTokenizer.from_pretrained(K["model"])
        mdl = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(a.dev).eval()
        eski = (K["kes"], K["maxlen"])
        K["kes"], K["maxlen"] = 4000, 512
        veri = {}
        try:
            for h in HEDEFLER:
                S = [json.loads(l) for l in open(f"{OUT}/uretim_{h['ad']}.jsonl",
                                                 encoding="utf-8")]
                sc = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0,
                          uzunluklar=[])
                E = CK.kodla(tk, mdl, [s["metin"] for s in S], "c1a", bs=48, sayac=sc)
                veri[h["ad"]] = np.asarray(E[:, 21, :], dtype=np.float16)
                print(f"  · {h['ad']} {len(S)} satir gömüldü", flush=True)
        finally:
            K["kes"], K["maxlen"] = eski
        np.savez_compressed(gomu + ".tmp", **veri)
        os.replace(gomu + ".tmp.npz", gomu)
        del mdl
        torch.cuda.empty_cache()
    G = np.load(gomu)
    vekil = {}
    for aile in sorted(set(AILE.values())):
        vekil[aile] = HV.vekil_kur(dislanan_aile=aile)[0]

    for h in HEDEFLER:
        S = [json.loads(l) for l in open(f"{OUT}/uretim_{h['ad']}.jsonl", encoding="utf-8")]
        E = np.asarray(G[h["ad"]], dtype=np.float32)
        p = vekil[AILE[h["ad"]]]([s["metin"] for s in S], E)
        F, _ = HV.yuzey_hizli([s["metin"] for s in S])
        kol = np.array([s["kol"] for s in S], dtype=object)
        kat = np.array([s["katman"] for s in S], int)
        man = json.load(open(f"{OUT}/manifest_{h['ad']}.json"))
        birincil = int(round(DERINLIK * man["L"]))
        R = {}
        for k in sorted(set(kat.tolist())):
            m0 = (kat == k) & (kol == "l0")
            e_ok = float(np.percentile(F["okunmazlik"][m0], 95))
            e_tk = float(np.percentile(F["tekrar"][m0], 95))
            kirli = (F["okunmazlik"] > e_ok) | (F["tekrar"] > e_tk)
            def T(msk):
                s = msk & ~kirli
                return (float((p[s] >= TAU).mean()) if s.sum() >= 20 else float("nan"),
                        int(s.sum()))

            def Pm(msk):
                s = msk & ~kirli
                return float(p[s].mean()) if s.sum() else float("nan")
            t0_, n0_ = T(m0)
            tg_, ng_ = T((kat == k) & (kol == "gercek"))
            pl = [T((kat == k) & (kol == f"plasebo{j}"))[0] for j in range(K_PLASEBO)]
            pl_d = np.array([v - t0_ for v in pl], dtype=float)
            band_p95 = float(np.nanpercentile(pl_d, 95))
            band_sd = float(np.nanstd(pl_d, ddof=1))
            delta = tg_ - t0_
            calisiyor = bool(np.isfinite(delta) and np.isfinite(band_p95) and band_sd > 0
                             and delta > band_p95 and delta >= Z * band_sd)
            kirli_oran = float(kirli[(kat == k)].mean())
            R[f"l{k}"] = dict(katman=int(k), birincil=bool(k == birincil),
                              T_capa=round(t0_, 4), n_capa=n0_,
                              T_gercek=None if not np.isfinite(tg_) else round(tg_, 4),
                              n_gercek=ng_, delta=None if not np.isfinite(delta)
                              else round(delta, 4),
                              plasebo_delta=[None if not np.isfinite(v) else round(v, 4)
                                             for v in pl_d],
                              band_p95=round(band_p95, 4), band_sd=round(band_sd, 4),
                              MDE=round(Z * band_sd, 4), kirli_oran=round(kirli_oran, 4),
                              boy_kelime=round(float(np.mean(F["log_n"][(kat == k)])), 3),
                              ikincil_p_ort=dict(
                                  l0=round(Pm(m0), 4),
                                  gercek=round(Pm((kat == k) & (kol == "gercek")), 4),
                                  plasebo=[round(Pm((kat == k) & (kol == f"plasebo{j}")), 4)
                                           for j in range(K_PLASEBO)]),
                              calisiyor=calisiyor)
            print(f"  {h['ad']:5s} ℓ{k:<3d} {'(birincil)' if k==birincil else '          '} "
                  f"T0={t0_:.4f} Tg={tg_:.4f} Δ={delta:+.4f} "
                  f"band_p95={band_p95:+.4f} sd={band_sd:.4f} MDE={Z*band_sd:.4f} "
                  f"kirli={kirli_oran:.3f} ⇒ {'CALISIYOR' if calisiyor else '—'}")
        sonuc[h["ad"]] = dict(rol=h["rol"], aile=AILE[h["ad"]], L=man["L"],
                              katmanlar=R, kalibrasyon=man["kalibrasyon"])
        if R[f"l{birincil}"]["kirli_oran"] >= 0.25:
            kapi_tamam = False

    def bay(ad):
        m = sonuc[ad]
        b = int(round(DERINLIK * m["L"]))
        v = m["katmanlar"][f"l{b}"]
        return None if v["delta"] is None else v["calisiyor"]

    h = verdict_transfer(bay("m7i"), bay("mn8i"), bay("yq9"), kapi_tamam)
    from datetime import datetime, timezone
    out = dict(damga_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               prereg="PREREG_SOY_TRANSFERI_2026-08-09.md @ 15b238e",
               kaynak=dict(ad=KAYNAK["ad"], hf=KAYNAK["hf"], katman=int(Z0["katman"]),
                           L=int(Z0["L"]), kararlilik=round(float(Z0["kararlilik"]), 4),
                           yaricap=round(float(Z0["yaricap"]), 2)),
               bicim="ham (sohbet sargisi YOK — K-TAVAN/D-20 geregi)",
               tau=TAU, kl_hedef=KL_HEDEF, k_plasebo=K_PLASEBO,
               hedefler=sonuc, kapi_tamam=kapi_tamam, verdict=h)
    json.dump(out, open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  ★★ K-TRANSFER HÜKMÜ: **{h}**\n→ {CIKTI}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", required=True, choices=("yon", "uret", "coz"))
    ap.add_argument("--hedef")
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=None)
    ap.add_argument("--yigin", type=int, default=34)
    a = ap.parse_args()
    _prova()
    if a.asama == "yon":
        kaynak_yon(a)
    elif a.asama == "uret":
        uret(a)
    else:
        coz(a)
