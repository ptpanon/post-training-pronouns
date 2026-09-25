#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, io, json, os, random, re, sys, time

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import numpy as np
import person_dial_minimal_pair_feasibility as KK
from reading_style_object import payda

KOK = os.environ.get("V61P_KOK", __DNH_DATA__ + "/v61_puanlama")
PREREG = f"{ROOT}/preregistration/prereg_v61_scoring_2026-09-11.md"
NOTICE = f"{ROOT}/preregistration/prerun_notice_v61_2026-09-11.md"
PREDICTION = f"{ROOT}/preregistration/prediction_v61_2026-09-11.md"
KUNYE = os.environ.get("V61P_KUNYE", f"{ROOT}/results/v61_scoring_generation_2026-09-12.json")
DUR = __DNH_DATA__ + "/DUR_V61"
HAVUZ = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
PUANLAYICI = (("armorm", "RLHFlow/ArmoRM-Llama3-8B-v0.1"),
              ("skywork", "Skywork/Skywork-Reward-Llama-3.1-8B"))
N_CIFT = 3000
N_GUC = 300
MDE_ORAN = 0.10
SEED = 20260911

TERS_DISI = {"your"}


def ters_kurallar():
    T = []
    for ad, rx, yeni in KK.RULE:
        if ad in TERS_DISI:
            continue
        T.append((ad, re.compile(r"\b" + re.escape(yeni) + r"\b", re.I), rx.pattern
                  .replace(r"\b", "").replace("(?i)", "")))
    return T


def uygula_ters(t):
    n = 0
    for ad, rx, yeni_s in ters_kurallar():
        def _d(m, y=yeni_s):
            return (y[0].upper() + y[1:]) if m.group(0)[:1].isupper() else y
        t, k = rx.subn(_d, t or "")
        n += k
    return t, n


def plasebo_takas(t, rnd):
    n, kar = KK.takaslar(t)
    if not t or kar <= 0:
        return t, 0
    kisi = re.compile(r"\byou\b|\byour\b|\byou're\b", re.I)
    yasak = [(m.start(), m.end()) for m in kisi.finditer(t)]
    def _cakisir(i, j):
        return any(not (j <= a or i >= b) for a, b in yasak)
    kelime = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)
              if not _cakisir(m.start(), m.end())]
    rnd.shuffle(kelime)
    sec, top = [], 0
    for i, j in kelime:
        if top >= kar:
            break
        sec.append((i, j)); top += (j - i)
    if len(sec) < 2:
        return t, 0
    sec.sort()
    kel = [t[i:j] for i, j in sec]
    karisik = kel[:]
    for _ in range(8):
        rnd.shuffle(karisik)
        if karisik != kel:
            break
    if karisik == kel:
        return t, 0
    out, son = [], 0
    for (i, j), yeni_k in zip(sec, karisik):
        out.append(t[son:i]); out.append(yeni_k); son = j
    out.append(t[son:])
    y = "".join(out)
    assert len(y) == len(t), "plasebo es-uzunluk DEGIL"
    return y, top


def _sha16(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def prova():
    s = []
    s.append(("kk2_uygula_prova", KK.uygula_prova()))
    a = "You should check your settings if you are unsure."
    ya, na = KK.uygula(a)
    s.append(("ileri_yon_takas", na > 0 and "you" not in ya.lower()))
    tb, nb = uygula_ters("One should check the settings if one is unsure.")
    s.append(("ters_yon_takas", nb > 0 and "you" in tb.lower()))
    tc, nc = uygula_ters("The settings are fine.")
    s.append(("ters_your_kapali", "your" not in tc.lower()))
    rnd = random.Random(SEED)
    p, kp = plasebo_takas(a, rnd)
    s.append(("plasebo_es_uzunluk", kp > 0 and len(p) == len(a)))
    s.append(("plasebo_kisi_bozmaz", p.lower().count("you") == a.lower().count("you")))
    gecen = sum(1 for _, g in s if g)
    payda("v61_puanlama_uretim_provasi", n_sinav=len(s), hal_gecen=gecen,
          red_dusen=len(s) - gecen)
    for ad, g in s:
        print(f"  [PROVA] {ad:22s} {'✓' if g else '✗'}")
    print(f"  ⇒ esik: {len(s)}/{len(s)} ⇒ EYLEM: düsen varsa PUANLAMA KOSMAZ", flush=True)
    return gecen == len(s)


def havuz_ciftleri(n_hedef, log=print):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    import filtre_verim as FV
    ds = load_dataset(HAVUZ, split="train")
    rnd = random.Random(SEED)
    ix = list(range(len(ds))); rnd.shuffle(ix)
    C, gorulen, red_uygunsuz, red_ters_yok, n_ters = [], set(), 0, 0, 0
    for k, i in enumerate(ix):
        if len(C) >= n_hedef:
            break
        x = ds[i]
        ist = x.get("prompt") or (x["chosen"][0]["content"] if x.get("chosen") else None)
        if not ist:
            continue
        h = hashlib.sha256(ist.encode()).hexdigest()[:16]
        if h in gorulen:
            continue
        for alan in ("chosen", "rejected"):
            t = FV._son_asistan_mesaj(x[alan]) if x.get(alan) else None
            if not t:
                continue
            ters = (n_ters < n_hedef // 2)
            if ters:
                kisisiz = t
                kisili, n = uygula_ters(t)
                if n == 0:
                    red_ters_yok += 1; continue
                n_ters += 1
            else:
                uy, _ = KK.uygun(t)
                if not uy:
                    red_uygunsuz += 1; continue
                kisili = t
                kisisiz, n = KK.uygula(t)
                if n == 0:
                    red_uygunsuz += 1; continue
            pla, kp = plasebo_takas(kisili, rnd)
            if kp == 0:
                continue
            if kisili == kisisiz:
                red_uygunsuz += 1
                if ters:
                    n_ters -= 1
                continue
            gorulen.add(h)
            C.append(dict(istem_i=len(gorulen) - 1, istem=ist, yon=("ters" if ters else "ileri"),
                          kisili=kisili, kisisiz=kisisiz, plasebo=pla,
                          n_takas=n, degisim_payi=float(KK.degisim_payi(kisili)),
                          kar_farki=len(kisisiz) - len(kisili)))
            break
    payda("v61_havuz_ciftleri", n_taranan=k + 1, n_cift=len(C),
          n_istem=len(gorulen), red_uygunsuz=red_uygunsuz, red_ters_yok=red_ters_yok,
          bekle={"n_cift": n_hedef})
    dp = np.array([c["degisim_payi"] for c in C]) if C else np.array([np.nan])
    log(f"{np.nanpercentile(dp,5):.4f}"
        f"{np.nanmedian(dp):.4f} {np.nanpercentile(dp,95):.4f}"
        f"")
    log(f"{sum(1 for c in C if c['yon']=='ileri')}"
        f"{sum(1 for c in C if c['yon']=='ters')}"
        f"")
    return C, dict(n_taranan=k + 1, n_istem=len(gorulen), red_uygunsuz=red_uygunsuz,
                   red_ters_yok=red_ters_yok,
                   degisim_payi=dict(p05=float(np.nanpercentile(dp, 5)),
                                     p50=float(np.nanmedian(dp)),
                                     p95=float(np.nanpercentile(dp, 95))))


class Puanlayici:

    def __init__(self, ad, depo, dev):
        import torch
        import transformers.models.llama.modeling_llama as _ml
        if not hasattr(_ml, "LLAMA_INPUTS_DOCSTRING"):
            _ml.LLAMA_INPUTS_DOCSTRING = ""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.ad, self.depo, self.dev = ad, depo, dev
        self.tok = AutoTokenizer.from_pretrained(depo)
        self.m = AutoModelForSequenceClassification.from_pretrained(
            depo, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map=dev, num_labels=1) if "Skywork" in depo else \
            AutoModelForSequenceClassification.from_pretrained(
                depo, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=dev)
        self.m.eval()
        self.torch = torch

    def puan(self, istem, cevap):
        msj = [{"role": "user", "content": istem}, {"role": "assistant", "content": cevap}]
        x = self.tok.apply_chat_template(msj, tokenize=True, return_tensors="pt",
                                         truncation=True, max_length=4096,
                                         return_dict=True)["input_ids"].to(self.dev)
        with self.torch.no_grad():
            o = self.m(x)
        s = getattr(o, "score", None)
        if s is None:
            s = o.logits[0]
        return float(np.asarray(s.float().cpu()).reshape(-1)[0])


def kos(ad, depo, C, dev, log=print):
    y = os.path.join(KOK, f"{ad}.jsonl")
    os.makedirs(KOK, exist_ok=True)
    var = 0
    if os.path.exists(y):
        var = sum(1 for l in io.open(y, encoding="utf-8") if l.strip())
        if var >= len(C):
            log(f"{ad} {var}")
            return dict(hal="ZATEN-VAR", n=var)
    P = Puanlayici(ad, depo, dev)
    t0 = time.time(); red = 0; yaz = 0
    with io.open(y, "a", encoding="utf-8") as f:
        for i, c in enumerate(C):
            if i < var:
                continue
            if os.path.exists(DUR):
                log(f"  ★★ DUR ⇒ {ad} {yaz} satirda durdu"); break
            try:
                pk = P.puan(c["istem"], c["kisili"])
                pz = P.puan(c["istem"], c["kisisiz"])
                pp = P.puan(c["istem"], c["plasebo"])
            except Exception as e:
                red += 1
                if red <= 3:
                    log(f"     red_puanlanamadi[{red}]: {str(e)[:90]}")
                continue
            f.write(json.dumps(dict(istem_i=c["istem_i"], yon=c["yon"],
                                    puan_kisili=pk, puan_kisisiz=pz, puan_plasebo=pp,
                                    degisim_payi=c["degisim_payi"],
                                    kar_farki=c["kar_farki"]), ensure_ascii=False) + "\n")
            yaz += 1
            if yaz == 1:
                dt = time.time() - t0
                log(f"  [W-71] {ad} ilk cift {dt:.1f} sn ⇒ ETA "
                    f"{dt*(len(C)-var)/60:.1f} dk (ölcülü) ⇒ esik 6 sa ⇒ EYLEM: "
                    f"asarsa kalan ciftler SONRAKI pencereye")
            if yaz % 250 == 0:
                f.flush()
                log(f"  [{ad}] {yaz}/{len(C)-var} · {(time.time()-t0)/60:.1f} dk")
    payda(f"v61_puanlama_{ad}", n_cift=len(C), hal_yazilan=yaz + var,
          red_puanlanamadi=red, hal_dk=(time.time() - t0) / 60)
    return dict(hal="KOSTU", n=yaz + var, red=red, dk=(time.time() - t0) / 60)


def guc_kapisi(ad, log=print):
    import arm_table as KT
    y = os.path.join(KOK, f"{ad}.jsonl")
    R = [json.loads(l) for l in io.open(y, encoding="utf-8") if l.strip()][:N_GUC]
    if len(R) < N_GUC:
        log(f"{ad} {len(R)} {N_GUC}")
        return None
    ist = np.array([int(r["istem_i"]) for r in R])
    d = np.array([r["puan_kisisiz"] - r["puan_kisili"] for r in R])
    sd_puan = float(np.nanstd([r["puan_kisili"] for r in R]))
    b = KT.kume_ort_boot(d, ist, nb=400, seed=SEED)
    mde = 1.645 * ((b[1] - b[0]) / (2 * 1.96)) if b else float("nan")
    gecti = bool(sd_puan > 0 and mde < MDE_ORAN * sd_puan)
    payda(f"v61_guc_kapisi_{ad}", n_cift=len(R), hal_mde=round(mde, 5),
          hal_bar=round(MDE_ORAN * sd_puan, 5), red_gecmedi=int(not gecti))
    log(f"  [KAPI · GÜC] {ad}: MDE={mde:.5f} · bar={MDE_ORAN}·sd={MDE_ORAN*sd_puan:.5f} "
        f"⇒ esik: MDE < bar ⇒ EYLEM: gecmezse TAM KOSU DEVAM ETMEZ ({'GECTI' if gecti else 'GECMEDI'})")
    return gecti


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--n", type=int, default=N_CIFT)
    ap.add_argument("--kuru", action="store_true")
    a = ap.parse_args()
    for y in (PREREG, NOTICE, PREDICTION):
        if not os.path.exists(y):
            raise SystemExit(f"{os.path.basename(y)}")
    if os.path.exists(DUR):
        print(f""); return 0
    if not prova():
        return 4
    if a.kuru:
        print("  ★ KURU: cift kurulumu sinanir, puanlama YOK", flush=True)
        C, k = havuz_ciftleri(min(a.n, 200))
        print(f"  ★ kuru cift {len(C)} · örnek: {C[0]['kisili'][:70]!r} → "
              f"{C[0]['kisisiz'][:70]!r}" if C else "  ★ cift YOK", flush=True)
        return 0
    C, kunye = havuz_ciftleri(a.n)
    if len(C) < a.n:
        print(f"{len(C)} {a.n}", flush=True)
    sonuc, ariza = {}, 0
    for ad, depo in PUANLAYICI:
        r0 = kos(ad, depo, C[:N_GUC], a.dev)
        g = guc_kapisi(ad)
        if g is False:
            sonuc[ad] = dict(**r0, guc="GECMEDI", tam_kosu=False)
            print(f"{ad}")
            continue
        r = kos(ad, depo, C, a.dev)
        sonuc[ad] = dict(**r, guc=("GECTI" if g else "ÖLCÜLEMEDI"), tam_kosu=True)
        ariza += int(r.get("red", 0) > 0.05 * len(C))
    json.dump(dict(alet="scripts/v61_scoring.py",
                   alet_sha16=_sha16(f"{ROOT}/scripts/v61_scoring.py"),
                   damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   prereg=os.path.basename(PREREG), prereg_sha256=_sha16(PREREG),
                   motor=dict(kk2_fizibilite=_sha16(f"{ROOT}/scripts/person_dial_minimal_pair_feasibility.py"),
                              kol_tablosu=_sha16(f"{ROOT}/scripts/arm_table.py")),
                   ters_disi_rule=sorted(TERS_DISI),
                   ters_serh=(""
                              ""),
                   havuz=HAVUZ, kok=KOK, n_hedef=a.n, cift_kunyesi=kunye,
                   sonuc=sonuc), io.open(KUNYE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    tam = all(v.get("tam_kosu") for v in sonuc.values()) and not ariza
    if tam:
        io.open(os.path.join(KOK, ".tamam"), "w", encoding="utf-8").write(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        print(f"  ★ .tamam yazildi ({KOK}/.tamam)", flush=True)
    else:
        print(f""
              f"", flush=True)
    payda("v61_puanlama_uretim", n_puanlayici=len(PUANLAYICI),
          hal_kosan=sum(1 for v in sonuc.values() if v.get("tam_kosu")),
          red_ariza=ariza)
    print(f"  ✓ {KUNYE} ⇒ esik: red_ariza>0 ⇒ CIKIS 3", flush=True)
    return 3 if ariza else 0


if __name__ == "__main__":
    sys.exit(main())
