#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import glob
import time
import argparse
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
import prefix_generation_pilot as UP
import prefix_okunmazlik as OK

OUT = __DNH_DATA__ + "/onek_korpus/yazar_batarya"
KOKLER = (f"{ROOT}/.hf/hub",
          "<storage>/huggingface/hub",
          f"{os.path.expanduser('~')}/.cache/huggingface/hub",
          __DNH_DATA__ + "/.hf_local/hub")
SEED = 20260808
N_CEKIM = 12
YENI_JETON = 160
DERINLIK = (0.25, 0.375, 0.5, 0.625, 0.75)
BAR_OKUNMAZ = 0.15
BAR_ORAN = 0.25

YAZARLAR = {
    "m7i":   dict(aile="mistral", hf="mistralai/Mistral-7B-Instruct-v0.3", boy=7.2),
    "mn3i":  dict(aile="mistral", hf="mistralai/Ministral-3-3B-Instruct-2512-BF16", boy=3.3),
    "mn8i":  dict(aile="mistral", hf="mistralai/Ministral-3-8B-Instruct-2512-BF16", boy=8.0),
    "g34i":  dict(aile="gemma",   hf="google/gemma-3-4b-it", boy=4.3),
    "g312i": dict(aile="gemma",   hf="google/gemma-3-12b-it", boy=12.2),
    "yg12":  dict(aile="gemma",   hf="google/gemma-4-12B-it", boy=12.0),
    "q25i":  dict(aile="qwen",    hf="Qwen/Qwen2.5-7B-Instruct", boy=7.6),
    "yq4":   dict(aile="qwen",    hf="Qwen/Qwen3.5-4B", boy=4.0),
    "yq9":   dict(aile="qwen",    hf="Qwen/Qwen3.5-9B", boy=9.0),
    "m7abl": dict(aile="mistral", hf="richardyoung/Mistral-7B-Instruct-v0.3-abliterated",
                  boy=7.2, es="m7i"),
}

YAZARLAR_TABAN = {
    "m7b":   dict(aile="mistral", hf="mistralai/Mistral-7B-v0.3", boy=7.2, es="m7i"),
    "mn3b":  dict(aile="mistral", hf="mistralai/Ministral-3-3B-Base-2512", boy=3.3, es="mn3i"),
    "mn8b":  dict(aile="mistral", hf="mistralai/Ministral-3-8B-Base-2512", boy=8.0, es="mn8i"),
    "g34b":  dict(aile="gemma",   hf="google/gemma-3-4b-pt", boy=4.3, es="g34i"),
    "g312b": dict(aile="gemma",   hf="google/gemma-3-12b-pt", boy=12.2, es="g312i"),
    "yg12b": dict(aile="gemma",   hf="google/gemma-4-12b", boy=12.0, es="yg12"),
    "q25b":  dict(aile="qwen",    hf="Qwen/Qwen2.5-7B", boy=7.6, es="q25i"),
    "yq9b":  dict(aile="qwen",    hf="Qwen/Qwen3.5-9B-Base", boy=9.0, es="yq9"),
}


def _itt():
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except Exception:
        return None


def snapshot(hf):
    ad = "models--" + hf.replace("/", "--")
    adaylar = []
    for kok in KOKLER:
        for s in sorted(glob.glob(f"{kok}/{ad}/snapshots/*")):
            cfg = os.path.exists(f"{s}/config.json")
            agr = bool(glob.glob(f"{s}/*.safetensors") or glob.glob(f"{s}/*.bin"))
            tk = bool(glob.glob(f"{s}/tokenizer.json") or glob.glob(f"{s}/tokenizer.model")
                      or glob.glob(f"{s}/vocab.json"))
            adaylar.append((s, cfg, agr, tk, len(os.listdir(s))))
    tam = sorted([(n, s) for s, c, a, t, n in adaylar if c and a and t])
    return (tam[-1][1] if tam else None), adaylar


def envanter():
    R, n_tam, n_bolunmus = {}, 0, 0
    for kol, m in YAZARLAR.items():
        s, adaylar = snapshot(m["hf"])
        L = d = None
        eksik = [os.path.basename(x[0]) for x in adaylar if not (x[1] and x[2] and x[3])]
        if s:
            c = json.load(open(f"{s}/config.json"))
            c = c.get("text_config", c)
            L, d = c.get("num_hidden_layers"), c.get("hidden_size")
            n_tam += 1
        elif adaylar:
            n_bolunmus += 1
        R[kol] = dict(aile=m["aile"], hf=m["hf"], boy_B=m["boy"], snapshot=s,
                      n_snapshot=len(adaylar), n_eksik_snapshot=len(eksik), tam=bool(s),
                      L=L, d=d,
                      katmanlar=[int(round(r * L)) for r in DERINLIK] if L else None,
                      vram_bf16_gb=round(m["boy"] * 2.0, 1))
        print(f"  {kol:6s} {m['aile']:8s} {m['boy']:5.1f}B  L={str(L):>3s} d={str(d):>5s}  "
              f"{'TAM' if s else '★ EKSIK'}  snap={len(adaylar)}")
    payda("yazar_envanter", n_yazar=len(YAZARLAR), n_aile=len(set(m["aile"] for m in
          YAZARLAR.values())), n_tam=n_tam, red_eksik=len(YAZARLAR) - n_tam,
          bekle={"n_yazar": 9, "n_aile": 3})
    if n_tam < len(YAZARLAR):
        print(f"{len(YAZARLAR)-n_tam}"
              f"{n_bolunmus}")
    return R


def istemler(basamak="B4"):
    from prefix_korpus_pilot import CERCEVE, DURUS
    import prefix_ladder as M
    if basamak not in M.SIRA:
        raise SystemExit(f"★ KAPI: bilinmeyen basamak {basamak!r} — {M.SIRA}")
    TON = {"sert": M.BASAMAKLAR[basamak], "sakin": CERCEVE["sakin"]}
    T = json.load(open(__DNH_DATA__ + "/onek_korpus/pilot_tezler.json",
                       encoding="utf-8"))
    isler = [(t, d, h) for t in T for d in DURUS for h in TON]
    payda("yazar_istem", n_tez=len(T), n_is=len(isler), n_ton=len(TON), n_durus=len(DURUS),
          bekle={"n_tez": 17, "n_is": 68, "n_ton": 2})
    return [dict(debate=t["debate"], tez=t["tez"], durus=d, ton_hedef=h,
                 onek=TON[h].format(T=t["tez"], P=DURUS[d])) for t, d, h in isler]


def kol_taban(a):
    return bool(getattr(a, "taban", False))


def uret(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    kip = getattr(a, "kip", "sohbet")
    tablo = YAZARLAR_TABAN if kol_taban(a) else YAZARLAR
    kol = a.yazar
    m = tablo[kol]
    snap, _ = snapshot(m["hf"])
    if snap is None:
        raise SystemExit(f"{kol}")
    os.makedirs(OUT, exist_ok=True)
    basamak = getattr(a, "basamak", "B4")
    kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket=f"yazar_{kol}")
    isler = istemler(basamak)
    if a.prova:
        isler = isler[:4]
        print("")
    tok = AutoTokenizer.from_pretrained(snap)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, sinif = None, None
    for ad, F in (("CausalLM", AutoModelForCausalLM),
                  ("ImageTextToText", _itt()), ("AutoModel", AutoModel)):
        if F is None:
            continue
        try:
            model = F.from_pretrained(snap, dtype=torch.bfloat16).to(a.dev)
            sinif = ad
            break
        except (ValueError, KeyError, TypeError, OSError) as e:
            print(f"  · {ad} olmadi: {type(e).__name__}", flush=True)
    if model is None:
        raise SystemExit(f"★ KAPI: {kol} hicbir sinifla yüklenemedi")
    model.eval()
    print(f"  · yüklendi: {sinif}", flush=True)
    cfg = model.config
    L = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    kats = [int(round(r * L)) for r in DERINLIK]
    print(f"★ {kol} · {os.path.basename(snap)} · L={L} · katmanlar {kats} · "
          f"istem {len(isler)} × cekim {N_CEKIM}", flush=True)

    def giydir(o):
        if kip == "ham":
            return o
        if kip == "sargi":
            import prefix_ladder as _M
            o = _M.SOHBET_TALIMATI + o
        try:
            return tok.apply_chat_template([{"role": "user", "content": o}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            return o
    ist = [giydir(j["onek"]) for j in isler]
    template = "sohbet" if ist[0] != isler[0]["onek"] else "ham"
    if kip == "ham" and template != "ham":
        raise SystemExit("★ KAPI: `--kip ham` istendi ama sargi uygulanmis — bicim kolu bozuk")
    if kip == "sargi":
        import prefix_ladder as _M
        if _M.SOHBET_TALIMATI not in ist[0]:
            raise SystemExit("★ KAPI: `--kip sargi` istendi ama TALIMAT istemde YOK")
        if template != "sohbet":
            raise SystemExit("★ KAPI: `--kip sargi` sohbet sablonu ister — model uygulamadi")

    ek = {"sohbet": "", "ham": "_ham", "sargi": "_sargi"}[kip] + \
         ("" if basamak == "B4" else f"_{basamak.lower()}")
    yol = f"{OUT}/uretim_{kol}{ek}.jsonl" + ("_prova" if a.prova else "")
    tmp = yol + f".tmp.{os.getpid()}"
    fh = open(tmp, "w", encoding="utf-8")
    torch.manual_seed(SEED)
    n_sat, t0 = 0, time.time()
    n_cekim = 1 if a.prova else N_CEKIM
    for c in range(n_cekim):
        for i in range(0, len(ist), a.yigin):
            par = ist[i:i + a.yigin]
            enc = tok(par, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024)
            enc = {k: v.to(a.dev) for k, v in enc.items()}
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=UP.SICAKLIK,
                                   top_p=UP.TOP_P, max_new_tokens=YENI_JETON,
                                   pad_token_id=tok.pad_token_id)
            yeni = g[:, enc["input_ids"].shape[1]:]
            for j, s in enumerate(yeni):
                t = tok.decode(s, skip_special_tokens=True)
                r = dict(isler[i + j]); r.pop("onek")
                r.update(yazar=kol, aile=m["aile"], cekim=c, satir_ix=n_sat, metin=t,
                         n_jeton=int((s != tok.pad_token_id).sum()))
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_sat += 1
        print(f"  cekim {c+1}/{N_CEKIM} · satir {n_sat} · {time.time()-t0:.0f}s", flush=True)
    fh.close()
    os.replace(tmp, yol)
    sure_uret = time.time() - t0

    if getattr(a, "metin_yalniz", False):
        man = dict(damga=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), yazar=kol,
                   aile=m["aile"], hf=m["hf"], snapshot=os.path.basename(snap), L=L,
                   katmanlar=[], derinlik=[], template=template, n_istem=len(isler),
                   n_cekim=n_cekim, n_satir=n_sat, yeni_jeton=YENI_JETON, seed=SEED,
                   basamak=basamak,
                   sicaklik=UP.SICAKLIK, top_p=UP.TOP_P, yigin=a.yigin, aktivasyon=False,
                   saniye_uretim=round(sure_uret, 1), tamam=True)
        json.dump(man, open(f"{OUT}/manifest_{kol}{ek}.json" + ("_prova" if a.prova else ""),
                            "w"), ensure_ascii=False, indent=1)
        payda(f"yazar_uret_{kol}{ek}", n_satir=n_sat, n_istem=len(isler),
              bekle={"n_satir": len(isler) * n_cekim})
        print(f"{yol} {n_sat} {sure_uret:.0f}",
              flush=True)
        return
    sat = [json.loads(l) for l in open(yol, encoding="utf-8")]
    tok.padding_side = "right"
    H = {k: np.zeros((len(sat), 0), dtype=np.float32) for k in kats}
    buf = {k: [] for k in kats}
    t1 = time.time()
    for i in range(0, len(sat), a.yigin):
        par = [s["metin"] if s["metin"].strip() else " " for s in sat[i:i + a.yigin]]
        enc = tok(par, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(a.dev) for k, v in enc.items()}
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True)
        msk = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
        for k in kats:
            h = o.hidden_states[k].float()
            buf[k].append(((h * msk).sum(1) / msk.sum(1).clamp(min=1)).cpu().numpy())
        del o
        torch.cuda.empty_cache()
    for k in kats:
        H[k] = np.concatenate(buf[k], 0).astype(np.float32)
    np.savez_compressed(f"{OUT}/akt_{kol}{ek}.npz" + ("_prova.npz" if a.prova else ""),
                        satir_ix=np.array([s["satir_ix"] for s in sat]),
                        **{f"h{k}": H[k] for k in kats})
    man = dict(damga=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), yazar=kol,
               aile=m["aile"], hf=m["hf"], snapshot=os.path.basename(snap), L=L,
               katmanlar=kats, derinlik=list(DERINLIK), template=template, n_istem=len(isler),
               n_cekim=N_CEKIM, n_satir=n_sat, yeni_jeton=YENI_JETON, seed=SEED,
               sicaklik=UP.SICAKLIK, top_p=UP.TOP_P, yigin=a.yigin, basamak=basamak,
               saniye_uretim=round(sure_uret, 1), saniye_kodlama=round(time.time() - t1, 1),
               tamam=True)
    json.dump(man, open(f"{OUT}/manifest_{kol}{ek}.json" + ("_prova" if a.prova else ""), "w"),
              ensure_ascii=False, indent=1)
    payda(f"yazar_uret_{kol}{ek}", n_satir=n_sat, n_istem=len(isler), n_katman=len(kats),
          bekle={"n_satir": len(isler) * n_cekim, "n_katman": 5})
    print(f"{yol} {n_sat} {sure_uret:.0f}"
          f"{time.time()-t1:.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", required=True, choices=("envanter", "uret"))
    ap.add_argument("--yazar")
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, default=None)
    ap.add_argument("--yigin", type=int, default=34)
    ap.add_argument("--prova", action="store_true")
    ap.add_argument("--taban", action="store_true", help="K-TAVAN eki: base yazar tablosu")
    ap.add_argument("--kip", default="sohbet", choices=("sohbet", "ham", "sargi"))
    ap.add_argument("--basamak", default="B4", choices=("B0", "B1", "B2", "B3", "B4"),
                    help="K-YAZAR-3: `sert` kolunun merdiven basamagi (varsayilan B4)")
    ap.add_argument("--metin-yalniz", dest="metin_yalniz", action="store_true")
    a = ap.parse_args()
    if a.asama == "envanter":
        R = envanter()
        json.dump(R, open(f"{ROOT}/unreleased/YAZAR_ENVANTER_2026-08-08.json", "w"),
                  ensure_ascii=False, indent=1)
        print(f"→ {ROOT}/unreleased/YAZAR_ENVANTER_2026-08-08.json")
    else:
        uret(a)


if __name__ == "__main__":
    main()
