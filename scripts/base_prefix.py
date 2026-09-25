#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, glob, argparse, collections, hashlib, random
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import weight_dosyalari as AGD

MNT = __DNH_DATA__ + ""
KOK = f"{MNT}/unreleased/saved-corpora"
GOMU = f"{MNT}/rk_p4/gomu"
OUT = f"{MNT}/rk_base_onek"
CIK = f"{ROOT}/unreleased/line"
SEED, N_EBEVEYN, K_ONEK, N_HEDEF = 20260809, 200, 8, 200
N_CEKIM, GEREKLI_N = 5, 2
PILOT_N, PILOT_CEKIM, MAKS_JETON = 6, 2, 320
os.makedirs(OUT, exist_ok=True)

MODELLER = {
    "qwen_base":     "<home>/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B",
    "qwen_instruct": "<home>/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct",
    "mistral_base":  f"{MNT}/.hf_local/hub/models--mistralai--Mistral-7B-v0.3",
    "mistral_instruct": "<home>/user/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3",
    "gemma_base":     __DNH_ROOT__ + "/.hf/hub/models--google--gemma-4-12b",
    "gemma_instruct": __DNH_ROOT__ + "/.hf/hub/models--google--gemma-4-12B-it",
}
BEK = {"qwen_base": (28, 3584), "qwen_instruct": (28, 3584),
       "mistral_base": (32, 4096), "mistral_instruct": (32, 4096),
       "gemma_base": (48, 3840), "gemma_instruct": (48, 3840)}
ZORUNLU = ("config.json", "tokenizer_config.json", "tokenizer.json")
X_KORPUS, Y_KORPUS = "subreddit-NeutralPolitics", "subreddit-Firearms"


def _snap(kok):
    s = sorted(glob.glob(f"{kok}/snapshots/*"))
    return s[-1] if s else None


def kapi_onkosul():
    red, det = collections.Counter(), {}
    for ad, kok in MODELLER.items():
        s = _snap(kok)
        if s is None:
            red["snapshot_yok"] += 1; det[ad] = "SNAPSHOT YOK"; continue
        eks = [f for f in ZORUNLU if not os.path.exists(f"{s}/{f}")]
        st = AGD.dosyalar(s)
        gb = sum(os.path.getsize(os.path.realpath(f)) for f in st) / 2**30
        c = json.load(open(f"{s}/config.json"))
        tc = c.get("text_config", c)
        uy = (tc["num_hidden_layers"], tc["hidden_size"]) != BEK[ad]
        red["eksik_dosya"] += len(eks); red["safetensors_yok"] += int(not st)
        red["anatomi_uyusmaz"] += int(uy)
        det[ad] = dict(snapshot=os.path.basename(s), n_safetensors=len(st), gb=round(gb, 1),
                       katman=tc["num_hidden_layers"], d=tc["hidden_size"],
                       vocab=tc.get("vocab_size", c.get("vocab_size")), eksik=eks,
                       anatomi_uyusmaz=bool(uy))
    payda("kapi_onkosul", n_model=len(MODELLER), n_zorunlu_dosya=len(ZORUNLU),
          red_toplam=sum(red.values()), **{f"red_{k}": v for k, v in red.items()})
    for ad, d in det.items():
        print(f"    {ad:18s} {d if isinstance(d, str) else f'{d[chr(34)] if False else d}'}"
              if isinstance(d, str) else
              f"    {ad:18s} L={d['katman']} d={d['d']} V={d['vocab']} · "
              f"{d['n_safetensors']} safetensors · {d['gb']} GB · eksik={d['eksik']}", flush=True)
    json.dump(det, open(f"{OUT}/kapi_onkosul.json", "w"), indent=1)
    if sum(red.values()):
        raise RuntimeError(f"ÖN-KOSUL KAPISI DÜSTÜ: {dict(red)}")
    print("  ✓ ön-kosul: dört model TAM, anatomi esli (base ↔ instruct)", flush=True)
    return det


def korpus_oku(c):
    U = {}
    for line in open(f"{KOK}/{c}/utterances.jsonl", encoding="utf-8", errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = (d.get("text") or "").strip()
        if len(t.split()) < 15 or len(t.split()) > 200:
            continue
        U[d["id"]] = (t, d.get("user"), d.get("root"))
    return U


def kapi_taklit(rng):
    paket = {}
    for rol, c in (("X", X_KORPUS), ("Y", Y_KORPUS)):
        U = korpus_oku(c)
        yaz = collections.defaultdict(list)
        for i, (t, u, r) in U.items():
            if u and u not in ("[deleted]", "AutoModerator", None):
                yaz[u].append((i, t, r))
        adaylar = sorted(yaz)
        rng.shuffle(adaylar)
        onek_y = adaylar[:K_ONEK]
        hedef_y = adaylar[K_ONEK:K_ONEK + N_HEDEF]
        onek = [(u,) + yaz[u][0] for u in onek_y]
        onek_ip0 = {o[3] for o in onek}
        hedef = []
        for u in adaylar[K_ONEK:]:
            kayit = yaz[u][0]
            if kayit[2] in onek_ip0:
                continue
            hedef.append((u,) + kayit)
            if len(hedef) >= N_HEDEF:
                break
        onek_ip = {o[3] for o in onek}; hedef_ip = {h[3] for h in hedef}
        cak_y = len({o[0] for o in onek} & {h[0] for h in hedef})
        cak_ip = len(onek_ip & hedef_ip)
        paket[rol] = dict(korpus=c, n_onek_yazar=len({o[0] for o in onek}),
                          n_hedef_yazar=len({h[0] for h in hedef}),
                          onek=[o[2] for o in onek], hedef=[h[2] for h in hedef],
                          red_yazar_cakismasi=cak_y, red_iplik_cakismasi=cak_ip)
    payda("kapi_taklit", n_rol=2, n_onek_ornek=K_ONEK, n_hedef=N_HEDEF,
          red_yazar_cakismasi=sum(paket[r]["red_yazar_cakismasi"] for r in paket),
          red_iplik_cakismasi=sum(paket[r]["red_iplik_cakismasi"] for r in paket),
          red_yetersiz_yazar=sum(int(paket[r]["n_onek_yazar"] < K_ONEK) for r in paket))
    for r in paket:
        print(f"    {r} ({paket[r]['korpus']}): önek {paket[r]['n_onek_yazar']} ayri yazar · "
              f"hedef {paket[r]['n_hedef_yazar']} · yazar-cakismasi "
              f"{paket[r]['red_yazar_cakismasi']} · iplik-cakismasi "
              f"{paket[r]['red_iplik_cakismasi']}", flush=True)
    json.dump({r: {k: v for k, v in paket[r].items() if k not in ("onek", "hedef")}
               for r in paket}, open(f"{OUT}/kapi_taklit_ozet.json", "w"), indent=1)
    json.dump(paket, open(f"{OUT}/kapi_taklit_paket.json", "w"), indent=1)
    kotu = sum(paket[r]["red_yazar_cakismasi"] + paket[r]["red_iplik_cakismasi"] for r in paket)
    az = sum(int(paket[r]["n_onek_yazar"] < K_ONEK) for r in paket)
    if kotu or az:
        raise RuntimeError(f"TAKLIT SINIRI KAPISI DÜSTÜ: cakisma={kotu} yetersiz_yazar={az}")
    print(""
          "", flush=True)
    return paket


def kapi_cihaz(gerek_gb=17):
    import torch
    bos = []
    for i in range(torch.cuda.device_count()):
        f, t = torch.cuda.mem_get_info(i)
        bos.append((f / 2**30, t / 2**30, i))
    print("    " + " · ".join(f"cuda:{i} bos {f:.1f}/{t:.0f} GB"
                              for f, t, i in sorted(bos, key=lambda x: x[2])), flush=True)
    uygun = [b for b in bos if b[0] >= gerek_gb]
    payda("kapi_cihaz", n_cihaz=len(bos), n_uygun=len(uygun),
          red_yetersiz=len(bos) - len(uygun))
    if not uygun:
        raise RuntimeError(f"{gerek_gb}")
    uygun.sort(reverse=True)
    d = f"cuda:{uygun[0][2]}"
    print(f"  ✓ cihaz: {d} (bos {uygun[0][0]:.1f} GB) — sahip izniyle", flush=True)
    return d


def onek_metni(paket, rol):
    return "\n\n".join(f"---\n{t}" for t in paket[rol]["onek"][:K_ONEK])


def kapi_uygulanabilirlik_ve_rejim(dev, paket, ebe, modeller=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from adil_mac import istem
    sonuc = {}
    onek = onek_metni(paket, "X")
    for ad in (modeller or ("qwen_base", "qwen_instruct", "mistral_base", "mistral_instruct")):
        s = _snap(MODELLER[ad])
        tok = AutoTokenizer.from_pretrained(s)
        pad_yok = tok.pad_token is None
        if pad_yok:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        m = AutoModelForCausalLM.from_pretrained(s, dtype=torch.float16, device_map=dev).eval()
        istemler = [onek + "\n\n---\n" + istem(None, e, "") for e in ebe[:PILOT_N]]
        enc = tok(istemler, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(dev)
        with torch.no_grad():
            g = m.generate(**enc, do_sample=True, temperature=0.9, top_p=0.95,
                           max_new_tokens=MAKS_JETON, num_return_sequences=PILOT_CEKIM,
                           pad_token_id=tok.pad_token_id)
        yeni = g[:, enc["input_ids"].shape[1]:]
        met = tok.batch_decode(yeni, skip_special_tokens=True)
        bos_n = sum(1 for t in met if len(t.strip().split()) < 5)
        def yin(t):
            w = t.split()
            if len(w) < 10:
                return 1.0
            gr = [" ".join(w[i:i + 5]) for i in range(len(w) - 4)]
            return collections.Counter(gr).most_common(1)[0][1] / len(gr)
        yr = [yin(t) for t in met]
        og = set()
        ow = onek.split()
        for i in range(len(ow) - 4):
            og.add(" ".join(ow[i:i + 5]))
        def kop(t):
            w = t.split()
            if len(w) < 5:
                return 0.0
            gr = [" ".join(w[i:i + 5]) for i in range(len(w) - 4)]
            return sum(1 for x in gr if x in og) / len(gr)
        kr = [kop(t) for t in met]
        kesik = sum(1 for i in range(yeni.shape[0])
                    if int(yeni[i].shape[0]) >= MAKS_JETON and
                    tok.eos_token_id not in yeni[i].tolist())
        sonuc[ad] = dict(n_cekim=len(met), bos=bos_n,
                         yineleme_medyan=float(np.median(yr)), yineleme_maks=float(max(yr)),
                         kopya_orani_medyan=float(np.median(kr)),
                         kopya_orani_maks=float(max(kr)), kesik=int(kesik),
                         pad_atandi=bool(pad_yok), pad=repr(tok.pad_token),
                         ort_kelime=float(np.mean([len(t.split()) for t in met])))
        print(f"    {ad:18s} bos {bos_n}/{len(met)} · yineleme med {np.median(yr):.3f} · "
              f"★kopya-orani med {np.median(kr):.3f} maks {max(kr):.3f} · kesik {kesik} · "
              f"ort {np.mean([len(t.split()) for t in met]):.0f} kelime · pad_atandi={pad_yok}",
              flush=True)
        with open(f"{OUT}/pilot__{ad}.jsonl", "w") as fh:
            for i, t in enumerate(met):
                fh.write(json.dumps({"i": i, "metin": t}, ensure_ascii=False) + "\n")
        del m
        torch.cuda.empty_cache()
    json.dump(sonuc, open(f"{OUT}/kapi_uygulanabilirlik.json", "w"), indent=1)
    kotu = {}
    for a, v in sonuc.items():
        p = v["bos"] / max(v["n_cekim"], 1)
        v["red_orani"] = float(p)
        v["butce_yeter"] = bool(N_CEKIM * (1 - p) >= GEREKLI_N)
        if not v["butce_yeter"]:
            kotu[a] = p
    P_BAR = 1 - GEREKLI_N / N_CEKIM
    payda("kapi_butce", n_model=len(sonuc), n_cekim_model_basina=PILOT_N * PILOT_CEKIM,
          red_butce_yetmez=len(kotu))
    print(f"{N_CEKIM} {GEREKLI_N} {P_BAR:.2f}"
          f"" +
          " · ".join(f"{a}={sonuc[a]['red_orani']:.3f}" for a in sonuc), flush=True)
    print(f"    kesik (BETIMLEYICI, 320-jeton penceresi): " +
          " · ".join(f"{a}={sonuc[a]['kesik']}/{sonuc[a]['n_cekim']}" for a in sonuc), flush=True)
    if kotu:
        raise RuntimeError(f"{kotu} {P_BAR:.2f}")
    json.dump(sonuc, open(f"{OUT}/kapi_uygulanabilirlik.json", "w"), indent=1)
    print(f"  ✓ uygulanabilirlik: {len(sonuc)} modelin {len(sonuc)}'i bütceyle uyumlu "
          "gecerli-n üretiyor", flush=True)
    return sonuc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu-gerek", type=float, default=17)
    ap.add_argument("--aile", default="ikili",
                    help="ikili = KARAR-42'nin iki ailesi (varsayilan, DEGISMEDI) · gemma = EK-6")
    a = ap.parse_args()
    AILE_MODEL = {"ikili": ("qwen_base", "qwen_instruct", "mistral_base", "mistral_instruct"),
                  "gemma": ("gemma_base", "gemma_instruct")}
    print("═" * 96); print("")
    rng = random.Random(SEED)
    print("\n[1] ÖN-KOSUL KAPISI (dosya düzeyinde)"); kapi_onkosul()
    print("\n[2] TAKLIT SINIRI KAPISI"); paket = kapi_taklit(rng)
    print("\n[3] CIHAZ KAPISI"); dev = kapi_cihaz(a.gpu_gerek)
    print("\n[4] UYGULANABILIRLIK + §7.4 REJIM + EK-1 KOPYA-ORANI")
    from adil_mac import ebeveyn_havuzu
    ebe = ebeveyn_havuzu(N_EBEVEYN)
    kapi_uygulanabilirlik_ve_rejim(dev, paket, ebe, modeller=AILE_MODEL[a.aile])
    print("")
