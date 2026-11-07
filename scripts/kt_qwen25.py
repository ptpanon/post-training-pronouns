import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", __DNH_ROOT__ + "/.hf")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kt_kodlama import load_split, BOLMELER, KES, KapsamHatasi

MODEL = "Qwen/Qwen2.5-7B"
OUT = __DNH_DATA__ + "/kt_qwen25"
MAXLEN = 512
ARSIV_MAXTOK = 128
TARAFLAR = ("ebeveyn", "cevap")
HAVUZLAR = ("sontoken", "ortalama_sinksiz")
DEV = "cuda:1"


def yukle_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    return tok, model


def kodla(tok, model, metinler, bs=32, sayac=None, maxlen=None, dev=None, duz_ort=False):
    import torch, numpy as np
    MAXLEN = globals()["MAXLEN"] if maxlen is None else maxlen
    DEV = globals()["DEV"] if dev is None else dev
    def _cfg(cfg, ad):
        if hasattr(cfg, ad):
            return getattr(cfg, ad)
        for alt in ("text_config", "language_model_config", "decoder_config"):
            if hasattr(cfg, alt) and hasattr(getattr(cfg, alt), ad):
                return getattr(getattr(cfg, alt), ad)
        raise AttributeError(f"{ad}")
    L = _cfg(model.config, "num_hidden_layers") + 1
    D = _cfg(model.config, "hidden_size")
    ST = np.zeros((len(metinler), L, D), np.float16)
    OR = np.zeros((len(metinler), L, D), np.float16)
    DZ = np.zeros((len(metinler), L, D), np.float16) if duz_ort else None
    uzun = []
    for i in range(0, len(metinler), bs):
        ch = [t if t.strip() else " " for t in metinler[i:i + bs]]
        b = tok(ch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
        m = b["attention_mask"]
        uzun += m.sum(1).tolist()
        with torch.no_grad():
            hs = torch.stack(model(**b, output_hidden_states=True).hidden_states)
        B_, S = m.shape
        son = m.sum(1) - 1
        st = hs[:, torch.arange(B_, device=hs.device), son, :]
        norms = hs.float().norm(dim=-1)
        gecerli = m.bool().unsqueeze(0).expand(L, -1, -1)
        med = torch.nanmedian(norms.masked_fill(~gecerli, float("nan")),
                              dim=2, keepdim=True).values
        tut = gecerli & (norms <= 8 * med)
        tut[:, :, 0] = False
        bos = tut.sum(2) == 0
        yedek = gecerli & (torch.arange(S, device=hs.device)[None, None] >= 1)
        tut = torch.where(bos.unsqueeze(2), yedek, tut)
        bos2 = tut.sum(2) == 0
        tut = torch.where(bos2.unsqueeze(2), gecerli, tut)
        if sayac is not None:
            sayac["gecerli_token"] += int(gecerli[0].sum())
            sayac["pos0_elenen"] += int(gecerli[0, :, 0].sum())
            sayac["sink_elenen"] += int((gecerli & (norms > 8 * med)).sum())
            sayac["yedege_dusen"] += int(bos.sum())
            sayac["ucuncu_basamak"] += int(bos2.sum())
            sayac["bos_havuz"] += int((tut.sum(2) == 0).sum())
        orp = (hs * tut.unsqueeze(-1)).sum(2) / tut.sum(2, keepdim=True).clamp(min=1)
        ST[i:i + len(ch)] = st.permute(1, 0, 2).float().cpu().numpy().astype(np.float16)
        OR[i:i + len(ch)] = orp.permute(1, 0, 2).float().cpu().numpy().astype(np.float16)
        if DZ is not None:
            gf = gecerli.unsqueeze(-1).to(hs.dtype)
            dzp = (hs * gf).sum(2) / gf.sum(2).clamp(min=1)
            DZ[i:i + len(ch)] = dzp.permute(1, 0, 2).float().cpu().numpy().astype(np.float16)
            del gf, dzp
        del hs, norms, tut, st, orp
    return (ST, OR, uzun, DZ) if DZ is not None else (ST, OR, uzun)


def onucus(n):
    import numpy as np
    print(f"══ ÖN-UCUS · {MODEL} · n={n} ══", flush=True)
    satir = load_split("train")[:n]
    met = [r["ebeveyn"] for r in satir]
    tok, model = yukle_model()
    print(f"  model: katman={model.config.num_hidden_layers} (+gömü ⇒ {model.config.num_hidden_layers+1} durum) "
          f"· dim={model.config.hidden_size} · dolgu={tok.padding_side} · pad={tok.pad_token!r} "
          f"· sohbet-sablonu UYGULANMADI (base)", flush=True)

    s1 = dict.fromkeys(("gecerli_token", "pos0_elenen", "sink_elenen", "yedege_dusen",
                        "ucuncu_basamak", "bos_havuz"), 0)
    A_st, A_or, uz = kodla(tok, model, met, bs=32, sayac=s1)
    B_st, B_or, _ = kodla(tok, model, met, bs=8)
    C_st, C_or, _ = kodla(tok, model, met, bs=32)

    def cos(X, Y):
        x = X.reshape(len(X), -1).astype(np.float32); y = Y.reshape(len(Y), -1).astype(np.float32)
        return ((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9))

    print(f"\n  [1] YIGIN-DEGISMEZLIGI (bs=32 ↔ bs=8) — payda {len(met)} metin")
    ok = True
    for ad, X, Y in (("sontoken", A_st, B_st), ("ortalama_sinksiz", A_or, B_or)):
        c = cos(X, Y); print(f"      {ad:18s} cos ort {c.mean():.6f} · min {c.min():.6f}")
        ok &= bool(c.min() > 0.9999)
    print(f"  [2] BELIRLENIMLILIK (ayni bs=32, iki kosu) — payda {len(met)}")
    for ad, X, Y in (("sontoken", A_st, C_st), ("ortalama_sinksiz", A_or, C_or)):
        c = cos(X, Y); print(f"      {ad:18s} cos min {c.min():.6f} · birebir {int((X==Y).all(axis=(1,2)).sum())}/{len(met)}")

    uz = np.array(uz)
    print(f"  [3] KESME — payda {len(uz)} metin (900-karakter kanonik kesmeden SONRA)")
    print(f"      token ort {uz.mean():.1f} · medyan {int(np.median(uz))} · maks {uz.max()}")
    print(f"      max_length={MAXLEN}'e carpan: {int((uz>=MAXLEN).sum())}/{len(uz)}")
    print(f"      arsivin MAXTOK={ARSIV_MAXTOK}'i asan (orijinalin KESECEGI): {int((uz>ARSIV_MAXTOK).sum())}/{len(uz)}")
    print(f"  [4] SINK-ELEME KAPSAMI — payda {s1['gecerli_token']} gecerli token")
    print(f"      pos0 elenen {s1['pos0_elenen']} · norm-sink elenen {s1['sink_elenen']} "
          f"(katman×token birimi) · yedege düsen satir-katman {s1['yedege_dusen']} "
          f"· ÜCÜNCÜ BASAMAK {s1['ucuncu_basamak']} · BOS HAVUZ {s1['bos_havuz']}")
    if s1["sink_elenen"] == 0:
        print("")
    if s1["bos_havuz"] != 0:
        raise KapsamHatasi(f"{s1['bos_havuz']}")
    print(f"\n  KARAR: yigin-degismezligi {'GECTI' if ok else 'KALDI'}")
    if not ok:
        raise SystemExit("[arm2] yigin-degismezligi KALDI — tam kosu YAPILMAZ")
    json.dump(dict(model=MODEL, n=len(met), maxlen=MAXLEN, kes_karakter=KES,
                   yigin_degismezligi="GECTI", token_ort=float(uz.mean()),
                   token_maks=int(uz.max()), maxlen_carpan=int((uz >= MAXLEN).sum()),
                   arsiv_maxtok_asan=int((uz > ARSIV_MAXTOK).sum()), sink_kapsam=s1),
              open("/tmp/kt_qwen25_onucus.json", "w"), ensure_ascii=False, indent=1)


def kos():
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time(); tok = model = None
    yazilan, atlanan = 0, 0
    for split in BOLMELER:
        satir = load_split(split)
        n = len(satir)
        for taraf in TARAFLAR:
            yollar = {h: f"{OUT}/{split}__{taraf}__{h}.fp16.npy" for h in HAVUZLAR}
            if all(os.path.exists(p) for p in yollar.values()):
                print(f"[atla] {split}/{taraf} — iki havuz da diskte", flush=True)
                atlanan += len(yollar); continue
            if model is None:
                tok, model = yukle_model()
                print(f"[model] {MODEL} · {model.config.num_hidden_layers+1} durum × "
                      f"{model.config.hidden_size} dim · {DEV}", flush=True)
            sayac = dict.fromkeys(("gecerli_token", "pos0_elenen", "sink_elenen",
                                   "yedege_dusen", "ucuncu_basamak", "bos_havuz"), 0)
            ST, OR, uz = kodla(tok, model, [r[taraf] for r in satir], bs=32, sayac=sayac)
            if sayac["bos_havuz"]:
                raise KapsamHatasi(f"[{split}/{taraf}] BOS HAVUZ {sayac['bos_havuz']} — YAZILMAZ")
            for h, A in (("sontoken", ST), ("ortalama_sinksiz", OR)):
                if A.shape[0] != n:
                    raise KapsamHatasi(f"[{split}/{taraf}/{h}] satir {A.shape[0]} ≠ beklenen {n}")
                np.save(yollar[h], A); yazilan += 1
            uz = np.array(uz)
            print(f"[kapsam] {split}/{taraf}: satir {n} · yazilan 2 · atlanan 0 · "
                  f"token ort {uz.mean():.1f} maks {uz.max()} · maxlen-carpan {int((uz>=MAXLEN).sum())} "
                  f"· sink-elenen {sayac['sink_elenen']} · ücüncü-basamak {sayac['ucuncu_basamak']} "
                  f"({sayac['ucuncu_basamak']//(model.config.num_hidden_layers+1)} satir) · bos-havuz 0", flush=True)
            del ST, OR
        meta = f"{OUT}/{split}__meta.json"
        if not os.path.exists(meta):
            json.dump([{k: r[k] for k in ("etiket", "forum_id", "dosya", "cumle_idx")}
                       for r in satir], open(meta, "w"))
        print(f"[kapsam] {split}: {n} satir · {len(set(r['forum_id'] for r in satir))} forum "
              f"· {len(set(r['dosya'] for r in satir))} dosya", flush=True)

    bek = [f"{s}__{t}__{h}.fp16.npy" for s in BOLMELER for t in TARAFLAR for h in HAVUZLAR]
    var = [f for f in bek if os.path.exists(f"{OUT}/{f}")]
    tot = sum(os.path.getsize(f"{OUT}/{f}") for f in var)
    print(f"\n[disk] {len(var)}/{len(bek)} dosya · {tot/1024**3:.2f} GB · {(time.time()-t0)/60:.1f} dk")
    if len(var) != len(bek):
        raise KapsamHatasi(f"EKSIK: {set(bek)-set(var)}")
    for f in var:
        a = np.load(f"{OUT}/{f}", mmap_mode="r")
        print(f"   {f:44s} {a.shape} {a.dtype}")
    print(f"[özet] yazilan {yazilan} · atlanan {atlanan} · test bölmesi ACILMADI")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogrula", type=int, default=0)
    ap.add_argument("--kos", action="store_true")
    a = ap.parse_args()
    if a.dogrula:
        onucus(a.dogrula)
    elif a.kos:
        kos()
    else:
        ap.error("--dogrula N ya da --kos")
