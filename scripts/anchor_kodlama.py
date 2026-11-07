import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", __DNH_ROOT__ + "/.hf")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from kt_kodlama import load_split, BOLMELER, KapsamHatasi
import muhafiz_sabit_konum as MSK

OUT = __DNH_DATA__ + "/kt_batarya"
MANF = __DNH_ROOT__ + "/unreleased/capa_a8_2026-07-30.json"
DEV = "cuda:1"
NFOLD = 5

KOLLAR = {
    "c1a": dict(model="intfloat/e5-large-v2", onek="query: ", kes=400, maxlen=192,
                havuz="ortalama", dtype="float32", capa=True,
                kayit=dict(kaynak="bs223_midlayer_disapere.py @ e20c41c", L=25, tepe=23,
                           sup={18: 0.781, 19: 0.780, 20: 0.780, 21: 0.793, 22: 0.791,
                                23: 0.800, 24: 0.796},
                           bolme="train+dev'de egitim → held-out TEST (2082 cift)")),
    "c2a": dict(model="BAAI/bge-m3", onek="", kes=1200, maxlen=512,
                havuz="cls", dtype="float32", capa=False,
                kayit=None),
    "c3a": dict(model="BAAI/bge-large-en-v1.5", onek="", kes=400, maxlen=512,
                havuz="cls", dtype="float32", capa=False,
                kayit=None),
    "c4a": dict(model="princeton-nlp/sup-simcse-roberta-large", onek="", kes=400,
                maxlen=512, havuz="cls", dtype="float32", capa=False, kayit=None,
                _sozlesme=""
                          ""),
    "c5a": dict(model="Qwen/Qwen3-Embedding-4B", onek="", kes=400, maxlen=512,
                havuz="sontoken", dtype="float32", capa=False, kayit=None,
                _sozlesme="talimat YOK (belge tarafi) · son-token havuzu · EOS'u "
                          "tokenizer ekler (ölcüldü) · L2 okuma aninda"),
    "c6a": dict(model="BAAI/bge-m3", onek="", kes=400, maxlen=512,
                havuz="cls", dtype="float32", capa=False, kayit=None,
                _sozlesme=""
                          ""
                          ""),
}
KESTIRICI = dict(max_iter=2000, C=0.5, class_weight="balanced")


def veri():
    ebe, cev, y, g, bol = [], [], [], [], []
    for s in BOLMELER:
        r = load_split(s)
        ebe += [x["ebeveyn"] for x in r]; cev += [x["cevap"] for x in r]
        y += [x["etiket"] for x in r]; g += [x["forum_id"] for x in r]; bol += [s] * len(r)
    return ebe, cev, np.array(y), np.array(g), np.array(bol)


def yukle_model(kol):
    import torch
    from transformers import AutoTokenizer, AutoModel
    K = KOLLAR[kol]
    tok = AutoTokenizer.from_pretrained(K["model"])
    m = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(DEV).eval()
    return tok, m


def kodla(tok, model, metinler, kol, bs=64, sayac=None):
    import torch
    K = KOLLAR[kol]
    out = []
    for i in range(0, len(metinler), bs):
        ch = [t if t.strip() else " " for t in metinler[i:i + bs]]
        enc = tok([K["onek"] + t[:K["kes"]] for t in ch], padding=True, truncation=True,
                  max_length=K["maxlen"], return_tensors="pt").to(DEV)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"]
        if sayac is not None:
            sayac["token_toplam"] += int(m.sum())
            sayac["satir"] += len(ch)
            sayac["maxlen_carpan"] += int((m.sum(1) >= K["maxlen"]).sum())
            sayac["bos_havuz"] += int((m.sum(1) == 0).sum())
            sayac["uzunluklar"] += m.sum(1).tolist()
        if K["havuz"] == "ortalama":
            mf = m.unsqueeze(-1).to(hs[0].dtype)
            out.append(np.stack([((h * mf).sum(1) / mf.sum(1).clamp(min=1)).float().cpu().numpy()
                                 for h in hs], 1))
        elif K["havuz"] == "cls":
            out.append(np.stack([h[:, 0, :].float().cpu().numpy() for h in hs], 1))
        elif K["havuz"] == "sontoken":
            son = m.cumsum(1).argmax(1)
            ix = torch.arange(m.shape[0], device=m.device)
            out.append(np.stack([h[ix, son, :].float().cpu().numpy() for h in hs], 1))
        else:
            raise KapsamHatasi(f"bilinmeyen havuz {K['havuz']}")
        del hs
    return np.concatenate(out, 0)


def l2(A):
    return A / np.clip(np.linalg.norm(A, axis=-1, keepdims=True), 1e-9, None)


def dogrula(n):
    import torch
    ebe, cev, y, g, bol = veri()
    print(f"[payda] {len(ebe)} cift yüklendi ({len(np.unique(g))} forum) · ön-ucus n={n}")
    for kol in KOLLAR:
        K = KOLLAR[kol]
        print(f"\n═══ {kol} · {K['model']} · ön-ek {K['onek']!r} · kes {K['kes']} kar / "
              f"{K['maxlen']} token · havuz {K['havuz']} · dtype {K['dtype']}")
        tok, model = yukle_model(kol)
        print(f"  gizli durum sayisi L = {model.config.num_hidden_layers + 1} · "
              f"dim {model.config.hidden_size}")
        sayac = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0, uzunluklar=[])
        E16 = kodla(tok, model, ebe[:n], kol, bs=16, sayac=sayac)
        E64 = kodla(tok, model, ebe[:n], kol, bs=64)
        E64b = kodla(tok, model, ebe[:n], kol, bs=64)
        u = np.array(sayac["uzunluklar"])
        print(f"  [payda] satir {sayac['satir']} · token {sayac['token_toplam']} · "
              f"uzunluk med {np.median(u):.0f} p95 {np.percentile(u,95):.0f} maks {u.max()}")
        print(f"  maxlen'e carpan: {sayac['maxlen_carpan']}/{sayac['satir']} "
              f"(%{100*sayac['maxlen_carpan']/sayac['satir']:.1f})")
        print(f"{sayac['bos_havuz']} {sayac['satir']}"
              f""
              f"{'✓' if sayac['bos_havuz'] == 0 else '✗ DUR'}")
        cs = (l2(E16) * l2(E64)).sum(-1)
        print(f"  yigin-degismezligi bs16↔bs64: cos min {cs.min():.6f} ort {cs.mean():.6f}")
        print(f"  belirlenimlilik (ayni yigin 2×): birebir ayni "
              f"{bool(np.array_equal(E64, E64b))} · maks |Δ| {np.abs(E64-E64b).max():.2e}")
        nf = int((~np.isfinite(E64)).sum())
        print(f"  SONLU: NaN/Inf {nf} · norm araligi {np.linalg.norm(E64,axis=-1).min():.2f}–"
              f"{np.linalg.norm(E64,axis=-1).max():.2f} · fp16 tasma siniri 65504 "
              f"({'güvenli' if np.linalg.norm(E64,axis=-1).max() < 6e4 else 'RISK'})")
        del model, tok, E16, E64, E64b
        torch.cuda.empty_cache()
    print("\n[ön-ucus] hicbir sey yazilmadi.")


def kapi_bolge(P, C, y, g, kol, etiket=""):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    katlar = list(GroupKFold(NFOLD).split(np.zeros(len(y)), y, g))
    L = P.shape[1]
    auc = np.zeros(L)
    for l in range(L):
        p, c = l2(P[:, l, :]), l2(C[:, l, :])
        X = np.column_stack([p, c, np.abs(p - c)])
        oof = np.full(len(y), np.nan)
        for tr, te in katlar:
            m = LogisticRegression(**KESTIRICI).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        auc[l] = roc_auc_score(y, oof)
    print(f"  [{kol}{etiket}] katman AUC (forum-kümeli GroupKFold({NFOLD}) OOF, n={len(y)}):")
    for l in range(L):
        kayit = (KOLLAR[kol]["kayit"] or {}).get("sup", {}).get(l)
        tag = ""
        if l == int(auc.argmax()):
            tag += "  ← TEPE"
        if l == L - 1:
            tag += "  ← son"
        ek = f"   [kayit {kayit:.3f}, Δ {auc[l]-kayit:+.3f}]" if kayit else ""
        if l >= L - 8 or l == int(auc.argmax()):
            print(f"     L{l:02d}  {auc[l]:.4f}{ek}{tag}")
    return auc


def kos(kollar):
    import torch
    t0 = time.time()
    ebe, cev, y, g, bol = veri()
    n_bek = {s: int((bol == s).sum()) for s in BOLMELER}
    print(f"[kapsam] {len(ebe)} cift · {len(np.unique(g))} forum · dispute %{100*y.mean():.1f} · "
          f"train {n_bek['train']} + dev {n_bek['dev']} · TEST ACILMADI", flush=True)
    R = {"_statu": "",
         "_sozlesme": "SOZLESME_capa_A8_2026-07-30.md @ 814c7df",
         "_kapsam": dict(n_cift=len(ebe), n_forum=int(len(np.unique(g))), bolmeler=n_bek,
                         test="ACILMADI", kestirici=KESTIRICI, nfold=NFOLD),
         "kollar": {}}
    if os.path.exists(MANF):
        R = json.load(open(MANF)); R["kollar"] = R.get("kollar", {})
    for kol in kollar:
        K = KOLLAR[kol]; tk = time.time()
        print(f"\n═══ {kol} · {K['model']} · ön-ek {K['onek']!r} · {K['kes']}kar/{K['maxlen']}tok "
              f"· havuz {K['havuz']} · dtype {K['dtype']} · capa {K['capa']}", flush=True)
        tok, model = yukle_model(kol)
        sayac = dict(token_toplam=0, satir=0, maxlen_carpan=0, bos_havuz=0, uzunluklar=[])
        P = kodla(tok, model, ebe, kol, sayac=sayac)
        Cv = kodla(tok, model, cev, kol, sayac=sayac)
        del model, tok
        torch.cuda.empty_cache()
        L, D = P.shape[1], P.shape[2]
        u = np.array(sayac["uzunluklar"])
        yaz, atla, kaps = 0, 0, {}
        for ad, A in (("ebeveyn", P), ("cevap", Cv)):
            for s in BOLMELER:
                sl = slice(0, n_bek["train"]) if s == "train" else slice(n_bek["train"], None)
                blok = A[sl]
                f = f"{OUT}/{kol}__{s}__{ad}__{K['havuz']}.fp16.npy"
                nf = int((~np.isfinite(blok)).sum())
                b16 = blok.astype(np.float16)
                nf16 = int((~np.isfinite(b16)).sum())
                tmp = f + ".tmp.npy"
                np.save(tmp, b16); os.replace(tmp, f)
                yaz += 1
                kaps[os.path.basename(f)] = dict(satir=int(blok.shape[0]), L=L, dim=D,
                                                 sonlu_olmayan_fp32=nf, sonlu_olmayan_fp16=nf16)
                print(f"  [kapsam-v2] {os.path.basename(f)}: yazilan 1 · atlanan 0 · "
                      f"SONLU (fp32 NaN/Inf {nf} · fp16 NaN/Inf {nf16}) · satir {blok.shape[0]}",
                      flush=True)
        if yaz == 0:
            raise KapsamHatasi("0 dosya yazildi — payda SIFIR, gecis degil HATA")
        print(f"  [payda] satir {sayac['satir']} (2×{len(ebe)}) · token {sayac['token_toplam']} · "
              f"uzunluk med {np.median(u):.0f} p95 {np.percentile(u,95):.0f} maks {u.max()} · "
              f"maxlen'e carpan {sayac['maxlen_carpan']} (%{100*sayac['maxlen_carpan']/sayac['satir']:.1f}) · "
              f"BOS HAVUZ {sayac['bos_havuz']} (yapisal 0)", flush=True)
        mh = MSK.tara(P, ebe, n=200, ad=f"{kol} ebeveyn ({K['havuz']})")
        auc32 = kapi_bolge(P, Cv, y, g, kol, " fp32")
        auc16 = kapi_bolge(P.astype(np.float16).astype(np.float32),
                           Cv.astype(np.float16).astype(np.float32), y, g, kol, " fp16")
        kayit = K["kayit"]
        oku = None
        if kayit:
            bolge = [l for l in kayit["sup"] if l < L]
            gz = {l: float(auc32[l]) for l in bolge}
            oku = dict(bolge=bolge, gozlenen={str(l): round(v, 4) for l, v in gz.items()},
                       kayit={str(l): kayit["sup"][l] for l in bolge},
                       mertebe_araligi=[round(min(gz.values()), 4), round(max(gz.values()), 4)],
                       kayit_araligi=[min(kayit["sup"][l] for l in bolge),
                                      max(kayit["sup"][l] for l in bolge)],
                       tepe_gozlenen=int(auc32.argmax()),
                       tepe_kayit=kayit["tepe"],
                       son_dusus_gozlenen=round(float(auc32[-1] - auc32.max()), 4),
                       son_dusus_kayit=round(kayit["sup"][L - 1] - max(kayit["sup"].values()), 4))
            print(f"  → CAPA BÖLGESI L{min(bolge)}–L{max(bolge)}: gözlenen "
                  f"{oku['mertebe_araligi'][0]:.4f}–{oku['mertebe_araligi'][1]:.4f} vs kayit "
                  f"{oku['kayit_araligi'][0]:.3f}–{oku['kayit_araligi'][1]:.3f} · "
                  f"tepe gözlenen L{oku['tepe_gozlenen']} / kayit L{oku['tepe_kayit']} · "
                  f"son-düsüs gözlenen {oku['son_dusus_gozlenen']:+.4f} / kayit "
                  f"{oku['son_dusus_kayit']:+.4f}", flush=True)
        else:
            print(""
                  "", flush=True)
        R["kollar"][kol] = dict(
            model=K["model"], onek=K["onek"], kes=K["kes"], maxlen=K["maxlen"],
            havuz=K["havuz"], dtype=K["dtype"], capa=K["capa"], L=L, dim=D,
            l2_yazimda=False, katman_auc_fp32=[round(float(x), 4) for x in auc32],
            katman_auc_fp16=[round(float(x), 4) for x in auc16],
            fp16_maks_sapma=round(float(np.abs(auc32 - auc16).max()), 5),
            capa_okuma=oku, kapsam=kaps, sabit_konum_muhafizi=mh,
            token=dict(medyan=float(np.median(u)), p95=float(np.percentile(u, 95)),
                       maks=int(u.max()), maxlen_carpan=sayac["maxlen_carpan"],
                       satir=sayac["satir"], bos_havuz=sayac["bos_havuz"]),
            sure_dk=round((time.time() - tk) / 60, 2))
        print(f"  fp16↔fp32 AUC maks sapma {R['kollar'][kol]['fp16_maks_sapma']:.5f} · "
              f"{R['kollar'][kol]['sure_dk']:.1f} dk", flush=True)
        del P, Cv
        json.dump(R, open(MANF, "w"), ensure_ascii=False, indent=1)
    json.dump(R, open(MANF, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {MANF} · toplam {(time.time()-t0)/60:.1f} dk", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogrula", type=int, default=0)
    ap.add_argument("--kos", action="store_true")
    ap.add_argument("--kol", default="c1a,c2a")
    a = ap.parse_args()
    if a.dogrula:
        dogrula(a.dogrula)
    elif a.kos:
        kos([k for k in a.kol.split(",") if k])
    else:
        ap.error("--dogrula N ya da --kos")
