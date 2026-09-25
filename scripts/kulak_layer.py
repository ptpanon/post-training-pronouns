import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, argparse, hashlib

os.environ.setdefault("HF_HOME", __DNH_ROOT__ + "/.hf")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "32")

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")

import numpy as np
import esdegerlik as E

MNT = __DNH_DATA__ + ""
KK = f"{MNT}/kulak_katman"
MAN = f"{ROOT}/unreleased/manifests"
ST5 = "cartgr/embeddings-for-preferences-st5-xl"
ADAPTER = f"{MNT}/unreleased/ear2_adapter"
DEV = "cuda:1"
MAXLEN = 256
HAV = ("ortalama", "sontoken")
KOLLAR = {"stock": None, "ear20": ADAPTER}
os.makedirs(KK, exist_ok=True)


def yaz(*a):
    print(*a, flush=True)


def yukle(adapter):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ST5, device=DEV)
    if adapter:
        try:
            m.load_adapter(adapter)
        except Exception:
            from peft import PeftModel
            m[0].auto_model = PeftModel.from_pretrained(m[0].auto_model, adapter)
    m = m.to(torch.bfloat16)
    return m[0].tokenizer, m[0].auto_model.eval(), m


def kodla(tok, govde, metinler, bs=64):
    import torch
    ilk = tok(["x"], return_tensors="pt").to(DEV)
    with torch.no_grad():
        L = len(govde(**ilk, output_hidden_states=True).hidden_states)
    D = govde.config.d_model
    ORT = np.zeros((len(metinler), L, D), np.float32)
    SON = np.zeros((len(metinler), L, D), np.float32)
    mx = 0.0
    uz = []
    for i in range(0, len(metinler), bs):
        ch = [t if t.strip() else " " for t in metinler[i:i + bs]]
        b = tok(ch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
        m = b["attention_mask"]; uz += m.sum(1).tolist()
        with torch.no_grad():
            hs = torch.stack(govde(**b, output_hidden_states=True).hidden_states)
        mf = m.unsqueeze(0).unsqueeze(-1).to(hs.dtype)
        ort = (hs * mf).sum(2) / mf.sum(2).clamp(min=1)
        son = hs[:, torch.arange(m.shape[0], device=hs.device), m.sum(1) - 1, :]
        o32 = ort.permute(1, 0, 2).float().cpu().numpy()
        s32 = son.permute(1, 0, 2).float().cpu().numpy()
        mx = max(mx, float(np.abs(o32).max()), float(np.abs(s32).max()))
        ORT[i:i + len(ch)] = o32; SON[i:i + len(ch)] = s32
        del hs, mf, ort, son, o32, s32
    return ORT, SON, np.array(uz), mx


def _kaydet(yol, A):
    gec = yol[:-4] + ".tmp.npy"
    np.save(gec, A); os.replace(gec, yol)


PARCA = [("dis", "train"), ("dis", "dev"), ("kia", "train"), ("kia", "heldout")]


def metinleri_al():
    P = {}
    for sp in ("train", "dev"):
        p, c, y, f = E.disapere(sp)
        P[("dis", sp)] = (p, c, y, f)
    (kp, kc, ky, kd), (hp, hc, hy, hd) = E.kialo()
    P[("kia", "train")] = (kp, kc, ky, kd)
    P[("kia", "heldout")] = (hp, hc, hy, hd)
    return P


def kodla_hepsi():
    t0 = time.time(); P = metinleri_al(); olcum = {}
    for kol, ad in KOLLAR.items():
        gerek = [(z, b, t) for z, b in PARCA for t in ("ebeveyn", "cevap")
                 if not all(os.path.exists(f"{KK}/{kol}__{z}__{b}__{t}__{h}.fp32.npy") for h in HAV)]
        if not gerek:
            yaz(f"  [atla] {kol} — 16 dosyanin hepsi diskte"); continue
        yaz(f"  · {kol} yükleniyor (adapter={'VAR' if ad else 'yok'})")
        tok, govde, _ = yukle(ad)
        for z, b in PARCA:
            p, c, y, f = P[(z, b)]
            for taraf, T in (("ebeveyn", p), ("cevap", c)):
                hedef = {h: f"{KK}/{kol}__{z}__{b}__{taraf}__{h}.fp32.npy" for h in HAV}
                if all(os.path.exists(v) for v in hedef.values()):
                    continue
                ORT, SON, uz, mx = kodla(tok, govde, T)
                for h, A in (("ortalama", ORT), ("sontoken", SON)):
                    if not np.isfinite(A).all():
                        raise E.KapsamHatasi(f"{kol}/{z}/{b}/{taraf}/{h}: SONLU DEGIL")
                    _kaydet(hedef[h], A)
                olcum[f"{kol}/{z}/{b}/{taraf}"] = dict(
                    n=len(T), durum=int(ORT.shape[1]), d=int(ORT.shape[2]),
                    tok_ort=float(uz.mean()), tok_maks=int(uz.max()),
                    carpan=int((uz >= MAXLEN).sum()), maks_buyukluk=mx,
                    fp16_tavani_asan=bool(mx > 65504))
                yaz(f"    {kol}/{z}/{b}/{taraf}: n={len(T)} durum={ORT.shape[1]} d={ORT.shape[2]} "
                    f"· token ort {uz.mean():.1f} maks {uz.max()} · "
                    f"max_length={MAXLEN}'e carpan **{int((uz>=MAXLEN).sum())}/{len(uz)}** "
                    f"· maks |h| = {mx:.4g} (fp16 tavani 65504 ⇒ "
                    f"{'ASIYOR, fp32 ZORUNLU' if mx > 65504 else 'asmiyor'})")
        del tok, govde
        import torch; torch.cuda.empty_cache()
    if olcum:
        json.dump(olcum, open(f"{KK}/kodlama_olcum.json", "w"), ensure_ascii=False, indent=1)
    yaz(f"  kodlama bitti ({time.time()-t0:.0f}s)")


def kat(kol, z, b, taraf, h, L):
    A = np.load(f"{KK}/{kol}__{z}__{b}__{taraf}__{h}.fp32.npy", mmap_mode="r")
    return np.asarray(A[:, L], np.float32)


def egri_kulak(kol, z, egit, olc, ytr, yte, nL):
    Z = E.ZEMIN["disapere" if z == "dis" else "kialo"]
    out = []
    for h in HAV:
        for L in range(nL):
            X = E.ozellik(kat(kol, z, egit, "ebeveyn", h, L), kat(kol, z, egit, "cevap", h, L))
            Y = E.ozellik(kat(kol, z, olc, "ebeveyn", h, L), kat(kol, z, olc, "cevap", h, L))
            out.append(dict(katman=L, havuz=h,
                            auc=E.auc(yte, E.probe(X, ytr, Y, Z["dengeli"], Z["iters"]))))
    return out


def egri_qwen(z, egit, olc, ytr, yte):
    zem = "disapere" if z == "dis" else "kialo"
    Z = E.ZEMIN[zem]
    nL = np.load(f"{E.QW}/dev__ebeveyn__sontoken.fp16.npy", mmap_mode="r").shape[1]
    out = []
    for h in E.HAVUZLAR:
        for L in range(nL):
            X = E.ozellik(*E.qw_yigin(zem, [egit], L, h))
            Y = E.ozellik(*E.qw_yigin(zem, [olc], L, h))
            out.append(dict(katman=L, havuz=h,
                            auc=E.auc(yte, E.probe(X, ytr, Y, Z["dengeli"], Z["iters"]))))
    return out


def egriler():
    t0 = time.time()
    _, _, ytr_d, _ = E.disapere("train"); _, _, yde, _ = E.disapere("dev")
    (_, _, ytr_k, _), (_, _, yhe, _) = E.kialo()
    nL = np.load(f"{KK}/stock__dis__dev__ebeveyn__ortalama.fp32.npy", mmap_mode="r").shape[1]
    yaz(f"{nL}")
    R = {"_statu": "KESIF/KARAKTERIZASYON — verdict YOK, yeni prereg YOK.",
         "_tarih": "2026-07-31", "_govde": dict(model=ST5, tip="T5EncoderModel",
                                                katman=24, durum=nL, d_model=1024,
                                                params_G=1.2417),
         "_protokol": {"dis": "",
                       "kia": "train→HELDOUT (mührün ÖLCÜM protokolü)"},
         "egriler": {}, "uc_nokta": {}}
    for z, egit, olc, ytr, yte in (("dis", "train", "dev", ytr_d, yde),
                                   ("kia", "train", "heldout", ytr_k, yhe)):
        for kol in KOLLAR:
            ts = time.time()
            R["egriler"][f"{kol}/{z}"] = egri_kulak(kol, z, egit, olc, ytr, yte, nL)
            b = max(R["egriler"][f"{kol}/{z}"], key=lambda r: r["auc"])
            yaz(f"  {kol}/{z}: tepe ℓ{b['katman']}/{b['havuz']} = {b['auc']:.4f} "
                f"({time.time()-ts:.0f}s)")
        ts = time.time()
        R["egriler"][f"qwen/{z}"] = egri_qwen(z, egit, olc, ytr, yte)
        b = max(R["egriler"][f"qwen/{z}"], key=lambda r: r["auc"])
        yaz(f"  qwen/{z}: tepe ℓ{b['katman']}/{b['havuz']} = {b['auc']:.4f} ({time.time()-ts:.0f}s)")

        zem = "disapere" if z == "dis" else "kialo"
        for kol in KOLLAR:
            Xtr = E.ozellik(*E.ear_yigin(zem, [egit], kol))
            Xte = E.ozellik(*E.ear_yigin(zem, [olc], kol))
            Z = E.ZEMIN[zem]
            a = E.auc(yte, E.probe(Xtr, ytr, Xte, Z["dengeli"], Z["iters"]))
            son = [r for r in R["egriler"][f"{kol}/{z}"]
                   if r["katman"] == nL - 1 and r["havuz"] == "ortalama"][0]["auc"]
            R["uc_nokta"][f"{kol}/{z}"] = dict(dense_768=a, son_katman_ortalama=son,
                                               dense_katkisi=a - son)
            yaz(f"  UC {kol}/{z}: son-katman+ort {son:.4f} → +Dense(768) {a:.4f} "
                f"(Dense katkisi {a-son:+.4f})")
    json.dump(R, open(f"{MAN}/kulak_katman_2026-07-31.json", "w"), ensure_ascii=False, indent=1)
    yaz(f"══ bitti ({time.time()-t0:.0f}s) → manifests/kulak_katman_2026-07-31.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodla", action="store_true")
    ap.add_argument("--egri", action="store_true")
    A = ap.parse_args()
    if A.kodla:
        kodla_hepsi()
    if A.egri:
        egriler()
    if not (A.kodla or A.egri):
        ap.print_help()
