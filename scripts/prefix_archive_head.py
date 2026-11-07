#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os
import sys
import json
import time
import glob
import hashlib
import argparse
import numpy as np

ROOT = __DNH_ROOT__ + ""
os.environ.setdefault("HF_HOME", __DNH_DATA__ + "/.hf_local")
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
import prefix_ladder as M
import prefix_ladder_gomu as G

OUT = __DNH_DATA__ + "/onek_korpus/arsiv_head"
OKURLAR = ("c1a", "c2a")
BS = 32


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def wo_yaz(mdl, okur):
    import torch
    W, b = [], []
    for lay in mdl.encoder.layer:
        d = lay.attention.output.dense
        W.append(d.weight.detach().cpu().numpy().T)
        b.append(d.bias.detach().cpu().numpy())
    W = np.stack(W).astype(np.float16)
    b = np.stack(b).astype(np.float32)
    pw, pb = f"{OUT}/WO_{okur}.fp16.npy", f"{OUT}/WOb_{okur}.fp32.npy"
    np.save(pw, W)
    np.save(pb, b)
    return dict(W_O=dict(yol=os.path.basename(pw), sekil=list(W.shape), sha256=sha(pw)),
                bias=dict(yol=os.path.basename(pb), sekil=list(b.shape), sha256=sha(pb)))


def kodla_ctx(tok, mdl, metinler, okur, dev):
    import torch
    import anchor_kodlama as CK
    K = CK.KOLLAR[okur]
    yakala = {}

    def kanca(i):
        def f(_mod, girdi):
            yakala[i] = girdi[0]
            return None
        return f

    tutamak = [lay.attention.output.dense.register_forward_pre_hook(kanca(i))
               for i, lay in enumerate(mdl.encoder.layer)]
    out, kes, maxlen = [], K["kes"], K["maxlen"]
    try:
        for i in range(0, len(metinler), BS):
            par = [K["onek"] + t[:kes] for t in metinler[i:i + BS]]
            b = tok(par, return_tensors="pt", padding=True, truncation=True,
                    max_length=maxlen).to(dev)
            with torch.no_grad():
                mdl(**b)
            m = b["attention_mask"].unsqueeze(-1).float()
            yig = []
            for l in range(len(tutamak)):
                C = yakala[l].float()
                yig.append(C[:, 0, :] if K["havuz"] == "cls"
                           else (C * m).sum(1) / m.sum(1).clamp(min=1e-9))
            out.append(torch.stack(yig, 1).cpu().numpy())
    finally:
        for t in tutamak:
            t.remove()
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModel
    import anchor_kodlama as CK
    os.makedirs(OUT, exist_ok=True)
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    C = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=False, etiket="arsiv_head")
    mp = f"{OUT}/manifest.json"
    MF = json.load(open(mp)) if os.path.exists(mp) else dict(hucreler={}, okurlar={})
    MF.update(damga_son=damga, gpu=C, statu="ARSIV — kalici, silme yok",
              format=dict(
                  nesne="W_O ÖNCESI birlesik dikkat baglami, katman basina havuzlanmis",
                  sekil="[n, L, d]",
                  head_geri_turetme=(""
                                     ""
                                     ""),
                  havuz_izdusum_degisme=("W_O dogrusal, ortalama-havuz dogrusal ⇒ havuzla-sonra-"
                                         "izdüsür = izdüsür-sonra-havuzla; `cls`'te tek token"),
                  yapilamaz="token-ICI head analizi (token ekseni saklanmadi; bedel beyanli)"))
    hucreler = [(k, b) for k in M.KOLLAR for b in M.SIRA]
    print(f"AKTIVASYON ARSIVI · damga {damga} · {len(OKURLAR)} okur × {len(hucreler)} hücre")
    for okur in OKURLAR:
        K = CK.KOLLAR[okur]
        tok = AutoTokenizer.from_pretrained(K["model"])
        mdl = AutoModel.from_pretrained(K["model"], dtype=torch.float32).to(a.dev).eval()
        cfg = mdl.config
        MF["okurlar"][okur] = dict(
            model=K["model"], L=int(cfg.num_hidden_layers), d=int(cfg.hidden_size),
            n_head=int(cfg.num_attention_heads),
            head_dim=int(cfg.hidden_size // cfg.num_attention_heads),
            havuz=K["havuz"], onek=K["onek"], kes=K["kes"], maxlen=K["maxlen"],
            **wo_yaz(mdl, okur))
        for kol, bas in hucreler:
            ad = f"{okur}__{kol}__{bas}"
            npy = f"{OUT}/CTX_{ad}.fp16.npy"
            if os.path.exists(npy):
                print(f"  ↷ ATLA {ad}")
                continue
            _, S = G.satirlar(kol, bas)
            met = [r[G.PENCERE] for r in S]
            t0 = time.time()
            X = kodla_ctx(tok, mdl, met, okur, a.dev)
            np.save(npy, X.astype(np.float16))
            MF["hucreler"][ad] = dict(okur=okur, merdiven_kol=kol, basamak=bas,
                                      n=int(X.shape[0]), sekil=list(X.shape),
                                      sha256=sha(npy),
                                      bayt=int(os.path.getsize(npy)),
                                      saniye=round(time.time() - t0, 1))
            payda(f"arsiv_{ad}", n_satir=int(X.shape[0]), n_katman=int(X.shape[1]),
                  bekle={"n_satir": len(met), "n_katman": int(cfg.num_hidden_layers)})
            print(f"  → {ad} {list(X.shape)} · {os.path.getsize(npy)/2**20:.1f} MB · "
                  f"sha {MF['hucreler'][ad]['sha256'][:16]} · "
                  f"{MF['hucreler'][ad]['saniye']}s", flush=True)
            del X
        del mdl
        torch.cuda.empty_cache()
    dosyalar = sorted(glob.glob(f"{OUT}/*.npy"))
    bayt = sum(os.path.getsize(p) for p in dosyalar)
    st = os.statvfs(OUT)
    MF["disk"] = dict(n_dosya=len(dosyalar), toplam_bayt=int(bayt),
                      toplam_GB=round(bayt / 2**30, 3),
                      bos_GB=round(st.f_bavail * st.f_frsize / 2**30, 1),
                      kok=OUT, silme="YOK")
    payda("arsiv_head", n_hucre=len(MF["hucreler"]), n_okur=len(OKURLAR),
          n_dosya=len(dosyalar), bekle={"n_hucre": 50, "n_okur": 2})
    json.dump(MF, open(mp, "w"), ensure_ascii=False, indent=1)
    print(f"\n★ DISK: {len(dosyalar)} dosya · {bayt/2**30:.2f} GB · "
          f"bos {MF['disk']['bos_GB']:.0f} GB · SILME YOK")
    print(f"→ {mp}")


if __name__ == "__main__":
    main()
