import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, glob, time, argparse, hashlib
os.environ.setdefault("HF_HOME", __DNH_ROOT__ + "/.hf")

ROOT = __DNH_ROOT__ + ""
DIS = f"{ROOT}/unreleased/final_dataset"
OUT = __DNH_DATA__ + "/kt_layers"
MODEL = "Qwen/Qwen3-Embedding-4B"
EMBED_URL = "http://172.16.180.28:8040/v1/embeddings"
BOLMELER = ("train", "dev")
KES = 900
DOKUNUS_DEFTERI = f"{ROOT}/unreleased/TEST_DOKUNUS_DEFTERI.md"


class KapsamHatasi(AssertionError):
    pass


def _dokunus_yaz(split, prereg, cagiran, n):
    import datetime
    yeni = not os.path.exists(DOKUNUS_DEFTERI)
    with open(DOKUNUS_DEFTERI, "a", encoding="utf-8") as fh:
        if yeni:
            fh.write(""
                     ""
                     ""
                     ""
                     ""
                     "")
        fh.write(f"| {datetime.datetime.now().isoformat(timespec='seconds')} | {split} | "
                 f"`{prereg}` | `{cagiran}` | {n} |\n")


def load_split(split, prereg=None):
    if split not in BOLMELER:
        if not (split == "test" and isinstance(prereg, str) and prereg.strip()):
            raise SystemExit(f"{split} {BOLMELER}"
                             f"")
    rows = []
    for f in sorted(glob.glob(f"{DIS}/{split}/*.json")):
        d = json.load(open(f)); revs = d["review_sentences"]
        fid = d["metadata"].get("forum_id")
        dosya = os.path.basename(f)
        for rs in d["rebuttal_sentences"]:
            st = rs.get("rebuttal_stance"); al = rs.get("alignment")
            if st not in ("dispute", "concur") or not al or al[1] is None:
                continue
            tgt = al[1]; idxs = tgt if isinstance(tgt, list) else [tgt]
            rtext = " ".join(revs[i]["text"] for i in idxs
                             if isinstance(i, int) and 0 <= i < len(revs))
            if not rtext or not rs["text"].strip():
                continue
            rows.append(dict(ebeveyn=rtext[:KES], cevap=rs["text"][:KES],
                             etiket=1 if st == "dispute" else 0, forum_id=fid, dosya=dosya,
                             cumle_idx=rs.get("sentence_index")))
    if not rows:
        raise KapsamHatasi(f"[kt·{split}] KAPSAM SIFIR — bu gecis degil HATA")
    if split == "test":
        import traceback
        yig = [x for x in traceback.extract_stack()[:-1] if "kt_kodlama" not in x.filename]
        _dokunus_yaz(split, prereg, os.path.basename(yig[-1].filename) if yig else "?", len(rows))
    return rows


def sunucu_gomu(texts):
    import urllib.request
    out = []
    for i in range(0, len(texts), 32):
        ch = texts[i:i + 32]
        body = json.dumps({"model": MODEL, "input": [t if t.strip() else " " for t in ch]}).encode()
        req = urllib.request.Request(EMBED_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())["data"]
        out += [e["embedding"] for e in sorted(d, key=lambda x: x["index"])]
    return out


def kodla(model, tok, metinler, eos, bs, maxlen, dev, katman_adim=1):
    import torch, numpy as np
    L = model.config.num_hidden_layers + 1
    kat = list(range(0, L, katman_adim))
    D = model.config.hidden_size
    ST = np.zeros((len(metinler), len(kat), D), dtype=np.float16)
    OR = np.zeros((len(metinler), len(kat), D), dtype=np.float16)
    uzun = []
    for i in range(0, len(metinler), bs):
        ch = [t if t.strip() else " " for t in metinler[i:i + bs]]
        if eos:
            ch = [t + tok.eos_token for t in ch]
        b = tok(ch, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
        uzun += b["attention_mask"].sum(1).tolist()
        with torch.no_grad():
            o = model(**b, output_hidden_states=True)
        m = b["attention_mask"]
        son = m.sum(1) - 1
        for kk, li in enumerate(kat):
            h = o.hidden_states[li]
            v = h[torch.arange(h.shape[0], device=dev), son]
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            ST[i:i + len(ch), kk] = v.cpu().numpy().astype(np.float16)
            mm = m.unsqueeze(-1).to(h.dtype)
            a = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
            a = torch.nn.functional.normalize(a.float(), dim=-1)
            OR[i:i + len(ch), kk] = a.cpu().numpy().astype(np.float16)
        del o
    return ST, OR, uzun, kat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogrula", type=int, default=0)
    ap.add_argument("--kos", action="store_true")
    ap.add_argument("--eos", type=int, default=1)
    ap.add_argument("--gpu", default="1"); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--katman-adim", type=int, default=1)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch, numpy as np
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time(); dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    L = model.config.num_hidden_layers + 1
    print(f"[kt] {MODEL} · {L} gizli durum · d={model.config.hidden_size} · "
          f"eos_token={tok.eos_token!r} · pad={tok.pad_token!r} · padding_side={tok.padding_side}",
          flush=True)

    if a.dogrula:
        rows = load_split("train")[:a.dogrula]
        metin = [r["ebeveyn"] for r in rows]
        print(f"\n══ DOGRULAMA · {len(metin)} metin ══", flush=True)
        srv = np.asarray(sunucu_gomu(metin), dtype=np.float32)
        srv /= np.linalg.norm(srv, axis=1, keepdims=True)
        R = {}
        for e in (0, 1):
            ST, OR, uz, kat = kodla(model, tok, metin, e, a.bs, a.maxlen, dev)
            son = ST[:, -1].astype(np.float32)
            cs = (son * srv).sum(1)
            co = (OR[:, -1].astype(np.float32) * srv).sum(1)
            print(f"  EOS={e}: son-token↔sunucu cos ort {cs.mean():.6f} min {cs.min():.6f} | "
                  f"ortalama-havuz↔sunucu cos ort {co.mean():.6f}")
            R[f"eos_{e}"] = dict(son_token_cos_ort=float(cs.mean()), son_token_cos_min=float(cs.min()),
                                 ortalama_cos_ort=float(co.mean()),
                                 token_uzunluk=dict(ort=float(np.mean(uz)), medyan=float(np.median(uz)),
                                                    maks=int(max(uz)),
                                                    kesmeye_carpan=int(sum(1 for u in uz if u >= a.maxlen)),
                                                    n=len(uz)))
        best = max((0, 1), key=lambda e: R[f"eos_{e}"]["son_token_cos_ort"])
        u = R[f"eos_{best}"]["token_uzunluk"]
        print(f"\n  ⇒ SEMA: EOS={best} (cos {R[f'eos_{best}']['son_token_cos_ort']:.6f})")
        print(f"  ⇒ KESME: {KES} karakter kesmeden sonra token ort {u['ort']:.1f} · "
              f"medyan {u['medyan']:.0f} · maks {u['maks']} · max_length={a.maxlen}'e carpan "
              f"**{u['kesmeye_carpan']}/{u['n']}**")
        R["secilen_eos"] = best; R["maxlen"] = a.maxlen; R["kes_karakter"] = KES
        os.makedirs(OUT, exist_ok=True)
        json.dump(R, open(f"{OUT}/dogrulama.json", "w"), ensure_ascii=False, indent=1)
        print(f"  -> {OUT}/dogrulama.json · hicbir temsil YAZILMADI ({(time.time()-t0)/60:.1f} dk)")
        return

    if not a.kos:
        raise SystemExit("[kt] --dogrula ya da --kos gerekli.")
    os.makedirs(OUT, exist_ok=True)
    nkat = len(range(0, L, a.katman_adim))
    for split in BOLMELER:
        rows = load_split(split)
        tahmin = len(rows) * 2 * nkat * model.config.hidden_size * 2 * 2 / 1e9
        print(f"\n[kt·{split}] {len(rows)} cift · {nkat} katman · "
              f"**HACIM TAHMINI {tahmin:.2f} GB** (2 taraf × 2 havuzlama, fp16)", flush=True)
        mfp = f"{OUT}/{split}__meta.json"
        for taraf in ("ebeveyn", "cevap"):
            hedef = {h: f"{OUT}/{split}__{taraf}__{h}.fp16.npy" for h in ("sontoken", "ortalama")}
            if all(os.path.exists(p) for p in hedef.values()) and os.path.exists(mfp):
                V = np.load(hedef["sontoken"], mmap_mode="r")
                if V.shape[0] == len(rows):
                    print(f"{split} {taraf} {V.shape}")
                    continue
            ST, OR, uz, kat = kodla(model, tok, [r[taraf] for r in rows], a.eos, a.bs,
                                    a.maxlen, dev, a.katman_adim)
            for h, arr in (("sontoken", ST), ("ortalama", OR)):
                tmp = hedef[h] + ".tmp"
                with open(tmp, "wb") as fh:
                    np.save(fh, arr)
                os.replace(tmp, hedef[h])
            atlanan = 0
            print(f"[kt·{split}·{taraf}] {ST.shape} YAZILDI · "
                  f"token ort {np.mean(uz):.1f} maks {max(uz)} · kesmeye carpan "
                  f"{sum(1 for u in uz if u >= a.maxlen)} · {(time.time()-t0)/60:.1f} dk",
                  flush=True)
            print(f"  └ KAPSAM [{split}·{taraf}]: yazilan {ST.shape[0]} + atlanan {atlanan} "
                  f"= {ST.shape[0]+atlanan} (beklenen {len(rows)})")
            if ST.shape[0] + atlanan != len(rows):
                raise KapsamHatasi(f"[{split}·{taraf}] defter tutmadi")
        meta = dict(model=MODEL, split=split, n=len(rows), katmanlar=kat, eos=a.eos,
                    maxlen=a.maxlen, kes_karakter=KES, havuzlamalar=["sontoken", "ortalama"],
                    kanonik_yukleyici="scripts/bs111_disapere.py::load_split",
                    test_bolmesi="ACILMADI",
                    satir=[{k: r[k] for k in ("etiket", "forum_id", "dosya", "cumle_idx")}
                           for r in rows])
        json.dump(meta, open(mfp + ".tmp", "w"), ensure_ascii=False)
        os.replace(mfp + ".tmp", mfp)
        print(f"[kt·{split}] meta yazildi ({len(rows)} satir)", flush=True)

    print("\n══ BAGIMSIZ DISK SAYIMI ══")
    for split in BOLMELER:
        m = json.load(open(f"{OUT}/{split}__meta.json"))
        for taraf in ("ebeveyn", "cevap"):
            for h in ("sontoken", "ortalama"):
                V = np.load(f"{OUT}/{split}__{taraf}__{h}.fp16.npy", mmap_mode="r")
                ok = V.shape == (m["n"], len(m["katmanlar"]), model.config.hidden_size)
                print(f"   {split:5s} {taraf:8s} {h:9s} {str(V.shape):22s} "
                      f"{'✓' if ok else '✗ BEKLENEN ' + str((m['n'], len(m['katmanlar']), 2560))}")
                if not ok:
                    raise KapsamHatasi("disk sayimi tutmadi")
    print(f"\n[kt] BITTI {(time.time()-t0)/60:.1f} dk -> {OUT}")


if __name__ == "__main__":
    main()
