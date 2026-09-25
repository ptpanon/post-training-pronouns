#!/usr/bin/env python
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C = __DNH_DATA__ + "/unreleased/cache"
KIALO = __DNH_DATA__ + "/unreleased/dataframe.csv"
OUT = "<scratch>"
MODEL = "cartgr/embeddings-for-preferences-st5-xl"
HELDOUT = {"gun-control", "vegan", "palestine", "harry-potter"}
rng = np.random.default_rng(0)


def l2(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def load_kialo():
    tr, ev = [], []
    with open(KIALO, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            s = r["Stance"].strip().lower(); p = r["Parent_claim"].strip(); c = r["Claim"].strip()
            if s in ("pro", "con") and p and c:
                rec = (p[:1200], c[:1200], 0 if s == "con" else 1, r["Debate_name"])
                (ev if r["Debate_name"] in HELDOUT else tr).append(rec)
    return tr, ev


def emb(model, texts, dev):
    return np.asarray(model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True, device=dev), np.float32)


def kialo_eval(model, tr, ev, dev, tag):
    trp = emb(model, [x[0] for x in tr], dev); trc = emb(model, [x[1] for x in tr], dev)
    evp = emb(model, [x[0] for x in ev], dev); evc = emb(model, [x[1] for x in ev], dev)
    ytr = np.array([1 - x[2] for x in tr]); yev = np.array([1 - x[2] for x in ev])
    ecos_ev = (evp * evc).sum(1)
    auc_cos = roc_auc_score(yev, -ecos_ev)
    Xtr = np.column_stack([trp, trc, np.abs(trp - trc)]); Xev = np.column_stack([evp, evc, np.abs(evp - evc)])
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, ytr)
    psup = lr.predict_proba(Xev)[:, 1]; auc_sup = roc_auc_score(yev, psup)
    q = np.quantile(ecos_ev, 0.75); mk = ecos_ev >= q
    auc_sup_q4 = roc_auc_score(yev[mk], psup[mk]) if len(set(yev[mk])) == 2 else float("nan")
    from collections import defaultdict
    byp = defaultdict(list)
    for i, x in enumerate(ev): byp[x[0][:80]].append(i)
    conc = tot = 0
    for p, ii in byp.items():
        pos = [i for i in ii if yev[i] == 1]; neg = [i for i in ii if yev[i] == 0]
        for a in pos:
            for b in neg:
                tot += 1; conc += (ecos_ev[a] < ecos_ev[b])
    wp = conc / tot if tot else float("nan")
    print(f"  [{tag:12s}] Kialo held-out: ear_cos AUC={auc_cos:.3f}  ear_sup AUC={auc_sup:.3f}  "
          f"(Q4 {auc_sup_q4:.3f})  within-parent={wp:.3f} (n_wp={tot})", flush=True)
    return dict(cos=auc_cos, sup=auc_sup, q4=auc_sup_q4, wp=wp)


def reddit_eval(model, dev, tag):
    import glob, html
    pairs = json.load(open(f"{C}/bs64_haiku_pairs.json"))
    lab = {}
    for f in sorted(glob.glob(f"{C}/bs64_haiku_out_*.json")):
        for e in json.load(open(f)): lab[int(e["idx"])] = e["label"]
    par, chi, yo = [], [], []
    for k, pr in enumerate(pairs):
        L = lab.get(int(pr["idx"]))
        if L in ("OPPOSE", "AGREE"):
            par.append(html.unescape(pr["parent"])); chi.append(html.unescape(pr["reply"])); yo.append(1 if L == "OPPOSE" else 0)
    yo = np.array(yo)
    ep = emb(model, par, dev); ec = emb(model, chi, dev)
    cos = (l2(ep) * l2(ec)).sum(1); auc_cos = roc_auc_score(yo, -cos)
    print(f"  [{tag:12s}] reddit transfer (Haiku labels, n={len(yo)}, oppose-rate {yo.mean():.2f}): ear_cos AUC={auc_cos:.3f}", flush=True)
    return dict(cos=auc_cos)


def main():
    t0 = time.time()
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from datasets import Dataset
    dev = "cuda:0"
    tr, ev = load_kialo()
    print(f"Kialo: train {len(tr)} pairs, held-out {len(ev)} pairs ({HELDOUT}) ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== STOCK ear baseline ===", flush=True)
    stock = SentenceTransformer(MODEL, device=dev).to(torch.bfloat16)
    s_k = kialo_eval(stock, tr, ev, dev, "stock")
    s_r = reddit_eval(stock, dev, "stock")
    del stock; torch.cuda.empty_cache()

    print(f"\n=== FINE-TUNE (LoRA r=16 on T5 q/v, lr 1e-4, 3 epochs, bf16, grad-ckpt) ({time.time()-t0:.0f}s) ===", flush=True)
    from peft import LoraConfig
    model = SentenceTransformer(MODEL, device=dev)
    model.add_adapter(LoraConfig(task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32,
                                 target_modules=["q", "v"], lora_dropout=0.05))
    ds = Dataset.from_dict({"sentence1": [x[0] for x in tr], "sentence2": [x[1] for x in tr],
                            "label": [float(x[2]) for x in tr]})
    loss = losses.CosineSimilarityLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=OUT, num_train_epochs=3, per_device_train_batch_size=16, learning_rate=1e-4,
        warmup_ratio=0.1, bf16=True, gradient_checkpointing=True, logging_steps=50, save_strategy="no",
        report_to=[], dataloader_drop_last=False)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    print(f"  trained ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== FINE-TUNED ear ===", flush=True)
    model = model.to(torch.bfloat16)
    f_k = kialo_eval(model, tr, ev, dev, "ear2")
    f_r = reddit_eval(model, dev, "ear2")

    print("\n================ VERDICT (Ear-2.0, prereg_bs95) ================", flush=True)
    print(f"  Kialo held-out ear_sup: stock {s_k['sup']:.3f} -> ear2 {f_k['sup']:.3f}  (Δ {f_k['sup']-s_k['sup']:+.3f})", flush=True)
    print(f"  Kialo held-out ear_cos: stock {s_k['cos']:.3f} -> ear2 {f_k['cos']:.3f}  (Δ {f_k['cos']-s_k['cos']:+.3f})", flush=True)
    print(f"  Kialo Q4 ear_sup:       stock {s_k['q4']:.3f} -> ear2 {f_k['q4']:.3f}  (Δ {f_k['q4']-s_k['q4']:+.3f})", flush=True)
    print(f"  Kialo within-parent:    stock {s_k['wp']:.3f} -> ear2 {f_k['wp']:.3f}  (Δ {f_k['wp']-s_k['wp']:+.3f})", flush=True)
    print(f"  reddit transfer ear_cos:stock {s_r['cos']:.3f} -> ear2 {f_r['cos']:.3f}  (Δ {f_r['cos']-s_r['cos']:+.3f})  [cross-family FORGETTING check]", flush=True)
    dk = f_k['sup'] - s_k['sup']; dr = f_r['cos'] - s_r['cos']
    print(f"\n  RULE: bank iff Kialo Δ>=+0.02 AND reddit Δ>=-0.02 (no forgetting). "
          f"Kialo Δ={dk:+.3f}, reddit Δ={dr:+.3f} => "
          f"{'BANK (Ear-2.0 works for this recipe/labels)' if dk>=0.02 and dr>=-0.02 else 'DEFLATE/NULL — see numbers'}", flush=True)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
