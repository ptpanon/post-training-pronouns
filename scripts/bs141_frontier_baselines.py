#!/usr/bin/env python
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", __DNH_ROOT__ + "/.hf")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, cohen_kappa_score

C = __DNH_DATA__ + "/unreleased/cache"
S = "<scratch>"
ST5 = "cartgr/embeddings-for-preferences-st5-xl"
MARKERS = ["but", "however", "although", "though", "actually", "no,", "not ", "n't", "disagree", "wrong", "false",
           "instead", "rather", "except", "unless", "nevertheless", "yet ", "on the contrary", "in fact", "however,"]


def l2(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
def feats(P, Cc): return np.column_stack([P, Cc, np.abs(P - Cc)]).astype(np.float32)


def st5_enc(adapter, texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(ST5, device="cuda:0")
    if adapter:
        try: m.load_adapter(adapter)
        except Exception:
            from peft import PeftModel; m[0].auto_model = PeftModel.from_pretrained(m[0].auto_model, adapter)
    m = m.to(torch.bfloat16)
    E = l2(np.asarray(m.encode([t[:1200] for t in texts], batch_size=160, normalize_embeddings=True,
                               convert_to_numpy=True, device="cuda:0", show_progress_bar=False), np.float32))
    del m; torch.cuda.empty_cache(); return E


def plain_enc(name, texts, prefix=""):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device="cuda:0")
    E = l2(np.asarray(m.encode([prefix + t[:1200] for t in texts], batch_size=128, normalize_embeddings=True,
                               convert_to_numpy=True, device="cuda:0", show_progress_bar=False), np.float32))
    del m; torch.cuda.empty_cache(); return E


def marker_feats(par, rep):
    def mv(t):
        t = " " + str(t).lower() + " "; return [t.count(mk) for mk in MARKERS]
    return np.array([mv(rep[i]) + mv(par[i]) for i in range(len(rep))], np.float32)


def tfidf_svd(par, rep, dim=300):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True)
    X = v.fit_transform([str(par[i]) + " [SEP] " + str(rep[i]) for i in range(len(rep))])
    return TruncatedSVD(min(dim, X.shape[1] - 1), random_state=0).fit_transform(X).astype(np.float32)


def oof(X, y, g):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(min(8, len(set(g)))).split(X, y, g):
        p[te] = LogisticRegression(max_iter=2000, C=0.5).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def cci(y, pa, pb, cl, boot=2000):
    cl = np.asarray(cl); uc = list(set(cl.tolist())); io = {c: np.where(cl == c)[0] for c in uc}
    r = np.random.default_rng(1); d = []
    for _ in range(boot):
        tk = np.concatenate([io[uc[k]] for k in r.integers(0, len(uc), len(uc))])
        if len(set(y[tk].tolist())) < 2: continue
        d.append(roc_auc_score(y[tk], pa[tk]) - roc_auc_score(y[tk], pb[tk]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(tag, y, mk4, grp, preds, order):
    print(f"\n### {tag} — Q4 n={int(mk4.sum())}, G(threads in Q4)={len(set(grp[mk4].tolist()))} ###", flush=True)
    print("  Q4 AUC:  " + "  ".join(f"{n}={roc_auc_score(y[mk4], preds[n][mk4]):.3f}" for n in order), flush=True)
    best_content = max(["stock", "e5", "bge", "content", "tfidf"], key=lambda n: roc_auc_score(y[mk4], preds[n][mk4]) if n in preds else 0)
    for n in order:
        if n == "ear": continue
        d, lo, hi = cci(y[mk4], preds["ear"][mk4], preds[n][mk4], grp[mk4])
        star = "  <== best content baseline" if n == best_content else ""
        print(f"    Δ(ear − {n:8s}) = {d:+.3f} thread-CI[{lo:+.3f},{hi:+.3f}]  {'DISJOINT>0' if lo>0 else 'straddles'}{star}", flush=True)


def main():
    t0 = time.time()
    z = np.load(f"{C}/bs62.npz", allow_pickle=True)
    cid = z["id"].astype(str); Eq = l2(z["Eq"].astype(np.float32)); root = z["root"].astype(str)
    idmap = {cid[i]: i for i in range(len(cid)) if cid[i]}

    labels = json.load(open(f"{C}/bs100_labels.json"))
    rows = [r for r in labels if r["label"] in ("OPPOSE", "AGREE") and r["pid"] in idmap and r["rid"] in idmap]
    pj = np.array([idmap[r["pid"]] for r in rows]); ci = np.array([idmap[r["rid"]] for r in rows]); grp = root[ci]
    y = np.array([1 if r["label"] == "OPPOSE" else 0 for r in rows])
    ccos = np.sum(Eq[pj] * Eq[ci], 1); mk4 = ccos >= np.quantile(ccos, 0.75)
    par = [r["parent"] for r in rows]; rep = [r["reply"] for r in rows]
    print(f"Panel A frontier pairs {len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    enc = {"ear": (st5_enc(f"{S}/ear2_adapter", par), st5_enc(f"{S}/ear2_adapter", rep)),
           "stock": (st5_enc(None, par), st5_enc(None, rep)),
           "e5": (plain_enc("intfloat/e5-large-v2", par, "query: "), plain_enc("intfloat/e5-large-v2", rep, "query: ")),
           "bge": (plain_enc("BAAI/bge-m3", par), plain_enc("BAAI/bge-m3", rep))}
    print(f"  encoded ear/stock/e5/bge ({time.time()-t0:.0f}s)", flush=True)
    predsA = {n: oof(feats(P, Cc), y, grp) for n, (P, Cc) in enc.items()}
    predsA["content"] = oof(feats(Eq[pj], Eq[ci]), y, grp)
    predsA["tfidf"] = oof(tfidf_svd(par, rep), y, grp)
    predsA["markers"] = oof(marker_feats(par, rep), y, grp)
    report("PANEL A (Qwen labels, full frontier)", y, mk4, grp, predsA, ["ear", "stock", "e5", "bge", "content", "tfidf", "markers"])

    pairs = json.load(open(f"{S}/frontier_xfam.json"))
    claude = {r["i"]: r["label"] for r in json.load(open(f"{C}/frontier_xfam_claude.json"))}
    rb = [p for p in pairs if p["pid"] in idmap and p["rid"] in idmap]
    pjb = np.array([idmap[p["pid"]] for p in rb]); cib = np.array([idmap[p["rid"]] for p in rb]); grpb = root[cib]
    ccb = np.array([p["ccos"] for p in rb]); m4b = ccb >= np.quantile(ccb, 0.75)
    parb = [p["parent"] for p in rb]; repb = [p["reply"] for p in rb]
    encB = {"ear": (st5_enc(f"{S}/ear2_adapter", parb), st5_enc(f"{S}/ear2_adapter", repb)),
            "stock": (st5_enc(None, parb), st5_enc(None, repb)),
            "e5": (plain_enc("intfloat/e5-large-v2", parb, "query: "), plain_enc("intfloat/e5-large-v2", repb, "query: ")),
            "bge": (plain_enc("BAAI/bge-m3", parb), plain_enc("BAAI/bge-m3", repb))}
    XB = {n: feats(P, Cc) for n, (P, Cc) in encB.items()}; XB["content"] = feats(Eq[pjb], Eq[cib])
    for fam, getlab in [("Qwen", lambda p: p["qwen"]), ("Claude", lambda p: claude.get(p["i"]))]:
        keep = np.array([i for i, p in enumerate(rb) if getlab(p) in ("OPPOSE", "AGREE")])
        yb = np.array([1 if getlab(rb[i]) == "OPPOSE" else 0 for i in keep]); m4 = m4b[keep]; gk = grpb[keep]
        preds = {n: oof(XB[n][keep], yb, gk) for n in ["ear", "stock", "e5", "bge", "content"]}
        report(f"PANEL B ({fam} labels, 1200-subset)", yb, m4, gk, preds, ["ear", "stock", "e5", "bge", "content"])
    print(f"\ndone bs141 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
