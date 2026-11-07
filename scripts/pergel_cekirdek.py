#!/usr/bin/env python3
import numpy as np
import torch


def foldlar(gruplar, k=5, seed=20260804):
    u = sorted(set(gruplar))
    rng = np.random.default_rng(seed)
    atama = {g: i % k for i, g in enumerate(rng.permutation(len(u)))}
    gid = {g: i for i, g in enumerate(u)}
    return np.array([atama[gid[g]] for g in gruplar])


class Cekirdek:

    def __init__(self, X, fold, dev="cuda:0", std=False, merkez=True):
        self.dev = dev
        self.std = std
        self.merkez = merkez
        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=dev)
        self.fold = torch.as_tensor(fold, dtype=torch.long, device=dev)
        self.n = self.X.shape[0]
        self.foldlar = sorted(set(fold.tolist()))
        self.hz = {}
        self.kirpilan_boyut = 0
        for j in self.foldlar:
            tr = (self.fold != j)
            te = ~tr
            mu = self.X[tr].mean(0) if merkez else torch.zeros_like(self.X[0])
            if std:
                s = self.X[tr].std(dim=0, unbiased=True)
                self.kirpilan_boyut = max(self.kirpilan_boyut, int((s < 1e-6).sum()))
                s = s.clamp_min(1e-6)
            else:
                s = None
            Z = self.X[te] - mu
            if std:
                Z = Z / s
            nz = Z.norm(dim=1).clamp_min(1e-12)
            self.hz[j] = dict(tr=tr, te=te, Z=Z, nz=nz, s=s, m=int(te.sum()))

    def oku_toplu(self, Y):
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.dev)
        K = Y.shape[0]
        A = torch.zeros((K, self.n), device=self.dev)
        for j in self.foldlar:
            h = self.hz[j]
            tr = h["tr"].float()
            W1 = Y * tr
            W0 = (1.0 - Y) * tr
            n1 = W1.sum(1, keepdim=True).clamp_min(1.0)
            n0 = W0.sum(1, keepdim=True).clamp_min(1.0)
            U = (W1 @ self.X) / n1 - (W0 @ self.X) / n0
            if self.std:
                U = U / h["s"]
            U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
            S = (h["Z"] @ U.T) / h["nz"][:, None]
            A[:, h["te"]] = S.T
        return A

    def bosalt(self):
        self.hz.clear()
        del self.X
        torch.cuda.empty_cache()

    def auc_toplu(self, Y, A):
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.dev)
        o = A.argsort(dim=1)
        r = torch.zeros_like(A)
        ar = torch.arange(1, self.n + 1, device=self.dev, dtype=A.dtype).expand_as(A)
        r.scatter_(1, o, ar)
        n1 = Y.sum(1)
        n0 = self.n - n1
        s1 = (r * Y).sum(1)
        paydas = n1 * n0
        auc = (s1 - n1 * (n1 + 1) / 2) / paydas.clamp_min(1.0)
        auc = torch.where(paydas > 0, auc, torch.full_like(auc, float("nan")))
        return auc.cpu().numpy()


class Permutator:

    def __init__(self, y, gruplar, seed=20260804):
        self.y = np.asarray(y)
        g = np.asarray(gruplar)
        u = {v: i for i, v in enumerate(sorted(set(g.tolist())))}
        self.g = np.array([u[v] for v in g])
        self.ordr = np.argsort(self.g, kind="stable")
        self.g_s = self.g[self.ordr].astype(np.float64)
        self.y_s = self.y[self.ordr]
        self.rng = np.random.default_rng(seed)
        self.n = len(self.y)

    def cek(self, K):
        key = self.g_s[None, :] + self.rng.random((K, self.n))
        p = np.argsort(key, axis=1)
        Ys = self.y_s[p]
        Y = np.empty((K, self.n), dtype=np.int8)
        Y[:, self.ordr] = Ys
        return Y

    def karisim(self, K, doz):
        Yp = self.cek(K)
        koru = self.rng.random((K, self.n)) < doz
        return np.where(koru, self.y[None, :], Yp).astype(np.int8)
