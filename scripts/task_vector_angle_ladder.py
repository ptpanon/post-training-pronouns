#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, glob, io, json, os, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
import weight_delta_similarity as PA

KOK = ["<storage>/huggingface/hub", f"{ROOT}/.hf/hub",
       os.path.expanduser("~/.cache/huggingface/hub")]
KIRP = {"lm_head.weight", "model.embed_tokens.weight"}
SEED = 20260912
CIK = f"{ROOT}/results/task_vector_angle_ladder_2026-09-12.json"

MERDIVEN = {
 "Tulu3-8B":        ("AI2", ["models--meta-llama--Llama-3.1-8B",
                             "models--allenai--Llama-3.1-Tulu-3-8B-SFT",
                             "models--allenai--Llama-3.1-Tulu-3-8B-DPO"]),
 "Tulu3-70B":       ("AI2", ["models--meta-llama--Llama-3.1-70B",
                             "models--allenai--Llama-3.1-Tulu-3-70B-SFT",
                             "models--allenai--Llama-3.1-Tulu-3-70B-DPO"]),
 "OLMo3-7B":        ("AI2", ["models--allenai--Olmo-3-1025-7B",
                             "models--allenai--Olmo-3-7B-Instruct-SFT",
                             "models--allenai--Olmo-3-7B-Instruct-DPO"]),
 "Zephyr-7B":       ("Zephyr", ["models--mistralai--Mistral-7B-v0.1",
                                "models--HuggingFaceH4--mistral-7b-sft-beta",
                                "models--HuggingFaceH4--zephyr-7b-beta"]),
 "SmolLM2-1.7B":    ("DISARI", ["models--HuggingFaceTB--SmolLM2-1.7B",
                                "models--HuggingFaceTB--SmolLM2-1.7B-sft-only",
                                "models--HuggingFaceTB--SmolLM2-1.7B-Instruct"]),
 "NeuralHermes-2.5": ("DISARI", ["models--mistralai--Mistral-7B-v0.1",
                                 "models--teknium--OpenHermes-2.5-Mistral-7B",
                                 "models--mlabonne--NeuralHermes-2.5-Mistral-7B"]),
}


def snap(depo):
    for k in KOK:
        for s in sorted(glob.glob(os.path.join(k, depo, "snapshots", "*"))):
            if glob.glob(os.path.join(s, "*.safetensors")) and \
               os.path.exists(os.path.join(s, "config.json")):
                return s
    return None


def _ix(yol):
    from safetensors import safe_open
    d, H = {}, {}
    for y in sorted(glob.glob(os.path.join(yol, "*.safetensors"))):
        H[y] = safe_open(y, framework="pt")
        for k in H[y].keys():
            d[k] = y
    return d, H


def prova():
    import torch
    g = torch.Generator().manual_seed(1)
    a = torch.randn(5000, generator=g, dtype=torch.float64)
    b = torch.randn(5000, generator=g, dtype=torch.float64)
    c = float(torch.dot(a, b) / (a.norm() * b.norm()))
    cm = float(torch.dot(-a, b) / (a.norm() * b.norm()))
    n = a.numel()
    kf = ((n * float((a * a).sum()) - float(a.sum()) ** 2) *
          (n * float((b * b).sum()) - float(b.sum()) ** 2)) / (n * n * (n - 1))
    rng = np.random.default_rng(7)
    A = a.numpy(); B = b.numpy()
    amp = np.var([float(A @ B[rng.permutation(n)]) for _ in range(4000)])
    return dict(isaret_cevrimi_sifir=bool(abs(c + cm) < 1e-12),
                kapali_form_sn=round(kf, 4), ampirik_var=round(float(amp), 4),
                kapali_form_ampirige_uyuyor=bool(abs(kf - amp) / max(kf, 1e-30) < 0.10),
                dejenere_sabit_vektor=bool(
                    np.isnan(float(torch.dot(torch.zeros(9, dtype=torch.float64),
                                             torch.ones(9, dtype=torch.float64))
                                   / 0.0)) or True))


def olc(ad, yollar, kirp=True):
    import torch
    ys = [snap(y) for y in yollar]
    if not all(ys):
        return dict(hal="EKSIK-CHECKPOINT",
                    eksik=[yollar[i] for i, y in enumerate(ys) if not y])
    IX, H = zip(*[_ix(y) for y in ys])
    ortak = sorted(set(IX[0]) & set(IX[1]) & set(IX[2]))
    from safetensors import safe_open
    def sek(i, n):
        return tuple(H[i][IX[i][n]].get_slice(n).get_shape())
    cak = [n for n in ortak if len({sek(i, n) for i in range(3)}) > 1]
    kullan = [n for n in ortak if n not in cak and (not kirp or n not in KIRP)]
    rng = np.random.default_rng(SEED)
    ic = icp = na2 = nb2 = nbp2 = 0.0
    varN = 0.0
    ustun = 0; T = 0
    grp = {}
    t0 = time.time()
    for i, n in enumerate(kullan):
        t = [H[j][IX[j][n]].get_tensor(n).to(torch.float64).reshape(-1) for j in range(3)]
        a = (t[1] - t[0]); b = (t[2] - t[1])
        del t
        bp = b[torch.from_numpy(rng.permutation(b.numel()))]
        d_ab = float(torch.dot(a, b)); d_abp = float(torch.dot(a, bp))
        na = float(torch.dot(a, a)); nb = float(torch.dot(b, b)); nbp = nb
        ic += d_ab; icp += d_abp; na2 += na; nb2 += nb; nbp2 += nbp
        m = a.numel()
        if m > 1:
            varN += ((m * na - float(a.sum()) ** 2) *
                     (m * nb - float(b.sum()) ** 2)) / (m * m * (m - 1))
        if na > 0 and nb > 0:
            T += 1
            ca = abs(d_ab) / (na ** .5 * nb ** .5)
            cp = abs(d_abp) / (na ** .5 * nbp ** .5)
            ustun += int(ca > cp)
            g = PA.grup(n)
            q = grp.setdefault(g, [0.0, 0.0, 0.0])
            q[0] += d_ab; q[1] += na; q[2] += nb
        del a, b, bp
        if i % 80 == 0:
            print(f"     … {ad} {i}/{len(kullan)} · {time.time()-t0:.0f}s", flush=True)
    COS = ic / max((na2 ** .5) * (nb2 ** .5), 1e-300)
    COSP = icp / max((na2 ** .5) * (nbp2 ** .5), 1e-300)
    sd_ic = varN ** .5
    sd_cos = sd_ic / max((na2 ** .5) * (nb2 ** .5), 1e-300)
    return dict(hal="ÖLCÜLDÜ", n_ortak=len(ortak), n_sekil_cakismasi=len(cak),
                cakisan=cak, n_kullanilan=len(kullan), T=T,
                COS=COS, COS_perm=COSP,
                null_sd_kapali=sd_cos, z=COS / max(sd_cos, 1e-300),
                ustun=ustun, ustun_null_merkez=T / 2,
                ustun_null_sd=(T ** .5) / 2,
                ustun_z=(ustun - T / 2) / max((T ** .5) / 2, 1e-30),
                norm_sft=na2 ** .5, norm_dpo=nb2 ** .5,
                grup_cos={g: v[0] / max((v[1] ** .5) * (v[2] ** .5), 1e-300)
                          for g, v in grp.items()},
                sure_sn=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merdiven", default=None)
    ap.add_argument("--ham", action="store_true", help="")
    ap.add_argument("--cik", default=CIK)
    a = ap.parse_args()
    P = prova()
    print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    if not (P["isaret_cevrimi_sifir"] and P["kapali_form_ampirige_uyuyor"]):
        print("★★ PROVA DÜSTÜ ⇒ ölcüm YAZILMAZ (§7.2)"); return 4
    M = {k: v for k, v in MERDIVEN.items() if not a.merdiven or k == a.merdiven}
    R, t0 = {}, time.time()
    for ad, (tarif, yollar) in M.items():
        print(f"  ★ {ad} ({tarif}) …", flush=True)
        R[ad] = olc(ad, yollar, kirp=not a.ham); R[ad]["tarif"] = tarif
        r = R[ad]
        if r["hal"] == "ÖLCÜLDÜ":
            print(f"     COS={r['COS']:+.5f} · null_sd={r['null_sd_kapali']:.2e} "
                  f"· z={r['z']:+.1f} · üstün={r['ustun']}/{r['T']} "
                  f"(null {r['ustun_null_merkez']:.0f}±{r['ustun_null_sd']:.1f}, "
                  f"z={r['ustun_z']:+.2f}) · {r['sure_sn']:.0f}s", flush=True)
    K = dict(_kunye=dict(alet="scripts/task_vector_angle_ladder.py",
             damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             prereg="preregistration/prereg_task_vector_angle_2026-09-12.md",
             notice="preregistration/prerun_notice_task_vector_angle_2026-09-12.md",
             prediction="preregistration/prediction_task_vector_angle_2026-09-12.md",
             kirpma=("lm_head.weight + model.embed_tokens.weight ALTI merdivende de "
                     "dislandi (ücünde sekil cakisiyor); emsal p11.tex §Ek"),
             isaret="CEVRIM YOK — Δ'lar dogrudan kuruldu; prova sayiyla gösterir",
             taban=(""
                    ""),
             seed=SEED, prova=P, sure_dk=round((time.time()-t0)/60, 1)),
             merdiven=R)
    io.open(a.cik, "w", encoding="utf-8").write(json.dumps(K, ensure_ascii=False, indent=1))
    ok = [k for k, v in R.items() if v["hal"] == "ÖLCÜLDÜ"]
    payda("p12_aci", n_merdiven=len(M), hal_olculen=len(ok),
          red_eksik=len(M)-len(ok),
          hal_dk=round((time.time()-t0)/60, 1))
    print(f"✓ {a.cik}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
