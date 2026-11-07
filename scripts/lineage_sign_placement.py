#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import lineage_permutation as SP

KAYNAK = f"{ROOT}/results/template_filtered_2026-09-07.json"
SOYKART = f"{ROOT}/results/lineage_permutation_2026-09-11.json"
CIK = f"{ROOT}/results/lineage_sign_placement_2026-09-15.json"


def sha16(y):
    return hashlib.sha256(open(y, "rb").read()).hexdigest()[:16]


def oku(ad, sgn, S):
    lab = np.array([S[n] for n in ad]); g = np.unique(lab, return_inverse=True)[1]
    Ig, n = SP._isaret_saf(sgn, g)
    rng = np.random.default_rng(SP.SEED); inull = np.empty(SP.NPERM)
    for i in range(SP.NPERM):
        p = rng.permutation(len(ad)); inull[i] = SP._isaret_saf(sgn, g[p])[0]
    uymayan = []
    for u in np.unique(lab):
        m = (lab == u) & (sgn != 0); k = sgn[m]
        if len(k) == 0:
            continue
        b = 1 if (k > 0).sum() >= (k < 0).sum() else -1
        uymayan += [ad[i] for i in np.where(m)[0] if sgn[i] != b]
    seviye = {u: int((lab == u).sum()) for u in np.unique(lab)}
    return dict(uyusan=int(Ig), n_isaretli=int(n), null_ort=float(inull.mean()),
                frac=float((inull >= Ig).mean()), uymayan=sorted(uymayan), seviye=seviye,
                n_seviye=len(seviye))


def main():
    K = json.load(io.open(KAYNAK, encoding="utf-8"))
    A = [a for a in K["aileler"] if a["hal"] == "ÖLCÜLDÜ"]
    POZ = set(K["sayim"]["sablonlu"]["pozitif"]); NEG = set(K["sayim"]["sablonlu"]["negatif"])
    ad = [a["aile"] for a in A]
    sgn = np.array([1 if n in POZ else (-1 if n in NEG else 0) for n in ad])
    rA = oku(ad, sgn, SP.SOY)
    rB = oku(ad, sgn, {k: ("AI2" if v == "OLMo-3" else v) for k, v in SP.SOY.items()})
    S = json.load(io.open(SOYKART, encoding="utf-8"))
    I = S["isaret"]
    red_esdeger = int(not (rA["uyusan"] == I["uyusan"] and rA["n_isaretli"] == I["n_isaretli"]
                           and abs(rA["null_ort"] - I["null_ort"]) <= 1e-9 and abs(rA["frac"] - I["frac"]) <= 1e-12))
    print(f"{len(ad)} {int((sgn != 0).sum())} {SP.NPERM} {SP.SEED}"
          f"{rA['uyusan']} {rA['n_isaretli']} {rA['frac']:.3f} {rB['uyusan']} {rB['n_isaretli']}"
          f"{rB['frac']:.3f} {red_esdeger}")
    if red_esdeger:
        print("★★ ESDEGERLIK DÜSTÜ — A yerlesimi SOY_PERMUTASYON kartini üretmiyor", rA, I); return 3
    red_rebuttal = int(not (rB["uyusan"] == 12 and rB["n_isaretli"] == 14 and round(rB["frac"], 3) == 0.092
                            and round(rB["null_ort"], 2) == 9.94))
    json.dump(dict(
        damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        SINIF="BETIM · mühürsüz · v75 oturum hesabinin karta alinmasi (Q7), yeni ölcüm degil",
        alet="scripts/lineage_sign_placement.py (lineage_permutation._isaret_saf ITHAL)",
        kaynaklar=[dict(yol="results/template_filtered_2026-09-07.json", sha256_16=sha16(KAYNAK)),
                   dict(yol="results/lineage_permutation_2026-09-11.json", sha256_16=sha16(SOYKART))],
        n_perm=SP.NPERM, seed=SP.SEED, bar=SP.BAR,
        referans="",
        okuma_A=dict(aciklama="OLMo-3 ayri seviye (alti seviye) — kâgidin birincil okumasi", **rA,
                     H=S["H"]["gozlenen"], H_null_ort=S["H"]["null_ort"], H_frac=S["H"]["frac"]),
        okuma_B=dict(aciklama="OLMo-3 AI2'ye katilmis (bes soy, §1'in sayimi)", **rB,
                     H=S["okuma_B"]["H"], H_null_ort=S["okuma_B"]["null_ort"], H_frac=S["okuma_B"]["frac"]),
        red_esdeger_A=red_esdeger, red_rebuttal_B=red_rebuttal),
        io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK} · red_rebuttal_B={red_rebuttal}")
    return 4 if red_rebuttal else 0


if __name__ == "__main__":
    sys.exit(main())
