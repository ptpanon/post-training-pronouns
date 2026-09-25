#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, os, re, subprocess, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
RULE = f"{ROOT}/preregistration/RULE_V99_C2B_UYUM_2026-09-21.md"
PAKET = f"{ROOT}/INSAN_PAKETI_V96_C2_B_100.md"
DIS = __DNH_DATA__ + "/v96_c2"
CIKTI = f"{ROOT}/results/v99_c2b_uyum_2026-09-21.json"
SINIF = ["ADDRESSED", "GENERIC", "UNDETERMINED"]
KISA = {"A": "ADDRESSED", "ADDRESSED": "ADDRESSED", "G": "GENERIC", "GENERIC": "GENERIC",
        "U": "UNDETERMINED", "UNDETERMINED": "UNDETERMINED"}
YARGIC = {"ADDRESSED": "ADDRESSED", "GENERIC": "GENERIC", "OTHER": "UNDETERMINED"}
B_CEKIM, SEED = 1000, 20260921


def rule_kapisi():
    h = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "-1", "--", os.path.relpath(RULE, ROOT)],
                       capture_output=True, text=True).stdout.strip()
    if not h:
        raise SystemExit("")
    b = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    if b not in open(RULE, encoding="utf-8").read():
        raise SystemExit(f"{b[:16]}")
    return h, b[:16]


def ayristir():
    cev, cift, sema_disi = {}, [], []
    for l in io.open(PAKET, encoding="utf-8"):
        m = re.match(r"^`h=(\d+) cevap=(\S+)`", l.strip())
        if not m:
            continue
        h, c = int(m.group(1)), m.group(2).strip().upper()
        if c not in KISA:
            sema_disi.append((h, c)); continue
        if h in cev:
            cift.append(h)
        cev[h] = KISA[c]
    return cev, cift, sema_disi


def kappa(i, y):
    po = float(np.mean(i == y))
    pe = float(sum(np.mean(i == s) * np.mean(y == s) for s in range(3)))
    return po, pe, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    h, b = rule_kapisi()
    cev, cift, sema_disi = ayristir()
    anah = {x["hucre"]: x["kimlik"] for x in map(json.loads, io.open(f"{DIS}/anahtar/ANAHTAR_V96_C2_B.jsonl", encoding="utf-8"))}
    YB = {x["kimlik"]: x["etiket"] for x in map(json.loads, io.open(f"{DIS}/B_yargi.jsonl", encoding="utf-8"))}
    OR = {x["kimlik"]: x for x in map(json.loads, io.open(f"{DIS}/B_orneklem.jsonl", encoding="utf-8"))}
    eksik = sorted(set(anah) - set(cev))
    satir = []
    for hc in sorted(cev):
        k = anah.get(hc)
        if k is None or k not in YB or YB[k] not in YARGIC:
            eksik.append(hc); continue
        satir.append(dict(hucre=hc, insan=cev[hc], yargic=YARGIC[YB[k]], aile=OR[k]["aile"], bacak=OR[k]["bacak"]))
    N = len(satir)
    print(f"  [PAYDA] v99_c2b: n_hucre_anahtar={len(anah)} · n_ayristirilan={len(cev)} · n_eslenen={N} · "
          f"red_eksik={len(eksik)} · red_cift={len(cift)} · red_sema_disi={len(sema_disi)} · [bekle] n_eslenen=100", flush=True)
    ix = {s: j for j, s in enumerate(SINIF)}
    I = np.array([ix[r["insan"]] for r in satir]); Y = np.array([ix[r["yargic"]] for r in satir])
    X = int((I == Y).sum()); po, pe, ka = kappa(I, Y)
    M = np.zeros((3, 3), int)
    for a, c in zip(I, Y):
        M[a, c] += 1
    sinif = {s: dict(insan_n=int(M[j].sum()), yargic_n=int(M[:, j].sum()), ortak=int(M[j, j]),
                     insan_sinifi_yargicta=float(M[j, j] / max(M[j].sum(), 1)),
                     yargic_sinifi_insanda=float(M[j, j] / max(M[:, j].sum(), 1))) for j, s in enumerate(SINIF)}
    aileler = sorted({r["aile"] for r in satir}); A = np.array([aileler.index(r["aile"]) for r in satir])
    rng = np.random.default_rng(SEED); bu, bk = [], []
    for _ in range(B_CEKIM):
        sec = rng.integers(0, len(aileler), len(aileler))
        idx = np.concatenate([np.where(A == a)[0] for a in sec])
        p, _, k = kappa(I[idx], Y[idx]); bu.append(p); bk.append(k)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    katman = {}
    for bc in sorted({r["bacak"] for r in satir}):
        j = np.array([r["bacak"] == bc for r in satir])
        katman[bc] = dict(n=int(j.sum()), uyum=int((I[j] == Y[j]).sum()))
    if N < 100 or cift or sema_disi:
        verdict = "OKUNAMADI"
    else:
        verdict = "GIRER" if X / N >= 0.80 else "GIRMEZ"
    kil = 0.78 <= (X / N if N else 0) <= 0.82
    ayrisan = [dict(hucre=r["hucre"], insan=r["insan"], yargic=r["yargic"], aile=r["aile"], bacak=r["bacak"])
               for r in satir if r["insan"] != r["yargic"]]
    out = dict(rule=h, alet=b, damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               paket_sha256=hashlib.sha256(open(PAKET, "rb").read()).hexdigest(),
               N=N, X=X, ham_uyum=X / N if N else None, ham_uyum_ci=ci(bu), kappa=ka, kappa_ci=ci(bk), p_e=pe,
               matris=dict(satir_insan=SINIF, sutun_yargic=SINIF, M=M.tolist()), sinif=sinif, katman=katman,
               eksik=eksik, cift=cift, sema_disi=sema_disi, verdict=verdict, kil_payi=kil, ayrisan=ayrisan,
               n_aile=len(aileler), bootstrap=dict(cekim=B_CEKIM, seed=SEED, kume="aile"))
    json.dump(out, io.open(CIKTI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ham uyum {X}/{N} = {X/N:.2f} [{ci(bu)[0]:.2f}, {ci(bu)[1]:.2f}] · κ {ka:.3f} [{ci(bk)[0]:.3f}, {ci(bk)[1]:.3f}] · "
          f"p_e {pe:.3f} ⇒ esik 0,80 ⇒ VERDICT {verdict}{' · KIL PAYI ⇒ baraj dosyasi' if kil else ''}", flush=True)
    print("  matris (satir insan, sütun yargic):", M.tolist(), "· katman", katman, flush=True)
    print(f"✓ {CIKTI}")
    return 0 if verdict != "OKUNAMADI" else 3


if __name__ == "__main__":
    sys.exit(main())
