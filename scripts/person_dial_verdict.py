#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, re, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import depersonalization16_neutral_run as M9
import form_count as FS
from reading_style_object import payda

CIK = os.environ.get("DIAL_VERDICT_CIK", f"{ROOT}/results/DIAL_VERDICT_2026-09-04.json")
DU_NULL_SD_PREREG = 0.0026626139817629182
ERRATA_1 = (""
            ""
            "")
KOL_YUVA = {"KISI_ASAGI": 844, "KISI_YUKARI": 357, "RASTGELE": 114, "KUVVET_ASAGI": 922}
REF_B = "RASTGELE"
KD_DESEN = "KOL_DEGER_DIAL_{kol}_m4_2026-08-30.json"
KASKAD = None
PREREG_ADI = "prereg_person_dial_2026-09-03.md @ 72600717 · gövde 33efcd27…"
TABAN = ("Tulu3-8B", "sft")
TABAN_KOK = ""
M = 4
BAR = 1.1
MDE_B = 1.29
NULL_SD_KUVVET = 0.00104
NB = 1000
PLASEBO_N = 200

_W = re.compile(r"\w+", re.U)


def distinct4(metinler):
    top, benzersiz = 0, set()
    for t in metinler:
        w = _W.findall((t or "").lower())
        for i in range(len(w) - 3):
            benzersiz.add(tuple(w[i:i + 4])); top += 1
    return (len(benzersiz) / top) if top else float("nan")


def _distinct4_provasi():
    tek = ["the cat sat on the mat"] * 50
    cesit = [f"alpha beta gamma delta {i} epsilon zeta" for i in range(50)]
    a, b = distinct4(tek), distinct4(cesit)
    ok = a < 0.10 and b > 0.60 and np.isnan(distinct4([""]))
    print(f"   distinct4 provasi: tekrar={a:.4f} (<0,10) · cesitli={b:.4f} (>0,60) · "
          f"bos=NaN ⇒ {'GECTI' if ok else 'DÜSTÜ'}")
    return ok


def olcu(V, C):
    t = MK.olc_toplam(V)
    return dict(M1=t["M1"], KUVVET=t["KUVVET"],
                SAHIS1=M9.sahis1_1k(V),
                JETON_CUMLE=float(V[:, 0].sum() / max(C.sum(), 1)))


def delta_kol(KOL, Vb, Cb, kolad, ref):
    out = {k: [] for k in ("M1", "KUVVET", "SAHIS1", "JETON_CUMLE")}
    for i in range(M):
        A = KOL[kolad]["V"][i]; CA = KOL[kolad]["C"][i]
        if ref == "SFT":
            ix = KOL[kolad]["esl"][i]
            oA = olcu(A[[y for _, y in ix]], CA[[y for _, y in ix]])
            oR = olcu(Vb[[x for x, _ in ix]], Cb[[x for x, _ in ix]])
        else:
            B = KOL[ref]["V"][i]; CB = KOL[ref]["C"][i]
            n = min(len(A), len(B))
            oA = olcu(A[:n], CA[:n]); oR = olcu(B[:n], CB[:n])
        for k in out:
            out[k].append(oA[k] - oR[k])
    return dict({k: float(np.mean(v)) for k, v in out.items()},
                _tekil={k: [float(x) for x in v] for k, v in out.items()})


def boot_kol(KOL, Vb, Cb, istb, kolad, ref, olcuad="M1"):
    rng = np.random.default_rng(20260904)
    ist = KOL[kolad]["ist"][0]; ad = np.unique(ist)
    ixk = {u: np.where(ist == u)[0] for u in ad}
    if ref != "SFT":
        istR = KOL[ref]["ist"][0]; ixR = {u: np.where(istR == u)[0] for u in np.unique(istR)}
    else:
        ixR = {u: np.where(istb == u)[0] for u in np.unique(istb)}
    out = []
    for _ in range(NB):
        sec = rng.choice(ad, size=len(ad), replace=True)
        a_ix = np.concatenate([ixk[u] for u in sec if u in ixk])
        r_ix = np.concatenate([ixR[u] for u in sec if u in ixR])
        d = []
        for i in range(M):
            A = KOL[kolad]["V"][i]; CA = KOL[kolad]["C"][i]
            oA = olcu(A[a_ix], CA[a_ix])
            if ref == "SFT":
                oR = olcu(Vb[r_ix], Cb[r_ix])
            else:
                B = KOL[ref]["V"][i]; CB = KOL[ref]["C"][i]
                rr = r_ix[r_ix < len(B)]
                oR = olcu(B[rr], CB[rr])
            d.append(oA[olcuad] - oR[olcuad])
        out.append(np.mean(d))
    o = np.array(out); o = o[~np.isnan(o)]
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)))


def verdict(dA, dB, ciB, plaB, kol):
    gecti_bar = abs(dB) >= BAR
    ayrik = (ciB[0] > 0) or (ciB[1] < 0)
    plasebo_ustu = abs(dB) > plaB
    if BAR <= abs(dB) < MDE_B and not ayrik:
        return "GECMEZ-MDE-ALTI", dict(bar=gecti_bar, ayrik=ayrik, plasebo=plasebo_ustu)
    gecti = gecti_bar and ayrik and plasebo_ustu
    return ("GECTI" if gecti else "GECMEZ"), dict(bar=gecti_bar, ayrik=ayrik,
                                                  plasebo=plasebo_ustu)


def kaskad(H):
    a, y = H["KISI_ASAGI"], H["KISI_YUKARI"]
    if a["hucre_A"] != a["hucre_B"] or y["hucre_A"] != y["hucre_B"]:
        kisi = "DOZ-BAGIMLI"
    else:
        asagi = a["hal_B"] == "GECTI" and a["dB"] <= -BAR
        yukari = y["hal_B"] == "GECTI" and y["dB"] >= BAR
        kisi = ("KADRAN-IKI-YÖN" if (asagi and yukari)
                else "KADRAN-TEK-YÖN" if (asagi or yukari) else "DOZ-ALTI")
    yerinde = all(H[k]["kuvvet_yerinde"] and H[k]["dU_yerinde"]
                  for k in ("KISI_ASAGI", "KISI_YUKARI"))
    katman = "KATMAN-AYRIK" if yerinde else "KATMAN-BIRLIKTE"
    kv = H["KUVVET_ASAGI"]
    kuvvet = "KUVVET-DOZ-ALTI" if kv["kuvvet_yerinde"] else "KUVVET-KADRAN"
    return dict(kisi=kisi, katman=katman, kuvvet=kuvvet)


def main():
    print("★ §7.1 DEJENERE PROVA (kosudan ÖNCE):", flush=True)
    ok = _distinct4_provasi()
    p = {
        "i_sifir_etki":  verdict(0.0, 0.0, (-0.5, 0.5), 0.2, "x")[0] == "GECMEZ",
        "ii_buyuk_ayrik": verdict(-3, -3.0, (-3.4, -2.6), 0.4, "x")[0] == "GECTI",
        "iii_bar_ustu_CI_sifiri_iceriyor": verdict(-1.2, -1.2, (-2.5, 0.3), 0.4, "x")[0] == "GECMEZ-MDE-ALTI",
        "iv_bar_alti": verdict(-0.9, -0.9, (-1.1, -0.7), 0.1, "x")[0] == "GECMEZ",
        "v_plasebo_yutuyor": verdict(-2.0, -2.0, (-2.4, -1.6), 2.5, "x")[0] == "GECMEZ",
    }
    print(f"   verdict fonksiyonu: {json.dumps(p, ensure_ascii=False)}")
    ok = ok and all(p.values())
    print(f"   ⇒ PROVA {'GECTI' if ok else 'DÜSTÜ'} ⇒ EYLEM: düserse cikis 4, verdict YAZILMAZ",
          flush=True)
    if not ok:
        sys.exit(4)

    nlp = FS._boru(); t0 = time.time()
    _veri_eski = MK.VERI
    if TABAN_KOK:
        MK.VERI = TABAN_KOK
    Rb = MK.oku(*TABAN)
    MK.VERI = _veri_eski
    if Rb is None:
        print(f"★ TABAN okunamadi: {TABAN_KOK or MK.VERI}/{TABAN[0]}/{TABAN[1]} ⇒ DUR")
        sys.exit(4)
    Vb = MK.satir_bilesenleri(nlp, Rb)
    Cb = Vb[:, -1]
    anb = {(r.get("kol"), r.get("cekim"), r.get("istem_i")): j for j, r in enumerate(Rb)}
    istb = np.array([r.get("istem_i") for r in Rb])
    print(f"  [{time.time()-t0:.0f}s] TABAN {TABAN[0]}/{TABAN[1]}: {len(Rb)} satir · "
          f"M1={MK.olc_toplam(Vb)['M1']:.4f}", flush=True)

    KOL = {}
    for kol, yuva in KOL_YUVA.items():
        V4, C4, ist4, d4, esl = [], [], [], [], []
        for i in range(M):
            c = f"ckpt-{yuva + i*1000}"
            R = MK.oku("miniDPO", c)
            if R is None:
                print(f"{kol} {c}"); continue
            V = MK.satir_bilesenleri(nlp, R)
            cift = [(anb[k], j) for j, r in enumerate(R)
                    if (k := (r.get("kol"), r.get("cekim"), r.get("istem_i"))) in anb]
            V4.append(V); C4.append(V[:, -1])
            ist4.append(np.array([r.get("istem_i") for r in R]))
            d4.append(distinct4([r["metin"] for r in R]))
            esl.append(cift)
            print(f"  [{time.time()-t0:.0f}s] {kol}/{c}: eslesen {len(cift)} · "
                  f"distinct4={d4[-1]:.4f}", flush=True)
        KOL[kol] = dict(V=V4, C=C4, ist=ist4, d4=d4, esl=esl)

    raf = {k: (float(np.mean(v["d4"])) < 0.60) for k, v in KOL.items()}
    payda("dial_dshelf", n_kol=len(KOL), red_raf=int(sum(raf.values())),
          hal_d4={k: round(float(np.mean(v["d4"])), 4) for k, v in KOL.items()})
    print(f"  ★ D-SHELF: distinct-4 < 0,60 olan kol = {[k for k,v in raf.items() if v] or 'YOK'}",
          flush=True)

    def delta(kolad, ref):
        return delta_kol(KOL, Vb, Cb, kolad, ref)

    def boot(kolad, ref, olcuad="M1"):
        return boot_kol(KOL, Vb, Cb, istb, kolad, ref, olcuad)

    H = {}
    for kol in KOL_YUVA:
        dA = delta(kol, "SFT"); dB = delta(kol, REF_B) if kol != REF_B else \
            {k: 0.0 for k in dA}
        ciB = boot(kol, REF_B) if kol != REF_B else (0.0, 0.0)
        ciA = boot(kol, "SFT")
        pla = M9.plasebo_esli(KOL[kol]["V"][0], KOL[kol]["ist"][0], n=PLASEBO_N)
        halB, bB = verdict(dA["M1"], dB["M1"], ciB, pla["p95"], kol)
        halA, bA = verdict(dA["M1"], dA["M1"], ciA, pla["p95"], kol)
        zK = abs(dB["KUVVET"]) / NULL_SD_KUVVET
        H[kol] = dict(dA=dA["M1"], dB=dB["M1"], ciA=ciA, ciB=ciB,
                      plasebo_p95=pla["p95"], plasebo_ort=pla["ort"],
                      hal_A=halA, hal_B=halB, bilesen_A=bA, bilesen_B=bB,
                      hucre_A=("GECTI" if halA == "GECTI" else "GECMEZ"),
                      hucre_B=("GECTI" if halB == "GECTI" else "GECMEZ"),
                      dKUVVET=dB["KUVVET"], z_kuvvet=float(zK),
                      kuvvet_yerinde=bool(zK < 1.0),
                      dSAHIS1=dB["SAHIS1"], dJETON_CUMLE=dB["JETON_CUMLE"],
                      dSAHIS1_A=dA["SAHIS1"], dJETON_CUMLE_A=dA["JETON_CUMLE"],
                      distinct4=float(np.mean(KOL[kol]["d4"])), raf=bool(raf[kol]))
        print(f"  ★ {kol}: ΔM1(B)={dB['M1']:+.4f} CI{np.round(ciB,4).tolist()} · "
              f"ΔM1(A)={dA['M1']:+.4f} · plasebo p95={pla['p95']:.4f} · "
              f"B={halB} · A={halA}", flush=True)

    for kol, yuva in KOL_YUVA.items():
        y = f"{ROOT}/results/" + KD_DESEN.format(kol=kol)
        d = json.load(open(y, encoding="utf-8"))
        dU = d.get("dU_PREREG_DOM", d.get("dU_DOM"))
        sd = d.get("dU_PREREG_DOM_sd") or 1e-9
        z = abs(dU) / sd
        zM = abs(dU) / DU_NULL_SD_PREREG
        H[kol]["dU_PREREG_DOM"] = dU
        H[kol]["z_dU"] = float(z)
        H[kol]["dU_yerinde"] = bool(z < 1.0)
        H[kol]["z_dU_PREREG_REF"] = float(zM)
        H[kol]["dU_yerinde_PREREG_REF"] = bool(zM < 1.0)

    ctx = dict(KOL=KOL, Vb=Vb, Cb=Cb, istb=istb, raf=raf)
    K = KASKAD(H, ctx) if KASKAD else kaskad(H)
    if all(k in K for k in ("kisi", "katman", "kuvvet")):
        print(f"\n★★★ VERDICT · kisi = {K['kisi']} · katman = {K['katman']} · "
              f"kuvvet kolu = {K['kuvvet']}", flush=True)
    else:
        print("\n★★★ VERDICT · " + " · ".join(f"{k} = {v}" for k, v in K.items()
                                            if isinstance(v, str)), flush=True)
    OUT = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg=PREREG_ADI,
               bar=BAR, mde_B=MDE_B, null_sd_kuvvet=NULL_SD_KUVVET, m=M, nb=NB,
               du_null_sd_prereg=DU_NULL_SD_PREREG, errata_1=ERRATA_1,
               taban_M1=float(MK.olc_toplam(Vb)["M1"]), kol=H, verdict=K)
    json.dump(OUT, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}", flush=True)
    payda("dial_verdict", n_kol=len(H), hal_gecen_B=sum(1 for v in H.values() if v["hal_B"] == "GECTI"),
          red_raf=int(sum(raf.values())))


if __name__ == "__main__":
    main()
