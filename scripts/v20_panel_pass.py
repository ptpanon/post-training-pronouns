#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import csv, json, os, re, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_olcu as MO
import addressee_run as MK
import form_count as FS
import rung_force_ci as BK

PANEL = __DNH_DATA__ + "/c1_panel"
ELICIT = __DNH_DATA__ + "/elicit"
A_CARD = f"{ROOT}/results/PREREG9_A_OLCUM_2026-08-31.json"
B_CARD = f"{ROOT}/results/depersonalization16_measurement_2026-09-01.json"
WAR = __DNH_DATA__ + "/unreleased/Ratings_Warriner_et_al.csv"
CIK = f"{ROOT}/results/v20_panel_2026-09-06.json"
ILK_N = 50
KELIME = re.compile(r"[A-Za-z']+")
SAHIS2 = MO.SAHIS2

ISARET = re.compile(r"\n\s*(?:User|Assistant|Human|Q)\s*:"
                    r"|<\|im_start\|>|<\|start_header_id\|>|\[INST\]|<start_of_turn>")


def warriner_sozluk():
    d = {}
    with open(WAR, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                d[r["Word"].strip().lower()] = float(r["D.Mean.Sum"])
            except (KeyError, ValueError):
                continue
    return d


def war_puan(t, S):
    ks = [w.lower() for w in KELIME.findall(t or "")]
    v = [S[w] for w in ks if w in S]
    return (float(np.mean(v)) if v else np.nan, len(v), len(ks))


def kes(t):
    m = ISARET.search(t or "")
    return (t[:m.start()], True) if m else (t, False)


def _prova():
    S = {"wrong": 2.0, "please": 5.0}
    t1, k1 = kes("abc\nUser: def")
    t2, k2 = kes("abc def")
    p, n, tot = war_puan("wrong please zzz", S)
    return {"i_kesici_kesiyor": t1 == "abc" and k1,
            "ii_kesici_dokunmuyor": t2 == "abc def" and not k2,
            "iii_warriner_ortalama": abs(p - 3.5) < 1e-9 and n == 2 and tot == 3,
            "iv_warriner_bos_nan": bool(np.isnan(war_puan("", S)[0])),
            "v_sahis2_ithal": bool(SAHIS2.search("your")),
            "vi_isaret_template_da_yakalar": kes("a<|im_start|>b")[1]}


def oku(yol):
    return [json.loads(l) for l in open(yol, encoding="utf-8")]


def bacak_olc(nlp, R, S):
    M = [r["metin"] or "" for r in R]
    KES = [kes(t) for t in M]
    isaretli = np.array([k for _, k in KES])
    V = MK.satir_bilesenleri(nlp, [{"metin": t} for t in M])
    Vk = MK.satir_bilesenleri(nlp, [{"metin": t} for t, _ in KES])
    s2_tam = np.array([len(SAHIS2.findall(t)) for t in M], float)
    s2_kes = np.array([len(SAHIS2.findall(t)) for t, _ in KES], float)
    W = np.array([war_puan(t, S) for t in M], dtype=float)
    ilk2 = np.zeros(len(M)); son2 = np.zeros(len(M))
    ilk_n = np.zeros(len(M)); son_n = np.zeros(len(M))
    for i, t in enumerate(M):
        ks = KELIME.findall(t)
        a, b = " ".join(ks[:ILK_N]), " ".join(ks[ILK_N:])
        ilk2[i] = len(SAHIS2.findall(a)); son2[i] = len(SAHIS2.findall(b))
        ilk_n[i] = len(ks[:ILK_N]); son_n[i] = len(ks[ILK_N:])
    ist = np.array([r.get("istem_i", -1) for r in R])
    onek = [r.get("onek") or "" for r in R]
    return dict(V=V, Vk=Vk, ist=ist, isaretli=isaretli,
                s2_tam=s2_tam, s2_kes=s2_kes, W=W,
                ilk2=ilk2, son2=son2, ilk_n=ilk_n, son_n=son_n, onek=onek)


def main():
    P = _prova(); print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    if not all(P.values()):
        return 4
    S = warriner_sozluk()
    print(f"{len(S):,}"
          f"")
    if len(S) < 10000:
        return 5
    A = json.load(open(A_CARD, encoding="utf-8"))
    aileler = [(k, v["kontrast"].split("→")) for k, v in A.items()
               if isinstance(v, dict) and v.get("sinif") == "BIRINCIL"]
    nlp = FS._boru(); OUT = {}; t0 = time.time()
    for i, (aile, (z0, z1)) in enumerate(aileler):
        y0, y1 = f"{PANEL}/{aile}/{z0}/uretim.jsonl", f"{PANEL}/{aile}/{z1}/uretim.jsonl"
        if not (os.path.exists(y0) and os.path.exists(y1)):
            print(f"{aile}"); continue
        D = {z: bacak_olc(nlp, oku(y), S) for z, y in ((z0, y0), (z1, y1))}
        r = {}
        for z in (z0, z1):
            d = D[z]; V = d["V"]
            r[z] = dict(
                n=int(len(V)),
                duzeltme=float(V[:, 3].sum()), duzeltme_kisisiz=float(V[:, 4].sum()),
                M3=float(V[:, 4].sum() / max(V[:, 3].sum(), 1)),
                isaretli=int(d["isaretli"].sum()),
                s2_isaret_sonrasi=float((d["s2_tam"] - d["s2_kes"]).sum()),
                s2_toplam=float(d["s2_tam"].sum()),
                warriner=float(np.nanmean(d["W"][:, 0])),
                warriner_kapsam=float(np.nansum(d["W"][:, 1]) / max(np.nansum(d["W"][:, 2]), 1)),
                M1_ilk=float(1000 * d["ilk2"].sum() / max(d["ilk_n"].sum(), 1)),
                M1_son=float(1000 * d["son2"].sum() / max(d["son_n"].sum(), 1)),
                M1_onek=float(1000 * sum(len(SAHIS2.findall(o)) for o in d["onek"])
                              / max(sum(len(KELIME.findall(o)) for o in d["onek"]), 1)))
        a, b = D[z0], D[z1]
        r["dM3"] = r[z1]["M3"] - r[z0]["M3"]
        r["dM3_ci"] = BK.kume_boot(b["V"], b["ist"], a["V"], a["ist"], "M3")[:2]
        r["dM1_tam"] = MK.olc_toplam(b["V"])["M1"] - MK.olc_toplam(a["V"])["M1"]
        r["dM1_kesik"] = MK.olc_toplam(b["Vk"])["M1"] - MK.olc_toplam(a["Vk"])["M1"]
        r["dM1_kesik_ci"] = BK.kume_boot(b["Vk"], b["ist"], a["Vk"], a["ist"], "M1")[:2]
        r["dWarriner"] = r[z1]["warriner"] - r[z0]["warriner"]
        r["kontrast"] = f"{z0}→{z1}"
        OUT[aile] = r
        print(f"  ★ {aile:18s} ΔM3 {r['dM3']:+.4f} CI [{r['dM3_ci'][0]:+.4f},"
              f"{r['dM3_ci'][1]:+.4f}] · ΔM1 tam {r['dM1_tam']:+.2f} → kesik "
              f"{r['dM1_kesik']:+.2f} · ΔWarr {r['dWarriner']:+.4f} "
              f"[{i+1}/{len(aileler)} · {(time.time()-t0)/60:.1f} dk]", flush=True)

    ay3 = [k for k, v in OUT.items() if v["dM3_ci"][0] * v["dM3_ci"][1] > 0]
    yuk3 = [k for k in ay3 if OUT[k]["dM3"] > 0]
    ayk = [k for k, v in OUT.items()
           if v["dM1_kesik_ci"][0] * v["dM1_kesik_ci"][1] > 0 and v["dM1_kesik"] < 0]
    print(f"\n  [PAYDA] aile={len(OUT)} · ΔM3 CI-ayrik={len(ay3)} (yukari {len(yuk3)}) "
          f"⇒ esik: ayrik yoksa EYLEM = özetin «corrections migrate» cümlesi daralir")
    print(f"{len(ayk)} {len(OUT)} {len(OUT)}"
          f"")
    print(f"{sum(1 for v in OUT.values() if v['dWarriner']<0)}"
          f"{len(OUT)}")
    S_ = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          alet="scripts/v20_panel_pass.py", borc="D-0907-V20 §1 a,b,f,g",
                          SINIF="ÖLCÜM — dört nicelik, tek gecis",
                          warriner=WAR, n_sozluk=len(S), ilk_n=ILK_N,
                          isaret_deseni=ISARET.pattern, prova=P,
                          motor="addressee_olcu.topla · addressee_run.olc_toplam · "
                                "rung_force_ci.kume_boot"),
              aile=OUT,
              _ozet=dict(n_aile=len(OUT), dM3_ayrik=ay3, dM3_ayrik_yukari=yuk3,
                         kesik_dM1_ayrik_asagi=len(ayk),
                         warriner_asagi=sum(1 for v in OUT.values() if v["dWarriner"] < 0)))
    json.dump(S_, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK} · {(time.time()-t0)/60:.1f} dk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
