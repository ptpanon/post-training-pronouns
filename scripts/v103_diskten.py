#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, io, json, os, re, subprocess, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
DIS = __DNH_DATA__ + ""
G = f"{ROOT}/results"
RULE = f"{ROOT}/preregistration/RULE_V103_DISKTEN_2026-09-23.md"
CIK = f"{G}/v103_diskten_2026-09-23.json"
TALIMAT_CARD = f"{G}/siradan_instruction_reading_2026-09-17.json"
SUZ_CARD = f"{G}/template_filtered_2026-09-07.json"
KK = f"{DIS}/dial_kol"
YET = f"{DIS}/c1_panel_yetenek"
CIP = f"{DIS}/c1_panel"

CIT = re.compile(r"```")
KOD_IPUCU = ("code", "python", "javascript", "java ", "c\\+\\+", "function", "program",
             "script", "compile", "api", "sql", "regex", "algorithm", "debug", "snippet")
KOD_IPUCU_RE = re.compile("|".join(KOD_IPUCU), re.I)
A_MODEL = ("Tulu3-8B", "Llama-3.1-8B")
D_KOL = (("kayit_sahissiz", "no pronoun"), ("kayit", "one you"), ("acik", "names the forms"))
PANEL16 = ("Qwen2.5-1.5B", "Qwen2.5-3B", "Mistral-7B-v0.3", "Qwen2.5-7B", "OLMo-3-7B", "Llama-3.1-8B",
           "Tulu3-8B", "Gemma-3-4B", "OLMo2-13B", "Gemma-3-12B", "Qwen2.5-14B", "Gemma-3-27B",
           "OLMo2-32B", "Qwen2.5-32B", "Qwen2.5-72B", "Llama-3.1-70B")
EKH4 = ("Gemma-4-12B", "Gemma-4-26B-A4B", "Gemma-4-E4B", "Mistral-Small-24B")


def sha(y):
    return hashlib.sha256(open(y, "rb").read()).hexdigest()


def rule_kapisi():
    h = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "-1", "--", os.path.relpath(RULE, ROOT)],
                       capture_output=True, text=True).stdout.strip()
    if not h:
        raise SystemExit("")
    b = sha(os.path.abspath(__file__))
    if b not in io.open(RULE, encoding="utf-8").read():
        raise SystemExit(f"{b[:16]}")
    return h, b[:16]


def damga():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def oku_jsonl(y):
    return [json.loads(l) for l in io.open(y, encoding="utf-8")] if os.path.exists(y) else None


def parca_a():
    K = json.load(io.open(TALIMAT_CARD, encoding="utf-8"))
    out = {"_kaynak": dict(card=os.path.relpath(TALIMAT_CARD, ROOT), sha256_16=sha(TALIMAT_CARD)[:16],
                           tanim="hizali düzey = M1.taban + M1.gozlenen (ayni hücreler, ayni card)")}
    for m in A_MODEL:
        a = K["aile"][m]
        t, gz = a["M1"]["taban"], a["M1"]["gozlenen"]
        out[m] = dict(M1_taban=round(t, 4), dM1=round(gz, 4), M1_hizali=round(t + gz, 4),
                      ci=[round(x, 4) for x in a["M1"]["ci"]], ayrik=bool(a["ayrik"] if "ayrik" in a else a["M1"]["ayrik"]),
                      n_tutulan=a["n_tutulan"])
    out["bant_hizali"] = K["bant"]["hizali"]
    out["taban_ayni_mi"] = dict(fark=round(out[A_MODEL[0]]["M1_taban"] - out[A_MODEL[1]]["M1_taban"], 4),
                                serh=""
                                     "")
    o0, o1 = out[A_MODEL[0]]["M1_hizali"], out[A_MODEL[1]]["M1_hizali"]
    out["kat"] = round(max(o0, o1) / min(o0, o1), 3) if min(o0, o1) > 0 else None
    print(f"{A_MODEL[0]} {o0:.2f} {A_MODEL[1]} {o1:.2f} {out['kat']}"
          f"{out[A_MODEL[0]]['M1_taban']:.2f} {out[A_MODEL[1]]['M1_taban']:.2f}"
          f"")
    return out


def parca_b():
    import form_count as FS, addressee_run as MK, rung_force_ci as BK
    import reviewer_readings_v87 as H87
    import template_filtered as SS
    S = json.load(io.open(SUZ_CARD, encoding="utf-8"))
    yuk = list(S["sayim"]["sablonlu"]["pozitif"])
    zem = {r["aile"]: (r["zemin_taban"], r["zemin_hizali"]) for r in S["aileler"]}
    nlp = FS._boru(); out = {"_kaynak": dict(card=os.path.relpath(SUZ_CARD, ROOT), sha256_16=sha(SUZ_CARD)[:16],
                                             yukselen=yuk, n=len(yuk),
                                             tanim=""
                                                   "")}
    t0 = time.time(); ETA = None
    for n, a in enumerate(yuk, 1):
        zt, zh = zem[a]
        Rt, Rh = SS.oku(SS.SAB, a, zt), SS.oku(SS.SAB, a, zh)
        Rct, Rch = SS.oku(SS.CIP, a, zt), SS.oku(SS.CIP, a, zh)
        if any(R is None or len(R) != SS.N_BEK for R in (Rt, Rh, Rct, Rch)):
            out[a] = dict(hal="DÖRT HÜCRE TAM DEGIL"); print(f"  ✗ {a}: dört hücre tam degil"); continue
        F = {k: SS.bayraklar(R) for k, R in (("st", Rt), ("sh", Rh), ("ct", Rct), ("ch", Rch))}
        K = {k: [SS.anahtar(r) for r in R] for k, R in (("st", Rt), ("sh", Rh), ("ct", Rct), ("ch", Rch))}
        kirli = set()
        for k in F:
            kirli |= {K[k][i] for i in np.where(F[k][0])[0]}
        mt = np.array([x not in kirli for x in K["st"]]); mh = np.array([x not in kirli for x in K["sh"]])
        kay_t, kay_h = H87.cumle_kayitlari(nlp, Rt), H87.cumle_kayitlari(nlp, Rh)
        Vt_tam, Vh_tam = H87.vektor(kay_t), H87.vektor(kay_h)
        Ct, Ch = MK.satir_bilesenleri(nlp, Rt), MK.satir_bilesenleri(nlp, Rh)
        e1 = abs(MK.olc_toplam(Vt_tam[mt])["M1"] - MK.olc_toplam(Ct[mt])["M1"])
        e2 = abs(MK.olc_toplam(Vh_tam[mh])["M1"] - MK.olc_toplam(Ch[mh])["M1"])
        if max(e1, e2) > 1e-6:
            out[a] = dict(hal=f"ESDEGERLIK DÜSTÜ ({e1:.2e}/{e2:.2e})")
            print(f"  ✗ {a}: esdegerlik düstü {e1:.2e}/{e2:.2e} ⇒ ÖLCÜLMEDI"); continue
        sec_t = [[j > 0 for j in range(len(cs))] for cs in kay_t]
        sec_h = [[j > 0 for j in range(len(cs))] for cs in kay_h]
        Vt_at, Vh_at = H87.vektor(kay_t, sec_t), H87.vektor(kay_h, sec_h)
        it = np.array([x.get("istem_i", -1) for x in Rt]); ih = np.array([x.get("istem_i", -1) for x in Rh])
        r = {}
        for ad, (Vt, Vh) in (("tam", (Vt_tam, Vh_tam)), ("ilk_cumle_atildi", (Vt_at, Vh_at))):
            d = MK.olc_toplam(Vh[mh])["M1"] - MK.olc_toplam(Vt[mt])["M1"]
            lo, hi, nk = BK.kume_boot(Vh[mh], ih[mh], Vt[mt], it[mt], "M1")
            r[ad] = dict(dM1=round(float(d), 4), ci=[round(lo, 4), round(hi, 4)],
                         ayrik=bool(lo * hi > 0), yon="yukari" if d > 0 else "asagi",
                         M1_taban=round(float(MK.olc_toplam(Vt[mt])["M1"]), 4),
                         M1_hizali=round(float(MK.olc_toplam(Vh[mh])["M1"]), 4), n_istem=int(nk))
        r["n_cumle_atilan_taban"] = int(sum(1 for cs in kay_t if cs))
        r["n_cumle_atilan_hizali"] = int(sum(1 for cs in kay_h if cs))
        r["pay_jeton_atilan_hizali"] = round(float(1 - Vh_at[mh][:, 0].sum() / max(Vh_tam[mh][:, 0].sum(), 1)), 4)
        r["pay_jeton_atilan_taban"] = round(float(1 - Vt_at[mt][:, 0].sum() / max(Vt_tam[mt][:, 0].sum(), 1)), 4)
        r["hal"] = "ÖLCÜLDÜ"; r["esdegerlik"] = [float(e1), float(e2)]
        out[a] = r
        if ETA is None:
            ETA = (time.time() - t0) * len(yuk) / 60
            print(f"{(time.time()-t0)/60:.1f} {ETA:.1f}"
                  f"")
        print(f"  ★ (b) {a:15s} tam ΔM1 {r['tam']['dM1']:+7.2f}{'*' if r['tam']['ayrik'] else ' '} · "
              f"ilk cümle atilinca {r['ilk_cumle_atildi']['dM1']:+7.2f}{'*' if r['ilk_cumle_atildi']['ayrik'] else ' '} · "
              f"atilan jeton payi hizali {r['pay_jeton_atilan_hizali']:.1%} [{n}/{len(yuk)} · "
              f"{(time.time()-t0)/60:.1f} dk]", flush=True)
    olc = [a for a in yuk if out.get(a, {}).get("hal") == "ÖLCÜLDÜ"]
    kalan = [a for a in olc if out[a]["ilk_cumle_atildi"]["dM1"] > 0 and out[a]["ilk_cumle_atildi"]["ayrik"]]
    out["_sayim"] = dict(n_yukselen=len(yuk), n_olculen=len(olc), n_hala_yukselen_ayrik=len(kalan),
                         hala_yukselen=kalan,
                         dusen=[a for a in olc if a not in kalan])
    print(f"  ★ (b) PAYDA: yükselen {len(yuk)} · ölcülen {len(olc)} · ilk cümle atilinca hâlâ yükselen-ayrik "
          f"{len(kalan)} {kalan}")
    return out


def _kod_jeton(t):
    t = t or ""
    top = len(t.split())
    p = [m.start() for m in CIT.finditer(t)]
    kod, i = 0, 0
    while i < len(p):
        a = p[i]
        b = p[i + 1] + 3 if i + 1 < len(p) else len(t)
        kod += len(t[a:b].split())
        i += 2
    return kod, top


def _yanit(m):
    if isinstance(m, list):
        return " ".join(x.get("content", "") for x in m if x.get("role") in (None, "assistant")) or \
               (m[-1].get("content", "") if m else "")
    return m or ""


def parca_c():
    out = {"_kaynak": dict(kol=[f"dial_kol/{k}.jsonl" for k in ("KISI_ASAGI", "RASTGELE")],
                           tanim=""
                                 "")}
    for kol in ("KISI_ASAGI", "RASTGELE"):
        y = f"{KK}/{kol}.jsonl"
        R = oku_jsonl(y)
        if R is None:
            out[kol] = dict(hal="DOSYA YOK"); continue
        acc = {"chosen": [0, 0, 0], "rejected": [0, 0, 0]}
        for r in R:
            for taraf in ("chosen", "rejected"):
                k, t = _kod_jeton(_yanit(r.get(taraf)))
                acc[taraf][0] += k; acc[taraf][1] += t; acc[taraf][2] += int(k > 0)
        d = {}
        for taraf, (k, t, n) in acc.items():
            d[taraf] = dict(kod_jeton_top=k, jeton_top=t, kod_jeton_pay=round(k / max(t, 1), 5),
                            kod_jeton_ort=round(k / len(R), 2), kod_iceren_pay=round(n / len(R), 4))
        d["fark"] = dict(kod_jeton_pay=round(d["chosen"]["kod_jeton_pay"] - d["rejected"]["kod_jeton_pay"], 5),
                         kod_jeton_ort=round(d["chosen"]["kod_jeton_ort"] - d["rejected"]["kod_jeton_ort"], 2),
                         kod_iceren_pay=round(d["chosen"]["kod_iceren_pay"] - d["rejected"]["kod_iceren_pay"], 4))
        d["n_cift"] = len(R); d["sha256_16"] = sha(y)[:16]
        out[kol] = d
        print(f"  ★ (c) {kol:12s} kod jeton payi chosen {d['chosen']['kod_jeton_pay']:.4f} · "
              f"rejected {d['rejected']['kod_jeton_pay']:.4f} · fark {d['fark']['kod_jeton_pay']:+.4f} · "
              f"yanit basina ort. fark {d['fark']['kod_jeton_ort']:+.1f} jeton")
    R = oku_jsonl(f"{CIP}/Llama-3.1-8B/instruct/uretim.jsonl")
    ist = sorted({(r.get("tez"), r.get("durus")) for r in R})
    onek = sorted({r.get("onek") or "" for r in R})
    tam = [f"{a} ({b})" for a, b in ist] + onek
    cit = [x for x in tam if CIT.search(x or "")]
    ipu = sorted({(x[:60], m.group(0)) for x in tam if (m := KOD_IPUCU_RE.search(x or ""))})
    out["munazara_paneli"] = dict(n_tez_durus=len(ist), n_onek=len(onek), n_istem_metni=len(tam),
                                  n_kod_citi=len(cit), n_kod_ipucu=len(ipu),
                                  ipucu_listesi=list(KOD_IPUCU), ornek=ipu[:5])
    print(f"{len(tam)} {len(ist)} {len(onek)}"
          f"{len(cit)} {len(ipu)}")
    return out


def parca_d():
    import form_count as FS, addressee_run as MK
    import reviewer_readings_v80 as H80
    import continuation_mode_exit as Q2
    import capability_verdict as YH
    nlp = FS._boru()
    def olc(R):
        V = MK.satir_bilesenleri(nlp, R)
        met = [r.get("metin") or "" for r in R]
        sat = [ln for t in met for ln in t.split("\n") if ln.strip()]
        liste = sum(1 for ln in sat if H80.SATIR.match(ln))
        ret = sum(1 for t in met if Q2.sinif(t) == "RET")
        return dict(n=len(R), jeton_ort=round(float(V[:, 0].mean()), 2),
                    liste_payi=round(liste / max(len(sat), 1), 4),
                    ret_orani=round(ret / max(len(R), 1), 4))
    aileler = sorted(x for x in os.listdir(f"{YET}/kayit") if os.path.isdir(f"{YET}/kayit/{x}"))
    out = {"_kaynak": dict(referans="c1_panel/<aile>/<hizali bacak> (cümlesiz)", kollar=[k for k, _ in D_KOL],
                           tanim="uzunluk = satir basina jeton ortalamasi (MK) · liste payi = H80.SATIR eslesen "
                                 "bos-olmayan satir payi · ret orani = Q2.sinif == RET payi")}
    t0 = time.time(); ETA = None
    for n, a in enumerate(aileler, 1):
        h_ad, h_yol = YH._bacak(a, YH.HIZALI_ADLARI)
        if not h_ad:
            out[a] = dict(hal="HIZALI BACAK YOK"); continue
        Rref = oku_jsonl(h_yol)
        r = dict(hizali_ad=h_ad, referans=olc(Rref))
        for kol, etiket in D_KOL:
            k_ad = next((x for x in YH.HIZALI_ADLARI if os.path.exists(f"{YET}/{kol}/{a}/{x}/uretim.jsonl")), None)
            y = f"{YET}/{kol}/{a}/{k_ad}/uretim.jsonl" if k_ad else ""
            R = oku_jsonl(y) if k_ad else None
            if R is None:
                r[kol] = dict(hal="YOK"); continue
            m = olc(R)
            m["fark"] = dict(jeton_ort=round(m["jeton_ort"] - r["referans"]["jeton_ort"], 2),
                             liste_payi=round(m["liste_payi"] - r["referans"]["liste_payi"], 4),
                             ret_orani=round(m["ret_orani"] - r["referans"]["ret_orani"], 4))
            m["etiket"] = etiket; m["bacak_ad"] = k_ad
            r[kol] = m
        out[a] = r
        if ETA is None:
            ETA = (time.time() - t0) * len(aileler) / 60
            print(f"{(time.time()-t0)/60:.1f} {ETA:.1f}"
                  f"")
        print(f"  ★ (d) {a:15s} " + " · ".join(
            f"{k}: Δjeton {r[k]['fark']['jeton_ort']:+6.1f} Δliste {r[k]['fark']['liste_payi']:+.3f} "
            f"Δret {r[k]['fark']['ret_orani']:+.3f}" for k, _ in D_KOL if "fark" in r.get(k, {}))
            + f"  [{n}/{len(aileler)} · {(time.time()-t0)/60:.1f} dk]", flush=True)
    olc_a = [a for a in aileler if "referans" in out.get(a, {})]
    med = {}
    for kol, etiket in D_KOL:
        v = {m: [out[a][kol]["fark"][m] for a in olc_a if "fark" in out[a].get(kol, {})] for m in
             ("jeton_ort", "liste_payi", "ret_orani")}
        med[kol] = dict(etiket=etiket, n=len(v["jeton_ort"]),
                        **{m: round(float(np.median(x)), 4) for m, x in v.items()})
    out["_medyan"] = med
    for kol, _ in D_KOL:
        m = med[kol]
        print(f"  ★ (d) MEDYAN {kol:16s} (n={m['n']}) Δuzunluk {m['jeton_ort']:+.1f} jeton · "
              f"Δliste payi {m['liste_payi']:+.4f} · Δret orani {m['ret_orani']:+.4f}")
    return out


def parca_e():
    P = f"{ROOT}/paper"
    kagit = io.open(f"{P}/p11.tex", encoding="utf-8").read()
    for y2 in sorted(os.listdir(f"{P}/fig")):
        if y2.endswith(".tex"):
            kagit += io.open(f"{P}/fig/{y2}", encoding="utf-8").read()
    kagit_l = kagit.lower()
    def parcalar(ad):
        return [t for t in re.findall(r"[A-Za-z]+", ad) if len(t) >= 4]
    def iz(ad):
        pl = parcalar(ad)
        say = {t: kagit_l.count(t.lower()) for t in pl}
        return say, (min(say.values()) if say else 0)
    out = {"_kaynak": dict(kok=os.path.relpath(CIP, DIS), panel16=list(PANEL16), ekH4=list(EKH4),
                           bizim_egitim=["miniDPO"],
                           tanim=""
                                 ""
                                 "")}
    izyok, uretilmemis = [], []
    for a in sorted(os.listdir(CIP)):
        p = f"{CIP}/{a}"
        if a.startswith("_") or not os.path.isdir(p):
            continue
        bacak, yarim = {}, {}
        for b in sorted(os.listdir(p)):
            if not os.path.isdir(f"{p}/{b}"):
                continue
            yol = f"{p}/{b}/uretim.jsonl"
            bacak[b] = len(oku_jsonl(yol) or []) if os.path.exists(yol) else 0
            kd = f"{p}/{b}/kol"
            yarim[b] = len([x for x in os.listdir(kd) if x.endswith(".jsonl")]) if os.path.isdir(kd) else 0
        say, en_seyrek = iz(a)
        sinif = ("PANEL-16" if a in PANEL16 else "EK-H-4" if a in EKH4 else
                 "BIZIM-EGITIM" if a == "miniDPO" else
                 ("KÂGITTA IZ YOK" if en_seyrek == 0 else "KÂGITTA IZ VAR"))
        r = dict(sinif=sinif, bacak=bacak, yarim_kol_dosyasi=yarim, n_satir=sum(bacak.values()),
                 kagit_gecis=say, en_seyrek=en_seyrek)
        out[a] = r
        if sinif == "KÂGITTA IZ YOK":
            izyok.append(a)
            if r["n_satir"] == 0:
                uretilmemis.append(a)
        print(f"{a:20s} {sinif:15s} {r['n_satir']:7d}"
              f"{sum(yarim.values()):3d} {say}")
    out["_sayim"] = dict(n_dizin=len([k for k in out if not k.startswith("_")]),
                         izyok=izyok, n_izyok=len(izyok),
                         izyok_ve_uretilmemis=uretilmemis,
                         izyok_ve_uretilmis=[a for a in izyok if a not in uretilmemis],
                         n_uretilip_disarida=len([a for a in izyok if a not in uretilmemis]))
    print(f"{out['_sayim']['n_dizin']} {len(izyok)} {izyok}"
          f"{uretilmemis}"
          f"{out['_sayim']['n_uretilip_disarida']}")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parca", default="a,b,c,d,e")
    a = ap.parse_args()
    h, b = rule_kapisi()
    K = json.load(io.open(CIK, encoding="utf-8")) if os.path.exists(CIK) else {}
    K["_kunye"] = dict(sinif="BETIM · bar YOK · sahip «EXECUTOR · v103» (5)", rule=h, alet=b,
                       alet_yolu="scripts/v103_diskten.py", damga_utc=damga())
    for p in a.parca.split(","):
        p = p.strip()
        if not p:
            continue
        print(f"\n════ PARCA ({p}) ════", flush=True)
        K[p] = dict(globals()[f"parca_{p}"]())
        json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"✓ {CIK} [parca {p}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
