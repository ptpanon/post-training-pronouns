#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from verdict_name_crosscount import capraz_say
import package1_kutuplar as KP
import serita_harman102 as SA
import a3_harman_oran as A3
import adim4_harman2 as A4
import prefix_overnight_doz as GD
import prefix_harsh_vekil as HV
import prefix_ladder as MD
import prefix_judge_calibration as KAL

OUT, CIKTI = SA.OUT, SA.CIKTI
KAPI_B = f"{ROOT}/results/SERITB_CETVEL_2026-08-12.json"
BLOK = 3000


def _yargila(R, es, rubrik="TON", altdizin="yargi", on_ek="SA", template_kullan=None):
    os.makedirs(f"{OUT}/{altdizin}", exist_ok=True)
    if template_kullan is None:
        template_kullan = (rubrik == "TON")
    S = None
    if template_kullan:
        S = json.load(open(A3.TEMPLATE, encoding="utf-8"))
        bek = S.pop("sha256_govde")
        if hashlib.sha256(json.dumps(S, ensure_ascii=False, indent=2,
                                     sort_keys=True).encode()).hexdigest() != bek:
            raise SystemExit("★ KAPI: template sha tutmadi")
    JET, cik, red = Counter(), {}, Counter()
    n_blok = -(-len(R) // BLOK)
    for b in range(n_blok):
        yol = f"{OUT}/{altdizin}/blok_{b:03d}.json"
        i0, i1 = b * BLOK, min((b + 1) * BLOK, len(R))
        sec = [{"id": f"{on_ek}{i:06d}", "metin": R[i]["metin"]} for i in range(i0, i1)]
        if os.path.exists(yol):
            d = json.load(open(yol, encoding="utf-8"))
            if len(d.get("cikti", {})) >= 0.75 * len(sec):
                cik.update(d["cikti"])
                red.update(d.get("red", {}))
                JET.update(d.get("jeton", {}))
                print(f"  [blok {b+1}/{n_blok}] ATLANDI (tam) · {len(d['cikti'])}", flush=True)
                continue
        j = Counter()
        t0 = time.time()
        c, r = KAL.yargila_esz(KAL.RUBRIK[rubrik], sec, es, template=S, jeton=j)
        tmp = yol + f".tmp{os.getpid()}"
        json.dump(dict(cikti=c, red=dict(r), jeton=dict(j)), open(tmp, "w"),
                  ensure_ascii=False)
        os.replace(tmp, yol)
        cik.update(c)
        red.update(r)
        JET.update(j)
        print(f"  [blok {b+1}/{n_blok}] {len(c)}/{len(sec)} · {time.time()-t0:.0f}s",
              flush=True)
    return cik, dict(red), dict(JET)


def _egriler(uy_ad):
    E = []
    for ton in (0, 1):
        for zem in (2, 3, 4, 5):
            kol = []
            for k in range(6):
                c = tuple(sorted([ton] * k + [zem] * (5 - k)))
                kol.append(uy_ad[c])
            E.append(dict(ad=f"{KP.KUTUP_AD[ton]}_zemin_{KP.KUTUP_AD[zem]}",
                          ton=KP.KUTUP_AD[ton], zemin=KP.KUTUP_AD[zem], kollar=tuple(kol)))
    return E


def _regresyon(y, X, kume, rng, B=SA.B_BOOT):
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    ku = sorted(set(kume.tolist()))
    Bh = np.empty((B, X.shape[1]))
    for b in range(B):
        ad = [ku[i] for i in rng.choice(len(ku), len(ku), replace=True)]
        idx = np.concatenate([np.flatnonzero(kume == a) for a in ad])
        Bh[b] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    return b0, Bh


def coz(a):
    t0 = time.time()
    capraz_say(f"{ROOT}/scripts/serita_harman102.py", "verdict_seritA", SA.ADLAR)
    man = json.load(open(f"{OUT}/manifest.json", encoding="utf-8"))
    sha = hashlib.sha256(open(f"{OUT}/uretim.jsonl", "rb").read()).hexdigest()
    if sha != man["sha256_uretim"]:
        raise SystemExit(f"★ KAPI: üretim sha {sha[:16]} ≠ manifest")
    if man["lafiz_hash"] != KP.LAFIZ_HASH:
        raise SystemExit("★ KAPI: lafiz hash degismis")
    R = [json.loads(l) for l in open(f"{OUT}/uretim.jsonl", encoding="utf-8")]
    K, uc_eksen = SA.kollar()
    uy, _ = KP.izgara()
    uy_ad = {tuple(sorted(c)): KP.hucre_adi(c) for c in uy}

    metin = [r["metin"] for r in R]
    F, _ = HV.yuzey_hizli(metin)
    okunmaz, tekrar = np.asarray(F["okunmazlik"]), np.asarray(F["tekrar"])
    kol = np.array([r["kol"] for r in R], dtype=object)
    debate = np.array([r["debate"] for r in R], dtype=object)
    capa_m = kol == "capa_k0"
    e_ok = max(float(np.percentile(okunmaz[capa_m], 95)), A3.D22_TABAN)
    e_tk = float(np.percentile(tekrar[capa_m], 95))
    TUM_ORNEK = [o for oo in KP.ORNEK for o in oo]
    yanki = np.array([any(o[:60] in r["metin"] for o in TUM_ORNEK) for r in R])
    temiz = (okunmaz <= e_ok) & (tekrar <= e_tk) & (~echo)
    payda("seritA_kapilar", n_satir=len(R), n_kol=len(set(kol.tolist())),
          n_temiz=int(temiz.sum()), red_yanki=int(echo.sum()),
          red_dejenere=int((~temiz & ~echo).sum()), bekle={"n_kol": 116})

    B_kapi = json.load(open(KAPI_B, encoding="utf-8"))["CETVEL_KAPISI"]
    print(f"[CETVEL-KAPISI] {B_kapi['karar']}", flush=True)
    cik, red, JET = _yargila(R, SA.ES)
    et = np.array([cik.get(f"SA{i:06d}", "") for i in range(len(R))], dtype=object)
    payda("seritA_yargi", n_satir=len(R), n_etiketli=int((et != "").sum()),
          **{f"red_{k}": v for k, v in red.items()}, bekle={"n_satir": len(R)})

    E = _egriler(uy_ad)
    tum_kol = tuple(dict.fromkeys([k for e in E for k in e["kollar"]]))
    h_all, n_all = A4.kol_orani(et, kol, temiz, tum_kol)
    H = dict(zip(tum_kol, [round(float(x), 4) for x in h_all]))
    NK = dict(zip(tum_kol, n_all))
    h_capa, n_capa = A4.kol_orani(et, kol, temiz, ("capa_k0",))
    rng = np.random.default_rng(SA.SEED)

    nl = A4.null_menzil(et, kol, temiz, np.random.default_rng(SA.SEED + 1),
                        K=SA.K_NULL, kollar=tum_kol, kume=debate)
    mde_g = SA.Z * float(np.std(nl, ddof=1))
    mde_r = mde_g / SA.KAPPA

    egri = []
    for e in E:
        HB = A4.kume_boot_egri(et, kol, debate, temiz, np.random.default_rng(SA.SEED + 2),
                               B=SA.B_BOOT, kollar=e["kollar"])
        hh = np.array([H[k] for k in e["kollar"]], float)
        Rb = HB[:, -1] - HB[:, 0]
        egri.append(dict(**{k: e[k] for k in ("ad", "ton", "zemin")},
                         kollar=list(e["kollar"]), h=[float(x) for x in hh],
                         n_karar=[NK[k] for k in e["kollar"]],
                         R=round(float(hh[-1] - hh[0]), 4),
                         R_kume_CI=[round(float(np.nanquantile(Rb, q)), 4)
                                    for q in (0.025, 0.975)],
                         SEKIL=GD.sekil(hh, mde_r)))
        egri[-1]["_Rb"] = Rb

    ciftler = []
    for i in range(len(egri)):
        for j in range(i + 1, len(egri)):
            if egri[i]["ton"] != egri[j]["ton"]:
                continue
            d = egri[i]["_Rb"] - egri[j]["_Rb"]
            ci = [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]
            ciftler.append(dict(a=egri[i]["ad"], b=egri[j]["ad"],
                                delta=round(egri[i]["R"] - egri[j]["R"], 4),
                                CI=[round(x, 4) for x in ci],
                                ayrik=bool(ci[0] > 0 or ci[1] < 0)))
    etkilesim = any(c["ayrik"] for c in ciftler)

    ku = sorted(set(debate.tolist()))
    dlt = []
    for _ in range(SA.K_NULL):
        p = rng.permutation(len(ku))
        A_, B_ = {ku[i] for i in p[:len(ku)//2]}, {ku[i] for i in p[len(ku)//2:]}
        ma = np.array([d in A_ for d in debate])
        ha = A4.kol_orani(et[ma], kol[ma], temiz[ma], ("capa_k0",))[0][0]
        hb = A4.kol_orani(et[~ma], kol[~ma], temiz[~ma], ("capa_k0",))[0][0]
        dlt.append(abs(float(ha - hb)))
    capa_p95 = float(np.nanquantile(dlt, 0.95))
    capa_kararsiz = bool(capa_p95 > mde_r)

    saf = [c for c in uy if len(set(c)) == 1]
    pl = {}
    for c in saf:
        ad, pad = KP.hucre_adi(c), f"plasebo_{KP.KUTUP_AD[c[0]]}"
        hh = A4.kol_orani(et, kol, temiz, (ad, pad))[0]
        pl[KP.KUTUP_AD[c[0]]] = dict(gercek=round(float(hh[0]), 4),
                                     plasebo=round(float(hh[1]), 4),
                                     delta=round(float(hh[0] - hh[1]), 4))
    tk = {}
    for c in saf + [uc_eksen]:
        ad = KP.hucre_adi(c)
        hh = A4.kol_orani(et, kol, temiz, (ad, f"takas_{ad}"))[0]
        tk[ad] = dict(t1=round(float(hh[0]), 4), t2=round(float(hh[1]), 4),
                      abs_delta=round(abs(float(hh[0] - hh[1])), 4))
    tk_null = np.abs(A4.null_menzil(et, kol, temiz, np.random.default_rng(SA.SEED + 6),
                                    K=SA.K_NULL, kollar=tum_kol, kume=debate))
    tk_taban = dict(null_merkez=round(float(np.mean(tk_null)), 4),
                    null_p95=round(float(np.nanquantile(tk_null, 0.95)), 4),
                    gozlenen_ort=round(float(np.mean([v["abs_delta"] for v in tk.values()])), 4))

    hucre_ad = {KP.hucre_adi(c): c for c in uy}
    m = temiz & np.isin(et, SA.KARAR_SINIF) & np.array([k in hucre_ad for k in kol])
    X = np.array([[list(hucre_ad[k]).count(p) for p in range(6)] for k in kol[m]], float)
    y = (et[m] == "HARSH").astype(float)
    b0, Bh = _regresyon(y, X, debate[m], np.random.default_rng(SA.SEED + 7))
    reg = {KP.KUTUP_AD[p]: dict(beta=round(float(b0[p]), 4),
                                CI=[round(float(np.nanquantile(Bh[:, p], q)), 4)
                                    for q in (0.025, 0.975)]) for p in range(6)}
    yuzey = {}
    mm = temiz & np.array([k in hucre_ad for k in kol])
    Xs = np.array([[list(hucre_ad[k]).count(p) for p in range(6)] for k in kol[mm]], float)
    for ad_v, v in (("okunmazlik", okunmaz), ("tekrar", tekrar),
                    ("boy", np.array([len(t) for t in metin], float))):
        s = np.asarray(v, float)[mm]
        s = (s - s.mean()) / (s.std() + 1e-9)
        bb, BB = _regresyon(s, Xs, debate[mm], np.random.default_rng(SA.SEED + 8), B=100)
        yuzey[ad_v] = {KP.KUTUP_AD[p]: dict(
            beta=round(float(bb[p]), 4),
            CI=[round(float(np.nanquantile(BB[:, p], q)), 4) for q in (0.025, 0.975)])
            for p in range(6)}

    ayirt = {KP.KUTUP_AD[c[0]]: H.get(KP.hucre_adi(c)) for c in saf}

    sekiller = [e["SEKIL"] for e in egri]
    n_karar_min = [min(e["n_karar"]) for e in egri]
    kapi_tamam = bool(int(temiz.sum()) > 0 and (et != "").mean() > 0.8)
    null_bozuk = bool(abs(float(np.mean(nl))) / max(float(np.std(nl, ddof=1)), 1e-9) > SA.Z)
    VERDICT = SA.verdict_seritA(n_karar_min, kapi_tamam, null_bozuk, capa_kararsiz,
                            etkilesim, sekiller)

    for e in egri:
        e.pop("_Rb", None)
    kayit = dict(
        damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prereg="results/PREREG_SERITA_HARMAN102_2026-08-12.md",
        manifest={k: man[k] for k in ("model", "lafiz_hash", "capa_hash", "n_kol",
                                      "n_satir", "n_cekim", "betik_sha", "sha256_uretim")},
        n_satir=len(R), n_temiz=int(temiz.sum()),
        kapilar=dict(okunmazlik_p95=round(e_ok, 4), tekrar_p95=round(e_tk, 4),
                     red_yanki=int(echo.sum()), etiketli_oran=round(float((et != "").mean()), 4)),
        capa=dict(h=round(float(h_capa[0]), 4), n=int(n_capa[0]),
                  yari_bolme_p95=round(capa_p95, 4), MDE=round(mde_r, 4),
                  KARARSIZ=capa_kararsiz,
                  serh=(""
                        ""
                        "")),
        tavan=dict(TONart_x5=H.get(KP.hucre_adi((0,) * 5)),
                   serh="§4.8 — tavan bu rejimde ÖLCÜLDÜ, HARMAN-2'den tasinmadi"),
        h=H, n_karar=NK, ayirt_edilebilirlik=ayirt,
        MDE=dict(gozlenen=round(mde_g, 4), gercek=round(mde_r, 4), kappa=SA.KAPPA,
                 null_ort=round(float(np.mean(nl)), 4),
                 null_sd=round(float(np.std(nl, ddof=1)), 4), null_merkez_bozuk=null_bozuk),
        egriler=egri, sekiller=sekiller,
        etkilesim=dict(ciftler=ciftler, n_cift=len(ciftler),
                       n_ayrik=sum(c["ayrik"] for c in ciftler), VAR=etkilesim,
                       serh=(""
                             ""
                             "")),
        regresyon=reg, yuzey_kolu=yuzey,
        plasebo=pl, konum_takasi=dict(kollar=tk, taban=tk_taban),
        CETVEL_KAPISI=B_kapi, D_A_OKUMASI="OKUMA-ERTELENDI" if not B_kapi["acilan"] else "ACIK",
        jeton_OLCULEN=JET, yargic_api_jetonu=0, es=SA.ES, adlar=list(SA.ADLAR),
        VERDICT=VERDICT, saniye=round(time.time() - t0, 1))
    gec = CIKTI + f".tmp{os.getpid()}"
    json.dump(kayit, open(gec, "w"), ensure_ascii=False, indent=1, default=str)
    os.replace(gec, CIKTI)
    print(f"\n★ VERDICT {VERDICT} · sekiller {Counter(sekiller)} · "
          f"ayrik cift {kayit['etkilesim']['n_ayrik']}/{len(ciftler)} · "
          f"MDE_gercek {mde_r:.4f} · capa {kayit['capa']['h']}\n→ {CIKTI}")
    return 0
