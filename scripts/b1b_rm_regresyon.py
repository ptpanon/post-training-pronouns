#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, io, json, math, os, subprocess, sys, time

WT = __DNH_ROOT__ + "-p11-gpu"
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import filtre_verim as FV
import addressee_olcu as MO
import form_count as FS
import pool_person_delta as HKD
import judge_signature as Y1
import judge_signature4 as Y4
import judge_signature5 as Y5
import hedge_counter as HS
import refusal_counter as RS
from reading_style_object import payda

DIZ = __DNH_DATA__ + "/gpu_haftasi/b1b"
OZN = f"{DIZ}/oznitelik.npz"
KIMLIK = f"{DIZ}/ornek_kimlik.json"
METIN = f"{DIZ}/ornek_metin.jsonl"
DUR = f"{DIZ}/B1B_DUR"
CARD = f"{WT}/results/b1b_rm_regresyon.json"
PREREG = os.environ.get("B1B_PREREG", f"{KOK}/reports/opus/PREREG_GPU_B1B_P11_2026-10-28.md")
SEED, N_SET, NB = 20261028, 20000, 1000
RMLER = ("armorm", "skywork")
KONTROL = ("modal_1k", "hedge_1k", "RED", "LISTEKOD", "LOG_JETON", "LOG_CUMLE",
           "SAHIS3_1k", "ACIKLAMA_SORUSU")
assert tuple(Y5.YUZEY) == KONTROL, "kâgidin YUZEY kümesi degisti ⇒ alet mühürle uyusmaz"
SAHIS3_DESEN = Y5.SAHIS3
KAPI_TEKRAR_N, KAPI_TEKRAR_TOL = 200, 1e-3
ETA_CIFT, ETA_ESIK_SA = 500, 4.0
OZN_ETA_OLCUM, OZN_ETA_ESIK_DK = 500, 60.0
BASARISIZ_ESIK = 0.01
H_, A_ = "\n\nHuman:", "\n\nAssistant:"


def _sha(y, n=None):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:n] if n else h.hexdigest()


def _sha16s(*parca):
    return hashlib.sha256("\x1f".join(parca).encode("utf-8")).hexdigest()[:16]


def _damga():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prereg_kapisi():
    alet = _sha(os.path.abspath(__file__), 16)
    if not os.path.exists(PREREG):
        _cik(2, f"{PREREG}")
    d = os.path.dirname(PREREG)
    iz = subprocess.run(["git", "-C", d, "ls-files", "--error-unmatch", PREREG],
                        capture_output=True, text=True)
    if iz.returncode != 0:
        _cik(2, f"{PREREG}")
    temiz = subprocess.run(["git", "-C", d, "diff", "--quiet", "HEAD", "--", PREREG])
    if temiz.returncode != 0:
        _cik(2, f"PREREG HEAD'e göre DEGISMIS (commit'siz düzenleme): {PREREG}")
    if alet not in io.open(PREREG, encoding="utf-8").read():
        _cik(2, f"{alet}"
                f"")
    cm = subprocess.run(["git", "-C", d, "log", "-1", "--format=%H", "--", PREREG],
                        capture_output=True, text=True).stdout.strip()
    print(f"{os.path.basename(PREREG)} {cm[:8]} {alet}"
          f"", flush=True)
    return dict(prereg=PREREG, prereg_commit=cm, alet_sha16=alet)


def _cik(kod, msj):
    print(f"★★ {msj} ⇒ cikis {kod}", flush=True)
    sys.exit(kod)


def _tek_tur_duz(s):
    if not s or s.count(H_) != 1 or s.count(A_) != 1:
        return False, ""
    i, j = s.find(H_), s.find(A_)
    if i > j:
        return False, ""
    return True, s[i + len(H_):j].strip()


def _tek_tur_mesaj(m):
    roller = tuple(x.get("role") for x in (m or []))
    if roller != ("user", "assistant"):
        return False, ""
    return True, m[0].get("content") or ""


def _cift(kip, ch, re_, pr):
    c, r = FV.cikar({"chosen": ch, "rejected": re_}, kip)
    if not c or not r:
        return "red_havuz_bos_taraf", None, None, None
    if kip == "duz":
        tc, pc = _tek_tur_duz(ch); tr, prr = _tek_tur_duz(re_); p = pc
    else:
        tc, pc = _tek_tur_mesaj(ch); tr, prr = _tek_tur_mesaj(re_)
        p = pr or pc
    if not (tc and tr):
        return "red_cok_tur", None, None, None
    if pc != prr:
        return "red_istem_uyusmaz", None, None, None
    if not (p or "").strip():
        return "red_bos_istem", None, None, None
    return "OK", p, c, r


def orneklem(log=print):
    from datasets import load_dataset
    rng = np.random.default_rng(SEED)
    P, K = [], {}
    for ki, s in enumerate(FV.SETLER):
        d = load_dataset(s["repo"], split=s["split"])
        N = len(d)
        CH, RE = d["chosen"], d["rejected"]
        PR = d["prompt"] if "prompt" in d.column_names else [None] * N
        say = dict(n_havuz=N, red_havuz_bos_taraf=0, red_cok_tur=0, red_istem_uyusmaz=0,
                   red_bos_istem=0)
        gec = []
        for i in range(N):
            du = _cift(s["cikarici"], CH[i], RE[i], PR[i])[0]
            if du == "OK":
                gec.append(i)
            else:
                say[du] += 1
        say["n_havuz_gecerli"] = N - say["red_havuz_bos_taraf"]
        say["n_rm_gecerli"] = len(gec)
        if len(gec) < N_SET:
            _cik(6, f"{s['ad']}: RM-gecerli {len(gec)} < {N_SET}")
        sec = np.sort(rng.choice(np.asarray(gec), N_SET, replace=False))
        for i in sec:
            i = int(i)
            _, p, c, r = _cift(s["cikarici"], CH[i], RE[i], PR[i])
            P.append(dict(id=f"{s['ad']}:{i}", korpus=ki, satir=i, istem=p, chosen=c, rejected=r))
        say["n_secilen"] = int(len(sec))
        say["veri_parmak_izi"] = getattr(d, "_fingerprint", None)
        K[s["ad"]] = say
        payda(f"b1b_evren_{s['ad']}", **{k: v for k, v in say.items() if k != "veri_parmak_izi"},
              bekle={"n_secilen": N_SET})
        del d, CH, RE, PR
    return P, K


def _yanit_ham(nlp, metinler, n_process=6):
    A, nc = MO.topla(nlp, metinler, n_process=n_process)
    H = np.array([HS.say(t) for t in metinler], float)
    S3 = np.array([len(SAHIS3_DESEN.findall(t)) for t in metinler], float)
    RD = np.array([RS.red_mi(t) for t in metinler], float)
    LK = np.array([Y4.liste_kod_payi(t) for t in metinler], float)
    AQ = np.array([Y5._aq(t) for t in metinler], float)
    return A, nc, dict(hedge=H, sahis3=S3, red=RD, listekod=LK, aq=AQ)


def oznitelik():
    os.makedirs(DIZ, exist_ok=True)
    t0 = time.time()
    P, K = orneklem()
    n = len(P)
    print(f"  [{time.time()-t0:.0f}s] örneklem kuruldu: {n:,} cift", flush=True)
    nlp = FS._boru()
    eta_dk = 0.0
    for ki, s in enumerate(FV.SETLER):
        orn = [p["chosen"] for p in P if p["korpus"] == ki][:OZN_ETA_OLCUM]
        t1 = time.time(); _yanit_ham(nlp, orn); dt = time.time() - t1
        hiz = len(orn) / max(dt, 1e-9)
        eta_dk += 2 * N_SET / hiz / 60
        print(f"  [KAPI/W-71] {s['ad']}: {hiz:.0f} yanit/sn (ölcülü, {len(orn)} yanit)", flush=True)
    print(f"  [KAPI/W-71] toplam ETA {eta_dk:.1f} dk ⇒ esik {OZN_ETA_ESIK_DK:.0f} dk ⇒ EYLEM: "
          f"asarsa DUR (cikis 5)", flush=True)
    if eta_dk > OZN_ETA_ESIK_DK:
        _cik(5, f"öznitelik ETA {eta_dk:.1f} dk > {OZN_ETA_ESIK_DK} dk")
    metin = [p["chosen"] for p in P] + [p["rejected"] for p in P]
    t1 = time.time()
    A, nc, X = _yanit_ham(nlp, metin)
    print(f"  [{time.time()-t0:.0f}s] sayaclar bitti ({2*n:,} yanit, {time.time()-t1:.0f} sn)", flush=True)
    jt = np.asarray(A["n_jeton"], float)
    c2, _, _ = HKD.bin_jetonda(A, 0, n)
    r2, _, _ = HKD.bin_jetonda(A, n, 2 * n)
    dM1 = c2 - r2
    m1z, _s1z, mdz, ljz, lcz, _D = Y1.kur_tasarim(A, nc, jt, np.zeros(2 * n, int), 1)
    k1000 = lambda x: 1000.0 * x / np.maximum(jt, 1)
    sut = {"modal_1k": mdz, "hedge_1k": Y1._z(k1000(X["hedge"])), "RED": Y1._z(X["red"]),
           "LISTEKOD": Y1._z(X["listekod"]), "LOG_JETON": ljz, "LOG_CUMLE": lcz,
           "SAHIS3_1k": Y1._z(k1000(X["sahis3"])), "ACIKLAMA_SORUSU": Y1._z(X["aq"])}
    D = {f"d_{k}": v[:n] - v[n:] for k, v in sut.items()}
    ham = {"modal_1k": k1000(A["n_modal"]), "LOG_JETON": np.log(np.maximum(jt, 1.0)),
           "LOG_CUMLE": np.log(np.maximum(nc, 1.0)), "M1": k1000(A["m1_sahis2"])}
    es = {k: float(np.max(np.abs(Y1._z(D[f"d_{k}"]) - Y1._z(ham[k][:n] - ham[k][n:]))))
          for k in ("modal_1k", "LOG_JETON", "LOG_CUMLE")}
    es["M1_kur_vs_bin_jetonda"] = float(np.max(np.abs(Y1._z(m1z[:n] - m1z[n:]) - Y1._z(dM1))))
    print(f"  [ESDEGERLIK] max|fark| {json.dumps({k: f'{v:.1e}' for k, v in es.items()})} ⇒ "
          f"esik 1e-9 ⇒ EYLEM: asan varsa cikis 6", flush=True)
    if max(es.values()) > 1e-9:
        _cik(6, "esdegerlik kapisi düstü")
    kume_of, kume = {}, np.zeros(n, np.int64)
    for i, p in enumerate(P):
        kume[i] = kume_of.setdefault(p["istem"], len(kume_of))
    nan = {k: int((~np.isfinite(v)).sum()) for k, v in {"dM1": dM1, **D}.items()}
    bos = sum(1 for t in metin if not t.strip())
    sd0 = [k for k, v in {"dM1": dM1, **D}.items() if np.nanstd(v) < 1e-12]
    for ki, s in enumerate(FV.SETLER):
        m = np.array([p["korpus"] == ki for p in P])
        print(f"  [PAYDA] {s['ad']:14s} cift={int(m.sum()):,} · küme={len(set(kume[m])):,} · "
              f"jeton=0 yanit={int((jt[:n][m] == 0).sum() + (jt[n:][m] == 0).sum())} · "
              f"cümle=0 yanit={int((nc[:n][m] == 0).sum() + (nc[n:][m] == 0).sum())}", flush=True)
    payda("b1b_oznitelik", n_cift=n, n_kume=len(kume_of), n_yanit=2 * n,
          red_bos_yanit=bos, red_nan_toplam=sum(nan.values()), red_sd_sifir=len(sd0),
          bekle={"n_cift": 3 * N_SET})
    print(f"  [PAYDA] NaN sütun basina {nan} · sd=0 sütun {sd0 or 'YOK'} ⇒ esik: NaN>0 ya da sd=0 "
          f"⇒ EYLEM: cikis 7 (dejenere)", flush=True)
    if sum(nan.values()) or sd0:
        _cik(7, "dejenere/NaN sütun")
    ids = np.array([p["id"] for p in P])
    np.savez_compressed(
        OZN, ids=ids, korpus=np.array([p["korpus"] for p in P], np.int8), kume=kume,
        dM1=dM1, m1_c=c2, m1_r=r2, **D,
        **{f"ham_{k}_c": np.asarray(A[k], float)[:n] for k in ("n_jeton", "m1_sahis2", "m5_yakin", "n_modal")},
        **{f"ham_{k}_r": np.asarray(A[k], float)[n:] for k in ("n_jeton", "m1_sahis2", "m5_yakin", "n_modal")},
        ham_nc_c=nc[:n], ham_nc_r=nc[n:],
        **{f"ham_{k}_c": v[:n] for k, v in X.items()}, **{f"ham_{k}_r": v[n:] for k, v in X.items()})
    with io.open(METIN, "w", encoding="utf-8") as f:
        for p in P:
            f.write(json.dumps(dict(id=p["id"], istem=p["istem"], chosen=p["chosen"],
                                    rejected=p["rejected"]), ensure_ascii=False) + "\n")
    kimlik = dict(
        _kunye=dict(damga_utc=_damga(), alet="scripts/b1b_rm_regresyon.py (p11-gpu-hafta)",
                    alet_sha16=_sha(os.path.abspath(__file__), 16), seed=SEED, n_set=N_SET,
                    evren="HAVUZ (havuz_kisi_deltasi --tam, 494 833 cift) ∩ RM-gecerli",
                    korpus=K, esdegerlik=es, n_kume=len(kume_of),
                    oznitelik_sha16=_sha(OZN, 16), metin_sha16=_sha(METIN, 16),
                    spacy="form_count._boru (en_core_web_sm, ner/lemmatizer haric)",
                    sahis3_deseni="judge_signature5.SAHIS3 (:77)"),
        cift=[dict(id=p["id"], korpus=FV.SETLER[p["korpus"]]["ad"], satir=p["satir"],
                   kume=int(kume[i]), istem_sha16=_sha16s(p["istem"]),
                   metin_sha16=_sha16s(p["istem"], p["chosen"], p["rejected"]))
              for i, p in enumerate(P)])
    json.dump(kimlik, io.open(KIMLIK, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"✓ {OZN} · {KIMLIK} · {METIN} · toplam {(time.time()-t0)/60:.1f} dk", flush=True)
    return 0


def _metinler():
    Z = np.load(OZN)
    kim = json.load(io.open(KIMLIK, encoding="utf-8"))["cift"]
    M = [json.loads(l) for l in io.open(METIN, encoding="utf-8") if l.strip()]
    ids = [str(x) for x in Z["ids"]]
    if not (len(M) == len(kim) == len(ids)) or any(
            m["id"] != k["id"] or m["id"] != i or
            _sha16s(m["istem"], m["chosen"], m["rejected"]) != k["metin_sha16"]
            for m, k, i in zip(M, kim, ids)):
        _cik(6, "metin ↔ kimlik ↔ npz HIZALAMA BOZUK (icerik sha'si)")
    return M


def puanla(rm, dev):
    K = prereg_kapisi()
    if os.path.exists(DUR):
        print(f"★ DUR bayragi ({DUR}) ⇒ KOSMAZ"); return 0
    import v61_scoring as V61
    import gpu_lock as GK
    depo = dict(V61.PUANLAYICI)[rm]
    M = _metinler(); N = len(M)
    C = GK.kilitle(dev, beklenen_fiziksel=GK.fiziksel_bekle(dev), tam=True,
                   etiket=f"b1b_{rm}", gereken_vram_gb=20)
    C["cevre"]["cikis_koku_modul_sabiti"] = C["cevre"].get("cikis_koku")
    C["cevre"]["cikis_koku"] = DIZ
    C.update(rm=rm, depo=depo, hf_home=os.environ.get("HF_HOME"), damga_utc=_damga(), **K)
    json.dump(C, io.open(f"{DIZ}/cevre_{rm}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    Pz = V61.Puanlayici(rm, depo, dev)

    def _p(m, taraf):
        try:
            s = Pz.puan(m["istem"], m[taraf])
            return (s, None) if math.isfinite(s) else (None, "sonlu-degil")
        except Exception as e:
            return None, str(e)[:120]

    sec = np.random.default_rng(SEED + 1).choice(N, KAPI_TEKRAR_N, replace=False)
    G = []
    for gecis in (1, 2):
        G.append([_p(M[int(i)], t)[0] for i in sec for t in ("chosen", "rejected")])
    a, b = (np.array([np.nan if v is None else v for v in g], float) for g in G)
    ok = np.isfinite(a) & np.isfinite(b)
    maxd = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else float("nan")
    json.dump(dict(rm=rm, n_cift=KAPI_TEKRAR_N, n_cagri=int(2 * len(a)), n_sonlu=int(ok.sum()),
                   max_fark=maxd, tol=KAPI_TEKRAR_TOL, ids=[M[int(i)]["id"] for i in sec],
                   gecis1=a.tolist(), gecis2=b.tolist(), damga_utc=_damga()),
              io.open(f"{DIZ}/kapi_tekrar_{rm}.json", "w", encoding="utf-8"))
    print(f"  [KAPI-TEKRAR] {rm}: {KAPI_TEKRAR_N} cift × 2 taraf × 2 gecis · sonlu {int(ok.sum())}/"
          f"{len(a)} · max|Δ| = {maxd:.2e} ⇒ esik {KAPI_TEKRAR_TOL:g} ⇒ EYLEM: asarsa ya da sonlu "
          f"olmayan varsa cikis 3", flush=True)
    if not (ok.all() and maxd <= KAPI_TEKRAR_TOL):
        _cik(3, f"KAPI-TEKRAR düstü ({rm})")
    y = f"{DIZ}/puan_{rm}.jsonl"
    yapilan = set()
    if os.path.exists(y):
        for l in io.open(y, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("puan") is not None:
                    yapilan.add((r["id"], r["taraf"]))
    sira = np.random.default_rng(SEED + 2).permutation(N)
    kalan = [int(j) for j in sira if any((M[int(j)]["id"], t) not in yapilan for t in ("chosen", "rejected"))]
    print(f"  [{rm}] {N:,} cift · zaten tam {N - len(kalan):,} · kalan {len(kalan):,}", flush=True)
    t0 = time.time(); yeni = 0; basarisiz = 0
    with io.open(y, "a", encoding="utf-8") as f:
        for j in kalan:
            if os.path.exists(DUR):
                print(f"  ★★ DUR ⇒ {rm} {yeni} ciftte durdu", flush=True); break
            m = M[j]
            for t in ("chosen", "rejected"):
                if (m["id"], t) in yapilan:
                    continue
                s, h = _p(m, t)
                basarisiz += int(s is None)
                if s is None and basarisiz <= 3:
                    print(f"     basarisiz[{basarisiz}] {m['id']}/{t}: {h}", flush=True)
                f.write(json.dumps(dict(id=m["id"], taraf=t, puan=s, hata=h)) + "\n")
            yeni += 1
            if yeni == ETA_CIFT:
                f.flush()
                spc = (time.time() - t0) / ETA_CIFT
                eta = spc * (len(kalan) - yeni) / 3600
                print(f"  [W-71] {rm}: ilk {ETA_CIFT} cift {spc:.3f} sn/cift (ölcülü) ⇒ kalan "
                      f"{len(kalan)-yeni:,} cift ETA {eta:.2f} sa ⇒ esik {ETA_ESIK_SA:g} sa ⇒ EYLEM: "
                      f"asarsa DUR (cikis 5)", flush=True)
                if eta > ETA_ESIK_SA:
                    _cik(5, f"ETA {eta:.2f} sa > {ETA_ESIK_SA} sa ({rm})")
            if yeni % 250 == 0:
                f.flush()
                print(f"  [{rm}] {yeni:,}/{len(kalan):,} · {(time.time()-t0)/60:.1f} dk · "
                      f"basarisiz {basarisiz}", flush=True)
    S = _puanlar(y)
    n_tam = sum(1 for m in M for t in ("chosen", "rejected") if (m["id"], t) in S)
    eksik = 2 * N - n_tam
    oran = eksik / (2 * N)
    if n_tam == 0:
        _cik(3, f"{rm}: hicbir sonlu puan yok")
    payda(f"b1b_puanla_{rm}", n_cagri_beklenen=2 * N, n_sonlu=n_tam, red_eksik=eksik,
          red_basarisiz_bu_kosu=basarisiz, hal_yeni_cift=yeni, atl_zaten_tam_cift=N - len(kalan))
    print(f"  [PAYDA] {rm}: sonlu {n_tam:,}/{2*N:,} · eksik/basarisiz {eksik:,} (%{100*oran:.2f}) "
          f"⇒ esik %{100*BASARISIZ_ESIK:g} ⇒ EYLEM: asarsa cikis 3", flush=True)
    if oran > BASARISIZ_ESIK:
        return 3
    io.open(f"{DIZ}/.tamam_{rm}", "w", encoding="utf-8").write(_damga() + "\n")
    return 0


def _puanlar(y):
    S = {}
    for l in io.open(y, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("puan") is not None and math.isfinite(r["puan"]):
                S[(r["id"], r["taraf"])] = float(r["puan"])
    return S


def okuma_adi(ci):
    if ci[1] < 0:
        return "CEZALANDIRIYOR"
    if ci[0] > 0:
        return "ODULLENDIRIYOR"
    return "AYIRT-EDILEMEDI"


def ols_kume_boot(y, x, K, F, g, nb=NB, seed=SEED):
    one = np.ones_like(x)
    X1 = np.column_stack([x, K, one, F]); X2 = np.column_stack([x, one, F])
    b1 = Y1._fit(X1, y); b2 = Y1._fit(X2, y)
    g = np.unique(g, return_inverse=True)[1]
    sira = np.argsort(g, kind="stable")
    _, bas, say = np.unique(g[sira], return_index=True, return_counts=True)
    k = len(bas); rng = np.random.default_rng(seed); B1 = np.empty(nb); B2 = np.empty(nb)
    for i in range(nb):
        sel = rng.choice(k, k, replace=True)
        L = say[sel]; T = int(L.sum())
        ix = sira[np.repeat(bas[sel] - (np.cumsum(L) - L), L) + np.arange(T)]
        B1[i] = Y1._fit(X1[ix], y[ix])[0]; B2[i] = Y1._fit(X2[ix], y[ix])[0]
    ci = lambda B: [float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))]
    return dict(beta_kontrollu=float(b1[0]), ci_kontrollu=ci(B1), se_boot_kontrollu=float(B1.std()),
                beta_kontrolsuz=float(b2[0]), ci_kontrolsuz=ci(B2), se_boot_kontrolsuz=float(B2.std()),
                beta_kontroller_betimsel={a: float(b1[1 + j]) for j, a in enumerate(KONTROL)},
                n_kume=int(k))


def cekirdek(rm, Z, kok):
    y_ = f"{kok}/puan_{rm}.jsonl"
    ids = [str(i) for i in Z["ids"]]; N = len(ids)
    if not os.path.exists(y_):
        return dict(ad="OLCULMEDI", neden="", n_cift=N, n_tam=0)
    S = _puanlar(y_)
    c = np.array([S.get((i, "chosen"), np.nan) for i in ids])
    r = np.array([S.get((i, "rejected"), np.nan) for i in ids])
    m = c - r; tam = np.isfinite(m); n_tam = int(tam.sum())
    if n_tam < (1 - BASARISIZ_ESIK) * N:
        return dict(ad="OLCULMEDI", neden=f"kapsam {n_tam}/{N} < %{100*(1-BASARISIZ_ESIK):g}",
                    n_cift=N, n_tam=n_tam)
    if np.std(m[tam]) < 1e-12:
        return dict(ad="OLCULMEDI", neden="DEJENERE-MARJ (sd=0)", n_cift=N, n_tam=n_tam)
    x = Y1._z(Z["dM1"][tam].astype(float))
    K = np.column_stack([Y1._z(Z[f"d_{k}"][tam].astype(float)) for k in KONTROL])
    kp = Z["korpus"][tam]; F = np.column_stack([kp == 1, kp == 2]).astype(float)
    R = ols_kume_boot(Y1._z(m[tam]), x, K, F, Z["kume"][tam])
    R.update(ad=okuma_adi(R["ci_kontrollu"]), n_cift=N, n_tam=n_tam, red_eksik=N - n_tam,
             n_korpus={FV.SETLER[q]["ad"]: int((kp == q).sum()) for q in range(3)},
             marj_ham_ort=float(m[tam].mean()), marj_ham_sd=float(m[tam].std()))
    return R


def analiz():
    K = prereg_kapisi()
    Z = np.load(OZN)
    O = {rm: cekirdek(rm, Z, DIZ) for rm in RMLER}
    for rm, R in O.items():
        if R["ad"] == "OLCULMEDI":
            print(f"  ★ {rm}: OLCULMEDI — {R['neden']}", flush=True)
        else:
            print(f"  ★ {rm}: β(ΔM1|8 kontrol) = {R['beta_kontrollu']:+.4f} SD CI {R['ci_kontrollu']} "
                  f"⇒ {R['ad']} · kontrolsüz {R['beta_kontrolsuz']:+.4f} CI {R['ci_kontrolsuz']} · "
                  f"n {R['n_tam']:,}/{R['n_cift']:,} · küme {R['n_kume']:,}", flush=True)
    kim = json.load(io.open(KIMLIK, encoding="utf-8"))["_kunye"]
    out = dict(_kunye=dict(damga_utc=_damga(), alet="scripts/b1b_rm_regresyon.py",
                           alet_sha16=K["alet_sha16"], prereg=K["prereg"], prereg_commit=K["prereg_commit"],
                           oznitelik_sha16=_sha(OZN, 16), ornek=kim, seed=SEED, nb=NB,
                           okuma=(""
                                  ""),
                           birincil="β(ΔM1) 8 kontrol + korpus FE", ikincil="β(ΔM1) korpus FE, isimsiz"),
               rm=O)
    os.makedirs(os.path.dirname(CARD), exist_ok=True)
    json.dump(out, io.open(CARD, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    eksik = [rm for rm, R in O.items() if R["ad"] == "OLCULMEDI"]
    print(f"✓ {CARD} · OLCULMEDI {eksik or 'YOK'} ⇒ esik 0 ⇒ EYLEM: varsa cikis 3", flush=True)
    return 3 if eksik else 0


def _sentetik_tasarim(rng, n_set=N_SET):
    n = 3 * n_set
    kume = rng.integers(0, int(0.8 * n), n)
    ortak = rng.normal(size=(n, 1))
    D = {f"d_{k}": 0.5 * ortak[:, 0] + rng.normal(size=n) for k in KONTROL}
    return dict(ids=np.array([f"s:{i}" for i in range(n)]), korpus=np.repeat(np.arange(3), n_set),
                kume=kume, dM1=0.4 * ortak[:, 0] + rng.normal(size=n), **D)


def prova(sentetik=False):
    S = {}
    S["hedge_counter.prova"] = all(r["tuttu"] for r in HS.prova())
    S["refusal_counter._prova"] = all(RS._prova().values())
    S["imzasi4._prova"] = all(Y4._prova()[0].values())
    S["imzasi5._prova"] = all(Y5._prova().values())
    hp, nlp = HKD._prova()
    S["pool_person_delta._prova"] = all(hp.values())
    from datasets import load_dataset
    d = load_dataset("openbmb/UltraFeedback")["train"]
    T = [c.get("response") or "" for i in range(50) for c in d[i]["completions"]]
    n = len(T); U = np.load(__DNH_DATA__ + "/uf_bilesen.npz", allow_pickle=True)
    A, nc, X = _yanit_ham(nlp, T, n_process=1)
    ref = {"hedge": "uf_hedge", "red": "uf_red", "listekod": "uf_listekod", "sahis3": "uf_sahis3",
           "aq": "uf_aciklama_sorusu"}
    es = {k: float(np.max(np.abs(X[k] - np.load(f"{__DNH_DATA__}/{v}.npy")[:n]))) for k, v in ref.items()}
    es.update({k: float(np.max(np.abs(np.asarray(A[k], float) - U[k][:n].astype(float))))
               for k in ("n_jeton", "m1_sahis2", "m5_yakin", "n_modal")})
    es["nc"] = float(np.max(np.abs(nc - U["nc"][:n])))
    S["kagit_onbellegi_esdegerlik"] = max(es.values()) == 0.0
    print(f"  [PROVA] kâgit önbellegi (openbmb/UltraFeedback ilk {n} yanit) max|fark| {es}", flush=True)
    rng = np.random.default_rng(SEED)
    gercek = os.path.exists(OZN) and not sentetik
    Z = dict(np.load(OZN)) if gercek else _sentetik_tasarim(rng)
    print(f"  [PROVA] tasarim: {'GERCEK oznitelik.npz' if gercek else 'SENTETIK'} · "
          f"{len(Z['ids']):,} cift · {len(np.unique(Z['kume'])):,} küme", flush=True)
    pk = f"{DIZ}/prova"; os.makedirs(pk, exist_ok=True)
    x = Y1._z(Z["dM1"].astype(float))
    Kz = np.column_stack([Y1._z(Z[f"d_{k}"].astype(float)) for k in KONTROL])
    gam = np.array([0.08, -0.06, -0.10, 0.07, 0.10, -0.05, 0.12, 0.06])
    fe = np.array([0.0, 0.3, -0.2])[Z["korpus"]]
    gk = np.unique(Z["kume"], return_inverse=True)[1]
    u = rng.normal(0, 0.3, gk.max() + 1)[gk]

    def yaz(et, marj, eksik_pay=0.0):
        taban = rng.normal(0, 2.0, len(marj))
        dus = rng.random(len(marj)) < eksik_pay
        with io.open(f"{pk}/puan_{et}.jsonl", "w", encoding="utf-8") as f:
            for i, idd in enumerate(Z["ids"]):
                f.write(json.dumps(dict(id=str(idd), taraf="chosen", puan=float(taban[i] + marj[i] / 2))) + "\n")
                f.write(json.dumps(dict(id=str(idd), taraf="rejected",
                                        puan=None if dus[i] else float(taban[i] - marj[i] / 2))) + "\n")

    beklenen = {-0.05: None, -0.10: "CEZALANDIRIYOR", 0.0: "AYIRT-EDILEMEDI", 0.10: "ODULLENDIRIYOR"}
    satir = []
    for b, bek in beklenen.items():
        s = b * x + Kz @ gam + fe + u
        e = rng.normal(0, math.sqrt(max(1 - s.var(), 1e-6)), len(s))
        marj = 0.5 + s + e
        et = f"ekim_{b:+.2f}"
        yaz(et, marj)
        R = cekirdek(et, Z, pk)
        gec = abs(R["beta_kontrollu"] - b) <= 0.01 and (bek is None or R["ad"] == bek)
        S[f"ekim_{b:+.2f}"] = gec
        satir.append(dict(ekilen=b, beklenen_ad=bek, **{k: R[k] for k in (
            "beta_kontrollu", "ci_kontrollu", "ad", "beta_kontrolsuz", "ci_kontrolsuz", "n_kume")},
            sd_marj_ekilen=float(np.std(marj))))
        print(f"  [PROVA] ekilen β {b:+.2f} ⇒ β̂ {R['beta_kontrollu']:+.4f} CI [{R['ci_kontrollu'][0]:+.4f},"
              f"{R['ci_kontrollu'][1]:+.4f}] ⇒ {R['ad']} (beklenen {bek or '—'}) · |fark| "
              f"{abs(R['beta_kontrollu']-b):.4f} ≤ 0,01 · kontrolsüz {R['beta_kontrolsuz']:+.4f} "
              f"{'✓' if gec else '★ DÜSTÜ'}", flush=True)
    S["olculmedi_dosya_yok"] = cekirdek("yok_boyle_rm", Z, pk)["ad"] == "OLCULMEDI"
    yaz("eksik2", 0.5 + 0.1 * x + rng.normal(size=len(x)), eksik_pay=0.02)
    R = cekirdek("eksik2", Z, pk); S["olculmedi_kapsam"] = R["ad"] == "OLCULMEDI"
    yaz("sabit", np.zeros(len(x)))
    R2 = cekirdek("sabit", Z, pk); S["olculmedi_sabit_marj"] = R2["ad"] == "OLCULMEDI"
    print(f"{S['']} {R['neden']} {R2['neden']}", flush=True)
    gecen = sum(bool(v) for v in S.values())
    payda("b1b_prova", n_sinav=len(S), hal_gecen=gecen, red_dusen=len(S) - gecen)
    for k, v in S.items():
        print(f"  [PROVA] {k:32s} {'✓' if v else '★ DÜSTÜ'}")
    json.dump(dict(damga_utc=_damga(), alet_sha16=_sha(os.path.abspath(__file__), 16),
                   tasarim="gercek" if gercek else "sentetik", sinav=S, ekim=satir, esdegerlik=es),
              io.open(f"{pk}/prova_sonuc.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ⇒ esik {len(S)}/{len(S)} ⇒ EYLEM: düsen varsa puanla/analiz KOSMAZ (cikis 4)", flush=True)
    return 0 if gecen == len(S) else 4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    alt = ap.add_subparsers(dest="kip", required=True)
    alt.add_parser("oznitelik")
    p = alt.add_parser("puanla"); p.add_argument("--rm", choices=RMLER, required=True)
    p.add_argument("--dev", default="cuda:0")
    alt.add_parser("analiz")
    pr = alt.add_parser("prova"); pr.add_argument("--sentetik", action="store_true")
    a = ap.parse_args()
    if a.kip == "oznitelik":
        sys.exit(oznitelik())
    if a.kip == "puanla":
        sys.exit(puanla(a.rm, a.dev))
    if a.kip == "analiz":
        sys.exit(analiz())
    sys.exit(prova(a.sentetik))
