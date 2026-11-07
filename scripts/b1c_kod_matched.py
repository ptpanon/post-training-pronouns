#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import contextlib, glob, hashlib, json, math, os, subprocess, sys, time

ANA = __DNH_ROOT__ + ""
AGAC = __DNH_ROOT__ + "-p11-gpu"
BU = os.path.abspath(__file__)
SURUCU = f"{AGAC}/scripts/b1c_kod_esli_kos.sh"
PAY = __DNH_DATA__ + ""
B1C = f"{PAY}/gpu_haftasi/b1c"

KOL, ET, SEED, M = "B1C_KOD", "DIAL_B1C_KOD", 20260830, 4
N_ISTEK = 3348
YON = "kol"
KOD_KIP = "yonlu"
YIGIN = 16
YUVA_BEKLENEN = 975
REF_YUVA = {"KISI_ASAGI": 844, "RASTGELE": 114}
ESLI_YUVA = {"ESLI": 771, "KISI_ASAGI": 844, "RASTGELE": 114}
ESLI_N, ESLI_ADIM = 7537, 458

B1C_KOL_KOK = f"{B1C}/kol"
KOL_JSONL = f"{B1C_KOL_KOK}/{KOL}.jsonl"
KUNYE = f"{B1C}/B1C_KOL_KUNYE_2026-10-29.json"
B0_CARD = {"veri": f"{B1C}/b0/B0_YONLU_VERIYONU_n3348.json",
           "kol": f"{B1C}/b0/B0_YONLU_KOLYONU_n3348.json",
           "kol_nmax": f"{B1C}/b0/B0_YONLU_KOLYONU_nmax.json"}
VERI_KIMLIK = f"{B1C}/b0/KIMLIK_VERIYONU_n3348.json"
B1C_CKPT, B1C_DS, B1C_HUB = f"{B1C}/dial_ckpt", f"{B1C}/hf_ds_cache", f"{B1C}/dpo_hub"
B1C_PANEL, B1C_GEM = f"{B1C}/c1_panel", f"{B1C}/results"
EGITIM_J = f"{B1C_GEM}/SEED_VARYANS_2026-08-30_{ET}_t{SEED}.json"
CKPT_DIR = f"{B1C_CKPT}/{ET}_t{SEED}"
KD = f"{B1C_GEM}/KOL_DEGER_{ET}_m{M}_2026-08-30.json"
PROVA_CIK = f"{B1C}/b1c_prova_esli.json"
VERDICT_CIK = f"{B1C}/b1c_kod_verdict_2026-10-29.json"
GUC_CIK = f"{B1C}/B1C_GUC_2026-10-29.json"

PAY_PANEL = f"{PAY}/c1_panel"
PAY_MINI = f"{PAY_PANEL}/miniDPO"
PAY_HUB = f"{PAY}/dpo_hub"
TABAN_DIR = f"{PAY_PANEL}/Tulu3-8B/sft"
HEDEF_KOL = f"{PAY}/dial_kol/KISI_ASAGI.jsonl"
REF_KD = {k: f"{ANA}/results/KOL_DEGER_DIAL_{k}_m4_2026-08-30.json" for k in ESLI_YUVA}
ESLI_VERDICT = f"{ANA}/results/matched_arm_verdict_2026-09-11.json"
ESLI_EGITIM = f"{ANA}/results/SEED_VARYANS_2026-08-30_DIAL_ESLI_t20260830.json"
ESLI_KAPI = f"{ANA}/results/matched_arm_gate_2026-09-10.json"
B0B = f"{ANA}/results/b0b_kod_ikili_2026-09-24.json"
B0_SHA = "4c37cb701c82dd3eee03d68cad51f963ab83d83ead3ab64b16e481a075305818"
HF_DS = ("<storage>/huggingface/datasets/allenai___llama-3.1-tulu-3-8b-preference-mixture/default/0.0.0/"
         "78a6f00785946cd24276c5dd075f83a143a3b1e6")
ESLI_COMMITLER = {"egitim_okuma": "44a07121", "verdict": "d0fde8db"}


def _dur(kod, mesaj):
    print(mesaj, flush=True)
    sys.exit(kod)


def _sha(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()


def _git(*a):
    r = subprocess.run(["git", *a], capture_output=True)
    return r.returncode, r.stdout


def _yol():
    cik = {os.path.dirname(BU), f"{AGAC}/scripts", f"{AGAC}/scripts",
           f"{ANA}/scripts", f"{ANA}/scripts"}
    kalan = [p for p in sys.path if not (p and os.path.abspath(p) in cik)]
    sys.path[:] = [f"{ANA}/scripts", f"{ANA}/scripts"] + kalan


def _kaynak(yazdir=True):
    dos = sorted({os.path.abspath(m.__file__) for ad, m in list(sys.modules.items())
                  if getattr(m, "__file__", None) and os.path.abspath(m.__file__) != BU
                  and os.path.abspath(m.__file__).startswith((ANA + "/", AGAC + "/"))})
    yabanci = [f for f in dos if f.startswith(AGAC + "/")]
    if yabanci:
        _dur(2, f"★★★ KOD KAYNAGI KARISIK: worktree'den yüklenen modül {yabanci} ⇒ DUR (tek kaynak ANA)")
    h = hashlib.sha256()
    for f in dos:
        h.update(f.encode()); h.update(_sha(f).encode())
    fp = h.hexdigest()[:16]
    if yazdir:
        _, hd = _git("-C", ANA, "rev-parse", "--short", "HEAD")
        print(f"  [KAYNAK] n_modul={len(dos)} · kök {ANA} · HEAD {hd.decode().strip()} · parmak-izi {fp} · "
              f"adaptör sha16 {_sha(BU)[:16]}", flush=True)
    return fp, dos


def _arg(argv, ad):
    return argv[argv.index(ad) + 1] if ad in argv and argv.index(ad) + 1 < len(argv) else None


def _bekle_arg(argv, beklenen):
    yanlis = {k: (_arg(argv, k), str(v)) for k, v in beklenen.items() if _arg(argv, k) != str(v)}
    if yanlis:
        _dur(2, f"★★★ BAYRAK UYUSMAZLIGI {yanlis} (verilen, beklenen) ⇒ KOSULMADI")


def _kunye(zorunlu=True):
    if not os.path.isfile(KUNYE) or not os.path.isfile(KOL_JSONL):
        if zorunlu:
            _dur(3, f"{KUNYE} {KOL_JSONL}")
        return None
    K = json.load(open(KUNYE, encoding="utf-8"))
    if not K.get("GECTI") or K.get("kol_sha256") != _sha(KOL_JSONL):
        if zorunlu:
            _dur(3, f"{str(K.get('kol_sha256'))[:16]} {K.get('GECTI')}")
        return None
    return K


def _yuva_yollari(y):
    out = []
    for i in range(M):
        k = y + i * 1000
        ad = f"models--project--minidpo-ckpt-{k:03d}"
        out.append((k, f"{B1C_PANEL}/miniDPO/ckpt-{k}", f"{PAY_MINI}/ckpt-{k}", f"{B1C_HUB}/{ad}", f"{PAY_HUB}/{ad}"))
    return out


def _yuva_dolu(y):
    dolu = [p for t in _yuva_yollari(y) for p in t[1:] if os.path.lexists(p)]
    for i in range(M):
        k = y + i * 1000
        for p in (glob.glob(f"{PAY}/gpu_haftasi/*/c1_panel/miniDPO/ckpt-{k}")
                  + glob.glob(f"{PAY}/gpu_haftasi/*/dpo_hub/models--project--minidpo-ckpt-{k:03d}")):
            if p not in dolu:
                dolu.append(p)
    return dolu


def _ozet(v):
    return round(float(v), 6)


def kur():
    K = _kunye(zorunlu=False)
    if K is not None:
        print(f"{K['n_kol']} {K['adim']} {K['kol_sha256'][:16]}",
              flush=True)
        return 0
    if os.path.exists(KOL_JSONL) or os.path.exists(KUNYE):
        _dur(3, f"{KOL_JSONL}")
    t0 = time.time()
    _yol()
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_CACHE"] = B1C_DS
    os.environ["B0_KOD_KIP"] = KOD_KIP
    os.environ["B0_CIK"] = B0_CARD["veri"]
    import numpy as np
    with contextlib.redirect_stdout(sys.stderr):
        import b0_kod_matched_feasibility as B0
        import three_arm as UK
        import person_dial_pair_sagligi as KCS
        import filtre_verim as FV
    if _sha(B0.__file__) != B0_SHA:
        _dur(2, f"{_sha(B0.__file__)[:16]}")
    if B0.HEDEF_KOL != HEDEF_KOL or not B0.CIK.startswith(B1C + "/"):
        _dur(2, f"★★★ B0.HEDEF_KOL {B0.HEDEF_KOL} ≠ {HEDEF_KOL} ya da B0.CIK {B0.CIK} b1c disinda ⇒ DUR")
    fp, _ = _kaynak()
    os.makedirs(f"{B1C}/b0", exist_ok=True)
    D = B0.tara()
    kimlik = D["kimlik"].tolist()
    yer = {k: i for i, k in enumerate(kimlik)}

    H = [json.loads(l) for l in open(HEDEF_KOL, encoding="utf-8")]
    ix = np.array([yer[r["id"]] for r in H])
    s = np.where(np.array([int(r["takas"]) for r in H]) == 1, -1.0, 1.0)
    Dk = {k: np.array(v, copy=True) for k, v in D.items()}
    for k in ("dM1", "dRed", "dKod", "dListe", "dBoy"):
        Dk[k] = Dk[k].astype(float)
        Dk[k][ix] = D[k][ix] * s
    sw = ix[s < 0]
    Dk["njc"][sw], Dk["njr"][sw] = D["njr"][sw], D["njc"][sw]
    f_m1 = float(np.abs(Dk["dM1"][ix] - np.array([r["M1_chosen"] - r["M1_rejected"] for r in H])).max())
    f_jt = int(np.abs((Dk["njc"][ix] - Dk["njr"][ix])
                      - np.array([r["n_jeton_chosen"] - r["n_jeton_rejected"] for r in H])).max())
    G_yon = bool(f_m1 <= 2e-4 and f_jt == 0 and len(set(ix.tolist())) == len(ix))
    print(f"  [PAYDA] b1c_yon: n_hedef={len(H)} · hal_takas={int((s < 0).sum())} · hal_maks_fark_dM1={f_m1:.2e} · "
          f"hal_maks_fark_jeton={f_jt} ⇒ esik dM1 ≤ 2e-4 (kol dosyasi 4 basamak) · jeton = 0 ⇒ "
          f"{'GECTI' if G_yon else 'DÜSTÜ'} ⇒ EYLEM: düserse cikis 3", flush=True)
    if not G_yon:
        _dur(3, "")

    def gecis(Dx, n_istek, cik):
        kayit, asil = [], UK.tabaka_esle

        def kayitli(idx_hedef, idx_kol, tab, rng, n=None, asgari=5):
            r = asil(idx_hedef, idx_kol, tab, rng, n=n, asgari=asgari)
            kayit.append(dict(n=n, sec=r[0], n_max=int(r[1]), ozet=r[2]))
            return r
        eski = (B0.tara, B0.N_ISTENEN, B0.CIK)
        UK.tabaka_esle, B0.tara, B0.N_ISTENEN, B0.CIK = kayitli, (lambda: Dx), n_istek, cik
        try:
            print(f"--- B0.main · n_istek={n_istek} · kip {KOD_KIP} · cikti {cik}", flush=True)
            B0.main()
        finally:
            UK.tabaka_esle = asil
            B0.tara, B0.N_ISTENEN, B0.CIK = eski
        card = json.load(open(cik, encoding="utf-8"))
        sec = [k["sec"] for k in kayit if k["n"] is not None]
        if len(kayit) < 4:
            _dur(4, f"★★★ B0.main beklenen 4 tavan cagrisini yapmadi ({len(kayit)}) ⇒ DUR")
        return card, (np.sort(sec[0]) if sec else None), kayit[3]["n_max"]

    card_v, sec_v, nmax_v = gecis(D, N_ISTEK, B0_CARD["veri"])
    b0b = json.load(open(B0B, encoding="utf-8"))
    G0 = bool(card_v["kod_kip_tavanlar"] == b0b["kod_kip_tavanlar"] and card_v["kisi_sifir"] == b0b["kisi_sifir"]
              and nmax_v == N_ISTEK and sec_v is not None)
    print(f"{card_v['kod_kip_tavanlar']} {b0b['kod_kip_tavanlar']}"
          f"{nmax_v} {N_ISTEK} {'GECTI' if G0 else 'DÜSTÜ'}", flush=True)
    if not G0:
        _dur(3, "")
    card_k, sec_k, nmax_k = gecis(Dk, N_ISTEK, B0_CARD["kol"])
    if sec_k is None:
        print(f"{N_ISTEK} {nmax_k}",
              flush=True)
        card_k2, sec_k, nmax_k2 = gecis(Dk, nmax_k, B0_CARD["kol_nmax"])
        if sec_k is None or nmax_k2 != nmax_k:
            _dur(3, "")
    sec = sec_k if YON == "kol" else sec_v
    N = int(len(sec))

    from datasets import Dataset, concatenate_datasets
    parca = sorted(glob.glob(f"{HF_DS}/*-train-*.arrow"))
    ds = concatenate_datasets([Dataset.from_file(f) for f in parca])
    G_sira = bool(len(ds) == len(kimlik) and list(ds["id"]) == kimlik)
    print(f"{len(parca)} {len(ds)}"
          f"{'BIREBIR' if G_sira else 'FARKLI'}", flush=True)
    if not G_sira:
        _dur(3, "")

    def satirlar(ii):
        return [ds[int(i)] for i in ii]

    def ham_olcu(R):
        c = [FV._son_asistan_mesaj(r["chosen"]) for r in R]
        rj = [FV._son_asistan_mesaj(r["rejected"]) for r in R]
        rc =np.array([KCS.red_var(t) for t in c], float); rr = np.array([KCS.red_var(t) for t in rj], float)
        kc = np.array([KCS.kod_var(t) for t in c], float); kr = np.array([KCS.kod_var(t) for t in rj], float)
        return dict(d_red_ham=_ozet(rc.mean() - rr.mean()), d_kod_ham=_ozet(kc.mean() - kr.mean()),
                    kod_secilen_ham=_ozet(kc.mean()), kod_reddedilen_ham=_ozet(kr.mean()))

    g = D["gecerli"].astype(bool)
    hedef = np.array(sorted(set(int(i) for i in ix if g[i])))

    def denge(ii, Dref):
        o = {f: _ozet(Dref[f][ii].mean()) for f in ("dRed", "dKod", "dListe", "dBoy", "dM1")}
        o["dJeton"] = _ozet((Dref["njc"][ii] - Dref["njr"][ii]).mean())
        o["pay_dKod_neg"] = _ozet((Dref["dKod"][ii] < 0).mean())
        return o
    Rh = [json.loads(l) for l in open(HEDEF_KOL, encoding="utf-8")]
    hedef_kol = dict(denge(hedef, Dk), **ham_olcu(Rh))
    hedef_veri = denge(hedef, D)
    R_k = satirlar(sec_k); R_v = satirlar(sec_v)
    kont_k = dict(denge(sec_k, D), **ham_olcu(R_k))
    kont_v = dict(denge(sec_v, D), **ham_olcu(R_v))
    esli = json.load(open(ESLI_KAPI, encoding="utf-8"))["ulasilan"]
    tablo = {"KISI_ASAGI_kol_yonu": hedef_kol, "KISI_ASAGI_veri_yonu": hedef_veri,
             f"B1C_kol_yonu_n{len(sec_k)}": kont_k, f"KAYITLI_veri_yonu_n{len(sec_v)}": kont_v,
             "ESLI_kayitli_kapi": {"d_red_ham": esli["dred"], "dJeton": esli["djeton"], "dM1": esli["ort_dM1"]}}
    fark = {ad: {k: _ozet(v[k] - hedef_kol[k]) for k in v if k in hedef_kol}
            for ad, v in tablo.items() if ad.startswith(("B1C", "KAYITLI"))}
    print(""
          "", flush=True)
    for ad, v in tablo.items():
        print(f"     {ad:<26} " + " · ".join(f"{k} {x:+.4f}" for k, x in v.items()), flush=True)
    for ad, v in fark.items():
        print(f"     fark {ad:<21} " + " · ".join(f"{k} {x:+.4f}" for k, x in v.items()), flush=True)
    bant = {ad: dict(red=bool(abs(v["d_red_ham"]) <= 0.0097), jeton=bool(abs(v["dJeton"]) <= 5.93))
            for ad, v in fark.items()}
    print(f"  [BETIM] ESLI §5 bandi (|Δred−hedef| ≤ 0,0097 · |Δjeton−hedef| ≤ 5,93) — kapi DEGIL: {bant}", flush=True)

    maks_m1 = float(np.abs(D["dM1"][sec]).max())
    G_notr = bool(maks_m1 == 0.0 and g[sec].all() and len(set(sec.tolist())) == N)
    kes = {}
    for ad in ("KISI_ASAGI", "RASTGELE", "ESLI"):
        ids = {json.loads(l)["id"] for l in open(f"{PAY}/dial_kol/{ad}.jsonl", encoding="utf-8")}
        kes[ad] = int(sum(1 for i in sec if kimlik[int(i)] in ids))
    print(f"  [PAYDA] b1c_kume: n_kol={N} · hal_maks_abs_dM1={maks_m1} · hal_gecerli={int(g[sec].sum())} · "
          f"red_yinelenen={N - len(set(sec.tolist()))} · kesisim {kes} (betim) ⇒ esik maks|ΔM1|=0 ⇒ "
          f"{'GECTI' if G_notr else 'DÜSTÜ'}", flush=True)

    os.makedirs(B1C_KOL_KOK, exist_ok=True)
    tmp = KOL_JSONL + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, r in zip(sec, R_k if YON == "kol" else R_v):
            i = int(i)
            f.write(json.dumps(dict(id=r["id"], source=r["source"], kol=KOL, yon="veri", prompt=r["prompt"],
                                    chosen=r["chosen"], rejected=r["rejected"], takas=0,
                                    dM1_kol=round(float(D["dM1"][i]), 4),
                                    n_jeton_chosen=int(D["njc"][i]), n_jeton_rejected=int(D["njr"][i])),
                               ensure_ascii=False) + "\n")
    os.replace(tmp, KOL_JSONL)
    ksha = _sha(KOL_JSONL)
    json.dump(dict(not_="",
                   kimlik=[kimlik[int(i)] for i in sec_v]), open(VERI_KIMLIK, "w", encoding="utf-8"), ensure_ascii=False)

    with contextlib.redirect_stdout(sys.stderr):
        import gozlem_pilot as GP
        import mini_dpo_feasibility as FZ
        from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FZ.SNAP)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ham = [json.loads(l) for l in open(KOL_JSONL, encoding="utf-8")]
    B = GP.boru_denetimi(tok, ham, 1536, "keep_start")
    dus = float(B["kesme"]["oran_SESSIZ_DUSEN"])
    adim = int(math.floor(N * (1 - dus) / YIGIN))
    G_boru = bool(B["kesme"]["HAL"] == "KABUL" and dus <= float(B["kesme"]["esik"]) and adim > 0)
    e_esli = json.load(open(ESLI_EGITIM, encoding="utf-8"))
    print(f"  ★ ADIM = ⌊{N}·(1−{dus})/{YIGIN}⌋ = {adim} (KK-3F G5 kurali) · ESLI: 458 TASINDI (KK: 7537·(1−0,0273)/16), "
          f"kendi borusu %{100*e_esli['boru']['kesme']['oran_SESSIZ_DUSEN']:.2f} ⇒ rule 463 verirdi · ESLI epok "
          f"{e_esli['egitim']['epoch']:.4f} ↔ B1c ≈ {adim*YIGIN/(N*(1-dus)):.4f}", flush=True)

    KAPI = dict(G0_kayitli_tavanlar_yeniden=G0, G_yon_kol_dosyasi=G_yon, G_satir_sirasi=G_sira,
                G_kisi_notr=G_notr, G_boru=G_boru)
    rapor = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 SINIF="KOL INSASI + DENGE ÖLCÜMÜ — egitim YOK · verdict YOK · prediction YOK",
                 direktif="sahip 29 Ekim 2026: «B1c: kod-blogu-esli kontrol 3 348 ciftle; MDE ve ulasilan güc raporda.»",
                 alet=dict(b0="scripts/b0_kod_matched_feasibility.py @ 2174db4b", b0_sha256=B0_SHA,
                           tabaka="three_arm.tabaka_esle", adaptor=BU, adaptor_sha16=_sha(BU)[:16], kod_parmak_izi=fp),
                 tanim=dict(kod_kip=KOD_KIP, kod_sinifi="ΔKod<0 (KCS._KOD «```», hijyenli metin)",
                            tabaka="digitize(ΔRed, hedef yüzdelik kenarlari) × digitize(Δlog-boy, hedef 4 yüzdelik) × kod",
                            aday="|ΔM1| ≤ p25(havuz) = 0 ⇒ kisi-nötr", seed_b0=B0.SEED, asgari_tabaka=5,
                            yon=YON, yon_gerekce=("KISI_ASAGI satirlarinin %47,7'si takasli; B0 hedefi veri yönünde "
                                                  "okuyordu ⇒ hedef kolun egitildigi yöne cevrildi (B0 gövdesi aynen)")),
                 n_istek=N_ISTEK, n_max={"veri_yonu": nmax_v, "kol_yonu": nmax_k},
                 n_secilen={"veri_yonu_n_istek_3348": int(len(sec_v)), "kol_yonu": int(len(sec_k))},
                 not_n=(""
                        f"{len(sec_v)}"),
                 esli_bandi_betim=band,
                 kip_tavanlari={"veri_yonu": card_v["kod_kip_tavanlar"], "kol_yonu": card_k["kod_kip_tavanlar"]},
                 onculdustu=(None if (nmax_k >= N_ISTEK and len(sec_v) == N_ISTEK) else
                             f"{nmax_k}"
                             f"{len(sec_v)} {N}"),
                 n_kol=N, kol_yolu=KOL_JSONL, kol_sha256=ksha, veri_yonu_kimlik=VERI_KIMLIK,
                 veri_yonu_kimlik_sha256=_sha(VERI_KIMLIK), b0_kartlari=B0_CARD,
                 denge=tablo, fark_hedefe=fark, kesisim=kes,
                 boru=dict(oran_sessiz_dusen=dus, tam=B), adim=adim, yigin=YIGIN,
                 adim_kurali="⌊N·(1 − sessiz düsen)/16⌋ (kk3f_kol.py G5; KK mührü §3: 7 537 → 458)",
                 esli_adim=dict(adim=ESLI_ADIM, n=ESLI_N, kaynak="KK konfigi tasindi (dial_gece_kuyrugu.sh:20)",
                                kendi_boru=e_esli["boru"]["kesme"]["oran_SESSIZ_DUSEN"], epok=e_esli["egitim"]["epoch"]),
                 not_b0_payda=(""
                               ""),
                 k3=(""),
                 kapilar=KAPI, GECTI=all(KAPI.values()), sure_sn=round(time.time() - t0, 1))
    json.dump(rapor, open(KUNYE, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"  [PAYDA] b1c_kur: n_kol={N} · hal_adim={adim} · red_kapi={sum(1 for v in KAPI.values() if not v)} ⇒ "
          f"esik red>0 ⇒ EYLEM: künye GECTI=False, egitim baslamaz", flush=True)
    print(f"✓ {KOL_JSONL} · sha256 {ksha}\n✓ {KUNYE} · {time.time()-t0:.0f} sn", flush=True)
    return 0 if all(KAPI.values()) else 3


def egit(argv):
    K = _kunye()
    _bekle_arg(argv, {"--kol": KOL, "--etiket": ET, "--n-cift": K["n_kol"], "--adim": K["adim"], "--seed": SEED})
    if "--kaydet" not in argv:
        _dur(2, "")
    _yol()
    os.environ["CKPT_KOK"] = B1C_CKPT
    os.environ["KOL_KOK"] = B1C_KOL_KOK
    import seed_variance as TV
    import gozlem_pilot as GP
    ds_erken = "datasets" in sys.modules
    os.environ["HF_DATASETS_CACHE"] = B1C_DS
    if ds_erken:
        import datasets.config as DC
        from pathlib import Path
        DC.HF_DATASETS_CACHE = Path(B1C_DS)
    TV.CKPT = B1C_CKPT
    TV.CIK = f"{B1C_GEM}/SEED_VARYANS_2026-08-30.json"
    os.makedirs(B1C_GEM, exist_ok=True); os.makedirs(B1C_DS, exist_ok=True)
    card = TV.CIK.replace(".json", f"_{ET}_t{SEED}.json")
    if card != EGITIM_J or not TV.CKPT.startswith(B1C + "/") or GP.KOL_KOK != B1C_KOL_KOK or TV.KOL_KOK != B1C_KOL_KOK:
        _dur(2, f"★★★ B1C YÖNLENDIRME DÜSTÜ (egitim): card {card} · ckpt {TV.CKPT} · GP.KOL_KOK {GP.KOL_KOK} ⇒ DUR")
    _kaynak()
    print(f"  [B1C-YÖNLENDIRME · egitim] CKPT {TV.CKPT} · cikti {TV.CKPT}/{ET}_t{SEED} · küme {TV.CKPT}/ds_{KOL}_"
          f"{K['n_kol']} · kol {GP.KOL_KOK}/{KOL}.jsonl (sha {K['kol_sha256'][:16]}) · card {card} · HF_DATASETS_CACHE "
          f"{os.environ['HF_DATASETS_CACHE']} (datasets import'ta {'YÜKLÜYDÜ' if ds_erken else 'yüklü degildi'})", flush=True)
    sys.argv = [TV.__file__] + list(argv)
    return TV.main()


def _okuma_yonlendir():
    with contextlib.redirect_stdout(sys.stderr):
        import mde_evaluate as MD
        import mini_dpo_egri as ME
        import serita_harman102 as SA
        import family_panel as C
    MD.TV = B1C_CKPT
    MD.ROOT = B1C
    ME.DPO_HUB = B1C_HUB
    SA.HF_KOKLERI[:] = [k for k in SA.HF_KOKLERI if os.path.abspath(k) != PAY_HUB] + [B1C_HUB]
    if not getattr(C.uretim_yolu, "_b1c", False):
        asil = C.uretim_yolu

        def uretim_yolu(cift_ad, zemin):
            y = asil(cift_ad, zemin)
            if y.startswith(PAY_MINI + "/"):
                return f"{B1C_PANEL}/miniDPO/" + y[len(PAY_MINI) + 1:]
            if zemin == "instruct":
                _dur(2, f"★★★ B1C YÖNLENDIRME DÜSTÜ: instruct yolu {y} eslenemedi ⇒ paylasilan köke yazim ⇒ DUR")
            return y
        uretim_yolu._b1c = True
        uretim_yolu._asil = asil
        C.uretim_yolu = uretim_yolu
    return MD, ME, SA, C


def oku(argv):
    K = _kunye()
    _bekle_arg(argv, {"--seed": SEED, "--etiket": ET, "--m": M, "--adim": K["adim"]})
    if "--prova" in argv:
        _dur(2, "★ `--prova` bu sarmalayicida kapali (paylasilan yollari basar) ⇒ `kuru` kipini kullan")
    _yol()
    MD, ME, SA, C = _okuma_yonlendir()
    y = MD.hub_no(ET, SEED)
    if y != YUVA_BEKLENEN:
        _dur(2, f"★★★ YUVA {y} ≠ beklenen {YUVA_BEKLENEN} ⇒ DUR")
    dolu = _yuva_dolu(y)
    eksik = [p for p in (f"{CKPT_DIR}/checkpoint-{K['adim']}/config.json", f"{TABAN_DIR}/X_e5.npy",
                         f"{TABAN_DIR}/uretim.jsonl") if not os.path.exists(p)]
    var_card = [p for p in (KD, f"{ANA}/results/KOL_DEGER_{ET}_m{M}_2026-08-30.json") if os.path.exists(p)]
    print(f"{M} {4*M} {len(dolu)} {len(eksik)}"
          f"{len(var_card)}", flush=True)
    if dolu or eksik or var_card:
        _dur(4, f"{dolu} {eksik} {var_card}")
    os.makedirs(f"{B1C_PANEL}/miniDPO", exist_ok=True); os.makedirs(B1C_GEM, exist_ok=True)
    _kaynak()
    C.PANEL = ME.panel_kur(y, f"models--project--minidpo-ckpt-{y:03d}")
    C.OUT_KOK = PAY_PANEL
    print(f"{MD.TV} {MD.TV} {ET} {SEED} {ME.DPO_HUB}"
          f"{SA.HF_KOKLERI} {C.uretim_yolu(f'miniDPO-{y:03d}', 'instruct')}"
          f"{C.uretim_yolu(f'miniDPO-{y:03d}', 'base')} {MD.ROOT}"
          f"{ET} {M}", flush=True)
    sys.argv = [MD.__file__] + list(argv)
    return MD.main()


def _verdict_hazirla():
    _yol()
    with contextlib.redirect_stdout(sys.stderr):
        import person_dial_verdict as KH
        import matched_arm_verdict as MEK
        import addressee_run as MK
        import mde_evaluate as MD
    if MEK.KH is not KH:
        _dur(2, "★★★ m_esli_kol_verdict baska bir dial_verdict nesnesi yükledi ⇒ DUR")
    KH.M = M
    KH.KD_DESEN = "KOL_DEGER_DIAL_{kol}_m4_2026-08-30.json"
    KH.REF_B = "RASTGELE"
    KH.ROOT = B1C
    MK.VERI = B1C_PANEL
    KH.TABAN_KOK = PAY_PANEL
    return KH, MEK, MK, MD


def ciftlik_kur(yuvalar):
    os.makedirs(f"{B1C_PANEL}/miniDPO", exist_ok=True); os.makedirs(B1C_GEM, exist_ok=True)
    n_bag = n_yeni = 0
    for kol, y in yuvalar.items():
        for i in range(M):
            k = y + i * 1000
            hed, bag = f"{PAY_MINI}/ckpt-{k}", f"{B1C_PANEL}/miniDPO/ckpt-{k}"
            if not os.path.isfile(f"{hed}/uretim.jsonl"):
                _dur(4, f"{hed}")
            if os.path.islink(bag):
                if os.readlink(bag) != hed:
                    _dur(4, f"★★★ bag {bag} → {os.readlink(bag)} ≠ {hed} ⇒ DUR")
            elif os.path.exists(bag):
                _dur(4, f"★★★ {bag} bag degil, gercek dizin ⇒ kol-aidiyeti belirsiz ⇒ DUR")
            else:
                os.symlink(hed, bag); n_yeni += 1
            n_bag += 1
        hed, bag = REF_KD[kol], f"{B1C_GEM}/{os.path.basename(REF_KD[kol])}"
        if not os.path.isfile(hed):
            _dur(4, f"★★★ referans card {hed} YOK ⇒ DUR")
        if os.path.islink(bag):
            if os.readlink(bag) != hed:
                _dur(4, f"★★★ bag {bag} yanlis hedef ⇒ DUR")
        elif os.path.exists(bag):
            _dur(4, f"★★★ {bag} bag degil ⇒ DUR")
        else:
            os.symlink(hed, bag); n_yeni += 1
        n_bag += 1
    print(f"  [PAYDA] b1c_ciftlik: n_bag={n_bag} · hal_yeni={n_yeni} · red_bozuk=0 (yuvalar {list(yuvalar.values())} × "
          f"{M} + {len(yuvalar)} card → paylasilan kök, SALT-OKUR)", flush=True)


def _yapraklar(x, on=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _yapraklar(v, f"{on}.{k}" if on else str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _yapraklar(v, f"{on}[{i}]")
    else:
        yield on, x


def _al(x, yol):
    import re
    for p in re.findall(r"[^.\[\]]+|\[\d+\]", yol):
        x = x[int(p[1:-1])] if p.startswith("[") else x[p]
    return x


def _sayi(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _kiyas(kayitli, yeni):
    en, n, uyusmayan = 0.0, 0, []
    for yol, v in _yapraklar(kayitli):
        n += 1
        try:
            w = _al(yeni, yol)
        except (KeyError, IndexError, TypeError):
            uyusmayan.append((yol, v, "YOK")); continue
        if _sayi(v) and _sayi(w):
            nv, nw = math.isnan(float(v)), math.isnan(float(w))
            if nv or nw:
                if nv != nw:
                    uyusmayan.append((yol, v, w))
            else:
                en = max(en, abs(float(v) - float(w)))
        elif v != w:
            uyusmayan.append((yol, v, w))
    return en, n, uyusmayan


def prova_gecerli(fp):
    if not os.path.exists(PROVA_CIK):
        return False, "prova künyesi YOK"
    d = json.load(open(PROVA_CIK, encoding="utf-8"))
    h, mu = d.get("verdict", {}), d.get("prereg", "")
    if h.get("prova") != "YENIDEN ÜRETILDI" or float(h.get("en_buyuk_fark", 1)) != 0.0 or h.get("n_uyusmayan", 1):
        return False, f"prova {h.get('prova')} · fark {h.get('en_buyuk_fark')} · uyusmayan {h.get('n_uyusmayan')}"
    if f"adaptör {_sha(BU)[:16]}" not in mu or f"kod {fp}" not in mu:
        return False, f"prova BAYAT (adaptör/kod degisti) · künye: {mu[:120]}"
    return True, f"prova GECERLI · fark {h['en_buyuk_fark']:.2e} · {h['n_alan']} alan · {d.get('damga_utc')}"


def prova(atla_gecerliyse=False):
    KH, MEK, MK, MD = _verdict_hazirla()
    fp, _ = _kaynak()
    if atla_gecerliyse:
        ok, neden = prova_gecerli(fp)
        if ok:
            print(f"{neden}", flush=True)
            return 0
        print(f"  ★ {neden} ⇒ prova kosuyor", flush=True)
    ciftlik_kur(ESLI_YUVA)
    kayit = json.load(open(ESLI_VERDICT, encoding="utf-8"))

    def kaskad_prova(H, ctx):
        h = MEK.esli_kaskad(H, ctx)
        e1, n1, u1 = _kiyas(kayit["verdict"], h)
        e2, n2, u2 = _kiyas(kayit["kol"], H)
        e3, n3, u3 = _kiyas({"taban_M1": kayit["taban_M1"]}, {"taban_M1": float(MK.olc_toplam(ctx["Vb"])["M1"])})
        en, u = max(e1, e2, e3), u1 + u2 + u3
        print(f"{n1+n2+n3} {n1} {n2} {en:.3e}"
              f"{len(u)} {' ' + str(u[:5]) if u else ''}"
              f"", flush=True)
        return dict(prova=("YENIDEN ÜRETILDI" if (en == 0.0 and not u) else "ÜRETILEMEDI"), en_buyuk_fark=en,
                    n_alan=n1 + n2 + n3, n_uyusmayan=len(u), uyusmayan=[list(map(str, x)) for x in u[:50]],
                    esli=h, kayit=os.path.basename(ESLI_VERDICT))
    KH.KOL_YUVA = dict(ESLI_YUVA)
    KH.KASKAD = kaskad_prova
    KH.PREREG_ADI = f"{_sha(BU)[:16]} {fp} {kayit['prereg']}"
    KH.CIK = PROVA_CIK
    print(f"★ B1C · PROVA · yuvalar {KH.KOL_YUVA} · M={KH.M} · B referansi {KH.REF_B} · MK.VERI {MK.VERI} · KH.ROOT "
          f"{KH.ROOT} · taban {KH.TABAN_KOK} · cikti {KH.CIK}", flush=True)
    KH.main()
    d = json.load(open(KH.CIK, encoding="utf-8"))["verdict"]
    print(f"  ★ PROVA: {d['prova']} · en büyük fark {d['en_buyuk_fark']:.2e} · {d['n_alan']} alan · ESLI dB "
          f"{d['esli']['ESLI']['dB']:+.6f} ciB {d['esli']['ESLI']['ciB']} (kayit {kayit['verdict']['ESLI']['dB']:+.6f})",
          flush=True)
    return 0 if d["prova"] == "YENIDEN ÜRETILDI" else 5


def _prereg():
    y = os.environ.get("B1C_PREREG", "")
    if not y:
        return None, "B1C_PREREG verilmedi"
    y = os.path.abspath(y)
    if not os.path.isfile(y):
        return None, f"{y}"
    d = os.path.dirname(y)
    r1, _ = _git("-C", d, "ls-files", "--error-unmatch", y)
    r2, c = _git("-C", d, "log", "-1", "--format=%h", "--", y)
    r3, _ = _git("-C", d, "diff", "--quiet", "HEAD", "--", y)
    if r1 or not c.strip():
        return None, f"prereg commit'li DEGIL: {y}"
    if r3:
        return None, f"mühürün calisma kopyasi commit'ten FARKLI: {y}"
    metin = open(y, encoding="utf-8").read()
    eksik = [f"{os.path.basename(p)} {_sha(p)[:16]}" for p in (SURUCU, BU) if _sha(p)[:16] not in metin]
    if eksik:
        return None, f"prereg aletlerin sha16'sini tasimiyor (W-3): {eksik}"
    return (os.path.basename(y), c.decode().strip()), "GECTI"


def verdict():
    m, neden = _prereg()
    if m is None:
        _dur(2, f"{neden}")
    K = _kunye()
    KH, MEK, MK, MD = _verdict_hazirla()
    fp, _ = _kaynak()
    ok, neden = prova_gecerli(fp)
    if not ok:
        _dur(5, f"★ {neden} ⇒ verdict kosusu YAPILMAZ (cikis 5)")
    if os.path.exists(VERDICT_CIK):
        _dur(7, f"{VERDICT_CIK}")
    y = MD.hub_no(ET, SEED)
    nu = sum(os.path.isfile(f"{B1C_PANEL}/miniDPO/ckpt-{y + i*1000}/uretim_kunye.json") for i in range(M))
    if y != YUVA_BEKLENEN or nu != M or not os.path.isfile(KD) or not os.path.isfile(EGITIM_J):
        _dur(4, f"{y} {nu} {M}"
                f"{'var' if os.path.isfile(KD) else 'YOK'} {'var' if os.path.isfile(EGITIM_J) else 'YOK'}")
    ciftlik_kur(REF_YUVA)
    KH.KOL_YUVA = {KOL: y, "KISI_ASAGI": REF_YUVA["KISI_ASAGI"], "RASTGELE": REF_YUVA["RASTGELE"]}
    KH.KASKAD = MEK.esli_kaskad
    KH.PREREG_ADI = (f"{m[0]} @ {m[1]} · B1C adaptör {_sha(BU)[:16]} · kod {fp} · kol sha {K['kol_sha256'][:16]} · "
                    f"n={K['n_kol']} · ADIM={K['adim']} · yol matched_arm_verdict.esli_kaskad")
    KH.CIK = VERDICT_CIK
    print(f"★ B1C · VERDICT · yuvalar {KH.KOL_YUVA} · M={KH.M} · B referansi {KH.REF_B} · MK.VERI {MK.VERI} · "
          f"egitim {EGITIM_J} · cikti {KH.CIK}", flush=True)
    KH.main()
    h = json.load(open(KH.CIK, encoding="utf-8"))["verdict"]
    e = json.load(open(ESLI_VERDICT, encoding="utf-8"))["verdict"]
    b = h[KOL]
    print(f"  ★★ VERDICT (dial_verdict + esli_kaskad): {KOL} dB {b['dB']:+.4f} · ciB [{b['ciB'][0]:+.4f}, {b['ciB'][1]:+.4f}] · "
          f"B={b['hal_B']} · A={b['hal_A']} · raf={b['raf']} · betim: KISI_ASAGI {h['KISI_ASAGI']['dB']:+.3f} (kayit "
          f"{e['KISI_ASAGI']['dB']:+.3f}) · ESLI kayit {e['ESLI']['dB']:+.3f} {e['ESLI']['ciB']}", flush=True)
    return 0


def guc():
    _yol()
    with contextlib.redirect_stdout(sys.stderr):
        import person_dial_verdict as KH
    d = json.load(open(ESLI_VERDICT, encoding="utf-8"))["verdict"]
    K = _kunye(zorunlu=False)
    z95, za = 1.959963984540054, 1.6448536269514722

    def phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    sd = {}
    for kol in ("ESLI", "KISI_ASAGI"):
        lo, hi = d[kol]["ciB"]
        sd[f"okuma_{kol}"] = (hi - lo) / (2 * z95)
    sd["seed_DIAL_MDE_B"] = KH.MDE_B / za
    dA = d["KISI_ASAGI"]["dB"]; dE = d["ESLI"]["dB"]
    n_b1c = K["n_kol"] if K else None
    etki = {"a_KISI_ASAGI_tam": dA,
            "b_dogrusal_doz_VARSAYIM_3348_7537": dA * 3348 / 7537,
            "c_ESLI": dE}
    if n_b1c and n_b1c != 3348:
        etki[f"b2_dogrusal_doz_VARSAYIM_{n_b1c}_7537"] = dA * n_b1c / 7537
    tablo = {e: {s: round(phi(abs(v) / sv - za), 4) for s, sv in sd.items()} for e, v in etki.items()}
    mde = {s: round(za * sv, 4) for s, sv in sd.items()}
    print("★ B1c · MDE + ULASILAN GÜC — YALNIZ KAYITLI KARTLAR (matched_arm_verdict_2026-09-11.json · person_dial_verdict.MDE_B)", flush=True)
    print(f"  formül: sd = (ciB_üst − ciB_alt) / (2·{z95:.4f})  [istem-kümeli bootstrap %95 CI → sd] · MDE = {za:.4f}·sd "
          f"(tek-tarafli %95) · güc = Φ(|δ|/sd − {za:.4f})", flush=True)
    for s, sv in sd.items():
        print(f"  sd[{s}] = {sv:.5f} ⇒ MDE {mde[s]:.4f} /1k", flush=True)
    print("  güc (δ × sd):", flush=True)
    for e, v in etki.items():
        print(f"    {e:<38} δ={etki[e]:+.4f} · " + " · ".join(f"{s} {p:.4f}" for s, p in tablo[e].items()), flush=True)
    print(f""
          f"{KH.MDE_B}"
          f""
          f"{KH.BAR}", flush=True)
    O = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="", kaynak=ESLI_VERDICT,
             formul=dict(sd="(ciB_ust - ciB_alt) / (2*1.959964)", mde="1.644854*sd (tek-tarafli %95)",
                         guc="Phi(|delta|/sd - 1.644854)", seed_sd="person_dial_verdict.MDE_B / 1.644854"),
             sd={k: round(v, 6) for k, v in sd.items()}, mde=mde, etki={k: round(v, 6) for k, v in etki.items()},
             guc=tablo, n_b1c=n_b1c, bar=KH.BAR,
             serh=["",
                   "(b) dogrusal-doz VARSAYIMI, ölcülmedi", "güc sifir-karsiti tek-tarafli z; verdict bari/plasebo ayrica"])
    os.makedirs(B1C, exist_ok=True)
    json.dump(O, open(GUC_CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {GUC_CIK}", flush=True)
    return 0


def kuru():
    say = {"n": 0, "red": 0}

    def k(ad, ok, ayr=""):
        say["n"] += 1; say["red"] += (not ok)
        print(f"  {'✓' if ok else '✗'} {ad}{' · ' + ayr if ayr else ''}", flush=True)

    _yol()
    print("— [K0] kol künyesi + küme —", flush=True)
    K = _kunye(zorunlu=False)
    k("künye GECTI + kol sha künyeyle ayni", K is not None, f"{KUNYE}")
    if K:
        n = sum(1 for _ in open(KOL_JSONL, encoding="utf-8"))
        kur_ = math.floor(K["n_kol"] * (1 - K["boru"]["oran_sessiz_dusen"]) / YIGIN)
        k(f"{K['n_kol']}", n == K["n_kol"], f"{n}")
        k(f"ADIM = ⌊N·(1−düsen)/16⌋ = {kur_} (künye {K['adim']})", kur_ == K["adim"] and 0 < K["adim"] < ESLI_ADIM)
        print(f"    küme: n_istek {K['n_istek']} · n_max {K['n_max']} · n_kol {K['n_kol']} · sha {K['kol_sha256'][:16]} · "
              f"ÖNCÜL: {K['onculdustu'] or 'yok'}", flush=True)
    os.environ["CKPT_KOK"] = B1C_CKPT; os.environ["KOL_KOK"] = B1C_KOL_KOK
    with contextlib.redirect_stdout(sys.stderr):
        import seed_variance as TV
        import gozlem_pilot as GP
    ds_erken = "datasets" in sys.modules
    k("egitim: `datasets` seed_variance import'unda yüklenmiyor", not ds_erken)
    k("egitim: TV.CIK import'ta donmus (yeniden atama gerekli)", TV.CIK.startswith(ANA + "/results/"), TV.CIK)
    k("egitim: KOL_KOK → b1c/kol (TV + GP)", TV.KOL_KOK == B1C_KOL_KOK and GP.KOL_KOK == B1C_KOL_KOK, GP.KOL_KOK)
    k("egitim: CKPT_KOK → b1c/dial_ckpt", TV.CKPT == B1C_CKPT, TV.CKPT)
    with contextlib.redirect_stdout(sys.stderr):
        MD, ME, SA, C = _okuma_yonlendir()
        KH, MEK, MK, _ = _verdict_hazirla()
        import mini_dpo_feasibility as FZ, depersonalization16_neutral_run, form_count, full_tarama
    fp, dos = _kaynak()

    print("", flush=True)
    nk, ad_ = (K["n_kol"], K["adim"]) if K else ("?", "?")
    print(f"  egitim : {B1C_CKPT}/ds_{KOL}_{nk}/ · {CKPT_DIR}/checkpoint-{ad_} · {EGITIM_J} · {B1C_DS}/", flush=True)
    y = MD.hub_no(ET, SEED)
    C.OUT_KOK = PAY_PANEL
    ins = []
    for t in _yuva_yollari(y):
        C.PANEL = ME.panel_kur(t[0], f"models--project--minidpo-ckpt-{t[0]:03d}")
        ins.append(C.uretim_yolu(f"miniDPO-{t[0]:03d}", "instruct"))
    base = C.uretim_yolu(f"miniDPO-{y + 3000:03d}", "base")
    print(f"  okuma  : {CKPT_DIR}/checkpoint-{{{y}…{y+3000}}} (bag) · {B1C_HUB}/models--project--minidpo-ckpt-"
          f"{{{y}…{y+3000}}}/ (bag) · {ins[0]} … {ins[-1]} · {KD}", flush=True)
    print(f"  prova  : {B1C_PANEL}/miniDPO/ckpt-{{771…3771, 844…3844, 114…3114}} (SALT-OKUR bag) · {B1C_GEM}/"
          f"KOL_DEGER_DIAL_{{ESLI,KISI_ASAGI,RASTGELE}}_m4_2026-08-30.json (bag) · {PROVA_CIK}", flush=True)
    print(f"  verdict  : {VERDICT_CIK} · güc: {GUC_CIK}", flush=True)
    k("okuma: 4 instruct yolu b1c'ye eslendi", all(p.startswith(B1C_PANEL + "/miniDPO/ckpt-") for p in ins), ins[0])
    k("okuma: taban yolu paylasilan Tulu3-8B/sft (salt-okur; X_e5.npy var)",
      base == TABAN_DIR and os.path.isfile(f"{TABAN_DIR}/X_e5.npy"), base)
    k("",
      PAY_HUB not in SA.HF_KOKLERI and B1C_HUB in SA.HF_KOKLERI)
    k("okuma: MD.TV/MD.ROOT/ME.DPO_HUB → b1c", MD.TV == B1C_CKPT and MD.ROOT == B1C and ME.DPO_HUB == B1C_HUB)
    k("verdict: KH.ROOT/MK.VERI → b1c · KH.TABAN_KOK → paylasilan c1_panel · esli_kaskad ayni KH",
      KH.ROOT == B1C and MK.VERI == B1C_PANEL and KH.TABAN_KOK == PAY_PANEL and MEK.KH is KH)

    print("— [K2] yuva / etiket boslugu —", flush=True)
    k(f"yuva hub_no({ET}) = {y}", y == YUVA_BEKLENEN and y not in ESLI_YUVA.values())
    dolu = _yuva_dolu(y)
    k(f"", not dolu, f"yuvalar {[y + i*1000 for i in range(M)]} · "
      f"red_dolu={len(dolu)}{' ' + str(dolu) if dolu else ''}")
    kartlar = [KD, f"{ANA}/results/KOL_DEGER_{ET}_m{M}_2026-08-30.json",
               f"{AGAC}/results/KOL_DEGER_{ET}_m{M}_2026-08-30.json", EGITIM_J, VERDICT_CIK,
               f"{ANA}/results/SEED_VARYANS_2026-08-30_{ET}_t{SEED}.json"]
    k("",
      not any(os.path.exists(p) for p in kartlar), str([p for p in kartlar if os.path.exists(p)]))
    izler = [p for r in (B1C_CKPT, f"{PAY}/seed_varyans", f"{PAY}/dial_ckpt", f"{PAY}/kk3_ckpt")
             for p in glob.glob(f"{r}/{ET}_t*")]
    k(f"egitim kökü `{ET}_t*` hicbir ckpt kökünde YOK (yarim iz yok)", not izler, str(izler))

    print("— [K3] salt-okur girdiler —", flush=True)
    snap_ok = os.path.isfile(f"{FZ.SNAP}/config.json") and any(f.endswith(".safetensors") for f in os.listdir(FZ.SNAP))
    k("egitim baslangic modeli FZ.SNAP (config + safetensors)", snap_ok, FZ.SNAP)
    with contextlib.redirect_stdout(sys.stderr):
        tsnap = SA._snapshot(None, model_adi=ME.TABAN_SNAP)
    k("", bool(tsnap), tsnap)
    k("taban üretimi (uretim.jsonl · uretim_kunye.json · X_e5.npy)",
      all(os.path.isfile(f"{TABAN_DIR}/{f}") for f in ("uretim.jsonl", "uretim_kunye.json", "X_e5.npy")))
    ref = [f"{PAY_MINI}/ckpt-{v + i*1000}/uretim.jsonl" for v in ESLI_YUVA.values() for i in range(M)]
    k("referans yuvalar 771/844/114 × 4 üretimli", all(os.path.isfile(p) for p in ref), f"{len(ref)} dosya")
    k("referans KOL_DEGER kartlari + ESLI verdict/egitim kartlari",
      all(os.path.isfile(p) for p in list(REF_KD.values()) + [ESLI_VERDICT, ESLI_EGITIM]))
    try:
        from huggingface_hub import try_to_load_from_cache as _c
        e5 = [_c("intfloat/e5-large-v2", f) for f in ("modules.json", "config.json", "model.safetensors",
                                                         "1_Pooling/config.json", "tokenizer.json")]
        k("e5-large-v2 gömücüsü yerel önbellekte (salt-okur denetim)", all(isinstance(p, str) for p in e5))
    except Exception as ex:
        k("e5-large-v2 yerel önbellek denetimi", False, str(ex)[:80])

    print("", flush=True)
    ekstra = [f"{ANA}/scripts/birim.sh", f"{ANA}/scripts/m_esli_kol_kos.py", f"{ANA}/scripts/b0_kod_matched_feasibility.py"]
    sat = []
    for f in dos + ekstra:
        rel = f[len(ANA) + 1:]
        s = _sha(f)
        bas = {}
        for ad, c in [("HEAD", "HEAD")] + list(ESLI_COMMITLER.items()):
            r, b = _git("-C", ANA, "show", f"{c}:{rel}")
            bas[ad] = (hashlib.sha256(b).hexdigest() == s) if r == 0 else None
        ag = f"{AGAC}/{rel}"
        bas["agac"] = os.path.isfile(ag) and _sha(ag) == s
        sat.append((rel, bas))
    n = len(sat)
    say_ = {ad: sum(1 for _, b in sat if b[ad]) for ad in ("HEAD", "agac", "egitim_okuma", "verdict")}
    k(f"{len(dos)}",
      say_["HEAD"] == n, f"{say_['HEAD']}/{n}")
    fark = [r for r, b in sat if not b["egitim_okuma"]]
    print(f"{say_['agac']} {n} {ESLI_COMMITLER['egitim_okuma']}"
          f"{say_['egitim_okuma']} {n} {ESLI_COMMITLER['verdict']} {say_['verdict']} {n}"
          f"{fark[:12]} {' …' if len(fark) > 12 else ''} {len(fark)}", flush=True)
    print(f"    [BETIM] kod parmak-izi {fp} · adaptör sha256 {_sha(BU)} · sürücü sha256 "
          f"{_sha(SURUCU) if os.path.isfile(SURUCU) else 'YOK'}", flush=True)

    print("— [K5] prova / prereg durumu (kapi degil, kosuda zorunlu) —", flush=True)
    r = subprocess.run([sys.executable, "-B", BU, "_fp"], capture_output=True, text=True)
    fp_h = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else "?"
    ok, neden = prova_gecerli(fp_h)
    print(f"    verdict-süreci kod parmak-izi {fp_h}", flush=True)
    print(f"    prova: {neden}{'' if ok else ' ⇒ kosu basinda CPU provasi kosar (~5 dk), düserse cikis 5'}", flush=True)
    mm, neden = _prereg()
    print(f"    prereg: {neden if mm is None else f'{mm[0]} @ {mm[1]} GECTI'}"
          f"{' ⇒ gercek kosu (KIP=tam|verdict) BASLAMAZ' if mm is None else ''}", flush=True)
    print(f"  [PAYDA] b1c_kuru_py: n_kontrol={say['n']} · hal_gecen={say['n'] - say['red']} · red_dusen={say['red']} "
          f"⇒ esik red>0 ⇒ EYLEM: kosu BASLATILMAZ (cikis 2)", flush=True)
    return 2 if say["red"] else 0


def yuva():
    _yol()
    with contextlib.redirect_stdout(sys.stderr):
        import mde_evaluate as MD
    print(MD.hub_no(ET, SEED))
    return 0


def main():
    if len(sys.argv) < 2:
        _dur(2, f"kullanim: {os.path.basename(BU)} kur|egit|oku|prova|prova-kapi|verdict|guc|kuru|yuva|prereg [bayraklar]")
    kip, argv = sys.argv[1], sys.argv[2:]
    if kip == "kur":
        return kur()
    if kip == "egit":
        return egit(argv)
    if kip == "oku":
        return oku(argv)
    if kip == "prova":
        return prova(atla_gecerliyse=False)
    if kip == "prova-kapi":
        return prova(atla_gecerliyse=True)
    if kip == "verdict":
        return verdict()
    if kip == "guc":
        return guc()
    if kip == "kuru":
        return kuru()
    if kip == "yuva":
        return yuva()
    if kip == "_fp":
        _verdict_hazirla()
        print(_kaynak(yazdir=False)[0])
        return 0
    if kip == "prereg":
        m, neden = _prereg()
        print(f"  [PREREG] {neden if m is None else f'{m[0]} @ {m[1]} · GECTI'}", flush=True)
        return 0 if m else 2
    _dur(2, f"bilinmeyen kip {kip}")


if __name__ == "__main__":
    sys.exit(main() or 0)
