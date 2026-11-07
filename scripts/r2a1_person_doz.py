#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import collections, contextlib, json, math, os, sys, time

ANA = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ANA}/scripts")
import b1c_kod_matched as B

KOK = f"{B.PAY}/gpu_haftasi/r2a1"
A1, A2, ET_A2, YUVA_A2 = "R2_A1", "B1C_KOD", "DIAL_B1C_KOD", 975
A2_KOK = f"{B.PAY}/gpu_haftasi/b1c"
A2_JSONL = f"{A2_KOK}/kol/B1C_KOD.jsonl"
A2_SHA = "38f299034147bb88"
A2_CARD = f"{ANA}/results/b1c_kod_verdict_2026-10-29.json"
N = 3346
ADLAR = ("IKISI-AYRIK", "YALNIZ-A2", "YALNIZ-A1", "IKISI-DE-DEGIL", "A1-TERS", "OLCULEMEZ")

B.BU, B.SURUCU = os.path.abspath(__file__), f"{ANA}/scripts/r2a1_kisi_doz_kos.sh"
B.B1C, B.KOL, B.ET, B.N_ISTEK = KOK, A1, "DIAL_R2_A1", N
B.YUVA_BEKLENEN = 531
B.B1C_KOL_KOK = f"{KOK}/kol"
B.KOL_JSONL = f"{B.B1C_KOL_KOK}/{A1}.jsonl"
B.KUNYE = f"{KOK}/r2a1_arm_manifest_2026-11-06.json"
B.B0_CARD = {"kol_nmax": f"{KOK}/b0/B0_YONLU_KOLYONU_n{N}.json"}
B.B1C_CKPT, B.B1C_DS, B.B1C_HUB = f"{KOK}/dial_ckpt", f"{KOK}/hf_ds_cache", f"{KOK}/dpo_hub"
B.B1C_PANEL, B.B1C_GEM = f"{KOK}/c1_panel", f"{KOK}/results"
B.EGITIM_J = f"{B.B1C_GEM}/SEED_VARYANS_2026-08-30_{B.ET}_t{B.SEED}.json"
B.CKPT_DIR = f"{B.B1C_CKPT}/{B.ET}_t{B.SEED}"
B.KD = f"{B.B1C_GEM}/KOL_DEGER_{B.ET}_m{B.M}_2026-08-30.json"
B.PROVA_CIK = f"{KOK}/r2a1_prova_esli.json"
B.VERDICT_CIK = f"{KOK}/R2A1_VERDICT_2026-11-06.json"
B.GUC_CIK = f"{KOK}/R2A1_GUC_YOK.json"


def _ozet(v):
    return round(float(v), 6)


def kur():
    K = B._kunye(zorunlu=False)
    if K is not None:
        print(f"{K['n_kol']} {K['adim']} {K['kol_sha256'][:16]}", flush=True)
        return 0
    if os.path.exists(B.KOL_JSONL) or os.path.exists(B.KUNYE):
        B._dur(3, f"{B.KOL_JSONL}")
    t0 = time.time()
    B._yol()
    os.makedirs(f"{KOK}/b0", exist_ok=True)
    os.environ.update(HF_DATASETS_OFFLINE="1", HF_DATASETS_CACHE=B.B1C_DS, B0_KOD_KIP=B.KOD_KIP, B0_CIK=B.B0_CARD["kol_nmax"])
    import numpy as np
    with contextlib.redirect_stdout(sys.stderr):
        import b0_kod_matched_feasibility as B0
        import three_arm as UK
        import person_dial_pair_sagligi as KCS
        import filtre_verim as FV
    if B._sha(B0.__file__) != B.B0_SHA:
        B._dur(2, f"{B._sha(B0.__file__)[:16]}")
    fp, _ = B._kaynak()
    D = B0.tara()
    kimlik = D["kimlik"].tolist(); yer = {k: i for i, k in enumerate(kimlik)}
    H = [json.loads(l) for l in open(B.HEDEF_KOL, encoding="utf-8")]
    ix = np.array([yer[r["id"]] for r in H])
    s = np.where(np.array([int(r["takas"]) for r in H]) == 1, -1.0, 1.0)
    Dk = {k: np.array(v, copy=True) for k, v in D.items()}
    for k in ("dM1", "dRed", "dKod", "dListe", "dBoy"):
        Dk[k] = Dk[k].astype(float); Dk[k][ix] = D[k][ix] * s
    sw = ix[s < 0]
    Dk["njc"][sw], Dk["njr"][sw] = D["njr"][sw], D["njc"][sw]
    f_m1 = float(np.abs(Dk["dM1"][ix] - np.array([r["M1_chosen"] - r["M1_rejected"] for r in H])).max())
    f_jt = int(np.abs((Dk["njc"][ix] - Dk["njr"][ix]) - np.array([r["n_jeton_chosen"] - r["n_jeton_rejected"] for r in H])).max())
    G_yon = bool(f_m1 <= 2e-4 and f_jt == 0 and len(set(ix.tolist())) == len(ix))
    print(f"  [PAYDA] r2a1_yon: n_hedef={len(H)} · hal_maks_fark_dM1={f_m1:.2e} · hal_maks_fark_jeton={f_jt} ⇒ esik dM1 ≤ 2e-4 · jeton = 0 ⇒ {'GECTI' if G_yon else 'DÜSTÜ'}", flush=True)
    if not G_yon:
        B._dur(3, "")
    kayit, asil = [], UK.tabaka_esle

    def kayitli(idx_hedef, idx_kol, tab, rng, n=None, asgari=5):
        r = asil(idx_hedef, idx_kol, tab, rng, n=n, asgari=asgari)
        kayit.append(dict(n=n, sec=r[0], n_max=int(r[1]), ozet=r[2], tab=tab, hedef=idx_hedef))
        return r
    eski = (B0.tara, B0.N_ISTENEN, B0.CIK)
    UK.tabaka_esle, B0.tara, B0.N_ISTENEN, B0.CIK = kayitli, (lambda: Dk), N, B.B0_CARD["kol_nmax"]
    try:
        B0.main()
    finally:
        UK.tabaka_esle = asil; B0.tara, B0.N_ISTENEN, B0.CIK = eski
    son = [k for k in kayit if k["n"] is not None][-1]
    tab, hedef, sec_a2 = son["tab"], son["hedef"], np.sort(son["sec"])
    ids_a2 = [json.loads(l)["id"] for l in open(A2_JSONL, encoding="utf-8")]
    G_a2 = bool(B._sha(A2_JSONL)[:16] == A2_SHA and {kimlik[int(i)] for i in sec_a2} == set(ids_a2))
    print(f"{len(ids_a2)} {len(sec_a2)} {G_a2} {B._sha(A2_JSONL)[:16]}", flush=True)
    if not G_a2:
        B._dur(3, "")
    sec_a1, _, _ = asil(hedef, hedef, tab, np.random.default_rng(B.SEED), n=N)
    sec_a1 = np.sort(sec_a1)
    c1, c2 = collections.Counter(tab[sec_a1].tolist()), collections.Counter(tab[sec_a2].tolist())
    G_tab = bool(c1 == c2 and len(sec_a1) == len(sec_a2) and len(set(sec_a1.tolist())) == len(sec_a1))
    print(f"  [PAYDA] r2a1_tabaka: n_A1={len(sec_a1)} · n_A2={len(sec_a2)} · n_tabaka={len(c1)} · red_tabaka_farki={sum(1 for k in set(c1) | set(c2) if c1[k] != c2[k])} ⇒ esik 0 ⇒ EYLEM: >0 ise cikis 3", flush=True)
    if not G_tab:
        B._dur(3, "")
    satir = {r["id"]: r for r in H}
    R1 = [satir[kimlik[int(i)]] for i in sec_a1]
    os.makedirs(B.B1C_KOL_KOK, exist_ok=True)
    tmp = B.KOL_JSONL + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in R1:
            f.write(json.dumps(dict(r, kol=A1), ensure_ascii=False) + "\n")
    os.replace(tmp, B.KOL_JSONL)
    ksha = B._sha(B.KOL_JSONL)
    R2 = [json.loads(l) for l in open(A2_JSONL, encoding="utf-8")]

    def denge(ii, Dref, R):
        o = {f: _ozet(Dref[f][ii].mean()) for f in ("dRed", "dKod", "dListe", "dBoy", "dM1")}
        o["dJeton"] = _ozet((Dref["njc"][ii] - Dref["njr"][ii]).mean())
        o["pay_dKod_neg"] = _ozet((Dref["dKod"][ii] < 0).mean())
        c = [FV._son_asistan_mesaj(r["chosen"]) for r in R]; rj = [FV._son_asistan_mesaj(r["rejected"]) for r in R]
        o["d_red_ham"] = _ozet(np.mean([KCS.red_var(t) for t in c]) - np.mean([KCS.red_var(t) for t in rj]))
        o["d_kod_ham"] = _ozet(np.mean([KCS.kod_var(t) for t in c]) - np.mean([KCS.kod_var(t) for t in rj]))
        return o
    g = D["gecerli"].astype(bool)
    tablo = {"KISI_ASAGI_7537": denge(np.array(sorted(set(int(i) for i in ix if g[i]))), Dk, H), "A1": denge(sec_a1, Dk, R1), "A2_B1C": denge(sec_a2, D, R2)}
    print("  ★ DENGE (egitilen yönde):", flush=True)
    for ad, v in tablo.items():
        print(f"     {ad:<16} " + " · ".join(f"{k} {x:+.4f}" for k, x in v.items()), flush=True)
    kes = len({kimlik[int(i)] for i in sec_a1} & set(ids_a2))
    with contextlib.redirect_stdout(sys.stderr):
        import gozlem_pilot as GP
        import mini_dpo_feasibility as FZ
        from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FZ.SNAP)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    Bo = GP.boru_denetimi(tok, [json.loads(l) for l in open(B.KOL_JSONL, encoding="utf-8")], 1536, "keep_start")
    dus = float(Bo["kesme"]["oran_SESSIZ_DUSEN"])
    adim = int(math.floor(N * (1 - dus) / B.YIGIN))
    G_boru = bool(Bo["kesme"]["HAL"] == "KABUL" and dus <= float(Bo["kesme"]["esik"]) and 0 < adim < B.ESLI_ADIM)
    print(f"  ★ ADIM = ⌊{N}·(1−{dus})/{B.YIGIN}⌋ = {adim} (kayitli rule; A2 = 206)", flush=True)
    KAPI = dict(G_yon_kol_dosyasi=G_yon, G_a2_yeniden_kuruldu=G_a2, G_tabaka_ayni=G_tab, G_boru=G_boru)
    rapor = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), SINIF="KOL INSASI + DENGE ÖLCÜMÜ — egitim YOK · verdict YOK",
                 direktif="sahip/Assistant «REBUTTAL-2» A1", n_istek=N, n_max=dict(a2_kol_yonu=int(son["n_max"])), onculdustu=f"«3 348» tavandir; A2 kurulabilir n = {N} (B1c ERRATA)",
                 alet=dict(b0_sha256=B.B0_SHA, adaptor=B.BU, adaptor_sha16=B._sha(B.BU)[:16], b1c_sha16=B._sha(B.__file__)[:16], kod_parmak_izi=fp),
                 tanim=dict(hedef=B.HEDEF_KOL, tabaka="B1c yönlü kipi: digitize(ΔRed) × digitize(Δlog-boy) × (ΔKod<0), kenarlar hedefin yüzdelikleri",
                            secim="three_arm.tabaka_esle(hedef, hedef, tab, default_rng(20260830), n=3346)", satir="KISI_ASAGI aynen, yalniz kol adi"),
                 n_kol=len(R1), n_a1_a2_ortak_kimlik=kes, kol_yolu=B.KOL_JSONL, kol_sha256=ksha, denge=tablo,
                 boru=dict(oran_sessiz_dusen=dus, tam=Bo), adim=adim, yigin=B.YIGIN, adim_kurali="⌊N·(1 − sessiz düsen)/16⌋ (kk3f_kol.py G5)",
                 a2=dict(kol=A2, kol_sha=A2_SHA, adim=206, n=N, prereg="b88d4993", sonuc="5d5d14a8", card=A2_CARD),
                 kapilar=KAPI, GECTI=all(KAPI.values()), sure_sn=round(time.time() - t0, 1))
    json.dump(rapor, open(B.KUNYE, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"  [PAYDA] r2a1_kur: n_kol={len(R1)} · hal_adim={adim} · red_kapi={sum(1 for v in KAPI.values() if not v)} ⇒ esik red>0 ⇒ EYLEM: künye GECTI=False, egitim baslamaz", flush=True)
    print(f"✓ {B.KOL_JSONL} · sha256 {ksha}\n✓ {B.KUNYE} · {time.time() - t0:.0f} sn", flush=True)
    return 0 if all(KAPI.values()) else 3


def durum(ci):
    return "negatif" if ci[1] < 0 else ("pozitif" if ci[0] > 0 else "kapsar")


def karar(a1, a2, raf):
    if raf:
        return "OLCULEMEZ"
    if a1 == "pozitif":
        return "A1-TERS"
    if a1 == "negatif":
        return "IKISI-AYRIK" if a2 != "kapsar" else "YALNIZ-A1"
    return "YALNIZ-A2" if a2 != "kapsar" else "IKISI-DE-DEGIL"


def adlar():
    import full_tarama as TT
    D = ("negatif", "pozitif", "kapsar")
    R = TT.tam_tarama(lambda a1, a2, raf: karar(a1, a2, raf), dict(a1=D, a2=D, raf=(False, True)), ADLAR, etiket="r2a1_tam_tarama")
    a2 = durum(json.load(open(A2_CARD, encoding="utf-8"))["kol"][A2]["ciB"])
    Rv = TT.tam_tarama(lambda a1, raf: karar(a1, a2, raf), dict(a1=D, raf=(False, True)), [a for a in ADLAR if karar("negatif", a2, False) == a or karar("kapsar", a2, False) == a or a in ("A1-TERS", "OLCULEMEZ")], etiket="r2a1_veri_yolu")
    TT.bas(R, "IZGARA"); TT.bas(Rv, f"VERI YOLU (A2 kayitli: {a2})")
    print(f"{sorted(Rv['ates_sayilari'])} {sorted(set(ADLAR) - set(Rv['ates_sayilari']))}", flush=True)
    return TT.kapi(R, Rv)


def _a2_bagla():
    os.makedirs(f"{B.B1C_PANEL}/miniDPO", exist_ok=True); os.makedirs(B.B1C_GEM, exist_ok=True)
    for i in range(B.M):
        k = YUVA_A2 + i * 1000
        hed, bag = f"{A2_KOK}/c1_panel/miniDPO/ckpt-{k}", f"{B.B1C_PANEL}/miniDPO/ckpt-{k}"
        if not os.path.isfile(f"{hed}/uretim.jsonl"):
            B._dur(4, f"{hed}")
        if not os.path.lexists(bag):
            os.symlink(hed, bag)
        elif os.path.realpath(bag) != os.path.realpath(hed):
            B._dur(4, f"★★★ bag {bag} yanlis hedef ⇒ DUR")
    hed, bag = f"{A2_KOK}/results/KOL_DEGER_{ET_A2}_m4_2026-08-30.json", f"{B.B1C_GEM}/KOL_DEGER_{ET_A2}_m4_2026-08-30.json"
    if not os.path.lexists(bag):
        os.symlink(hed, bag)


def verdict(prova=False):
    m, neden = B._prereg()
    if m is None and not prova:
        B._dur(2, f"{neden}")
    K = B._kunye()
    KH, MEK, MK, MD = B._verdict_hazirla()
    fp, _ = B._kaynak()
    ok, neden = B.prova_gecerli(fp)
    if not ok:
        B._dur(5, f"★ {neden} ⇒ verdict kosusu YAPILMAZ (cikis 5)")
    with contextlib.redirect_stdout(sys.stderr):
        import depersonalization16_neutral_run as M9
        import person_dial_pair_sagligi as KCS
    cik = f"{KOK}/R2A1_VERDICT_YOL_PROVASI.json" if prova else B.VERDICT_CIK
    if os.path.exists(cik) and not prova:
        B._dur(7, f"{cik}")
    y = MD.hub_no(B.ET, B.SEED)
    nu = sum(os.path.isfile(f"{B.B1C_PANEL}/miniDPO/ckpt-{y + i * 1000}/uretim_kunye.json") for i in range(B.M))
    if not prova and (y != B.YUVA_BEKLENEN or nu != B.M or not os.path.isfile(B.KD) or not os.path.isfile(B.EGITIM_J)):
        B._dur(4, f"{y} {nu} {B.M}")
    B.ciftlik_kur(B.REF_YUVA); _a2_bagla()
    kol1 = "KISI_ASAGI" if prova else A1
    a2k = json.load(open(A2_CARD, encoding="utf-8"))["kol"][A2]

    def kaskad(H, ctx):
        h = MEK.esli_kaskad(H, ctx)
        KOLD, Vb, Cb, istb = ctx["KOL"], ctx["Vb"], ctx["Cb"], ctx["istb"]
        fark = max(abs(H[A2]["dB"] - a2k["dB"]), abs(H[A2]["dA"] - a2k["dA"]), *(abs(p - q) for p, q in zip(list(H[A2]["ciB"]) + list(H[A2]["ciA"]), a2k["ciB"] + a2k["ciA"])))
        print(f"  [PAYDA] r2a1_a2_esitlik: en_buyuk_fark={fark:.2e} ⇒ esik 0 ⇒ EYLEM: >0 ise ALET-KAYDI, cikis 7", flush=True)
        if fark != 0.0:
            B._dur(7, "")
        E = {}
        for kol in dict.fromkeys((kol1, A2, "KISI_ASAGI")):
            pl1 = M9.plasebo_esli(KOLD[kol]["V"][0], KOLD[kol]["ist"][0], n=KH.PLASEBO_N, olc=M9.sahis1_1k)
            E[kol] = dict(dM1_B=H[kol]["dB"], ci_M1_B=list(H[kol]["ciB"]), dM1_A=H[kol]["dA"], ci_M1_A=list(H[kol]["ciA"]), plasebo_M1_p95=H[kol]["plasebo_p95"],
                          d1st_B=H[kol]["dSAHIS1"], ci_1st_B=list(KH.boot_kol(KOLD, Vb, Cb, istb, kol, KH.REF_B, "SAHIS1")),
                          d1st_A=H[kol]["dSAHIS1_A"], ci_1st_A=list(KH.boot_kol(KOLD, Vb, Cb, istb, kol, "SFT", "SAHIS1")), plasebo_1st_p95=pl1["p95"], distinct4=H[kol]["distinct4"])
        dC = KH.delta_kol(KOLD, Vb, Cb, kol1, A2)
        kont = dict(dM1=dC["M1"], ci_M1=list(KH.boot_kol(KOLD, Vb, Cb, istb, kol1, A2, "M1")), d1st=dC["SAHIS1"], ci_1st=list(KH.boot_kol(KOLD, Vb, Cb, istb, kol1, A2, "SAHIS1")),
                    pay_A2_bolu_A1=(H[A2]["dB"] / H[kol1]["dB"]) if H[kol1]["dB"] else None)
        Rb = None
        eski = MK.VERI
        MK.VERI = KH.TABAN_KOK; Rb = MK.oku(*KH.TABAN); MK.VERI = eski

        def prof(R):
            t = [r["metin"] for r in R]
            return dict(red=float(sum(KCS.red_var(x) for x in t) / len(t)), kod=float(sum(KCS.kod_var(x) for x in t) / len(t)),
                        liste=float(sum(KCS.liste_payi(x) for x in t) / len(t)), kar=float(sum(len(x or "") for x in t) / len(t)))
        tb = prof(Rb); P = {}
        for kol, yv in KH.KOL_YUVA.items():
            ps = [prof(MK.oku("miniDPO", f"ckpt-{yv + i * 1000}")) for i in range(B.M)]
            ort = {k: float(sum(p[k] for p in ps) / B.M) for k in tb}
            P[kol] = dict(ort=ort, delta_SFT={k: ort[k] - tb[k] for k in tb}, tekil=ps)
        raf = bool(H[kol1]["raf"] or H[A2]["raf"])
        ad = karar(durum(H[kol1]["ciB"]), durum(H[A2]["ciB"]), raf)
        print(f"  ★★ {'YOL-PROVASI (verdict degil)' if prova else 'R2-A1'} · ad {ad} · {kol1} dB {H[kol1]['dB']:+.3f} {E[kol1]['ci_M1_B']} · A2 dB {H[A2]['dB']:+.3f} · fark {kont['dM1']:+.3f} {kont['ci_M1']} · Δ1st {E[kol1]['d1st_B']:+.3f} ↔ {E[A2]['d1st_B']:+.3f}", flush=True)
        return dict(ad=ad, esli_kaskad=h, okuma=E, kontrast_A1_A2=kont, profil=dict(taban_SFT=tb, kol=P), a2_kayit_esitlik=fark)
    KH.KOL_YUVA = {kol1: (B.REF_YUVA["KISI_ASAGI"] if prova else y), A2: YUVA_A2, "KISI_ASAGI": B.REF_YUVA["KISI_ASAGI"], "RASTGELE": B.REF_YUVA["RASTGELE"]}
    KH.KASKAD = kaskad
    KH.PREREG_ADI = ("YOL PROVASI · verdict degil" if prova else f"{m[0]} @ {m[1]}") + f" · adaptör {B._sha(B.BU)[:16]} · kod {fp} · kol sha {K['kol_sha256'][:16]} · n={K['n_kol']} · ADIM={K['adim']}"
    KH.CIK = cik
    print(f"★ R2A1 · VERDICT · yuvalar {KH.KOL_YUVA} · M={KH.M} · B referansi {KH.REF_B} · cikti {KH.CIK}", flush=True)
    KH.main()
    return 0


def main():
    if len(sys.argv) < 2:
        B._dur(2, f"kullanim: {os.path.basename(B.BU)} kur|egit|oku|prova|prova-kapi|verdict|kuru|yuva|prereg|_fp|adlar")
    kip, argv = sys.argv[1], sys.argv[2:]
    if kip == "kur":
        return kur()
    if kip == "verdict":
        return verdict()
    if kip == "verdict-prova":
        return verdict(prova=True)
    if kip == "adlar":
        return adlar()
    if kip in ("egit", "oku"):
        return getattr(B, kip)(argv)
    if kip == "prova":
        return B.prova(atla_gecerliyse=False)
    if kip == "prova-kapi":
        return B.prova(atla_gecerliyse=True)
    if kip == "kuru":
        return B.kuru()
    if kip == "yuva":
        return B.yuva()
    if kip == "_fp":
        B._verdict_hazirla(); print(B._kaynak(yazdir=False)[0]); return 0
    if kip == "prereg":
        m, neden = B._prereg()
        print(f"  [PREREG] {neden if m is None else f'{m[0]} @ {m[1]} · GECTI'}", flush=True)
        return 0 if m else 2
    B._dur(2, f"bilinmeyen kip {kip}")


if __name__ == "__main__":
    sys.exit(main() or 0)
