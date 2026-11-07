#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

VERI_ASIL = __DNH_DATA__ + "/c1_panel/Llama3SimPO"
GECICI = os.environ.get("SIMPO_GECICI", __DNH_DATA__ + "/simpo_verdict_gecici")
CIK = os.environ.get("SIMPO_CIK", f"{ROOT}/results/simpo_second_person_verdict_2026-09-13.json")
ADLAR_1 = ("ZAYIF", "AYNI-ISARET-SimPO", "TERS", "ÖLCÜLEMEZ")
ADLAR_2 = ("AYIRT-EDILEMEZ", "SimPO-DAHA-KÜCÜK", "SimPO-DAHA-BÜYÜK")


def karar1(dM1, ci_alt, ci_ust, null_sd):
    mde = 1.645 * null_sd
    if abs(dM1) < mde:
        return "ZAYIF"
    if dM1 < 0 and ci_ust < 0:
        return "AYNI-ISARET-SimPO"
    if dM1 > 0 and ci_alt > 0:
        return "TERS"
    return "ÖLCÜLEMEZ"


def mutlak_aralik(a, b):
    if a >= 0:
        return (a, b)
    if b <= 0:
        return (-b, -a)
    return (0.0, max(-a, b))


def karar2(d_simpo, ci_simpo, d_dpo, ci_dpo):
    s, d = mutlak_aralik(*ci_simpo), mutlak_aralik(*ci_dpo)
    if s[0] <= d[1] and d[0] <= s[1]:
        return "AYIRT-EDILEMEZ"
    return "SimPO-DAHA-KÜCÜK" if abs(d_simpo) < abs(d_dpo) else "SimPO-DAHA-BÜYÜK"


def dize_suz(R, S):
    on = {(r.get("kol"), r.get("cekim"), r.get("istem_i")): r.get("onek") for r in S}
    return [r for r in R if on.get((r.get("kol"), r.get("cekim"), r.get("istem_i"))) == r.get("onek")]


def raf_mi(d4_taban, d4_uc):
    return d4_taban < 0.60 or d4_uc < 0.60


def prova():
    _S = [dict(kol="a", cekim=0, istem_i=0, onek="x"), dict(kol="a", cekim=0, istem_i=1, onek="y")]
    _R = [dict(kol="a", cekim=0, istem_i=0, onek="x"), dict(kol="a", cekim=0, istem_i=1, onek="Y"),
          dict(kol="b", cekim=0, istem_i=0, onek="x")]
    return {
        "0a_dize_kapisi_farkli_oneki_dusurur": len(dize_suz(_R, _S)) == 1,
        "0b_dize_kapisi_eslesmeyeni_dusurur": dize_suz(_R, _S)[0]["istem_i"] == 0,
        "0c_raf_taban": raf_mi(0.55, 0.75) and raf_mi(0.75, 0.59) and not raf_mi(0.60, 0.60),
        "i_zayif_mde_alti": karar1(-0.5, -0.9, -0.1, 0.5) == "ZAYIF",
        "ii_ayni_isaret": karar1(-3.0, -3.8, -2.2, 0.5) == "AYNI-ISARET-SimPO",
        "iii_ters": karar1(+2.6, +1.8, +3.5, 0.5) == "TERS",
        "iv_mde_ustu_ci_sifirda_olculemez": karar1(-1.2, -2.4, +0.1, 0.5) == "ÖLCÜLEMEZ",
        "v_zayif_ci_ayriksa_bile_once": karar1(-0.6, -0.9, -0.3, 0.5) == "ZAYIF",
        "vi_abs_aralik_negatif": mutlak_aralik(-3.0, -1.0) == (1.0, 3.0),
        "vii_abs_aralik_sifir": mutlak_aralik(-1.0, 2.0) == (0.0, 2.0),
        "viii_ortusen": karar2(-3.0, (-3.5, -2.5), -3.2, (-3.8, -2.6)) == "AYIRT-EDILEMEZ",
        "ix_kucuk": karar2(-1.0, (-1.4, -0.6), -4.0, (-4.6, -3.4)) == "SimPO-DAHA-KÜCÜK",
        "x_buyuk": karar2(-6.0, (-6.6, -5.4), -2.0, (-2.5, -1.5)) == "SimPO-DAHA-BÜYÜK",
        "xi_zit_isaret_buyukluk": karar2(+3.0, (+2.5, +3.5), -1.0, (-1.3, -0.7)) == "SimPO-DAHA-BÜYÜK",
    }


def tarama():
    import full_tarama as TT
    g = [round(x, 3) for x in np.linspace(-5, 5, 41)]
    R1 = TT.tam_tarama(karar1, dict(dM1=g, ci_alt=g, ci_ust=g, null_sd=[0.3, 0.6, 1.2]),
                       ADLAR_1, etiket="simpo_k1_izgara")
    R1v = TT.tam_tarama(lambda dM1, y, null_sd, ci_alt, ci_ust: karar1(dM1, ci_alt, ci_ust, null_sd),
                        dict(dM1=g, y=[0.5, 0.9, 1.5], null_sd=[0.47, 0.6]), ADLAR_1,
                        etiket="simpo_k1_veri_yolu",
                        turetilmis={"ci_alt": lambda k: k["dM1"] - k["y"], "ci_ust": lambda k: k["dM1"] + k["y"]})
    h = [round(x, 2) for x in np.linspace(-6, 6, 25)]
    R2 = TT.tam_tarama(lambda ds, dd, ys, yd, ci_s, ci_d: karar2(ds, ci_s, dd, ci_d),
                       dict(ds=h, dd=h, ys=[0.4, 0.9], yd=[0.4, 0.9]), ADLAR_2, etiket="simpo_k2_veri_yolu",
                       turetilmis={"ci_s": lambda k: (k["ds"] - k["ys"], k["ds"] + k["ys"]),
                                   "ci_d": lambda k: (k["dd"] - k["yd"], k["dd"] + k["yd"])})
    for R, b in ((R1, "k1 izgara"), (R1v, "k1 veri-yolu"), (R2, "k2 veri-yolu")):
        TT.bas(R, f"TAM TARAMA · {b}")
    V = TT.veri_yolu_karsilastir(R1, R1v, etiket="simpo_k1_iz_vy")
    return TT.kapi(R1, R1v, R2, V)


def main():
    P = prova()
    print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False), flush=True)
    if not all(P.values()):
        print("   ⇒ DÜSTÜ ⇒ cikis 4"); return 4
    if "--tarama" in sys.argv:
        return tarama()
    import addressee_run as MK
    import depersonalization16_neutral_run as M9
    import person_dial_verdict as KH
    t0 = time.time()
    veri = f"{GECICI}/veri"
    for ad, uc in (("Llama3SimPO_DPO", "dpo"), ("Llama3SimPO_SimPO", "simpo")):
        os.makedirs(f"{veri}/{ad}", exist_ok=True)
        for b in ("sft", uc):
            y = f"{veri}/{ad}/{b}"
            if not os.path.islink(y):
                os.symlink(f"{VERI_ASIL}/{b}", y)
    os.makedirs(f"{GECICI}/results", exist_ok=True)
    MK.VERI = veri; MK.ROOT = GECICI
    MK.MERDIVEN = {"Llama3SimPO_DPO": ("sft", ["sft", "dpo"], "dpo"),
                   "Llama3SimPO_SimPO": ("sft", ["sft", "simpo"], "simpo")}
    asil_oku, asil_sb = MK.oku, MK.satir_bilesenleri
    KAYIT = {"anahtar": None}; VON = {}; RON = {}; DIZE = {}

    def oku(a, b, kok=None):
        R = asil_oku(a, b, kok)
        if R is None:
            return None
        if b != "sft":
            tut = dize_suz(R, asil_oku(a, "sft", kok))
            DIZE[b] = dict(n_satir=len(R), n_ayni_onek=len(tut), n_dusen=len(R) - len(tut))
            R = tut
        KAYIT["anahtar"] = b; RON[b] = R
        return R

    def sb(nlp, R):
        b = KAYIT["anahtar"]
        if b not in VON:
            VON[b] = asil_sb(nlp, R)
        return VON[b]
    MK.oku, MK.satir_bilesenleri = oku, sb
    MK.kos()
    S = json.load(open(f"{GECICI}/results/prereg8_measurement_2026-09-01.json", encoding="utf-8"))

    d4 = {b: float(KH.distinct4([r["metin"] for r in RON[b]])) for b in RON}
    ist = np.array([r.get("istem_i") for r in RON["sft"]])
    pla = M9.plasebo_esli(VON["sft"], ist)
    O = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             prereg="prereg_simpo_2026-09-10.md @ 20914b5c · NOTICE 23589472 · prediction d6292408",
             protokol="CIPLAK", motor="addressee_run.kos (ithal, yapilandirma) · 200 permütasyon · 1000 bootstrap",
             dize_kapisi=DIZE, distinct4=d4, plasebo_esli_sft=pla, basamak={})
    for anahtar, uc, etiket in (("Llama3SimPO_DPO", "dpo", "SFT→DPO"), ("Llama3SimPO_SimPO", "simpo", "SFT→SimPO")):
        f = S[anahtar]["fark"]["M1"]
        raf = raf_mi(d4["sft"], d4[uc])
        ad = None if raf else karar1(f["gozlenen"], f["ci"][0], f["ci"][1], f["null_sd"])
        O["basamak"][etiket] = dict(
            dM1=f["gozlenen"], ci=f["ci"], null_sd=f["null_sd"], null_ort=f["null_ort"],
            MDE=1.645 * f["null_sd"], frac_null_asan=f["frac_null_asan"],
            ad=ad, hukme_girmez=("DEJENERE-RAF (distinct-4 < 0,60)" if raf else None),
            plasebo_yutuyor=bool(abs(f["gozlenen"]) <= pla["p95"]),
            n_cift=S[anahtar]["n_cift"], n_istem=S[anahtar]["n_istem"],
            dKUVVET=S[anahtar]["fark"]["KUVVET"]["gozlenen"],
            basamak_tablosu=S[anahtar]["basamak_tablosu"])
    bs, bd = O["basamak"]["SFT→SimPO"], O["basamak"]["SFT→DPO"]
    if bs["ad"] is None or bd["ad"] is None:
        O["ikinci_soru"] = dict(ad=None, not_="")
    else:
        O["ikinci_soru"] = dict(ad=karar2(bs["dM1"], bs["ci"], bd["dM1"], bd["ci"]),
                                abs_ci_simpo=mutlak_aralik(*bs["ci"]), abs_ci_dpo=mutlak_aralik(*bd["ci"]),
                                dpo_referans_verebilir=bd["ad"] == "AYNI-ISARET-SimPO")
    O["sure_sn"] = round(time.time() - t0, 1)
    json.dump(O, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for e, b in O["basamak"].items():
        print(f"★★★ {e}: ΔM1 {b['dM1']:+.4f} CI [{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}] · null_sd {b['null_sd']:.4f} "
              f"· MDE {b['MDE']:.4f} ⇒ **{b['ad'] or b['hukme_girmez']}**"
              f"{' · PLASEBO-YUTUYOR' if b['plasebo_yutuyor'] else ''}", flush=True)
    print(f"★★★ ikinci soru ⇒ **{O['ikinci_soru']['ad']}** · dize kapisi {DIZE} · distinct-4 {d4} "
          f"· plasebo p95 {pla['p95']:.4f} · {O['sure_sn']} sn", flush=True)
    payda("simpo_verdict", n_basamak=2, hal_ad_simpo=str(bs["ad"]), hal_ad_dpo=str(bd["ad"]),
          red_dize_dusen=sum(v["n_dusen"] for v in DIZE.values()),
          red_raf=sum(1 for b in O["basamak"].values() if b["hukme_girmez"]))
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
