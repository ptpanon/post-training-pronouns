#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import itertools, json, math, sys, os
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda


def izgara(alt, ust, n):
    if n < 2:
        return [float(alt)]
    return [alt + (ust - alt) * i / (n - 1) for i in range(n)]


def tam_tarama(verdict_fn, eksenler, adlar, etiket="tam_tarama", turetilmis=None):
    anahtarlar = list(eksenler)
    ates, ornek = {}, {}
    n_hucre = 0
    for kombo in itertools.product(*(eksenler[k] for k in anahtarlar)):
        kw = dict(zip(anahtarlar, kombo))
        for ad_t, fn_t in (turetilmis or {}).items():
            kw[ad_t] = fn_t(kw)
        r = verdict_fn(**kw)
        ad, neden = (r if isinstance(r, tuple) else (r, ""))
        k = f"{ad}·{neden}" if neden else ad
        ates[k] = ates.get(k, 0) + 1
        ornek.setdefault(k, {a: round(v, 6) if isinstance(v, float) else v
                             for a, v in kw.items()})
        n_hucre += 1
    dogan = {k.split("·")[0] for k in ates}
    dogamayan = sorted(set(adlar) - dogan)
    fazla = sorted(dogan - set(adlar))
    R = dict(
        KIP=("veri-yolu" if turetilmis else "izgara"),
        turetilmis_arglar=sorted(turetilmis or {}),
        n_hucre=n_hucre, n_eksen=len(anahtarlar),
        eksen_boyu={k: len(v) for k, v in eksenler.items()},
        ates_sayilari=dict(sorted(ates.items(), key=lambda x: -x[1])),
        pay={k: round(v / n_hucre, 4) for k, v in ates.items()},
        ilk_ornek=ornek,
        DOGAMAYAN_ADLAR=dogamayan,
        MUHURDE_OLMAYAN_ADLAR=fazla,
        GECTI=(not dogamayan and not fazla),
        _serh=(""
               ""
               ""))
    payda(etiket, n_hucre_taranan=n_hucre, n_ad_muhurde=len(adlar),
          n_ad_dogan=len(dogan), red_dogamayan=len(dogamayan),
          red_muhurde_olmayan=len(fazla))
    return R


def veri_yolu_karsilastir(R_izgara, R_veri, etiket="veri_yolu"):
    gi = {k.split("·")[0] for k in R_izgara["ates_sayilari"]}
    gv = {k.split("·")[0] for k in R_veri["ates_sayilari"]}
    yok = sorted(gi - gv)
    ters = sorted(gv - gi)
    R = dict(n_ad_izgarada=len(gi), n_ad_veride=len(gv),
             IZGARADA_VAR_VERIDE_YOK=yok, VERIDE_VAR_IZGARADA_YOK=ters,
             GECTI=(not yok and not ters),
             _serh=("Izgara bagimsiz süpürür; veri-yolu bagimliligi tasir. "
                    "Fark, K-1'in ölcülmemis yarisidir (W-369)."))
    payda(etiket, n_ad_izgarada=len(gi), n_ad_veride=len(gv),
          red_izgarada_var_veride_yok=len(yok), red_veride_var_izgarada_yok=len(ters))
    return R


def kapi(*raporlar, etiket="tam_tarama_kapisi"):
    dusen = [i for i, R in enumerate(raporlar) if not R.get("GECTI")]
    payda(etiket, n_rapor=len(raporlar), red_dusen_rapor=len(dusen))
    if dusen:
        print(f"   ★ KAPI DÜSTÜ — rapor #{dusen} · cikis 3", flush=True)
        return 3
    print("   ✓ KAPI GECTI — her ad hem izgarada hem VERI YOLUNDA doguyor", flush=True)
    return 0


def bas(R, baslik="TAM TARAMA"):
    print(f"★ {baslik} · {R['n_hucre']} hücre · {R['n_eksen']} eksen")
    for k, v in R["ates_sayilari"].items():
        print(f"   {k:38} {v:6}  %{100*R['pay'][k]:5.1f}   ör. {R['ilk_ornek'][k]}")
    if R["DOGAMAYAN_ADLAR"]:
        print(f"   ★ DOGAMAYAN: {R['DOGAMAYAN_ADLAR']}")
    if R["MUHURDE_OLMAYAN_ADLAR"]:
        print(f"   ★ MÜHÜRDE OLMAYAN: {R['MUHURDE_OLMAYAN_ADLAR']}")
    print(f"   ⇒ {'GECTI' if R['GECTI'] else 'DÜSTÜ'}")


def prova():
    t = {}
    ADLAR = ("A", "B", "C", "D")

    def kusurlu(x, y):
        if y >= x - 0.15: return "D_ONLEYEN", "kapi"
        if x >= 0.35: return "A", ""
        if x <= 0.05: return "D", ""
        return "B", ""

    def saglam(x, y):
        if y >= 0.5: return "D", "kapi"
        if x >= 0.35: return "A", ""
        if x <= 0.05: return "C", ""
        return "B", ""

    ek = dict(x=izgara(0, 1, 21), y=izgara(0, 1, 11))
    R1 = tam_tarama(kusurlu, ek, ADLAR, "prova_kusurlu")
    R2 = tam_tarama(saglam, ek, ADLAR, "prova_saglam")
    t["p1_kusurlu_dogamayani_bulur"] = "D" in R1["DOGAMAYAN_ADLAR"]
    t["p2_kusurlu_fazla_adi_bulur"] = "D_ONLEYEN" in R1["MUHURDE_OLMAYAN_ADLAR"]
    t["p3_kusurlu_dusar"] = R1["GECTI"] is False
    t["p4_saglam_gecer"] = R2["GECTI"] is True
    t["p5_paylar_toplami_1"] = abs(sum(R2["pay"].values()) - 1.0) < 1e-6
    t["p6_hucre_sayisi_carpim"] = R2["n_hucre"] == 21 * 11
    t["p7_uc_noktalar_dahil"] = izgara(0, 1, 3) == [0.0, 0.5, 1.0]
    t["p8_ornek_her_ad_icin"] = set(R2["ilk_ornek"]) == set(R2["ates_sayilari"])

    ADLAR2 = ("SIKISIYOR", "GENISLIYOR", "NULL-AYIRT-EDEMEDI", "FARK-YOK")

    def kaskad(d, frac):
        if -0.02 <= d <= 0.02: return "FARK-YOK", ""
        if frac > 0.05:        return "NULL-AYIRT-EDEMEDI", ""
        if d > 0:              return "SIKISIYOR", ""
        return "GENISLIYOR", ""

    def _frac_tek_yanli(kw):
        return float(0.5 * (1.0 - math.erf(kw["d"] / (0.1 * math.sqrt(2.0)))))

    ek2 = dict(d=izgara(-0.5, 0.5, 41), frac=izgara(0.0, 1.0, 21))
    Ri = tam_tarama(kaskad, ek2, ADLAR2, "prova_izgara_tek_yanli")
    Rv = tam_tarama(kaskad, dict(d=ek2["d"]), ADLAR2, "prova_veri_tek_yanli",
                    turetilmis={"frac": _frac_tek_yanli})
    C = veri_yolu_karsilastir(Ri, Rv, "prova_karsilastir")
    t["p9_izgara_GENISLIYORu_dogurur"] = "GENISLIYOR" in {k.split("·")[0]
                                                          for k in Ri["ates_sayilari"]}
    t["p10_veri_yolu_GENISLIYORu_DOGURMAZ"] = "GENISLIYOR" in Rv["DOGAMAYAN_ADLAR"]
    t["p11_karsilastirma_yakalar"] = C["IZGARADA_VAR_VERIDE_YOK"] == ["GENISLIYOR"]
    t["p12_kapi_cikis_3"] = kapi(C, etiket="prova_kapi_dusen") == 3
    t["p13_kapi_cikis_0"] = kapi(dict(GECTI=True), etiket="prova_kapi_gecen") == 0
    print("[PROVA]", json.dumps(t, ensure_ascii=False))
    payda("prova_tam_tarama", n_kontrol=len(t),
          red_dusen=sum(1 for v in t.values() if not v))
    return t


if __name__ == "__main__":
    t = prova()
    sys.exit(0 if all(t.values()) else 1)
