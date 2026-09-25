#!/usr/bin/env python3
import json

MIRAS = {
    "payda": dict(
        soru="n = 0 ya da beklenenden kücük mü?",
        kaynak="§8 — «payda sifir ya da beklenenden kücükse sonuc PASS degil HATADIR»"),
    "rejim-kurulamadi": dict(
        soru="null merkezlenmemis mi (|ort|/sd > 1,645)?",
        kaynak=""),
    "tavan-yapisik-null": dict(
        soru="null_p95 tavana yapisik mi (≥ 1 − 1e−9, ya da menzilin üst ucunda)? "
             "Yapisiksa POZITIF DAL ATESLENEMEZ.",
        kaynak="V-7 prereg-1; ★ atlandigi ölcülmüs vaka: KAPI-1(a), W-305"),
    "taban-sifir-degil": dict(
        soru="isareti olmayan bir istatistigi SIFIRDAN mi okuyorum?",
        kaynak="§4 md-10 — pozitif-tanimli istatistigin tabani null MERKEZIDIR"),
    "tavan-olculdu": dict(
        soru="esigin TAVANI ölcüldü mü (ulasilabilir mi)?",
        kaynak="§4 md-8 — tavani ölcülmemis esik SAHTE NULL üretir; W-252"),
    "bar-bicimi": dict(
        soru="menzili sinirli bir istatistikte `ort + zσ` mi kullaniyorum?",
        kaynak="§4 md-9 — ampirik yüzdelik; parametrik bar TASAR"),
    "birim-esligi": dict(
        soru="kiyasladigim iki nicelik AYNI BIRIMDE mi (etki ↔ gürültü)?",
        kaynak="★ W-306 — `sd_ham` ile `ort_norm` kiyaslandi, MDE ~25× sisti"),
    "yokluk-hukmu": dict(
        soru=""
             "",
        kaynak="§8 + W-302 (desen-bicimi) + ★ W-308 (ad-uzayi)"),
    "yer-tutucu": dict(
        soru="",
        kaynak="§8 yer-tutucu sözlesmesi"),
    "kill-hareketli": dict(
        soru="her kill-testin ISTENDIGI KIPTE atesledigi GÖRÜLDÜ mü?",
        kaynak="§7.2 + K-2e"),
    "ad-tutarliligi": dict(
        soru=""
             ""
             "",
        kaynak=""
               ""
               ""),
    "nesne-kunye": dict(
        soru=""
             "",
        kaynak=""
               ""
               ""
               ""
               ""
               ""),
}


ADRES_DESENI = __import__("re").compile(
    r"^[\w./-]+(?::\d+|\s+§\s*[\w.-]+)$")



_AYIR = lambda t: [w for w in __import__("re").split(r"[\W_]+", str(t).lower()) if w]
DURAK = {"bir", "bu", "ve", "ile", "icin", "icin", "olan", "olarak", "gibi",
         "ama", "degil", "degil", "her", "daha", "cok", "cok", "var", "yok",
         "dir", "dir", "de", "da", "ki", "mi", "mi", "ise", "hem", "ya"}

def denetle_nesneler(nesneler):
    N = dict(nesneler or {})
    cok_satir, kisa, yanki = [], [], []
    for ad, k in N.items():
        k = str(k or "")
        if "\n" in k.strip():
            cok_satir.append(ad); continue
        if len(k.strip()) < 20:
            kisa.append(ad); continue
        ad_kok = {w[:5] for w in _AYIR(ad) if w}
        yeni_kel = [w for w in _AYIR(k)
                    if len(w) > 2 and w not in DURAK and w[:5] not in ad_kok]
        if len(yeni_kel) < 3:
            echo.append(ad)
    return {"PAYDA": {"n_nesne": len(N), "red_cok_satir": len(cok_satir),
                      "red_kisa": len(kisa), "red_ad_yankisi": len(echo)},
            "COK_SATIR": cok_satir, "KISA": kisa, "AD_YANKISI": echo,
            "GECTI": not (cok_satir or kisa or echo) and len(N) > 0}


def denetle(prereg_kapilari, gerekmeyen=None, adres=None, adres_zorunlu=True,
            nesneler=None):
    ele = set(prereg_kapilari); muaf = dict(gerekmeyen or {}); adr = dict(adres or {})
    NK = denetle_nesneler(nesneler) if nesneler is not None else None
    kotu = [k for k in ele | set(muaf) | set(adr) if k not in MIRAS]
    eksik = [k for k in MIRAS if k not in ele and k not in muaf]
    adres_yok = sorted(k for k in ele if not str(adr.get(k, "")).strip())
    adres_bozuk = sorted(k for k in ele
                         if str(adr.get(k, "")).strip()
                         and not ADRES_DESENI.match(str(adr[k]).strip()))
    return {"PAYDA": {"n_miras": len(MIRAS), "n_ele_alinan": len(ele),
                      "n_muaf": len(muaf), "n_adresli": len(ele) - len(adres_yok),
                      "red_eksik": len(eksik),
                      "red_adres_yok": len(adres_yok),
                      "red_adres_bozuk": len(adres_bozuk),
                      "red_tanimsiz_anahtar": len(kotu)},
            "EKSIK": eksik, "TANIMSIZ": kotu,
            "ADRES_YOK": adres_yok, "ADRES_BOZUK": adres_bozuk,
            "adresler": adr, "adres_zorunlu": bool(adres_zorunlu),
            "muafiyetler": muaf, "NESNE_KUNYE": NK,
            "GECTI": (not eksik and not kotu
                      and (not adres_zorunlu or (not adres_yok and not adres_bozuk))
                      and (NK is None or NK["GECTI"]))}


def bas(R):
    P = R["PAYDA"]
    print(f"  [PAYDA] miras_kapi: n_miras={P['n_miras']} · ele_alinan={P['n_ele_alinan']} "
          f"· muaf={P['n_muaf']} · red_eksik={P['red_eksik']} "
          f"· red_tanimsiz={P['red_tanimsiz_anahtar']}")
    for k in R["EKSIK"]:
        print(f"   ✘ EKSIK · {k}: {MIRAS[k]['soru']}")
        print(f"        kaynak: {MIRAS[k]['kaynak']}")
    for k in R["TANIMSIZ"]:
        print(f"   ✘ TANIMSIZ ANAHTAR: {k}")
    for k in R.get("ADRES_YOK", []):
        print(f"   ✘ ADRESSIZ · {k}: kapi anildi ama NEREDE gecildigi yazilmadi (W-311)")
    for k in R.get("ADRES_BOZUK", []):
        print(f"{k} {R['adresler'][k]!r}")
    for k, g in R["muafiyetler"].items():
        print(f"   ○ muaf · {k} — {g}")
    NK = R.get("NESNE_KUNYE")
    if NK is not None:
        q = NK["PAYDA"]
        print(f"{q['n_nesne']}"
              f"{q['']} {q['red_kisa']}"
              f"{q['red_ad_yankisi']}")
        for ad in NK["COK_SATIR"] + NK["KISA"] + NK["AD_YANKISI"]:
            print(f"   ✘ KÜNYE KUSURLU · {ad}")
    if R.get("adres_zorunlu") is False:
        print("")
    print("   ⇒ " + ("GECTI" if R["GECTI"] else "KAPANMAZ"))
    return R


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--ornek", action="store_true", help="W-305 vakasini yeniden oynat")
    a = ap.parse_args()
    if a.liste:
        print(json.dumps(MIRAS, ensure_ascii=False, indent=1))
    if a.ornek:
        print("★ W-305 yeniden oynatma — KAPI-1(a) mührünün BEYAN ETTIGI kapilar:")
        bas(denetle(["payda", "rejim-kurulamadi", "yer-tutucu"]))
    if not (a.liste or a.ornek):
        print("★ --liste · --ornek")
