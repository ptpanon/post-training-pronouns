#!/usr/bin/env python3
import re, sys, tempfile
import fitz

KENAR_SOL, KENAR_SAG = 0.15, 0.83


def olc(pdf):
    d = fitz.open(pdf)
    say = dict(vekalet_dipnot=0, verdict_kenar=0, mavi_karakter=0, borc_listesi=0, eksik_kutu=0)
    ornek = {}
    for n, pg in enumerate(d, start=1):
        W = pg.rect.width
        metin = pg.get_text("text")
        duz = " ".join(metin.split())
        for ad, rx in (("vekalet_dipnot", r"VEK[ÂA]LET"),
                       ("borc_listesi", r"BOR[CC]\s*L[II]?STES[II]?|what this draft still owes"),
                       ("eksik_kutu", r"\[\s*(?:EKS[II]?K|SAH[II]?P\s+SE[CC]ECEK)\s*\]")):
            k = len(re.findall(rx, duz))
            if k:
                say[ad] += k; ornek.setdefault(ad, f"s.{n}")
        for b in pg.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    t = s["text"].strip()
                    if not t:
                        continue
                    c = s["color"]; r, g, bl = (c >> 16) & 255, (c >> 8) & 255, c & 255
                    if r <= 30 and g <= 30 and bl >= 100:
                        say["mavi_karakter"] += len(t); ornek.setdefault("mavi_karakter", f"s.{n}: {t[:40]}")
                    kenar = s["bbox"][0] > W * KENAR_SAG or s["bbox"][2] < W * KENAR_SOL
                    if kenar and re.search(r"Mon|TT|Courier|Mono", s["font"]):
                        say["verdict_kenar"] += 1; ornek.setdefault("verdict_kenar", f"s.{n}: {t[:40]}")
    return say, ornek, len(d)


def sentetik_pozitif():
    d = fitz.open(); pg = d.new_page(width=612, height=792)
    pg.insert_text((72, 100), "Body text in blue", color=(0, 0, 140 / 255), fontname="helv", fontsize=10)
    pg.insert_text((540, 200), "VERDICT_TAG", fontname="cour", fontsize=6)
    pg.insert_text((72, 300), "BORC LISTESI --- what this draft still owes", fontname="helv", fontsize=10)
    pg.insert_text((72, 320), "[EKSK] a number still owed", fontname="helv", fontsize=10)
    pg.insert_text((72, 700), "IMZALI-VEKALET --- bu paragraf taslaktan girdi", fontname="helv", fontsize=7)
    y = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name; d.save(y); return y


def main():
    if sys.argv[1] == "--prova":
        taslak, gonderim = sys.argv[2], sys.argv[3]
        st, _, _ = olc(taslak); sg, _, _ = olc(gonderim); ss, _, _ = olc(sentetik_pozitif())
        p = {"i_gonderim_temiz": sum(sg.values()) == 0,
             "ii_taslakta_atesler": (st["mavi_karakter"] > 0 and st["verdict_kenar"] > 0 and st["vekalet_dipnot"] > 0
                                     and st["borc_listesi"] > 0),
             "iii_sentetikte_bes_sinif_atesler": all(v > 0 for v in ss.values())}
        print(f"★ KAPI-26 PROVA · taslak {st} · gönderim {sg} · sentetik {ss} ⇒ {p}")
        return 0 if all(p.values()) else 4
    say, ornek, n = olc(sys.argv[1])
    ihlal = sum(1 for v in say.values() if v)
    print(f"   ★ KAPI-26 · gönderim kipi · {sys.argv[1]} · sayfa {n} · " + " · ".join(f"{k}={v}" for k, v in say.items())
          + f" · IHLAL SINIFI = {ihlal} ⇒ esik 0 ⇒ EYLEM: >0 ise DERLEME DÜSER")
    for k, v in ornek.items():
        print(f"     ✗ {k}: {v}")
    return 26 if ihlal else 0


if __name__ == "__main__":
    sys.exit(main())
