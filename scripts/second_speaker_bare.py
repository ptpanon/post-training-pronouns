#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, re, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
import template_robustness_24 as S24
import template_robustness as SG
import urial_run as UK
import arm_table as KT
from reading_style_object import payda

KOK = os.environ.get("KONUSMA_KOK", __DNH_DATA__ + "/c1_panel")
CIK = os.environ.get("KONUSMA_CIK", f"{ROOT}/results/second_speaker_bare_2026-09-11.json")
NB = int(os.environ.get("KONUSMA_NB", "500"))
SEED = 20260911
NP = int(os.environ.get("KONUSMA_NP", "8"))

ALINTI = re.compile(r'"[^"\n]{1,1200}"|“[^”\n]{1,1200}”|'
                    r'«[^»\n]{1,1200}»|‘[^’\n]{1,1200}’')
ETIKET = re.compile(r"^[ \t]{0,4}(?:[A-Z][A-Za-z0-9.'’-]{0,14}"
                    r"(?:[ ][A-Z0-9][A-Za-z0-9.'’-]{0,14}){0,2}|SPEAKER[ ]?\d{1,2})"
                    r"[ \t]{0,2}:[ \t]+(?=\S)")
TIRE = re.compile(r"^[ \t]{0,4}[—–][ \t]{0,2}(?=\S)")
KONUSMA_FIILI = {
    "say", "says", "said", "saying", "tell", "tells", "told", "telling",
    "ask", "asks", "asked", "asking", "reply", "replies", "replied",
    "answer", "answers", "answered", "argue", "argues", "argued",
    "claim", "claims", "claimed", "insist", "insists", "insisted",
    "respond", "responds", "responded", "state", "states", "stated",
    "explain", "explains", "explained", "add", "adds", "added",
    "mention", "mentions", "mentioned", "suggest", "suggests", "suggested",
    "counter", "counters", "countered", "object", "objects", "objected",
    "write", "writes", "wrote", "note", "notes", "noted", "admit", "admits",
    "admitted", "declare", "declares", "declared", "shout", "shouts",
    "shouted", "whisper", "whispers", "whispered",
}


def _satir_araliklari(m, red=None):
    sat = m.split("\n")
    aday, adet = [], {}
    o = 0
    for satir in sat:
        me = ETIKET.match(satir)
        if me:
            ad = satir[:me.end()].strip()
            adet[ad] = adet.get(ad, 0) + 1
            aday.append((o, o + len(satir), ad))
        elif TIRE.match(satir):
            aday.append((o, o + len(satir), None))
        o += len(satir) + 1
    out = []
    for a, b, ad in aday:
        if ad is None or adet.get(ad, 0) >= 2:
            out.append((a, b))
        elif red is not None:
            red[ad] = red.get(ad, 0) + 1
    return out


BIRINCI_IKINCI = {"i", "we", "you", "us", "me", "myself", "ourselves", "yourself"}


def _ucuncu_kisi_ozne(t):
    for c in t.children:
        if c.dep_ in ("nsubj", "nsubjpass"):
            return c.lower_ not in BIRINCI_IKINCI
    return False


def _aktarilan_araliklari(doc):
    out = []
    for t in doc:
        if (t.lower_ in KONUSMA_FIILI and t.pos_ in ("VERB", "AUX")
                and _ucuncu_kisi_ozne(t)):
            for c in t.children:
                if c.dep_ == "ccomp":
                    s = c.subtree
                    ks = list(s)
                    if ks:
                        out.append((ks[0].idx, ks[-1].idx + len(ks[-1].text)))
    return out


def _birlestir(ar):
    ar = sorted(a for a in ar if a[1] > a[0])
    if not ar:
        return []
    out = [list(ar[0])]
    for a, b in ar[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def _kes(m, ar):
    if not ar:
        return m
    p, o = [], 0
    for a, b in ar:
        p.append(m[o:a]); p.append(" "); o = b
    p.append(m[o:])
    return "".join(p)


def _bosluk_kuyrugu(m):
    i = m.find("\n\n")
    return [(i, len(m))] if i >= 0 else []


def araliklar(doc, m, red=None):
    A = [(x.start(), x.end()) for x in ALINTI.finditer(m)]
    K = _satir_araliklari(m, red)
    R = _aktarilan_araliklari(doc)
    return A, K, R


def prova(nlp):
    a = ('He said that you are wrong.\nAlice: you should stop.\n'
         'Alice: you never listen.\n"you again," he cried.')
    b = ("Note: you should stop.\nFirst: you must read this.\n"
         "Conclusion: you were right.\nStep 1: you begin here.")
    da, db = nlp(a), nlp(b)
    Aa, Ka, Ra = araliklar(da, a)
    Ab, Kb, Rb = araliklar(db, b)
    ka = _kes(a, _birlestir(Aa + Ka + Ra))
    ok = (len(Aa) >= 1 and len(Ka) >= 2 and len(Ra) >= 1
          and not Ab and not Kb and not Rb
          and ka.count("you") == 0 and _kes(b, _birlestir(Ab + Kb + Rb)) == b)
    payda("konusma_ciplak_prova", n_sinav=6,
          hal_alinti=len(Aa), hal_konusmaci=len(Ka), hal_aktarilan=len(Ra),
          red_yanlis_pozitif=len(Ab) + len(Kb) + len(Rb),
          hal_kalan_you=ka.count("you"))
    print(f"{len(Aa)} {len(Ka)}"
          f"{len(Ra)} {ka.count('you')}"
          f"{len(Ab)+len(Kb)+len(Rb)}"
          f"", flush=True)
    return ok


def _oku(a, bacak):
    R = MK.oku(a, bacak, kok=KOK)
    return R


def _bacak(nlp, R):
    M = [r["metin"] for r in R]
    nA = nK = nR = 0
    kar, tav = [], []
    for m, doc in zip(M, nlp.pipe(M, batch_size=64, n_process=NP)):
        A, K, Rr = araliklar(doc, m)
        nA += len(A); nK += len(K); nR += len(Rr)
        u = _birlestir(A + K + Rr)
        kar.append({"metin": _kes(m, u)})
        tav.append({"metin": _kes(m, _birlestir(u + _bosluk_kuyrugu(m)))})
    return kar, tav, dict(n_alinti=nA, n_konusmaci=nK, n_aktarilan=nR)


def main():
    t0 = time.time()
    nlp = FS._boru()
    if not prova(nlp):
        print("★ PROVA DÜSTÜ ⇒ ölcüm YAZILMAZ"); return 4
    HAV = {k["ad"].split("·")[0]: dict(k)
           for k in (S24.KOLLAR + SG.KOLLAR + UK.cift_kollar())}
    AILE = [(a, HAV[a]["taban"].split("/", 1)[1], HAV[a]["hizali"].split("/", 1)[1])
            for a in UK.TEK_CARD + UK.CIFT_CARD]
    OUT = {}
    for a, zt, zh in AILE:
        Rt, Rh = _oku(a, zt), _oku(a, zh)
        if Rt is None or Rh is None:
            OUT[a] = dict(hal="BACAK-EKSIK"); print(f"  ✗ {a}: bacak eksik"); continue
        it = np.array([int(r["istem_i"]) for r in Rt])
        ih = np.array([int(r["istem_i"]) for r in Rh])
        kt, tt, st = _bacak(nlp, Rt)
        kh, th, sh = _bacak(nlp, Rh)
        Vt = MK.satir_bilesenleri(nlp, Rt); Vh = MK.satir_bilesenleri(nlp, Rh)
        Kt = MK.satir_bilesenleri(nlp, kt); Kh = MK.satir_bilesenleri(nlp, kh)
        Tt = MK.satir_bilesenleri(nlp, tt); Th = MK.satir_bilesenleri(nlp, th)
        j = MK.ALAN.index("m1_sahis2")
        jj = MK.ALAN.index("n_jeton")
        s2t, s2h = float(Vt[:, j].sum()), float(Vh[:, j].sum())
        s2tk, s2hk = float(Kt[:, j].sum()), float(Kh[:, j].sum())
        M1 = lambda V: float(MK.olc_toplam(V)["M1"])
        rec = dict(
            hal="ÖLCÜLDÜ", n_satir=len(Rt),
            tespit_taban=st, tespit_hizali=sh,
            pay_you_araliklarda_taban=round((s2t - s2tk) / max(s2t, 1), 4),
            pay_you_araliklarda_hizali=round((s2h - s2hk) / max(s2h, 1), 4),
            M1_taban=round(M1(Vt), 4), M1_hizali=round(M1(Vh), 4),
            M1_taban_kesik=round(M1(Kt), 4), M1_hizali_kesik=round(M1(Kh), 4),
            M1_taban_tavan=round(M1(Tt), 4), M1_hizali_tavan=round(M1(Th), 4),
            dM1=round(M1(Vh) - M1(Vt), 4),
            dM1_kesik=round(M1(Kh) - M1(Kt), 4),
            dM1_tavan=round(M1(Th) - M1(Tt), 4),
            pay_jeton_kesildi_taban=round(1 - float(Kt[:, jj].sum()) /
                                          max(float(Vt[:, jj].sum()), 1), 4),
            pay_jeton_tavan_kesildi_taban=round(1 - float(Tt[:, jj].sum()) /
                                                max(float(Vt[:, jj].sum()), 1), 4),
            M1_yon_kesik=("artti" if M1(Kt) > M1(Vt) else "azaldi"),
            M1_yon_tavan=("artti" if M1(Tt) > M1(Vt) else "azaldi"),
        )
        for ad, (Va, Vb) in (("ci", (Vt, Vh)), ("ci_kesik", (Kt, Kh)),
                             ("ci_tavan", (Tt, Th))):
            ci = KT._boot(Vb, ih, Va, it, nb=NB, seed=SEED)
            rec[ad] = [round(x, 4) for x in ci] if ci else None
            rec[ad.replace("ci", "ayrik") if ad != "ci" else "ayrik"] = bool(
                ci and (ci[0] > 0 or ci[1] < 0))
        OUT[a] = rec
        print(f"  [{time.time()-t0:5.0f}s] {a:16s} you'larin %"
              f"{100*rec['pay_you_araliklarda_taban']:4.1f}'i (taban) / %"
              f"{100*rec['pay_you_araliklarda_hizali']:4.1f}'i (hizali) aralikta · "
              f"ΔM1 {rec['dM1']:+7.2f} → kesik {rec['dM1_kesik']:+7.2f} → tavan "
              f"{rec['dM1_tavan']:+7.2f}", flush=True)
    ol = [v for v in OUT.values() if v.get("hal") == "ÖLCÜLDÜ"]
    n_tam = sum(1 for v in ol if v["dM1"] < 0 and v["ayrik"])
    n_kes = sum(1 for v in ol if v["dM1_kesik"] < 0 and v["ayrik_kesik"])
    n_tav = sum(1 for v in ol if v["dM1_tavan"] < 0 and v["ayrik_tavan"])
    payda("konusma_ciplak", n_aile=len(AILE), hal_olculen=len(ol),
          hal_asagi_ayrik_tam=n_tam, hal_asagi_ayrik_kesik=n_kes,
          hal_asagi_ayrik_tavan=n_tav,
          hal_liste_elle_yazildi=1, red_bacak_eksik=len(AILE) - len(ol))
    N = len(AILE)
    print(f"{n_tam} {N}"
          f"{n_kes} {N} {n_tav} {N} {len(ol)} {N}"
          f"{N}"
          f"{N}"
          f"", flush=True)
    K = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF=""
                   "",
             alet="scripts/second_speaker_bare.py", borc="D-0911-V33 §B/1 · HAKEM-ASSISTANT W2",
             kok=KOK, n_bootstrap=NB, seed=SEED, kume_birimi="istem_i",
             sinirlar=["düz kesme isareti tirnagi sayilmaz ⇒ ALINTI alt sinir",
                       "konusma fiili listesi ELLE yazildi ⇒ ERRATA-2 sinifi",
                       "",
                       ""],
             sayim=dict(tam=n_tam, kesik=n_kes, tavan=n_tav, payda=len(AILE),
                        olculen=len(ol)),
             aile=OUT)
    json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}  ({(time.time()-t0)/60:.1f} dk)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
