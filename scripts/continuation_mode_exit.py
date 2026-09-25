#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import sys, json, re, time, random, collections
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
CARD = f"{ROOT}/results/PREREG9_A_OLCUM_2026-08-31.json"
RET = re.compile(r"^(I'm sorry|I am sorry|I apologi[sz]e|I can't|I cannot|I won't|I will not|As an AI|As a language model|Sorry,)", re.I)
ASI = re.compile(r"^(Sure|Certainly|Of course|Absolutely|Here's|Here is|Here are|I'd be happy|I would be happy|Great question|Thank you for|Thanks for|Alright|Let's|Let me|Okay|OK[,.!])", re.I)
MET = re.compile(r"^((This|The) (text|passage|argument|statement|debate|discussion|excerpt|thesis|prompt|conversation|dialogue|exchange|quote)|It (seems|looks like|appears)|In summary|To summari[sz]e|Summary|You('ve| have) (provided|shared|presented|written|given))", re.I)


def sinif(m):
    s = re.sub(r'^[\s"\'“”#*>\-]+', '', m or '')[:160]
    if RET.match(s): return "RET"
    if ASI.match(s): return "ASISTAN"
    if MET.match(s): return "META"
    return "DEVAM"


_BOL = re.compile(r"(?<=[.!?])\s+")
TUR = re.compile(r"\b(Assistant|User|Human|System)\s*:|<\|[A-Za-z_]+\|>|<start_of_turn>|<end_of_turn>|\[/?INST\]|<</?SYS>>|</?s>|###\s*(Instruction|Response|Input)")
MD = re.compile(r"^\s{0,3}#{1,6}\s+\S|^\s{0,3}[-*+\u2022]\s+\S|^\s{0,3}\d{1,2}[.)]\s+\S|\*\*[^*\n]{1,200}\*\*|__[^_\n]{1,200}__", re.M)


def tavan_bayrak(m):
    m = m or ""
    isa = False
    for satir in m.split("\n"):
        for c in _BOL.split(satir):
            u = re.sub(r'^[\s"\'“”#*>\-]+', '', c)[:160]
            if u and (RET.match(u) or ASI.match(u)):
                isa = True; break
        if isa:
            break
    return isa, bool(TUR.search(m)), bool(MD.search(m))


def tavan():
    import addressee_run as MK, rung_force_ci as BK
    t0 = time.time(); out = {}; red = 0; T = {"taban": [0, 0, 0, 0, 0], "hizali": [0, 0, 0, 0, 0]}
    for aile, v in birincil().items():
        k0, k1 = v["kontrast"].split("→"); L = {}; sinif_say = {}
        for bacak, z in (("taban", k0), ("hizali", k1)):
            V = np.load(f"V_{aile}_{z}.npy"); I = np.load(f"I_{aile}_{z}.npy"); R = MK.oku(aile, z)
            assert len(R) == len(V) == len(I), (aile, z)
            B = np.array([tavan_bayrak(r["metin"]) for r in R])
            at = B.any(axis=1); L[bacak] = (V, I, at)
            sinif_say[bacak] = dict(n=len(R), atilan=int(at.sum()), isaretci=int(B[:, 0].sum()), tur=int(B[:, 1].sum()), markdown=int(B[:, 2].sum()))
            for i, x in enumerate((len(R), int(at.sum()), int(B[:, 0].sum()), int(B[:, 1].sum()), int(B[:, 2].sum()))): T[bacak][i] += x
            if abs(MK.olc_toplam(V)["M1"] - v["basamak_tablosu"][z]["M1"]) > 1e-6: red += 1
        (Vt, It, ct), (Vh, Ih, ch) = L["taban"], L["hizali"]
        d_all = MK.olc_toplam(Vh)["M1"] - MK.olc_toplam(Vt)["M1"]
        d_t = MK.olc_toplam(Vh[~ch])["M1"] - MK.olc_toplam(Vt[~ct])["M1"]
        lo, hi, nk = BK.kume_boot(Vh[~ch], Ih[~ch], Vt[~ct], It[~ct], "M1")
        out[aile] = dict(dM1_tum=round(d_all, 4), dM1_tavan=round(d_t, 4), ci_tavan=[round(lo, 3), round(hi, 3)],
                         ayrik_asagi=bool(hi < 0), n_istem=nk, **{f"{b}_{k}": x for b in sinif_say for k, x in sinif_say[b].items()},
                         ikinci_sahis_payi_atilan_hizali=round(float(Vh[ch, 1].sum() / max(Vh[:, 1].sum(), 1)), 4))
    n_ay = sum(o["ayrik_asagi"] for o in out.values())
    en = max(out, key=lambda a: abs(out[a]["dM1_tavan"] - out[a]["dM1_tum"]))
    json.dump(dict(sinif="BETIM · mühürsüz · bilerek fazla kesen tavan · kabul siniri yok", rule="preregistration/rule_continuation_mode_exit_ceiling.md",
                   alet="addressee_run.olc_toplam + rung_force_ci.kume_boot (ITHAL)", NB=BK.NB, seed=BK.SEED,
                   red_esdeger_M1=red, n_aile=len(out), n_ayrik_asagi_tavan=n_ay,
                   toplam={b: dict(zip(("n", "atilan", "isaretci", "tur", "markdown"), T[b])) for b in T},
                   en_buyuk_kayma=dict(aile=en, kayma=round(abs(out[en]["dM1_tavan"] - out[en]["dM1_tum"]), 4)),
                   aile=out, sure_sn=round(time.time() - t0, 1)),
              open("Q2_TAVAN.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(out)} {red} {T['hizali'][1]} {T['hizali'][0]}"
          f"{T['taban'][1]} {T['taban'][0]} {n_ay} {en}"
          f"")


def birincil():
    A = json.load(open(CARD))
    return {k: v for k, v in A.items() if isinstance(v, dict) and v.get("sinif") == "BIRINCIL"}


def sayim():
    import addressee_run as MK
    t0 = time.time(); out = {}; ornek = {}; rng = random.Random(20260915)
    for aile, v in birincil().items():
        k0, k1 = v["kontrast"].split("→"); out[aile] = {}
        for bacak, z in (("taban", k0), ("hizali", k1)):
            R = MK.oku(aile, z)
            c = collections.Counter(sinif(r["metin"]) for r in R)
            out[aile][bacak] = dict(zemin=z, n=len(R), **{k: c.get(k, 0) for k in ("RET", "ASISTAN", "META", "DEVAM")})
            for cls in ("RET", "ASISTAN", "META"):
                xs = [r["metin"][:140].replace("\n", " ") for r in R if sinif(r["metin"]) == cls]
                if xs: ornek[f"{aile}/{bacak}/{cls}"] = rng.sample(xs, min(5, len(xs)))
    json.dump(dict(rule="REBUTTAL_CEVAPLAR_2026-09-15.md Q2", n_aile=len(out), aile=out, sure_sn=round(time.time() - t0, 1)),
              open("Q2_SAYIM.json", "w"), ensure_ascii=False, indent=1)
    json.dump(ornek, open("Q2_ORNEK.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(out)} {2*len(out)}")


def bilesen(aile):
    import addressee_run as MK, form_count as FS
    v = birincil()[aile]; k0, k1 = v["kontrast"].split("→"); nlp = FS._boru(); t = time.time()
    for z in (k0, k1):
        R = MK.oku(aile, z)
        np.save(f"V_{aile}_{z}.npy", MK.satir_bilesenleri(nlp, R))
        np.save(f"I_{aile}_{z}.npy", np.array([r["istem_i"] for r in R]))
    print(aile, "tamam", round(time.time() - t, 1), "sn", flush=True)


def duyarlilik():
    import addressee_run as MK, rung_force_ci as BK
    out = {}; red = 0
    for aile, v in birincil().items():
        k0, k1 = v["kontrast"].split("→"); L = {}
        for bacak, z in (("taban", k0), ("hizali", k1)):
            V = np.load(f"V_{aile}_{z}.npy"); I = np.load(f"I_{aile}_{z}.npy")
            cik = np.array([sinif(r["metin"]) != "DEVAM" for r in MK.oku(aile, z)])
            L[bacak] = (V, I, cik)
            if abs(MK.olc_toplam(V)["M1"] - v["basamak_tablosu"][z]["M1"]) > 1e-6: red += 1
        (Vt, It, ct), (Vh, Ih, ch) = L["taban"], L["hizali"]
        d_all = MK.olc_toplam(Vh)["M1"] - MK.olc_toplam(Vt)["M1"]
        d_dev = MK.olc_toplam(Vh[~ch])["M1"] - MK.olc_toplam(Vt[~ct])["M1"]
        lo, hi, nk = BK.kume_boot(Vh[~ch], Ih[~ch], Vt[~ct], It[~ct], "M1")
        out[aile] = dict(dM1_tum=round(d_all, 4), dM1_card=round(v["fark"]["M1"]["gozlenen"], 4), dM1_devam=round(d_dev, 4),
                         ci_devam=[round(lo, 3), round(hi, 3)], ayrik_asagi=bool(hi < 0), n_istem=nk,
                         cikis_hizali=int(ch.sum()), cikis_taban=int(ct.sum()),
                         ikinci_sahis_payi_cikis_hizali=round(float(Vh[ch, 1].sum() / max(Vh[:, 1].sum(), 1)), 4),
                         ikinci_sahis_payi_cikis_taban=round(float(Vt[ct, 1].sum() / max(Vt[:, 1].sum(), 1)), 4))
    n_ay = sum(o["ayrik_asagi"] for o in out.values())
    json.dump(dict(sinif="BETIM · mühürsüz · rebuttal (kâgitta yok)", rule="REBUTTAL_CEVAPLAR_2026-09-15.md Q2",
                   alet="addressee_run.olc_toplam + rung_force_ci.kume_boot (ITHAL)", NB=BK.NB, seed=BK.SEED,
                   red_esdeger_M1=red, n_aile=len(out), n_ayrik_asagi_devam=n_ay, aile=out),
              open("Q2_DUYARLILIK.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(out)} {red} {n_ay}"
          f"")


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else "sayim"
    {"sayim": sayim, "duyarlilik": duyarlilik, "tavan": tavan}.get(k, lambda: bilesen(sys.argv[2]))()
