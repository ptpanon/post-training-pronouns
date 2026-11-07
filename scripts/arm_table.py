#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
import addressee_run as MK
import form_count as FS
import k1b_generation as K1BU
import template_robustness_24 as S24
import template_robustness as SG
import urial_run as UK
from reading_style_object import payda

KOK = os.environ.get("KOL_TABLO_KOK", __DNH_DATA__ + "/c1_panel")
CIK = os.environ.get("KOL_TABLO_CIK", f"{ROOT}/results/KOL_TABLOSU_2026-09-10.json")
NB = int(os.environ.get("KOL_TABLO_NB", "500"))
SEED = 20260910
N_BEK = 6528


def _boot(Vh, ih, Vt, it, nb=NB, seed=SEED):
    ka = {u: np.where(it == u)[0] for u in np.unique(it)}
    kb = {u: np.where(ih == u)[0] for u in np.unique(ih)}
    ortak = [u for u in ka if u in kb]
    if not ortak:
        return None
    rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        sec = rng.choice(ortak, size=len(ortak), replace=True)
        a = np.concatenate([ka[u] for u in sec]); b = np.concatenate([kb[u] for u in sec])
        out.append(MK.olc_toplam(Vh[b])["M1"] - MK.olc_toplam(Vt[a])["M1"])
    o = np.array(out, dtype=float); o = o[~np.isnan(o)]
    if not len(o):
        return None
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def kume_ort_boot(x, kume, nb=NB, seed=SEED):
    x = np.asarray(x, dtype=float); kume = np.asarray(kume)
    m = ~np.isnan(x)
    x, kume = x[m], kume[m]
    if not len(x):
        return None
    ks = np.unique(kume)
    ix = {k: np.where(kume == k)[0] for k in ks}
    rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        sec = rng.choice(ks, size=len(ks), replace=True)
        j = np.concatenate([ix[k] for k in sec])
        out.append(float(x[j].mean()))
    o = np.array(out, dtype=float); o = o[~np.isnan(o)]
    if not len(o):
        return None
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)),
            float(x.mean()), int(len(ks)))


def _kume_ort_boot_provasi():
    x = np.full(200, 3.0)
    r = kume_ort_boot(x, np.arange(200), nb=200, seed=1)
    sabit_ok = r is not None and abs(r[1] - r[0]) < 1e-9
    rng = np.random.default_rng(7)
    y = np.concatenate([rng.normal(m, 0.1, 20) for m in rng.normal(0, 1.0, 10)])
    kume_satir = np.arange(len(y))
    kume_istem = np.repeat(np.arange(10), 20)
    a = kume_ort_boot(y, kume_satir, nb=400, seed=2)
    b = kume_ort_boot(y, kume_istem, nb=400, seed=2)
    genis_ok = (b[1] - b[0]) > (a[1] - a[0])
    payda("kume_ort_boot_prova", n_sinav=2, hal_sabit_sifir=int(sabit_ok),
          hal_kume_genisletti=int(genis_ok),
          hal_satir_genislik=round(a[1] - a[0], 4),
          hal_istem_genislik=round(b[1] - b[0], 4),
          red_uyusmaz=int(not (sabit_ok and genis_ok)))
    print(f"{sabit_ok}"
          f"{a[1]-a[0]:.4f} {b[1]-b[0]:.4f}"
          f"{genis_ok}"
          f"", flush=True)
    return sabit_ok and genis_ok


def main():
    t0 = time.time(); nlp = FS._boru()
    HAV = {k["ad"].split("·")[0]: dict(k)
           for k in (S24.KOLLAR + SG.KOLLAR + UK.cift_kollar())}
    KOL = [k[0] for k in K1BU.kollar(ton=True)]
    AILE = [(a, HAV[a]["taban"].split("/", 1)[1], HAV[a]["hizali"].split("/", 1)[1])
            for a in UK.TEK_CARD + UK.CIFT_CARD]
    payda("kol_tablosu_evren", n_aile=len(AILE), n_kol=len(KOL), red_eksik=abs(16 - len(AILE)))

    onek_kol, onek_ilk = {}, None
    ONEK = {}
    ayrisan = 0
    SONUC = {}
    for a, zt, zh in AILE:
        Rt, Rh = MK.oku(a, zt, kok=KOK), MK.oku(a, zh, kok=KOK)
        if Rt is None or Rh is None or len(Rt) != N_BEK or len(Rh) != N_BEK:
            SONUC[a] = dict(hal="BACAK-EKSIK",
                            taban=(None if Rt is None else len(Rt)),
                            hizali=(None if Rh is None else len(Rh)))
            print(f"  ✗ {a}: BACAK-EKSIK", flush=True); continue
        Vt = MK.satir_bilesenleri(nlp, Rt); Vh = MK.satir_bilesenleri(nlp, Rh)
        kt = np.array([r["kol"] for r in Rt]); kh = np.array([r["kol"] for r in Rh])
        it = np.array([r["istem_i"] for r in Rt]); ih = np.array([r["istem_i"] for r in Rh])
        if onek_ilk is None:
            def _ortak_sonek(ss):
                s0 = min(ss, key=len); n = 0
                while n < len(s0) and all(x.endswith(s0[len(s0) - n - 1:]) for x in ss):
                    n += 1
                return s0[len(s0) - n:] if n else ""
            grup = {}
            for r in Rt:
                grup.setdefault((r["istem_i"], r["cekim"]), {})[r["kol"]] = r["onek"]
            onek_ilk = {k: [] for k in KOL}
            _suz = []
            for g, dk in grup.items():
                if len(dk) != len(KOL):
                    continue
                suf = _ortak_sonek(list(dk.values())); _suz.append(len(suf))
                for k, o in dk.items():
                    onek_ilk[k].append(o[:len(o) - len(suf)] if suf else o)
            for k in KOL:
                Vo = MK.satir_bilesenleri(nlp, [{"metin": o} for o in onek_ilk[k]])
                t = MK.olc_toplam(Vo); sv = Vo.sum(axis=0)
                ONEK[k] = dict(n_onek=len(onek_ilk[k]), M1_onek=round(t["M1"], 4),
                               n_sahis2=int(sv[MK.ALAN.index("m1_sahis2")]),
                               n_jeton=int(sv[MK.ALAN.index("n_jeton")]))
            payda("kol_onek_ayristirma", n_grup=len(grup),
                  hal_ortak_sonek_ort=int(sum(_suz) / max(len(_suz), 1)),
                  hal_you_tasiyan_kol=sum(1 for k in KOL if ONEK[k]["n_sahis2"] > 0),
                  red_grup_eksik=sum(1 for dk in grup.values() if len(dk) != len(KOL)))
            print("  ★ kol önegi ayristirildi · 2. kisi TASIYAN kol: "
                  + (", ".join(f"{k}({ONEK[k]['n_sahis2']})" for k in KOL
                               if ONEK[k]["n_sahis2"] > 0) or "YOK"), flush=True)
        else:
            _g2 = {}
            for r in Rt:
                _g2.setdefault((r["istem_i"], r["cekim"]), {})[r["kol"]] = r["onek"]
            if len(_g2) != len(grup):
                ayrisan += 1
        satir = {}
        for k in KOL:
            mt, mh = kt == k, kh == k
            if not mt.any() or not mh.any():
                satir[k] = dict(hal="KOL-YOK"); continue
            M1t = float(MK.olc_toplam(Vt[mt])["M1"]); M1h = float(MK.olc_toplam(Vh[mh])["M1"])
            ci = _boot(Vh[mh], ih[mh], Vt[mt], it[mt])
            satir[k] = dict(hal="ÖLCÜLDÜ", M1_taban=round(M1t, 4), M1_hizali=round(M1h, 4),
                            dM1=round(M1h - M1t, 4), ci=[round(x, 4) for x in ci] if ci else None,
                            ayrik=bool(ci and (ci[0] > 0 or ci[1] < 0)))
        yf = [k for k in KOL if ONEK[k]["n_sahis2"] == 0]
        if yf:
            my_t = np.isin(kt, yf); my_h = np.isin(kh, yf)
            M1t = float(MK.olc_toplam(Vt[my_t])["M1"]); M1h = float(MK.olc_toplam(Vh[my_h])["M1"])
            ci = _boot(Vh[my_h], ih[my_h], Vt[my_t], it[my_t])
            havuz = dict(n_kol=len(yf), kollar=yf, M1_taban=round(M1t, 4),
                         M1_hizali=round(M1h, 4), dM1=round(M1h - M1t, 4),
                         ci=[round(x, 4) for x in ci] if ci else None,
                         ayrik=bool(ci and (ci[0] > 0 or ci[1] < 0)))
        else:
            havuz = dict(n_kol=0, hal="YOU-FREE-KOL-YOK")
        tum_t = float(MK.olc_toplam(Vt)["M1"]); tum_h = float(MK.olc_toplam(Vh)["M1"])
        SONUC[a] = dict(hal="ÖLCÜLDÜ", kol=satir, you_free=havuz,
                        tum_panel=dict(M1_taban=round(tum_t, 4), M1_hizali=round(tum_h, 4),
                                       dM1=round(tum_h - tum_t, 4)))
        n_ayrik = sum(1 for v in satir.values() if v.get("ayrik"))
        n_poz = sum(1 for v in satir.values() if v.get("ayrik") and v["dM1"] > 0)
        print(f"  [{time.time()-t0:5.0f}s] {a:16s} tüm-panel ΔM1 {tum_h-tum_t:+7.2f} · "
              f"you-free({havuz.get('n_kol')}) {havuz.get('dM1')} "
              f"{havuz.get('ci')} · kol CI-ayrik {n_ayrik}/16 (poz {n_poz})", flush=True)

    olculen = [a for a, v in SONUC.items() if v.get("hal") == "ÖLCÜLDÜ"]
    yf_ayrik = sum(1 for a in olculen if SONUC[a]["you_free"].get("ayrik")
                   and SONUC[a]["you_free"]["dM1"] < 0)
    tum_ayrik = sum(1 for a in olculen if SONUC[a]["tum_panel"]["dM1"] < 0)
    payda("kol_tablosu", n_aile=len(AILE), hal_olculen=len(olculen),
          hal_you_free_kol=sum(1 for k in KOL if ONEK[k]["n_sahis2"] == 0),
          hal_you_free_asagi_ayrik=yf_ayrik, hal_tum_panel_asagi=tum_ayrik,
          red_bacak_eksik=len(AILE) - len(olculen), red_onek_ayrisan=ayrisan)
    K = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="KESIF/BETIM — bar YOK · ad YOK · prediction YOK (§10)",
             alet="scripts/arm_table.py", borc="D-0910-V32 §B/1.2",
             kok=KOK, n_bootstrap=NB, seed=SEED, kume_birimi="istem_i",
             serh=(""
                   ""),
             onek=ONEK, aile=SONUC)
    json.dump(K, io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ {CIK}  ({(time.time()-t0)/60:.1f} dk)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
