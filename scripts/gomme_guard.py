import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, json, hashlib
import numpy as np

ROOT = __DNH_ROOT__ + ""
GOL = f"{ROOT}/unreleased/gol"

RED_KURALI = {
    "arsipel": "bos_dize",
    "gol": "kaynak_dolgu",
}


class BosMetinHatasi(ValueError):
    pass


class SozlesmeHatasi(AssertionError):
    pass


class KapsamHatasi(AssertionError):
    pass


def dogrula_metin(metin, baglam=""):
    bos = [i for i, t in enumerate(metin) if not (t and str(t).strip())]
    if bos:
        raise BosMetinHatasi(
            f"{baglam} {len(bos)} {len(metin)}"
            f"{bos[:5]}"
            f"")
    return metin


def yer_tutucu_mu(t, uzay, kaynak=None):
    if uzay not in RED_KURALI:
        raise SozlesmeHatasi(
            f"{uzay}"
            f"")
    if RED_KURALI[uzay] == "bos_dize":
        return not (t and str(t).strip())
    if kaynak is None:
        raise SozlesmeHatasi("kaynak-dolgu kolunda `kaynak` ZORUNLU: kaynak-dolgu bos-dize "
                             "testiyle GÖRÜNMEZ.")
    return (len(str(t).split()) < 5) or (str(t) == str(kaynak)[:1200])


def uretim_maskesi(KEY, uzay, NPARA, N, kaynak_metin=None, kesisim=None):
    if uzay not in RED_KURALI:
        raise SozlesmeHatasi(
            f"{KEY} {uzay}"
            f"")
    M = np.zeros((NPARA, N), dtype=bool)
    ayrinti = {}
    for d in range(NPARA):
        rw = json.load(open(f"{GOL}/gen/{KEY}/draw_{d}.json"))["rewrites"]
        if len(rw) != N:
            if kesisim is None:
                raise SozlesmeHatasi(f"{KEY} {d} {len(rw)} {N}"
                                     f"")
            rw = [rw[i] for i in kesisim]
        if RED_KURALI[uzay] == "bos_dize":
            yt = np.array([yer_tutucu_mu(t, uzay) for t in rw])
            ayrinti[d] = dict(bos_dize=int(yt.sum()))
        else:
            if kaynak_metin is None:
                raise SozlesmeHatasi(f"[{KEY}] göl kolunda kaynak_metin ZORUNLU: kaynak-dolgu "
                                     f"bos-dize testiyle GÖRÜNMEZ (CLAUDE_v1.md §5-(1b) ≡ v2 §8(1b)).")
            yt = np.array([yer_tutucu_mu(t, uzay, s) for t, s in zip(rw, kaynak_metin)])
            ayrinti[d] = dict(kisa_5kelime=int(sum(len(str(t).split()) < 5 for t in rw)),
                              kaynakla_ozdes=int(sum(str(t) == str(s)[:1200]
                                                     for t, s in zip(rw, kaynak_metin))),
                              birlesik=int(yt.sum()))
        M[d] = ~yt
    rapor = dict(kol=KEY, uzay=uzay, red_kurali=RED_KURALI[uzay], NPARA=NPARA, N=N,
                 uretilen=int(M.sum()), yer_tutucu=int((~M).sum()),
                 yer_tutucu_orani=float((~M).sum() / M.size),
                 cekim_basina=ayrinti,
                 maske_doluluk=dict(toplam_hucre=int(M.size), True_sayisi=int(M.sum()),
                                    False_sayisi=int((~M).sum())),
                 kapsam=dict(okunan_cekim_dosyasi=int(NPARA), incelenen_satir=int(NPARA * N),
                             kol_basina_dolduruldu=True))
    if M.size == 0 or NPARA == 0:
        raise SozlesmeHatasi(f"{KEY}"
                             f"")
    return M, rapor


def vektor_teshisi(V, kaynak_satirlari=None, esik=0.999999):
    V = np.asarray(V)
    nrm = np.linalg.norm(V, axis=1)
    sifir = np.where(nrm < 1e-8)[0]

    anahtar = np.stack([V[:, 0], V[:, -1], V[:, V.shape[1] // 2], nrm], axis=1)
    _, ters, say = np.unique(anahtar, axis=0, return_inverse=True, return_counts=True)
    kumeler = []
    for g in np.where(say > 1)[0]:
        idx = np.where(ters == g)[0]
        alt = {}
        for i in idx:
            alt.setdefault(V[i].tobytes(), []).append(int(i))
        kumeler += [v for v in alt.values() if len(v) > 1]
    kumeler.sort(key=len, reverse=True)
    yinelenen = int(sum(len(k) for k in kumeler))

    out = dict(n=int(len(V)),
               sifir_satir=int(len(sifir)), sifir_ilk=sifir[:5].tolist(),
               yinelenen_satir=yinelenen, yigin_sayisi=len(kumeler),
               en_buyuk_yigin=int(len(kumeler[0])) if kumeler else 0,
               en_buyuk_yigin_ilk=kumeler[0][:5] if kumeler else [],
               kaynakla_ozdes=None, kaynakla_ozdes_ilk=[])
    if kaynak_satirlari is not None:
        K = np.asarray(kaynak_satirlari)
        if K.ndim == 1:
            K = V[K] if K.max() < len(V) else None
        if K is not None:
            c = np.einsum('ij,ij->i', V, K) / np.maximum(nrm * np.linalg.norm(K, axis=1), 1e-12)
            oz = np.where(c > esik)[0]
            out["kaynakla_ozdes"] = int(len(oz)); out["kaynakla_ozdes_ilk"] = oz[:5].tolist()
    return out


def kimlik_parmak_izi(KEY, V, M_uretildi, enc, n_ornek=64, seed=20260728):
    V = np.asarray(V); yt = ~np.asarray(M_uretildi).reshape(-1)
    gec = np.where(~yt)[0]
    if len(gec) == 0:
        raise SozlesmeHatasi(f"{KEY}")
    rng = np.random.default_rng(seed)
    sec = np.sort(rng.choice(gec, size=min(n_ornek, len(gec)), replace=False))
    h = hashlib.sha256()
    for i in sec:
        h.update(np.ascontiguousarray(V[i]).tobytes())
    hm = hashlib.sha256(np.ascontiguousarray(np.asarray(M_uretildi)).tobytes()).hexdigest()[:16]
    return dict(kol=KEY, enc=enc, satir=int(V.shape[0]), boyut=int(V.shape[1]),
                n_ornek=int(len(sec)), seed=seed,
                ornek_sha=h.hexdigest()[:24],
                ilk_uretilmis_idx=int(gec[0]),
                ilk_uretilmis_sha=hashlib.sha256(
                    np.ascontiguousarray(V[gec[0]]).tobytes()).hexdigest()[:16],
                son_uretilmis_idx=int(gec[-1]),
                son_uretilmis_sha=hashlib.sha256(
                    np.ascontiguousarray(V[gec[-1]]).tobytes()).hexdigest()[:16],
                maske_sha=hm, yer_tutucu=int(yt.sum()),
                kapsam=dict(uretilmis_havuz=int(len(gec)), ornek_alinan=int(len(sec)),
                            toplam_satir=int(V.shape[0])))


def dogrula_parmak_izi(yeni, manifest_fp, yaz=print):
    import os
    ters = {}
    for k, v in yeni.items():
        ters.setdefault(v["ornek_sha"], []).append(k)
    cakisan = {h: ks for h, ks in ters.items() if len(ks) > 1}
    if cakisan:
        raise SozlesmeHatasi(
            f"{cakisan}"
            f"")
    if not os.path.exists(manifest_fp):
        json.dump(dict(_taban_tarih="ilk yazim", kol=yeni), open(manifest_fp, "w"),
                  ensure_ascii=False, indent=1)
        yaz(f"{len(yeni)}"
            f"")
        return dict(mod="taban_yazildi", sapan=[])
    eski = json.load(open(manifest_fp))["kol"]
    sapan = [k for k, v in yeni.items()
             if k in eski and any(eski[k].get(a) != v.get(a)
                                  for a in ("satir", "ornek_sha", "maske_sha", "yer_tutucu"))]
    if sapan:
        raise SozlesmeHatasi(f"{sapan}")
    yaz(f"[PARMAK-IZI] {len(yeni)} kol manifest'le UYUSTU; cakisma yok.")
    return dict(mod="dogrulandi", sapan=[])


def on_ucus(KEY, M_metin, V, kaynak_satirlari=None, kaynak_gecerli=None, sikilik="dur", yaz=print):
    NPARA, N = M_metin.shape
    assert len(V) == NPARA * N, f"{KEY} {len(V)} {NPARA} {N} {NPARA*N}"
    yt = ~M_metin.reshape(-1)
    kg = np.ones(len(V), dtype=bool) if kaynak_gecerli is None else np.asarray(kaynak_gecerli)
    tes = vektor_teshisi(V, kaynak_satirlari)
    n_metin = int(yt.sum())

    varyant, kanit = "yok", {}
    ytk = yt & kg
    if n_metin:
        Vy = np.asarray(V)[yt]
        nrm = np.linalg.norm(Vy, axis=1)
        if (nrm < 1e-8).all():
            varyant = "sifir-vektor"; kanit = dict(norm_max=float(nrm.max()))
        else:
            Vn = Vy / np.maximum(nrm[:, None], 1e-12)
            ic = Vn @ Vn[0]
            if ic.min() > 0.999999:
                varyant = "sabit-dolgu"; kanit = dict(yigin_ici_min_cos=float(ic.min()),
                                                      norm=float(nrm[0]))
            elif kaynak_satirlari is not None and int(ytk.sum()) > 0:
                Va_ = np.asarray(V)[ytk]; K = np.asarray(kaynak_satirlari)[ytk]
                c = np.einsum('ij,ij->i', Va_, K) / np.maximum(
                    np.linalg.norm(Va_, axis=1) * np.linalg.norm(K, axis=1), 1e-12)
                kanit = dict(kaynak_kapsami=int(ytk.sum()), kaynak_kapsami_orani=
                             float(ytk.sum() / max(yt.sum(), 1)),
                             kaynakla_medyan_cos=float(np.median(c)),
                             yigin_ici_min_cos=float(ic.min()))
                varyant = "kaynak-dolgu" if float(np.median(c)) > 0.999999 else "COZULMEDI"
            else:
                varyant = "COZULMEDI"; kanit = dict(yigin_ici_min_cos=float(ic.min()),
                                                    kaynak_kapsami=int(ytk.sum()))
        if n_metin == 1 and varyant == "sabit-dolgu":
            varyant = "sabit-dolgu?tek-ornek"; kanit["uyari"] = "n=1 ⇒ yigin testi bos yere dogru"

    Va = np.asarray(V)
    karsilastirilan = yt
    if varyant == "sifir-vektor":
        n_vektor = int((np.linalg.norm(Va, axis=1) < 1e-8).sum())
    elif varyant.startswith("sabit-dolgu"):
        hedef = Va[yt][0]
        n_vektor = int((Va @ (hedef / max(np.linalg.norm(hedef), 1e-12)) > 0.999999).sum())
    elif varyant == "kaynak-dolgu":
        karsilastirilan = yt & kg
        K = np.asarray(kaynak_satirlari)[kg]; Vk = Va[kg]
        c = np.einsum('ij,ij->i', Vk, K) / np.maximum(
            np.linalg.norm(Vk, axis=1) * np.linalg.norm(K, axis=1), 1e-12)
        n_vektor = int((c > 0.999999).sum())
    else:
        n_vektor = 0 if varyant == "yok" else -1
    n_metin_karsilastirilan = int(karsilastirilan.sum())

    rapor = dict(kol=KEY, varyant=varyant, kanit=kanit,
                 metin_tarafi=n_metin, metin_tarafi_karsilastirilan=n_metin_karsilastirilan,
                 vektor_tarafi=n_vektor,
                 eslesti=bool(n_metin_karsilastirilan == n_vektor),
                 oran=float(n_metin / (NPARA * N)),
                 maske_doluluk=dict(True_sayisi=int(M_metin.sum()), False_sayisi=n_metin),
                 vektor_teshisi=tes)
    rapor["kapsam"] = dict(incelenen_satir=int(len(V)), incelenen_hucre=int(M_metin.size),
                           kaynagi_bilinen_satir=int(kg.sum()),
                           kaynak_kapsami_orani=float(kg.mean()),
                           cekim_sayisi=int(NPARA), slot_sayisi=int(N))
    if len(V) == 0 or M_metin.size == 0:
        raise SozlesmeHatasi(f"{KEY}"
                             f"")
    ek = "" if n_metin_karsilastirilan == n_metin else f" (kaynak-kapsami {n_metin_karsilastirilan})"
    yaz(f"[ÖN-UCUS {KEY}] varyant={varyant}  metin={n_metin}{ek}  vektör={n_vektor}  "
        f"eslesti={rapor['eslesti']}  oran={rapor['oran']:.3%}  kanit={kanit}")
    yaz(f"{len(V)} {M_metin.size}"
        f"{int(kg.sum())} {kg.mean():.1%} {NPARA} {N}")
    if not rapor["eslesti"]:
        msg = (f"{KEY} {n_metin_karsilastirilan}"
               f"{n_vektor} {varyant}"
               f"")
        if sikilik == "dur":
            raise SozlesmeHatasi(msg)
        yaz("UYARI " + msg)
    return rapor



class HizaHatasi(AssertionError):
    pass


def seyrek_defter(KEY, kok, npara, N, yaz=print):
    defter, top_k, top_a = {}, 0, 0
    for d in range(npara):
        fp, afp = f"{kok}/{KEY}/draw_{d}.json", f"{kok}/{KEY}/draw_{d}__atlanan.json"
        if not (os.path.exists(fp) and os.path.exists(afp)):
            raise HizaHatasi(f"{KEY} {d}")
        j, at = json.load(open(fp)), json.load(open(afp))
        idx = list(j["idx"]); atl = [p for p, _ in at]
        if len(j["rewrites"]) != len(idx):
            raise HizaHatasi(f"[{KEY}·{d}] idx {len(idx)} != rewrites {len(j['rewrites'])}")
        si, sa = set(idx), set(atl)
        if len(si) != len(idx):
            raise HizaHatasi(f"[{KEY}·{d}] idx YINELENMIS ({len(idx) - len(si)} tekrar)")
        if len(sa) != len(atl):
            raise HizaHatasi(f"[{KEY}·{d}] atlanan YINELENMIS ({len(atl) - len(sa)} tekrar)")
        ortak = si & sa
        if ortak:
            raise HizaHatasi(f"[{KEY}·{d}] idx ∩ atlanan BOS DEGIL ({len(ortak)} konum, "
                             f"ör. {sorted(ortak)[:5]})")
        eksik = set(range(N)) - si - sa
        if eksik:
            raise HizaHatasi(f"[{KEY}·{d}] {len(eksik)} konum HICBIR yerde yok "
                             f"(ör. {sorted(eksik)[:5]}) — idx ∪ atlanan ≠ N")
        if j.get("N") != N:
            raise HizaHatasi(f"[{KEY}·{d}] dosyadaki N {j.get('N')} != beklenen {N}")
        defter[d] = dict(idx=idx, atlanan=atl, metin=j["rewrites"])
        top_k += len(idx); top_a += len(atl)
    if top_k == 0:
        raise KapsamHatasi(f"{KEY}")
    yaz(f"  └ KAPSAM [{KEY}]: yazilan {top_k} + atlanan {top_a} = {top_k + top_a} "
        f"(beklenen {npara * N}) · {npara} cekimin {len(defter)}'i incelendi ✓")
    return defter


def seyrek_hiza(KEY, V, konum, defter, yaz=print):
    n1 = int(np.asarray(V).shape[0]); n2 = len(konum)
    if n1 != n2:
        raise HizaHatasi(f"{KEY} {n1} {n2}")
    say = {}
    for _, d in konum:
        say[d] = say.get(d, 0) + 1
    for d, v in defter.items():
        if say.get(d, 0) != len(v["idx"]):
            raise HizaHatasi(f"{KEY} {d} {say.get(d, 0)}"
                             f"{len(v['idx'])}")
    bekl = [[p, d] for d in sorted(defter) for p in defter[d]["idx"]]
    if [list(x) for x in konum] != bekl:
        ilk = next((i for i, (a, b) in enumerate(zip([list(x) for x in konum], bekl)) if a != b), 0)
        raise HizaHatasi(f"{KEY}"
                         f"{ilk} {list(konum[ilk])} {bekl[ilk]}")
    yaz(f"  └ HIZA [{KEY}]: satir {n1} = konum {n2} ✓ · cekim kirilimi "
        f"{ {d: say[d] for d in sorted(say)} } defterle birebir ✓ · sira birebir ✓")
    return True


def aidiyet_seyrek(KEY, V, konum, defter, enc, n_ornek=24, seed=20260729, esik=0.999999,
                   yaz=print):
    V = np.asarray(V)
    rng = np.random.default_rng(seed)
    sec = np.sort(rng.choice(len(konum), size=min(n_ornek, len(konum)), replace=False))
    yer = {}
    for d in sorted(defter):
        for k, p in enumerate(defter[d]["idx"]):
            yer[(p, d)] = defter[d]["metin"][k]
    metin = [yer[(konum[i][0], konum[i][1])] for i in sec]
    W = np.asarray(enc(metin))
    cos = (V[sec] * W).sum(1) / (np.linalg.norm(V[sec], axis=1) * np.linalg.norm(W, axis=1))
    kotu = int((cos < esik).sum())
    yaz(f"  └ AIDIYET [{KEY}]: {len(sec)} satir · cos min {cos.min():.6f} · ort {cos.mean():.6f} "
        f"· esik-alti {kotu}")
    if kotu:
        raise HizaHatasi(f"{KEY} {kotu} {len(sec)}"
                         f"{cos.min():.6f}")
    return float(cos.min())
