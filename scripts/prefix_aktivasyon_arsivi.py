#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, time, hashlib, argparse
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from gpu_lock import kilitle
import prefix_steering_generation as SU

KORPUS = __DNH_DATA__ + "/onek_korpus"
ARSIV = f"{KORPUS}/aktivasyon_arsivi"
MANIFEST = f"{ARSIV}/manifest.json"
CIKTI = f"{ROOT}/unreleased/ARSIV_MANIFEST_2026-08-08.json"
KATMANLAR = (8, 12, 16, 20, 24, 28)
ALAN = "metin_160"
PARCA = 816
MIN_UZUNLUK = 40


def kaynaklar():
    import glob
    K = []
    for korpus, desen in (("ton_dengeli", "ton__mistral.jsonl"),
                          ("dominance_tam", "domtam__mistral.jsonl"),
                          ("merdiven", "mistral_*.jsonl")):
        for yol in sorted(glob.glob(f"{KORPUS}/{korpus}/{desen}")):
            n = sum(1 for _ in open(yol, encoding="utf-8"))
            for b in range(0, n, PARCA):
                K.append(dict(korpus=korpus, dosya=os.path.basename(yol), yol=yol,
                              bas=b, son=min(b + PARCA, n),
                              ad=f"{korpus}__{os.path.basename(yol)[:-6]}__p{b//PARCA:02d}"))
    return K


def kimlik_sha(satirlar):
    h = hashlib.sha256()
    for r in satirlar:
        h.update((r.get("debate", "") + "|" + r.get("durus", "") + "|" +
                  str(r.get("cekim", "")) + "|" +
                  hashlib.sha1((r.get(ALAN) or "").encode()).hexdigest()[:16]
                  + "\n").encode())
    return h.hexdigest()[:16]


def shard_tam_mi(s):
    met, npz = f"{ARSIV}/{s['ad']}.meta.json", f"{ARSIV}/{s['ad']}.npz"
    if not (os.path.exists(met) and os.path.exists(npz)):
        return False, "dosya yok"
    try:
        m = json.load(open(met, encoding="utf-8"))
        if not m.get("tamam") or m.get("sema") != 2:
            return False, "tamam=False ya da eski sema"
        S0 = [json.loads(l) for l in open(s["yol"], encoding="utf-8")][s["bas"]:s["son"]]
        if m.get("kimlik_sha_s0") != kimlik_sha(S0):
            return False, "KIMLIK SHA UYUSMUYOR"
        z = np.load(npz)
        n = int(z["h_tam"].shape[0])
        if n != int(z["satir_ix"].shape[0]):
            return False, f"indeks {z['satir_ix'].shape[0]} != satir {n}"
        if n != int(m.get("n_satir", -1)):
            return False, ""
        if int(z["h_tam"].shape[1]) != len(KATMANLAR):
            return False, ""
        if n and int(z["satir_ix"].max()) >= len(S0):
            return False, "indeks ham dilimin disinda"
        return True, "tam"
    except Exception as e:
        return False, f"okunamadi: {e}"


def yukle(korpus=None, katman=16, seg=True, attn=True):
    li = list(KATMANLAR).index(katman)
    H, Hs, A, R, KOL = [], [], [], [], []
    istenen = arsivde = 0
    for s in kaynaklar():
        if korpus and s["korpus"] != korpus:
            continue
        tam, nicin = shard_tam_mi(s)
        if not tam:
            raise RuntimeError(f"{s['ad']} {nicin}")
        S0 = [json.loads(l) for l in open(s["yol"], encoding="utf-8")][s["bas"]:s["son"]]
        z = np.load(f"{ARSIV}/{s['ad']}.npz")
        ix = z["satir_ix"]
        istenen += len(S0); arsivde += len(ix)
        H.append(z["h_tam"][:, li, :].astype(np.float64))
        if seg:
            Hs.append(z["h_seg"][:, li, :, :].astype(np.float64))
        if attn:
            A.append(z["a_tam"][:, li, :].astype(np.float64))
        R.extend(S0[int(j)] for j in ix)
        KOL.extend([s["ad"]] * len(ix))
    H = np.concatenate(H)
    Hs = np.concatenate(Hs) if seg else None
    A = np.concatenate(A) if attn else None
    payda("arsiv_okuma", n_istenen=istenen, n_arsivde=arsivde, n_eslesen=len(R),
          n_boyut=int(H.shape[1]), bekle={"n_eslesen": arsivde})
    if len(R) != H.shape[0]:
        raise RuntimeError(f"ESLEME KAYMASI: {len(R)} satir ↔ {H.shape[0]} vektör")
    return H, Hs, A, R, np.array(KOL)


def katman_ortalamasi(hidden_states, bas=0, katmanlar=None, ofs=1, dtype=np.float16):
    T = hidden_states[0].shape[1]
    if T - bas < 1:
        raise ValueError(f"bos dilim: T={T} bas={bas} ⇒ ortalama TANIMSIZ")
    ix = range(len(hidden_states)) if katmanlar is None else [L + ofs for L in katmanlar]
    return np.stack([hidden_states[i][0, bas:, :].float().mean(0).cpu().numpy()
                     for i in ix]).astype(dtype)


P_ONEK = 64


def kayip_profili(logits, ids, n_on, p_onek=P_ONEK, dtype=np.float32):
    import torch
    T = int(ids.shape[1])
    if n_on < 2:
        raise ValueError(f"ALIM dilimi bos: n_on={n_on} ⇒ önek kaybi TANIMSIZ")
    lp = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
    nll = (-lp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)).cpu().numpy()
    k_on, k_ur = nll[:n_on - 1], nll[n_on - 1:]
    prof = np.full(p_onek, np.nan, dtype=dtype)
    maske = np.zeros(p_onek, dtype=bool)
    al = min(p_onek, len(k_on))
    if al:
        prof[p_onek - al:] = k_on[-al:]
        maske[p_onek - al:] = True
    kes = [int(round(len(k_on) * j / 3)) for j in range(4)]
    seg = np.array([k_on[kes[j]:max(kes[j + 1], kes[j] + 1)].mean() for j in range(3)],
                   dtype=dtype)
    return (prof, maske, dtype(k_on.mean()),
            dtype(k_ur.mean()) if len(k_ur) else dtype(np.nan), seg)


def prova_kayip():
    import torch
    V, gecti, dusen = 512, 0, []

    def kontrol(ad, kosul):
        nonlocal gecti
        if kosul:
            gecti += 1
        else:
            dusen.append(ad)
        print(f"    {'✓' if kosul else '✗'} {ad}")

    ids = torch.arange(1, 21).unsqueeze(0)
    lg = torch.zeros(1, 20, V)
    prof, mk, ko, ku, seg = kayip_profili(lg, ids, n_on=10)
    kontrol(f"düz logit ⇒ NLL = log({V}) = {np.log(V):.6f}  [ölcülen {ko:.6f}]",
            abs(float(ko) - np.log(V)) < 1e-5)
    kontrol("", abs(float(ku) - np.log(V)) < 1e-5)
    kontrol(f"kisa önekte sol yari NaN + maske False ({int(mk.sum())}/{P_ONEK} dolu)",
            int(mk.sum()) == 9 and bool(np.isnan(prof[0])) and not bool(mk[0]))
    lg2 = torch.zeros(1, 20, V); lg2[0, 8, :] = 0.0; lg2[0, 8, ids[0, 9]] = 50.0
    prof2, _, _, _, _ = kayip_profili(lg2, ids, n_on=10)
    kontrol("saga hizali: son hücre = önegin SON konum kaybi (≈0 kuruldu)",
            float(prof2[-1]) < 1e-3 and float(prof2[-2]) > 1.0)
    try:
        kayip_profili(lg, ids, n_on=1); kontrol("n_on=1 ⇒ ValueError", False)
    except ValueError:
        kontrol("n_on=1 ⇒ ValueError (sessiz NaN yok)", True)
    print(f"  [PROVA PAYDASI] denetlenen {gecti + len(dusen)} · gecen {gecti} · "
          f"düsen {len(dusen)}{' → ' + ', '.join(dusen) if dusen else ''}")
    return not dusen


def kayip_tam_mi(s, p_onek=P_ONEK):
    met, npz = f"{ARSIV}/{s['ad']}.kayip.meta.json", f"{ARSIV}/{s['ad']}.kayip.npz"
    if not (os.path.exists(met) and os.path.exists(npz)):
        return False, "dosya yok"
    try:
        m = json.load(open(met, encoding="utf-8"))
        if not m.get("tamam") or m.get("sema") != "kayip-1":
            return False, "tamam=False ya da eski sema"
        if int(m.get("p_onek", -1)) != p_onek:
            return False, f"pencere boyu {m.get('p_onek')} != {p_onek}"
        S0 = [json.loads(l) for l in open(s["yol"], encoding="utf-8")][s["bas"]:s["son"]]
        if m.get("kimlik_sha_s0") != kimlik_sha(S0):
            return False, "KIMLIK SHA UYUSMUYOR"
        z = np.load(npz)
        n = int(z["kayip_onek_son"].shape[0])
        if n != int(z["satir_ix"].shape[0]) or n != int(m.get("n_satir", -1)):
            return False, ""
        if n and int(z["satir_ix"].max()) >= len(S0):
            return False, "indeks ham dilimin disinda"
        return True, "tam"
    except Exception as e:
        return False, f"okunamadi: {e}"


def atomik_kaydet(yol, **diziler):
    gec = yol + f".tmp{os.getpid()}"
    with open(gec, "wb") as f:
        np.savez(f, **diziler)
        f.flush(); os.fsync(f.fileno())
    os.rename(gec, yol)


def kos_kayip(a, model, tok, K, cev, damga):
    import torch
    sayac = dict(istenen=0, uretilen=0, atlanan_tam=0)
    red = dict(kisa_metin=0, kisa_dizi=0, kisa_onek=0, nan_profil=0)
    kayit, t00 = [], time.time()
    eta_basildi = False
    for s in K:
        if a.yalniz and s["ad"] != a.yalniz:
            continue
        tam, nicin = kayip_tam_mi(s)
        if a.devam and tam:
            sayac["atlanan_tam"] += 1
            print(f"  [ATLA] {s['ad']} — {nicin}")
            kayit.append(dict(kol=s["ad"], atlandi=True, neden=nicin, tamam=True))
            continue
        S0 = [json.loads(l) for l in open(s["yol"], encoding="utf-8")][s["bas"]:s["son"]]
        sayac["istenen"] += len(S0)
        S, S_ix = [], []
        for j, r in enumerate(S0):
            if len((r.get(ALAN) or "").strip()) < MIN_UZUNLUK:
                red["kisa_metin"] += 1
                continue
            S.append(r); S_ix.append(j)
        sha_in = kimlik_sha(S0)
        PR = np.full((len(S), P_ONEK), np.nan, dtype=np.float32)
        MK = np.zeros((len(S), P_ONEK), dtype=bool)
        SEG = np.full((len(S), 3), np.nan, dtype=np.float32)
        KO = np.full(len(S), np.nan, dtype=np.float32)
        KU = np.full(len(S), np.nan, dtype=np.float32)
        NO = np.zeros(len(S), dtype=np.int32)
        NU = np.zeros(len(S), dtype=np.int32)
        IX = np.zeros(len(S), dtype=np.int32)
        t0, i = time.time(), 0
        for r, j0 in zip(S, S_ix):
            met = (r.get(ALAN) or "").strip()
            e_on = tok(r["onek"], return_tensors="pt").input_ids.to(a.dev)
            e = tok(r["onek"] + " " + met, return_tensors="pt").input_ids.to(a.dev)
            n_on = int(e_on.shape[1])
            if e.shape[1] <= n_on + 6:
                red["kisa_dizi"] += 1
                continue
            if n_on < 2:
                red["kisa_onek"] += 1
                continue
            with torch.no_grad():
                o = model(e)
            PR[i], MK[i], KO[i], KU[i], SEG[i] = kayip_profili(o.logits, e, n_on)
            NO[i], NU[i], IX[i] = n_on, int(e.shape[1]) - n_on, j0
            if not np.isfinite(KO[i]):
                red["nan_profil"] += 1
            i += 1
            del o
        PR, MK, SEG, KO, KU, NO, NU, IX = (PR[:i], MK[:i], SEG[:i], KO[:i],
                                           KU[:i], NO[:i], NU[:i], IX[:i])
        sayac["uretilen"] += i
        atomik_kaydet(f"{ARSIV}/{s['ad']}.kayip.npz", kayip_onek_son=PR,
                      maske_onek_son=MK, kayip_seg=SEG, kayip_onek_ort=KO,
                      kayip_uretim_ort=KU, n_tok_onek=NO, n_tok_uretim=NU,
                      satir_ix=IX)
        meta = dict(kol=s["ad"], ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    sema="kayip-1", korpus=s["korpus"], dosya=s["dosya"],
                    bas=s["bas"], son=s["son"], n_satir=int(i), n_ham=len(S0),
                    kimlik_sha_s0=sha_in, p_onek=P_ONEK, alan=ALAN, tamam=True,
                    hizalama="SAGA (basamak isareti önekin SONUNDA)",
                    anahtarlar=dict(
                        kayip_onek_son=f"[n,{P_ONEK}] son {P_ONEK} önek konumunun NLL'i, "
                                       "saga hizali, bos yer NaN",
                        maske_onek_son=f"[n,{P_ONEK}] bool — NaN olmayan konumlar",
                        kayip_seg="[n,3] önegin üc esit diliminin ortalama NLL'i",
                        kayip_onek_ort="[n] ALIM: tüm önek konumlarinin ortalama NLL'i",
                        kayip_uretim_ort="",
                        satir_ix=""),
                    saniye=round(time.time() - t0, 1))
        gec = f"{ARSIV}/{s['ad']}.kayip.meta.json.tmp{os.getpid()}"
        json.dump(meta, open(gec, "w"), ensure_ascii=False, indent=1)
        os.rename(gec, f"{ARSIV}/{s['ad']}.kayip.meta.json")
        kayit.append(meta)
        sn = time.time() - t0
        print(f"  [{s['ad']:44s}] n={i:4d} · {sn:5.0f}s · sha {sha_in} · "
              f"toplam {time.time()-t00:.0f}s", flush=True)
        if not eta_basildi:
            kalan = sum(1 for s2 in K if not kayip_tam_mi(s2)[0])
            print(f"  [ETA · ÖLCÜLDÜ] ilk shard {sn:.0f}s ⇒ kalan {kalan} shard icin "
                  f"≈ {kalan * sn / 60:.1f} dk", flush=True)
            eta_basildi = True
    tam_liste = [(s["ad"], *kayip_tam_mi(s)) for s in K]
    n_tam = sum(1 for _, t, _ in tam_liste if t)
    n_arsivde = sum(json.load(open(f"{ARSIV}/{s['ad']}.kayip.meta.json",
                                   encoding="utf-8"))["n_satir"]
                    for s in K if os.path.exists(f"{ARSIV}/{s['ad']}.kayip.meta.json"))
    payda("arsiv_kayip", n_shard=len(K), n_shard_tam=n_tam, n_satir_arsivde=n_arsivde,
          uretilen_bu_kosuda=sayac["uretilen"], red_kisa_metin=red["kisa_metin"],
          red_kisa_dizi=red["kisa_dizi"], red_kisa_onek=red["kisa_onek"],
          red_nan_profil=red["nan_profil"],
          bekle={"n_shard": len(K), "n_shard_tam": len(K)})
    sayac["arsivde"] = n_arsivde
    M = dict(damga=damga, sinif="HASAT · ALIM YARISI — verdict YOK", cevre=cev,
             alan=ALAN, model=SU.MODEL, kok=ARSIV, p_onek=P_ONEK, sayac=sayac,
             red=red, n_shard=len(K), n_shard_tam=n_tam, kollar=kayit,
             saklamaz="konum-basina logit · sözlük-üstü dagilim · gemma satirlari",
             tuketici_notu="")
    yol = f"{ARSIV}/manifest_kayip.json"
    json.dump(M, open(yol, "w"), ensure_ascii=False, indent=1)
    json.dump({k: v for k, v in M.items() if k != "kollar"},
              open(f"{ROOT}/unreleased/ARSIV_KAYIP_MANIFEST_2026-08-17.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n★ ALIM ARSIVI: {n_tam}/{len(K)} shard tam · {n_arsivde} satir · red {red}")
    print(f"→ {yol}")
    return 0


def main():
    import torch, glob as _g
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--bekle-gpu", type=int, required=True)
    ap.add_argument("--devam", action="store_true")
    ap.add_argument("--yalniz", default=None, help="tek shard adi (hata ayiklama)")
    ap.add_argument("--kayip", action="store_true",
                    help=""
                         "")
    a = ap.parse_args()
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(ARSIV, exist_ok=True)
    print(f"AKTIVASYON ARSIVI · damga {damga} · katmanlar {KATMANLAR} · alan {ALAN}")

    cev = kilitle(a.dev, beklenen_fiziksel=a.bekle_gpu, tam=True, etiket="arsiv")
    K = kaynaklar()
    on_kosul = {
        "hf_home": bool(os.environ.get("HF_HOME")),
        "gpu_kilidi": cev.get("fiziksel_gpu") == a.bekle_gpu,
        "vram_bos_gb>=40": float(cev["cevre"]["vram_bos_gb"]) >= 40,
        "cikis_koku_yazilabilir": os.access(ARSIV, os.W_OK),
        "shard_var": len(K) > 0,
    }
    payda("on_ucus", n_on_kosul=len(on_kosul), n_gecen_kosul=sum(on_kosul.values()),
          red_dusen=len(on_kosul) - sum(on_kosul.values()),
          bekle={"n_on_kosul": 5, "n_gecen_kosul": 5})
    if not all(on_kosul.values()):
        raise RuntimeError("ÖN-UCUS DÜSTÜ: " +
                           ", ".join(k for k, v in on_kosul.items() if not v))
    bos_gb = float(os.statvfs(ARSIV).f_bavail) * os.statvfs(ARSIV).f_frsize / 1e9
    satir_bayt = len(KATMANLAR) * 5 * 4096 * 2
    tahmin_gb = sum(s["son"] - s["bas"] for s in K) * satir_bayt / 1e9
    print(f"  [DISK] bos {bos_gb:.0f} GB · tahmini arsiv {tahmin_gb:.2f} GB · "
          f"satir basina {satir_bayt/1024:.0f} KB")
    if tahmin_gb > bos_gb * 0.5:
        raise RuntimeError(f"DISK BÜTCESI: {tahmin_gb:.1f} GB > bosun yarisi")

    snap = sorted(_g.glob(f"{SU.MODEL}/snapshots/*"))[-1]
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float16).to(a.dev).eval()
    nh = model.config.num_attention_heads
    dh = model.config.hidden_size // nh

    if a.kayip:
        return kos_kayip(a, model, tok, K, cev, damga)

    tut = {}

    def yap_kanca(L):
        def kanca(mod, girdi):
            tut[L] = girdi[0].detach()
        return kanca
    for L in KATMANLAR:
        model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(yap_kanca(L))

    sayac = dict(istenen=0, uretilen=0, atlanan_tam=0)
    red = dict(kisa_metin=0, kisa_dizi=0)
    kayit, t00 = [], time.time()
    for s in K:
        if a.yalniz and s["ad"] != a.yalniz:
            continue
        tam, nicin = shard_tam_mi(s)
        if a.devam and tam:
            sayac["atlanan_tam"] += 1
            print(f"  [ATLA] {s['ad']} — {nicin}")
            kayit.append(dict(kol=s["ad"], atlandi=True, neden=nicin, tamam=True))
            continue
        S0 = [json.loads(l) for l in open(s["yol"], encoding="utf-8")][s["bas"]:s["son"]]
        sayac["istenen"] += len(S0)
        S, S_ix = [], []
        for j, r in enumerate(S0):
            if len((r.get(ALAN) or "").strip()) < MIN_UZUNLUK:
                red["kisa_metin"] += 1
                continue
            S.append(r); S_ix.append(j)
        sha_in = kimlik_sha(S0)
        H = np.zeros((len(S), len(KATMANLAR), 4096), dtype=np.float16)
        Hs = np.zeros((len(S), len(KATMANLAR), 3, 4096), dtype=np.float16)
        A = np.zeros((len(S), len(KATMANLAR), 4096), dtype=np.float16)
        NT = np.zeros(len(S), dtype=np.int32)
        IX = np.zeros(len(S), dtype=np.int32)
        t0, tut_i = time.time(), 0
        for r, j0 in zip(S, S_ix):
            met = (r.get(ALAN) or "").strip()
            e_on = tok(r["onek"], return_tensors="pt").input_ids.to(a.dev)
            e = tok(r["onek"] + " " + met, return_tensors="pt").input_ids.to(a.dev)
            n_on = int(e_on.shape[1])
            if e.shape[1] <= n_on + 6:
                red["kisa_dizi"] += 1
                continue
            with torch.no_grad():
                o = model(e, output_hidden_states=True)
            n_ur = int(e.shape[1]) - n_on
            kes = [n_on + int(round(n_ur * j / 3)) for j in range(4)]
            H[tut_i] = katman_ortalamasi(o.hidden_states, n_on, KATMANLAR)
            for li, L in enumerate(KATMANLAR):
                h = o.hidden_states[L + 1][0, n_on:, :].float()
                for j in range(3):
                    b, c = kes[j] - n_on, max(kes[j + 1] - n_on, kes[j] - n_on + 1)
                    Hs[tut_i, li, j] = h[b:c].mean(0).cpu().numpy().astype(np.float16)
                A[tut_i, li] = tut[L][0, n_on:, :].float().mean(0).cpu().numpy().astype(np.float16)
            NT[tut_i] = n_ur
            IX[tut_i] = j0
            tut_i += 1
            del o
        H, Hs, A, NT, IX = H[:tut_i], Hs[:tut_i], A[:tut_i], NT[:tut_i], IX[:tut_i]
        sayac["uretilen"] += tut_i
        atomik_kaydet(f"{ARSIV}/{s['ad']}.npz", h_tam=H, h_seg=Hs, a_tam=A,
                      n_uretilen=NT, satir_ix=IX)
        meta = dict(kol=s["ad"], ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    sema=2, korpus=s["korpus"], dosya=s["dosya"], bas=s["bas"], son=s["son"],
                    n_satir=int(tut_i), n_ham=len(S0), kimlik_sha_s0=sha_in,
                    katmanlar=list(KATMANLAR),
                    alan=ALAN, n_head=nh, d_head=dh, tamam=True,
                    anahtarlar=dict(h_tam="[n,6,4096] artik-akim ort", satir_ix="",
                                    h_seg="[n,6,3,4096] üc dilim ort",
                                    a_tam="[n,6,4096] o_proj GIRDISI (head-geri-türetilebilir)",
                                    n_uretilen="[n] üretilen jeton sayisi"),
                    saniye=round(time.time() - t0, 1))
        gec = f"{ARSIV}/{s['ad']}.meta.json.tmp{os.getpid()}"
        json.dump(meta, open(gec, "w"), ensure_ascii=False, indent=1)
        os.rename(gec, f"{ARSIV}/{s['ad']}.meta.json")
        kayit.append(meta)
        print(f"  [{s['ad']:44s}] n={tut_i:4d} · {time.time()-t0:5.0f}s · "
              f"sha {sha_in} · toplam {time.time()-t00:.0f}s", flush=True)

    tam_liste = [(s["ad"], *shard_tam_mi(s)) for s in K]
    n_tam = sum(1 for _, t, _ in tam_liste if t)
    boyut_gb = sum(os.path.getsize(f"{ARSIV}/{f}") for f in os.listdir(ARSIV)
                   if f.endswith(".npz")) / 1e9
    import math as _m
    bekle_shard = sum(_m.ceil(sum(1 for _ in open(y, encoding="utf-8")) / PARCA)
                      for y in sorted({s["yol"] for s in K}))
    n_arsivde = sum(json.load(open(f"{ARSIV}/{s['ad']}.meta.json", encoding="utf-8"))["n_satir"]
                    for s in K if os.path.exists(f"{ARSIV}/{s['ad']}.meta.json"))
    payda("arsiv", n_shard=len(K), n_shard_tam=n_tam, n_satir_arsivde=n_arsivde,
          n_shard_beklenen=bekle_shard, uretilen_bu_kosuda=sayac["uretilen"],
          red_kisa_metin=red["kisa_metin"], red_kisa_dizi=red["kisa_dizi"],
          bekle={"n_shard": bekle_shard, "n_shard_tam": bekle_shard})
    sayac["arsivde"] = n_arsivde
    M = dict(damga=damga, sinif="HASAT — verdict YOK", cevre=cev, katmanlar=list(KATMANLAR),
             alan=ALAN, model=SU.MODEL, n_head=nh, d_head=dh, kok=ARSIV,
             sayac=sayac, red=red, n_shard=len(K), n_shard_tam=n_tam,
             disk_gb=round(boyut_gb, 3), kollar=kayit,
             saklamaz=""
                      "",
             tuketici_notu="head h'nin katkisi = W_o[:,128h:128(h+1)] @ a_tam[128h:128(h+1)]")
    json.dump(M, open(MANIFEST, "w"), ensure_ascii=False, indent=1)
    json.dump({k: v for k, v in M.items() if k != "kollar"},
              open(CIKTI, "w"), ensure_ascii=False, indent=1)
    print(f"\n★ ARSIV: {n_tam}/{len(K)} shard tam · {sayac['uretilen']} satir · "
          f"{boyut_gb:.2f} GB · red {red}")
    print(f"→ {MANIFEST}\n→ {CIKTI}")


if __name__ == "__main__":
    if "--prova" in sys.argv:
        print("ALIM PROVASI · kayip_profili — GPU YOK, dejenere girdi (§7.1)")
        sys.exit(0 if prova_kayip() else 1)
    main()
