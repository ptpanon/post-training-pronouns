#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, json, os, subprocess, sys, threading, time

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

CIFT = __DNH_DATA__ + "/c1_kip_takas/kip_takas.jsonl"
CIFT_SHA16 = "5aa4f3b747e265bc"
SNAP = ("<storage>/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT/"
        "snapshots/f2a0b46b0cfda21003c6141b1ff837b7e165524d")
DUZ = "emir→duz"
CIK = f"{ROOT}/results/MINI_DPO_FIZIBILITE_2026-08-29.json"
PAY = 0.85
TAMPON_MIB = 10240
KENDI_PAYI_MIB = 2048


SAHIP_PAYI_MIB = int(os.environ.get("PROJECT_SAHIP_PAYI_MIB", str(KENDI_PAYI_MIB)))


def kapi_hesapla(bos_mib, rejim="PAYLI"):
    t = (KENDI_PAYI_MIB if rejim == "PROD-YOK"
         else SAHIP_PAYI_MIB if rejim == "SAHIP-PENCERESI" else TAMPON_MIB)
    return max(int(bos_mib) - t, 0)
PROD_PENCERE = os.environ.get("PROD_PENCERE",
                              __DNH_DATA__ + "/PROD_PENCERE")


def pencere_gecerli(yol=None, simdi=None):
    import datetime as _dt
    y = yol or PROD_PENCERE
    if not os.path.exists(y):
        return False, None, 0
    try:
        ilk = open(y, encoding="utf-8").readline().strip().split()[0]
        bit = _dt.datetime.strptime(ilk, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc)
    except Exception:
        return False, None, 0
    n = simdi or _dt.datetime.now(_dt.timezone.utc)
    kalan = int((bit - n).total_seconds() // 60)
    return (kalan > 0), ilk, max(kalan, 0)


UCLAR = (("gomme", "http://172.16.180.28:8040/health"),
         ("uretim", "http://172.16.180.28:80/health"))


def sha16(y):
    import hashlib
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def gpu_bos_mib(i=0):
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used",
                        "--format=csv,noheader,nounits", "-i", str(i)],
                       capture_output=True, text=True).stdout.strip()
    bos, kul = (int(x) for x in o.split(","))
    return bos, kul


def gpu_bos_mib_kararli(i=0, n=4, bekle=4.0, tolerans=512, onceki_pid=None,
                        oku=None):
    import time as _t
    f = oku or (lambda k: gpu_bos_mib(k)[0])
    if onceki_pid:
        for _ in range(int(3600 / max(bekle, 0.1))):
            if not os.path.exists(f"/proc/{onceki_pid}"):
                break
            _t.sleep(bekle)
    ok = []
    for k in range(max(n, 2)):
        ok.append(int(f(i)))
        if len(ok) >= 2 and abs(ok[-1] - ok[-2]) <= tolerans:
            return ok[-1], True, ok
        if k < n - 1:
            _t.sleep(bekle)
    return min(ok), False, ok


def _prova_kararli():
    sabit = iter([70000, 70100, 70050])
    a = gpu_bos_mib_kararli(oku=lambda i: next(sabit), bekle=0.0)
    oynak = iter([10000, 70000, 12000, 68000])
    b = gpu_bos_mib_kararli(oku=lambda i: next(oynak), bekle=0.0)
    return {"i_kararli_yakalanir": a[1] is True and a[0] == 70100,
            "ii_kararsiz_yakalanir": b[1] is False and b[0] == 10000,
            "iii_kararsizda_en_kucuk": b[0] == min(b[2])}


def prod_rejimi(dev=0):
    ac, bit, kalan = pencere_gecerli()
    if ac:
        _, kul = gpu_bos_mib(dev)
        return "SAHIP-PENCERESI", {"bayrak": f"bitis {bit} · kalan {kalan} dk"}, kul
    sag = prod_saglik()
    hepsi_200 = all(v == 200 for v in sag.values())
    ulasilamaz = all(isinstance(v, str) and v.startswith("HATA:") for v in sag.values())
    _, kul = gpu_bos_mib(dev)
    kartta_yuk = kul > 1024
    if hepsi_200:
        return "PAYLI", sag, kul
    if ulasilamaz and not kartta_yuk:
        return "PROD-YOK", sag, kul
    return "SAGLIKSIZ", sag, kul


def prod_kapisi(dev=0, etiket=""):
    rej, sag, kul = prod_rejimi(dev)
    print(f"[PROD KAPISI{(' · ' + etiket) if etiket else ''}] rejim **{rej}** · "
          f"uclar {sag} · kartta {kul} MiB ⇒ "
          f"{'DUR' if rej == 'SAGLIKSIZ' else 'KOS'}", flush=True)
    payda("prod_kapisi", n_uc=len(sag), hal_rejim_saglikli=int(rej != "SAGLIKSIZ"),
          red_sagliksiz=int(rej == "SAGLIKSIZ"))
    if rej == "SAGLIKSIZ":
        raise SystemExit(f"★ PROD SAGLIKSIZ {sag} (kartta {kul} MiB) ⇒ BASLAMAZ")
    return rej, sag


def prod_saglik():
    import urllib.request
    d = {}
    for ad, u in UCLAR:
        try:
            with urllib.request.urlopen(u, timeout=5) as r:
                d[ad] = r.status
        except Exception as e:
            d[ad] = f"HATA:{type(e).__name__}"
    return d


def ciftleri_kur():
    R = [json.loads(l) for l in open(CIFT, encoding="utf-8")]
    ic, red_ayni = [], 0
    for r in R:
        a_duz, b_duz = r["kip_a"] == DUZ, r["kip_b"] == DUZ
        if a_duz == b_duz:
            red_ayni += 1
            continue
        d, ri = ("a", "b") if a_duz else ("b", "a")
        ic.append({"prompt": "", "chosen": r["metin_" + d], "rejected": r["metin_" + ri]})
    payda("mini_dpo_cift", n_satir=len(R), hal_capraz=len(ic),
          hal_tekil_metin=len({x["chosen"] for x in ic}), red_ayni_tabaka=red_ayni,
          bekle={"hal_capraz": 98})
    if len(ic) != 98:
        raise SystemExit(f"{len(ic)}")
    return ic


class Gozcu(threading.Thread):

    def __init__(self, tavan_mib, durdurucu=True):
        super().__init__(daemon=True)
        self.tavan, self.tepe_kul, self.dur, self.ihlal = tavan_mib, 0, False, False
        self.durdurucu = durdurucu
        self.dur_istendi = False

    def _tur(self, kul):
        self.tepe_kul = max(self.tepe_kul, kul)
        if kul > self.tavan:
            self.ihlal = True
            if self.durdurucu:
                self.dur_istendi = True
        return self.ihlal

    def run(self):
        while not self.dur:
            try:
                _, kul = gpu_bos_mib(0)
                self._tur(kul)
            except Exception:
                pass
            time.sleep(1.0)

    def sert_kur(self, kapi_mib, dev=0):
        import torch
        toplam = int(torch.cuda.get_device_properties(dev).total_memory / 2**20)
        oran = kapi_mib / toplam
        torch.cuda.set_per_process_memory_fraction(oran, dev)
        print(f"{kapi_mib} {toplam}"
              f"{oran:.3f}"
              f"",
              flush=True)
        self.sert = dict(oran=round(oran, 4), kapi_mib=kapi_mib, toplam_mib=toplam)
        return self.sert

    def callback(self):
        from transformers import TrainerCallback
        dis = self

        class K(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                try:
                    _, kul = gpu_bos_mib(0)
                except Exception:
                    return control
                dis._tur(kul)
                if dis.dur_istendi:
                    control.should_training_stop = True
                    print(f"{kul}"
                          f"{dis.tavan}",
                          flush=True)
                return control
        return K()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--adim", type=int, default=3)
    ap.add_argument("--etiket", required=True)
    a = ap.parse_args()

    s = sha16(CIFT)
    bos0, kul0 = gpu_bos_mib(0)
    sag = prod_saglik()
    print(f"[ÖN-UCUS] HF_HOME={os.environ.get('HF_HOME')} · "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[ÖN-UCUS] GPU0 bos={bos0} MiB · kullanilan={kul0} MiB · prod={sag}", flush=True)
    print(f"[ÖN-UCUS] cift sha={s} (mühürlü {CIFT_SHA16}) · uzunluk={a.max_length}", flush=True)
    if s != CIFT_SHA16:
        raise SystemExit(f"★ VERI KAYDI: {s} ≠ {CIFT_SHA16} ⇒ DUR")
    if not os.path.isdir(SNAP):
        raise SystemExit(f"★ SNAPSHOT YOK: {SNAP}")
    if any(v != 200 for v in sag.values()):
        raise SystemExit(f"{sag}")

    ic = ciftleri_kur()
    kapi_mib = int(PAY * bos0)
    tavan_card = kul0 + kapi_mib
    print(f"{int(PAY*100)} {bos0} {kapi_mib}"
          f"{tavan_card}",
          flush=True)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    g = Gozcu(tavan_card); g.start()
    t0 = time.time()
    hata = None
    try:
        tok = AutoTokenizer.from_pretrained(SNAP)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16).to("cuda")
        model.config.use_cache = False
        ds = Dataset.from_list(ic)
        cfg = DPOConfig(
            output_dir=f"{__DNH_DATA__}/mini_dpo_prova/{a.etiket}",
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            max_steps=a.adim, learning_rate=5e-7, beta=0.1,
            max_length=a.max_length, bf16=True, gradient_checkpointing=True,
            optim="paged_adamw_8bit", precompute_ref_log_probs=True,
            logging_steps=1, save_strategy="no", report_to=[],
            dataloader_num_workers=0, seed=20260829)
        tr = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
        torch.cuda.reset_peak_memory_stats()
        tr.train()
        tahsis_tepe = torch.cuda.max_memory_allocated() / 2**20
        ayrilan_tepe = torch.cuda.max_memory_reserved() / 2**20
    except torch.cuda.OutOfMemoryError as e:
        hata = "OOM"; tahsis_tepe = ayrilan_tepe = float("nan")
        print(f"★ OOM: {str(e)[:200]}", flush=True)
    except Exception as e:
        hata = f"{type(e).__name__}: {str(e)[:300]}"
        tahsis_tepe = ayrilan_tepe = float("nan")
        print(f"★ HATA: {hata}", flush=True)
    finally:
        g.dur = True; g.join(timeout=3)
    sn = time.time() - t0
    card_tepe = g.tepe_kul
    card_bizim = card_tepe - kul0

    if hata == "OOM" or card_bizim > bos0:
        hal = "SIGMADI"
    elif hata:
        hal = "ARIZA"
    elif card_bizim > kapi_mib:
        hal = "SINIRDA"
    else:
        hal = "SIGDI"
    payda(f"mini_dpo_fizibilite_{a.etiket}", n_adim=a.adim, n_cift=len(ic),
          hal_card_tepe_mib=card_tepe, hal_bizim_mib=card_bizim,
          hal_tahsis_tepe_mib=round(tahsis_tepe, 1) if tahsis_tepe == tahsis_tepe else -1,
          red_ihlal=int(g.ihlal), red_ariza=int(bool(hata)))
    sag2 = prod_saglik()
    out = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               PREREG="prereg_mini_dpo_8b_2026-08-29.md", SINIF="KESIF",
               etiket=a.etiket, max_length=a.max_length, adim=a.adim,
               gpu0_bos_baslangic_mib=bos0, gpu0_prod_taban_mib=kul0,
               kapi_mib=kapi_mib, card_tepe_mib=card_tepe, bizim_tepe_mib=card_bizim,
               torch_tahsis_tepe_mib=None if tahsis_tepe != tahsis_tepe else round(tahsis_tepe, 1),
               torch_ayrilan_tepe_mib=None if ayrilan_tepe != ayrilan_tepe else round(ayrilan_tepe, 1),
               saniye=round(sn, 1), HAL=hal, hata=hata,
               prod_saglik_once=sag, prod_saglik_sonra=sag2, gozcu_ihlal=g.ihlal)
    eski = json.load(open(CIK, encoding="utf-8")) if os.path.exists(CIK) else {"olcum": []}
    eski["olcum"] = [x for x in eski["olcum"] if x["etiket"] != a.etiket] + [out]
    json.dump(eski, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n★ {a.etiket} · uzunluk {a.max_length} · **{hal}**")
    print(f"   card tepe {card_tepe} MiB · bizim {card_bizim} MiB · kapi {kapi_mib} MiB"
          f" · torch tahsis {out['torch_tahsis_tepe_mib']} MiB · {sn:.0f} sn")
    print(f"   prod: {sag} → {sag2}")
    print(f"→ {CIK}")
    return 3 if hal == "ARIZA" else 0


if __name__ == "__main__":
    raise SystemExit(main())
