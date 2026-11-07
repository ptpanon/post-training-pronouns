#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import gpu_lock as RK
import argparse, json, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
from reading_style_object import payda
import mini_dpo_feasibility as FZ

KOL_KOK = os.environ.get("KOL_KOK", __DNH_DATA__ + "/uc_kol")
CIK = f"{ROOT}/results/GOZLEM_PILOT_2026-08-29.json"
KOK = __DNH_DATA__ + "/gozlem_pilot_ckpt"
KESME_ESIK = 0.10




def _son_asistan(m):
    for x in reversed(m or []):
        if x.get("role") == "assistant":
            return (x.get("content") or "").strip()
    return ""


def veri(kol, n):
    yol = f"{KOL_KOK}/{kol}.jsonl"
    if not os.path.exists(yol):
        raise SystemExit(f"{yol}")
    R = [json.loads(l) for l in open(yol, encoding="utf-8")][:n]
    return [dict(prompt=[{"role": "user", "content": r["prompt"]}],
                 chosen=[{"role": "assistant", "content": _son_asistan(r["chosen"])}],
                 rejected=[{"role": "assistant", "content": _son_asistan(r["rejected"])}])
            for r in R], R


def boru_denetimi(tok, ham, maxlen, trunc="keep_start"):
    O = {}
    p0 = tok.apply_chat_template([{"role": "user", "content": ham[0]["prompt"]}],
                                 tokenize=False, add_generation_prompt=True)
    ids = tok(p0, add_special_tokens=True)["input_ids"]
    bos = tok.bos_token_id
    n_bos = sum(1 for t in ids[:3] if t == bos)
    O["template"] = dict(template_var=tok.chat_template is not None,
                       ilk_80=p0[:80], n_bos_bas=n_bos,
                       HAL="CIFT-BOS" if n_bos > 1 else "TEK-BOS")

    def _p(r):
        return tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
    lp = np.array([len(tok(_p(r), add_special_tokens=False)["input_ids"]) for r in ham])
    lc = np.array([len(tok(_son_asistan(r["chosen"]), add_special_tokens=False)["input_ids"])
                   for r in ham])
    lr = np.array([len(tok(_son_asistan(r["rejected"]), add_special_tokens=False)["input_ids"])
                   for r in ham])
    dusen = (lp >= maxlen) if trunc == "keep_start" else np.zeros_like(lp, dtype=bool)
    kalan_yer = np.maximum(maxlen - lp, 0)
    kismen = (~dusen) & (kalan_yer < np.minimum(lc, lr))
    o_dus = float(dusen.mean()); o_kis = float(kismen.mean())
    O["kesme"] = dict(max_length=maxlen, truncation_mode=trunc,
                      istem_jeton_ort=round(float(lp.mean()), 1),
                      istem_jeton_p50=int(np.median(lp)),
                      istem_jeton_p95=int(np.percentile(lp, 95)),
                      chosen_jeton_ort=round(float(lc.mean()), 1),
                      rejected_jeton_ort=round(float(lr.mean()), 1),
                      cevaba_kalan_jeton_ort=round(float(kalan_yer.mean()), 1),
                      oran_SESSIZ_DUSEN=round(o_dus, 4),
                      oran_cevap_KISMEN_kesilen=round(o_kis, 4),
                      esik=KESME_ESIK,
                      HAL=("VERI-DÜSÜYOR" if o_dus > KESME_ESIK else
                           ("KESME-ASIRI" if o_kis > KESME_ESIK else "KABUL")))
    print(f"{lp.mean():.0f} {np.percentile(lp,95):.0f}"
          f"{lc.mean():.0f} {lr.mean():.0f} {maxlen} {trunc}"
          f"{o_dus*100:.1f} {o_kis*100:.1f}"
          f"{KESME_ESIK*100:.0f}"
          f"", flush=True)

    O["dolgu"] = dict(pad_token=tok.pad_token, pad_token_id=tok.pad_token_id,
                      eos_token=tok.eos_token, padding_side=tok.padding_side,
                      pad_eos_ayni=tok.pad_token_id == tok.eos_token_id)
    print(f"      dolgu: pad={tok.pad_token!r}({tok.pad_token_id}) taraf={tok.padding_side} "
          f"· pad==eos? {O['dolgu']['pad_eos_ayni']}", flush=True)

    ayni = sum(1 for r in ham if _son_asistan(r["chosen"]) == _son_asistan(r["rejected"]))
    O["ozdes_cift"] = dict(n=ayni, oran=round(ayni / len(ham), 4))
    payda("gozlem_boru", n_cift=len(ham), red_ozdes=ayni,
          red_sessiz_dusen=int(dusen.sum()), hal_kesilen_kismen=round(o_kis, 4),
          hal_cift_bos=int(n_bos > 1))
    return O


def parti_denetimi(tr):
    import torch
    dl = tr.get_train_dataloader()
    b = next(iter(dl))
    O = {"anahtarlar": sorted(b.keys())}
    for k, v in b.items():
        if hasattr(v, "shape"):
            O[k] = dict(sekil=list(v.shape), dtype=str(v.dtype))
    for k in b:
        if hasattr(b[k], "shape") and ("label" in k or "completion_mask" == k):
            m = (b[k] == -100) if "label" in k else (b[k] == 0)
            O[f"{k}_maske"] = dict(
                anlam="prompt/dolgu (kayiba girmeyen) konum orani",
                n_maskeli=int(m.sum()), n_toplam=int(m.numel()),
                oran=round(float(m.float().mean()), 4))
    print(f"      parti anahtarlari: {O['anahtarlar']}", flush=True)
    for k, v in O.items():
        if k.endswith("_maske"):
            print(f"      {k}: {v['n_maskeli']}/{v['n_toplam']} = %{v['oran']*100:.1f} "
                  f"maskeli (-100)", flush=True)
    payda("gozlem_parti", n_anahtar=len(O["anahtarlar"]),
          hal_maske_alani=sum(1 for k in O if k.endswith("_maske")))
    return O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", default="EMIR_ARTI")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--adim", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=128,
                    help="★ PREREG-3'ten TASINAN sayi — sinanacak olan budur")
    ap.add_argument("--truncation-mode", default="keep_start")
    ap.add_argument("--sadece-denetim", action="store_true",
                    help="GPU'suz: yalniz boru denetimi (kesme/template/dolgu)")
    a = ap.parse_args()

    ic, ham = veri(a.kol, a.n)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FZ.SNAP)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"★ GÖZLEM-PILOT · kol={a.kol} · n={len(ic)} — BORU DENETIMI", flush=True)
    B = boru_denetimi(tok, ham, a.max_length, a.truncation_mode)

    O = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             DAMGA="",
             SINIF="GÖZLEM (prereg YOK · bar YOK · ad YOK · prediction YOK)",
             kol=a.kol, n=len(ic), boru=B)
    if a.sadece_denetim:
        json.dump(O, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n→ {CIK}  (yalniz denetim)"); return 0

    bos0, kul0 = FZ.gpu_bos_mib(RK.fiziksel_bekle("cuda:0"))
    _rejim, sag = FZ.prod_kapisi(RK.fiziksel_bekle("cuda:0"), "gozlem_pilot.py")
    kapi = FZ.kapi_hesapla(bos0, _rejim); tavan = kul0 + kapi
    print(f"[ÖN-UCUS] GPU0 bos={bos0} kullanilan={kul0} · prod={sag}\n"
          f"[KAPI] bütce = bos {bos0} − TAMPON {FZ.TAMPON_MIB} = {kapi} MiB ⇒ card tavani {tavan} MiB "
          f"⇒ EYLEM: asilirsa gözcü bayrak basar, is_ kendini durdurur (prod KESILMEZ)",
          flush=True)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import DPOTrainer, DPOConfig
    os.makedirs(KOK, exist_ok=True)
    g = FZ.Gozcu(tavan)
    sert = g.sert_kur(kapi)
    g.start(); t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(FZ.SNAP, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    cfg = DPOConfig(output_dir=KOK, per_device_train_batch_size=1,
                    gradient_accumulation_steps=8, max_steps=a.adim,
                    learning_rate=5e-7, beta=0.1,
                    max_length=a.max_length, truncation_mode=a.truncation_mode,
                    bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
                    precompute_ref_log_probs=True, logging_steps=5,
                    save_strategy="no", report_to=[], dataloader_num_workers=0,
                    seed=20260829)
    tr = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(ic),
                    processing_class=tok)
    tr.add_callback(g.callback())
    O["parti"] = parti_denetimi(tr)
    torch.cuda.reset_peak_memory_stats()
    cik = tr.train()
    O["tepe_tahsis_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    g.dur = True; g.join(timeout=3)
    sag2 = FZ.prod_saglik()
    O.update(sert_kapi=sert,
             yumusak_kapi=dict(atesledi=bool(g.dur_istendi), tepe_gorulen=g.tepe_kul),
             egitim=cik.metrics, card_tepe_mib=g.tepe_kul,
             bizim_tepe_mib=g.tepe_kul - kul0, kapi_mib=kapi, gozcu_ihlal=g.ihlal,
             prod_once=sag, prod_sonra=sag2, saniye=round(time.time() - t0, 1))
    payda("gozlem_pilot", n_adim=a.adim, n_cift=len(ic),
          hal_bizim_mib=g.tepe_kul - kul0, red_gozcu_ihlal=int(g.ihlal))
    json.dump(O, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"\n★ GÖZLEM-PILOT BITTI · {(time.time()-t0)/60:.1f} dk · "
          f"bizim tepe {g.tepe_kul-kul0} MiB (kapi {kapi}) · ihlal={g.ihlal}\n"
          f"   prod: {sag} → {sag2}\n→ {CIK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
