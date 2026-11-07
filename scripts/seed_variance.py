#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import gpu_lock as RK
import argparse, json, os, sys, time

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("HF_HOME", "<storage>/huggingface")
os.environ["HF_DATASETS_CACHE"] = __DNH_DATA__ + "/hf_ds_cache"
from reading_style_object import payda
import mini_dpo_feasibility as FZ
import gozlem_pilot as GP

KOL_KOK = os.environ.get("KOL_KOK", __DNH_DATA__ + "/uc_kol")
CKPT = os.environ.get("CKPT_KOK", __DNH_DATA__ + "/seed_varyans")
CIK = f"{ROOT}/results/SEED_VARYANS_2026-08-30.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kol", default="EMIR_ARTI")
    ap.add_argument("--n-cift", dest="n", type=int, default=1024)
    ap.add_argument("--adim", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--birikim", type=int, default=8)
    ap.add_argument("--etiket", default="probe")
    ap.add_argument("--kaydet", action="store_true",
                    help="checkpoint yaz (FSDP tam-durum toplama YAVAS; probe'da KAPALI)")
    a = ap.parse_args()

    yerel = int(os.environ.get("LOCAL_RANK", "0"))
    bas = yerel == 0
    if bas:
        _rejim, _sag = FZ.prod_kapisi(RK.fiziksel_bekle("cuda:0"), "seed_varyans")
        bos0, kul0 = FZ.gpu_bos_mib(RK.fiziksel_bekle("cuda:0"))
        kapi = FZ.kapi_hesapla(bos0, _rejim)
        _pay = bos0 - kapi
        print(f"[KAPI] rejim {_rejim} ⇒ bos {bos0} − pay {_pay} = {kapi} MiB "
              f"⇒ EYLEM: gözcü DURDURUCU, asilirsa is_ kendini durdurur", flush=True)

    import torch
    import torch.distributed as dist
    from datasets import Dataset, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    ic, ham = GP.veri(a.kol, a.n)
    tok = AutoTokenizer.from_pretrained(FZ.SNAP)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if bas:
        B = GP.boru_denetimi(tok, ham, a.max_length, "keep_start")
        print(f"  ★ boru: sessizce düsen %{B['kesme']['oran_SESSIZ_DUSEN']*100:.1f} · "
              f"kismen kesilen %{B['kesme']['oran_cevap_KISMEN_kesilen']*100:.1f}", flush=True)
    else:
        B = None

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    ds_yol = f"{CKPT}/ds_{a.kol}_{len(ic)}"
    if bas and not os.path.exists(f"{ds_yol}/dataset_info.json"):
        os.makedirs(CKPT, exist_ok=True)
        Dataset.from_list(ic).save_to_disk(ds_yol)
        print(f"  ★ küme diske yazildi: {ds_yol}", flush=True)
    dist.barrier()
    DS = load_from_disk(ds_yol)

    out = f"{CKPT}/{a.etiket}_t{a.seed}"
    os.makedirs(out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(FZ.SNAP, dtype=torch.bfloat16)
    model.config.use_cache = False
    torch.cuda.set_device(yerel)
    model = model.to(f"cuda:{yerel}")
    cfg = DPOConfig(
        output_dir=out, per_device_train_batch_size=1,
        gradient_accumulation_steps=a.birikim, max_steps=a.adim,
        learning_rate=5e-7, beta=0.1, max_length=a.max_length,
        bf16=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
        precompute_ref_log_probs=True, logging_steps=5,
        save_strategy=("steps" if a.kaydet else "no"),
        save_steps=max(a.adim, 1), save_only_model=True,
        save_total_limit=1, report_to=[], dataloader_num_workers=0, seed=a.seed,
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
                     "activation_checkpointing": True,
                     "use_orig_params": True})
    tr = DPOTrainer(model=model, args=cfg, train_dataset=DS, processing_class=tok)
    g = None
    if bas:
        g = FZ.Gozcu(kul0 + kapi); g.start()
        tr.add_callback(g.callback())
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    cik = tr.train()
    sn = time.time() - t0
    dist.barrier()
    if not bas:
        dist.destroy_process_group()
        return 0
    g.dur = True; g.join(timeout=3)
    tepe = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    card = FZ.gpu_bos_mib(RK.fiziksel_bekle("cuda:0"))[1]
    payda("seed_varyans", n_adim=a.adim, n_cift=len(ic),
          hal_saniye=round(sn, 1), hal_tahsis_tepe_mib=tepe,
          hal_card_tepe_mib=g.tepe_kul, red_gozcu_ihlal=int(g.ihlal))
    O = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             SINIF="KESIF (bar yok · ad yok · prediction yok)", etiket=a.etiket,
             kol=a.kol, seed=a.seed, n_cift=len(ic), adim=a.adim,
             max_length=a.max_length, birikim=a.birikim,
             optim="adamw_torch", fsdp="full_shard auto_wrap", n_card=2,
             boru=B, saniye=round(sn, 1),
             saniye_basina_adim=round(sn / max(a.adim, 1), 2),
             tahsis_tepe_mib=tepe, card_tepe_mib=g.tepe_kul, card_son=card,
             kapi_mib=kapi, gozcu_ihlal=g.ihlal, dur_istendi=g.dur_istendi,
             egitim=cik.metrics, ckpt=out)
    yol = CIK.replace(".json", f"_{a.etiket}_t{a.seed}.json")
    json.dump(O, open(yol, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"\n★ {a.etiket}/t{a.seed} BITTI · {sn/60:.2f} dk · {sn/max(a.adim,1):.2f} sn/adim · "
          f"tahsis tepe {tepe} MiB · card tepe {g.tepe_kul} (kapi {kapi}) · ihlal {g.ihlal}")
    print(f"→ {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
