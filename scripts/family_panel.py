#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import argparse, hashlib, json, glob, os, sys, time
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda
from verdict_name_crosscount import capraz_say
from gpu_lock import kilitle
import gpu_lock as RKG
import serita_harman102 as SA
import a3_harman_oran as A3
import k1b_generation as K1BU
import k1b_resolve as K1BC
from step1_stage1_resolve import _merkez_fark, prova_ilkel

OUT_KOK = os.environ.get("PROJECT_OUT_KOK", __DNH_DATA__ + "/c1_panel")
CIKTI = PREREG = NOTICE = PREDICTION = None

EKSENLER = ("ARO", "DOM", "TON")
N_CEKIM, YIGIN = 12, 17
K_NULL, B_BOOT, SEED = 400, 2000, 20260815
BOS_BAR = 0.05
N_CIFT_BAR = 3
ADLAR = ("TÜM-AILELER-SILER", "IMZA-AILEYE-GÖRE",
         "DOM-HER-YERDE-ÖLÜ-TEYIT", "KAPI/ÖLCÜLEMEZ")
ADLAR_ARO = ("ARO-DOKUNULMAMIS", "ARO-IKI-YÖNLÜ", "ARO-TEK-YÖNLÜ-POZ",
             "ARO-TEK-YÖNLÜ-NEG", "KAPI/ÖLCÜLEMEZ")

PANEL_C1 = [
    dict(ad="Qwen2.5-7B",      base="models--Qwen--Qwen2.5-7B",
         instruct="models--Qwen--Qwen2.5-7B-Instruct",              tarif="RLHF (Alibaba)"),
    dict(ad="Mistral-7B-v0.3", base="models--mistralai--Mistral-7B-v0.3",
         instruct="models--mistralai--Mistral-7B-Instruct-v0.3",    tarif="SFT (Mistral) — ★ ANA HATTIN KENDI ALETI"),
    dict(ad="OLMo-3-7B",       base="models--allenai--Olmo-3-1025-7B",
         instruct="models--allenai--Olmo-3-7B-Instruct",            tarif="SFT→DPO→RLVR (tam acik tarif)"),
    dict(ad="Llama-3.1-8B",    base="models--meta-llama--Llama-3.1-8B",
         instruct="models--meta-llama--Llama-3.1-8B-Instruct",      tarif="SFT+DPO (Meta)"),
    dict(ad="Gemma-3-4B",      base="models--google--gemma-3-4b-pt",
         instruct="models--google--gemma-3-4b-it",                  tarif="Google — ★ ÖLCEK KONTROLÜ (aile ici)"),
    dict(ad="Gemma-3-12B",     base="models--google--gemma-3-12b-pt",
         instruct="models--google--gemma-3-12b-it",                 tarif="Google — ★ SAHIP HIPOTEZI (Gemma-EKLER)"),
]

PANEL_OLCEK = [
    dict(ad="Qwen2.5-32B",     base="models--Qwen--Qwen2.5-32B",
         instruct="models--Qwen--Qwen2.5-32B-Instruct",
         tarif="RLHF (Alibaba) — ★ C1'deki Qwen2.5-7B'nin ×4,6 ölcegi"),
    dict(ad="Qwen2.5-72B",     base="models--Qwen--Qwen2.5-72B",
         instruct="models--Qwen--Qwen2.5-72B-Instruct",
         tarif="RLHF (Alibaba) — ★ ×10 ölcek; TP=2 ister"),
    dict(ad="Gemma-4-26B-A4B", base="models--google--gemma-4-26B-A4B",
         instruct="models--google--gemma-4-26B-A4B-it",
         tarif="Google MoE (26B toplam / 4B etkin) — ★ Gemma ailesinin ölcek ucu"),
]

PANEL_OLCEK2 = [
    dict(ad="Qwen2.5-1.5B", base="models--Qwen--Qwen2.5-1.5B",
         instruct="models--Qwen--Qwen2.5-1.5B-Instruct",
         aile="Qwen2.5", sira=1, asama=None, tip="dense", p_toplam=1.5, p_etkin=1.5,
         tarif="RLHF (Alibaba) — ölcek ekseninin ALT ucu"),
    dict(ad="Qwen2.5-3B", base="models--Qwen--Qwen2.5-3B",
         instruct="models--Qwen--Qwen2.5-3B-Instruct",
         aile="Qwen2.5", sira=2, asama=None, tip="dense", p_toplam=3.0, p_etkin=3.0,
         tarif="RLHF (Alibaba) — alt uc ikinci nokta"),
    dict(ad="Qwen2.5-14B", base="models--Qwen--Qwen2.5-14B",
         instruct="models--Qwen--Qwen2.5-14B-Instruct",
         aile="Qwen2.5", sira=3, asama=None, tip="dense", p_toplam=14.0, p_etkin=14.0,
         tarif=""),
    dict(ad="OLMo2-13B·taban→SFT", base="models--allenai--OLMo-2-1124-13B",
         instruct="models--allenai--OLMo-2-1124-13B-SFT",
         anahtar={"base": "OLMo2-13B/taban", "instruct": "OLMo2-13B/sft"},
         aile="OLMo2-13B", sira=None, asama=1, tip="dense", p_toplam=13.0, p_etkin=13.0,
         tarif="AI2 merdiveni — 1. adim (taban→SFT)"),
    dict(ad="OLMo2-13B·SFT→DPO", base="models--allenai--OLMo-2-1124-13B-SFT",
         instruct="models--allenai--OLMo-2-1124-13B-DPO",
         anahtar={"base": "OLMo2-13B/sft", "instruct": "OLMo2-13B/dpo"},
         aile="OLMo2-13B", sira=None, asama=2, tip="dense", p_toplam=13.0, p_etkin=13.0,
         tarif="AI2 merdiveni — 2. adim (SFT→DPO)"),
    dict(ad="OLMo2-13B·DPO→Inst", base="models--allenai--OLMo-2-1124-13B-DPO",
         instruct="models--allenai--OLMo-2-1124-13B-Instruct",
         anahtar={"base": "OLMo2-13B/dpo", "instruct": "OLMo2-13B/instruct"},
         aile="OLMo2-13B", sira=None, asama=3, tip="dense", p_toplam=13.0, p_etkin=13.0,
         tarif="AI2 merdiveni — 3. adim (DPO→Instruct)"),
    dict(ad="Mistral-Small-24B", base="models--mistralai--Mistral-Small-24B-Base-2501",
         instruct="models--mistralai--Mistral-Small-24B-Instruct-2501",
         aile="Mistral-Small", sira=1, asama=None, tip="dense", p_toplam=23.6, p_etkin=23.6,
         tarif="Mistral 2501 — dense orta-üst"),
    dict(ad="Gemma-3-27B", base="models--google--gemma-3-27b-pt",
         instruct="models--google--gemma-3-27b-it",
         aile="Gemma-3", sira=3, asama=None, tip="dense", p_toplam=27.0, p_etkin=27.0,
         tarif="★ B3 HAKEMI — Gemma ailesinin DENSE 27B ucu (26B-A4B MoE'nin karsiti)"),
    dict(ad="OLMo2-32B·taban→SFT", base="models--allenai--OLMo-2-0325-32B",
         instruct="models--allenai--OLMo-2-0325-32B-SFT",
         anahtar={"base": "OLMo2-32B/taban", "instruct": "OLMo2-32B/sft"},
         aile="OLMo2-32B", sira=None, asama=1, tip="dense", p_toplam=32.0, p_etkin=32.0,
         tarif="AI2 merdiveni (32B) — 1. adim"),
    dict(ad="OLMo2-32B·SFT→DPO", base="models--allenai--OLMo-2-0325-32B-SFT",
         instruct="models--allenai--OLMo-2-0325-32B-DPO",
         anahtar={"base": "OLMo2-32B/sft", "instruct": "OLMo2-32B/dpo"},
         aile="OLMo2-32B", sira=None, asama=2, tip="dense", p_toplam=32.0, p_etkin=32.0,
         tarif="AI2 merdiveni (32B) — 2. adim"),
    dict(ad="OLMo2-32B·DPO→Inst", base="models--allenai--OLMo-2-0325-32B-DPO",
         instruct="models--allenai--OLMo-2-0325-32B-Instruct",
         anahtar={"base": "OLMo2-32B/dpo", "instruct": "OLMo2-32B/instruct"},
         aile="OLMo2-32B", sira=None, asama=3, tip="dense", p_toplam=32.0, p_etkin=32.0,
         tarif="AI2 merdiveni (32B) — 3. adim"),
    dict(ad="Mixtral-8x7B", base="models--mistralai--Mixtral-8x7B-v0.1",
         instruct="models--mistralai--Mixtral-8x7B-Instruct-v0.1",
         aile="Mixtral", sira=1, asama=None, tip="MoE", p_toplam=46.7, p_etkin=12.9,
         tarif="★ MoE KONTROLÜ — toplam 46,7B / etkin 12,9B (26B-A4B'nin ikinci örnegi)"),
    dict(ad="Llama-3.1-70B", base="models--meta-llama--Llama-3.1-70B",
         instruct="models--meta-llama--Llama-3.1-70B-Instruct",
         aile="Llama-3.1", sira=2, asama=None, tip="dense", p_toplam=70.6, p_etkin=70.6,
         tarif="Meta RLHF — C1'deki 8B'nin ×8,8 ölcegi; iki card ister"),
]

PANEL_OLCEK3 = [
    dict(ad="Tulu3-8B·taban→SFT", base="models--meta-llama--Llama-3.1-8B",
         instruct="models--allenai--Llama-3.1-Tulu-3-8B-SFT",
         anahtar={"base": "Tulu3-8B/taban", "instruct": "Tulu3-8B/sft"},
         aile="Tulu3-8B", sira=None, asama=1, tip="dense", p_toplam=8.0, p_etkin=8.0,
         tarif="AI2 Tülu-3 merdiveni — 1. adim (Llama-3.1-8B tabani → SFT)"),
    dict(ad="Tulu3-8B·SFT→DPO", base="models--allenai--Llama-3.1-Tulu-3-8B-SFT",
         instruct="models--allenai--Llama-3.1-Tulu-3-8B-DPO",
         anahtar={"base": "Tulu3-8B/sft", "instruct": "Tulu3-8B/dpo"},
         aile="Tulu3-8B", sira=None, asama=2, tip="dense", p_toplam=8.0, p_etkin=8.0,
         tarif="AI2 Tülu-3 merdiveni — 2. adim (SFT→DPO)"),
    dict(ad="Tulu3-8B·DPO→RL", base="models--allenai--Llama-3.1-Tulu-3-8B-DPO",
         instruct="models--allenai--Llama-3.1-Tulu-3-8B",
         anahtar={"base": "Tulu3-8B/dpo", "instruct": "Tulu3-8B/rl"},
         aile="Tulu3-8B", sira=None, asama=3, tip="dense", p_toplam=8.0, p_etkin=8.0,
         tarif="AI2 Tülu-3 merdiveni — 3. adim (DPO→RLVR; final Tülu-3-8B)"),
    dict(ad="Tulu3-8B·taban→RL", base="models--meta-llama--Llama-3.1-8B",
         instruct="models--allenai--Llama-3.1-Tulu-3-8B",
         anahtar={"base": "Tulu3-8B/taban", "instruct": "Tulu3-8B/rl"},
         aile="Tulu3-8B", sira=None, asama=None, tip="dense", p_toplam=8.0, p_etkin=8.0,
         tarif="★ UCTAN-UCA — ÖLCEK-2'de EKSIKTI; sekle GIRMEZ (asama=None)"),
]

PANEL_OLMO3_MERDIVEN = [
    dict(ad="OLMo3-7B·taban→SFT", base="models--allenai--Olmo-3-1025-7B",
         instruct="models--allenai--Olmo-3-7B-Instruct-SFT",
         anahtar={"base": "OLMo3-7B/taban", "instruct": "OLMo3-7B/sft"},
         aile="OLMo3-7B", sira=None, asama=1, tip="dense", p_toplam=7.0, p_etkin=7.0,
         tarif="AI2 OLMo-3 merdiveni — 1. adim (taban → SFT)"),
    dict(ad="OLMo3-7B·SFT→DPO", base="models--allenai--Olmo-3-7B-Instruct-SFT",
         instruct="models--allenai--Olmo-3-7B-Instruct-DPO",
         anahtar={"base": "OLMo3-7B/sft", "instruct": "OLMo3-7B/dpo"},
         aile="OLMo3-7B", sira=None, asama=2, tip="dense", p_toplam=7.0, p_etkin=7.0,
         tarif="AI2 OLMo-3 merdiveni — 2. adim (SFT→DPO)"),
    dict(ad="OLMo3-7B·DPO→RLVR", base="models--allenai--Olmo-3-7B-Instruct-DPO",
         instruct="models--allenai--Olmo-3-7B-Instruct",
         anahtar={"base": "OLMo3-7B/dpo", "instruct": "OLMo3-7B/rlvr"},
         aile="OLMo3-7B", sira=None, asama=3, tip="dense", p_toplam=7.0, p_etkin=7.0,
         tarif="AI2 OLMo-3 merdiveni — 3. adim (DPO→RLVR; final Olmo-3-7B-Instruct)"),
    dict(ad="OLMo3-7B·taban→RLVR", base="models--allenai--Olmo-3-1025-7B",
         instruct="models--allenai--Olmo-3-7B-Instruct",
         anahtar={"base": "OLMo3-7B/taban", "instruct": "OLMo3-7B/rlvr"},
         aile="OLMo3-7B", sira=None, asama=None, tip="dense", p_toplam=7.0, p_etkin=7.0,
         tarif=""
               ""),
]

PANEL_OLMO2_7B = [
    dict(ad="OLMo2-7B·taban→SFT", base="models--allenai--OLMo-2-1124-7B",
         instruct="models--allenai--OLMo-2-1124-7B-SFT",
         anahtar={"base": "OLMo2-7B/taban", "instruct": "OLMo2-7B/sft"},
         aile="OLMo2-7B", sira=None, asama=1, tip="dense", p_toplam=7.3, p_etkin=7.3,
         tarif="AI2 OLMo-2 merdiveni, 7B — 1. adim (taban → SFT)"),
    dict(ad="OLMo2-7B·SFT→DPO", base="models--allenai--OLMo-2-1124-7B-SFT",
         instruct="models--allenai--OLMo-2-1124-7B-DPO",
         anahtar={"base": "OLMo2-7B/sft", "instruct": "OLMo2-7B/dpo"},
         aile="OLMo2-7B", sira=None, asama=2, tip="dense", p_toplam=7.3, p_etkin=7.3,
         tarif="AI2 OLMo-2 merdiveni, 7B — 2. adim (SFT→DPO)"),
    dict(ad="OLMo2-7B·DPO→Inst", base="models--allenai--OLMo-2-1124-7B-DPO",
         instruct="models--allenai--OLMo-2-1124-7B-Instruct",
         anahtar={"base": "OLMo2-7B/dpo", "instruct": "OLMo2-7B/instruct"},
         aile="OLMo2-7B", sira=None, asama=3, tip="dense", p_toplam=7.3, p_etkin=7.3,
         tarif="AI2 OLMo-2 merdiveni, 7B — 3. adim (DPO→Instruct)"),
    dict(ad="OLMo2-7B·taban→Inst", base="models--allenai--OLMo-2-1124-7B",
         instruct="models--allenai--OLMo-2-1124-7B-Instruct",
         anahtar={"base": "OLMo2-7B/taban", "instruct": "OLMo2-7B/instruct"},
         aile="OLMo2-7B", sira=None, asama=4, tip="dense", p_toplam=7.3, p_etkin=7.3,
         tarif="AI2 OLMo-2 merdiveni, 7B — uctan uca (taban → Instruct)"),
]

PANEL_SIMPO = [
    dict(ad="Llama3SimPO·SFT→DPO",
         base="models--princeton-nlp--Llama-3-Base-8B-SFT",
         instruct="models--princeton-nlp--Llama-3-Base-8B-SFT-DPO",
         anahtar={"base": "Llama3SimPO/sft", "instruct": "Llama3SimPO/dpo"},
         aile="Llama3SimPO", sira=None, asama=1, tip="dense",
         p_toplam=8.0, p_etkin=8.0,
         tarif="SimPO karsilastirmasi — referans kol (ayni SFT tabani, DPO)"),
    dict(ad="Llama3SimPO·SFT→SimPO",
         base="models--princeton-nlp--Llama-3-Base-8B-SFT",
         instruct="models--princeton-nlp--Llama-3-Base-8B-SFT-SimPO",
         anahtar={"base": "Llama3SimPO/sft", "instruct": "Llama3SimPO/simpo"},
         aile="Llama3SimPO", sira=None, asama=2, tip="dense",
         p_toplam=8.0, p_etkin=8.0,
         tarif="SimPO karsilastirmasi — sinanan kol (referanssiz objektif)"),
]

PANEL_TULU70B = [
    dict(ad="Tulu3-70B·taban→SFT", base="models--meta-llama--Llama-3.1-70B",
         instruct="models--allenai--Llama-3.1-Tulu-3-70B-SFT",
         anahtar={"base": "Tulu3-70B/taban", "instruct": "Tulu3-70B/sft"},
         aile="Tulu3-70B", sira=None, asama=1, tip="dense", p_toplam=70.0,
         p_etkin=70.0,
         tarif="AI2 Tülu-3 merdiveni, 70B — 1. adim (Llama-3.1-70B tabani → SFT)"),
    dict(ad="Tulu3-70B·SFT→DPO", base="models--allenai--Llama-3.1-Tulu-3-70B-SFT",
         instruct="models--allenai--Llama-3.1-Tulu-3-70B-DPO",
         anahtar={"base": "Tulu3-70B/sft", "instruct": "Tulu3-70B/dpo"},
         aile="Tulu3-70B", sira=None, asama=2, tip="dense", p_toplam=70.0,
         p_etkin=70.0,
         tarif="AI2 Tülu-3 merdiveni, 70B — 2. adim (SFT→DPO); tercih adimi"),
]

PANEL_BASAMAK = PANEL_OLMO3_MERDIVEN + PANEL_OLCEK3

PANEL_ZEPHYR = [
    dict(ad="Zephyr-7B·taban→SFT", base="models--mistralai--Mistral-7B-v0.1",
         instruct="models--HuggingFaceH4--mistral-7b-sft-beta",
         anahtar={"base": "Zephyr-7B/taban", "instruct": "Zephyr-7B/sft"},
         aile="Zephyr-7B", sira=None, asama=1, tip="dense", p_toplam=7.2, p_etkin=7.2,
         tarif="H4 Zephyr merdiveni — 1. adim (Mistral-7B-v0.1 tabani → SFT)"),
    dict(ad="Zephyr-7B·SFT→DPO", base="models--HuggingFaceH4--mistral-7b-sft-beta",
         instruct="models--HuggingFaceH4--zephyr-7b-beta",
         anahtar={"base": "Zephyr-7B/sft", "instruct": "Zephyr-7B/dpo"},
         aile="Zephyr-7B", sira=None, asama=2, tip="dense", p_toplam=7.2, p_etkin=7.2,
         tarif="H4 Zephyr merdiveni — 2. adim (SFT→DPO; final zephyr-7b-beta)"),
    dict(ad="Zephyr-7B·taban→DPO", base="models--mistralai--Mistral-7B-v0.1",
         instruct="models--HuggingFaceH4--zephyr-7b-beta",
         anahtar={"base": "Zephyr-7B/taban", "instruct": "Zephyr-7B/dpo"},
         aile="Zephyr-7B", sira=None, asama=None, tip="dense", p_toplam=7.2, p_etkin=7.2,
         tarif="★ UCTAN-UCA — sekle GIRMEZ (asama=None); N-toplam özdesliginin ikinci ayagi"),
]

_OLMO2_13B = [c for c in PANEL_OLCEK2 if c.get("aile") == "OLMo2-13B"]
_OLMO2_13B_UU = [dict(ad="OLMo2-13B·taban→Inst",
                      base="models--allenai--OLMo-2-1124-13B",
                      instruct="models--allenai--OLMo-2-1124-13B-Instruct",
                      anahtar={"base": "OLMo2-13B/taban", "instruct": "OLMo2-13B/instruct"},
                      aile="OLMo2-13B", sira=None, asama=None, tip="dense",
                      p_toplam=13.0, p_etkin=13.0,
                      tarif="★ UCTAN-UCA — H3'te eklendi (N-toplam özdesligi icin)")]
PANEL_HASAT_H3 = _OLMO2_13B + _OLMO2_13B_UU + PANEL_ZEPHYR

PANEL_F5_YENI = [
    dict(ad="Qwen2.5-1.5B", base="models--Qwen--Qwen2.5-1.5B",
         instruct="models--Qwen--Qwen2.5-1.5B-Instruct",
         tarif="SFT+DPO+GRPO (Alibaba) — ★ ÖLCEK TABANI"),
    dict(ad="Zephyr-7B", base="models--mistralai--Mistral-7B-v0.1",
         instruct="models--HuggingFaceH4--zephyr-7b-beta",
         tarif="★ SAF dSFT+dDPO (H4) — ödül modeli YOK"),
    dict(ad="Falcon-H1-7B", base="models--tiiuae--Falcon-H1-7B-Base",
         instruct="models--tiiuae--Falcon-H1-7B-Instruct",
         tarif="★ HIBRIT MIMARI (Mamba+dikkat, TII)"),
    dict(ad="Tulu-3-8B", base="models--allenai--Llama-3.1-Tulu-3-8B-SFT",
         instruct="models--allenai--Llama-3.1-Tulu-3-8B",
         tarif="★ DPO→RLVR (AI2) — taban bacagi SFT'dir"),
    dict(ad="Granite-4.1-8B", base="models--ibm-granite--granite-4.1-8b-base",
         instruct="models--ibm-granite--granite-4.1-8b",
         tarif="IBM SFT+RLHF"),
    dict(ad="Ministral-3-8B", base="models--mistralai--Ministral-3-8B-Base-2512",
         instruct="models--mistralai--Ministral-3-8B-Instruct-2512-BF16",
         tarif="Mistral tercih hizalamasi"),
    dict(ad="OLMo-2-13B", base="models--allenai--OLMo-2-1124-13B",
         instruct="models--allenai--OLMo-2-1124-13B-Instruct",
         tarif="★ RLVR (AI2, tam acik tarif)"),
    dict(ad="Qwen2.5-14B", base="models--Qwen--Qwen2.5-14B",
         instruct="models--Qwen--Qwen2.5-14B-Instruct",
         tarif="SFT+DPO+GRPO (Alibaba) — ★ AILE-ICI ÖLCEK"),
    dict(ad="Mistral-Small-24B", base="models--mistralai--Mistral-Small-24B-Base-2501",
         instruct="models--mistralai--Mistral-Small-24B-Instruct-2501",
         tarif="SFT+tercih (Mistral)"),
    dict(ad="OLMo-2-32B", base="models--allenai--OLMo-2-0325-32B",
         instruct="models--allenai--OLMo-2-0325-32B-Instruct",
         tarif="★ RLVR, 32B — ★ VRAM KIL-PAYI (bkz. EK PREREG §3)"),
    dict(ad="Gemma-3-27B", base="models--google--gemma-3-27b-pt",
         instruct="models--google--gemma-3-27b-it",
         tarif="Google — ★ MINISTRAL TAKASI (ERRATA-1), aile-ici ölcek"),
]
PANEL_F5 = PANEL_C1 + PANEL_F5_YENI

PANELLER = {"c1": PANEL_C1, "olcek": PANEL_OLCEK, "tulu70b": PANEL_TULU70B,
            "olcek2": PANEL_OLCEK2, "olcek3": PANEL_OLCEK3,
            "basamak": PANEL_BASAMAK, "hasat_h3": PANEL_HASAT_H3,
            "olmo2_7b": PANEL_OLMO2_7B, "simpo": PANEL_SIMPO,
            "f5": PANEL_F5}
BELGELER = {
    "c1":    ("C1_AILE_PANELI_2026-08-15.json", "prereg_family_panel_2026-08-15.md",
              "prerun_notice_2026-08-15.md", "prediction_2026-08-15.md"),
    "olcek": ("P5_OLCEK_KOLU_2026-08-21.json", "prereg_scale_arm_2026-08-21.md",
              "NOTICE_P5_OLCEK_2026-08-21.md", "prediction_2026-08-21.md"),
    "olcek2": ("P7_OLCEK2_2026-08-21.json", "prereg_scale2_2026-08-21.md",
               "NOTICE_P7_OLCEK2_2026-08-21.md", "prediction_overnight_2026-08-21.md"),
    "olcek3": ("P_OLCEK3_TULU3_2026-08-22.json", "prereg_scale3_tulu3_2026-08-22.md",
               "NOTICE_P7_OLCEK2_2026-08-21.md", "prediction_scale3_2026-08-22.md"),
    "simpo": ("SIMPO_KARNE_2026-09-10.json", "prereg_simpo_2026-09-10.md",
              "NOTICE_SIMPO_2026-09-10.md", "prediction_simpo_2026-09-10.md"),
    "olmo2_7b": ("OKUMA_BASAMAK_KARNE_2026-08-24.json", "PREREG_OKUMA_BASAMAK_2026-08-24.md",
                 "NOTICE_OKUMA_BASAMAK_2026-08-24.md", "PREDICTION_OKUMA_BASAMAK_assistant_2026-08-24.md"),
    "basamak": ("OKUMA_BASAMAK_KARNE_2026-08-24.json", "PREREG_OKUMA_BASAMAK_2026-08-24.md",
                "NOTICE_OKUMA_BASAMAK_2026-08-24.md", "PREDICTION_OKUMA_BASAMAK_assistant_2026-08-24.md"),
    "tulu70b": ("TULU70B_KARNE_2026-09-10.json", "PREREG_OKUMA_BASAMAK_2026-08-24.md",
                "NOTICE_OKUMA_BASAMAK_2026-08-24.md", "PREDICTION_OKUMA_BASAMAK_assistant_2026-08-24.md"),
    "f5": ("shadow_reading_16_scorecard_2026-08-25.json", "prereg_shadow_reading_16_2026-08-25.md",
           "NOTICE_F5_16_2026-08-25.md", "prediction_f5_2026-08-25.md"),
    "hasat_h3": ("H3_BASAMAK_HAM_2026-08-24.json", "—KESIF-MÜHÜRSÜZ—",
                 "—KESIF-HABERSIZ—", "—KESIF-BAHISSIZ—"),
}


def _gemma(panel):
    return tuple(c["ad"] for c in panel if c["ad"].lower().startswith("gemma"))


KARAR_MERKEZLI = {"c1": False, "olcek": False, "olcek2": False, "olcek3": True,
                  "basamak": False, "hasat_h3": False, "f5": False}
MERKEZLI = False

PANEL, GEMMA = None, None


def panel_sec(ad):
    global PANEL, GEMMA, CIKTI, PREREG, NOTICE, PREDICTION, MERKEZLI
    PANEL = PANELLER[ad]
    MERKEZLI = KARAR_MERKEZLI.get(ad, False)
    GEMMA = _gemma(PANEL)
    CIKTI, PREREG, NOTICE, PREDICTION = (f"{ROOT}/results/{x}" for x in BELGELER[ad])
    return ad


panel_sec("c1")


def verdict_c1(n_gecen, dom_hic_birikmedi, dU_dom_ayrik_poz):
    if n_gecen < N_CIFT_BAR:
        return "KAPI/ÖLCÜLEMEZ"
    if dom_hic_birikmedi:
        return "DOM-HER-YERDE-ÖLÜ-TEYIT"
    if all(dU_dom_ayrik_poz):
        return "TÜM-AILELER-SILER"
    return "IMZA-AILEYE-GÖRE"


def verdict_aro(n_gecen, n_ayrik_poz, n_ayrik_neg):
    if n_gecen < N_CIFT_BAR:
        return "KAPI/ÖLCÜLEMEZ"
    if n_ayrik_poz == 0 and n_ayrik_neg == 0:
        return "ARO-DOKUNULMAMIS"
    if n_ayrik_poz > 0 and n_ayrik_neg > 0:
        return "ARO-IKI-YÖNLÜ"
    if n_ayrik_poz > 0:
        return "ARO-TEK-YÖNLÜ-POZ"
    return "ARO-TEK-YÖNLÜ-NEG"


def sekil_ic_maksimum(sirali_dU):
    if len(sirali_dU) < 3:
        return "ÖLCÜLEMEZ", None
    ix = max(range(len(sirali_dU)), key=lambda i: sirali_dU[i])
    return ("IC-MAKSIMUM" if 0 < ix < len(sirali_dU) - 1 else "UC-MAKSIMUM"), ix


def prova():
    assert verdict_c1(2, False, [True]) == "KAPI/ÖLCÜLEMEZ"
    assert verdict_c1(6, True, [True] * 6) == "DOM-HER-YERDE-ÖLÜ-TEYIT"
    assert verdict_c1(6, False, [True] * 6) == "TÜM-AILELER-SILER"
    assert verdict_c1(6, False, [True] * 5 + [False]) == "IMZA-AILEYE-GÖRE"
    assert verdict_c1(3, False, [False] * 3) == "IMZA-AILEYE-GÖRE"
    prova_ilkel()
    assert _gemma(PANEL_C1) == ("Gemma-3-4B", "Gemma-3-12B"), _gemma(PANEL_C1)
    assert _gemma(PANEL_OLCEK) == ("Gemma-4-26B-A4B",), _gemma(PANEL_OLCEK)
    payda("c1_gemma_uyelik", n_panel=2, n_uye_c1=len(_gemma(PANEL_C1)),
          n_uye_olcek=len(_gemma(PANEL_OLCEK)), red_bos_kume=0)
    u = capraz_say(__file__, "verdict_c1", set(ADLAR))
    payda("c1_prova", n_dal=4, n_ad=len(ADLAR), n_ad_kodda=len(u), red_kapsanmayan=0)
    assert verdict_aro(2, 3, 1) == "KAPI/ÖLCÜLEMEZ"
    assert verdict_aro(6, 0, 0) == "ARO-DOKUNULMAMIS"
    assert verdict_aro(6, 2, 1) == "ARO-IKI-YÖNLÜ"
    assert verdict_aro(6, 2, 0) == "ARO-TEK-YÖNLÜ-POZ"
    assert verdict_aro(6, 0, 2) == "ARO-TEK-YÖNLÜ-NEG"
    assert sekil_ic_maksimum([1.0, 2.0]) == ("ÖLCÜLEMEZ", None)
    assert sekil_ic_maksimum([1.0, 3.0, 2.0]) == ("IC-MAKSIMUM", 1)
    assert sekil_ic_maksimum([3.0, 1.0, 2.0]) == ("UC-MAKSIMUM", 0)
    assert sekil_ic_maksimum([1.0, 2.0, 3.0]) == ("UC-MAKSIMUM", 2)
    import full_tarama as TT
    n_bar = N_CIFT_BAR
    izgara = dict(n_gecen=[0, n_bar - 1, n_bar, 6],
                  n_ayrik_poz=[0, 1, 3, 6], n_ayrik_neg=[0, 1, 3, 6])
    R_izg = TT.tam_tarama(verdict_aro, izgara, set(ADLAR_ARO), etiket="aro_izgara")
    veri = dict(n_gecen=[0, n_bar - 1, n_bar, 6], _pay=[0, 1, 3, 6], _pas=[0.0, 0.4, 1.0])
    R_ver = TT.tam_tarama(
        lambda n_gecen, n_ayrik_poz, n_ayrik_neg, **_: verdict_aro(n_gecen, n_ayrik_poz, n_ayrik_neg),
        veri, set(ADLAR_ARO), etiket="aro_veri_yolu",
        turetilmis=dict(
            n_ayrik_poz=lambda kw: min(kw["_pay"], kw["n_gecen"]),
            n_ayrik_neg=lambda kw: int(min(kw["_pas"] * kw["n_gecen"], max(0, kw["n_gecen"] - min(kw["_pay"], kw["n_gecen"]))))))
    TT.veri_yolu_karsilastir(R_izg, R_ver, etiket="aro_veri_yolu_kiyas")
    TT.kapi(R_izg, R_ver, etiket="aro_kf_kapisi")
    u2 = capraz_say(__file__, "verdict_aro", set(ADLAR_ARO))
    payda("c1_prova_aro", n_dal=5, n_ad=len(ADLAR_ARO), n_ad_kodda=len(u2),
          red_kapsanmayan=0)
    return True


def _imza(snap):
    c = json.load(open(f"{snap}/config.json", encoding="utf-8"))
    g = glob.glob(f"{snap}/generation_config.json")
    gc = json.load(open(g[0], encoding="utf-8")) if g else {}
    t = glob.glob(f"{snap}/tokenizer_config.json")
    tk = json.load(open(t[0], encoding="utf-8")) if t else {}
    tx = c.get("text_config") or {}
    return dict(eos=c.get("eos_token_id"), tip=c.get("model_type"),
                L=c.get("num_hidden_layers") or tx.get("num_hidden_layers"),
                d=c.get("hidden_size") or tx.get("hidden_size"),
                orn=bool(gc.get("temperature") is not None or gc.get("top_k") is not None
                         or gc.get("top_p") is not None),
                template=bool(tk.get("chat_template")))


def _agirlik_izi(snap):
    f = sorted(glob.glob(f"{snap}/*.safetensors")) or sorted(glob.glob(f"{snap}/*.bin"))
    h = hashlib.sha256()
    for p in f:
        h.update(os.path.basename(p).encode()); h.update(str(os.path.getsize(p)).encode())
    with open(f[0], "rb") as fh:
        h.update(fh.read(1 << 20))
    return h.hexdigest()[:16], len(f)


def k0_kimlik(snap, cift, zemin):
    bek = cift["base"] if zemin == "base" else cift["instruct"]
    yol = bek.replace("models--", "").replace("--", "/") in snap.replace("models--", "").replace("--", "/")
    iz, nf = _agirlik_izi(snap)
    ib, ii = _imza(SNAP[cift["ad"]]["base"]), _imza(SNAP[cift["ad"]]["instruct"])
    ayirt = [k for k in ("eos", "orn", "template") if ib[k] != ii[k]]
    kendi = _imza(snap)
    dogru = (all(kendi[k] == (ib if zemin == "base" else ii)[k] for k in ayirt)
             if ayirt else None)
    payda(f"c1_k0kimlik_{cift['ad']}_{zemin}", n_isaret_denetlenen=3,
          n_isaret_MEVCUT=2 + int(bool(ayirt)), hal_ayirt_edici_alan=len(ayirt),
          red_yol=int(not yol), red_config_uyumsuz=int(dogru is False))
    print(f"  [K0-KIMLIK] yol {'✔' if yol else '✗'} · agirlik-izi {iz} ({nf} dosya) · "
          f"config-ayirt-edici {ayirt or '★ YOK — depo adina dayanir'}"
          f"{'' if not ayirt else (' ✔' if dogru else ' ✗')}", flush=True)
    if not yol or dogru is False:
        raise SystemExit("★ K0-KIMLIK DÜSTÜ — yüklenen model beklenen zemin degil")
    return dict(yol=yol, agirlik_izi=iz, n_agirlik_dosyasi=nf, ayirt_edici_alanlar=ayirt,
                config_ayirt_edici_VAR=bool(ayirt), imza=kendi)


SNAP = {}


def snaplari_coz(zorunlu=None):
    eksik = []
    for c in PANEL:
        SNAP[c["ad"]] = {}
        for z in ("base", "instruct"):
            try:
                SNAP[c["ad"]][z] = SA._snapshot(None, model_adi=c[z])
            except Exception as e:
                eksik.append(f"{c['ad']}/{z}: {str(e)[:80]}")
    payda("c1_envanter", n_cift=len(PANEL), n_zemin=2 * len(PANEL),
          red_eksik=len(eksik),
          bekle={"n_cift": len(PANEL), "n_zemin": 2 * len(PANEL)})
    if eksik:
        print(f"  [ENVANTER] eksik {len(eksik)}/{2*len(PANEL)} zemin: "
              f"{[e.split(':')[0] for e in eksik]}", flush=True)
    if zorunlu:
        c_ad, z_ad = zorunlu
        if SNAP.get(c_ad, {}).get(z_ad) is None:
            raise SystemExit(f"★ ENVANTER DÜSTÜ (zorunlu zemin): {c_ad}/{z_ad} — "
                             f"{[e for e in eksik if e.startswith(c_ad + '/' + z_ad)]}")
    return SNAP


def uretim_yolu(cift_ad, zemin):
    c = next((c for c in PANEL if c["ad"] == cift_ad), None)
    ah = (c or {}).get("anahtar") or {}
    return f"{OUT_KOK}/{ah.get(zemin, f'{cift_ad}/{zemin}')}"


def model_yukle(snap, dev, coklu_card=0, etiket="", ty=None):
    import torch
    from transformers import AutoModelForCausalLM
    ty = ty if ty is not None else time.time()
    ck = int(coklu_card or 0)
    if ck > 1:
        import torch as _t
        n_gor = _t.cuda.device_count()
        payda(f"c1_card_gorunurlugu_{etiket}", n_card_gorunen=n_gor,
              n_card_istenen=ck, red_yetersiz=int(n_gor < ck))
        if n_gor < ck:
            raise SystemExit(f"{ck}"
                             f"{n_gor}"
                             f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
        pay = {i: f"{int(_t.cuda.get_device_properties(i).total_memory / 2**30 * 0.92)}GiB"
               for i in range(ck)}
        model = AutoModelForCausalLM.from_pretrained(
            snap, dtype=torch.bfloat16, device_map="auto", max_memory=pay).eval()
        yerlesim = dict(getattr(model, "hf_device_map", {}) or {})
        n_card = len(set(yerlesim.values()))
        payda(f"c1_coklu_card_{etiket}", n_card_istenen=ck,
              n_card_kullanilan=n_card, n_modul=len(yerlesim), red_cpu_offload=int(
                  any(str(v) in ("cpu", "disk") for v in yerlesim.values())))
        print(f"  [COK-CARD] {n_card} card · {len(yerlesim)} modül · pay={pay}", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).to(dev).eval()
        yerlesim = {"(hepsi)": dev}
    print(f"  [W-72] yükleme {time.time()-ty:.1f} sn — ÖLCÜLDÜ", flush=True)
    return model, yerlesim, ck


def uret(a):
    import torch
    t0 = time.time(); prova(); snaplari_coz(zorunlu=(a.cift, a.zemin))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cift = next(c for c in PANEL if c["ad"] == a.cift)
    out = uretim_yolu(a.cift, a.zemin)
    K = K1BU.kollar(ton=True); K1BU.k0a_yuva(K)
    ist = A3.istemler()
    bekle_satir = len(ist) * N_CEKIM
    _bek = (a.bekle_gpu if getattr(a, "bekle_gpu", None) is not None
            else RKG.fiziksel_bekle(a.dev))
    C = kilitle(a.dev, beklenen_fiziksel=_bek, tam=False,
                etiket=f"c1_{a.cift}_{a.zemin}")
    os.makedirs(f"{out}/kol", exist_ok=True)

    ty = time.time(); snap = SNAP[a.cift][a.zemin]
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kimlik = k0_kimlik(snap, cift, a.zemin)
    model, yerlesim, ck = model_yukle(snap, a.dev, getattr(a, "coklu_card", 0),
                                     etiket=f"{a.cift}_{a.zemin}", ty=ty)
    yuk_sn = round(time.time() - ty, 1)
    _tepe = lambda: max(int(torch.cuda.max_memory_allocated(i) / 2**20)
                        for i in range(torch.cuda.device_count()))

    atlanan, t_ilk = 0, None
    for ki, kol in enumerate(K):
        yol = f"{out}/kol/{SA._gv(kol[0])}.jsonl"
        if os.path.exists(yol):
            if sum(1 for _ in open(yol, encoding="utf-8")) >= bekle_satir:
                atlanan += 1; continue
            os.remove(yol)
        tk = time.time()
        SA._kol_uret(model, tok, argparse.Namespace(dev=a.dev, yigin=YIGIN),
                     kol, ist, N_CEKIM, yol)
        if t_ilk is None:
            t_ilk = time.time() - tk
            print(f"  [W-71] ilk kol {t_ilk:.1f} sn ⇒ ETA "
                  f"{t_ilk*(len(K)-atlanan)/60:.1f} dk (yükleme HARIC, ölcülü)", flush=True)
        print(f"  [{ki+1}/{len(K)}] {kol[0]:<22} · {time.time()-t0:.0f}s", flush=True)

    R, kol_say = [], {}
    for ad, *_ in K:
        p = f"{out}/kol/{SA._gv(ad)}.jsonl"
        s = [json.loads(l) for l in open(p, encoding="utf-8")]
        kol_say[ad] = len(s); R += s
    with open(f"{out}/uretim.jsonl", "w", encoding="utf-8") as fh:
        for r in R:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    bos = sum(1 for r in R if not r["metin"].strip())
    payda(f"c1_uretim_{a.cift}_{a.zemin}", n_kol=len(K), n_satir=len(R),
          n_istem=len(ist), n_cekim=N_CEKIM, red_bos_metin=bos,
          bekle={"n_kol": 16, "n_satir": 6528})
    print(f"  [K0-bos] bos metin {bos}/{len(R)} = {bos/len(R):.4f} (bar {BOS_BAR})", flush=True)
    json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   HALKA="üretim", cift=a.cift, zemin=a.zemin, model=snap, kimlik=kimlik,
                   cevre=C, yerlesim=yerlesim, coklu_card=ck,
                   tepe_vram_MiB=_tepe(),
                   seed=getattr(SA, "SEED", None),
                   seed_kaynagi="serita_harman102.SEED",
                   TAM=True, n_kol=len(K), n_satir=len(R), kol_satir=kol_say,
                   atlanan_kol=atlanan, n_cekim=N_CEKIM, yigin=YIGIN, red_bos_metin=bos,
                   bos_orani=round(bos / len(R), 6), yukleme_sn=yuk_sn,
                   saniye=round(time.time() - t0, 1)),
              open(f"{out}/uretim_kunye.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(f"  ★ ÜRETIM TAM · {a.cift}/{a.zemin} · {time.time()-t0:.0f} sn", flush=True)
    return 0


def _zemin_oku(cift_ad, zemin, dev, X_donustur=None, motor="e5"):
    out = uretim_yolu(cift_ad, zemin)
    U = json.load(open(f"{out}/uretim_kunye.json", encoding="utf-8"))
    R = [json.loads(l) for l in open(f"{out}/uretim.jsonl", encoding="utf-8")]
    bos = sum(1 for r in R if not r["metin"].strip())
    payda(f"c1_yukle_{cift_ad}_{zemin}", n_satir=len(R),
          n_kol=len(set(r["kol"] for r in R)), n_istem=len(set(r["istem_i"] for r in R)),
          n_cekim=len(set(r["cekim"] for r in R)), red_bos_metin=bos,
          bekle={"n_satir": 6528, "n_kol": 16, "n_istem": 34, "n_cekim": 12})
    kapi_bos = (bos / len(R)) <= BOS_BAR

    yx = f"{out}/X_{motor}.npy"
    if os.path.exists(yx):
        X = np.load(yx).astype(np.float64)
    elif motor == "e5":
        K1BC.OUT = out
        X = K1BC.gom(R, dev)
    else:
        raise SystemExit(f"★ motor gömüsü diskte YOK: {yx}")
    kol = np.array([r["kol"] for r in R]); ist = np.array([r["istem_i"] for r in R])
    K34 = np.unique(ist)
    CIFT = {(k, i): np.flatnonzero((kol == k) & (ist == i))
            for k in np.unique(kol) for i in K34}
    sec = lambda ks, iss: np.concatenate([CIFT[(k, i)] for k in ks for i in iss])
    aileler = K1BC.aile_adlari(EKSENLER)
    AD = sorted(set(kol.tolist()))
    X1, X4, K22 = aileler
    eksik = [n for e in EKSENLER for n in list(X1[e]) + list(X4[e]) if n not in AD]
    payda(f"c1_hucre_{cift_ad}_{zemin}", n_kol=len(AD), red_eksik_ad=len(eksik),
          bekle={"n_kol": 16})
    if eksik:
        raise SystemExit(f"★ hücre adi diskte YOK: {sorted(set(eksik))}")

    if X_donustur is not None:
        X = X_donustur(X, sec, aileler, K34)

    rb = np.random.default_rng(SEED)
    hav = sec(["ARO+x1", "ARO-x1"], K34); n1 = len(sec(["ARO+x1"], K34))
    null = np.array([_merkez_fark(X, hav[(p := rb.permutation(len(hav)))[:n1]], hav[p[n1:]])
                     for _ in range(K_NULL)])
    nm, nsd = float(null.mean()), float(null.std())

    gh = lambda e, al, iss: K1BC.g_hat(X, sec, aileler, nm, e, al, iss)
    OL = {e: dict(g_x1=round(gh(e, "x1", K34), 6), g_x4=round(gh(e, "x4", K34), 6))
          for e in EKSENLER}
    for e in EKSENLER:
        OL[e]["U"] = round(OL[e]["g_x4"] - OL[e]["g_x1"], 6)

    rbb = np.random.default_rng(SEED + 5)
    Bt = {e: [] for e in EKSENLER}
    for _ in range(B_BOOT):
        ks = rbb.choice(K34, len(K34), replace=True)
        for e in EKSENLER:
            Bt[e].append(gh(e, "x4", ks) - gh(e, "x1", ks))
    for e in EKSENLER:
        v = np.array(Bt[e])
        OL[e]["U_CI"] = [round(float(np.percentile(v, 2.5)), 6),
                         round(float(np.percentile(v, 97.5)), 6)]
        OL[e]["U_sd"] = round(float(v.std()), 6)
        OL[e]["MDE_1645sd"] = round(1.645 * float(v.std()), 6)
        OL[e]["AYRIK_POZ"] = bool(OL[e]["U_CI"][0] > 0)
    rp = np.random.default_rng(SEED + 11)
    UN = {}
    for e in EKSENLER:
        v = []
        for _ in range(K_NULL):
            g = {}
            for al, (ap, an) in (("x1", X1[e]), ("x4", X4[e])):
                A, B_ = sec([ap], K34), sec([an], K34)
                hp = np.concatenate([A, B_]); rp.shuffle(hp)
                g[al] = _merkez_fark(X, hp[:len(A)], hp[len(A):]) - nm
            v.append(g["x4"] - g["x1"])
        UN[e] = np.array(v)
        OL[e]["Unull_ort"] = round(float(UN[e].mean()), 6)
        OL[e]["Unull_sd"] = round(float(UN[e].std()), 6)
        OL[e]["REJIM"] = ("KURULAMADI" if abs(UN[e].mean()) / (UN[e].std() or 1) > 1.645
                          else "kuruldu")
        OL[e]["frac_null_gecen"] = round(float((UN[e] >= OL[e]["U"]).mean()), 4)
    payda(f"c1_plasebo_{cift_ad}_{zemin}", n_eksen=len(EKSENLER), n_cekim=K_NULL,
          red_rejim_kurulamadi=sum(1 for e in EKSENLER if OL[e]["REJIM"] == "KURULAMADI"))

    return dict(olcum=OL, null=dict(ort=round(nm, 6), sd=round(nsd, 6)),
                kapi_bos=bool(kapi_bos), bos_orani=round(bos / len(R), 6),
                n_satir=len(R), n_kume=int(len(K34)), model=U["model"],
                kimlik=U["kimlik"]), {e: np.array(Bt[e]) for e in EKSENLER}, UN


def coz(a):
    t0 = time.time(); prova(); snaplari_coz()
    for y in (PREREG, NOTICE, PREDICTION):
        if not os.path.exists(y):
            raise SystemExit(f"{os.path.basename(y)}")
    payda("c1_b43", n_belge=3, red_eksik=0)

    P, atlanan = {}, []
    ETA_K = 3.0
    _cift_sn, _ilk_eta = [], None
    for c in PANEL:
        _tc = time.time()
        try:
            zb, Bb, UNb = _zemin_oku(c["ad"], "base", a.dev)
            zi, Bi, UNi = _zemin_oku(c["ad"], "instruct", a.dev)
        except Exception as e:
            atlanan.append(f"{c['ad']}: {e}"); print(f"  [ATLANDI] {c['ad']}: {e}", flush=True)
            continue
        d = {}
        for e in EKSENLER:
            v = Bb[e] - Bi[e]
            ci = [round(float(np.percentile(v, 2.5)), 6), round(float(np.percentile(v, 97.5)), 6)]
            sd = float(v.std())
            dU = round(zb["olcum"][e]["U"] - zi["olcum"][e]["U"], 6)
            d[e] = dict(dU=dU, dU_CI=ci, dU_sd=round(sd, 6),
                        MDE_1645sd=round(1.645 * sd, 6),
                        null_sd_kati=round(abs(dU) / sd, 3) if sd > 0 else None,
                        AYRIK_POZ=bool(ci[0] > 0), AYRIK_NEG=bool(ci[1] < 0))
            d[e]["CONTINGENT"] = bool(not d[e]["AYRIK_POZ"] and not d[e]["AYRIK_NEG"]
                                      and d[e]["null_sd_kati"] is not None
                                      and 1.0 <= d[e]["null_sd_kati"] < 1.645)
            dn = UNb[e] - UNi[e]
            d[e]["dU_plasebo_ort"] = round(float(dn.mean()), 6)
            d[e]["dU_plasebo_sd"] = round(float(dn.std()), 6)
            d[e]["dU_plasebo_frac"] = round(float((np.abs(dn) >= abs(dU)).mean()), 4)
            d[e]["PLASEBO_MERKEZLI"] = bool(abs(dn.mean()) / (dn.std() or 1) <= 1.645)
            m = float(dn.mean())
            d[e]["null_merkezi"] = round(m, 6)
            d[e]["dU_M"] = round(dU - m, 6)
            d[e]["dU_CI_M"] = [round(ci[0] - m, 6), round(ci[1] - m, 6)]
            d[e]["null_sd_kati_M"] = round(abs(dU - m) / sd, 3) if sd > 0 else None
            d[e]["AYRIK_POZ_M"] = bool(ci[0] - m > 0)
            d[e]["AYRIK_NEG_M"] = bool(ci[1] - m < 0)
            d[e]["CONTINGENT_M"] = bool(not d[e]["AYRIK_POZ_M"] and not d[e]["AYRIK_NEG_M"]
                                        and d[e]["null_sd_kati_M"] is not None
                                        and 1.0 <= d[e]["null_sd_kati_M"] < 1.645)
            d[e]["dU_plasebo_frac_M"] = round(
                float((np.abs(dn - m) >= abs(dU - m)).mean()), 4)
            d[e]["AYRIK_POZ_HAM"] = d[e]["AYRIK_POZ"]
            d[e]["AYRIK_NEG_HAM"] = d[e]["AYRIK_NEG"]
            d[e]["KARAR_TABANI"] = "null_merkezi" if MERKEZLI else "sifir"
            if MERKEZLI:
                d[e]["AYRIK_POZ"] = d[e]["AYRIK_POZ_M"]
                d[e]["AYRIK_NEG"] = d[e]["AYRIK_NEG_M"]
                d[e]["CONTINGENT"] = d[e]["CONTINGENT_M"]
        poz = zb["olcum"]["ARO"]["AYRIK_POZ"] and zi["olcum"]["ARO"]["AYRIK_POZ"]
        P[c["ad"]] = dict(tarif=c["tarif"], POZ_KONTROL=bool(poz),
                          kapi_bos=bool(zb["kapi_bos"] and zi["kapi_bos"]),
                          base=zb, instruct=zi, delta=d)
        print(f"  [{c['ad']:<16}] POZ-KONTROL {'✔' if poz else '✗'} · "
              + " · ".join(f"Δ{e} {d[e]['dU']:+.5f} "
                           f"{'>0' if d[e]['AYRIK_POZ'] else ('<0' if d[e]['AYRIK_NEG'] else '~')}"
                           for e in EKSENLER), flush=True)
        _dt = time.time() - _tc
        _cift_sn.append(_dt)
        if _ilk_eta is None:
            _ilk_eta = _dt * len(PANEL)
            print(f"  [W-428] ilk cift {_dt:.1f} sn ⇒ ETA {_ilk_eta / 60:.1f} dk "
                  f"({len(PANEL)} cift, ÖLCÜLDÜ) ⇒ esik {ETA_K}× = "
                  f"{ETA_K * _ilk_eta / 60:.1f} dk ⇒ asilirsa UYAR, süre künyeye yazilir",
                  flush=True)
        _ge = sum(_cift_sn)
        _hal = (""
                if _ge > ETA_K * _ilk_eta else "icinde")
        print(f"  [SÜRE] {c['ad']}: {_dt:.1f} sn · gecen {_ge / 60:.1f} dk / esik "
              f"{ETA_K * _ilk_eta / 60:.1f} dk ⇒ {_hal}", flush=True)

    rejim_yok = [k for k, v in P.items()
                 if v["base"]["olcum"]["DOM"]["REJIM"] == "KURULAMADI"
                 or v["instruct"]["olcum"]["DOM"]["REJIM"] == "KURULAMADI"
                 or not v["delta"]["DOM"]["PLASEBO_MERKEZLI"]]
    gecen = [k for k, v in P.items()
             if v["POZ_KONTROL"] and v["kapi_bos"] and k not in rejim_yok]
    gecmeyen = [k for k in P if k not in gecen and k not in rejim_yok]
    dom_olu = all(not (P[k]["base"]["olcum"]["DOM"]["AYRIK_POZ"]
                       or P[k]["instruct"]["olcum"]["DOM"]["AYRIK_POZ"]) for k in gecen) \
        if gecen else False
    ayrik = [P[k]["delta"]["DOM"]["AYRIK_POZ"] for k in gecen]
    H = verdict_c1(len(gecen), dom_olu, ayrik)

    imza_ayrik = (any(P[k]["delta"]["DOM"]["AYRIK_POZ"] for k in gecen)
                  and any(P[k]["delta"]["DOM"]["AYRIK_NEG"] for k in gecen))
    gemma = {k: P[k]["delta"]["DOM"] for k in gecen if k in GEMMA}
    gemma_ekler = bool(gemma) and all(v["AYRIK_NEG"] for v in gemma.values())

    rejim_yok_aro = [k for k, v in P.items()
                     if v["base"]["olcum"]["ARO"]["REJIM"] == "KURULAMADI"
                     or v["instruct"]["olcum"]["ARO"]["REJIM"] == "KURULAMADI"
                     or not v["delta"]["ARO"]["PLASEBO_MERKEZLI"]]
    gecen_aro = [k for k, v in P.items()
                 if v["POZ_KONTROL"] and v["kapi_bos"] and k not in rejim_yok_aro]
    aro_poz = [k for k in gecen_aro if P[k]["delta"]["ARO"]["AYRIK_POZ"]]
    aro_neg = [k for k in gecen_aro if P[k]["delta"]["ARO"]["AYRIK_NEG"]]
    H_ARO = verdict_aro(len(gecen_aro), len(aro_poz), len(aro_neg))
    tarif = {c["ad"]: c for c in PANEL}
    SEKIL = {}
    for eksen, alan in (("SEKIL_OLCEK", "sira"), ("SEKIL_ASAMA", "asama")):
        for ail in sorted({tarif[k].get("aile") for k in gecen if tarif[k].get("aile")}):
            uye = [(tarif[k][alan], k) for k in gecen
                   if tarif[k].get("aile") == ail and tarif[k].get(alan) is not None]
            if not uye:
                continue
            uye.sort()
            dU = [P[k]["delta"]["DOM"]["dU"] for _, k in uye]
            hal, ix = sekil_ic_maksimum(dU)
            SEKIL.setdefault(eksen, {})[ail] = dict(
                uyeler=[k for _, k in uye], dU=dU, HAL=hal,
                maksimum=(uye[ix][1] if ix is not None else None), n=len(uye))
    payda("c1_p7_ekler", n_cift_okunan=len(P), n_gecen_aro=max(len(gecen_aro), 0) or len(P),
          n_sekil_ekseni=2, red_rejim_kurulamadi_aro=len(rejim_yok_aro))
    print(f"  [ARO · KARAR EKSENI] verdict={H_ARO} · gecen {len(gecen_aro)} · "
          f"ayrik+ {len(aro_poz)} {aro_poz} · ayrik− {len(aro_neg)} {aro_neg} · "
          f"rejim-kurulamadi {len(rejim_yok_aro)}", flush=True)
    for eks, d in SEKIL.items():
        for ail, v in d.items():
            print(f"  [{eks}] {ail}: {v['HAL']} (maks={v['maksimum']}, n={v['n']})", flush=True)

    payda("c1_coz_sure", n_cift_olculen=len(_cift_sn),
          hal_toplam_sn=int(sum(_cift_sn)),
          hal_eta_ilk_ciftten_sn=int(_ilk_eta or 0),
          hal_esik_asildi=int(bool(_ilk_eta and sum(_cift_sn) > ETA_K * _ilk_eta)))
    payda("c1_panel", n_cift_denetlenen=len(PANEL), n_cift_okunan=len(P),
          n_eksen=len(EKSENLER), B_boot=B_BOOT, red_atlanan=len(atlanan),
          red_poz_dusen=len(gecmeyen), red_rejim_kurulamadi=len(rejim_yok))
    print(f"  [KURNE · RULE-3] gecen {len(gecen)} · gecmeyen {len(gecmeyen)} · "
          f"rejim-kurulamadi {len(rejim_yok)} {rejim_yok or ''}", flush=True)
    json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   prereg=os.path.basename(PREREG), VERDICT=H,
                   KARNE_RULE3=dict(gecen=len(gecen), gecmeyen=len(gecmeyen),
                                     rejim_kurulamadi=len(rejim_yok)),
                   rejim_kurulamayan=rejim_yok, gecmeyen_cifter=gecmeyen,
                   n_cift_gecen=len(gecen), gecen_cifter=gecen, atlanan=atlanan,
                   ALT_IMZA_AYRIK=bool(imza_ayrik), ALT_GEMMA_EKLER=bool(gemma_ekler),
                   ALT_DOM_HIC_BIRIKMEDI=bool(dom_olu), eksenler=list(EKSENLER),
                   VERDICT_ARO=H_ARO,
                   ARO=dict(gecen=gecen_aro, n_gecen=len(gecen_aro),
                            ayrik_poz=aro_poz, ayrik_neg=aro_neg,
                            rejim_kurulamadi=rejim_yok_aro, adlar=list(ADLAR_ARO)),
                   SEKIL=SEKIL,
                   panel=P, N_CIFT_BAR=N_CIFT_BAR, B_boot=B_BOOT, K_null=K_NULL,
                   betik_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
                   saniye=round(time.time() - t0, 1)),
              open(CIKTI, "w"), ensure_ascii=False, indent=1, default=str)
    print(f"{H_ARO} {len(gecen_aro)} {len(PANEL)}"
          f"{H} {len(gecen)} {len(PANEL)}"
          f"{imza_ayrik}"
          f"{gemma_ekler}"
          f"{dom_olu}", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--halka", required=True, choices=["prova", "uret", "coz"])
    p.add_argument("--cift", default=None)
    p.add_argument("--zemin", default=None, choices=["base", "instruct"])
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--bekle-gpu", dest="bekle_gpu", type=int, default=None)
    p.add_argument("--panel", default="c1", choices=sorted(PANELLER))
    p.add_argument("--coklu-card", dest="coklu_card", type=int, default=0)
    a = p.parse_args()
    panel_sec(a.panel)
    print(f"  [PANEL] {a.panel} · {len(PANEL)} cift · GEMMA={GEMMA}", flush=True)
    if a.halka == "prova":
        sys.exit(0 if (prova() and snaplari_coz(
            zorunlu=((a.cift, a.zemin) if (a.cift and a.zemin) else None))) else 1)
    sys.exit(uret(a) if a.halka == "uret" else coz(a))
