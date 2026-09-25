#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import csv, glob, json, math, os, re, sys
ROOT = __DNH_ROOT__ + ""
KUTU = __DNH_DATA__ + "/teslim_annotator/KP-e799"
ANAH = __DNH_DATA__ + "/anahtar_annotator/KP-e799/anahtar.jsonl"
sys.path.insert(0, f"{ROOT}/scripts")
import dejenerelik_olcer as DJ

KELIME = re.compile(r"[a-z']+")


def sozluk():
    yol = f"{ROOT}/unreleased/Ratings_Warriner_et_al.csv"
    W = {}
    with open(yol, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                W[r["Word"].strip().lower()] = float(r["D.Mean.Sum"])
            except (ValueError, KeyError):
                continue
    return W, yol


def d_puan(metin, W):
    ks = [W[w] for w in KELIME.findall((metin or "").lower()) if w in W]
    return (sum(ks) / len(ks), len(ks)) if ks else (None, 0)


def z_iki_ornek(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (ma - mb) / se if se > 0 else None


def _prova():
    W = {"good": 6.0, "bad": 2.0}
    return {
        "i_bos_none": d_puan("", W) == (None, 0),
        "ii_sozluk_disi_none": d_puan("zzz qqq", W) == (None, 0),
        "iii_ortalama": abs(d_puan("good bad", W)[0] - 4.0) < 1e-9,
        "iv_kapsam_sayisi": d_puan("good good bad", W)[1] == 3,
        "v_z_sifir_varyans_none": z_iki_ornek([1, 1, 1], [2, 2, 2]) is None,
        "v_b_z_isaret": z_iki_ornek([1, 2, 1, 2], [3, 4, 3, 4]) < 0,
        "vi_z_tek_eleman_none": z_iki_ornek([1], [2]) is None,
        "vii_z_ozdes_sifir_civari": abs(z_iki_ornek([1, 2, 3], [1, 2, 3])) < 1e-9,
        "viii_dejenere_ates": DJ.bayrakla("aaaa aaaa aaaa aaaa aaaa aaaa")["f_uzun"] is True,
        "ix_temiz_ates_etmez": DJ.bayrakla(
            "The committee reviewed the proposal and asked for two changes.")["f_uzun"] is False,
    }
