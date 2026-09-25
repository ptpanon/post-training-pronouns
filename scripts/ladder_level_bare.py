#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
import addressee_run as MK, form_count as FS
from reading_style_object import payda
V = __DNH_DATA__ + ""
G = __DNH_ROOT__ + "/results"
MERD = {
 "Tulu3-8B":        ("c1_panel", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo"), ("rl", "rl")]),
 "OLMo2-13B":       ("c1_panel", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo"), ("instruct", "instruct")]),
 "OLMo2-32B":       ("c1_panel", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo"), ("instruct", "instruct")]),
 "Zephyr-7B":       ("c1_panel", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo")]),
 "OLMo3-7B":        ("c1_panel", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo"), ("rlvr", "rlvr")]),
 "Tulu3-70B":       ("c1_panel", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo")]),
 "OLMo2-7B":        ("c1_panel", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo"), ("instruct", "instruct")]),
 "SmolLM2-1.7B":    ("c1_panel_v61", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo")]),
 "NeuralHermes-2.5": ("c1_panel_v61", [("base", "taban"), ("sft", "sft"), ("dpo", "dpo")]),
}
t0 = time.time(); nlp = FS._boru(); out = {}; n_bacak = 0
K16 = json.load(open(f"{G}/YETENEK_KARNE_16_2026-09-13.json"))
hiz = sorted((r["M1_hizali"], a) for a, r in K16["aile"].items())
tab = sorted((r["M1_taban"], a) for a, r in K16["aile"].items())
for ad, (kok, basamaklar) in MERD.items():
    out[ad] = {"kok": kok}
    for b, dz in basamaklar:
        R = [json.loads(l) for l in io.open(f"{V}/{kok}/{ad}/{dz}/uretim.jsonl", encoding="utf-8")]
        M1 = float(MK.olc_toplam(MK.satir_bilesenleri(nlp, R))["M1"]); n_bacak += 1
        out[ad][b] = dict(M1=round(M1, 4), n=len(R))
        print(f"  {ad:<17} {b:<9} M1 {M1:7.3f}  n {len(R)}", flush=True)
    L = [out[ad][b]["M1"] for b, _ in basamaklar]
    out[ad]["sft_zemine_mesafe"] = round(out[ad]["sft"]["M1"] - hiz[0][0], 4)
    out[ad]["tercih_adimi"] = round(out[ad]["dpo"]["M1"] - out[ad]["sft"]["M1"], 4)
    out[ad]["sft_bandda_konum"] = round((out[ad]["sft"]["M1"] - hiz[0][0]) / (hiz[-1][0] - hiz[0][0]), 4)
res = dict(_kunye=dict(alet="scripts/ladder_level_bare.py", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       sinif="DÜZEY OKUMASI — verdict kurmaz", protokol="ciplak devam istemi",
                       band_kaynagi="YETENEK_KARNE_16_2026-09-13.json (M1_hizali / M1_taban, 16 aile)",
                       sure_dk=round((time.time() - t0) / 60, 1)),
           hizali_band=dict(zemin=hiz[0], tavan=hiz[-1]), taban_band=dict(alt=tab[0], ust=tab[-1]), merdiven=out)
json.dump(res, open(f"{G}/ladder_level_bare_2026-09-15.json", "w"), ensure_ascii=False, indent=1)
payda("merdiven_duzey_ciplak", n_merdiven=len(MERD), n_bacak=n_bacak, hal_dk=res["_kunye"]["sure_dk"])
print("hizali band", hiz[0], hiz[-1], "· taban band", tab[0], tab[-1])
for ad in MERD:
    o = out[ad]; print(f"  {ad:<17} SFT {o['sft']['M1']:6.2f} · zemine mesafe {o['sft_zemine_mesafe']:+6.2f} · bantta konum {o['sft_bandda_konum']:+.2f} · tercih adimi {o['tercih_adimi']:+6.2f}")
