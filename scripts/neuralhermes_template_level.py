#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys, time
import numpy as np
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
import addressee_run as MK, form_count as FS, rung_force_ci as BK
from reading_style_object import payda
K = __DNH_DATA__ + "/c1_panel_v61_template/NeuralHermes-2.5"
t0 = time.time(); nlp = FS._boru(); V, I, n = {}, {}, {}
for b in ("sft", "dpo"):
    R = [json.loads(l) for l in io.open(f"{K}/{b}/uretim.jsonl", encoding="utf-8")]
    V[b] = MK.satir_bilesenleri(nlp, R); I[b] = np.array([x.get("istem_i", -1) for x in R]); n[b] = len(R)
m = {b: float(MK.olc_toplam(V[b])["M1"]) for b in V}
lo, hi, nk = BK.kume_boot(V["dpo"], I["dpo"], V["sft"], I["sft"], "M1")
out = dict(_kunye=dict(alet="scripts/neuralhermes_template_level.py", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       kok=K, sinif="DÜZEY OKUMASI — verdict kurmaz", zemin_kaynagi="p11.tex Ek «Absolute densities behind the templated ladder shares»: floor of the whole templated aligned band 2.01"),
           M1_sft=round(m["sft"], 4), M1_dpo=round(m["dpo"], 4), dM1_tercih=round(m["dpo"] - m["sft"], 4),
           ci=[round(lo, 4), round(hi, 4)], ayrik=bool(lo * hi > 0), n_istem=int(nk), n_satir=n,
           sft_sablonlu_zemine_mesafe=round(m["sft"] - 2.01, 4))
json.dump(out, open(__DNH_ROOT__ + "/results/neuralhermes_template_level_2026-09-15.json", "w"), ensure_ascii=False, indent=1)
payda("neuralhermes_template_duzey", n_bacak=2, hal_dk=round((time.time() - t0) / 60, 1))
print(json.dumps({k: v for k, v in out.items() if k != "_kunye"}, ensure_ascii=False))
