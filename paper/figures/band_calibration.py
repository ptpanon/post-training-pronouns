#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, io, json, sys
ROOT = __DNH_ROOT__ + ""
OUT = f"{ROOT}/paper/figures"
CARD = "results/spread_calibration_2026-09-14.json"
KIT_BANT = "results/rebuttal_materials_v78_band_no_system_line_2026-09-15.json"
SAYI = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def main():
    K = json.load(io.open(f"{ROOT}/{CARD}", encoding="utf-8"))
    B = json.load(io.open(f"{OUT}/BANT.meta.json", encoding="utf-8"))
    Y = K["yayilim"]
    m1_t, m1_h = Y["taban"]["M1"]["max_min"], Y["hizali"]["M1"]["max_min"]
    assert abs(m1_h - B["oran_hizali"]) < 1e-3 and abs(m1_t - B["oran_taban"]) < 1e-3
    lk_t, lk_h = Y["taban"]["listekod_payi"]["max_min"], Y["hizali"]["listekod_payi"]["max_min"]
    uz_t, uz_h = Y["taban"]["uzunluk_jeton"]["max_min"], Y["hizali"]["uzunluk_jeton"]["max_min"]
    lk_min_t = Y["taban"]["listekod_payi"]["min"]
    assert lk_h < lk_t and uz_h < uz_t, ""
    assert m1_h > m1_t, ""
    assert Y["hizali"]["red_orani"]["max_min"] is None and Y["taban"]["red_orani"]["max_min"] is None
    S = json.load(io.open(f"{ROOT}/{KIT_BANT}", encoding="utf-8"))
    assert S["n_aile"] == 16 and S["suzgec"].startswith("YOK"), (S["n_aile"], S["suzgec"])
    st_tum, st_tem = S["bant_tum"]["oran"], S["bant_temiz"]["oran"]
    assert st_tem < st_tum
    olc = [a for a in K["aileler"] if a["hal"] == "ÖLCÜLDÜ"]
    assert len(olc) == 16, len(olc)
    n_sifir_h = sum(1 for a in olc if a["hizali"]["red_orani"] == 0)
    n_sifir_t = sum(1 for a in olc if a["taban"]["red_orani"] == 0)
    assert n_sifir_h in SAYI and n_sifir_t in SAYI, (n_sifir_h, n_sifir_t)
    print(f"{len(olc)} {m1_t:.2f} {m1_h:.2f} {lk_t:.2f} {lk_h:.2f} {lk_min_t:.4f}"
          f"{uz_t:.3f} {uz_h:.3f} {n_sifir_h} {n_sifir_t}"
          f"")
    s = (f"The spread of \\S\\ref{{sec:deperson}} is read here against quantities alignment does manage, on the same "
         f"chat-template cells after the same four-quadrant filter and with the same measure, the ratio of the largest model "
         f"value to the smallest across the sixteen models. Second-person density widens from ${m1_t:.1f}\\times$ across "
         f"the bases to ${m1_h:.1f}\\times$ across the aligned checkpoints, while the share of list and code lines narrows "
         f"from ${lk_t:.1f}\\times$ to ${lk_h:.1f}\\times$ and response length from ${uz_t:.2f}\\times$ to "
         f"${uz_h:.2f}\\times$: alignment draws together the quantities it manages and pulls this one apart. "
         f"Read with the default system line absent on every model and without the four-quadrant filter, the aligned band "
         f"spans ${st_tum:.1f}\\times$, and ${st_tem:.1f}\\times$ with echo and turn-marker rows dropped, so the spread is not "
         f"those two inputs. A ratio of "
         f"extremes is unstable where a minimum nears zero --- the smallest base share of list and code lines is "
         f"${lk_min_t:.3f}$ --- and refusal rate, zero in {SAYI[n_sifir_h]} aligned models and in {SAYI[n_sifir_t]} "
         f"base, cannot be read this way at all. The comparison is descriptive and was not pre-registered; it gives a "
         f"reference class, not a null.")
    io.open(f"{OUT}/BANT_KALIB.tex", "w", encoding="utf-8").write(s + "\n")
    json.dump(dict(table="BANT_KALIB", ciktilar=["BANT_KALIB.tex"], sources=[dict(yol=y, sha256_16=hashlib.sha256(open(f"{ROOT}/{y}", "rb").read()).hexdigest()[:16]) for y in (CARD, KIT_BANT)],
                   payda=dict(M1=[m1_t, m1_h], listekod=[lk_t, lk_h], listekod_min_taban=lk_min_t, uzunluk=[uz_t, uz_h], satirsiz_bant=dict(tum=st_tum, temiz=st_tem),
                              red_sifir_aile=dict(hizali=n_sifir_h, taban=n_sifir_t)),
                   note="renders the descriptive spread calibration as an Appendix E paragraph; no statistic computed here"),
              io.open(f"{OUT}/BANT_KALIB.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  ✓ BANT_KALIB.tex ({len(s)} kar) · «{s}»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
