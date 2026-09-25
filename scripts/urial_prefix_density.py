#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import json, re, sys, time
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
import addressee_run as MK
import form_count as FS
from reading_style_object import payda

KOK = __DNH_ROOT__ + ""
ISTEM = __DNH_DATA__ + "/unreleased/inst_1k.txt"
KUNYE = f"{KOK}/results/URIAL_ISTEM_KUNYE.json"
CIK = f"{KOK}/results/v27_urial_prefix_density_2026-09-09.json"
BEK = [53.1915, 13.2743, 6.8027]


def main():
    nlp = FS._boru()
    txt = open(ISTEM, encoding="utf-8").read()
    parts = re.split(r"(?m)^# Query:", txt)
    baslik, ornekler = parts[0], ["# Query:" + p for p in parts[1:]]

    def m1(ms):
        return float(MK.olc_toplam(MK.satir_bilesenleri(
            nlp, [dict(metin=m) for m in ms]))["M1"])

    def cevap(o):
        m = re.search(r"# Answer:\s*```(.*?)```", o, re.S)
        return m.group(1) if m else ""

    cev = [cevap(o) for o in ornekler]
    tek = [round(m1([c]), 4) for c in cev]
    sapma = max(abs(a - b) for a, b in zip(tek, BEK)) if len(tek) == len(BEK) else 9e9
    print(f"{len(cev)} {BEK}"
          f"{tek} {sapma:.4f}"
          f"")
    if len(cev) != 3 or sapma > 0.001:
        print("  ★★ ESDEGERLIK DÜSTÜ ⇒ ölcüm yazilmadi"); return 3
    D = dict(_kunye=dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         alet="scripts/urial_prefix_density.py",
                         motor="addressee_run.olc_toplam (ITHAL, edit yok)",
                         borc="D-0910-V27 §1 · PREREG_URIAL ERRATA-1",
                         istem=ISTEM, esdegerlik_sapma=sapma, n_ornek=len(cev)),
             cevaplar_tek=tek, cevaplar_uc=round(m1(cev), 4),
             baslik=round(m1([baslik]), 4), baslik_kelime=len(baslik.split()),
             baslik_arti_cevaplar=round(m1([baslik] + cev), 4),
             onek_butun=round(m1([txt]), 4))
    json.dump(D, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("urial_onek", n_parca=3, hal_baslik=D["baslik"], hal_butun=D["onek_butun"],
          red_esdegerlik=0)
    print(f"  baslik {D['baslik']} · üc cevap {D['cevaplar_uc']} · bütün {D['onek_butun']}")
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
