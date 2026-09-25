#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/v97_a14_vif_yeniden_fit_2026-09-19.json"
YUZEY_AD = {"LOG_JETON": r"$\log$ tokens", "LOG_JETON_CUMLE": r"$\log$(tokens per sentence)", "RED": "the refusal term"}


def main():
    K, pk = load(CARD)
    if K["alet"]["hal"] != "ESDEGER":
        print(""); return 3
    O = K["okuma_sonucu"]; v = O["i"]["VIF"]
    kisi = max(v[a] for a in ("M1_1k", "SAHIS1_1k", "SAHIS3_1k"))
    uz = max(v["LOG_JETON"], v["LOG_CUMLE"])
    a, b = O["ii"], O["Fb"]
    s = (rf"Variance inflation in the eight-control fit is at most ${uz:.1f}$, on the two length terms, and at most ${kisi:.1f}$ on "
         rf"the person terms. Writing length as $\log$ tokens and $\log$(tokens per sentence) leaves the second-person coefficient at "
         rf"${a['fit']['M1_1k']['beta']:.3f}$ and puts the margin against the largest surface term ({YUZEY_AD.get(a['en_guclu_yuzey'], a['en_guclu_yuzey'])}) "
         rf"at ${a['marj_yuzey']:.2f}$; keeping only $\log$(tokens per sentence) gives ${b['fit']['M1_1k']['beta']:.3f}$ "
         rf"$[{b['fit']['M1_1k']['ci'][0]:.3f},{b['fit']['M1_1k']['ci'][1]:.3f}]$ and a margin of ${b['marj_yuzey']:.2f}$. The registered "
         r"reading is the one in the table, and these refits are descriptive.")
    io.open(f"{OUT}/A14_VIF.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/a14_vif.py)\n" + s)
    json.dump(dict(table="A14_VIF", ciktilar=["A14_VIF.tex"], sources=[pk],
                   payda=dict(vif_uzunluk=uz, vif_kisi=kisi, Fa_marj=a["marj_yuzey"], Fb_marj=b["marj_yuzey"],
                              Fa_beta=a["fit"]["M1_1k"]["beta"], Fb_beta=b["fit"]["M1_1k"]["beta"]),
                   note="renders the R3 VIF/refit card; no statistic computed here"),
              io.open(f"{OUT}/A14_VIF.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  [PAYDA] a14_vif: VIF uzunluk {uz:.2f} · kisi {kisi:.2f} · F-a marj {a['marj_yuzey']:.3f} · F-b marj {b['marj_yuzey']:.3f} ⇒ esik: ESDEGER ⇒ EYLEM: degilse YAZILMAZ")
    print("✓ A14_VIF.tex"); return 0


if __name__ == "__main__":
    sys.exit(main())
