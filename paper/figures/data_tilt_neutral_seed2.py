#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/data_tilt_neutral_seed2_verdict_2026-09-13.json"
KK3 = "results/data_tilt_verdict_2026-09-07.json"


def main():
    K, pk = load(CARD)
    K3, pk3 = load(KK3)
    h, n1, t2 = K["verdict"], K["kol"]["KK3_NOTR"], K["kol"]["KK3_NOTR_T2"]
    assert h["kk3t2"] == "SEED-GÖRÜNMEZ", h["kk3t2"]
    assert h["ciT"][0] <= 0 <= h["ciT"][1], h["ciT"]
    assert t2["bilesen_A"] == {"bar": False, "ayrik": True, "plasebo": False}, t2["bilesen_A"]
    assert n1["bilesen_A"] == {"bar": True, "ayrik": True, "plasebo": True}, n1["bilesen_A"]
    assert abs(n1["dA"] - K3["verdict"]["okuma_A"]["KK3_NOTR"]) < 1e-9
    assert abs(abs(h["ref_tilt"]) - abs(K3["verdict"]["okuma_A"]["KK3_TABAN"] - K3["verdict"]["okuma_A"]["KK3_NOTR"])) < 1e-9
    pay_bar, pay_pla = abs(n1["dA"]) - K["bar"], abs(n1["dA"]) - n1["plasebo_p95"]
    assert pay_bar > 0 and pay_pla > 0
    print(f"  [PAYDA] kk3t2_seed: dA_T2={t2['dA']:.4f} · p95_T2={t2['plasebo_p95']:.4f} · pay_bar={pay_bar:.4f} · "
          f"pay_pla={pay_pla:.4f} · dT={h['dT']:+.4f} {h['ciT']} · ref={h['ref_tilt']:.4f} "
          f"⇒ esik: ad/bilesen degisirse ⇒ EYLEM: cümle YAZILMAZ")
    s = (f"Retrained with a second training seed, the flattened run moves $M_1$ by ${t2['dA']:.2f}$ and does not "
         f"clear its own paired placebo (${t2['plasebo_p95']:.2f}$); the first seed had cleared the pre-set bar by "
         f"${pay_bar:.2f}$ and its placebo by ${pay_pla:.2f}$. The two seeds differ by ${h['dT']:+.2f}$ "
         f"$[{h['ciT'][0]:+.2f},{h['ciT'][1]:+.2f}]$ per thousand, against the ${h['ref_tilt']:.2f}$ that separates "
         f"the flattened run from the natural one, a difference the registered rule reads as not separable from "
         f"prompt noise. That interval carries the sampling of prompts, not the variance across training seeds, "
         f"which one further seed cannot estimate, so the seeds are not shown to agree.")
    io.open(f"{OUT}/KK3T2_SEED.tex", "w", encoding="utf-8").write(s + "\n")
    json.dump(dict(table="KK3T2_SEED", ciktilar=["KK3T2_SEED.tex"], sources=[pk, pk3],
                   payda=dict(dA_T2=round(t2["dA"], 4), p95_T2=round(t2["plasebo_p95"], 4),
                              pay_bar=round(pay_bar, 4), pay_pla=round(pay_pla, 4),
                              dT=round(h["dT"], 4), ciT=[round(x, 4) for x in h["ciT"]], ref=round(h["ref_tilt"], 4)),
                   note="renders the sealed KK-3T2 reading; the second-seed share on the card is not used"),
              io.open(f"{OUT}/KK3T2_SEED.meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  ✓ KK3T2_SEED.tex · «{s}»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
