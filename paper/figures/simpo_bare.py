#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, load

CARD = "results/simpo_second_person_verdict_2026-09-13.json"


def main():
    K, pk = load(CARD)
    s_, d_ = K["basamak"]["SFT→SimPO"], K["basamak"]["SFT→DPO"]
    assert s_["ad"] == d_["ad"] == "AYNI-ISARET-SimPO", (s_["ad"], d_["ad"])
    assert K["ikinci_soru"]["ad"] == "SimPO-DAHA-BÜYÜK", K["ikinci_soru"]["ad"]
    assert not s_["plasebo_yutuyor"] and not d_["plasebo_yutuyor"]
    print(f"  [PAYDA] simpo_ciplak: SimPO={s_['dM1']:.4f} {s_['ci']} · DPO={d_['dM1']:.4f} {d_['ci']} "
          f"· ikinci={K['ikinci_soru']['ad']} ⇒ esik: ad degisirse ⇒ EYLEM: cümle YAZILMAZ")
    s = (f"both withdraw the second person on the raw-continuation protocol, the reference-free one further "
         f"(${s_['dM1']:.2f}$ $[{s_['ci'][0]:.2f},{s_['ci'][1]:.2f}]$ against ${d_['dM1']:.2f}$ "
         f"$[{d_['ci'][0]:.2f},{d_['ci'][1]:.2f}]$ per thousand), so the withdrawal is not specific to DPO; "
         f"each checkpoint was trained on its publisher's own pipeline, so the two objectives are not "
         f"compared at an equal data size.")
    io.open(f"{OUT}/SIMPO_CIPLAK.tex", "w", encoding="utf-8").write(s + "\n")
    json.dump(dict(table="SIMPO_CIPLAK", ciktilar=["SIMPO_CIPLAK.tex"], sources=[pk],
                   payda=dict(simpo=round(s_["dM1"], 4), dpo=round(d_["dM1"], 4)),
                   note="renders the sealed SimPO raw-continuation-protocol reading; no statistic computed here"),
              io.open(f"{OUT}/SIMPO_CIPLAK.meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  ✓ SIMPO_CIPLAK.tex · «{s}»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
