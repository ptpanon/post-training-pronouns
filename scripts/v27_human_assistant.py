#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, os, gzip, json, random, sys, time
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
import human_band as IB
import form_count as FS
from reading_style_object import payda

KOK = __DNH_ROOT__ + ""
KAYNAK = (__DNH_DATA__ + "/.hf/hub/datasets--OpenAssistant--oasst1/"
          "snapshots/*/2023-04-12_oasst_ready.messages.jsonl.gz")
CIK = os.environ.get("V27_CIK", f"{KOK}/results/V27_INSAN_ASISTAN_2026-09-09.json")
SEED = 20260910


def rezervuar(rol, n=IB.N_HEDEF):
    yol = sorted(glob.glob(KAYNAK))
    if not yol:
        return None, dict(hal="KAYNAK-YOK", desen=KAYNAK)
    rng = random.Random(SEED); R = []; gor = kisa = dil = rolsuz = uygun = 0
    with gzip.open(yol[0], "rt", encoding="utf-8") as f:
        for ham in f:
            try:
                d = json.loads(ham)
            except Exception:
                continue
            gor += 1
            if d.get("role") != rol:
                rolsuz += 1; continue
            if d.get("lang") != "en":
                dil += 1; continue
            t = (d.get("text") or "").strip()
            if len(t.split()) < IB.ASGARI_KELIME:
                kisa += 1; continue
            kay = dict(text=t, meta=dict(tree=d.get("message_tree_id")))
            uygun += 1
            IB.rezervuar_al(R, kay, uygun, n, rng)
    return R, dict(n_satir_gorulen=gor, n_uygun=uygun, red_rol=rolsuz, red_dil=dil,
                   red_kisa=kisa, n_secilen=len(R), kaynak=yol[0].split("/")[-1])


def main():
    nlp = FS._boru(); t0 = time.time()
    D = {"_kunye": dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        alet="scripts/v27_human_assistant.py", borc="D-0910-V27 §2h",
                        SINIF="KESIF/BETIM — bar YOK · ad YOK · prediction YOK (§10)",
                        motor="human_band.olc (ITHAL, edit yok)",
                        n_hedef=IB.N_HEDEF, asgari_kelime=IB.ASGARI_KELIME,
                        kume="message_tree_id", dil="en", seed=SEED,
                        not_="ham metin repoya YAZILMADI (§9)")}
    for rol in ("assistant", "prompter"):
        R, p = rezervuar(rol)
        print(f"  [PAYDA] oasst_{rol}: " +
              " · ".join(f"{k}={v}" for k, v in p.items()) +
              f" ⇒ esik: n_secilen≥2000 ⇒ EYLEM: altindaysa ÖLCÜLEMEZ", flush=True)
        if R is None or len(R) < 2000:
            D[rol] = dict(hal="ÖLCÜLEMEZ", payda=p); continue
        D[rol] = dict(hal="ÖLCÜLDÜ", payda=p,
                      **IB.olc(nlp, R, lambda d: (d.get("meta") or {}).get("tree")))
        v = D[rol]
        print(f"  [{time.time()-t0:.0f}s] {rol}: n={v['n']} n_kume={v['n_kume']} "
              f"M1={v['M1']:.3f} CI={[round(x,2) for x in v['M1_ci']]} "
              f"M3={v['M3']:.3f} jeton_ort={v['jeton_ort']:.1f}", flush=True)
    json.dump(D, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("v27_insan_asistan", n_rol=2,
          hal_olculen=sum(1 for r in ("assistant", "prompter")
                          if D.get(r, {}).get("hal") == "ÖLCÜLDÜ"), red_olculemez=0)
    print(f"✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
