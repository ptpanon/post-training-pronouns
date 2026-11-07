#!/usr/bin/env python
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reading_style_object import payda

SK = __DNH_DATA__ + "/c1_panel_template"
KOK = __DNH_ROOT__ + ""
CIK = f"{KOK}/results/TEMPLATE_TABAN_KIRLILIK_2026-09-06.json"


def _onekler(yol, n=200):
    o = []
    with io.open(yol, encoding="utf-8") as fh:
        for i, l in enumerate(fh):
            if i >= n:
                break
            o.append(json.loads(l)["onek"])
    return o


def _zarf(onek, govde):
    i = onek.find(govde)
    return (onek[:i], onek[i + len(govde):]) if i >= 0 else (None, None)


def dize_esit(aile, zemin, ref, n=200, kok=None):
    K = kok or SK
    ya, yb = f"{K}/{aile}/{zemin}/uretim.jsonl", f"{K}/{aile}/{ref}/uretim.jsonl"
    if not (os.path.exists(ya) and os.path.exists(yb)):
        return None, 0, 0, None, None
    oa, ob = _onekler(ya, n), _onekler(yb, n)
    m = min(len(oa), len(ob))
    farkli, orn_a, orn_b = 0, None, None
    for a, b in zip(oa[:m], ob[:m]):
        if a == b:
            continue
        farkli += 1
        if orn_a is None:
            p = 0
            while p < min(len(a), len(b)) and a[p] == b[p]:
                p += 1
            orn_a, orn_b = a[:p + 60], b[:p + 60]
    return (farkli == 0), m, farkli, orn_a, orn_b


def main():
    REF = ("instruct", "rl")
    ciftler = []
    for d in sorted(os.listdir(SK)):
        if not os.path.isdir(f"{SK}/{d}"):
            continue
        zeminler = [z for z in sorted(os.listdir(f"{SK}/{d}"))
                    if os.path.exists(f"{SK}/{d}/{z}/uretim.jsonl")]
        ref = next((r for r in REF if r in zeminler), None)
        if ref is None:
            continue
        for z in zeminler:
            if z != ref:
                ciftler.append((d, z, ref))
    satir, n_ayni, n_farkli = [], 0, 0
    for ad, zem, ref in ciftler:
        esit, n_es, farkli, orn_a, orn_b = dize_esit(ad, zem, ref)
        etiket = f"{ad}/{zem}↔{ref}"
        hal = ("CÖZÜLEMEDI" if esit is None else ("AYNI-DIZE" if esit else "FARKLI-DIZE"))
        n_ayni += hal == "AYNI-DIZE"
        n_farkli += hal == "FARKLI-DIZE"
        satir.append(dict(kol=etiket, hal=hal, n_karsilastirilan=n_es, n_farkli=farkli,
                          taban_bas=orn_a, hizali_bas=orn_b))
        print(f"{etiket:34s} {hal:12s} {farkli} {n_es}", flush=True)
        if farkli:
            print(f"      taban  : {orn_a[:120]!r}")
            print(f"      hizali : {orn_b[:120]!r}")

    payda("template_taban_kirlilik", n_cift=len(ciftler), hal_ayni=n_ayni,
          red_farkli=n_farkli)
    print(f""
          f""
          f"")
    json.dump(dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   SINIF="ONARIM/TESHIS · yalniz DÜSÜRME yetkisi",
                   kaynak_kusur="W-972", kollar=satir,
                   payda=dict(n_cift=len(ciftler), hal_ayni=n_ayni, red_farkli=n_farkli)),
              io.open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"★ künye: {CIK}")


if __name__ == "__main__":
    main()
