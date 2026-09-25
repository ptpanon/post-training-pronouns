#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, os, re, sys


def atif_denetimi(log):
    if not os.path.exists(log):
        return None, []
    t = io.open(log, encoding="utf-8", errors="ignore").read()
    kes = t.rfind("Rerunning")
    son = t[kes:] if kes > 0 else t
    k = sorted(set(re.findall(r"Citation `([^']+)' on page", son)))
    return len(re.findall(r"Citation `[^']+' on page", son)), k


def bant_denetimi(tex, ek):
    kay = [tex] + ek
    bulgu, ihlal = 0, []
    P1 = re.compile(r"\$([\d.,]+)\$\s*(?:to|--)\s*\$([\d.,]+)\$[^.$]{0,60}?"
                    r"a\s*\$([\d.,]+)\\times\$", re.S)
    for y in kay:
        if not os.path.exists(y):
            continue
        t = re.sub(r"(?<!\\)%.*", "", io.open(y, encoding="utf-8").read())
        t = re.sub(r"\s+", " ", t)
        for m in P1.finditer(t):
            a, b, r = (float(x.replace(",", "")) for x in m.groups())
            bulgu += 1
            if a <= 0:
                continue
            if abs(round(b / a) - r) > 0.5:
                ihlal.append((os.path.basename(y), m.group(0)[:90],
                              f"{b}/{a}={b/a:.2f} ⇒ {round(b/a)}×, metin {r:.0f}×"))
    return bulgu, ihlal


def main():
    tex = sys.argv[1] if len(sys.argv) > 1 else "p11.tex"
    kok = os.path.dirname(os.path.abspath(tex)) or "."
    log = os.path.join(kok, "p11_submission.log")
    sys.path.insert(0, __DNH_ROOT__ + "/scripts")
    from tex_tree import agac as _agac
    ek = [os.path.join(kok, y)
          for y in _agac(os.path.basename(tex), kok, kip="submission")[0][1:]]
    n_at, anahtar = atif_denetimi(log)
    n_b, ihl_b = bant_denetimi(tex, ek)
    print(f"★ KAPI-19 · (A) tanimsiz atif = {0 if n_at is None else len(anahtar)} "
          f"(son gecis, {n_at if n_at is not None else 'log YOK'} satir) · "
          f"(B) bant↔oran bulgusu = {n_b} · celiski = {len(ihl_b)} "
          f"⇒ esik 0/0 ⇒ EYLEM: >0 ise DERLEME DÜSER")
    for a in anahtar:
        print(f"  ✗ TANIMSIZ ATIF: {a}")
    for y, met, ne in ihl_b:
        print(f"  ✗ BANT↔ORAN: {y} · {met} · {ne}")
    print(f"  [PAYDA] kapi19: n_kaynak={1+len(ek)} · n_bant_bulgu={n_b} · "
          f"n_atif_anahtar={0 if n_at is None else len(anahtar)} · "
          f"red_log_yok={int(n_at is None)}")
    if n_at is None:
        print("")
    return 19 if (anahtar or ihl_b) else 0


if __name__ == "__main__":
    sys.exit(main())
