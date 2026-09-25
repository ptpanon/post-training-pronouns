#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, hashlib, time
ROOT = __DNH_ROOT__ + ""
CARD = f"{ROOT}/results/template_filtered_2026-09-07.json"
OUT = os.path.dirname(os.path.abspath(__file__))


def sha16(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def main():
    K = json.load(io.open(CARD, encoding="utf-8"))
    A = [a for a in K["aileler"] if a.get("hal") == "ÖLCÜLDÜ"]
    tb = [a["sbl"]["M1_taban"] for a in A]
    hz = [a["sbl"]["M1_hizali"] for a in A]
    rt, rh = max(tb) / min(tb), max(hz) / min(hz)
    print(f"{len(A)} {min(tb):.2f} {max(tb):.2f}"
          f"{rt:.2f} {min(hz):.2f} {max(hz):.2f} {rh:.2f}"
          f"")
    io.open(f"{OUT}/BANT_OZET.tex", "w", encoding="utf-8").write(
        r"the aligned checkpoints run from $%.1f$ to $%.1f$ per thousand, a "
        r"$%.1f\times$ span" % (min(hz), max(hz), rh) + "\n")
    io.open(f"{OUT}/BANT_REL.tex", "w", encoding="utf-8").write(
        r"the base band spans $%.1f\times$, the aligned band $%.1f\times$"
        % (rt, rh) + "\n")
    import numpy as _np
    _g = lambda x: float(_np.exp(_np.std(_np.log([v for v in x if v > 0]), ddof=1)))
    gt, gh = _g(tb), _g(hz)
    print(f"  [PAYDA] bant_gsd: n_taban={len(tb)} · n_hizali={len(hz)} · n_atlanan={sum(1 for v in tb+hz if v <= 0)} "
          f"· gsd_taban={gt:.4f} · gsd_hizali={gh:.4f} ⇒ esik: atlanan>0 ⇒ EYLEM: künyede adiyla basilir")
    io.open(f"{OUT}/BANT_GSD.tex", "w", encoding="utf-8").write(
        r"geometric standard deviations of $%.2f$ for the base checkpoints and $%.2f$ for the aligned ones"
        % (gt, gh) + "\n")
    json.dump(dict(table="BANT", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=os.path.relpath(CARD, ROOT), sha256_16=sha16(CARD))],
                   okuma="dejenerelik SÜZÜLMÜS (2×2'nin yaslandigi card)",
                   taban=[min(tb), max(tb)], hizali=[min(hz), max(hz)],
                   oran_taban=rt, oran_hizali=rh, n_aile=len(A)),
              io.open(f"{OUT}/BANT.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("✓ BANT_OZET.tex + BANT_REL.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
