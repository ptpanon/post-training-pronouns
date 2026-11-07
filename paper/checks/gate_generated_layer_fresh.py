#!/usr/bin/env python3
from __future__ import annotations
import hashlib, os, subprocess, sys

PY = __import__("sys").executable
ATLA = {"style.py", "elicited16_loader.py", "urial_verdict_loader.py", "four_seed_loader.py"}
SONRA = {"f0_example.py": "example_pair.py"}


def _sha(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""):
            h.update(p)
    return h.hexdigest()[:16]


def tara(fig):
    return {f: _sha(os.path.join(fig, f)) for f in sorted(os.listdir(fig)) if f.endswith(".tex")}


def tex_yorum_sil(fig):
    n_dosya = n_satir = 0
    for f in sorted(os.listdir(fig)):
        if not f.endswith(".tex"):
            continue
        p = os.path.join(fig, f)
        t = open(p, encoding="utf-8").read()
        s = t.split("\n")
        k = [x for x in s if not x.lstrip().startswith("%")]
        if len(k) != len(s):
            open(p, "w", encoding="utf-8").write("\n".join(k))
            n_dosya += 1; n_satir += len(s) - len(k)
    return n_dosya, n_satir


def main():
    fig = os.path.abspath(sys.argv[1])
    kur = "--kur" in sys.argv
    once = tara(fig)
    uretici = [f for f in sorted(os.listdir(fig)) if f.endswith(".py") and f not in ATLA]
    for t, u in SONRA.items():
        if t in uretici and u in uretici:
            uretici.remove(t); uretici.insert(uretici.index(u) + 1, t)
    kosan = reddeden = ariza = 0
    arizalar, red = [], []
    for g in uretici:
        r = subprocess.run([PY, os.path.join(fig, g)], capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            kosan += 1
        elif r.returncode == 3:
            reddeden += 1; red.append(g)
        else:
            ariza += 1; arizalar.append((g, r.returncode, (r.stderr or r.stdout).strip().split("\n")[-1][:120]))
    if os.environ.get("P11_TEX_YORUMSUZ") == "1":
        yd, ys = tex_yorum_sil(fig)
        print(f"   [PAYDA] kapi22_yayim_kipi: üretici ciktisindan silinen tam-satir % yorum {ys} ({yd} dosya) "
              f"⇒ karsilastirma YAYIM biciminde (yorumsuz)")
    sonra = tara(fig)
    bayat = sorted(set(once) | set(sonra))
    bayat = [f for f in bayat if once.get(f) != sonra.get(f)]
    print(f"{len(uretici)} {len(ATLA)}"
          f"{kosan} {reddeden} {ariza}"
          f"{len(bayat)}"
          f"")
    for f in bayat:
        print(f"     ✗ BAYAT: fig/{f} · commit'li hâl kartiyla uyusmuyordu (tazelendi: "
              f"{once.get(f, 'YOK')} → {sonra.get(f, 'SILINDI')})")
    for g in red:
        print(f"{g}")
    for g, rc, m in arizalar:
        print(f"     ✗✗ ARIZA: fig/{g} rc={rc} :: {m}")
    print(f"   [PAYDA] kapi22: n_uretici={len(uretici)} · n_kosan={kosan} · n_atlanan={len(ATLA)} · "
          f"red_bayat={len(bayat)} · red_ariza={ariza} · hal_rule_reddi={reddeden}")
    if ariza:
        return 3
    if bayat and kur:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
