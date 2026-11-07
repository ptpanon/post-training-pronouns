#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")

import glob
import os
import sys

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

DESENLER = ("*.safetensors", "pytorch_model-*.bin", "pytorch_model.bin")
INDEKSLER = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def dosyalar(snapshot: str, coz_bag: bool = True) -> list[str]:
    out = []
    for d in DESENLER:
        for p in glob.glob(os.path.join(snapshot, d)):
            if not coz_bag or os.path.exists(os.path.realpath(p)):
                out.append(p)
    return sorted(set(out))


def bicim(yollar) -> str | None:
    if not yollar:
        return None
    b = os.path.basename(yollar[0])
    return "safetensors" if b.endswith(".safetensors") else "pytorch_model.bin"


def bayt(yollar) -> int:
    return sum(os.path.getsize(os.path.realpath(p)) for p in yollar)


def tam(snapshot: str, tokenizer_da: bool = True) -> tuple[bool, dict]:
    ag = dosyalar(snapshot)
    ix = [i for i in INDEKSLER if os.path.exists(os.path.join(snapshot, i))]
    cfg = os.path.exists(os.path.join(snapshot, "config.json"))
    tk = [f for f in ("tokenizer.json", "tokenizer.model", "vocab.json")
          if os.path.exists(os.path.join(snapshot, f))]
    d = dict(n_agirlik=len(ag), hal_bicim=bicim(ag), hal_gib=round(bayt(ag) / 2**30, 2),
             n_indeks=len(ix), hal_config=cfg, n_tokenizer=len(tk))
    ok = bool(ag) and cfg and (bool(tk) or not tokenizer_da)
    if len(ag) > 1 and not ix:
        d["red_indeks_yok"] = 1
        ok = False
    return ok, d


def sec(depo: str) -> str | None:
    aday = sorted(g for g in glob.glob(os.path.join(depo, "snapshots", "*"))
                  if os.path.isdir(g))
    tamlar = [g for g in aday if tam(g)[0]]
    return tamlar[-1] if tamlar else None


def _prova() -> int:
    import json
    import shutil
    import tempfile
    kok = tempfile.mkdtemp(prefix="agirlik_prova_")
    try:
        kip = []
        a = os.path.join(kok, "a"); os.makedirs(a)
        for f in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                  "model.safetensors.index.json", "config.json", "tokenizer.json"):
            open(os.path.join(a, f), "wb").write(b"x" * 1024)
        kip.append(("safetensors-TAM", tam(a)[0], True))
        b = os.path.join(kok, "b"); os.makedirs(b)
        for f in ("pytorch_model-00001-of-00003.bin", "pytorch_model-00002-of-00003.bin",
                  "pytorch_model-00003-of-00003.bin", "pytorch_model.bin.index.json",
                  "config.json", "vocab.json"):
            open(os.path.join(b, f), "wb").write(b"x" * 1024)
        kip.append(("bin-TAM ★", tam(b)[0], True))
        c = os.path.join(kok, "c"); os.makedirs(c)
        for f in ("config.json", "tokenizer.json"):
            open(os.path.join(c, f), "wb").write(b"x")
        kip.append(("agirliksiz", tam(c)[0], False))
        d_ = os.path.join(kok, "d"); os.makedirs(d_)
        for f in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                  "config.json", "tokenizer.json"):
            open(os.path.join(d_, f), "wb").write(b"x")
        kip.append(("indekssiz-parcali", tam(d_)[0], False))
        e = os.path.join(kok, "e"); os.makedirs(e)
        os.symlink(os.path.join(kok, "olmayan.safetensors"),
                   os.path.join(e, "model.safetensors"))
        for f in ("config.json", "tokenizer.json"):
            open(os.path.join(e, f), "wb").write(b"x")
        kip.append(("kirik-bag", tam(e)[0], False))
        kip.append(("bicim-adi", bicim(dosyalar(b)) == "pytorch_model.bin", True))
        dep = os.path.join(kok, "depo"); os.makedirs(os.path.join(dep, "snapshots"))
        import shutil as _sh
        _sh.copytree(a, os.path.join(dep, "snapshots", "0aaa"))
        _bos = os.path.join(dep, "snapshots", "9zzz"); os.makedirs(_bos)
        open(os.path.join(_bos, "model.safetensors"), "wb").write(b"x")
        kor = sorted(glob.glob(os.path.join(dep, "snapshots", "*")))[-1]
        kip.append(("secici-kor-yanlis", os.path.basename(kor) == "9zzz", True))
        kip.append(("secici-TAM-secer ★", os.path.basename(sec(dep)) == "0aaa", True))
        gecen = sum(1 for _, g, bek in kip if g == bek)
        for ad, g, bek in kip:
            print(f"    {ad:<20} → {g}  (beklenen {bek}) {'✔' if g == bek else '★'}")
        payda("agirlik_prova", n_kip=len(kip), hal_gecen=gecen,
              red_sapan=len(kip) - gecen)
        print(f"  [PAYDA] n_kip={len(kip)} · hal_gecen={gecen} ⇒ esik {len(kip)}/{len(kip)} "
              f"⇒ {'PROVA GECTI' if gecen == len(kip) else '★ PROVA DÜSTÜ'}")
        return 0 if gecen == len(kip) else 5
    finally:
        shutil.rmtree(kok, ignore_errors=True)


def _denetle() -> int:
    import re
    import subprocess
    r = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    dar, genis, n = [], [], 0
    _st = re.compile(r"glob\([^)]*\*\.safetensors")
    _bin = re.compile(r"\*\.bin|pytorch_model")
    for y in r.stdout.splitlines():
        if y.endswith("weight_dosyalari.py"):
            continue
        try:
            m = open(os.path.join(ROOT, y), encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for satir_no, l in enumerate(m.splitlines(), 1):
            if _st.search(l):
                n += 1
                (genis if _bin.search(l) else dar).append(f"{y}:{satir_no}")
    payda("agirlik_desen_envanteri", n_govde=n, hal_genis=len(genis),
          hal_dar=len(dar))
    print(f"{len(dar)}")
    for x in dar:
        print(f"      {x}")
    print(f"  ⇒ esik hal_dar=0 ⇒ {'TEMIZ' if not dar else '★ SÜPÜRME BORCU — kütüge'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--denetle":
        sys.exit(_denetle())
    sys.exit(_prova())
