#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json
ROOT = __DNH_ROOT__ + ""
FIG = f"{ROOT}/paper/figures"
CARD = f"{ROOT}/results/qwen_base_own_template_2026-09-24.json"
SIRA = ("Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B")


def s(x, n=2):
    return f"${x:+.{n}f}$"


def main():
    K = json.load(io.open(CARD, encoding="utf-8")); A = K["aile"]
    assert tuple(A) == SIRA, list(A)
    boy = {a: a.split("-")[1] for a in SIRA}
    d1 = {a: A[a]["d1st"] for a in SIRA}; d2 = {a: A[a]["d2nd"] for a in SIRA}
    assert all(d1[a]["ayrik"] and d1[a]["delta"] < 0 for a in SIRA), "«I» dördünde ayrik düsmüyor"
    asagi = [a for a in SIRA if d2[a]["ayrik"] and d2[a]["delta"] < 0]
    yukari = [a for a in SIRA if d2[a]["ayrik"] and d2[a]["delta"] > 0]
    belirsiz = [a for a in SIRA if not d2[a]["ayrik"]]
    assert (asagi, yukari, belirsiz) == (["Qwen2.5-3B", "Qwen2.5-7B"], ["Qwen2.5-1.5B"], ["Qwen2.5-14B"]), (asagi, yukari, belirsiz)
    n = {a: A[a]["n_tutulan"] for a in SIRA}; N = A[SIRA[0]]["n_cift"]
    assert all(A[a]["n_bos_tutulan"] == 0 for a in SIRA)
    diger = [n[a] for a in SIRA if a != "Qwen2.5-1.5B"]
    v = lambda x: f"{x:,}".replace(",", "{,}")
    metin = (
        r"\paragraph*{Qwen2.5 bases read through their own chat template.}" "\n"
        r"\vekalet{Qwen2.5's base checkpoints ship a chat template of their own, so the chat-template contrast "
        r"can also be read with each checkpoint in its own template and the base in distribution. The two "
        r"templates render our prompts identically except for the default system line (``You are a helpful assistant.'' against ``You are "
        r"Qwen, created by Alibaba Cloud. You are a helpful assistant.''). We read four of the six sizes "
        r"(1.5B, 3B, 7B and 14B); the 32B generations exist and were not read, and no 72B base generations were "
        r"made. ``I'' falls in all four (" + ", ".join(s(d1[a]["delta"]) for a in SIRA)
        + r" per 1,000 tokens), every interval excluding zero. ``You'' falls in 3B and 7B ("
        + s(d2["Qwen2.5-3B"]["delta"]) + " and " + s(d2["Qwen2.5-7B"]["delta"]) + r"), is unresolved in 14B ("
        + s(d2["Qwen2.5-14B"]["delta"]) + r", interval covering zero) and rises in 1.5B ("
        + s(d2["Qwen2.5-1.5B"]["delta"]) + r"), where the degeneracy filter keeps $" + v(n["Qwen2.5-1.5B"])
        + r"$ of $" + v(N) + r"$ pairs against $" + v(min(diger)) + r"$ to $" + v(max(diger))
        + r"$ in the other three. With the base in distribution the pattern is the one \S\ref{sec:deperson} "
        r"reports under the aligned template: ``I'' falls and ``you'' splits. The reading is descriptive and "
        r"was registered before it was computed.}" "\n")
    io.open(f"{FIG}/QWEN_KENDI_D3.tex", "w", encoding="utf-8").write(
        "% * URETILDI - elle duzenlenmez (figures/qwen_own_template.py)\n" + metin)
    json.dump(dict(table="QWEN_KENDI_D3", sources=[{"yol": "results/qwen_base_own_template_2026-09-24.json"}],
                   ciktilar=["QWEN_KENDI_D3.tex"], n_aile=len(SIRA),
                   sayim_d2nd=dict(asagi=len(asagi), yukari=len(yukari), belirsiz=len(belirsiz))),
              io.open(f"{FIG}/QWEN_KENDI_D3.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(asagi)} {len(yukari)} {len(belirsiz)}"
          f"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
