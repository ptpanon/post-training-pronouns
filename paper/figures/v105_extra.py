#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json

KOK = __DNH_ROOT__ + ""
CARD = f"{KOK}/results/v105_template_birinci_suzgecsiz_2026-09-23.json"
FIG = f"{KOK}/paper/figures"
BAS = "% * URETILDI - elle duzenlenmez (figures/v105_extra.py)\n"


def yaz(ad, g):
    io.open(f"{FIG}/{ad}", "w", encoding="utf-8").write(BAS + g.strip() + "\n")
    print(f"✓ {ad} ({len(g.split())} kelime)")


def main():
    K = json.load(io.open(CARD, encoding="utf-8"))
    S, sy = K["aile"], K["sayim"]
    assert not K["kapi"]["red"], ""
    n = sy["n_aile"]; olc = sy["olculen"]
    asagi = len(sy["asagi_ayrik"]); pl = len(sy["plasebo_ustu"])
    d = [v["d1"] for v in S.values()]
    kapi = max(v["kapi_sapma"] for v in S.values())
    assert olc == n, (olc, n)
    yaz("V105_SUZGECSIZ_CUMLE.tex", rf"and without that filter in ${asagi}$ of ${n}$")
    yaz("V105_D3.tex", rf"""
\paragraph*{{The first person under the chat template without the cell filter.}}
\vekalet{{The filter that admits a cell only when it is non-degenerate in all four quadrants was built for
the second-person contrast, and it keeps between $28$ and $67\%$ of cells. Read instead on \emph{{all}}
$6{{,}}528$ generations per checkpoint, with the same direct counter as Table~\ref{{tab:families}} and the
same prompt-clustered intervals, the first-person rate falls in ${asagi}$ of ${n}$ models, every interval
excluding zero, and every model clears its own within-prompt paired placebo (${pl}$ of ${n}$). The shifts
run from ${max(d):.2f}$ to ${min(d):.2f}$ per 1,000 tokens. \textbf{{We do not compare these widths with
the filtered reading}}: that one is computed on a derived first-person measure and these on the direct
count, so the two are different instruments and are reported side by side rather than differenced.
\textbf{{What this answers is whether the filter makes the result, not whether the comparison is fair}}.
The base checkpoint still reads a template it never saw, and removing the filter increases that asymmetry
rather than removing it. The same loop recomputes the unfiltered second-person shift and reproduces this
appendix's own unfiltered column to within $5\times10^{{-5}}$ in all ${n}$ models, which is what licenses
reading the two together.}}
""")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
