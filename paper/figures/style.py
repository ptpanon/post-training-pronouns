#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = __DNH_ROOT__ + ""
OUT = f"{ROOT}/paper/figures"
COL = 3.25
MDE = 1.645
BAND_LO = 1.00

C = dict(pos="#1b6ca8", neg="#c1440e", null="#8a8f98", band="#d9dce1",
         band2="#eef0f2", ink="#1a1a1a", light="#b8bcc4", accent="#7a4fa3")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": C["ink"],
    "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "legend.fontsize": 6, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def load(rel):
    p = f"{ROOT}/{rel}" if not rel.startswith("/") else rel
    return json.load(open(p, encoding="utf-8")), dict(yol=rel, sha256_16=sha16(p))


def null_band(ax, horizontal=True, lo=BAND_LO, hi=MDE):
    f = ax.axhspan if not horizontal else ax.axvspan
    f(-hi, hi, color=C["band2"], lw=0, zorder=0)
    f(-lo, lo, color=C["band"], lw=0, zorder=0)
    (ax.axvline if horizontal else ax.axhline)(0, color=C["null"], lw=0.6, zorder=1)


def save(fig, ad, kaynaklar, altyazi):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{ad}.{ext}")
    plt.close(fig)
    json.dump(dict(figure=ad, caption_draft=altyazi, sources=kaynaklar,
                   note="renders pre-specified measurements; computes no new statistic"),
              open(f"{OUT}/{ad}.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  ✓ {ad}.pdf / .png  ← {len(kaynaklar)} source(s)")


EN = {
    "SECICILIK-CÖZÜLMEDI": "selectivity unresolved",
    "OKUMA-SAKINMASI-VAR": "addressee-selective reading",
    "OKUMA-KOLAYLASMASI-VAR": "addressee-facilitated reading",
    "BAGLAM-GENEL": "context-general",
    "BAGLAM-KOLAYLASTIRICI-GENEL": "context-facilitating (general)",
    "OKUMA-SÖZDIZIM": "syntax-selective",
    "OKUMA-SESSIZ": "no context trace",
    "KAPI/ÖLCÜLEMEZ": "unmeasurable (gate)",
    "BAGLAM-YADIRGATIYOR": "context impedes",
    "MUHATAP-SECICI": "addressee-selective",
    "SÖZDIZIM-SECICI": "syntax-selective",
    "BAGLAM-IZI-YOK": "no context trace",
    "BAGLAM-CATISIK": "context conflicted",
    "MUHATAP-KOLAYLASTIRIYOR": "addressee facilitates",
    "SÖZDIZIM-KOLAYLASTIRIYOR": "syntax facilitates",
    "BAGLAM-KOLAYLASTIRIYOR": "context facilitates",
    "ILAN-SESSIZ": "assertion silent",
    "ILAN-DAGINIK": "assertion scattered",
    "ILAN-KOLAYLASTIRICI-GENEL": "assertion facilitates (general)",
    "ILAN-YADIRGATICI-GENEL": "assertion impedes (general)",
    "ILAN-KOLAYLASTIRIYOR": "assertion facilitates",
    "ILAN-YADIRGATIYOR": "assertion impedes",
    "ILAN-IZ-YOK": "assertion silent",
    "BASAMAK-SESSIZ": "no stage disjoint",
    "BASAMAK-CATISIK": "stages conflict",
    "BASAMAK-DAGILMIS": "effect spread over stages",
    "IMZA-AILEYE-GÖRE": "signature is model-specific",
    "IC-MAKSIMUM": "interior maximum",
    "YALNIZ-OKUYUCU": "reader side only",
    "UC-MAKSIMUM": "boundary maximum",
    "AYRIM-VAR": "separation present",
    "D-DISINDA": "travel is off-axis",
    "IKISI-DE": "both columns",
    "IZ-YOK": "no trace",
    "ÖLCÜLEMEZ": "unmeasurable",
    "SILME-TEK-DELTADA": "erasure lives in the single delta",
    "DAGINIK": "scattered",
    "DELTA-SILER": "delta erases it",
    "DELTA-KISMEN": "delta erases it partly",
    "DELTA-SILMEZ": "delta does not erase it",
    "PANEL-KAPI": "panel unmeasurable (gate)",
    "PROTOKOLE-BAGIMLI": "protocol-dependent",
    "DÜSER": "falls", "DÜSMEZ": "does not fall",
    "GERI-GELIR": "returns", "GELMEZ": "does not return",
    "IGNE-FORMDA-GÖRÜNÜR": "needle visible in form",
    "IGNE-FORMDA-GÖRÜNMEZ": "needle invisible in form",
    "CAPA-KURULAMADI-DAVRANISSAL": "behavioural anchor not established",
    "CETVEL-D'YI-GÖSTERIR": "ruler tracks D",
    "GÖSTERMEZ": "ruler does not track D",
    "KIP-TASIR": "mood carries it", "YALINLIK-TASIR": "brevity carries it",
    "SABLONA-BAGIMLI": "template-dependent",
    "DPO-BASAMAGI-GÜRBÜZ": "DPO stage is robust",
}


def en(ad):
    if ad in EN:
        return EN[ad]
    for tr, eng in EN.items():
        if ad.endswith("-KURAR") and ad[:-6] in ("SFT", "DPO", "RLVR"):
            return f"{ad[:-6]} builds it"
        if ad.endswith("-SÖNDÜRÜR") and ad[:-9] in ("SFT", "DPO", "RLVR"):
            return f"{ad[:-9]} damps it"
    return ad


def t1_adi(a, tex=True):
    s = str(a).replace("--", "-").replace("Qwen-2.5", "Qwen2.5").replace("OLMo2-", "OLMo-2-").replace("Tulu3-", "Tulu-3-")
    s = s.replace("Tulu-3-", 'T\\"ulu-3-')
    return s if tex else s.replace('T\\"ulu', "T\u00fclu")
