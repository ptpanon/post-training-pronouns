#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys, time, hashlib
ROOT = __DNH_ROOT__ + ""
OUT = os.path.dirname(os.path.abspath(__file__))
CARD = "results/person_dial_seeds_2_3_verdict_2026-09-15.json"
AD = {"KISI_ASAGI": "person-down", "KISI_YUKARI": "person-up"}
TOH = {"t1": "1 (20260830)", "t2": "2 (20260915)", "t3": "3 (20260916)"}


def f(x): return f"{x:+.2f}".replace("-", "$-$").replace("+", "$+$")
def ci(c): return f"[{f(c[0])}, {f(c[1])}]"


def main():
    p = f"{ROOT}/{CARD}"; D = json.load(io.open(p, encoding="utf-8"))
    H = D["verdict"]
    if H.get("genel") in (None, "KOSULMADI"):
        print(f"  ★★ verdict {H.get('genel')} ⇒ tablo basilmaz"); return 3
    L = [r"\paragraph*{The person-selected runs at three training seeds.}",
         r"\vekalet{Both person runs were retrained twice more from the same pairs and the same supervised checkpoint, changing only the training seed, "
         r"and read over the same four generation seeds. The registered bar was the sign of the displacement against the data-size-matched unselected run "
         r"in each new seed; magnitudes are given as ranges only.}",
         r"\begin{center}\scriptsize\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{llrrrr}\toprule",
         r"run & training seed & $\Delta M_1$ vs.\ unselected & 95\% CI & minus seed 1 [CI] & $\Delta M_1$ vs.\ SFT \\ \midrule"]
    for kol in ("KISI_ASAGI", "KISI_YUKARI"):
        K = H["kol"][kol]
        for t in ("t1", "t2", "t3"):
            T = K["seed"][t]
            fk = ("---" if t == "t1" else f"{f(K['seed1_ile_fark'][t]['fark'])} {ci(K['seed1_ile_fark'][t]['ci'])}")
            L.append(f"{AD[kol] if t == 't1' else ''} & {TOH[t]} & {f(T['dB'])} & {ci(T['ciB'])} & {fk} & {f(T['dA'])} \\\\")
        a = K["aralik_dB"]
        L.append(f" & range, three seeds & [{f(a[0])}, {f(a[1])}] & & & \\\\")
        if kol == "KISI_ASAGI":
            L.append(r"\midrule")
    L += [r"\bottomrule\end{tabular}\end{center}"]
    io.open(f"{OUT}/DIAL_SEED23.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/person_dial_seeds_2_3.py)\n" + "\n".join(L) + "\n")
    json.dump(dict(table="DIAL_SEED23", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   sources=[dict(yol=CARD, sha256_16=hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])],
                   verdict=dict(genel=H["genel"], asagi=H.get("kisi_asagi"), yukari=H.get("kisi_yukari")),
                   note="renders the sealed verdict card; computes no new statistic"),
              io.open(f"{OUT}/DIAL_SEED23.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✓ DIAL_SEED23.tex · verdict {H['genel']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
