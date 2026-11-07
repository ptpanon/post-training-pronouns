#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, subprocess, sys, time
import numpy as np
ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
import example_pair as OC

CARD_SS = f"{ROOT}/results/template_filtered_2026-09-07.json"
AILELER = ["Gemma-3-27B", "Tulu3-8B"]
NM = {"Gemma-3-27B": "Gemma-3-27B", "Tulu3-8B": r"T\"ulu-3-8B"}
KURALLAR = {"preregistration/rule_example_pair_selection.md": "ceb38255", "preregistration/rule_reviewer6_readings_2026-09-17.md": "84f79b65"}


def rule_commitleri():
    for yol, bek in KURALLAR.items():
        h = subprocess.run(["git", "-C", ROOT, "log", "--format=%h", "--abbrev=8", "--diff-filter=A", "--", yol],
                           capture_output=True, text=True, check=True).stdout.split()
        assert len(h) == 1 and h[0].startswith(bek), (yol, h)


def sec(nlp, MK, aile, L, ilk_kol, jargon, anon, istem_kaynagi=None):
    ist = np.array([r["istem_i"] for r in L["taban"][0]]); ish = np.array([r["istem_i"] for r in L["hizali"][0]])
    dp = {}
    for p in sorted(set(ist.tolist())):
        mt = L["taban"][1][ist == p]; mh = L["hizali"][1][ish == p]
        dp[p] = 1000 * mh[:, 1].sum() / mh[:, 0].sum() - 1000 * mt[:, 1].sum() / mt[:, 0].sum()
    med = float(np.median(list(dp.values())))
    sira = sorted(dp, key=lambda p: (abs(dp[p] - med), p))
    dusen = []
    for p in sira:
        bul = lambda R: next((r for r in R if r["kol"] == ilk_kol and r["cekim"] == 0 and r["istem_i"] == p), None)
        rt, rh = bul(L["taban"][0]), bul(L["hizali"][0])
        if not rt or not rh or not rt.get("metin") or not rh.get("metin"):
            dusen.append((p, "missing row")); continue
        ham_istem = rt["onek"] if istem_kaynagi is None else istem_kaynagi[(ilk_kol, 0, p)]
        if istem_kaynagi is not None:
            assert ham_istem and ham_istem in rt["onek"] and ham_istem in rh["onek"], (aile, p, "")
        ist_m, nj_i, kes_i = OC.kes(nlp, ham_istem)
        tm, nj_t, kes_t = OC.kes(nlp, rt["metin"]); hm, nj_h, kes_h = OC.kes(nlp, rh["metin"])
        birlesik = " ".join((ist_m, tm, hm))
        if OC.UYGUNSUZ.search(birlesik):
            dusen.append((p, "uygunsuz icerik listesi")); continue
        if any(j.search(birlesik) for j in jargon) or anon.search(birlesik):
            dusen.append((p, "derleme kapisi sözlügü")); continue
        return dict(aile=aile, istem_i=p, dM1_istem=dp[p], medyan=med, kol=ilk_kol, cekim=0, tez=rt.get("tez"), durus=rt.get("durus"),
                    dusen=dusen, istem=ist_m, taban=tm, hizali=hm, n_jeton=dict(istem=nj_i, taban=nj_t, hizali=nj_h),
                    kesildi=dict(istem=kes_i, taban=kes_t, hizali=kes_h),
                    sha_metin=dict(istem=OC.sha16b(ist_m.encode()), taban=OC.sha16b(tm.encode()), hizali=OC.sha16b(hm.encode())),
                    sha_uretim=dict(taban=OC.sha16f(L["taban"][2]), hizali=OC.sha16f(L["hizali"][2])), n_istem=len(dp))
    return None


def main():
    rule_commitleri()
    import addressee_run as MK, form_count as FS, template_filtered as SS
    t0 = time.time(); nlp = FS._boru()
    A = json.load(open(OC.CARD, encoding="utf-8"))
    ilk_kol = [k for k, v in json.load(open(OC.KOLT, encoding="utf-8"))["onek"].items() if v["n_sahis2"] == 0][0]
    jargon, anon = OC.kapi_sozlukleri()
    eski = {s["aile"]: s for s in json.load(open(f"{OUT}/ORNEK_CIFT.meta.json", encoding="utf-8"))["secim"]}
    for aile in OC.AILELER:
        v = A[aile]; k0, k1 = v["kontrast"].split("→"); L = {}
        for bacak, z in (("taban", k0), ("hizali", k1)):
            R = MK.oku(aile, z); L[bacak] = (R, MK.satir_bilesenleri(nlp, R), f"{MK.VERI}/{aile}/{z}/uretim.jsonl")
        s = sec(nlp, MK, aile, L, ilk_kol, jargon, anon); e = eski[aile]
        ayni = (s["istem_i"] == e["istem_i"] and abs(s["dM1_istem"] - e["dM1_istem"]) <= 1e-9 and abs(s["medyan"] - e["medyan"]) <= 1e-9
                and [list(d) for d in s["dusen"]] == [list(d) for d in e["dusen"]] and s["sha_metin"] == e["sha_metin"] and s["sha_uretim"] == e["sha_uretim"])
        print(f"{aile} {'AYNI' if ayni else 'FARKLI'}", flush=True)
        assert ayni, (aile, s["istem_i"], e["istem_i"])
    S = {r["aile"]: r for r in json.load(open(CARD_SS, encoding="utf-8"))["aileler"]}
    secim = []
    for aile in AILELER:
        zt, zh = S[aile]["zemin_taban"], S[aile]["zemin_hizali"]; L = {}
        for bacak, z in (("taban", zt), ("hizali", zh)):
            R = SS.oku(SS.SAB, aile, z); L[bacak] = (R, MK.satir_bilesenleri(nlp, R), f"{SS.SAB}/{aile}/{z}/uretim.jsonl")
        d = MK.olc_toplam(L["hizali"][1])["M1"] - MK.olc_toplam(L["taban"][1])["M1"]
        print(f"  [PAYDA] ornek_template KAPI-2 {aile}: süzgecsiz ΔM1 {d:.4f} · card {S[aile]['sbl']['dM1_suzgecsiz']} ⇒ esik 4 ondalik ⇒ EYLEM: tutmazsa YAZMA", flush=True)
        assert round(d, 4) == S[aile]["sbl"]["dM1_suzgecsiz"], (aile, d)
        k0 = A[aile]["kontrast"].split("→")[0] if aile in A else zt
        Rc = MK.oku(aile, k0)
        kaynak = {(r["kol"], r["cekim"], r["istem_i"]): r["onek"] for r in Rc if r["kol"] == ilk_kol and r["cekim"] == 0}
        s = sec(nlp, MK, aile, L, ilk_kol, jargon, anon, istem_kaynagi=kaynak)
        s["bacak"] = dict(taban=zt, hizali=zh); secim.append(s)
        print(f"  [PAYDA] ornek_template {aile}: istem {s['n_istem']} · medyan ΔM1 {s['medyan']:+.3f} · secilen istem {s['istem_i']} "
              f"(ΔM1_p {s['dM1_istem']:+.3f}) · düsen aday {len(s['dusen'])} {s['dusen']}", flush=True)
    t = [(r"\paragraph*{The same rule through each model's own template.}"),
         (r"\vekalet{The rule above was applied unchanged to the panel that wraps the same prompt strings in each model's own chat "
          r"template (Appendix~\ref{app:D3}), for Gemma-3-27B and T\"ulu-3-8B. Per-prompt $\Delta M_1$ uses every row of both checkpoints, "
          r"unfiltered. Only the aligned continuation is printed; the prompt is the user content inside the template, and the base "
          r"continuation is identified by its hash.}"), ""]
    for s in secim:
        dus = ("no candidate was skipped" if not s["dusen"] else
               "%d earlier candidate%s skipped (%s)" % (len(s["dusen"]), "" if len(s["dusen"]) == 1 else "s",
                                                        "; ".join(sorted({d[1] for d in s["dusen"]}))))
        dus = dus.replace("uygunsuz icerik listesi", "offensive-term list").replace("derleme kapisi sözlügü", "build-check list")
        t += [r"\paragraph*{%s, own template.}" % NM[s["aile"]],
              r"\vekalet{Model median $\Delta M_1$ $%+.2f$ per thousand; prompt %d, $\Delta M_1$ $%+.2f$; %s. "
              r"Generation files: base \texttt{%s}, aligned \texttt{%s}; base continuation \texttt{%s}.}" % (
                  s["medyan"], s["istem_i"], s["dM1_istem"], dus, s["sha_uretim"]["taban"], s["sha_uretim"]["hizali"], s["sha_metin"]["taban"]),
              r"\begin{quote}\small\raggedright",
              r"\textbf{Prompt} (\texttt{%s}): %s\par\smallskip" % (s["sha_metin"]["istem"], OC.tex_kac(s["istem"])),
              r"\textbf{Aligned} (\texttt{%s}): %s" % (s["sha_metin"]["hizali"], OC.tex_kac(s["hizali"])),
              r"\end{quote}", ""]
    io.open(f"{OUT}/ORNEK_TEMPLATE.tex", "w", encoding="utf-8").write("% * URETILDI - elle duzenlenmez (figures/example_template.py)\n" + "\n".join(t))
    json.dump(dict(table="ORNEK_TEMPLATE", damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rule=list(KURALLAR),
                   sources=[dict(yol=os.path.relpath(p, ROOT), sha256_16=OC.sha16f(p)) for p in (OC.CARD, OC.KOLT, CARD_SS)],
                   secim=[{k: v for k, v in s.items() if k not in ("istem", "taban", "hizali")} for s in secim],
                   sure_dk=round((time.time() - t0) / 60, 1)),
              io.open(f"{OUT}/ORNEK_TEMPLATE.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("✓ ORNEK_TEMPLATE.tex + ORNEK_TEMPLATE.meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
