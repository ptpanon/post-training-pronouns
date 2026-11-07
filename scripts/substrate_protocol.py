#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import glob, json, os, re, sys

KOK = __DNH_ROOT__ + ""
NOTR = __DNH_DATA__ + "/c1_panel"
ELIC_KUNYE = f"{KOK}/unreleased/elicit"
sys.path.insert(0, f"{KOK}/scripts")
import elicited16_loader as E16
IZ = re.compile(r"<\|im_start\|>|<\|start_header_id\|>|\[INST\]|<start_of_turn>|"
                r"<\|assistant\|>|<\|user\|>|<\|endoftext\|>")


def _prova():
    return {"i_template_izi_yakalanir": bool(IZ.search("<|im_start|>user\nx")),
            "ii_ciplak_iskele_temiz": not IZ.search("User: x\nAssistant:"),
            "iii_bos_metin_temiz": not IZ.search("")}


def notr_zemin():
    out, n_dizin, n_kunye, n_onek = [], 0, 0, 0
    for ky in sorted(glob.glob(f"{NOTR}/*/*/uretim_kunye.json")):
        n_dizin += 1
        aile, zemin = ky.split("/")[-3], ky.split("/")[-2]
        try:
            k = json.load(open(ky, encoding="utf-8")); n_kunye += 1
        except Exception:
            k = {}
        uy = os.path.join(os.path.dirname(ky), "uretim.jsonl")
        onek, izli = None, None
        if os.path.exists(uy):
            with open(uy, encoding="utf-8") as fh:
                ilk = fh.readline()
            if ilk:
                onek = (json.loads(ilk).get("onek") or "")[:400]
                izli = bool(IZ.search(onek)); n_onek += 1
        sab = (k.get("kimlik") or {}).get("imza", {}).get("template")
        out.append(dict(aile=aile, zemin=zemin, kunye_template=sab,
                        onek_template_izi=izli))
    return out, dict(n_dizin=n_dizin, n_kunye=n_kunye, n_onek_okundu=n_onek)


def elicit_zemin():
    out = []
    for ky in sorted(glob.glob(f"{ELIC_KUNYE}/KUNYE_m9_*.json")):
        b = os.path.basename(ky)[len("KUNYE_m9_"):-len(".json")]
        aile, _, bacak = b.rpartition("_")
        k = json.load(open(ky, encoding="utf-8"))
        out.append(dict(aile=aile, bacak=bacak, sablonlu=k.get("sablonlu"),
                        iskele=k.get("iskele"), hf_ad=k.get("hf_ad")))
    return out


def main():
    P = _prova(); print("★ §7.1 PROVA:", json.dumps(P, ensure_ascii=False))
    if not all(P.values()):
        print("★ PROVA DÜSTÜ ⇒ ölcüm YAZILMADI"); return 4

    N, npay = notr_zemin()
    izli = [r for r in N if r["onek_template_izi"]]
    yetenek = [r for r in N if r["kunye_template"] is True]
    print(f"  [PAYDA] nötr: dizin={npay['n_dizin']} · künye={npay['n_kunye']} · "
          f"önek okundu={npay['n_onek_okundu']} · künyede template=True {len(yetenek)} "
          f"⇒ ÖNEKTE template izi **{len(izli)}** ⇒ esik >0 ⇒ EYLEM: W-965 geri alinir")
    if npay["n_onek_okundu"] != npay["n_dizin"]:
        print("  ★ PAYDA EKSIK ⇒ HATA (§8): her dizinin öneki okunmaliydi"); return 3

    E = elicit_zemin()
    B, _pB = E16.yukle()
    olculen = sorted(k for k, v in B.items()
                     if not k.startswith("_") and isinstance(v, dict) and "fark" in v)
    ec = {}
    for r in E:
        ec.setdefault(r["aile"], {})[r["bacak"]] = r
    ayrik_prot, ayni_prot, eksik = [], [], []
    for a in olculen:
        d = ec.get(a)
        if not d or "base" not in d or "instruct" not in d:
            eksik.append(a); continue
        (ayni_prot if d["base"]["sablonlu"] == d["instruct"]["sablonlu"]
         else ayrik_prot).append(a)
    print(f"{len(E)} {len(olculen)}"
          f"{len(eksik)} {len(ayrik_prot)}"
          f"{len(ayni_prot)}")
    if eksik:
        print(f"  ★ KÜNYESI EKSIK: {eksik} ⇒ HATA (§8)"); return 3

    cikti = dict(
        _kunye=dict(damga_utc=__import__("subprocess").run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True,
            text=True).stdout.strip(),
            alet="scripts/substrate_protocol.py", borc="W-965",
            kaynak=dict(notr=NOTR, elicit=ELIC_KUNYE, tablo_karti=_pB["yol"],
                        tablo_karti_sha16=_pB["sha256_16"]),
            prova=P, payda_notr=npay,
            payda_elicit=dict(n_kunye=len(E), n_olculen=len(olculen),
                              n_ayrik_protokol=len(ayrik_prot),
                              n_ayni_protokol=len(ayni_prot))),
        NOTR_PROTOKOL=dict(
            hal="TEK PROTOKOL — ciplak iskele, her bacakta",
            n_okunan=npay["n_onek_okundu"],
            onekte_template_izi=len(izli), kunyede_template_yetenek=len(yetenek),
            not_=""),
        ELICIT_PROTOKOL=dict(
            hal="IKI PROTOKOL — sablonu olan bacakta template, olmayanda ciplak iskele",
            iskele="User: {q}\nAssistant:",
            farkli_protokol=ayrik_prot, ayni_protokol=ayni_prot),
        AILELER=N, ELICIT=E)
    yol = f"{KOK}/results/substrate_protocol_2026-09-06.json"
    json.dump(cikti, open(yol, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"★ YAZILDI: {yol}")
    print(f"★ SERH GEREKLI MI: {'EVET' if ayrik_prot else 'HAYIR'} "
          f"({len(ayrik_prot)}/{len(olculen)} ailede bacaklar farkli formatta)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
