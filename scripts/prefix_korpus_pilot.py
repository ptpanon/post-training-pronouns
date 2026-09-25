#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, csv, argparse, collections, hashlib, re
import numpy as np

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda

OUT = __DNH_DATA__ + "/onek_korpus"
KIALO = __DNH_DATA__ + "/unreleased/dataframe.csv"
SEED, N_CEKIM, MAKS_JETON, SICAKLIK, TOP_P = 20260805, 2, 160, 0.9, 0.95
os.makedirs(OUT, exist_ok=True)

MODELLER = {
    "qwen05_base": __DNH_DATA__ + "/.hf_local/hub/models--Qwen--Qwen2.5-0.5B",
    "phi2_base":   f"{ROOT}/.hf/hub/models--microsoft--phi-2",
}

CERCEVE = {
    "sakin": "Considering the arguments carefully, I think {T} is {P}, because",
    "sert":  "Honestly, anyone can see {T} is completely {P}. Here is why:",
}
DURUS = {"arti": "right", "eksi": "wrong"}
DURUS_SOZLUGU = {
    "arti": {"right", "correct", "true", "valid", "sensible", "reasonable"},
    "eksi": {"wrong", "incorrect", "false", "invalid", "nonsense", "unreasonable"},
}
ISI_SOZ = {"obviously", "clearly", "absurd", "ridiculous", "stupid", "idiotic", "nonsense",
           "insane", "ludicrous", "outrageous", "pathetic", "garbage"}
ARGUMAN_SOZ = {"because", "therefore", "however", "although", "evidence", "reason", "argue",
               "suppose", "instead", "moreover", "furthermore", "thus", "hence", "since"}
ISLEV_SOZ = {"the", "of", "and", "to", "a", "in", "is", "it", "that", "for", "as", "with",
             "be", "on", "not", "this", "but", "by", "are", "or", "an", "have", "from"}
KELIME = re.compile(r"[a-z']+")


def _jet(t):
    return KELIME.findall(str(t).lower())


def tekrar_orani(t):
    j = _jet(t)
    if len(j) < 6:
        return 0.0
    g = [tuple(j[i:i + 3]) for i in range(len(j) - 2)]
    return 1.0 - len(set(g)) / len(g)


def tezler(n_debate=None):
    per, red = collections.defaultdict(list), collections.Counter()
    with open(KIALO, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            c = " ".join(str(r["Claim"]).split())
            if not (40 <= len(c) <= 180):
                red["uzunluk"] += 1; continue
            try:
                d = int(r["Depth"])
            except Exception:
                red["derinlik_yok"] += 1; continue
            per[r["Debate_name"]].append((d, c))
    out = []
    for deb in sorted(per):
        en_sig = sorted(per[deb], key=lambda x: (x[0], x[1]))[:1]
        for d, c in en_sig:
            out.append(dict(debate=deb, derinlik=d, tez=c))
    payda("pilot_tez", n_debate=len(per), n_tez=len(out),
          red_uzunluk=red["uzunluk"], red_derinlik_yok=red["derinlik_yok"])
    if n_debate:
        out = out[:n_debate]
    return out


def onek(tez, durus, isi):
    return CERCEVE[isi].format(T=tez, P=DURUS[durus])


def kapi_cihaz(gerek_gb):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("")
    ser, top = torch.cuda.mem_get_info(0)
    bos = ser / 2**30
    print(f"[ÖN-UCUS] GPU0 bos={bos:.1f} GB · gereken≈{gerek_gb:.1f} GB · "
          f"HF_HOME={os.environ.get('HF_HOME', '(yok)')}", flush=True)
    if bos < gerek_gb:
        raise RuntimeError(f"{bos:.1f} {gerek_gb:.1f}"
                           f"")
    return bos


def uret(ad, tez_list, dev="cuda:0"):
    import torch, glob
    from transformers import AutoTokenizer, AutoModelForCausalLM
    snap = sorted(glob.glob(f"{MODELLER[ad]}/snapshots/*"))[-1]
    tok = AutoTokenizer.from_pretrained(snap)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float16).to(dev).eval()
    torch.manual_seed(SEED)
    sat = []
    isler = [(t, d, i) for t in tez_list for d in DURUS for i in CERCEVE]
    for b in range(0, len(isler), 8):
        parti = isler[b:b + 8]
        ist = [onek(t["tez"], d, i) for t, d, i in parti]
        enc = tok(ist, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(dev)
        with torch.no_grad():
            g = m.generate(**enc, do_sample=True, temperature=SICAKLIK, top_p=TOP_P,
                           max_new_tokens=MAKS_JETON, num_return_sequences=N_CEKIM,
                           pad_token_id=tok.pad_token_id)
        yeni = g[:, enc["input_ids"].shape[1]:]
        met = tok.batch_decode(yeni, skip_special_tokens=True)
        for j, (t, d, i) in enumerate(parti):
            for c in range(N_CEKIM):
                k = j * N_CEKIM + c
                kes = int((yeni[k] != tok.pad_token_id).sum()) >= MAKS_JETON
                sat.append(dict(uretici=ad, debate=t["debate"], tez=t["tez"], durus=d, isi=i,
                                cekim=c, kesik=int(kes), onek=ist[j], metin=met[k]))
        print(f"  {ad}: {min(b+8, len(isler))}/{len(isler)} istem", flush=True)
    del m
    torch.cuda.empty_cache()
    return sat


def olc(s):
    t = str(s["metin"])
    w = _jet(t)
    ow = _jet(s["onek"])
    tw = set(_jet(s["tez"])) - ISLEV_SOZ
    og = {" ".join(ow[i:i + 5]) for i in range(max(0, len(ow) - 4))}
    gr = [" ".join(w[i:i + 5]) for i in range(max(0, len(w) - 4))]
    kendi = DURUS_SOZLUGU[s["durus"]]
    karsi = DURUS_SOZLUGU["eksi" if s["durus"] == "arti" else "arti"]
    ws = set(w)
    return dict(
        n_kelime=len(w),
        gecerli=int(len(t.strip().split()) >= 5),
        kopya5=round(sum(1 for x in gr if x in og) / len(gr), 5) if gr else 0.0,
        yanki_kendi=int(bool(ws & kendi)),
        yanki_karsi=int(bool(ws & karsi)),
        konu_ortusme=round(len(ws & tw) / max(1, len(tw)), 5),
        arguman=int(bool(ws & ARGUMAN_SOZ)),
        isi_n=sum(1 for x in w if x in ISI_SOZ),
        islev_pay=round(sum(1 for x in w if x in ISLEV_SOZ) / max(1, len(w)), 5),
        yineleme=round(collections.Counter(gr).most_common(1)[0][1] / len(gr), 5) if gr else 1.0,
    )


def ozetle(sat):
    for s in sat:
        s.update(olc(s))
    tab = {}
    for ad in sorted({s["uretici"] for s in sat}):
        for d in ("arti", "eksi"):
            for i in ("sakin", "sert"):
                g = [s for s in sat if s["uretici"] == ad and s["durus"] == d and s["isi"] == i]
                if not g:
                    continue
                A = lambda k: float(np.mean([s[k] for s in g]))
                tab[f"{ad}/{d}/{i}"] = dict(
                    n=len(g), gecerli=round(A("gecerli"), 4),
                    kelime_medyan=int(np.median([s["n_kelime"] for s in g])),
                    kopya5_p95=round(float(np.percentile([s["kopya5"] for s in g], 95)), 4),
                    yanki_kendi=round(A("yanki_kendi"), 4), yanki_karsi=round(A("yanki_karsi"), 4),
                    konu_ortusme=round(A("konu_ortusme"), 4), arguman=round(A("arguman"), 4),
                    isi=round(A("isi_n"), 4), islev_pay=round(A("islev_pay"), 4),
                    yineleme_medyan=round(float(np.median([s["yineleme"] for s in g])), 4))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tez", type=int, default=17)
    ap.add_argument("--modeller", default="qwen05_base,phi2_base")
    a = ap.parse_args()

    T = tezler(a.n_tez)
    json.dump(T, open(f"{OUT}/pilot_tezler.json", "w"), ensure_ascii=False, indent=1)
    onekler = [dict(durus=d, isi=i, template=CERCEVE[i].format(T="{T}", P=DURUS[d]))
               for d in DURUS for i in CERCEVE]
    json.dump(onekler, open(f"{OUT}/pilot_onekler.json", "w"), ensure_ascii=False, indent=1)
    print(f"[GIRDI] {len(T)} tez · {len(onekler)} önek sablonu → diske yazildi", flush=True)

    kapi_cihaz(7.0)
    sat = []
    for ad in a.modeller.split(","):
        sat += uret(ad, T)
    tab = ozetle(sat)

    with open(f"{OUT}/pilot_satirlar.jsonl", "w", encoding="utf-8") as fh:
        for s in sat:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    json.dump(tab, open(f"{OUT}/pilot_ozet.json", "w"), ensure_ascii=False, indent=1)

    print()
    print(f"{'üretici/durus/isi':<28}{'n':>4}{'gecer':>7}{'kel~':>6}{'kop95':>7}"
          f"{'yanki+':>8}{'yanki−':>8}{'konu':>7}{'arg':>6}{'isi':>6}{'yin~':>7}")
    print("─" * 94)
    for k in sorted(tab):
        v = tab[k]
        print(f"{k:<28}{v['n']:>4}{v['gecerli']:>7.2f}{v['kelime_medyan']:>6}"
              f"{v['kopya5_p95']:>7.3f}{v['yanki_kendi']:>8.2f}{v['yanki_karsi']:>8.2f}"
              f"{v['konu_ortusme']:>7.3f}{v['arguman']:>6.2f}{v['isi']:>6.2f}"
              f"{v['yineleme_medyan']:>7.3f}")
    print("─" * 94)
    payda("pilot_onek", n_satir=len(sat), n_tez=len(T), n_hucre=len(tab),
          n_uretici=len(a.modeller.split(",")))
    n_gecersiz = sum(1 for s in sat if not s["gecerli"])
    print(f"{n_gecersiz} {len(sat)}"
          f"{n_gecersiz/max(1,len(sat)):.2%} {len(tab)}")
    print(f"→ {OUT}/pilot_satirlar.jsonl · pilot_ozet.json · pilot_tezler.json")


if __name__ == "__main__":
    main()
