#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, glob, time, argparse, platform, subprocess, types, hashlib, inspect
sys.path.insert(0, __DNH_ROOT__ + "/scripts")

try:
    import sklearn
    _SAPTIRMA = []
except ModuleNotFoundError:
    _SAPTIRMA = ["esdegerlik", "okuma_tarzi"]
    for _m in _SAPTIRMA:
        _s = types.ModuleType(_m)
        _s.TARZLAR, _s.tarz_hesapla, _s.ozellik = {}, None, None
        sys.modules[_m] = _s
from reading_style_object import payda
import weight_dosyalari as AGD
from prefix_korpus_pilot import CERCEVE, DURUS, onek, OUT as OUT_KOK

PAYDA_HASH = hashlib.sha1(inspect.getsource(payda).encode()).hexdigest()[:12]

OUT = f"{OUT_KOK}/uretim_pilot"
SEED, SICAKLIK, TOP_P = 20260805, 0.9, 0.95
N_CEKIM, B_ANA, B_KAPI = 12, 640, 160
PLASEBO_AILESI = {
    "P1": {"arti": "right", "eksi": "correct"},
    "P2": {"arti": "blicket", "eksi": "dax"},
    "P3": {"arti": "recent", "eksi": "ancient"},
}
DURUS_PLASEBO = PLASEBO_AILESI["P3"]
BUTCELER = (160, 320, 640)


def cevre(model_yolu, ad):
    import torch, vllm
    bos, top = torch.cuda.mem_get_info(0)
    c = dict(
        motor="vllm", motor_surum=vllm.__version__, torch=torch.__version__,
        imaj=os.environ.get("ONEK_IMAJ", "(bilinmiyor)"),
        gpu_adi=torch.cuda.get_device_name(0),
        vram_bos_gb=round(bos / 2**30, 2), vram_top_gb=round(top / 2**30, 2),
        hf_home=os.environ.get("HF_HOME", "(yok)"),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "(yok)")),
        gpu_host_index=os.environ.get("ONEK_GPU", "(yok)"),
        python=platform.python_version(), model_yolu=model_yolu, yazar=ad,
        payda_muhafiz_hash=PAYDA_HASH, saptirilan_moduller=_SAPTIRMA or "(yok — sklearn var)",
        cikis_koku=OUT, seed=SEED, sicaklik=SICAKLIK, top_p=TOP_P,
        n_cekim=N_CEKIM, b_ana=B_ANA, b_kapi=B_KAPI, butceler=list(BUTCELER),
        damga_utc=subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                 capture_output=True, text=True).stdout.strip(),
    )
    snaplar = sorted(glob.glob(f"{model_yolu}/snapshots/*"))
    agirlik = AGD.dosyalar(snaplar[-1]) if snaplar else []
    c["snapshot"] = os.path.basename(snaplar[-1]) if snaplar else "(yok)"
    c["n_agirlik_dosyasi"] = len(agirlik)
    c["model_gercek_yol"] = os.path.realpath(model_yolu)
    if not agirlik:
        raise RuntimeError(
            f"{model_yolu}"
            f"{os.path.realpath(model_yolu)}"
            f"")

    eksik = [k for k, v in c.items() if v in ("(yok)", "(bilinmiyor)", None)]
    print(f"[CEVRE] {json.dumps(c, ensure_ascii=False)}", flush=True)
    if eksik:
        raise RuntimeError(f"{eksik}")
    return c


def istemler():
    T = json.load(open(f"{OUT_KOK}/pilot_tezler.json", encoding="utf-8"))
    isler = [(t, d, i) for t in T for d in DURUS for i in CERCEVE]
    payda("uretim_istem", n_tez=len(T), n_is=len(isler), n_durus=len(DURUS),
          n_cerceve=len(CERCEVE), bekle={"n_tez": 17, "n_is": 68})
    return T, isler


def kos(llm, tok, isler, T, ad, b_maks, etiket, n_cekim=None, plasebo=False, pkod=None):
    from vllm import SamplingParams
    nc = n_cekim or N_CEKIM
    if plasebo:
        from prefix_korpus_pilot import CERCEVE as _C
        _P = PLASEBO_AILESI[pkod] if pkod else DURUS_PLASEBO
        _ok = lambda t, d, i: _C[i].format(T=t, P=_P[d])
    else:
        _ok = lambda t, d, i: onek(t, d, i)
    ist = [_ok(t["tez"], d, i) for t, d, i in isler]
    sp = SamplingParams(temperature=SICAKLIK, top_p=TOP_P, max_tokens=b_maks,
                        n=nc, seed=SEED)
    t0 = time.time()
    cikti = llm.generate(ist, sp)
    dt = time.time() - t0

    sat, n_jeton = [], 0
    for (t, d, i), o in zip(isler, cikti):
        for c, s in enumerate(o.outputs):
            tid = list(s.token_ids)
            n_jeton += len(tid)
            r = dict(uretici=ad, debate=t["debate"], tez=t["tez"], durus=d, isi=i,
                     cekim=c, n_token=len(tid), bitis=s.finish_reason,
                     kol=(pkod or "plasebo") if plasebo else "gercek",
                     onek=_ok(t["tez"], d, i))
            for B in BUTCELER:
                if B <= b_maks:
                    r[f"metin_{B}"] = tok.decode(tid[:B], skip_special_tokens=True)
            sat.append(r)
    yol = f"{OUT}/{etiket}__{ad}.jsonl"
    tmp = yol + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in sat:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, yol)
    hiz = n_jeton / dt
    payda(f"uretim_{etiket}_{ad}", n_satir=len(sat), n_is=len(isler), n_tez=len(T),
          bekle={"n_satir": len(isler) * nc, "n_is": 68})
    print(f"  [HIZ] {ad}/{etiket}: {n_jeton} jeton / {dt:.1f}s = {hiz:.0f} jeton/s "
          f"· ortalama {n_jeton/len(sat):.0f} jeton/satir", flush=True)
    return dict(etiket=etiket, n_satir=len(sat), n_jeton=n_jeton, saniye=round(dt, 1),
                jeton_sn=round(hiz, 1), yol=yol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ad", required=True)
    ap.add_argument("--n-cekim", type=int, default=N_CEKIM)
    ap.add_argument("--out-alt", default="uretim_pilot")
    ap.add_argument("--plasebo", action="store_true")
    ap.add_argument("--plasebo-aile", default="", help="virgullu kod listesi, or. P1,P2,P3")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    globals()["OUT"] = f"{OUT_KOK}/{a.out_alt}"
    globals()["SEED"] = a.seed
    os.makedirs(OUT, exist_ok=True)
    c = cevre(a.model, a.ad)
    from vllm import LLM
    snap = sorted(glob.glob(f"{a.model}/snapshots/*"))[-1]
    T, isler = istemler()
    llm = LLM(model=snap, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=2048, disable_log_stats=True, seed=SEED)
    tok = llm.get_tokenizer()
    c["snapshot"] = os.path.basename(snap)
    c["n_cekim"] = a.n_cekim
    c["kosular"] = [kos(llm, tok, isler, T, a.ad, B_ANA, "ana", a.n_cekim),
                    kos(llm, tok, isler, T, a.ad, B_KAPI, "kapi160", a.n_cekim)]
    if a.plasebo:
        c["kosular"].append(kos(llm, tok, isler, T, a.ad, B_ANA, "plasebo",
                                a.n_cekim, plasebo=True))
    for pk in [x for x in a.plasebo_aile.split(",") if x]:
        c["kosular"].append(kos(llm, tok, isler, T, a.ad, B_ANA, pk,
                                a.n_cekim, plasebo=True, pkod=pk))
    c["plasebo_ailesi"] = {k: v for k, v in PLASEBO_AILESI.items()
                           if k in a.plasebo_aile.split(",")}
    json.dump(c, open(f"{OUT}/manifest__{a.ad}.json", "w"), ensure_ascii=False, indent=1)
    print(f"→ {OUT}/manifest__{a.ad}.json")


if __name__ == "__main__":
    main()
