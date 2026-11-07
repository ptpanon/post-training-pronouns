#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys, json, subprocess, argparse

ROOT = __DNH_ROOT__ + ""
sys.path.insert(0, f"{ROOT}/scripts"); sys.path.insert(0, f"{ROOT}/scripts")
from reading_style_object import payda


def smi_kartlari():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True)
    out = {}
    for l in r.stdout.strip().splitlines():
        i, u, kul, top = [x.strip() for x in l.split(",")]
        out[u.replace("GPU-", "")] = dict(index=int(i), kullanilan=kul, toplam=top)
    return out


def fiziksel_bekle(dev="cuda:0"):
    i = int(dev.split(":")[1]) if ":" in str(dev) else 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return i
    p = [x.strip() for x in cvd.split(",") if x.strip()]
    return int(p[i]) if i < len(p) and p[i].lstrip("-").isdigit() else i


def kilitle(dev="cuda:0", beklenen_fiziksel=None, tam=False, etiket="gpu",
            gereken_vram_gb=25):
    import torch
    cvd_once = os.environ.get("CUDA_VISIBLE_DEVICES")
    i = int(dev.split(":")[1]) if ":" in dev else 0
    torch.zeros(1, device=f"cuda:{i}")
    p = torch.cuda.get_device_properties(i)
    uuid = str(getattr(p, "uuid", "(yok)"))
    CARD = smi_kartlari()
    kayit = CARD.get(uuid)
    C = dict(dev=dev, torch_index=i, gpu_adi=p.name, gpu_uuid=uuid,
             fiziksel_gpu=(kayit or {}).get("index"), cvd_once=cvd_once,
             cvd_kilitten_sonra=os.environ.get("CUDA_VISIBLE_DEVICES"),
             smi_card_sayisi=len(CARD), smi_kullanilan=(kayit or {}).get("kullanilan"))
    if kayit is None:
        raise RuntimeError(f"{uuid}"
                           f"")
    if beklenen_fiziksel is not None and C["fiziksel_gpu"] != beklenen_fiziksel:
        raise RuntimeError(f"{beklenen_fiziksel}"
                           f"{C['fiziksel_gpu']} {uuid[:8]}"
                           f"")
    if tam:
        import prefix_layer_taramasi as KT
        C["cevre"] = KT.on_ucus_cevresi(f"cuda:{i}", gereken_vram_gb=gereken_vram_gb)
    print(f"  [GPU KILIDI] {dev} → **FIZIKSEL GPU{C['fiziksel_gpu']}** · uuid {uuid[:8]}… · "
          f"{p.name} · smi kullanilan {C['smi_kullanilan']} · CVD {cvd_once}→"
          f"{C['cvd_kilitten_sonra']} · kilit ✓")
    payda(f"gpu_kilidi_{etiket}", n_card=len(CARD), n_kilit=1,
          bekle={"n_card": 1, "n_kilit": 1})
    return C


def _test():
    savlar = {}
    CARD = smi_kartlari()
    savlar["A_smi_okunuyor"] = dict(gecti=bool(CARD), n_card=len(CARD))

    def alt(kod):
        env = dict(os.environ); env.pop("CUDA_VISIBLE_DEVICES", None)
        r = subprocess.run([f"{ROOT}/.venv/bin/python", "-c", kod], capture_output=True,
                           text=True, env=env, cwd=ROOT)
        s = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
        return json.loads(s[-1]) if s else dict(hata=(r.stderr.strip().splitlines() or ["?"])[-1])

    ana = (f"import os,sys,json; sys.path.insert(0,{ROOT!r}+'/scripts')\n"
           "os.environ['CUDA_VISIBLE_DEVICES']='1'\n"
           "from gpu_lock import kilitle\n")
    b = alt(ana + "C=kilitle('cuda:0')\n"
                  "os.environ['CUDA_VISIBLE_DEVICES']='0'\n"
                  "import torch\n"
                  "u2=str(torch.cuda.get_device_properties(0).uuid)\n"
                  "print(json.dumps({'ilk':C['gpu_uuid'],'sonra':u2,"
                  "'fiziksel':C['fiziksel_gpu']}))\n")
    savlar["B_kilit_tutuyor"] = dict(gecti=bool(b.get("ilk") and b.get("ilk") == b.get("sonra")),
                                     **b)
    c = alt(ana + "try:\n kilitle('cuda:0', beklenen_fiziksel=99)\n print(json.dumps({'hata':None}))\n"
                  "except RuntimeError as e:\n import json; print(json.dumps({'hata':str(e)[:60]}))\n")
    savlar["C_kapi_atesliyor"] = dict(gecti=bool(c.get("hata")), mesaj=c.get("hata"))
    d = alt(ana + "C=kilitle('cuda:0')\n"
                  f"sys.path.insert(0,{ROOT!r}+'/scripts')\n"
                  "from prefix_muhurlu_kosu import burrows_z\n"
                  "_=burrows_z(['the cat sat on the mat'])\n"
                  "import torch\n"
                  "print(json.dumps({'fiziksel':C['fiziksel_gpu'],"
                  "'sonra_uuid':str(torch.cuda.get_device_properties(0).uuid),"
                  "'cvd':os.environ.get('CUDA_VISIBLE_DEVICES')}))\n")
    savlar["D_zincire_karsi"] = dict(gecti=bool(d.get("fiziksel") == 1), **d)
    print("★ GPU KILIDI TESTI")
    for k, v in savlar.items():
        print(f"  {k:20s} {'GECTI ✓' if v.get('gecti') else 'DÜSTÜ ✗'}  "
              f"{ {a: b for a, b in v.items() if a != 'gecti'} }")
    hepsi = all(v.get("gecti") for v in savlar.values())
    payda("gpu_kilidi_testi", n_sav=len(savlar),
          red_dusen=sum(1 for v in savlar.values() if not v.get("gecti")),
          bekle={"n_sav": 4})
    json.dump(dict(savlar=savlar, hepsi=bool(hepsi)),
              open(f"{ROOT}/unreleased/GPU_KILIDI_TESTI_2026-08-07.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"  ⇒ {'DÖRT SAV DA GECTI ✓' if hepsi else 'TEST DÜSTÜ ✗'}")
    return hepsi


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true")
    ap.add_argument("--dev", default="cuda:0"); ap.add_argument("--bekle", type=int, default=None)
    a = ap.parse_args()
    if a.test:
        sys.exit(0 if _test() else 1)
    print(json.dumps(kilitle(a.dev, a.bekle), ensure_ascii=False, indent=1))
