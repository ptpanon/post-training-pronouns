import numpy as np


class KapsamHatasi(AssertionError):
    pass


def _l2(A):
    return A / np.clip(np.linalg.norm(A, axis=-1, keepdims=True), 1e-9, None)


def tara(E, metinler=None, n=200, cos_esik=0.9999, bas=True, ad=""):
    if E.ndim != 3:
        raise KapsamHatasi(f"{E.shape}")
    m = int(min(n, len(E)))
    if m < 2:
        raise KapsamHatasi(f"{m}")
    L = int(E.shape[1])
    if metinler is None:
        taban, taban_kaynak = 0, "VERILMEDI (0 varsayildi — dogrulanmamis)"
    else:
        taban = int(sum(1 for t in metinler[1:m] if t == metinler[0]))
        taban_kaynak = "metin-özdesligi"
    sat = []
    for l in range(L):
        a = np.asarray(E[:m, l, :], np.float32)
        ozdes = int((a[1:] == a[0]).all(1).sum())
        nn = _l2(a)
        cos = float((nn @ nn.T)[np.triu_indices(m, 1)].mean())
        sat.append(dict(katman=l, ozdes=ozdes, cos_ort=round(cos, 6),
                        bayrak=bool(ozdes > taban or cos > cos_esik)))
    bayrakli = [s["katman"] for s in sat if s["bayrak"]]
    R = dict(taban=taban, taban_kaynak=taban_kaynak, payda_satir=m, L=L,
             cos_esik=cos_esik, bayrakli_katmanlar=bayrakli, katmanlar=sat)
    if bas:
        print(f"  [sabit-konum · {ad}] payda: {m} satir × {L} katman · TABAN {taban}/{m-1} "
              f"({taban_kaynak}) · bayrakli katman: {bayrakli or 'YOK'}")
        for s in sat:
            if s["bayrak"]:
                print(f"{s['katman']:02d} {s['ozdes']} {m-1}"
                      f"{taban} {s['cos_ort']:.6f}")
    return R
