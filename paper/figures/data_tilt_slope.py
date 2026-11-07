#!/usr/bin/env python3
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import io, json, os, sys
sys.path.insert(0, __DNH_ROOT__ + "/paper/figures")
from style import OUT, ROOT, load

CARD = "results/data_tilt_verdict_2026-09-07.json"
DIK = "results/weight_delta_similarity_2026-09-05.json"
DIK_CIFT = "KISI_ASAGI|POZKON"


def main():
    K, pk = load(CARD)
    D, pd = load(DIK)
    A = K["verdict"]["okuma_A"]
    tb, nt, uc = abs(A["KK3_TABAN"]), abs(A["KK3_NOTR"]), abs(A["KK3_UCKAT"])
    geri, kalan, kat = (tb - nt) / tb, nt / tb, uc / tb
    print(f"  [PAYDA] kk3_egim: taban={tb:.4f} · notr={nt:.4f} · uckat={uc:.4f} · "
          f"geri={geri:.4f} · kalan={kalan:.4f} · kat={kat:.4f} ⇒ esik: geri+kalan≠1 ⇒ HATA")
    assert abs(geri + kalan - 1) < 1e-9, "geri + kalan ≠ 1"
    T3, pt3 = load("results/data_tilt_neutral_seed3_verdict_2026-09-15.json")
    assert T3["verdict"]["kk3t3"] == "ISARET-TUTTU" and T3["verdict"]["alet"] == "ESDEGER", T3["verdict"]
    _tk = {"KK3_NOTR": "t1", "KK3_NOTR_T2": "t2", "KK3_NOTR_T3": "t3"}
    _gecen = {_tk[k]: (1 - T3["verdict"]["kalan_pay"][k]) for k in _tk if T3["kol"][k]["hal_A"] == "GECTI"}
    _pay = sorted(round(100 * v) for v in _gecen.values())
    _n_gecen, _n_seed = len(_gecen), len(_tk)
    _say = {1: "one", 2: "two", 3: "three"}
    _yuzde = (f"${_pay[0]}$--${_pay[-1]}\\%$" if _pay[0] != _pay[-1] else f"${_pay[0]}\\%$")
    _kalan = sorted(round(100 * T3["verdict"]["kalan_pay"][k]) for k in _tk if T3["kol"][k]["hal_A"] == "GECTI")
    _kalan_s = (f"${_kalan[0]}$--${_kalan[-1]}\\%$" if _kalan[0] != _kalan[-1] else f"${_kalan[0]}\\%$")
    _seed_s = f"in {_say[_n_gecen]} of {_say[_n_seed]} training seeds"
    print(f"  [PAYDA] kk3_seed_uc: n_seed={_n_seed} · hal_plasebo_gecen={_n_gecen} · pay={_pay} · kalan={_kalan} "
          f"⇒ esik: gecen 0 ise ⇒ EYLEM: yüzde cümlesi YAZILMAZ")
    assert _n_gecen >= 1
    _ceyrek = all(20 <= p <= 30 for p in _pay)
    print(f"{len(_pay)} {_pay} {'SÖZEL' if _ceyrek else 'YÜZDE'}"
          f"")
    _oran_s = "about a quarter" if _ceyrek else f"{_yuzde}"
    _kalan_c = "the rest is unidentified" if _ceyrek else f"{_kalan_s} remains"
    H = K["verdict"]
    _bar_kat = H["fark"] / H["plasebo_p95"]
    _kalan_hepsi = [T3["verdict"]["kalan_pay"][k] for k in _tk]
    _cogu = all(v > 0.5 for v in _kalan_hepsi)
    print(f"  [PAYDA] kk3_kayitli: fark={H['fark']:.4f} · p95={H['plasebo_p95']:.4f} · bar_kat={_bar_kat:.3f} · "
          f"kat={kat:.3f} · kalan={[round(v, 3) for v in _kalan_hepsi]} ⇒ esik: kat>3 «more than triples», "
          f"kalan>0,5 (3/3) «most in place» ⇒ EYLEM: tutmazsa sözel bicim YAZILMAZ")
    assert H["bar"] and H["plasebo"] and kat > 3 and _cogu, (H["bar"], H["plasebo"], kat, _kalan_hepsi)
    s = (f"The registered contrast, flat against tripled, clears its bar ${_bar_kat:.1f}\\times$ on one "
         f"training seed, with no range across seeds: tripling the pronoun gap more than triples the withdrawal "
         f"(${kat:.1f}\\times$). Zeroing it accounts for {_oran_s} of the withdrawal at this data size, "
         f"{_seed_s}; {_kalan_c}.")
    _kaynak_seed = []
    _huk_y, _esl_y = "results/data_tilt_natural_verdict_2026-09-16.json", "results/data_tilt_natural_matched_fark_2026-09-17.json"
    if os.path.exists(f"{ROOT}/{_huk_y}") and os.path.exists(f"{ROOT}/{_esl_y}"):
        TB, ptb = load(_huk_y); ES, pes = load(_esl_y)
        _ad = TB["verdict"]["kk3taban"]
        assert ES["verdict_adi"] == _ad, (ES["verdict_adi"], _ad)
        if _ad in ("ISARET-TUTTU", "ISARET-DÖNDÜ") and TB["verdict"]["alet"] == "ESDEGER" and ES.get("havuz"):
            _a0, _a1 = TB["verdict"]["dA_aralik"]; _h = ES["havuz"]
            _isaret = "" if _ad == "ISARET-TUTTU" else ", not one sign"
            s = (f"The registered contrast, flat against tripled, clears its bar ${_bar_kat:.1f}\\times$ on one "
                 f"training seed: tripling the pronoun gap more than triples the withdrawal "
                 f"(${kat:.1f}\\times$). Zeroing it accounts for {_oran_s} of the withdrawal at this data size, "
                 f"{_seed_s}; {_kalan_c}. At three seeds the run with the natural pronoun gap moves $M_1$ by "
                 f"${_a0:+.2f}$ to ${_a1:+.2f}${_isaret}; pooled, zeroed minus natural is "
                 f"${_h['fark']:+.2f}$ $[{_h['ci'][0]:+.2f},{_h['ci'][1]:+.2f}]$.")
            _kaynak_seed = [ptb, pes]
        print(f"  [PAYDA] kk3_egim_seed: verdict={_ad} · alet={TB['verdict']['alet']} · havuz={'VAR' if ES.get('havuz') else 'YOK'} · "
              f"hal_yazildi={bool(_kaynak_seed)} ⇒ esik: TUTTU/DÖNDÜ ∧ ESDEGER ∧ havuz ⇒ EYLEM: degilse v87b lafzi")
    else:
        print("")
    _uc_y = "results/kk3uckat_t2_verdict_2026-09-18.json"
    _uc_tuttu = False
    if os.path.exists(f"{ROOT}/{_uc_y}"):
        UC, puc = load(_uc_y); _hu = UC["verdict"]
        print(f"{_hu.get('kk3uckat')} {_hu.get('alet')} {_hu.get('karsitlik_aralik')}"
              f"")
        if _hu.get("kk3uckat") == "ISARET-TUTTU" and _hu.get("alet") == "ESDEGER":
            _uc_tuttu = True
            _u0, _u1 = _hu["karsitlik_aralik"]
            _eski = "on one training seed: tripling"
            assert _eski in s, s[:120]
            s = s.replace(_eski, f"on one training seed and keeps its sign at a second (${_u1:+.2f}$ and ${_u0:+.2f}$ per "
                                 f"thousand; Appendix~\\ref{{app:D7}}): tripling", 1)
            _kaynak_seed = _kaynak_seed + [puc]
    if _uc_tuttu and _kaynak_seed and len(_kaynak_seed) >= 3:
        _es = TB["verdict"]["duz_eksi_dogal_esli"]
        _pay3 = sorted(v["fark"] / abs(v["dogal"]) for v in _es.values())
        _oran_s = ("a quarter to a half" if 0.20 <= _pay3[0] <= 0.30 and 0.45 <= _pay3[-1] <= 0.55
                   else f"${100*_pay3[0]:.0f}$--${100*_pay3[-1]:.0f}\\%$")
        _seed_s = f"across {_say[len(_pay3)]} training seeds"
        _cogu3 = all(1 - v > 0.5 for v in _pay3)
        print(f"{len(_pay3)} {[round(v, 4) for v in _pay3]} {_cogu3}"
              f"")
        assert _cogu3, _pay3
        s = (f"Changing the pronoun gap in the training pairs changes the drop. Tripling the gap more than triples "
             f"the drop (${kat:.1f}$ times). The registered contrast, tripled gap against gap removed, clears its bar ${_bar_kat:.1f}$ times over on "
             f"one training seed and keeps its sign on a second (${_u1:+.2f}$ and ${_u0:+.2f}$ per 1,000 tokens). "
             f"Removing the gap accounts for {_oran_s} of the drop, {_seed_s}, so most of the drop remains. Across "
             f"three seeds the run with the natural gap moves the second-person rate by ${_a0:+.2f}$ to "
             f"${_a1:+.2f}${_isaret}, and removing the gap changes that by ${_h['fark']:+.2f}$ "
             f"$[{_h['ci'][0]:+.2f},{_h['ci'][1]:+.2f}]$ when the seeds are pooled.")
    io.open(f"{OUT}/KK3_EGIM.tex", "w", encoding="utf-8").write(s + "\n")
    for kol in ("KK3_NOTR", "KK3_TABAN"):
        tv = sorted(f for f in os.listdir(f"{ROOT}/results")
                    if f.startswith(f"SEED_VARYANS_2026-08-30_{kol}_t"))
        assert tv == [f"SEED_VARYANS_2026-08-30_{kol}_t20260906.json"], (kol, tv)
    T2, pt2 = load("results/data_tilt_neutral_seed2_verdict_2026-09-13.json")
    assert T2["verdict"]["kk3t2"] == "SEED-GÖRÜNMEZ" and T2["verdict"]["aile_bari_hal_A"] == "GECMEZ", T2["verdict"]
    print(f"{T2['verdict']['kk3t2']}"
          f"{T2['verdict']['aile_bari_hal_A']}")
    o = (f"removing that gap leaves most of the drop. Tripling the gap more than triples the drop in one training seed"
         + (" and keeps its sign in a second" if _uc_tuttu else ""))
    print(f"  [PAYDA] kk3_ozet: kat={kat:.3f} · cogu={_cogu} · uckat_t2_tuttu={_uc_tuttu} ⇒ esik kat>3 ∧ cogu ⇒ "
          f"EYLEM: ikinci seed tutmazsa yan cümle düser")
    io.open(f"{OUT}/KK3_OZET.tex", "w", encoding="utf-8").write(o + "\n")
    c = D["kosinus"][DIK_CIFT]["TOPLAM"]
    kol = list(D["_kunye"]["kollar"])
    print(f"  [PAYDA] kk3_dik: cift={DIK_CIFT} · cos={c:.6f} · n_kol={len(kol)} "
          f"⇒ esik: cift kartta yoksa ⇒ EYLEM: cümle YAZILMAZ")
    d = (f"their cosine is ${c:+.4f}$, orthogonal to within the residue eight "
         f"billion coordinates leave")
    io.open(f"{OUT}/KK3_DIK.tex", "w", encoding="utf-8").write(d + "\n")
    json.dump(dict(table="KK3_EGIM",
                   ciktilar=["KK3_EGIM.tex", "KK3_DIK.tex", "KK3_OZET.tex"],
                   sources=[pk, pd, pt2, pt3] + _kaynak_seed, diklik=dict(cift=DIK_CIFT, cos=c),
                   payda=dict(geri=round(geri, 4), kalan=round(kalan, 4), kat=round(kat, 4)),
                   note="renders the sealed KK-3 cascade reading; no statistic computed here"),
              io.open(f"{OUT}/KK3_EGIM.meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  ✓ {OUT}/KK3_EGIM.tex · «{s}»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
