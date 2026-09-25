#!/usr/bin/env python3
import ast

__all__ = ["adlari_topla", "capraz_say"]


def _dizgeler(dugum):
    if dugum is None:
        return set()
    if isinstance(dugum, ast.Constant):
        return {dugum.value} if isinstance(dugum.value, str) else set()
    if isinstance(dugum, ast.IfExp):
        return _dizgeler(dugum.body) | _dizgeler(dugum.orelse)
    if isinstance(dugum, ast.BoolOp):
        out = set()
        for v in dugum.values:
            out |= _dizgeler(v)
        return out
    return set()


def adlari_topla(kaynak_yolu, fonksiyon):
    agac = ast.parse(open(kaynak_yolu, encoding="utf-8").read())
    fn = next((n for n in ast.walk(agac)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == fonksiyon), None)
    if fn is None:
        raise SystemExit(f"{fonksiyon} {kaynak_yolu}")
    adlar, kor = set(), 0
    for n in ast.walk(fn):
        if isinstance(n, ast.Return):
            g = _dizgeler(n.value)
            if g:
                adlar |= g
            elif n.value is not None:
                kor += 1
    return adlar, kor


def capraz_say(kaynak_yolu, fonksiyon, prereg_adlari, bilinen=()):
    kodda, kor = adlari_topla(kaynak_yolu, fonksiyon)
    kodda |= set(bilinen)
    prereg = set(prereg_adlari)
    fark = sorted(kodda ^ prereg)
    print(f"  [D44] kodda {len(kodda)} · mühürde {len(prereg)} · fark {fark or '—'} · "
          f"görülemeyen dal {kor}" + (f" · ELLE BEYAN {len(bilinen)} (denetlenmedi)"
                                      if bilinen else ""))
    if kor and not bilinen:
        raise SystemExit(f"{kor}"
                         f"")
    if kodda != prereg:
        raise SystemExit(f"★ D44 DÜSTÜ: kodun ad kümesi mühürle esit degil — fark {fark}")
    return kodda
