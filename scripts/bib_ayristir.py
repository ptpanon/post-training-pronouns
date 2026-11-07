#!/usr/bin/env python3
from __future__ import annotations
import re


def _deger_sonu(t, i):
    if t[i] == "{":
        d, j = 1, i + 1
        while j < len(t) and d:
            d += {"{": 1, "}": -1}.get(t[j], 0) if t[j - 1] != "\\" else 0
            j += 1
        return t[i + 1:j - 1], j
    if t[i] == '"':
        d, j = 0, i + 1
        while j < len(t) and not (t[j] == '"' and d == 0):
            d += {"{": 1, "}": -1}.get(t[j], 0)
            j += 1
        return t[i + 1:j], j + 1
    m = re.compile(r"[^,\s}]+").match(t, i)
    return m.group(0), m.end()


def oku(yol):
    t = open(yol, encoding="utf-8").read(); G = {}
    for m in re.finditer(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", t, re.M):
        tip = m.group(1).lower()
        if tip in ("comment", "string", "preamble"):
            continue
        i = m.end(); alan = {}
        while True:
            while i < len(t) and t[i] in " \t\r\n,":
                i += 1
            if i >= len(t) or t[i] == "}":
                break
            a = re.compile(r"([A-Za-z_\-]+)\s*=\s*").match(t, i)
            if not a:
                raise ValueError(f"{yol}: {m.group(2)} alan okunamadi @ {t[i:i+40]!r}")
            v, i = _deger_sonu(t, a.end())
            alan[a.group(1).lower()] = " ".join(v.split())
        G[m.group(2)] = dict(tip=tip, alan=alan, ham=t[m.start():i + 1], dosya=yol, satir=t.count("\n", 0, m.start()) + 1, bas=m.start(), son=i + 1)
    return G


def yazarlar(alan_author):
    if not alan_author:
        return []
    par, d, cur, i = [], 0, "", 0
    s = alan_author
    while i < len(s):
        if s[i] == "{": d += 1
        if s[i] == "}": d -= 1
        if d == 0 and s[i:i + 5].lower() == " and ":
            par.append(cur.strip()); cur = ""; i += 5; continue
        cur += s[i]; i += 1
    par.append(cur.strip())
    out = []
    for p in par:
        if p.startswith("{") and p.endswith("}") and p.count("{") == 1:
            out.append(("", p[1:-1])); continue
        if p.lower() == "others":
            out.append(("", "others")); continue
        if "," in p:
            a, b = p.split(",", 1); out.append((b.strip(), a.strip()))
        else:
            w = p.split()
            k = len(w) - 1
            while k > 0 and w[k - 1][0].islower():
                k -= 1
            out.append((" ".join(w[:k]), " ".join(w[k:])))
    return out
