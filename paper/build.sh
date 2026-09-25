#!/usr/bin/env bash
set -u
TEC=${DNH_DATA}/.venv/bin/tectonic
cd "$(dirname "$0")"
if [ "${P11_KAPI22:-1}" = "1" ]; then
  ${PYTHON:-python3} \
    ${DNH_ROOT}/paper/checks/gate_generated_layer_fresh.py fig || exit 3
else
  echo ""
fi

${PYTHON:-python3} \
  ${DNH_ROOT}/paper/checks/gate_draft_phrases.py
rc23=$?
[ "$rc23" != "0" ] && { echo ""; exit 23; }
${PYTHON:-python3} ${DNH_ROOT}/paper/checks/sozluk_gate.py
rc27=$?
[ "$rc27" != "0" ] && { echo ""; exit 27; }
${PYTHON:-python3} ${DNH_ROOT}/paper/checks/kutuk_gate.py || echo "   ★★ KAPI-24 kendisi ariza verdi (uyari kipi, derleme sürer)"
${PYTHON:-python3} ${DNH_ROOT}/paper/checks/card_izi_gate.py || echo "   ★★ KAPI-25 kendisi ariza verdi (uyari kipi, derleme sürer)"

for kip in draft submission; do
  v=1; [ "$kip" = submission ] && v=0
  sed "s|^\\\\providecommand{\\\\taslakkipi}{1}|\\\\providecommand{\\\\taslakkipi}{$v}|" \
      p11.tex > "p11_${kip}.tex"
  echo "── ${kip} (taslakkipi=$v)"
  timeout 600 $TEC -X compile --print "p11_${kip}.tex" > "p11_${kip}.log" 2>&1
  rc=$?
  ham=$(grep -c "^\[EKSIK " "p11_${kip}.log" || true)
  pay=$(grep -o "\[EKSIK-PAYDA\] toplam=[0-9]*" "p11_${kip}.log" | tail -1)
  n=${pay##*=}
  echo "$rc ${n:-?} $ham $pay"
  grep "^\[EKSIK " "p11_${kip}.log" | sort -u | sed 's/^/     /'
  [ $rc -ne 0 ] && { echo "★ ARIZA: $kip derlenmedi"; tail -3 "p11_${kip}.log"; exit 3; }
  ${PYTHON:-python3} \
    ${DNH_ROOT}/paper/checks/tex_tree.py p11.tex \
    | head -1
  if [ "$kip" = submission ]; then
    ${PYTHON:-python3} \
      ${DNH_ROOT}/paper/checks/tounicode_onar.py p11_submission.pdf
    rcTU=$?
    [ "$rcTU" != "0" ] && { echo "★★ METIN KATMANI ONARIMI DÜSTÜ (rc=$rcTU)"; exit 31; }
  fi

  if [ "$kip" = submission ]; then
    tr=$($TEC --version >/dev/null 2>&1; ${PYTHON:-python3} - <<'PYX'
from pypdf import PdfReader
import sys
t="".join((p.extract_text() or "") for p in PdfReader("p11_submission.pdf").pages)
print(sum(t.count(w) for w in ("BORC","BORC LISTESI","borc")))
PYX
)
    echo "   ★ BORC_KAPISI · submission'da Türkce borc dizgisi = ${tr} ⇒ esik 0 ⇒ EYLEM: >0 ise derleme DÜSER"
    [ "${tr:-1}" != "0" ] && { echo "★★ KAPI DÜSTÜ: gönderim PDF'i borc sayfasi tasiyor"; exit 4; }
  fi
done
eks=$(${PYTHON:-python3} - <<'PYX'
import re, glob
import sys as _s; _s.path.insert(0, "${DNH_ROOT}/scripts")
from tex_agaci import agac as _agac
_AGAC, _EKSIK = _agac("p11.tex")
s=""
for f in _AGAC:
    try: s+=open(f,encoding="utf-8").read()
    except Exception: pass
d=re.sub(r"(?m)^\s*%.*$","",s)
t=set(re.findall(r"\\label\{([^}]+)\}",d)); a=re.findall(r"\\ref\{([^}]+)\}",d)
print(len(set(x for x in a if x not in t)))
PYX
)
echo "   ★ REF_KAPISI · tanimsiz etiket = ${eks} ⇒ esik 0 ⇒ EYLEM: >0 ise derleme DÜSER"
[ "${eks:-1}" != "0" ] && { echo "★★ REF_KAPISI DÜSTÜ"; exit 5; }
tr=$(${PYTHON:-python3} - <<'PYX'
from pypdf import PdfReader
import re
t=" ".join((p.extract_text() or "") for p in PdfReader("p11_submission.pdf").pages)
print(len(re.findall(r"(?<![\w])(taban|merdiven|basamak|sayac|verdict|ölcüm|yargic|seed|kisi|kol)(?![\w])", t)))
PYX
)
echo "   ★ ANON_KAPISI · gönderim PDF'inde Türkce terim = ${tr} ⇒ esik 0 ⇒ EYLEM: >0 ise derleme DÜSER"
[ "${tr:-1}" != "0" ] && { echo "★★ ANON_KAPISI DÜSTÜ — cift-kör sizintisi"; exit 6; }
${PYTHON:-python3} - <<'PYX'
import glob, re, sys
try:
    import pypdf
except Exception:
    print(""); sys.exit(0)
TR = re.compile(r"(?<![\w])(taban|merdiven|basamak|saya\u00e7|h\u00fck\u00fcm|\u00f6l\u00e7\u00fcm|"
                r"yarg\u0131\u00e7|seed|ki\u015fi|kol|tarif|tercih|kuvvet|d\u00fczeltme|aile)(?![\w])", re.I)
vur = {}
for f in sorted(glob.glob("fig/*.pdf")):
    try: x = " ".join((pg.extract_text() or "") for pg in pypdf.PdfReader(f).pages)
    except Exception: continue
    h = sorted(set(TR.findall(x)))
    if h: vur[f] = h
print(f"   \u2605 ANON_FIGUR \u00b7 taranan {len(glob.glob('fig/*.pdf'))} figur PDF'i \u00b7 "
      f"T\u00fcrk\u00e7e ta\u015fiyan = {len(vur)} \u21d2 e\u015fik 0 \u21d2 EYLEM: >0 ise \u00e7evir")
for f, h in vur.items(): print(f"     \u2717 {f}: {h}")
sys.exit(0)
PYX
${PYTHON:-python3} ${DNH_ROOT}/paper/checks/anon_path_gate.py p11_submission.pdf "fig/*.pdf"
rca2=$?
[ "$rca2" != "0" ] && { echo "★★ ANON-2 DÜSTÜ — gönderim PDF'inde depo izi"; exit 6; }

${PYTHON:-python3} ${DNH_ROOT}/paper/checks/gonderim_kipi_gate.py p11_submission.pdf
k26=$?
[ "$k26" != "0" ] && { echo "★★ KAPI-26 DÜSTÜ — gönderim PDF'inde taslak izi var"; exit 26; }

${PYTHON:-python3} - <<'PYX'
import re, sys
ham = open("p11.tex", encoding="utf-8").read()
t = re.sub(r"(?<!\\)%.*", "", ham)
i_app = t.index("\\appendix")
FLOAT = re.compile(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", re.S)
bolge = [(m.start(), m.end()) for m in FLOAT.finditer(t)]
E = [(m.group(1), m.start()) for m in re.finditer(r"\\label\{((?:tab|fig):[^}]+)\}", t)]
import os as _os
for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", t):
    f = m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"
    if not _os.path.exists(f): continue
    try: s = open(f, encoding="utf-8").read()
    except Exception: continue
    for mm in re.finditer(r"\\label\{((?:tab|fig):[^}]+)\}", s):
        E.append((mm.group(1), m.start())); bolge.append((m.start(), m.end()))
konum = dict(E); ana = {e for e, p in E if p < i_app}
ek = {e for e, p in E if p >= i_app} - ana
NIT = re.compile(r"Appendix~?\s*$|Appendix\s+(?:Figure|Table)~?\s*$")
S = {}
for m in re.finditer(r"\\ref\{((?:tab|fig):[^}]+)\}", t):
    ad, p = m.group(1), m.start(); kp = konum.get(ad)
    kendi = kp is not None and any(a <= p < b and a <= kp < b for a, b in bolge)
    S.setdefault(ad, []).append((p, kendi, p < i_app, bool(NIT.search(t[max(0, p-45):p]))))
for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", t):
    f = m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"
    if not _os.path.exists(f): continue
    try: s = re.sub(r"(?<!\\)%.*", "", open(f, encoding="utf-8").read())
    except Exception: continue
    fb = [(mm.start(), mm.end()) for mm in FLOAT.finditer(s)]
    lab_in = {mm.group(1): mm.start() for mm in re.finditer(r"\\label\{((?:tab|fig):[^}]+)\}", s)}
    for r in re.finditer(r"\\ref\{((?:tab|fig):[^}]+)\}", s):
        ad, q = r.group(1), r.start(); kq = lab_in.get(ad)
        kendi = kq is not None and any(a <= q < b and a <= kq < b for a, b in fb)
        S.setdefault(ad, []).append((m.start(), kendi, m.start() < i_app, bool(NIT.search(s[max(0, q-45):q]))))
d1 =sorted(e for e, _ in E if not [c for c in S.get(e, []) if not c[1]])
d2 = sorted(e for e in ana if not [c for c in S.get(e, []) if not c[1] and c[2]])
d3 = sorted((e, c[0]) for e in ek for c in S.get(e, []) if not c[1] and c[2] and not c[3])
n = sum(len(v) for v in S.values()); nk = sum(1 for v in S.values() for c in v if c[1])
print(f"{len(konum)} {len(ana)} {len(ek)} {n}"
      f"{nk} {len(d1)}"
      f"{len(d2)} {len(d3)}"
      f"")
for e in d1: print(f"     \u2717 D1 ATIFSIZ: {e}")
for e in d2: print(f"     \u2717 D2 ana metinde ama atif yalniz EK'ten: {e}")
for e, p in d3:
    print(f"     \u2717 D3 satir {t[:p].count(chr(10))+1}: {e} \u2014 ana metinden nitelemesiz "
          f"(\u00abAppendix Figure/Table\u00bb yaz)")
sys.exit(0 if not (d1 or d2 or d3) else 7)
PYX
rc2=$?
[ $rc2 -ne 0 ] && { echo "★★ REF-2 KAPISI DÜSTÜ"; exit 7; }

${PYTHON:-python3} - <<'PYX'
from pypdf import PdfReader
import re, sys
ad = set()
def gez(f):
    try: f = f.get_object()
    except Exception: return
    b = f.get("/BaseFont")
    if b: ad.add(str(b))
    for d in (f.get("/DescendantFonts") or []): gez(d)
for p in PdfReader("p11_submission.pdf").pages:
    res = (p.get("/Resources") or {}).get("/Font") or {}
    try: res = res.get_object()
    except Exception: pass
    for k in res: gez(res[k])
IST = {"kalin": r"(Bold|Medi|Semib|Black|-Bd|bx)",
       "kalin_serif": r"(Rom|Times|ptm|Serif).*(Bold|Medi|Bd)|(Bold|Medi|Bd).*(Rom|Times|Serif)",
       "monospace": r"(Mono|MonL|Courier|CM(TT|SLTT))",
       "italik": r"(Ital|Oblique|CMTI|CMMI)"}
bul = {k: sorted(a for a in ad if re.search(v, a, re.I)) for k, v in IST.items()}
eks = [k for k, v in bul.items() if not v]
print(f"   ★ FONT_KAPISI · n_gomulu={len(ad)} · " +
      " · ".join(f"{k}={len(v)}" for k, v in bul.items()) +
      " ⇒ esik her biri ≥1 ⇒ EYLEM: eksikse DERLEME DÜSER")
for k in eks: print(f"{k}")
if not eks:
    print("     " + " · ".join(f"{k}: {v[0].split('+')[-1]}" for k, v in bul.items()))
kosu = []
def _gor(t, cm, tm, fd, fs):
    if t and t.strip():
        try: kosu.append((t, str((fd or {}).get("/BaseFont"))))
        except Exception: pass
for _p in PdfReader("p11_submission.pdf").pages:
    _p.extract_text(visitor_text=_gor)
HED = {"kalin_serif": (r"(Rom|Times|Serif).*(Medi|Bold|Bd)", "asymmetry"),
       "monospace":   (r"(MonL|Mono|Courier)", "inst_1k")}
red = 0
for _ad, (_d, _o) in HED.items():
    _v = [(t, f) for t, f in kosu if f and re.search(_d, f, re.I) and _o.lower() in t.lower()]
    red += 0 if _v else 1
    print(f"     {'OK' if _v else 'XX'} render {_ad}: «{_o}» {'o yuzle cizilmis' if _v else 'O YUZLE CIZILMEMIS'}"
          + (f" ({_v[0][1].split('+')[-1]})" if _v else ""))
print(f"     ⇒ render esik: 2/2 · red={red} ⇒ EYLEM: red>0 ise DERLEME DÜSER")
sys.exit(0 if not eks and red == 0 else 8)
PYX
rc3=$?
[ $rc3 -ne 0 ] && { echo "★★ FONT_KAPISI DÜSTÜ"; exit 8; }

${PYTHON:-python3} - <<'PYX'
from pypdf import PdfReader
import glob, re, sys
def duz(f):
    try: t = " ".join((p.extract_text() or "") for p in PdfReader(f).pages)
    except Exception: return ""
    return re.sub(r"\s+", " ", t.replace("-\n", "").replace("­", ""))
YASAK = {
 "bank*": r"\bbank(ed|ing|s)?\b", "seal*": r"\bseal(ed|ing|s)?\b",
 "bet*": r"\bbets?\b|\bbetting\b", "verdict*": r"\bverdicts?\b",
 "kill-test": r"\bkill[\s\-]?tests?\b", "POZKON": r"\bPOZKON\b",
 "kesif": r"KE[SS]I?IF|\bkesif\b", "olculemez": r"[ÖO]L[CC][ÜU]LEMEZ",
 "ajan-adi": r"\b(Pat[ii]ka|Annotator|Assistant|Line|Auditor)\b",
 "W-numarasi": r"\bW-\d{2,4}\b", "CARD": r"\bKART\b",
 "REPORT": r"D[ÖO]N[ÜU][SS]", "NOTICE": r"\bHABER\b",
 "tr-sinif": r"(?-i:\b(?:RET|ASISTAN|ASISTAN|META|DEVAM)\b)",
 "tr-verdict-adi": r"(?-i:\b(?:ALET-KAYDI|DEJENERE-RAF|EKSIK-PAYDA|OZGUL-IMZA|TAM-HAVUZ|ISARET-TUTTU|ISARET-DONDU|KOSULMADI)\b|\b[A-ZCGIÖSÜ]*[CGIÖSÜ][A-ZCGIÖSÜ]*-[A-ZCGIÖSÜ]+\b|\b[A-ZCGIÖSÜ]+-[A-ZCGIÖSÜ]*[CGIÖSÜ][A-ZCGIÖSÜ]*\b)",
}
hedef = [("govde", "p11_submission.pdf")] + [("figur", f) for f in sorted(glob.glob("fig/*.pdf"))]
metin = {f: duz(f) for _, f in hedef}
top, vur = 0, []
for ad, d in YASAK.items():
    for _, f in hedef:
        for m in re.finditer(d, metin[f], re.I):
            top += 1
            vur.append((ad, f, metin[f][max(0, m.start()-45):m.start()+35]))
print(f"   ★ JARGON_KAPISI · taranan {len(hedef)} PDF ({sum(len(v) for v in metin.values())} kar) · "
      f"yasak terim sinifi {len(YASAK)} · IHLAL = {top} ⇒ esik 0 ⇒ EYLEM: >0 ise DERLEME DÜSER")
for ad, f, c in vur[:12]:
    print(f"     ✗ [{ad}] {f}: …{c}…")
sys.exit(0 if top == 0 else 10)
PYX
rc4=$?
[ $rc4 -ne 0 ] && { echo "★★ JARGON_KAPISI DÜSTÜ — ic sözlük gönderim PDF'ine sizmis"; exit 10; }

${PYTHON:-python3} ../../../paper/checks/sayfa_gate.py p11_submission.pdf
rcS=$?
[ $rcS -ne 0 ] && { echo "$rcS"; exit 28; }
${PYTHON:-python3} ../../../paper/checks/sayi_density.py p11_submission.pdf
${PYTHON:-python3} ../../../paper/checks/sentence_uzunlugu.py --tex p11.tex
${PYTHON:-python3} - <<'PYQ18'
import re, io, json, os, datetime as dt
SON = dt.date(2026, 9, 16)
bugun = dt.datetime.now(dt.timezone.utc).date()
if bugun > SON:
    print(f"   \u2605 KAPI-18 \u00b7 ozet haritasi \u00b7 PENCERE KAPANDI ({SON}) \u21d2 kosmadi")
    raise SystemExit(0)
HAR = "OZET_HARITASI.json"
if not os.path.exists(HAR):
    print(f"{HAR}")
    raise SystemExit(0)
D = json.load(io.open(HAR, encoding="utf-8"))
ham = io.open("p11.tex", encoding="utf-8").read()
a = ham.index("\\begin{abstract}"); b = ham.index("\\end{abstract}")
t = re.sub(r"(?<!\\)%.*", "", ham[a+len("\\begin{abstract}"):b])
t = t.replace("\\vekalet{", " ").strip()
if t.endswith("}"): t = t[:-1]
t = re.sub(r"\s+", " ", t)
cum = re.split(r"(?<=[.!?])\s+(?=[A-Z\\\\])", t)
def norm(x): return re.sub(r"\s+", " ", x).strip()
H = {norm(c["metin"]): c for c in D["cumle"]}
eslesmeyen = [c for c in cum if norm(c) not in H]
bayat = [k for k in H if k not in [norm(c) for c in cum]]
kayipkaynak = []
for c in cum:
    e = H.get(norm(c))
    if not e:
        continue
    if str(e.get("prereg", "")).startswith("SAYI-YOK"):
        continue
    for yol in re.findall(r"[A-Za-z0-9_./\-]+\.(?:json|tex)", str(e.get("sayi_kaynagi") or "")):
        tam = yol if os.path.exists(yol) else os.path.join("..", "..", "..", yol)
        if not os.path.exists(yol) and not os.path.exists(tam):
            kayipkaynak.append((e["i"], yol))
print(f"   \u2605 KAPI-18 \u00b7 ozet haritasi \u00b7 cumle {len(cum)} \u00b7 haritada {len(H)} \u00b7 "
      f"ESLESMEYEN = {len(eslesmeyen)} \u00b7 BAYAT = {len(bayat)} \u00b7 KAYIP-KAYNAK = "
      f"{len(kayipkaynak)} \u21d2 esik 0/0/0 \u21d2 EYLEM: UYARIR, derlemeyi DUSURMEZ "
      f"(pencere {SON} kapanir)")
for c in eslesmeyen[:6]:
    print(f"     \u2717 haritasiz cumle: {c[:90]}...")
for k in bayat[:6]:
    print(f"     \u2717 BAYAT harita satiri (ozette yok): {k[:90]}...")
for i, y in kayipkaynak[:6]:
    print(f"     \u2717 cumle[{i}] kaynagi DISKTE YOK: {y}")
PYQ18
${PYTHON:-python3} - <<'PYQ20'
import re, fitz
BAR = 290
import sys as _s; _s.path.insert(0, "${DNH_ROOT}/scripts")
from p11_ozet_penceresi import ozet_jetonlari
tok, _tani = ozet_jetonlari("p11_submission.pdf")
if tok is None:
    print(f"{_tani}")
else:
    n = len([x for x in tok if re.search(r"[A-Za-z0-9]", x)])
    print(f"   ★ KAPI-20 · pencere: {_tani}")
    print(f"   ★ KAPI-20 · özet (BASILI PDF, \\input acik) · n_jeton={len(tok)} · "
          f"n_kelime={n} ⇒ esik {BAR} ⇒ EYLEM: >{BAR} ise UYARIR (kesim sahibin; "
          f"derlemeyi DÜSÜRMEZ)")
    if n > BAR:
        print(f"   ★★ KAPI-20 UYARI: özet tavani ASILDI — {n - BAR} kelime fazla")
PYQ20

${PYTHON:-python3} - <<'PYQ17'
import re, io, sys
ham = io.open("p11.tex", encoding="utf-8").read()
t = re.sub(r"(?<!\\)%.*", "", ham)
i_app = t.index("\\appendix")
def kunye(metin):
    out = []
    for m in re.finditer(r"\\caption\{", metin):
        d, j = 0, m.end() - 1
        for k in range(j, len(metin)):
            if metin[k] == "{": d += 1
            elif metin[k] == "}":
                d -= 1
                if d == 0:
                    out.append(metin[m.end():k]); break
    return out
def kelime(c):
    c = re.sub(r"\\[a-zA-Z]+|\$[^$]*\$", " ", c)
    return len(re.findall(r"[A-Za-z][A-Za-z'\u2019\-]*", c))
hedef = []
for c in kunye(t[:i_app]):
    hedef.append(("p11.tex", c))
import sys as _s; _s.path.insert(0, "${DNH_ROOT}/scripts")
from tex_agaci import agac as _agac
_ana = t[:i_app]
for f in _agac("p11.tex")[0][1:]:
    if ("\\input{" + f[:-4] + "}") not in _ana and ("\\input{" + f + "}") not in _ana:
        continue
    try:
        g = io.open(f, encoding="utf-8").read()
    except Exception:
        continue
    for c in kunye(g):
        hedef.append((f, c))
ihlal = [(f, kelime(c)) for f, c in hedef if kelime(c) > 80]
print(f"   \u2605 KAPI-17 \u00b7 kunye uzunlugu \u00b7 ana metin kunyesi {len(hedef)} \u00b7 "
      f"IHLAL = {len(ihlal)} \u21d2 esik 80 kelime \u21d2 EYLEM: >80 ise bulgu cumlesi "
      f"GOVDEYE (*_body), serh EK'e (*_note) \u2014 kunye BETIMLER")
for f, n in ihlal:
    print(f"     \u2717 {f}: {n} kelime")
if ihlal:
    sys.exit(12)
PYQ17
rc17=$?
[ $rc17 -ne 0 ] && { echo "★★ KAPI-17 DÜSTÜ — ana metin künyesi 80 kelimeyi asiyor"; exit 12; }

${PYTHON:-python3} - <<'PYQ16'
import re, glob, io, sys
import sys as _s; _s.path.insert(0, "${DNH_ROOT}/scripts")
from tex_agaci import agac as _agac
_AGAC, _EKSIK = _agac("p11.tex")
pat = re.compile(r"\\\\([A-Za-z]+)")
n_d = n_a = 0; org = []
for f in _AGAC:
    n_d += 1
    s = io.open(f, encoding="utf-8").read()
    for m in pat.finditer(s):
        n_a += 1
        org.append(f"{f}: \\\\{m.group(1)}")
print(f"   ★ KAPI-16 · ikiye katlanmis makro · taranan {n_d} kaynak · IHLAL = {n_a} "
      f"⇒ esik 0 ⇒ EYLEM: >0 ise ÜRETICIDE tek ters böluye indir, .tex elle düzeltilmez")
for x in org[:10]:
    print("     " + x)
if n_a:
    sys.exit(11)
PYQ16
rc16=$?
[ $rc16 -ne 0 ] && { echo "★★ KAPI-16 DÜSTÜ — ikiye katlanmis makro PDF'e düz metin olarak düser"; exit 11; }

${PYTHON:-python3} - <<'PYX'
import re, sys
from pypdf import PdfReader
ham = open("p11.tex", encoding="utf-8").read()
ham = re.sub(r"(?<!\\)%.*", "", ham)
def _verdict_sil(s):
    out, i = [], 0
    while True:
        j = s.find("\\verdict{", i)
        if j < 0:
            out.append(s[i:]); break
        out.append(s[i:j]); k, d = j + 7, 1
        while k < len(s) and d:
            d += (s[k] == "{") - (s[k] == "}"); k += 1
        i = k
    return "".join(out)
import sys as _s; _s.path.insert(0, "${DNH_ROOT}/scripts")
from tex_agaci import agac as _agac
def _metin_kipi(x):
    x = _verdict_sil(re.sub(r"(?<!\\)%.*", "", x))
    _p = re.split(r"(\$[^$]*\$)", x)
    return "".join(q for i, q in enumerate(_p) if i % 2 == 0)
ham = _verdict_sil(ham)
parca = re.split(r"(\$[^$]*\$)", ham)
metin = "".join(p for i, p in enumerate(parca) if i % 2 == 0)
import glob
MINUS = chr(0x2212)
n_minus = metin.count(MINUS)
_suclu = []
for _f in _agac("p11.tex")[0][1:]:
    try: _n = _metin_kipi(open(_f, encoding="utf-8").read()).count(MINUS)
    except OSError: continue
    if _n: _suclu.append((_f, _n)); n_minus += _n
for _f, _n in _suclu:
    print(f"     \u2717 {_f}: METIN kipinde U+2212 = {_n}")
print(f"   \u2605 KAPI-10 · d\u00fc\u015fen glif · METIN kipinde U+2212 = {n_minus} "
      f"\u21d2 e\u015fik 0 \u21d2 EYLEM: >0 ise DERLEME D\u00dc\u015eER")
pdf = " ".join((pg.extract_text() or "") for pg in PdfReader("p11_submission.pdf").pages)
metin = re.sub(r"\\ifnum\\taslakkipi=1.*?\\fi", "", metin, flags=re.S)
TR = set("cCgGiIöÖsSüÜâÂîÛ")
kaynak = {c for c in metin if ord(c) > 127 and c not in TR}
eksik = sorted(c for c in kaynak if c not in pdf)
def _yorumsuz(t, bib=False):
    if bib:
        t = "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("%"))
    else:
        t = re.sub(r"(?<!\\\\)%.*", "", t)
    return "".join(x for i, x in enumerate(re.split(r"(\$[^$]*\$)", t)) if i % 2 == 0)

ek_kaynak = {}
import os as _os
_bibs = []
for _m in re.finditer(r"\\bibliography\{([^}]*)\}", open("p11.tex", encoding="utf-8").read()):
    for _b in _m.group(1).split(","):
        _b = _b.strip()
        if _b and _os.path.exists(_b + ".bib"):
            _bibs.append(_b + ".bib")
for yol in sorted(glob.glob("fig/*.tex")) + sorted(set(_bibs)):
    try:
        g = _yorumsuz(open(yol, encoding="utf-8").read(), bib=yol.endswith(".bib"))
    except OSError:
        continue
    for c in g:
        if ord(c) > 127 and c not in pdf:
            ek_kaynak.setdefault(c, []).append(yol)
print(f"   \u2605 KAPI-10b (report mode) \u00b7 non-ASCII in source (Turkish excluded) {len(kaynak)} "
      f"\u00b7 missing from PDF {len(eksik)} \u21d2 ACTION: inspect, does not stop")
if eksik:
    print("     \u2717 " + " ".join(f"U+{ord(c):04X}({c})" for c in eksik[:12]))
print(f"   \u2605 KAPI-10c \u00b7 KAYIP GL\u0130F \u00b7 taranan {len(glob.glob('fig/*.tex'))+1} "
      f"yan kaynak (fig/*.tex + bibliography'deki bib'ler) \u00b7 PDF'te bulunmayan karakter "
      f"{len(ek_kaynak)} \u21d2 e\u015fik 0 \u21d2 EYLEM: >0 ise ka\u00e7\u0131\u015f dizisi yaz")
for c, ff in sorted(ek_kaynak.items()):
    print(f"     \u2717 U+{ord(c):04X}({c}) \u2190 {', '.join(sorted(set(ff)))}")
sys.exit(9 if n_minus else 0)
PYX
rc10=$?
[ $rc10 -ne 0 ] && { echo "★★ KAPI-10 DÜSTÜ: metin kipinde U+2212 var, PDF'e düsmeyebilir"; exit 9; }
${PYTHON:-python3} - <<'PYX'
import glob, json, hashlib, os, re
ROOT = "${DNH_ROOT}"
HEX = re.compile(r"^[0-9a-f]{16}$")
def sha16(y):
    h = hashlib.sha256()
    with open(y, "rb") as f:
        for p in iter(lambda: f.read(1 << 20), b""): h.update(p)
    return h.hexdigest()[:16]
def coz(y):
    for c in (y, f"{ROOT}/{y}", f"{ROOT}/results/{os.path.basename(y)}"):
        if os.path.exists(c): return c
    return None
n_k = n_s = 0; BAYAT = []; EKSIK = 0; YOLYOK = 0
for m in sorted(glob.glob("fig/*.meta.json")):
    d = json.load(open(m, encoding="utf-8")); n_k += 1
    for s in d.get("sources") or []:
        y = s.get("yol") or s.get("path"); e = s.get("sha256_16"); n_s += 1
        if not y or not e or not HEX.match(str(e)): EKSIK += 1; continue
        t = coz(y)
        if not t: YOLYOK += 1; continue
        if sha16(t) != e: BAYAT.append((os.path.basename(m), y, e, sha16(t)))
print(f"{n_k} {n_s} {len(BAYAT)}"
      f"{EKSIK} {YOLYOK}"
      f"")
for b, y, e, n in BAYAT:
    print(f"     ★★ BAYAT {b} ← {y}\n        {e} → {n}")
PYX

${PYTHON:-python3} - <<'PYX'
import glob, json, os, re
YOL = re.compile(r"[\"']((?:docs|data)/[A-Za-z0-9_./-]+\.json)[\"']")
EK = {"t1_families": "T1_families", "t1_blind_panel": "T1_blind_panel",
      "t2_template": "T2_template", "t3_coverage": "T3_coverage",
      "t4_rungs": "T4_rungs", "a1_breadth_table": "A1_breadth",
      "seedcov": "SEEDCOV"}
n_u = 0; KUNYE_YOK = []; BEYAN_DISI = []
for py in sorted(glob.glob("fig/*.py")):
    ad = EK.get(os.path.basename(py)[:-3])
    if not ad:
        continue
    n_u += 1
    src = open(py, encoding="utf-8").read()
    okunan = set(YOL.findall(src)) | set(YOL.findall(src.replace('"\n', '').replace('"', '', 0)))
    okunan = {y for y in okunan if os.path.exists(
        "${DNH_ROOT}/" + y)}
    m = f"fig/{ad}.meta.json"
    if not os.path.exists(m):
        KUNYE_YOK.append((ad, py)); continue
    beyan = {s.get("yol") for s in (json.load(open(m, encoding="utf-8")).get("sources") or [])}
    for y in sorted(okunan - beyan):
        BEYAN_DISI.append((ad, y))
print(f"   ★ KAPI-11b kunye-kapsami · uretici {n_u} · KUNYE-YOK {len(KUNYE_YOK)} "
      f"· BEYAN-DISI {len(BEYAN_DISI)} ⇒ esik 0 ⇒ EYLEM: uretici kunyeye "
      f"kaynagi EKLER (rapor kipi: uyarir, durdurmaz)")
for a, y in KUNYE_YOK:
    print(f"     ★★ KUNYE-YOK {a} ← uretici {y}")
for a, y in BEYAN_DISI:
    print(f"{a} {y}")
PYX

${PYTHON:-python3} \
  ${DNH_ROOT}/paper/checks/gate_ref_target.py
k14=$?
[ "$k14" != "0" ] && { echo ""; exit 14; }

${PYTHON:-python3} \
  ${DNH_ROOT}/paper/checks/gate_count_consistency.py \
  "$(pwd)/p11.tex" | sed 's/^/   /'
rc15=${PIPESTATUS[0]}
[ "$rc15" != "0" ] && { echo ""; exit 15; }

${PYTHON:-python3} \
  ${DNH_ROOT}/paper/checks/gate_undefined_citation_band_ratio.py \
  "$(pwd)/p11.tex" | sed 's/^/   /'
rc19=${PIPESTATUS[0]}
[ "$rc19" != "0" ] && { echo "★★ KAPI-19 DÜSTÜ — tanimsiz atif ya da bant↔oran celiskisi"; exit 19; }

${PYTHON:-python3} \
  ${DNH_ROOT}/paper/checks/gate_table_rows_printed.py \
  "$(pwd)/p11.tex" p11_submission.pdf
rc21=$?
[ "$rc21" != "0" ] && { echo "★★ KAPI-21 DÜSTÜ — tablonun basili satiri kaynaktakinden az ya da sayfadan tasan metin var"; exit 21; }

echo "[PAYDA] derle: n_kip=2 · hal_pdf=$(ls -1 p11_draft.pdf p11_submission.pdf 2>/dev/null | wc -l)"
