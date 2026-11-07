#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import hashlib, json, re, sys, time
sys.path.insert(0, __DNH_ROOT__ + "/scripts")
import addressee_run as MK
import form_count as FS
from reading_style_object import payda

KOK = __DNH_ROOT__ + ""
KAYNAK = __DNH_DATA__ + "/unreleased/inst_1k.txt"
HEDEF = __DNH_DATA__ + "/unreleased/inst_1k_kisisiz.txt"
CIK = f"{KOK}/results/v28_impersonal_urial_manifest.json"

HEDEFLI = [
    ("between a human and an AI assistant (you).",
     "between a human and an AI assistant."),
    ('Users place their queries under "# Query:", and your responses are under  "# Answer:".',
     'Users place their queries under "# Query:", and the assistant\'s responses are under  "# Answer:".'),
    ("You are a helpful, respectful, and honest assistant.",
     "The assistant is helpful, respectful, and honest."),
    ("You should always answer as helpfully as possible while ensuring safety.",
     "The assistant should always answer as helpfully as possible while ensuring safety."),
    ("Your answers should be well-structured and provide detailed information.",
     "The answers should be well-structured and provide detailed information."),
    ("Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.",
     "The responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful."),
    ("Your response must be socially responsibly, and thus you can reject to answer some controversial topics.",
     "The response must be socially responsible, and thus the assistant can reject to answer some controversial topics."),
    ("You are a detective interrogating a suspect.",
     "A detective is interrogating a suspect."),
    ("How do you get them to confess",
     "How does one get them to confess"),
    ("Can you tell me some common types of renewable energy sources?",
     "What are some common types of renewable energy sources?"),
]
GENEL = [
    (r"\bYou're\b", "One is"), (r"\byou're\b", "one is"),
    (r"\bYou've\b", "One has"), (r"\byou've\b", "one has"),
    (r"\bYou'll\b", "One will"), (r"\byou'll\b", "one will"),
    (r"\bYou'd\b", "One would"), (r"\byou'd\b", "one would"),
    (r"\bYourself\b", "Oneself"), (r"\byourself\b", "oneself"),
    (r"\bYours\b", "One's"), (r"\byours\b", "one's"),
    (r"\bYour\b", "One's"), (r"\byour\b", "one's"),
    (r"\bYou\b", "One"), (r"\byou\b", "one"),
]
SAHIS2 = re.compile(r"\b(you|your|yours|yourself|yourselves|"
                    r"you're|you've|you'll|you'd)\b", re.I)


def donustur(t):
    n_hedefli = 0
    for a, b in HEDEFLI:
        if a in t:
            t = t.replace(a, b); n_hedefli += 1
    n_genel = 0
    for pat, rep in GENEL:
        t, k = re.subn(pat, rep, t)
        n_genel += k
    return t, n_hedefli, n_genel


def main():
    nlp = FS._boru()
    ham = open(KAYNAK, encoding="utf-8").read()
    yeni, nh, ng = donustur(ham)
    cerceve = {k: (ham.count(k), yeni.count(k))
               for k in ("# Instruction", "# Query:", "# Answer:", "```")}
    cerceve["satir"] = (ham.count("\n"), yeni.count("\n"))
    cerceve_tam = all(a == b for a, b in cerceve.values())
    artik = sorted(set(m.group(0) for m in SAHIS2.finditer(yeni)))

    def m1(ms):
        return float(MK.olc_toplam(MK.satir_bilesenleri(
            nlp, [dict(metin=m) for m in ms]))["M1"])

    parts = re.split(r"(?m)^# Query:", yeni)
    bas_y, orn_y = parts[0], ["# Query:" + p for p in parts[1:]]
    cev_y = [re.search(r"# Answer:\s*```(.*?)```", o, re.S).group(1) for o in orn_y]
    olc = dict(baslik=round(m1([bas_y]), 4), cevaplar_uc=round(m1(cev_y), 4),
               onek_butun=round(m1([yeni]), 4))
    print(f"{nh} {len(HEDEFLI)}"
          f"{ng} {cerceve_tam}"
          f"{len(artik)} {artik}")
    print(f"{olc['baslik']} {olc['cevaplar_uc']}"
          f"{olc['onek_butun']}")
    if not cerceve_tam:
        print(f"  ★★ CERCEVE BOZULDU {cerceve} ⇒ dosya YAZILMADI"); return 3
    if artik:
        print(f""); return 3
    open(HEDEF, "w", encoding="utf-8").write(yeni)
    sha = hashlib.sha256(yeni.encode()).hexdigest()
    D = dict(damga_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             alet="scripts/v28_impersonal_urial.py", borc="D-0910-V28 §1",
             kaynak=dict(dosya=KAYNAK,
                         sha256="3a3989a9c5a58293362de475be08218cc552544882b8a8356b558d03c1a91f06"),
             hedef=dict(dosya=HEDEF, sha256=sha, bayt=len(yeni.encode())),
             donusum=dict(n_hedefli=nh, n_genel=ng, cerceve=cerceve,
                          artik_ikinci_sahis=artik),
             M1_kisisiz=olc,
             M1_kaynak=dict(baslik=60.1504, cevaplar_uc=16.2866, onek_butun=21.2515),
             SERH=(""
                   ""))
    json.dump(D, open(CIK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    payda("v28_kisisiz_urial", n_rule=len(HEDEFLI) + len(GENEL), hal_hedefli=nh,
          hal_genel=ng, red_artik=len(artik), red_cerceve=0)
    print(f"  sha256 {sha[:16]}…  ✓ {HEDEF}\n✓ {CIK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
