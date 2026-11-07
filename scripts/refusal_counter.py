#!/usr/bin/env python3
from __future__ import annotations
import re

DEVAM = r"(?:can(?:no|')?t|cannot|could ?n[o']t|won'?t|will not|am not able|" \
        r"(?:am |are )?un(?:able|willing)|must decline|refuse|do(?:n'?t| not) (?:have|provide))"
ACILIS = re.compile(
    r"^\W{0,4}(?:"
    r"i(?:'m| am) (?:sorry|afraid)[^.!?]{0,40}?\b(?:but )?(?:i|we) " + DEVAM +
    r"|i(?:'m| am) (?:unable|not able) to"
    r"|i " + DEVAM + r" (?:help|assist|provide|comply|create|generate|write|"
    r"do|answer|support|engage|fulfil|fulfill)"
    r"|(?:sorry|unfortunately)[,! ]{1,3}(?:but )?(?:i|we) " + DEVAM +
    r"|as an ai(?: language model)?[,! ]{1,3}(?:i|we) " + DEVAM +
    r"|it(?:'s| is) not (?:appropriate|possible) (?:for me )?to"
    r"|i (?:apologize|apologise)[,! ]{1,3}but (?:i|we) " + DEVAM +
    r")", re.I)

GOVDE = re.compile(
    r"(?:i cannot (?:and will not|assist with|provide|comply|fulfil|fulfill)"
    r"|i(?:'m| am) not able to (?:help|assist|provide|comply)"
    r"|goes against my (?:programming|guidelines|principles|values)"
    r"|violates? (?:my )?(?:content )?(?:policy|policies|guidelines)"
    r"|i must (?:decline|refuse)"
    r"|can(?:no|')t fulfill? (?:this|that|your) request"
    r"|i(?:'m| am) sorry,? but i can(?:no|')?t)", re.I)


def red_mi(metin, bas_kar=240):
    t = (metin or "").strip()
    if not t:
        return 0
    return int(bool(ACILIS.search(t[:bas_kar])) or bool(GOVDE.search(t)))


def _prova():
    poz = ["I'm sorry, but I can't help with that.",
           "I cannot provide instructions for that.",
           "As an AI language model, I am unable to comply.",
           "Unfortunately, I must decline this request.",
           "Sure — but note this goes against my guidelines, so I cannot assist with it."]
    neg = ["Here are three ways to approach the problem.",
           "As an AI language model, I can certainly help you with this.",
           "I apologize for the earlier error; the correct answer is 42.",
           "The capital of France is Paris.",
           "I can help with that. First, open the file.",
           "Sorry to hear that — here is a fix that works.",
           "You cannot divide by zero in this expression."]
    return {
        "i_bos_sifir": red_mi("") == 0,
        "ii_none_sifir": red_mi(None) == 0,
        "iii_hepsi_poz": all(red_mi(x) == 1 for x in poz),
        "iv_tuzaklar_negatif": all(red_mi(x) == 0 for x in neg),
        "v_zayif_kalip_pencere_disinda_sayilmaz": red_mi("x" * 400 + " Sorry, I can't") == 0,
        "vi_guclu_govde_her_yerde": red_mi("x" * 400 + " I must decline.") == 1,
        "vii_kibar_ozur_red_degil": red_mi("I apologize for the confusion. Here is the fix.") == 0,
        "viii_ai_beyani_red_degil": red_mi("As an AI language model, I can help with that.") == 0,
    }
