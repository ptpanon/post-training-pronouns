#!/usr/bin/env python3
from __future__ import annotations
import os as _dnh_os
__DNH_ROOT__ = _dnh_os.environ.get("DNH_ROOT", ".")
__DNH_DATA__ = _dnh_os.environ.get("DNH_DATA", "data")
import os, sys
KOK = __DNH_ROOT__ + ""
sys.path.insert(0, f"{KOK}/scripts")

KOL = "kayit_sahissiz"
TALIMAT_SAHISSIZ = ("Write the continuation informally, personally and "
                    "conversationally, rather than neutrally or encyclopedically.\n\n")
ZINCIR = dict(PREREG=f"{KOK}/preregistration/prereg_capability_pronoun_free_2026-09-14.md",
              NOTICE=f"{KOK}/preregistration/prerun_notice_capability_pronoun_free_2026-09-14.md",
              PREDICTION=f"{KOK}/preregistration/prediction_capability_pronoun_free_2026-09-14.md")


def kur():
    import capability_arm as YK
    if KOL in YK.TALIMAT and YK.TALIMAT[KOL] != TALIMAT_SAHISSIZ:
        raise SystemExit(f"★ {KOL} anahtari yetenek_kol'da BASKA lafizla var ⇒ kosu YOK")
    YK.TALIMAT[KOL] = TALIMAT_SAHISSIZ
    YK.ZINCIR = dict(ZINCIR)
    return YK


if __name__ == "__main__":
    if "--kol" in sys.argv:
        _v = sys.argv[sys.argv.index("--kol") + 1]
        if _v != KOL:
            print(f"{KOL} {_v}")
            sys.exit(4)
    else:
        sys.argv += ["--kol", KOL]
    sys.exit(kur().main())
