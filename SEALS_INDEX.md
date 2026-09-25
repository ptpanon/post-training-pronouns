# Registration index

The registrations behind the paper (pre-registrations, pre-run notices, predictions and reading rules) were written in Turkish. The original texts are released at camera-ready. This index carries, for each registration, the bar in one English sentence, the outcome names it fixed in advance and its recorded score, and for each file its SHA-256 digest and the UTC timestamp of the commit that added it (and of the last change, where an erratum was prepended later).

Scores count the predictions as their verdict documents recorded them. *Operational* predictions (whether runs would finish) and *base-rate* statements are listed apart from the substantive ones; *not scored* covers predictions recorded as unscorable, unmeasurable or open, or with no written score.

## The registered bare-prompt direction of the second person (Depersonalization, $M_1$)

Internal name: `PREREG-9`

- **Bar.** Count the 16 families whose base-to-instruct change in second-person density (M1) on bare prompts is negative with a CI excluding zero: ≤3 is `DESEN-YOK`, 4–11 YEREL, ≥12 `GENIS`.
- **Outcome names.** `DESEN-YOK` · `YEREL` · `GENIS`
- **Score.** 3 substantive predictions: 2 held, 1 failed (`B9b`), 0 not scored; operational: `B9c` not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_depersonalization16_2026-08-31.md` | `72a7c4f4968175a5a5b6f236437e75aa451e65e080ae39593c89c79a91c3e13c` | 2026-08-31T17:20Z | 2026-08-31T21:54Z |

## The task-matched ground: the neutral panel's prompts inside the URIAL frame

Internal name: `PREREG-URIAL`

- **Bar.** Per family, ΔM1 = instruct minus base second-person density inside the same URIAL prompt: `BICIM-ÖZELLIGI` if |ΔM1| < MDE (1.645 × matched-placebo null-sd), `AGIRLIK-ÖZELLIGI` if ≥ MDE downward, TERS if ≥ MDE upward; fewer than 8 legs is `ÖLCÜLEMEZ-KAPSAM`.
- **Outcome names.** `BICIM-ÖZELLIGI` · `AGIRLIK-ÖZELLIGI` · `TERS` · `ÖLCÜLEMEZ-KAPSAM`
- **Score.** 6 substantive predictions: 2 held, 3 failed (`URIAL-F2`, `URIAL-F3`, `URIAL-F5`), 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_urial_2026-09-07.md` | `3d438f44d61951c27c62b138176062e15546f86d8af979ae6a1dac6adc827067` | 2026-09-07T12:35Z | 2026-09-10T09:40Z |
| `prediction_urial_2026-09-07.md` | `46d02f04d8e18803b41745cc23c6038dfd7b546b18adbcaf550a408b44ff2268` | 2026-09-07T14:29Z | — |

## The default system line, present or absent inside the same string

Internal name: `M-SISTEM-2`

- **Bar.** Per family, diff = (instruct − base second-person density with a neutral system line) − (the same without it): `SATIR-ETKISIZ` if |diff| < MDE in ≥12 families, `SATIR-ETKILI` if its CI excludes zero in ≥12 families, else `KARISIK`; under 12 measured families is `ÖLCÜLEMEZ`.
- **Outcome names.** `SATIR-ETKISIZ` · `SATIR-ETKILI` · `KARISIK` · `ÖLCÜLEMEZ (payda<12)`
- **Score.** 4 substantive predictions: 3 held, 1 failed (`F-MS2-2`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_system_prompt2_2026-09-13.md` | `8ed892c08f4a0343386f4a543ed10481f4a2685921352cd91ae796d30a388f52` | 2026-09-13T10:07Z | — |
| `prediction_system_prompt2_2026-09-13.md` | `7c60ba30c22fbef4526d08f7d8d90c9e6beefee5f13058f512317cae0fab30d0` | 2026-09-13T10:10Z | — |

## Two ladders outside the AI2 and Zephyr recipes (SmolLM2-1.7B, NeuralHermes-2.5)

Internal name: `V61-MERDIVEN`

- **Bar.** Per ladder and protocol: `AYNI-ADIM` if the preference step's ΔM1 is negative, its prompt-clustered CI excludes zero, and it exceeds the SFT step in magnitude with non-overlapping intervals; `FARKLI-ADIM` if the SFT step does so instead; otherwise `ÖLCÜLEMEZ`.
- **Outcome names.** `AYNI-ADIM` · `FARKLI-ADIM` · `ÖLCÜLEMEZ`
- **Score.** 2 substantive predictions: 1 held, 1 failed (`F-V61-1a`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_v61_ladder_2026-09-11.md` | `4eeddebf617d0504a90d28cf0a126dd4bfc55600904d10a04387bb20c9f3c191` | 2026-09-11T17:25Z | 2026-09-12T09:39Z |
| `prediction_v61_2026-09-11.md` | `68e8e606b517370e26e2f1793b370b42ba86229126d53d1837cea3496436bcbc` | 2026-09-11T17:27Z | 2026-09-14T10:55Z |
| `prerun_notice_v61_2026-09-11.md` | `f005a03f80dee9a089b9075aae61d91f5d5d23f66afc77b38654c9217669b2af` | 2026-09-11T17:26Z | — |

## Re-scoring person-swapped responses with two public reward models

Internal name: `V61-PUANLAMA`

- **Bar.** Per reward model, Δ = score(person-free rewrite) − score(original): CEZALANDIRIYOR if Δ > 0, CI excludes zero and outside the placebo band; `ÖDÜLLENDIRIYOR` if Δ < 0 likewise; SIFIR if the CI covers zero with adequate MDE; else `ÖLCÜLEMEZ`; `AYRISTI` if scorers disagree.
- **Outcome names.** `CEZALANDIRIYOR` · `ÖDÜLLENDIRIYOR` · `SIFIR` · `ÖLCÜLEMEZ` · `AYRISTI`
- **Score.** 2 substantive predictions: 0 held, 1 failed (`F-V61-2a`), 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_v61_scoring_2026-09-11.md` | `37939182ab7442430f4fc74c6a780a0005bd08adf33fe311824e387f22a0decd` | 2026-09-11T17:25Z | 2026-09-12T09:39Z |
| `prediction_v61_2026-09-11.md` | `68e8e606b517370e26e2f1793b370b42ba86229126d53d1837cea3496436bcbc` | 2026-09-11T17:27Z | 2026-09-14T10:55Z |

## The person dial (four arms of 7,537 preference pairs selected on person marking)

Internal name: `KISI KADRANI`

- **Bar.** An arm's second-person shift passes only if |Δ| ≥ 1.1 per 1k tokens, its CI excludes zero, and it beats the matched placebo, read primarily against the random-label arm; `KADRAN-IKI-YÖN` needs the down arm ≤ −1.1 and the up arm ≥ +1.1.
- **Outcome names.** `KADRAN-IKI-YÖN` · `KADRAN-TEK-YÖN` · `DOZ-ALTI` · `DOZ-BAGIMLI` · `KATMAN-AYRIK` · `KATMAN-BIRLIKTE` · `KUVVET-KADRAN` · `KUVVET-DOZ-ALTI`
- **Score.** 9 substantive predictions: 0 held, 4 failed (`Q5`, `Q7`, `Q8`, `Q9`), 5 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_person_dial_2026-09-03.md` | `8c59de6ec0dae875b9c3a74cd902fa9383aebb307388e117de8ef8ada7fba824` | 2026-09-03T20:15Z | 2026-09-04T06:38Z |
| `prediction_person_dial_2026-09-03.md` | `1612a89322d4d413496d0abcd2b4744a302d944d734bee0c19c3e222463860d3` | 2026-09-03T20:19Z | — |

## A third targeted run on minimal pairs built by exchanging a single determiner

Internal name: `KK-2`

- **Bar.** Minimal-pair arms pass only if |Δ| ≥ 1.1 per 1k, the prompt-clustered CI excludes zero and the matched placebo is beaten; 1.1–1.29 without CI separation is `GECMEZ-MDE-ALTI`; if the dial passes, force staying in place gives `KATMAN-AYRIK` and force moving with it `KATMAN-BIRLIKTE`.
- **Outcome names.** `ÖLCÜLEMEZ-DOZ` · `KATMAN-AYRIK` · `KATMAN-BIRLIKTE` · `DOZ-BAGIMLI`
- **Score.** 9 substantive predictions: 2 held, 4 failed (`Q3`, `Q5`, `Q8`, `Q9`), 3 not scored; operational: `KAPI-1 (D-SHELF)` not scored, `KAPI-2 (adim sayisi)` not scored, `KAPI-3 (gözcü ihlali)` not scored, `KAPI-4 (boy serhi)` not scored, `KAPI-5 (MP_RASTGELE bari)` not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_person_dial_minimal_pair_2026-09-04.md` | `db69579eadcb3621a45e4b8e4459c94233758bd1777d4d6ec98ed09292f698ec` | 2026-09-04T13:10Z | 2026-09-04T22:52Z |
| `prediction_person_dial_minimal_pair_2026-09-04.md` | `5550542c5dde2be97540bfc0230d9e5ba7dd656317f8f42e6afea0029f13f53c` | 2026-09-04T22:58Z | — |

## Pairs without person content

Internal name: `KK-3F`

- **Bar.** `FARKSIZ-CEKIYOR`: second-person shift vs SFT with CI wholly below zero, above its placebo p95, and ≤ −0.5612 (half the neutral arm's −1.1224); `FARKSIZ-CEKMIYOR`: that CI not wholly negative, arm-minus-neutral CI above zero, shift > −0.5612; otherwise `ÖLCÜLEMEZ-ARADA`.
- **Outcome names.** `FARKSIZ-CEKIYOR` · `FARKSIZ-CEKMIYOR` · `ÖLCÜLEMEZ-ARADA` · `RAF` · `ALET-KAYDI` · `KOL-KURULAMADI` · `KOSULMADI`
- **Score.** 3 substantive predictions: 0 held, 3 failed (`F-KK3F-1`, `F-KK3F-2`, `F-KK3F-3`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_data_tilt_zero_difference_2026-09-13.md` | `a8775b80595a2dd157d9730f4f094a0324347ed0953f4dc1ff29acf15bfea870` | 2026-09-13T22:37Z | — |
| `prediction_data_tilt_zero_difference_2026-09-13.md` | `3a155ba889fac9ce09bb65ee74b8027dd37535ea34f57b22df5ab2aa9f12f960` | 2026-09-13T22:39Z | — |

## The reference-free objective: released DPO and SimPO checkpoints from one supervised base

Internal name: `SimPO`

- **Bar.** Per ladder and protocol, first match wins: ZAYIF if |ΔM1| (endpoint minus shared SFT) < MDE = 1.645 × permutation null-sd; AYNI-ISARET-SimPO if ΔM1 < 0 and its CI excludes zero; TERS if ΔM1 > 0 likewise; else `ÖLCÜLEMEZ`; size versus DPO by CI overlap.
- **Outcome names.** `ZAYIF` · `AYNI-ISARET-SimPO` · `TERS` · `ÖLCÜLEMEZ` · `AYIRT-EDILEMEZ` · `SimPO-DAHA-KÜCÜK` · `SimPO-DAHA-BÜYÜK`
- **Score.** 3 substantive predictions: 1 held, 1 failed (`F-SIMPO-2`), 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_simpo_2026-09-10.md` | `a81b766b6e57876127f5028628079524b755393746d6279e61ad809488aad5ca` | 2026-09-09T13:07Z | — |
| `prediction_simpo_2026-09-10.md` | `109778c1707fd3b96deb17d072dfc8d10de652712d0e1acdc9943bcf02cbbc17` | 2026-09-09T13:09Z | 2026-09-09T13:11Z |

## The scorer fit: a joint fit with first-person density and two surface controls

Internal name: `YARGIC-IMZASI-2`

- **Bar.** In a joint fit, the bar is |β_M1| ≥ 0.05 with CI excluding zero and β_M1 < 0; bar met with margin |β_M1|/max(|β_modal|, |β_hedge|) ≥ 1.5 is `ÖZGÜL-IMZA`, bar met with margin < 1.5 is `GENEL-ÜSLUP-CEZASI`, otherwise `IMZA-YOK`.
- **Outcome names.** `ÖZGÜL-IMZA` · `GENEL-ÜSLUP-CEZASI` · `IMZA-YOK`
- **Score.** 6 substantive predictions: 3 held, 3 failed (`Q2`, `Q3`, `Q5`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_judge_signature_2_2026-09-03.md` | `56fc3c769d38657385934002c1ca7d1866af4b34d875d43289bdb2d96ea8a865` | 2026-09-03T17:36Z | — |
| `prediction_judge_signature_2_2026-09-03.md` | `324c100a7c86910d01a695bfbae2043f5bde4ca93f27b982ab7983dc6c9cff5e` | 2026-09-03T17:36Z | — |

## The mechanical form counters (bare imperative, second-person obligation, hedge, interrogative directive, hortative)

Internal name: `FORM-SAYIM`

- **Bar.** A contrast cell is `DÜSER` if the aligned-minus-base pooled per-sentence rate is negative and frac2 ≤ 0.05 (two-sided empirical percentile against a K=2000 cluster label-swap null), `DÜSMEZ` otherwise, `ÖLCÜLEMEZ` if the counter failed validation; R2 cells read the placebo contrast first.
- **Outcome names.** `DÜSER` · `DÜSMEZ` · `ÖLCÜLEMEZ` · `PROTOKOLE-BAGIMLI` · `IGNE-FORMDA-GÖRÜNÜR` · `IGNE-FORMDA-GÖRÜNMEZ` · `PLASEBO-KIRLENDI`
- **Score.** 12 substantive predictions: 4 held, 8 failed (`BF-1`, `BF-2`, `BF-3`, `BF-6`, `PF-2`, `PF-3`, `PF-4`, `PF-6`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_form_count_2026-08-28.md` | `356dc0111616c346a254c39366cc1488c9e46306a6985ad74e7961a9eb942ee9` | 2026-08-28T09:18Z | — |
| `prediction_form.md` | `f1c0ba2ec7054cce84e2ca79cb9b5a441c8d794aa8dde93ed18b2070e522dc70` | 2026-08-28T09:18Z | — |
| `prediction_form_executor.md` | `9e06dc491f6213baa9fa11c07274de6e23e29a20d3fad9972c73956fce8e8b27` | 2026-08-28T09:18Z | — |

## The reading cascade: identical text scored under a real and a matched pseudo-neutral context

Internal name: `F5-16`

- **Bar.** A family channel separates at two-sided empirical frac2 ≤ 0.05 with ≥30 clusters; the 16-family panel is YAYGIN if ≥`8/16` separate positively, ORTA at shares 0.25–0.50, SEYREK below 0.25, YOK at 0, `TERS-SINIF-VAR` if ≥`3/16` separate negatively, `KAPI/ÖLCÜLEMEZ` under 8 measurable.
- **Outcome names.** `YAYGIN` · `ORTA` · `SEYREK` · `YOK` · `TERS-SINIF-VAR` · `KAPI/ÖLCÜLEMEZ`
- **Score.** 6 substantive predictions: 5 held, 1 failed (`BF5-2`), 0 not scored; operational: `BF5-7` failed.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_shadow_reading_16_2026-08-25.md` | `a7df48a079cdb6302750bc6645b911bc43632d5a3c0dbf0ed1b363aebdf257ec` | 2026-08-25T16:28Z | — |
| `extra_prereg_shadow_reading_16_panel_2026-08-25.md` | `e74365244feec2a1b89e4aabedf1e93afacae3b40482a769e51acde9c1321b87` | 2026-08-25T21:35Z | 2026-08-25T23:50Z |
| `prediction_f5_2026-08-25.md` | `18660a4e5c09ebac0eb10a622ed7969f4684f9a619a9ecdd7e9bc47f7e1275f8` | 2026-08-25T21:37Z | — |

## Injecting the pre-specified weight-space dominance direction

Internal name: `D2V2-N`

- **Bar.** Per arm, the SFT→DPO shift along the dominance-gradient axis counts only if frac_yön ≤ 0.05 against a K=200 anisotropy-matched direction null and no placebo control matches it; the negative-control arm moving too gives panel `CETVEL-AYIRT-ETMIYOR`, dominance arm alone `AYRIM-VAR`.
- **Outcome names.** `KAPI/ÖLCÜLEMEZ` · `IZ-YOK` · `D-EKSENINDE` · `D-DISINDA` · `IKISI-DE` · `AYRIM-VAR` · `CETVEL-AYIRT-ETMIYOR` · `IKISI-DE-SESSIZ` · `TERS` · `EGRILIK-DÜSTÜ` · `EGRILIK-ARTTI` · `EGRILIK-FARKI-YOK`
- **Score.** 6 substantive predictions: 2 held, 2 failed (`BDN-4`, `BDN-5`), 2 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_steering_yon_null_2026-08-23.md` | `3d9f31b30bbc217d1577a421e23d9c37627d2d3cf27c06a824562ce31704aa89` | 2026-08-23T10:47Z | 2026-08-23T14:32Z |
| `prediction_steering_2026-08-23.md` | `4f0b4225c5b41556e3200706ebc4f8cf767f8fda2674d1303e3d968fe8b25270` | 2026-08-23T10:48Z | — |

## Registration `R2`

Internal name: `R2`

- **Bar.** Per family, `GERI-GELIR` if the steered arm's blind-judge dominance share is two-sidedly separated above placebo and rises monotonically over three KL doses, `KISMI` if separated but not monotone, `GERI-GELMEZ` if not separated, BOZULUR if mean retries steered/unsteered exceed 1.50; panel `KIP-YÖNÜ-TASINIR` at ≥`4/6` `GERI-GELIR`.
- **Outcome names.** `GERI-GELIR` · `KISMI` · `GERI-GELMEZ` · `BOZULUR` · `KAPI/ÖLCÜLEMEZ` · `KIP-YÖNÜ-TASINIR` · `TASINMAZ` · `AILEYE-GÖRE`
- **Score.** 4 substantive predictions: 1 held, 1 failed (`BR2-1`), 2 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_restorasyon2_2026-08-26.md` | `ec9e1f4352c2da5df404b0a3b17fe7610666f4ca81697f76cf8ae73866a39fae` | 2026-08-26T12:43Z | 2026-08-26T20:22Z |
| `prereg_r2_faz2_package_2026-08-27.md` | `8e20a431d18580b08e96c43bfc9e42865e5a7d011777367107b84516480747d1` | 2026-08-27T12:06Z | 2026-08-27T13:42Z |
| `prediction_r2.md` | `6b9611914ce6568ae59ff38cc3c5e3f00718df5f33f63d7a4026af54e70c1dd0` | 2026-08-26T14:48Z | — |

## The grammatical mood swap: rule-based imperative–request minimal pairs under a short and a long frame

Internal name: `E-YALINLIK`

- **Bar.** Per family, on aligned-minus-base per-token NLL with a verb-clustered 95% bootstrap (K=2000): `KIP-TASIR` if both mood contrasts separate and neither length contrast does, `YALINLIK-TASIR` if the reverse, `IKISI-DE` if all four separate, KAPI if none; `BASKIN-RICA-VURUR` if the dominant request separates toward the imperative.
- **Outcome names.** `KIP-TASIR` · `YALINLIK-TASIR` · `IKISI-DE` · `KAPI` · `BASKIN-RICA-VURUR`
- **Score.** 3 substantive predictions: 1 held, 1 failed (`BE-2`), 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_plainness_2026-08-27.md` | `f26cc044e4203ebd281cce69510829e416b7555ce1bc2229e92c361b1ad235c3` | 2026-08-27T21:23Z | 2026-08-28T11:16Z |
| `prediction_e.md` | `08f36252aec0c143d6103c72e04695ed9d8820a5d4644cc694a48a2514cd8e8e` | 2026-08-27T21:23Z | — |
| `prerun_notice_plainness_2026-08-27.md` | `b60e10fc97a8225a22c98f06b55328ac032a107ed2d0fb51d766fa5ffe6853b9` | 2026-08-27T21:23Z | — |

## Registration `E799`

Internal name: `E799`

- **Bar.** Command-minus-neutral mean dominance separates at two-sided frac2 ≤ 0.05 against a K=200 row-level context-label shuffle null; panel `REFLEKS-GENEL` if ≥`4/6` measurable cells are `REFLEKS-DÜSER` and none rises, `KIP-ÖZGÜLLÜGÜ-GENEL` if ≥`4/6` cells separate for command while declaration stays silent.
- **Outcome names.** `REFLEKS-DÜSER` · `REFLEKS-YÜKSELIR` · `REFLEKS-SESSIZ` · `KAPI/ÖLCÜLEMEZ` · `REFLEKS-GENEL` · `REFLEKS-TERS` · `REFLEKS-CATISIK` · `REFLEKS-DAGINIK` · `REFLEKS-YOK` · `KIP-ÖZGÜL` · `KIP-GENEL` · `KIP-TERS` · `KIP-SESSIZ` · `KIP-ÖZGÜLLÜGÜ-GENEL` · `KIP-ÖZGÜLLÜGÜ-YOK` · `KIP-ÖZGÜLLÜGÜ-DAGINIK`
- **Score.** 9 substantive predictions: 0 held, 0 failed, 9 not scored; operational: `PB-1` held, `PB-5` failed.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `kutu_muhru_kpe799_2026-08-25.md` | `24f30991634961b177743686a2be71ec4ec882fd2e6ffd32e598627245cc8a31` | 2026-08-25T07:01Z | — |
| `prereg_shadow_refleks_2026-08-25.md` | `543e0f1a63536cae7b60efb94b227c32aa31771f029a3dc31260aac84babb550` | 2026-08-25T06:40Z | — |
| `prediction_overnight2_2026-08-24.md` | `164b58f6c1c39bdbbe73e5057399044ca4c1fcc92dfca8bfa8ccc63b43a6b835` | 2026-08-24T20:03Z | — |

## Undoing the preference delta by interpolation

Internal name: `D2`

- **Bar.** Carried unchanged from `D-DPO-DELTA`: over five locked α points, `SILME-TEK-DELTADA` needs |Spearman ρ| = 1 and base-closeness within 25% of the DPO gap; `DAGINIK` if the Rademacher sign placebo (K=20) satisfies both in over 0.05 of draws.
- **Outcome names.** `SILME-TEK-DELTADA` · `DAGINIK` · `KAPI` · `DELTA-SILER` · `DELTA-KISMEN` · `DELTA-SILMEZ` · `PANEL-KAPI`
- **Score.** 2 substantive predictions: 0 held, 2 failed (`BD2-1`, `BD2-2`), 0 not scored; operational: `BD2-3` held.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_d2_delta_genisletme_2026-08-28.md` | `7bf7ec897644b538444c895d7e79343c12df55b019fc576db4ea5ba72288c8f9` | 2026-08-28T06:48Z | 2026-08-28T11:16Z |
| `prereg_d_dpo_delta_2026-08-27.md` | `d8e05f5fcfbb5b3109482b069a5d3fd30a735f1e8ab6c2122f70ead131021960` | 2026-08-27T21:16Z | 2026-08-27T22:20Z |
| `prediction_d2.md` | `e73817289e8d1af146f97b609baa9fd12e6bc7fb9e88cfdc855b5efa4d257a30` | 2026-08-28T07:04Z | — |
| `prerun_notice_d_dpo_delta_2026-08-27.md` | `bc4776cb9be781bece97bb8de9c1d7edb7871fb171f74ad4159ad64511f9f594` | 2026-08-27T21:16Z | — |

## The behavioural anchor: the supervised weights stepped either way along the dominance direction at a fixed KL budget

Internal name: `C3`

- **Bar.** Carried from the C2 body-arm seal: `CETVEL-D`'`YI-GÖSTERIR` if ΔT(+ε) < 0 < ΔT(−ε) with both branches separated at two-sided frac2 ≤ 0.05, over 98 pairs at KL budget 0.0875; this seal only raised the placebo count from 20 to 60.
- **Outcome names.** `CETVEL-D'YI-GÖSTERIR` · `TERS-GÖSTERIR` · `GÖSTERMEZ` · `KAPI` · `CAPA-DAVRANISSAL` · `CAPA-TERS` · `CAPA-KURULAMADI-DAVRANISSAL` · `PANEL-KAPI`
- **Score.** 3 substantive predictions: 1 held, 2 failed (`BC3-1`, `BC3-2`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_c3_buyutme_2026-08-28.md` | `6d6e66cd7cae3167eb7210c141b636fe28b3cc48fe8b8bb22f03682898b96cde` | 2026-08-28T06:49Z | 2026-08-28T06:56Z |
| `prereg_conflict_probe_body_arm_2026-08-28.md` | `7ae44556fdbd8ec2b610be1f4219d6ae9aa4ce79c2b809aa139d03ee59bf9610` | 2026-08-28T00:06Z | — |
| `prediction_c3.md` | `ad7f295323de1eda37855d5b853ff67f5f8c3bf795cba7080355b5eeef6c04f7` | 2026-08-28T07:04Z | — |
| `prerun_notice_conflict_probe_body_arm_2026-08-28.md` | `8814ed15e32933241f021a8b0a5de562a2e90a090c1c4cb0304c3a484f70e529` | 2026-08-28T00:06Z | — |
| `prerun_notice_c3_buyutme_2026-08-28.md` | `08c26c46599e207986580dcafc8e9fdfd2489a1413d5d085412823752a4bfd8d` | 2026-08-28T06:49Z | — |

## Rung-by-rung reading with the embedding estimator

Internal name: `P7-ÖLCEK2`

- **Bar.** With at least 3 pairs passing the positive control: `TÜM-AILELER-SILER` if every passing pair's ΔU_DOM is CI-separated positive, `DOM-HER-YERDE-ÖLÜ-TEYIT` if dominance never accumulated, else `IMZA-AILEYE-GÖRE`; `ARO-DOKUNULMAMIS` if no pair's ΔU_ARO separates, otherwise two-way or one-way ARO names.
- **Outcome names.** `TÜM-AILELER-SILER` · `IMZA-AILEYE-GÖRE` · `DOM-HER-YERDE-ÖLÜ-TEYIT` · `KAPI/ÖLCÜLEMEZ` · `ARO-DOKUNULMAMIS` · `ARO-IKI-YÖNLÜ` · `ARO-TEK-YÖNLÜ-POZ` · `ARO-TEK-YÖNLÜ-NEG` · `IC-MAKSIMUM` · `UC-MAKSIMUM` · `ÖLCÜLEMEZ`
- **Score.** 5 substantive predictions: 1 held, 4 failed (`B1`, `B2`, `B3`, `B5`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_scale2_2026-08-21.md` | `8ebc6d9d2b76aa4cc18960da805662e9b804904f9f506562b90312641b796db8` | 2026-08-21T22:04Z | 2026-08-22T23:05Z |
| `prediction_overnight_2026-08-21.md` | `873c994183e65e04e9448af2a1d57b5060138ed49e50df0d218436d471b8923a` | 2026-08-21T22:05Z | 2026-08-21T22:07Z |

## Registration `RLVR OLMo-2-7B`

Internal name: `RLVR OLMo-2-7B`

- **Bar.** Over six consecutive RLVR sub-steps: `KAPI/ÖLCÜLEMEZ` if fewer than 3 pairs pass the positive control, `ARO-DOKUNULMAMIS` if no cell's ΔU_ARO interval excludes zero, `ARO-IKI-YÖNLÜ` if both signs separate, otherwise a one-way ARO name.
- **Outcome names.** `KAPI/ÖLCÜLEMEZ` · `ARO-DOKUNULMAMIS` · `ARO-IKI-YÖNLÜ` · `ARO-TEK-YÖNLÜ-POZ` · `ARO-TEK-YÖNLÜ-NEG` · `IC-MAKSIMUM` · `UC-MAKSIMUM` · `ÖLCÜLEMEZ`
- **Score.** 2 substantive predictions: 2 held, 0 failed, 0 not scored; operational: `BR-4` held, `BR-5` held; base-rate: `BR-3` failed.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_rlvr_olmo2_7b_2026-08-23.md` | `870d911f4c25617a151f1b6086793d2e9af388e7fb2b248f578add49e39dee16` | 2026-08-23T11:04Z | — |
| `prediction_rlvr_executor_2026-08-23.md` | `0f7580582b5a9b890c7505803c52ca6a8b403cb6d1cb4ff2f4e51cd9c54dd664` | 2026-08-23T11:04Z | — |

## Registration `ACI`

Internal name: `ACI`

- **Bar.** Per ladder, from the cosine of SFT and preference weight deltas: `ÖLCÜLEMEZ` if loading fails or under 50 shared tensors, `FARKLI-EKSEN` if |COS| < 0.01, `AYNI-EKSEN` if |COS| ≥ 0.10 and tensor-permutation excess z ≥ 1.645, else `ARA-BANT`; panel `ÖLCÜLEMEZ` under four ladders.
- **Outcome names.** `ÖLCÜLEMEZ` · `FARKLI-EKSEN` · `AYNI-EKSEN` · `ARA-BANT` · `AYRISMIYOR` · `RECETEYE GÖRE AYRISIYOR`
- **Score.** 7 substantive predictions: 6 held, 0 failed, 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_task_vector_angle_2026-09-12.md` | `45274413c3a91a161178eac04f909e21d612492c3f8c3fe3cc63182004c4c2c7` | 2026-09-12T21:54Z | — |
| `prediction_task_vector_angle_2026-09-12.md` | `66adb1a82c73b9550865e2aad8a490338401a0baea30df5d819b9ead51ab5a8b` | 2026-09-12T21:57Z | 2026-09-12T23:45Z |
| `prerun_notice_task_vector_angle_2026-09-12.md` | `d1f6a152cbefd13dfbe5eaa8eb7fdc6c7d02db6a66b4e7990906ed7f9e5f6b10` | 2026-09-12T21:55Z | — |

## Registration `YETENEK`

Internal name: `YETENEK`

- **Bar.** Recovery = [M1(instructed arm) − M1(aligned)] / [M1(base) − M1(aligned)] per family; YETENEK `SAGLAM` if median recovery ≥ 0.50, YETENEK YOK if ≤ 0, `KISMÎ` in between; `ÖLCÜLEMEZ` first if under 12 measurable families or over 4 aligned cells with per-text distinct-4 < 0.60.
- **Outcome names.** `ÖLCÜLEMEZ` · `YETENEK YOK` · `YETENEK SAGLAM` · `KISMÎ`
- **Score.** 4 substantive predictions: 4 held, 0 failed, 0 not scored; operational: `F-YET-5` held.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_capability_2026-09-12.md` | `923cddfc0217d9792918cbfc3c51d8545f9495ad3152c86d104e299fbec5a7ca` | 2026-09-13T00:06Z | — |
| `prediction_capability_2026-09-12.md` | `50d826f246568a971e5c2208a7a35385939859785e6ddc433a57d2c7652eceb3` | 2026-09-13T00:08Z | 2026-09-13T09:00Z |

## Registration `PREREG-9b`

Internal name: `PREREG-9b`

- **Bar.** Arm B is re-read on a length-matched first-word window; families with a negative, CI-excluding-zero second-person shift are counted (`GENIS` ≥12, YEREL 4–11, `DESEN-YOK` ≤3); if full-text and window counts differ by 2 or more families, the window reading becomes primary for arm B.
- **Outcome names.** `GENIS` · `YEREL` · `DESEN-YOK` · `TAVAN-SINIRLI`
- **Score.** 1 substantive prediction: 1 held, 0 failed, 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_9b_uzunluk_matched_2026-09-01.md` | `e47a4972b75f8584992258dbcae2469275a1a406c1c1e6b00fae243061bedcd3` | 2026-08-31T20:32Z | — |

## Registration `D-DPO-DELTA`

Internal name: `D-DPO-DELTA`

- **Bar.** Over five locked interpolation points, `SILME-TEK-DELTADA` needs |Spearman ρ(α, T)| = 1 for the imperative-minus-request NLL gap T and |T(−0.5) − T_base| ≤ 0.25·|T(1) − T_base|; `DAGINIK` if either fails or the sign placebo does the same; panel `DELTA-SILER` needs ≥2 cells.
- **Outcome names.** `SILME-TEK-DELTADA` · `DAGINIK` · `KAPI` · `DELTA-SILER` · `DELTA-KISMEN` · `DELTA-SILMEZ` · `PANEL-KAPI`
- **Score.** 2 substantive predictions: 2 held, 0 failed, 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_d_dpo_delta_2026-08-27.md` | `d8e05f5fcfbb5b3109482b069a5d3fd30a735f1e8ab6c2122f70ead131021960` | 2026-08-27T21:16Z | 2026-08-27T22:20Z |
| `prediction_d.md` | `a42d7a7a528e31f6016353a4d113d735aacab4d04bcda68180b82b9c0424588f` | 2026-08-27T21:16Z | — |

## Registration `P-ÖLCEK3`

Internal name: `P-ÖLCEK3`

- **Bar.** `ÖLCEK-2` cascades applied to Tülu-3-8B's four cells, but a cell's ΔU_DOM or ΔU_ARO counts as separated only if its paired cluster-bootstrap interval, shifted by the cell's own placebo-null mean, excludes zero; 1.0–1.645 null-sd is CONTINGENT and not positive.
- **Outcome names.** `TÜM-AILELER-SILER` · `IMZA-AILEYE-GÖRE` · `DOM-HER-YERDE-ÖLÜ-TEYIT` · `KAPI/ÖLCÜLEMEZ` · `ARO-DOKUNULMAMIS` · `ARO-IKI-YÖNLÜ` · `ARO-TEK-YÖNLÜ-POZ` · `ARO-TEK-YÖNLÜ-NEG` · `IC-MAKSIMUM` · `UC-MAKSIMUM` · `ÖLCÜLEMEZ` · `YÖN-UYUMLU` · `YÖN-CELISIR` · `UCTAN-UCA-ÖLCÜLEMEZ`
- **Score.** 5 substantive predictions: 5 held, 0 failed, 0 not scored; base-rate: `SAHIP-HIPOTEZI` failed.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_scale3_tulu3_2026-08-22.md` | `2d0238eae08474f7a06c4ced1527d6dbace597c97e1b96c108c75ffc34b1f879` | 2026-08-22T23:00Z | 2026-08-23T00:36Z |
| `prediction_scale3_2026-08-22.md` | `d990005aae0740866b8753db31c094054200e9778a344ce3208c4be7925fb6ac` | 2026-08-22T23:01Z | — |

## Registration `MERDIVEN-TEMPLATE`

Internal name: `MERDIVEN-TEMPLATE`

- **Bar.** Each ladder's SFT→DPO step compares its ΔU_DOM verdict name under bare prompting with the name under chat templates (`AYNI-AD` or `AD-DEGISTI`); panel `DPO-BASAMAGI-GÜRBÜZ` if at least 2 measurable ladders keep the same name, else `DPO-BASAMAGI-SABLONA-BAGIMLI`; fewer than 2 measurable is KAPI.
- **Outcome names.** `AYNI-AD` · `AD-DEGISTI` · `KAPI` · `DPO-BASAMAGI-GÜRBÜZ` · `DPO-BASAMAGI-SABLONA-BAGIMLI`
- **Score.** 4 substantive predictions: 4 held, 0 failed, 0 not scored; operational: `BM-5` held.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_ladder_template_2026-08-27.md` | `f4d6521103208da997ee7600d4c9a41ecb50c5fd553d267dcdd208112c1ab4ff` | 2026-08-27T05:59Z | 2026-08-27T07:50Z |
| `prediction_ladder.md` | `9b6894008646af141eda0724df79a8c58ec1d37f63a04871936aa6590e45d4d2` | 2026-08-27T06:20Z | — |

## Registration `M-SISTEM`

Internal name: `M-SISTEM`

- **Bar.** Per family, diff = (instruct − base second-person density with a neutral system line) − (same with an empty system role): `SATIR-ETKISIZ` if |diff| < MDE in ≥12 families, `SATIR-ETKILI` if its CI excludes zero in ≥12, else `KARISIK`; under 12 measured is `ÖLCÜLEMEZ`.
- **Outcome names.** `SATIR-ETKISIZ` · `SATIR-ETKILI` · `KARISIK` · `ÖLCÜLEMEZ (payda<12)`
- **Score.** 1 substantive prediction: 0 held, 0 failed, 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_system_prompt_2026-09-10.md` | `a8df075e7b25785e1798150526a46cb7326681e8bc9136c1230b45000944f330` | 2026-09-09T17:27Z | 2026-09-10T20:23Z |
| `prediction_system_prompt_2026-09-10.md` | `f541197a5beb0029b53058338f1072c73c9d60724bb2fa8f104e19de8d5d95ba` | 2026-09-09T17:30Z | — |

## Registration `URIAL-2`

Internal name: `URIAL-2`

- **Bar.** ΔM1_task = M1(instruct model with its own chat template) − M1(base model with the URIAL prefix); |ΔM1_task| < MDE (1.645 × matched-placebo null-sd) is `BICIM-ÖZELLIGI`, ≥ MDE downward is `AGIRLIK-ÖZELLIGI`, ≥ MDE upward is TERS; fewer than 8 legs is `ÖLCÜLEMEZ-KAPSAM`.
- **Outcome names.** `BICIM-ÖZELLIGI` · `AGIRLIK-ÖZELLIGI` · `TERS` · `ÖLCÜLEMEZ-KAPSAM`
- **Score.** 1 substantive prediction: 0 held, 0 failed, 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_urial_2_2026-09-07.md` | `b4bbecb539d28e59d320622397b8eb773b396dd1327058ad4adc0da1539704e3` | 2026-09-07T15:02Z | 2026-09-10T09:40Z |
| `prediction_urial_2026-09-07.md` | `46d02f04d8e18803b41745cc23c6038dfd7b546b18adbcaf550a408b44ff2268` | 2026-09-07T14:29Z | — |

## Registration `KK-3`

Internal name: `KK-3`

- **Bar.** All three required: |ΔM1(`NÖTR`) − ΔM1(`ÜC-KAT`)| ≥ 1.1 per 1k tokens, a CI-disjoint prompt-clustered bootstrap interval, and exceeding the matched placebo; in the 1.1–1.29 band without CI separation the result is `GECMEZ-MDE-ALTI`.
- **Outcome names.** `VERI-EGIMI-TASIYOR` · `VERI-EGIMI-TASIMIYOR` · `ÖLCÜLEMEZ-GÜC` · `DOZ-BAGIMLI` · `GECMEZ-MDE-ALTI`
- **Score.** 2 substantive predictions: 0 held, 0 failed, 2 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `draft_prereg_data_tilt_2026-09-05.md` | `8021a36abbf695ca5556caf58fee87abce04ef92960fa576e9c7f74bd65b861b` | 2026-09-05T11:35Z | 2026-09-06T08:31Z |

## Registration `KGK v3`

Internal name: `KGK v3`

- **Bar.** Per judge arm, the preference difference (Your-swap minus Their-swap control) must be ≥ 0.07 with a pair-bootstrap 95% CI excluding zero to count as separated; both arms → `KGK-TUTTU`, one arm → `KGK-YARGIC-BAGIMLI`, neither → `KGK-NULL`.
- **Outcome names.** `AYRISTI` · `KGK-TUTTU` · `KGK-YARGIC-BAGIMLI` · `KGK-NULL`
- **Score.** 1 substantive prediction: 0 held, 0 failed, 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_person_restoration_v3_2026-09-03.md` | `e31034353a978dcf8af58fb2094ea50ca1ef83fd6e56539113111c47989ceb6c` | 2026-09-03T11:12Z | 2026-09-03T12:55Z |
| `prerun_notice_person_restoration_v3_2026-09-03.md` | `14ab8676e81ecd5b0e2713c823b47da67518a04ba02c653fe81fdd8902f6796d` | 2026-09-03T11:14Z | — |

## Registration `PREREG-7`

Internal name: `PREREG-7`

- **Bar.** For primary contrast P1 (status-act rate, SFT→RL, first-window reading), `YARGIC-BAGIMSIZ` requires the log-odds ratio to be positive in both the Claude and Qwen judges and each to exceed its own LOR-scale MDE; otherwise `YARGIC-BAGIMLI`.
- **Outcome names.** `YARGIC-BAGIMSIZ` · `ISARET-UYUMLU` · `YARGIC-BAGIMLI`
- **Score.** 7 substantive predictions: 0 held, 0 failed, 7 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_7_draft_statu_teli_2026-08-31.md` | `ea22db7fa98d582a7ae7321d8804f95faa345eb088ed0855a9d18805b467812c` | 2026-08-31T12:03Z | 2026-08-31T13:42Z |

## Registration `M-KOL-TEMPLATE`

Internal name: `M-KOL-TEMPLATE`

- **Bar.** Under the Tülu-3 template, each person arm's Δ = M1(arm) − M1(RASTGELE) per 1k tokens passes only if |Δ| ≥ 1.1, CI-disjoint and above placebo; both person arms passing in their own direction → `KADRAN-SABLONDA-DA`, both |Δ| < MDE (1.29) → `KADRAN-SUREKLILIK`.
- **Outcome names.** `KADRAN-SABLONDA-DA` · `KADRAN-SUREKLILIK` · `KARISIK` · `ÖLCÜLEMEZ-DEJENERE` · `DOZ-BAGIMLI`
- **Score.** 2 substantive predictions: 0 held, 0 failed, 2 not scored.
- **Renamed in this release.** The outcome name `KADRAN-SUREKLILIK` was renamed in this release; the rename changes no rule, reading or score.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_arm_template_2026-09-10.md` | `47ce349b9d307ceca70a57ba5108099ebb8175815fcf67779b44958cce0cb1f3` | 2026-09-09T17:27Z | — |
| `prediction_arm_template_2026-09-10.md` | `18ca6bf4e8b1b146081e8b71077c832152c64a6fed17f9d7004aceb5d60a7ec4` | 2026-09-09T17:30Z | — |

## Registration `M-ESLI-KOL`

Internal name: `M-ESLI-KOL`

- **Bar.** Referenced to KISI_ASAGI's own measured |dB| = 6.89 per 1k, the length- and refusal-matched arm's |Δ| ≥ 3.44 (50%) is `ESLI-KOL-CEKIYOR`, < 1.72 (25%) is `ESLI-KOL-CEKMIYOR`, in between ARADA; if the matching acceptance band fails, `ESLEME-KURULAMADI`.
- **Outcome names.** `ESLI-KOL-CEKIYOR` · `ESLI-KOL-CEKMIYOR` · `ARADA` · `ESLEME-KURULAMADI`
- **Score.** 1 substantive prediction: 0 held, 0 failed, 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_matched_arm_2026-09-10.md` | `f73c1b98f0bfd895a250f571d721430ad6a8e1350744be94716aff5ccb44070c` | 2026-09-09T17:27Z | 2026-09-09T17:33Z |
| `prediction_matched_arm_2026-09-10.md` | `1d9262f8720be64ab4568c60d917620e85d54c6217b936d907d17d578da55d25` | 2026-09-09T17:30Z | — |

## Registration `C-DAVRANISSAL`

Internal name: `C-DAVRANISSAL`

- **Bar.** A branch is disjoint if frac2 = mean(|ΔT_placebo| ≥ |ΔT_real|) ≤ 0.05 over K = 20 direction placebos (two-sided, empirical); a cell is `CETVEL-D`'`YI-GÖSTERIR` if ΔT(+ε) < 0 < ΔT(−ε) with both branches disjoint; the panel needs ≥2 such cells and none reversed.
- **Outcome names.** `CETVEL-D'YI-GÖSTERIR` · `TERS-GÖSTERIR` · `GÖSTERMEZ` · `KAPI` · `CAPA-DAVRANISSAL` · `CAPA-TERS` · `CAPA-KURULAMADI-DAVRANISSAL` · `PANEL-KAPI`
- **Score.** 1 substantive prediction: 0 held, 0 failed, 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_davranissal_anchor_2026-08-27.md` | `9639e370cd275fcf084921f7c294d970a79879d93f60865ecf024840985f6545` | 2026-08-27T21:06Z | 2026-08-28T00:03Z |
| `prediction_c.md` | `e040548da47e0e6ff6041ed75fd63f89b5eefeb77229d6d281c13d869b39880e` | 2026-08-27T21:06Z | — |
| `prerun_notice_davranissal_anchor_2026-08-27.md` | `e3fbee3e42efd2aeb5ffd253b7f10173250410768a3e452b2244f8e68987b654` | 2026-08-27T21:06Z | — |

## Registration `PREREG-10`

Internal name: `PREREG-10`

- **Bar.** Per ladder (CI-disjoint = cluster-bootstrap CI excludes zero): SFT→DPO step disjoint and negative → `TERCIH-ADIMI-TASIR`; else base→SFT disjoint → `ASISTAN-EGITIMI`; neither but end-to-end disjoint → `DAGILMIS`; one summary name only if `3/3` ladders agree, otherwise `MERDIVENE-BAGIMLI`.
- **Outcome names.** `TERCIH-ADIMI-TASIR` · `ASISTAN-EGITIMI` · `DAGILMIS` · `MERDIVENE-BAGIMLI`
- **Score.** No prediction was registered.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_rung_table_2026-09-01.md` | `0522a5092d6c1e41a90da8eba55315e0cc27e67848405c2c41727e3ee9994e30` | 2026-09-01T10:57Z | — |

## Registration `GÖMME-AYAGI`

Internal name: `GÖMME-AYAGI`

- **Bar.** No bar or outcome name was fixed.
- **Outcome names.** none fixed
- **Score.** No prediction was registered.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_gomme_ayagi_2026-08-28.md` | `9b7f3e5954189bc7c7a8e9b32de257ab5b9f0b33472179162ea19c07e05fabc3` | 2026-08-28T21:27Z | — |

## Registration `PREREG-8`

Internal name: `PREREG-8`

- **Bar.** Each SFT-to-final-step difference (M1–M5 and the total-force control) is read by the two-sided empirical percentile frac(null ≥ |observed|) over 200 within-prompt step permutations, with no cutoff value written; a difference inside the null spread is `ÖLCÜLEMEZ`.
- **Outcome names.** `ÖLCÜLEMEZ`
- **Score.** 3 substantive predictions: 0 held, 1 failed (`B8a`), 2 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_8_addressee_silme_2026-09-01.md` | `af9c749bb6259553a15873dd4eff7687185e06dbec1327d7283f9726176d9480` | 2026-08-31T16:23Z | — |

## The first scorer fit, whose specificity margin set the later bar

Internal name: `YARGIC-IMZASI-1`

- **Bar.** Each coefficient (second- and first-person density predicting the UltraFeedback overall score, within prompt and model, length-controlled) passes if |β_std| ≥ 0.05 sd/sd, its prompt-clustered bootstrap CI excludes zero, and its sign is negative.
- **Outcome names.** `IMZA-VAR` · `KISMI` · `SAPTANDI-AMA-ÖNEMSIZ` · `ÖLCÜLEMEZ` · `TERS`
- **Score.** 6 substantive predictions: 3 held, 3 failed (`P2`, `P3`, `P4`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_judge_signature_2026-09-03.md` | `aca02dca49857660c0d56296b0e3d81f570e5b21e5c1b9c269905021cef6d102` | 2026-09-03T16:43Z | — |
| `prediction_judge_signature_2026-09-03.md` | `c169b5ab8afe48875091772c0ba350a93eb538e6c466fdcc4549a616c1f6dcf5` | 2026-09-03T16:44Z | — |

## Registration `KK3T2`

Internal name: `KK3T2`

- **Bar.** The `NÖTR` arm retrained with a second training seed is read against the original `NÖTR` arm (dT): CI excluding zero with |dT| ≥ 0.4164 (the measured `NÖTR` − TABAN difference) → `SEED-PAYI-ASAR`; CI excluding zero with smaller |dT| → `SEED-GÖRÜNÜR`; CI including zero → `SEED-GÖRÜNMEZ`.
- **Outcome names.** `SEED-PAYI-ASAR` · `SEED-GÖRÜNÜR` · `SEED-GÖRÜNMEZ` · `RAF` · `ALET-KAYDI` · `KOSULMADI`
- **Score.** 2 substantive predictions: 2 held, 0 failed, 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_data_tilt_neutral_seed2_2026-09-13.md` | `4bc98f7eeb80a2378bcabef41898845d1030356fa71470f72b1fdeb8319a388c` | 2026-09-13T23:43Z | — |
| `prediction_data_tilt_neutral_seed2_2026-09-13.md` | `5d8884b9297bc497de323d93099bda328956ce91224e7104991a9b520874c8d3` | 2026-09-13T23:44Z | — |

## Registration `KK3T3`

Internal name: `KK3T3`

- **Bar.** Sign bar, no numeric threshold: `ISARET-TUTTU` if the `NÖTR` arm's depersonalization sign (reading A, arm − SFT) is the same nonzero sign in all three training seeds, otherwise `ISARET-DÖNDÜ`; magnitude is printed only as a range, never pooled.
- **Outcome names.** `ISARET-TUTTU` · `ISARET-DÖNDÜ` · `RAF` · `ALET-KAYDI` · `KOSULMADI`
- **Score.** 2 substantive predictions: 2 held, 0 failed, 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_data_tilt_neutral_seed3_2026-09-15.md` | `8843e219504371e7cb333abcee5ace847d48368e5a7a9be4ac5d18842e58276e` | 2026-09-15T21:09Z | 2026-09-15T22:57Z |
| `draft_prereg_data_tilt_neutral_seed3_2026-09-15.md` | `d5ddbf8caae6552bd2495a795209f025a6f7230626ed7c599b130f1a69607808` | 2026-09-15T19:38Z | — |
| `prediction_data_tilt_neutral_seed3_2026-09-15.md` | `72df2eae4b786599f99621c2e55fbe694c9850c4aba14585a87e141c4622bb3f` | 2026-09-15T21:10Z | — |

## Registration `KK3TABAN`

Internal name: `KK3TABAN`

- **Bar.** Sign bar, no numeric threshold: `ISARET-TUTTU` if the natural-slope arm (KK3_TABAN) keeps seed 1's sign (downward versus SFT, reading A) in both new training seeds, otherwise `ISARET-DÖNDÜ`; magnitude is printed only as a [min, max] range.
- **Outcome names.** `ISARET-TUTTU` · `ISARET-DÖNDÜ` · `RAF` · `ALET-KAYDI` · `KOSULMADI`
- **Score.** 3 substantive predictions: 0 held, 0 failed, 3 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_data_tilt_natural_seeds_2_3_2026-09-16.md` | `fde3105ff499a6dca2f78e35a35b2dede22e32b0ee50d3e72ac27bd0d6087ba5` | 2026-09-16T12:19Z | 2026-09-16T12:28Z |
| `prediction_data_tilt_natural_seeds_2_3_2026-09-16.md` | `a7a731b5481ea300304387817ba05b587adfda26157d6f2548ddf573d4dc17a6` | 2026-09-16T12:20Z | — |

## Registration `KISI_SEED23`

Internal name: `KISI_SEED23`

- **Bar.** Sign bar, no numeric threshold: per person arm (KISI_ASAGI, KISI_YUKARI), `ISARET-TUTTU` if the displacement sign (dB, reading B versus RASTGELE) in both new training seeds matches seed 1 (exact zero does not count), else `ISARET-DÖNDÜ`; both arms holding → `KADRAN-TOHUMDA-TUTTU`.
- **Outcome names.** `ISARET-TUTTU` · `ISARET-DÖNDÜ` · `RAF` · `ALET-KAYDI` · `KOSULMADI` · `KADRAN-TOHUMDA-TUTTU` · `KADRAN-TOHUMDA-DÖNDÜ`
- **Score.** 3 substantive predictions: 3 held, 0 failed, 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_person_dial_seeds_2_3_2026-09-15.md` | `db049dde3a7e8be1ae880b3b8375f129bf66fef249b9a801d0d307d7d806d010` | 2026-09-15T16:13Z | — |
| `prediction_person_dial_seeds_2_3_2026-09-15.md` | `f6c837d5945a9f3be3e2454fdfff040d34e843a877faae23bac858048f553fd6` | 2026-09-15T16:15Z | — |

## Registration `SEED_UC_AILE`

Internal name: `SEED_UC_AILE`

- **Bar.** Sign bar, no numeric threshold: per family (Llama-3.`1-70B`, Qwen2.`5-72B`, Qwen2.`5-32B`), `ISARET-TUTTU` if all four new generation seeds keep the seed-1 ΔM1 sign (zero counts as different), else `ISARET-DÖNDÜ`; all three holding → `ÜCÜ-DE-TUTTU`, any flip → `EN-AZ-BIRI-DÖNDÜ`.
- **Outcome names.** `ISARET-TUTTU` · `ISARET-DÖNDÜ` · `RAF` · `KOSULMADI` · `ÜCÜ-DE-TUTTU` · `EN-AZ-BIRI-DÖNDÜ`
- **Score.** 2 substantive predictions: 2 held, 0 failed, 0 not scored; operational: `F-TUA-3` held.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_three_family_four_seed_2026-09-15.md` | `edd1fd48c34958195e953ac3753829c65751fee2faf0aec015a3fc7c9c907fbb` | 2026-09-15T20:56Z | — |
| `draft_prereg_three_family_four_seed_2026-09-15.md` | `7b06eb7bc4d784757be00e2f60daf721744f60095b2966556dd176efbc49cdd5` | 2026-09-15T19:38Z | — |
| `prediction_three_family_four_seed_2026-09-15.md` | `af14119fbc55e3bebd6b818f0413ad201bb1496a1f37bf749c7c062533f2f91a` | 2026-09-15T20:57Z | — |

## Registration `SEED_DENETIMI`

Internal name: `SEED_DENETIMI`

- **Bar.** Per published form-delta entry, |z| = |mean over four seeds − published single-seed value| / sd over four seeds: below 1.0 → `SEED-GÜRBÜZ`, 1.0–1.645 → `ÖLCÜLEMEZ` (contingent band), above 1.645 → `SEED-DUYARLI`.
- **Outcome names.** `SEED-GÜRBÜZ` · `ÖLCÜLEMEZ` · `SEED-DUYARLI`
- **Score.** No prediction was registered.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_seed_audit_2026-09-01.md` | `db6a5e0e7e6dba2d94265498861e9e4265fbaf408d4606f087f5941a5b7ab0ef` | 2026-08-31T20:36Z | — |

## Registration `YETENEK_SAHISSIZ`

Internal name: `YETENEK_SAHISSIZ`

- **Bar.** Median over 16 families of recovery = (M1 pronoun-free arm − M1 aligned)/(M1 base − M1 aligned): `ÖLCÜLEMEZ` if <12 measurable families or >4 degenerate cells; ≤0 YETENEK YOK; <0.50 `KISMÎ`; 0.50–1 `SAGLAM-ASMASIZ`; >1 `ASMA`; a copy slot compares D = 1.7181 − median with E = 0.5220.
- **Outcome names.** `ÖLCÜLEMEZ` · `YETENEK YOK` · `KISMÎ` · `SAGLAM-ASMASIZ` · `ASMA` · `KOPYADAN-BÜYÜK-DÜSÜS` · `KOPYADAN-BÜYÜK-ARTIS` · `KOPYA-ICINDE`
- **Score.** 3 substantive predictions: 3 held, 0 failed, 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_capability_pronoun_free_2026-09-14.md` | `c77b252e9e571ef113e373f49c1b1ec60b9894d25bb5b14628ae3fbd0f5849d3` | 2026-09-14T19:47Z | — |
| `prediction_capability_pronoun_free_2026-09-14.md` | `9798d73f4f9deae30d20f40d7dd4cde392e78f1dccc383e2453c66d54dc4d59b` | 2026-09-14T19:49Z | 2026-09-15T06:01Z |
| `prerun_notice_capability_pronoun_free_2026-09-14.md` | `0cff063602cff5d42f92776cabf941010c1aeaad50287ecc0f0f831e24d9d5ba` | 2026-09-14T19:48Z | — |

## The low-dose targeted preference arms (four DPO arms on form-selected pairs)

Internal name: `TERAZI (KOL-GECESI)`

- **Bar.** With ΔU_DOM = U(base) − U(arm) and MDE 0.00438, bet B1 passes only if ΔU_DOM(`EMIR`↑) > +0.00438 and ΔU_DOM(`EMIR`↓) < −0.00438; B2 passes only if all four arms have |ΔU_DOM| ≤ 0.00438; B3 is recorded when exactly one arm crosses.
- **Outcome names.** none fixed
- **Score.** 3 substantive predictions: 1 held, 1 failed (`B1`), 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_arm_gecesi_2026-08-30.md` | `833d5bc58237ca61c108fe3f06c9113c4584ab3cc9679954e11b19e8c3eff694` | 2026-08-30T19:50Z | — |
| `prereg_base_cizgisi_2026-08-30.md` | `de3c68a3991f2300dee6aaf5e8593ab58274f7a56efcf71b25f19daaafd58bbc` | 2026-08-30T19:49Z | — |
| `prerun_notice_three_arm_2026-08-29.md` | `8fa8cfacbdfd7297c9afe8ec301083e2abe3d5c9fbd98d2a455f8630b8703170` | 2026-08-29T20:22Z | 2026-08-30T11:34Z |

## Registration `MINI-DPO-8B`

Internal name: `MINI-DPO-8B`

- **Bar.** No acceptance bar (discovery class); the only pre-written rule is a VRAM gate: peak memory ≤ 85% of measured free memory (`SIGDI`) opens step 2, while 85–100% (SINIRDA) or above 100%/OOM (`SIGMADI`) does not.
- **Outcome names.** none fixed
- **Score.** 1 substantive prediction: 0 held, 0 failed, 1 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_mini_dpo_8b_2026-08-29.md` | `994ece4b958340f6829790dd2912570396f1ace351b04fb1dfe305729fbb2891` | 2026-08-29T10:01Z | 2026-08-30T11:05Z |
| `mini_dpo_reading_anahtari_2026-08-29.md` | `37b4a36e9b2486b57efb203da6dec4baf5fc7351d14755675f4102d34fb27139` | 2026-08-29T10:32Z | — |

## Registration `TABAN-CIZGISI`

Internal name: `TABAN-CIZGISI`

- **Bar.** No bar or outcome name was fixed.
- **Outcome names.** none fixed
- **Score.** No prediction was registered.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_base_cizgisi_2026-08-30.md` | `de3c68a3991f2300dee6aaf5e8593ab58274f7a56efcf71b25f19daaafd58bbc` | 2026-08-30T19:49Z | — |

## Registration `RLVR-TÜLÜ (dogrulama)`

Internal name: `RLVR-TÜLÜ (dogrulama)`

- **Bar.** The paper's values 2.89 and 2.56 are compared with M1 (second-person forms per 1k tokens, unfiltered primary reading) measured at RLVR step_60 and step_360; if both lie within ±0.05, `KAYNAK-DOGRULANDI`, otherwise `KAYNAK-BULUNAMADI`.
- **Outcome names.** `KAYNAK-DOGRULANDI` · `KAYNAK-BULUNAMADI`
- **Score.** No prediction was registered.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_rlvr_tulu_2026-09-06.md` | `f840a8b3bd26920042ff45e04100d3baf8ff6736c707d37f62d550eb80aaf7e5` | 2026-09-06T08:19Z | — |

## Registration `TEMPLATE-GÜRBÜZLÜGÜ`

Internal name: `TEMPLATE-GÜRBÜZLÜGÜ`

- **Bar.** For nine aligned arms, the ΔU_DOM verdict name under bare versus chat-template prompting is compared: ≥`7/9` `AYNI-AD` → `TEMPLATE-GÜRBÜZ`, ≥`3/9` `AD-DEGISTI` → `SABLONA-BAGIMLI`, in between → `KISMÎ-GÜRBÜZ`, fewer than 5 measurable → `KAPI/ÖLCÜLEMEZ`.
- **Outcome names.** `AYNI-AD` · `AD-DEGISTI` · `KAPI/ÖLCÜLEMEZ` · `TEMPLATE-GÜRBÜZ` · `SABLONA-BAGIMLI` · `KISMÎ-GÜRBÜZ`
- **Score.** 3 substantive predictions: 0 held, 3 failed (`BT-1`, `BT-2`, `BT-3`), 0 not scored; operational: `BT-4` failed.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_template_robustness_2026-08-26.md` | `0ad90508a8a1840a8a8b0202c9c3c791dae11e6ff0a0a5bd6f7fac33d6b92b26` | 2026-08-26T12:40Z | 2026-08-26T14:56Z |
| `prediction_template.md` | `3cee83c0ea3c40fdc2cdb3bf12451bc26265f7c22397119c72b88e5b09fadd8b` | 2026-08-26T14:48Z | — |

## Registration `TEMPLATE-24`

Internal name: `TEMPLATE-24`

- **Bar.** Per arm, the ΔU_DOM name bare versus templated is `AYNI-AD`, `AD-DEGISTI` or KAPI; the panel is `SABLONA-GÜRBÜZ` if `AYNI-AD` ≥ `2/3` of non-KAPI arms, otherwise `SABLONA-BAGIMLI`, and KAPI if fewer than two arms are measurable.
- **Outcome names.** `AYNI-AD` · `AD-DEGISTI` · `KAPI` · `SABLONA-GÜRBÜZ` · `SABLONA-BAGIMLI`
- **Score.** 4 substantive predictions: 4 held, 0 failed, 0 not scored; operational: `BP-5` held.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_template_robustness_24_2026-08-27.md` | `0034a30e3f923b83316e43bec46bceb31d23d57464dfdbaa3bee6b903698e42d` | 2026-08-27T14:25Z | 2026-08-27T14:41Z |
| `prediction_24.md` | `9640ee31a171c9ee2f7f48f6730239532bc8c7159a9a3267f63c9ce7c983dc9e` | 2026-08-27T14:25Z | 2026-08-27T20:14Z |

## Registration `C1-AILE-PANELI`

Internal name: `C1-AILE-PANELI`

- **Bar.** Per base↔instruct pair, ΔU_DOM = U(base) − U(instruct) with a paired 2,000-draw prompt-cluster bootstrap CI: fewer than 3 pairs passing controls → `KAPI/ÖLCÜLEMEZ`; no U_DOM CI above zero anywhere → `DOM-HER-YERDE-ÖLÜ-TEYIT`; every passing pair CI-disjoint > 0 → `TÜM-AILELER-SILER`; otherwise `IMZA-AILEYE-GÖRE`.
- **Outcome names.** `KAPI/ÖLCÜLEMEZ` · `DOM-HER-YERDE-ÖLÜ-TEYIT` · `TÜM-AILELER-SILER` · `IMZA-AILEYE-GÖRE`
- **Score.** 17 substantive predictions: 2 held, 0 failed, 15 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_family_panel_2026-08-15.md` | `9746d9746ec1c0321c7508369a1267303a088c1b52fa909f06b92f5bdc823784` | 2026-08-15T10:27Z | 2026-08-21T07:55Z |
| `prediction_2026-08-15.md` | `4ed1afcce1be40f1135290a18cae4b06292a45a5333f976b8509e34133bf988f` | 2026-08-15T10:31Z | — |
| `prerun_notice_2026-08-15.md` | `c23a8eaed5e3cd0c0f887ecd69020b7f3adc69f181d90aefd1609468eb95968f` | 2026-08-15T10:28Z | — |

## Registration `P5-ÖLCEK-KOLU`

Internal name: `P5-ÖLCEK-KOLU`

- **Bar.** C1's decision cascade and constants are imported unchanged for three larger pairs (Qwen2.`5-32B`, Qwen2.`5-72B`, gemma-4-26B-A4B), each with its own nulls; the pair bar N_CIFT_BAR = 3 equals the panel size, so losing any pair yields `KAPI/ÖLCÜLEMEZ`.
- **Outcome names.** `TÜM-AILELER-SILER` · `IMZA-AILEYE-GÖRE` · `DOM-HER-YERDE-ÖLÜ-TEYIT` · `KAPI/ÖLCÜLEMEZ`
- **Score.** 3 substantive predictions: 2 held, 1 failed (`Ikincil`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_scale_arm_2026-08-21.md` | `252c805797d17036bb290a0167563e5a186111dc5ea0f9cce5b001c15735a2ed` | 2026-08-21T15:36Z | 2026-08-21T21:51Z |
| `prediction_2026-08-21.md` | `267ec52370452259f5eb3ff1e69d7a7363e0d05f4af76a2e2ef6aa3a1e0fa698` | 2026-08-21T16:26Z | — |

## Registration `TEMPLATE-MERDIVEN (A1)`

Internal name: `TEMPLATE-MERDIVEN (A1)`

- **Bar.** No bar or outcome name was fixed.
- **Outcome names.** none fixed
- **Score.** No prediction was registered.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_template_holes_2026-08-28.md` | `d8309fd20462356a80a56e5555522acc333457b657d3d5403a86de3954c55e97` | 2026-08-28T21:27Z | 2026-08-29T09:20Z |

## Registration `PREREG_N2_BASE_SONDASI_2026-08-14`

Internal name: `PREREG_N2_BASE_SONDASI_2026-08-14`

- **Bar.** If the arousal positive control passes, dominance counts as accumulating with dose when the lower CI95 bound of U_DOM (x4 minus x1 centred separation, prompt-cluster bootstrap) is above 0; otherwise the lower CI95 bound of D22 above 0 decides; any failed gate is unmeasurable.
- **Outcome names.** `BASE-BIRIKIR-HIZALAMA-SILMIS` · `BASE-DE-BIRIKMEZ` · `KISMI/ARA` · `KAPI/ÖLCÜLEMEZ`
- **Score.** Recorded verdict `BASE-BIRIKIR-HIZALAMA-SILMIS` (its name later narrowed by an erratum to the seal); of the two probability predictions, the assistant's was scored the winner.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_n2_base_probe_2026-08-14_55a88e.md` | `8af7a13c9e5c75e3becc142771de42060d4572d3937a4734a1904a5d32eee831` | 2026-08-14T19:52Z | 2026-08-15T12:59Z |
| `prereg_n2_base_probe_2026-08-14_5a9d6d.md` | `af9f77f01e3b50103a269f8005b6d06dd8eee27b56aaf6d889d74afa102221d9` | 2026-08-14T19:52Z | 2026-08-21T07:55Z |

## Pre-run notice `NOTICE_C2_2026-08-15`

Internal name: `NOTICE_C2_2026-08-15`

- **Bar.** Pre-run notice: states that no decision statistic of the `C-2` conflict-probe seal had been computed; carries no bar.
- **Outcome names.** `ADAY-SINIFI-DOGAR` · `YINE-DUYARSIZ` · `KARMA` · `KAPI/ÖLCÜLEMEZ`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prerun_notice_conflict_probe_2026-08-15.md` | `70a3df3436b5611a8e5e1ef4bcc2680f616966f1d1f5d738435d5e719e95238f` | 2026-08-15T10:48Z | — |

## Pre-run notice `NOTICE_C3_2026-08-15`

Internal name: `NOTICE_C3_2026-08-15`

- **Bar.** Pre-run notice: states that no decision statistic of the `C-3` M1 clean-pipeline seal had been computed (declared only half-open: the run was going and point estimates had already been seen); carries no bar.
- **Outcome names.** `TAVANA-YAKIN-GECIYOR` · `KISMEN` · `GECMIYOR` · `REJIM-KURULAMADI` · `ÖLCÜLEMEZ`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prerun_notice_c3_2026-08-15.md` | `fd50c7d218c12c953b0b96e7c3824637b30bdb4a80fc3ab701c92238f2f1cb68` | 2026-08-15T10:39Z | — |

## Pre-run notice `NOTICE_C_YE_CMV_2026-08-25`

Internal name: `NOTICE_C_YE_CMV_2026-08-25`

- **Bar.** Pre-run notice: states that no decision statistic of the c_YE exam on cga-cmv seal had been computed (the raw cga-cmv generations existed but had not been read); carries no bar.
- **Outcome names.** none fixed
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prerun_notice_ye_cmv_2026-08-25.md` | `6f54c8faae352da007068c613349f6aa4ed87b584d1f7b097c4ac5cb2284bd58` | 2026-08-25T06:55Z | — |

## Registration `PREREG_N2B_HIZALAMA_FARKI_2026-08-14`

Internal name: `PREREG_N2B_HIZALAMA_FARKI_2026-08-14`

- **Bar.** If the arousal positive control passes on both base and instruct Qwen2.`5-7B`, the paired prompt-cluster bootstrap CI95 of ΔU_DOM (base minus instruct dominance dose-accumulation) decides: lower bound above 0 means erasure, upper bound below 0 amplification, CI containing 0 no effect.
- **Outcome names.** `HIZALAMA-SILIYOR` · `HIZALAMA-BÜYÜTÜYOR` · `HIZALAMA-DOKUNMUYOR` · `KAPI/ÖLCÜLEMEZ`
- **Score.** Recorded verdict `HIZALAMA-DOKUNMUYOR`; one prediction left open, no winner recorded; later flagged as contingent (1.17 sd).

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_n2b_hizalama_difference_2026-08-14.md` | `2b8cd11e271f9c06529e476b2286e1ea647645210b456ff47698311bc6dd6eb5` | 2026-08-14T20:17Z | — |

## Registration `PREREG_N2_MASKE_ALIM_2026-08-20`

Internal name: `PREREG_N2_MASKE_ALIM_2026-08-20`

- **Bar.** Each adjacent-rung pair is read as raw total NLL and per-token NLL: opposite signs are unmeasurable (length confound); sign consistency below 0.65 is direction-reversing; |mean| below MDE = 1.645·sd/√n is not-read; fewer than 200 pairs or a missing shard is unmeasurable.
- **Outcome names.** `MERDIVEN-OKUNUYOR` · `MERDIVEN-OKUNMUYOR` · `YÖN-DÖNEN` · `ÖLCÜLEMEZ`
- **Score.** Not scored: no pre-run notice or prediction is on file; the registered verdict `MERDIVEN-OKUNMUYOR` stands.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_n2_maske_alim_2026-08-20.md` | `2ec4fd698eb8e5edcaed8207f401364cee44c1933e2f89795ab24ba24c5b20ef` | 2026-08-20T10:29Z | 2026-08-20T11:50Z |

## Registration `PREREG_RESTORASYON_2026-08-25`

Internal name: `PREREG_RESTORASYON_2026-08-25`

- **Bar.** Restoration counts per family if the steered aligned arm's dominance share is two-sided disjoint from the KL-matched placebo (frac2 ≤ 0.05) and closes ≥0.50 of the base–aligned gap; panel needs ≥`4/6`; AUC < 0.70, KL deviation > 10% or gap under 30 pp is unmeasurable.
- **Outcome names.** `GERI-GELIR` · `KISMI` · `GERI-GELMEZ` · `BOZULUR` · `KAPI/ÖLCÜLEMEZ` · `SAKINMA-DOGRULANDI` · `SILINME` · `AILEYE-GÖRE`
- **Score.** Both panels `KAPI/ÖLCÜLEMEZ` (unmeasurable); recorded prediction scores: assistant 3/3 · executor 1/4 (BR-1, BR-4, PR-1, PR-4 not scorable).

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_restorasyon_2026-08-25.md` | `94d671f7cc4ab38ae9841fb816866b51fa9028b5954fce8e5cc1ffaa385f0cb9` | 2026-08-25T13:11Z | 2026-08-29T10:01Z |

## Pre-run notice `NOTICE_CAPRAZ_ENJEKSIYON_2026-08-09`

Internal name: `NOTICE_CAPRAZ_ENJEKSIYON_2026-08-09`

- **Bar.** Pre-run notice: states that no decision statistic of the cross-model injection (`K-TRANSFER-MATRISI`) seal had been computed; carries no bar.
- **Outcome names.** `KÖSEGEN-BASKIN` · `SOY-BLOK` · `KARISIK` · `ÖLCÜLEMEZ`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prerun_notice_capraz_enjeksiyon_2026-08-09.md` | `542c188bb015990c6529887cf0d6a93b4232865922da7480e7e393f8388edd6a` | 2026-08-09T16:00Z | — |

## Reading rule `RULE_ORNEK`

Internal name: `RULE_ORNEK`

- **Bar.** Fixes, before selection, how the Appendix H example pair is chosen: the prompt whose ΔM1 is closest to the family median, the first arm with no second-person tokens, draw 0, first 120 tokens, named drop rules; no acceptance bar.
- **Outcome names.** none fixed
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_example_pair_selection.md` | `334bb902299a9d2c5e512f2e0b931e60955576da953a00673a2fa1f3a2653394` | 2026-09-15T16:20Z | — |

## Reading rule `RULE_TAVAN`

Internal name: `RULE_TAVAN`

- **Bar.** Fixes a deliberately over-cutting ceiling reading: drop any generation with a refusal or assistant marker at a line or sentence start, a turn marker, or markdown structure, then recompute ΔM1 with a prompt-clustered 95% CI; no acceptance bar.
- **Outcome names.** none fixed
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_continuation_mode_exit_ceiling.md` | `98b844294cfd2ea00a2bbc3e2898aca28808738911f1502189df38968a3652c6` | 2026-09-15T15:35Z | — |

## Reading rule `RULE_DUZYAZI_SAHIS1_2026-09-15`

Internal name: `RULE_DUZYAZI_SAHIS1_2026-09-15`

- **Bar.** Fixes how the first-person change (capital 'US' excluded) is counted in prose-only lines of bare-prompt output, with prompt-clustered CIs classed down, up or null; no acceptance bar, though the owner's wording rule keys on `16/16` down.
- **Outcome names.** `asagi` · `yukari` · `null`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_prose_first_person_2026-09-15.md` | `06bf42f10cd930015b9287bcb8d098800cbb7a13c2b62e8b94a1fbd1e01e5427` | 2026-09-15T20:58Z | — |

## Reading rule `RULE_HAKEM6_OKUMALARI_2026-09-17`

Internal name: `RULE_HAKEM6_OKUMALARI_2026-09-17`

- **Bar.** Fixes five descriptive referee re-readings (`ELICIT-99` levels, shared aligned leg, mode rule on the templated panel, five stimulus classes, two templated examples) with equivalence gates; no acceptance bar; a family-mode cell enters a band only with at least 80 rows.
- **Outcome names.** `asagi` · `yukari` · `null` · `ALET-KAYDI` · `AYNI ÜRETIM · IKI SÜZGEC` · `FARKLI ÜRETIM`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_reviewer6_readings_2026-09-17.md` | `76be7323494818325bcdc6bec9c7eb147d999859e7e0661af272af73516a0cfe` | 2026-09-17T07:38Z | — |

## Reading rule `RULE_HAKEM_OKUMALARI_2026-09-15`

Internal name: `RULE_HAKEM_OKUMALARI_2026-09-15`

- **Bar.** Fixes six descriptive referee re-readings (Q1–Q5, W2) with prompt-clustered CIs classed down/up/null and equivalence gates; its only threshold is Q3's two-sided exact sign test over pre-defined dependency units, read against the paper's Bonferroni bar 0.`05/10` = 0.005.
- **Outcome names.** `asagi` · `yukari` · `null` · `ALET-KAYDI` · `karisik`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_reviewer_readings_2026-09-15.md` | `6fcd7b5238cf215b7567f345001f7c6e9911a0cc6eef5e2e1a77bee74c5685d6` | 2026-09-15T19:30Z | — |

## Reading rule `RULE_V84_OKUMALARI_2026-09-16`

Internal name: `RULE_V84_OKUMALARI_2026-09-16`

- **Bar.** Only item (a) carries a bar: of the seven families that rise under the template, a family stays if ΔM1 > 0 with CI lower bound > 0 after three simultaneous exclusions; ≥3 staying passes, <3 stops; items (b)–(e) are descriptive.
- **Outcome names.** `KALDI` · `DÜSTÜ` · `ÖLCÜLEMEZ` · `GECTI` · `DUR` · `KOSULMADI`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v84_readings_2026-09-16.md` | `98767c73ee3e92a4b7049462f2593f9b1bc1aac852009e65abf22ece05db6507` | 2026-09-16T06:17Z | — |

## Reading rule `RULE_V87_OKUMALARI_2026-09-16`

Internal name: `RULE_V87_OKUMALARI_2026-09-16`

- **Bar.** Declares the readings descriptive (no acceptance bar or verdict name) but fixes beforehand: each card/objective hit gets one context label; the arm-split placebo passes per family if |ΔM1| exceeds its 200-draw p95, and enters §1 only if ≥`12/16` pass.
- **Outcome names.** `ILGISIZ` · `ANILIR` · `METRIK` · `AMAC-KISI` · `AMAC-YAKIN` · `KISI` · `YAKIN` · `HAYIR` · `CARD-YOK` · `METIN-YOK` · `ALET-KAYDI`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v87_readings_2026-09-16.md` | `6a7fdc67a6902794609b33c93be8dde3447cf8a6136a338d9f352aee4952efb1` | 2026-09-16T13:22Z | — |

## Reading rule `RULE_V90_DUSEN_SAYIM_2026-09-17`

Internal name: `RULE_V90_DUSEN_SAYIM_2026-09-17`

- **Bar.** Fixes the failed-prediction count: a registration counts if its seal pre-wrote a numeric bar or named outcome space and the paper reports its result; a failure is a binding 'did not hold/fell' score, excluding unmeasurable, unscorable, procedural and base-rate items; no acceptance bar.
- **Outcome names.** `CÖZÜLEMEDI`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v90_failed_count_2026-09-17.md` | `f74a9776925c8506cdacf2c5ceec6d3abba81d3b3fcd18dbfb7ccdab950a9844` | 2026-09-17T14:59Z | — |

## Reading rule `RULE_V90_OKUMALARI_2026-09-17`

Internal name: `RULE_V90_OKUMALARI_2026-09-17`

- **Bar.** Fixes three descriptive CPU readings with equivalence gates: a thesis-clustered CI over 17 clusters for Table 1, a paired within-prompt placebo p95 (200 draws) for plain-minus-natural, and a one-line table of the 19 failed registrations; no acceptance bar.
- **Outcome names.** `asagi` · `yukari` · `null` · `ALET-KAYDI`
- **Score.** Not applicable.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v90_readings_2026-09-17.md` | `d1101a292b007425f541490e7922b4c214f0bc786c037fe8e58429476619db6c` | 2026-09-17T14:02Z | — |

## Registration `SIRADAN_TALIMAT` (ordinary instructions)

Internal name: `PREREG_SIRADAN_TALIMAT_2026-09-17`

- **Bar.** No decision rule: the registration is descriptive, no number is read as held or failed, and entry into the paper needs a separate instruction from the author.
- **Outcome names.** none fixed
- **Score.** No prediction was registered (descriptive class).

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_siradan_instruction_2026-09-17.md` | `05378517ad4210a7490c94fadfdfd00bc65f9214ac85d78cb3a6e3dc73bb15de` | 2026-09-17T08:17Z | — |

## Registration `TEMPLATE_KADRAN` (the dial under the chat template)

Internal name: `PREREG_TEMPLATE_KADRAN_BETIM_2026-09-17`

- **Bar.** No acceptance bar and no outcome name; the one rule fixed in advance is the degeneracy floor, an arm with distinct-4 below 0.60 is shelved and printed by name, and every slot must carry 6,528 complete rows.
- **Outcome names.** none fixed
- **Score.** No prediction was registered (descriptive class).

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_template_kadran_betim_2026-09-17.md` | `ae3de11b933fadb4c351f312c4c9a902db39db158a93fb0e47b025203e2189af` | 2026-09-17T08:33Z | — |

## Reading rule `RULE_V91_KONUSAN_SAYIMI` (direct first-person count, ordinary instructions)

Internal name: `RULE_V91_KONUSAN_SAYIMI_2026-09-17`

- **Bar.** Fixes, before counting, a direct first-person count (capitalised US excluded) on the ordinary-instruction generations through the same pipeline, gated on reproducing the reading's second-person numbers exactly; no acceptance bar.
- **Outcome names.** `asagi` · `yukari` · `null`
- **Score.** Not applicable (reading rule).

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v91_konusan_sayimi_2026-09-17.md` | `f3abd754837e939ef9093c9f6a7dbc5416d6129257de07a127c21956199d13c5` | 2026-09-17T22:40Z | 2026-09-18T16:16Z |

## Registration `KK3UCKAT_T2` (tripled-tilt arm, second training seed)

Internal name: `PREREG_KK3UCKAT_T2_2026-09-18`

- **Bar.** The sign of tripled minus flat at the same training seed must match the first seed's (negative); the magnitude is reported only as a range over the two seeds.
- **Outcome names.** `ISARET-TUTTU` · `ISARET-DÖNDÜ` · `RAF` · `ALET-KAYDI` · `KOSULMADI`
- **Score.** Recorded verdict `ISARET-TUTTU` (the sign held at the second seed); the blind predictions are not yet scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_kk3uckat_t2_2026-09-18.md` | `b5e6e72a8a446ed15de211aafe4fa75709db5cb5a8f558297f1e2d87f8150e80` | 2026-09-18T06:47Z | — |
| `prerun_notice_kk3uckat_t2_2026-09-18.md` | `ce0cd54ab89f947ddf8721ecd916b24b08a5e9ccdc4c88b27f3f1b74699e629e` | 2026-09-18T06:48Z | — |
| `prediction_kk3uckat_t2_2026-09-18.md` | `437c2366458d8a155bd534bf2352cf2a2069f4ee9cd7a5f502f2a0e9aad96169` | 2026-09-18T06:52Z | — |

## The seventh family (Gemma-4-12B base and instruct) read against the signature of its predecessor, Gemma-3-12B

Internal name: `M-3 YEDINCI-AILE`

- **Bar.** Per axis (ARO, DOM, TON), Gemma-4-12B's ΔU = U(base) − U(instruct) is classed POZ, NEG or BANT by whether its paired 2,000-draw prompt-cluster bootstrap 95% CI excludes zero, and compared with Gemma-3-12B's class from the six-pair panel: if any of four gates fails (positive control, empty text, `RULE-3` regime, centred polarity placebo) `KAPI/ÖLCÜLEMEZ`; otherwise ≥2 axes with the opposite separated sign `NESIL-FARKI`, all 3 axes in the same class `IMZA-TEKRAR`, 1–2 `KISMI`, 0 `IMZA-YOK`.
- **Outcome names.** `KAPI/ÖLCÜLEMEZ` · `NESIL-FARKI` · `IMZA-TEKRAR` · `KISMI` · `IMZA-YOK`
- **Score.** 10 substantive predictions: 3 held, 2 failed (`ÖM3`, `ÖM4`), 5 not scored (`Ö8`–`Ö12`); operational: the two gate-passing probabilities (0.86, 0.80) not scored; name-level probabilities: the reached name had 0.20 from both bettors, recorded as a tie, not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_impersonal_correction_seventh_family_2026-08-15.md` | `58db51a7f24d5e2fac4362cbf442db2ca45e55c78b8a3ab3257d15369dfbc6c2` | 2026-08-15T21:03Z | 2026-08-21T07:55Z |
| `prediction_seventh_family_2026-08-15.md` | `ac11825f0122ff26584b81bc24f18901182361f5c1dd2ac28eeb5a0d0747f9fc` | 2026-08-15T13:50Z | — |
| `prediction_impersonal_correction_2026-08-15.md` | `8e8dfacb7224f533b610329fdbfc66cfa2d401b15f3a4ad3f7e58c57dba179b9` | 2026-08-15T21:05Z | — |

## Texts that address the reader versus texts that do not, as a moderator of the base-to-instruct change in pole separation

Internal name: `MUHATAP`

- **Bar.** Per (pair × axis) cell, D = ΔU(texts addressing the reader) − ΔU(texts not addressing it) with a paired prompt-cluster bootstrap CI: fewer than 24 shared prompts `KAPI/ÖLCÜLEMEZ`, an uncentred null (|mean|/sd > 1.645) `NULL-AYIRT-EDEMEDI`, a directional CI whose polarity-placebo fraction exceeds 0.05 `PLASEBO-AYIRT-EDEMEDI`, CI above zero `MUHATAP-BÜYÜTÜR`, below zero `MUHATAP-KÜCÜLTÜR`, else `MUHATAP-FARKI-YOK`; per axis the panel is `ÖLCÜLEMEZ-PANEL` under 8 measurable cells, `MUHATAP-TASIMAZ` with no directional cell, `MUHATAP-TASIYICI` if one direction covers ≥`2/3` of measurable cells across ≥3 families, `MUHATAP-AILEYE-BAGLI` if both directions occur, else `MUHATAP-KARMA`.
- **Outcome names.** `KAPI/ÖLCÜLEMEZ` · `NULL-AYIRT-EDEMEDI` · `PLASEBO-AYIRT-EDEMEDI` · `MUHATAP-BÜYÜTÜR` · `MUHATAP-KÜCÜLTÜR` · `MUHATAP-FARKI-YOK` · `ÖLCÜLEMEZ-PANEL` · `MUHATAP-TASIYICI` · `MUHATAP-AILEYE-BAGLI` · `MUHATAP-KARMA` · `MUHATAP-TASIMAZ`
- **Score.** 7 substantive predictions: 1 held, 3 failed (`BM1`, `BM6`, `BM7`), 3 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_addressee_2026-08-22.md` | `f68b2c52e29d2b446534afe9e0f6ced658b28371cc0b1a441c15c4cbd7e8810e` | 2026-08-22T18:21Z | 2026-08-22T19:19Z |
| `prerun_notice_addressee_2026-08-22.md` | `ad523936cb0da69895d011d05e4f3617be072715b110d3929142e470d05d5ed1` | 2026-08-22T18:21Z | — |
| `prediction_addressee_2026-08-22.md` | `64ec94b259b02d2c1c36456ec05b3ea75c0c37dbbe6d664db8ff8ee3cfed97e4` | 2026-08-22T19:13Z | — |

## The chosen side of preference pairs on which the blind judge was unanimous about dominance

Internal name: `SECIM-KUYRUK`

- **Bar.** In each preference box (human hh-rlhf, n = 165; GPT-4-written Tülu-3, n = 57), among pairs where the blind judge named the same text more dominant in all three passes, the share whose chosen side is the less dominant one is tested against binomial p = 0.5: n < 20 `KAPI/ÖLCÜLEMEZ`, a Wilson CI excluding 0.5 `KUYRUK-YÖN-VAR`, else `YÖN-YOK`; the panel is `CELISKILI` if the two boxes separate in opposite directions, otherwise `KUYRUK-YÖN-VAR` if at least one box separates.
- **Outcome names.** `KUYRUK-YÖN-VAR` · `YÖN-YOK` · `KAPI/ÖLCÜLEMEZ` · `CELISKILI`
- **Score.** 3 substantive predictions: 2 held, 1 failed (`BK-2`), 0 not scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_secim_queue_2026-08-27.md` | `7dcfc73f89c42829523409531b440a2f52d125fa58013ef75b9df838b98d2152` | 2026-08-27T11:27Z | — |
| `prerun_notice_secim_queue_2026-08-27.md` | `14e57213983a0d454ca0616fa2d85d348786c3581200df8e775166d7d0e8ebad` | 2026-08-27T11:29Z | — |
| `prediction_queue.md` | `39bdee8163d5f1ec03c5f5f2efde1853008b8f36509c58050f0d6ed49cc96157` | 2026-08-27T11:48Z | — |

## Reading rule `RULE_V96_Q1_BIRINCISIZ_ONEK` (person change on prefix arms with no first-person content)

Internal name: `RULE_V96_Q1_BIRINCISIZ_ONEK_2026-09-18`

- **Bar.** Descriptive, no decision rule: on the raw-continuation panel (original seed, 16 families, aligned − base), reads derived and direct Δ1st and ΔM1 on the four prefix arms with no first-person content (K4) and on the two with neither person (K2), each family classed down/up/null by a prompt-clustered 1,000-draw CI, gated on reproducing Table 1's Δ1st on all 16 arms to ≤1e-9.
- **Outcome names.** `asagi` · `yukari` · `null` · `ALET-KAYDI`
- **Score.** Descriptive; no bar and no predictions.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v96_q1_birincisiz_prefix_2026-09-18.md` | `53b00867aeeb8422d85fe58f3a37a3556b0423df00fa60f0acb69fe5be68bc96` | 2026-09-18T20:00Z | — |

## Reading rule `RULE_V96_Q2_YARGIC` (judge labels: reply or continuation per output, addressed or generic 'you' per sentence)

Internal name: `RULE_V96_Q2_YARGIC_2026-09-18`

- **Bar.** Descriptive, no decision rule: a Qwen2.5-32B-Instruct judge (temperature 0, fixed prompts) labels 6,268 templated-panel outputs `REPLY/CONTINUE/OTHER` and 2,000 second-person sentences `ADDRESSED/GENERIC/OTHER`, from which the rule reads the opener rule's precision and recall, the within-mode band (max/min over families whose mode share is ≥20%), and addressed-only ΔM1.
- **Outcome names.** `REPLY` · `CONTINUE` · `OTHER` · `ADDRESSED` · `GENERIC` · `SEMA-DISI` · `HATA`
- **Score.** Descriptive; no bar and no predictions.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v96_q2_judge_2026-09-18.md` | `0e7a97a5296e3977610601109178fb41e09fbbc0eda518ca4ae6f0ce6aa3cd04` | 2026-09-18T21:38Z | — |

## Reading rule `RULE_V96_Q4_URIAL_OZDES` (ordinary instructions: aligned leg on the identical URIAL string, base leg in the person-free frame)

Internal name: `RULE_V96_Q4_URIAL_OZDES_2026-09-18`

- **Bar.** Descriptive, no decision rule: on the 300 ordinary instructions, generates aligned legs on the base leg's bit-identical URIAL string and base legs in the person-free frame, and reads ΔM1 and direct Δ1st for K1 (identical string) and K2 (person-free base) with B1's engine, classed down/up/null and counted beside B1's `16/16` (second person) and `12/16` (first person).
- **Outcome names.** `asagi` · `yukari` · `null` · `ALET-KAYDI`
- **Score.** Descriptive; no bar and no predictions.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v96_q4_urial_ozdes_2026-09-18.md` | `b882e429b3669f72b5d99c08108bf68d0d6608c81bd8f4a98328b6aad78fc2b4` | 2026-09-18T21:25Z | — |

## Reading rule `RULE_V97_CPU_OKUMALARI` (five CPU readings: direct first person, code-fence cut, VIF refit, ELICIT on 15 models, generation cap)

Internal name: `RULE_V97_CPU_OKUMALARI_2026-09-19`

- **Bar.** Descriptive, no bar or outcome name: fixes five CPU readings, each behind its own equivalence gate — Table 1's direct first-person change over four seeds (R1), ΔM1 with the aligned leg also cut at the first code fence (R2), VIFs and two named refits of Table A14 (R3), ELICIT ΔM3 on 15 models (R4), and the share of outputs reaching the generation cap, n ≥ cap − 3 (R5).
- **Outcome names.** `asagi` · `yukari` · `null` · `ALET-KAYDI`
- **Score.** Descriptive; no bar and no predictions.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v97_cpu_readings_2026-09-19.md` | `7e9892877ff84efa94ba88a4824f342456dfb6008d72d35cff33873dc18fd87c` | 2026-09-19T16:24Z | 2026-09-19T16:56Z |

## Reading rule `RULE_V98_CPU_OKUMALARI` (five CPU readings: prose lines, mode split, position, pronoun-free pairs, reverse direction)

Internal name: `RULE_V98_CPU_OKUMALARI_2026-09-21`

- **Bar.** Descriptive, no bar or outcome name: fixes one path each for five CPU readings — prose-line and per-sentence ΔM1 and direct Δ1st in cells B1 and K2 (R1, gated on reproducing the recorded B1 numbers to ≤1e-9), a weighted between/within-mode split of templated ln M1 over cells with mode share ≥0.05 (R2), the first 160 tokens versus the rest (R3), chosen − rejected differences on the 37,426 pronoun-free pairs (R4), and the reverse direction (pronouns added to a person-free original) in both reward models (R5).
- **Outcome names.** `asagi` · `yukari` · `null` · `ALET-KAYDI`
- **Score.** Descriptive; no bar and no predictions.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v98_cpu_readings_2026-09-21.md` | `87a4206445fb2f6c8531b4e9e10953557b1f9d20bf67df6400ca5740df791737` | 2026-09-21T08:58Z | 2026-09-21T09:27Z |

## Registration `V96_K3_KISISIZ_OZDES` (ordinary instructions, both legs on the identical person-free URIAL string)

Internal name: `PREREG_V96_K3_KISISIZ_OZDES_2026-09-19`

- **Bar.** No decision rule: with both checkpoints reading the identical person-free URIAL string on the 300 ordinary instructions, prints for six measures (ΔM1 and direct Δ1st on full output, and on prose lines per token and per sentence) how many of 16 models are down with a CI excluding zero and how many of those also clear the paired-placebo p95.
- **Outcome names.** none fixed
- **Score.** Descriptive; no bar and no predictions.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_v96_k3_impersonal_ozdes_2026-09-19.md` | `2770effff6f30531713a5e25d44d619e5633e298e2e35060cb78430e643e01e1` | 2026-09-19T08:00Z | 2026-09-19T08:02Z |

## Registration `V97_Q4_TUR` (genre-fixed raw continuation: four genre headers, same thesis scaffold, both legs on the same string)

Internal name: `PREREG_V97_Q4_TUR_2026-09-20`

- **Bar.** No decision rule: on raw continuation with one of four person-free genre headers (blog post, forum reply, how-to guide, news report) before the same thesis scaffold, prints per genre and pooled, for six measures, how many of 16 models are down, up or null and how many down models clear the paired placebo, plus a regex genre-agreement matrix whose residual blog share is read as an upper bound.
- **Outcome names.** none fixed
- **Score.** Descriptive; no bar and no predictions.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_v97_q4_turn_2026-09-20.md` | `954cddd80492349dc9ac170d3cca5ed359847584a9297f87df2d970a976db300` | 2026-09-20T22:24Z | — |

## Reading rule `RULE_V99_C2B_UYUM` (human–judge agreement on addressed versus generic 'you', package B)

Internal name: `RULE_V99_C2B_UYUM_2026-09-21`

- **Bar.** Raw human–judge agreement on the 100 blind package-B items decides: `GIRER` if `X/N` ≥ 0.80 with N = 100 (the addressed-'you' sentence enters §2), `GIRMEZ` if `X/N` < 0.80 with N = 100 (it stays out; the agreement goes to Appendix C and a line to Limitations), OKUNAMADI if N < 100 or any answer is off-schema or doubled; κ is printed with no bar.
- **Outcome names.** `GIRER` · `GIRMEZ` · `OKUNAMADI`
- **Score.** Recorded verdict `GIRMEZ` (raw agreement 76/100, κ 0.546). The agent's bet `F-V99C2B-2` (`GIRMEZ`, p = 0.53) is recorded as held in the card's commit; `F-V99C2B-1` (`GIRER`, 0.45) and the operational `F-V99C2B-3` (`OKUNAMADI`, 0.02) carry no written score; the owner slot was not opened.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v99_c2b_uyum_2026-09-21.md` | `6c492ae68a7dfda0a0e5b5090b01d8dcbcd9e03b8fb61948c144664fdcf2caa7` | 2026-09-21T15:39Z | — |
| `prerun_notice_v99_c2b_uyum_2026-09-21.md` | `a747f8ee76a18c5d6a1c88387ca25c0d5c1bd45320f1de9e461eafaa7a691292` | 2026-09-21T15:40Z | — |
| `prediction_v99_c2b_uyum_2026-09-21.md` | `29c71ba23f93251dd94ce98d08c27d8c8d29d3b6a00ff6a7ad8c284788410c4d` | 2026-09-21T15:40Z | — |

## Registration `V98_ELICIT_OZDES` (ELICIT-99 with both legs wrapped in the aligned model's chat template)

Internal name: `PREREG_V98_ELICIT_OZDES_2026-09-21`

- **Bar.** No decision rule: on the 99 `ELICIT-99` prompts × 4 draws with both checkpoints wrapped in the aligned model's chat template (identical string, measured per leg), prints six measures and how many of 16 models are down, up or null and how many down models clear the paired placebo; a model whose strings diverge is written `DIZE-AYRIK` and left out of the count.
- **Outcome names.** none fixed
- **Score.** Descriptive; no bar and no predictions. Read (card V98_ELICIT_OZDES): all 16 models on the identical string; "you" falls in 12, rises in 1 and 3 show no clear change, 9 clear the paired placebo; "I" falls in 15.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_v98_elicit_ozdes_2026-09-21.md` | `bc86870441c6f2483b21f82ae67b0eec300fc90c0c7fd3e37db9ad9bf2cef3b4` | 2026-09-21T09:13Z | 2026-09-21T17:37Z |

## Registration `V98_KK3F_T2` (pronoun-free-pairs arm, second training seed)

Internal name: `PREREG_V98_KK3F_T2_2026-09-21`

- **Bar.** With only the training seed changed (20260913 → 20260922) on the pronoun-free-pairs arm: `TUTAR` if the second seed's ΔM1 on the debate panel is negative and its prompt-clustered 95% CI excludes zero, `DÜSER` if the sign is positive or the CI covers zero, `ÖLCÜLEMEZ` if the arm falls below the distinct-4 < 0.60 shelf or training does not finish (no score either way).
- **Outcome names.** `TUTAR` · `DÜSER` · `ÖLCÜLEMEZ`
- **Score.** `TUTAR`: second-seed ΔM1 −1.684, 95% CI [−2.536, −0.878] (seed 1: −1.204); placebo p95 1.014, distinct-4 shelf clear, 8 of 8 generations. Agent prediction F-V98T2-1 (`TUTAR`, 0.60) matches the outcome; owner slot not opened.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_v98_data_tilt_zero_difference_t2_2026-09-21.md` | `8fd0a649d877a2b6ada50b8d2d7bb2d0719f04678e5a4f065a684ec1d396b564` | 2026-09-21T09:13Z | 2026-09-22T05:54Z |
| `prerun_notice_v98_data_tilt_zero_difference_t2_2026-09-21.md` | `6fbd1c679971ac1746610bbf446bab2d8e2fb265ca552fff5ab17f091f94ca3c` | 2026-09-21T15:33Z | — |
| `prediction_v98_data_tilt_zero_difference_t2_2026-09-21.md` | `a48e0979b9546b7a2b5b1a3236022d6cf615dc7036b6b46be38a53f336646eed` | 2026-09-21T15:33Z | — |

## Registration `V99_SONNET_YARGIC` (second judge, Claude Sonnet, calibrated on the two 100-item human packages)

Internal name: `RULE_V99_SONNET_YARGIC_2026-09-21`

- **Bar.** Per package (A reply/continue, B addressed/generic), a second judge labels the same 100 items as the human labeller (three repeats, majority label): `GECER` if raw agreement is at least 80 of 100 with no failed call and a clean model check, `KALIR` if below 80, `OKUNAMADI` if any call fails or the model check does not pass; `GECER` lets the full set be labelled under a second registration.
- **Outcome names.** `GECER` · `KALIR` · `OKUNAMADI`
- **Score.** Package A: `KALIR` (78 of 100, κ 0.53, two items short of the bar; bar file BARAJ_V99_SONNET_A). Package B: `GECER` (84 of 100, κ 0.69). Agent predictions A2 (`KALIR`, 0.60) and B1 (`GECER`, 0.50) match the outcomes; owner slot not opened.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v99_sonnet_judge_2026-09-21.md` | `78bb443bee9a7fcc85d062bab4ff0f15af2e1551a9eec621004bab4b9c6cf1b4` | 2026-09-21T19:42Z | — |
| `prerun_notice_v99_sonnet_judge_2026-09-21.md` | `2ff14369a3618a9d83244ba656c750c55220a934227a28480981a17a5b7f35ba` | 2026-09-21T19:42Z | — |
| `prediction_v99_sonnet_judge_2026-09-21.md` | `060b7541e443307028907d32fa100a009cdfc6ff6fa90e0567a644465191516d` | 2026-09-21T19:43Z | — |
| `baraj_v99_sonnet_bare_imperative_2026-09-21.md` | `32f1c0500148214edd212e8725656b91153a79f7917ca4e43579006ce7784576` | 2026-09-21T19:48Z | — |

## Registration `V100_KELIME_PAYDA` (the main contrast per 1,000 words instead of tokens)

Internal name: `RULE_V100_KELIME_PAYDA_2026-09-22`

- **Bar.** The canonical readings are re-run with the denominator changed to word tokens (a tokenizer token carrying at least one letter), after each first reproduces its canonical card exactly with the token denominator: `GÖVDEYE` if the raw-continuation ΔM1 (four pooled seeds) falls with an interval excluding zero and clears the paired placebo in at least 12 of 16 models, `LIMITATIONS` if in fewer than 12, `OKUNAMADI` if the equivalence gate fails or 16 models cannot be read.
- **Outcome names.** `GÖVDEYE` · `LIMITATIONS` · `OKUNAMADI`
- **Score.** `GÖVDEYE`: 16 of 16 fall and clear the paired placebo per 1,000 words (16 of 16 per 1,000 tokens); Table 1 reproduced with 0 differences. Agent prediction F-V100K-1 (`GÖVDEYE`, 0.60) matches the outcome; owner slot not opened.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v100_kelime_payda_2026-09-22.md` | `63ae903a19ebde5e7a5e8ceb6086dc0d506f736767fdfdd1f8ce880d3ce36925` | 2026-09-22T08:53Z | 2026-09-22T09:31Z |
| `prerun_notice_v100_kelime_payda_2026-09-22.md` | `36519dd301ff532a42a19fcb341f512c1dbd8c57500a2e85341df55ef9fa07aa` | 2026-09-22T08:53Z | — |
| `prediction_v100_kelime_payda_2026-09-22.md` | `80d56bc49727ad6af207131c26acd7598ff578ca636cdd87801e45858ed57939` | 2026-09-22T08:53Z | — |

## Registration `V102_TAZE_B_UYUM` (second judge against the human labeller on 30 fresh items)

Internal name: `RULE_V102_TAZE_B_UYUM_2026-09-23`

- **Bar.** Descriptive, no bar: on 30 addressed-or-generic items drawn from the same 2,000 and labelled after the second judge had been chosen, none of them among the 100 used to choose it, the raw agreement between the judge and the labeller is counted and written into the paper whatever it is; per-class and per-leg agreement is named in the package frame's own `OLCULEMEZ_DOGAR` list (cells of 15 or fewer) and is printed without carrying a verdict.
- **Outcome names.** none fixed
- **Score.** Second judge 28 of 30 (kappa 0.859), first judge 23 of 30 (0.539), the two judges 25 of 30; 0 unreadable and 0 out-of-schema labels. Agent predictions: 28 or more for the second judge at 0.35 and the first judge below the second at 0.70 both match the outcome (Brier 0.545); owner slot not opened.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v102_taze_uyum_2026-09-23.md` | `758ece3aa8aee06c3bc1456cb0214aeea2c1e20576c2f837f7f4df8cb64160ff` | 2026-09-23T07:19Z | — |
| `prerun_notice_v102_taze_uyum_2026-09-23.md` | `79c2f15c5b9d752fc7205ae895293e12cdef3561ce891525c71d32182a8fc482` | 2026-09-23T07:20Z | — |
| `prediction_v102_taze_uyum_2026-09-23.md` | `2cb5857971f607b07f14731305f543413ee0be1075e5b2d8dcc821041a0ab805` | 2026-09-23T07:20Z | — |

## Registration `V103_DISKTEN` (five descriptive readings taken from existing generations, answering reviewer questions)

Internal name: `RULE_V103_DISKTEN_2026-09-23`

- **Bar.** Descriptive, no bar, five readings each with a single analysis path fixed before counting: (a) the two pipelines on one base read on the 300 ordinary instructions in their own formats; (b) the chat-template contrast for the models whose rate rises, recomputed with the first sentence dropped from both checkpoints on the same filtered cells, behind an equivalence gate of 1e-6 against the canonical counter; (c) the token volume inside fenced code blocks in the pronoun-selected and unselected preference pairs, and a scan of the debate prompts for code fences and for fifteen programming cues fixed before the count; (d) length, list share and refusal rate for the three versions of the restoring sentence against the same checkpoint without it; (e) every model directory under the panel's generation root, classified with its printed evidence and its generated row count. Whatever each reading gives is written, including where it does not support the paper.
- **Outcome names.** none fixed
- **Score.** (a) 2.71 and 3.86 per 1,000 tokens, a 1.43-fold gap against the 20.5-fold of the chat-template condition, so that contrast does not carry to the instruction set; (b) the rise survives in 5 of 7 after the first sentence is dropped, median shift +14.76 to +10.15; (c) the chosen side carries 1.36 points more of its tokens in code although fewer of its responses contain any, and the debate prompts contain 0 code fences in 5,011 distinct strings; (d) length and refusal rate barely move while the list share falls by 0.250 to 0.323; (e) 0 models were generated and left out of the paper. Agent predictions: 3 of 5 match (mean Brier 0.655); the two that failed both substituted a related quantity for the measured one, and the verdict says so. Owner slot not opened.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v103_diskten_2026-09-23.md` | `59c22860ffc2331f28096a0ab853069a1428ed927af07e2c0c801cd844b42b2d` | 2026-09-23T08:17Z | 2026-09-23T08:45Z |
| `prerun_notice_v103_diskten_2026-09-23.md` | `2ebedbf61cebae80e1611b01a304f3a42bf80d7b1a35b5576d11c9db7ae5d6d3` | 2026-09-23T08:19Z | — |
| `prediction_v103_diskten_2026-09-23.md` | `5ef64369a7be83fe1632ef4ae5affb921459e3ad77e4bb5342b2cea60b1dd24d` | 2026-09-23T08:19Z | — |

## Registration `V104_TABAN_BIRINCI` (the base checkpoints' first-person level, the half of the first-person shift the paper had not printed)

Internal name: `RULE_V104_TABAN_BIRINCI_2026-09-23`

- **Bar.** Descriptive, no bar, one analysis path fixed before the reading: the base and aligned first-person levels are recomputed from the same component matrices and the same primitives that produced the published first-person shift, and each model is admitted only if aligned minus base reproduces that published shift to within 1e-9. A single model failing that gate means no card is written at all, because a partial column would leave an empty cell and an empty cell is missing data. Where the resulting column is printed is decided by the build, not by preference: it goes in the main table if the table still fits its line width and the nine-page block holds, and in the full table with a pointer from the results section otherwise.
- **Outcome names.** none fixed
- **Score.** All 16 models passed the gate with a residual of exactly zero, so the published shift is unchanged and is now printed as two levels. Base first-person density spans 17.61 to 28.49 per 1,000 tokens (1.62-fold, median 22.83), against 1.39-fold and a median of 11.60 for the base second person on the same prompts; the two checkpoints that share a base read the same value twice. The column was placed in the main table and the build measured it 45.02 pt too wide, so it went to the full table with a pointer, as the rule required. Agent predictions: all four modal branches match the outcome (mean Brier 0.224); owner slot not opened.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v104_base_birinci_2026-09-23.md` | `645684f59a05f0fa39293da3145a42cee217080b5cff77b8dfca0acdd7d6283a` | 2026-09-23T13:55Z | — |
| `prerun_notice_v104_base_birinci_2026-09-23.md` | `cf2ecd02373936d1d8887c3e9c8f0ae55e25af315fa9f594b23c8644a3c44651` | 2026-09-23T13:55Z | — |
| `prediction_v104_base_birinci_2026-09-23.md` | `b82c26438fd66c82270454a77e3f277eba5b460809b8f1f3dc5d90e98e77df05` | 2026-09-23T13:56Z | — |

## Registration `V105_TEMPLATE_BIRINCI_SUZGECSIZ` (the first person under the chat template with the cell filter removed)

Internal name: `RULE_V105_TEMPLATE_BIRINCI_SUZGECSIZ_2026-09-23`

- **Bar.** Descriptive, no bar, one analysis path fixed before counting: the first-person rate under each model's own chat template is read on every generation rather than on the cells the degeneracy filter admits, with the direct counter used by the main table rather than the derived measure used by the filtered reading, prompt-clustered intervals, and a within-prompt paired placebo built on the base leg. Each model is admitted only if the same loop reproduces this appendix's own unfiltered second-person column to within 1e-4; a model that fails is not measured and stays in the denominator. Whatever the reading gives is written, including a result that would weaken the paper.
- **Outcome names.** none fixed
- **Score.** All 16 models passed the gate, the largest residual being 4.9e-05. The first-person rate falls in 16 of 16 with every interval excluding zero and every model above its paired placebo, the shifts running from -8.27 to -29.50 per 1,000 tokens. The reading was requested because a reviewer named it as one of two findings that would have lowered the paper's score; it did not. Agent predictions: all four match (mean Brier 0.148), but the verdict records that the reasoning behind the headline prediction was backwards, since the unfiltered fall widened where the prediction expected it to narrow. Owner slot not opened.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `rule_v105_template_birinci_suzgecsiz_2026-09-23.md` | `c8db723b2b3375b035781ef550f5e60f06a19af67859ae2f013975883dc37aef` | 2026-09-23T14:49Z | — |
| `prerun_notice_v105_template_birinci_suzgecsiz_2026-09-23.md` | `abadc87906a5c0a5a6c0a120045a42a15d6a9f33336443bdc7e3d7860946e579` | 2026-09-23T14:50Z | — |
| `prediction_v105_template_birinci_suzgecsiz_2026-09-23.md` | `13a4dcd0ec6deb8918c7c7e23df0f95ccbf447783797688b25eed311fbbdf379` | 2026-09-23T14:50Z | — |

## Registration `OLMO2_7B_TEMPLATE` (OLMo-2-7B's four checkpoints read through one chat template)

Internal name: `PREREG_OLMO2_7B_TEMPLATE_2026-09-24`

- **Bar.** Descriptive, with no bar and no outcome name, and one analysis path fixed before any generation: OLMo-2-7B's base, SFT, DPO and Instruct checkpoints are generated under the chat template the three aligned checkpoints share, the base reading that same string, and each stage change in the second-person rate is read with the registered stage tool, prompt-clustered intervals and a paired placebo; an output set that fails the content, empty-output or identical-string check is not read.
- **Outcome names.** none fixed
- **Score.** All four output sets passed the checks. Base to SFT -19.25, SFT to DPO -1.70 and DPO to Instruct -0.15 per 1,000 tokens, each interval excluding zero; the stage rule, read as written, gives the preference-stage branch while SFT carries 91% of the change. Agent predictions: three of six match.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_olmo2_7b_template_2026-09-24.md` | `939df037e46cb7e04d745c867d24c5b41b33e47735357e2bbd0f17188100b82b` | 2026-09-24T19:58Z | 2026-09-24T19:59Z |
| `prerun_notice_olmo2_7b_template_2026-09-24.md` | `67bf534d6ce94d45cd573a9e4f0ea83a32ba8de9e0353740b6115f90b9ae4fa7` | 2026-09-24T20:00Z | — |
| `prediction_olmo2_7b_template_executor_2026-09-24.md` | `bb2a1b2775a969187a27029fd30c006c4612cfb6790a20cd491870237fa1fc70` | 2026-09-24T20:00Z | 2026-09-24T20:22Z |

## Registration `TULU70B_TEMPLATE` (Tulu-3-70B's base, SFT and DPO checkpoints read through one chat template)

Internal name: `PREREG_TULU70B_TEMPLATE_2026-09-24`

- **Bar.** Descriptive, with no bar and no outcome name: Tulu-3-70B's base, SFT and DPO checkpoints are generated under the chat template SFT and DPO share, the base reading that same string, and both stage changes are read with the registered stage tool, prompt-clustered intervals and a paired placebo; an output set that fails the content, empty-output or identical-string check is not read, and a stage that cannot finish before a fixed time is not started.
- **Outcome names.** none fixed
- **Score.** All three output sets passed the registered checks; the base and SFT sets carry 18 and 69 empty generations, under the registered 5% bar, and dropping every pair with an empty side leaves both changes in place. Base to SFT +0.25 with an interval covering zero, SFT to DPO -5.45 with an interval excluding zero: under the chat template the preference stage carries the change, unlike the other four sequences read this way. The post-preference stage was not run. Agent predictions: three of six match.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_tulu70b_template_2026-09-24.md` | `c9258bba05d83cd63e6402a7111e18b25ff688e911173cff00359fd0deb6005a` | 2026-09-24T21:06Z | — |
| `prerun_notice_tulu70b_template_2026-09-24.md` | `38dfccc60cd84ced2b766b20b1b5e350b2b19a3298a56877a718a5ec4d3ec9c6` | 2026-09-24T21:07Z | — |
| `prediction_tulu70b_template_executor_2026-09-24.md` | `ffd3887ee8a0fe8adcf1edd1a195695b29d11900d5b16ac9bfbf8857dac9cf34` | 2026-09-24T21:07Z | — |

## Registration `QWEN_TABAN_KENDI_TEMPLATE` (four Qwen2.5 bases read through their own chat template)

Internal name: `PREREG_QWEN_TABAN_KENDI_TEMPLATE_2026-09-24`

- **Bar.** Descriptive, with no bar: for Qwen2.`5-1`.5B, -3B, -7B and -14B, whose base checkpoints ship a chat template, each base is read against its aligned checkpoint with both in their own template, on generations that already existed, with the pair reader used by the appendix readings.
- **Outcome names.** none fixed
- **Score.** "I" falls in all four; "you" falls in 3B and 7B, rises in 1.5B, where the degeneracy filter keeps 2,594 of 6,528 pairs, and is unresolved in 14B. Agent predictions: not yet scored.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_qwen_base_own_template_2026-09-24.md` | `44d1fa9b0e685cb91bbf830933d5db8b91cb12632cf5237de4174410242d0ba5` | 2026-09-24T21:14Z | — |
| `prerun_notice_qwen_base_own_template_2026-09-24.md` | `f3507726b32664df463cfbeb542119ede61c52a2836dd67bf518a323d09e587c` | 2026-09-24T21:14Z | — |
| `prediction_qwen_base_own_template_executor_2026-09-24.md` | `e269faac6ef502d235545d471ddb698c4fe9548ebbcdbc3723c31df9c4e9c25e` | 2026-09-24T21:15Z | — |

## Registration `JH_B100` (the cue classifier of Appendix D.1 against the author's labels for 100 second-person sentences)

Internal name: `PREREG_JH_B100_2026-09-25`

- **Bar.** Descriptive, with no bar and no outcome name: on the 100 sentences of the author-labelled package B, the precision of the cue classifier's generic label and of its addressed label against the author's labels, with the classifier and the answer parser embedded in the registration by body and `SHA-256`, an author label of undetermined counted as not matching, and model-clustered 95% intervals from 2,000 draws with a fixed seed.
- **Outcome names.** none fixed
- **Score.** Descriptive; no bar and no predictions. Read (card JH_B100): the classifier labels 17 of the 100; its generic label is right in 5 of 8 and its addressed label in 6 of 9 (intervals [0.25, 1.00] and [0.31, 1.00]); the author read 78 of the 83 it leaves undetermined as addressed or generic.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_jh_b100_2026-09-25.md` | `6a014d0b0e96b07944d096a7298f77a91c8f846a501081da7783489cd2304566` | 2026-09-25T18:09Z | — |
| `prerun_notice_jh_b100_2026-09-25.md` | `995424f856317c15dac00fb31d3d8d0e65db20866fd2a40e985a6ba93371a372` | 2026-09-25T18:09Z | — |

## Registration `JH_YARGIC_JENERIK` (the generic half of second-person use read with the validated label judge)

Internal name: `PREREG_JH_YARGIC_JENERIK_2026-09-25`

- **Bar.** Descriptive, with one decision rule and no outcome name: on raw continuation, for each of the 16 models, the change from base to aligned in the generic rate (the second-person rate times the judge-labelled generic share of its cell, from the second label judge's labels of 2,000 sentences), with prompt-clustered 95% intervals (1,000 draws) and a paired placebo (the base checkpoint's outputs split into halves within each prompt, 200 splits); the Discussion sentence on generic you stays if the rate falls in at least 12 of 16 models, goes to the authors if it rises in at least 12, and is removed otherwise.
- **Outcome names.** none fixed
- **Score.** Read (card JH_YARGIC_JENERIK): the generic rate falls in 16 of 16 models and rises in none; 12 of the falls have intervals below zero and 8 exceed their paired placebo; the Discussion sentence stays. No predictions were written: the numbers from which the statistic can be derived had been seen before registration, as the pre-run notice states.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `prereg_jh_judge_generic_2026-09-25.md` | `7bcca4b55ac9b0c8a191b92af287e75f92964ca53398bbcb6679b8aceab6702f` | 2026-09-25T19:49Z | 2026-09-25T20:12Z |
| `prerun_notice_jh_judge_generic_2026-09-25.md` | `12d851509672d0b460e7ec473157e0226a61e154e72c3c3d4bf775e5836398bb` | 2026-09-25T19:50Z | — |

## Verdict reports

Reports that scored the registrations above; released at camera-ready with them.

| file | SHA-256 | added (UTC) | last change (UTC) |
|---|---|---|---|
| `banka_u3.errata_u4_2026-08-26.md` | `a999cc7172a8cbcf36e7e6f3b1858807ad23512940264133674e5c228c4df00e` | 2026-08-26T12:33Z | 2026-08-27T13:12Z |
| `banka_u3_2026-08-23.md` | `c5e667b41b78b4544f499a728208a9739f99e22f44a5d1d2a0c61df197f01c5a` | 2026-08-23T15:07Z | 2026-08-31T08:20Z |
| `baraj_sorgu_judge_marj_2026-09-10.md` | `69bba32af72dc7b5d81c6d708a12f6cae03457e043e5169d23d21df29cdafffc` | 2026-09-09T13:36Z | — |
| `baraj_ysz_kopya_2026-09-15.md` | `c9f07d9820b4e4b5fa17e7f627e310a68ed52e4d492d7762bc46b7871ba88b01` | 2026-09-15T06:01Z | — |
| `composition_2026-08-31.md` | `74c7373a0edce3b9b97183e3bd251a0fdd2e7c3303a25edae13ee535d22e4081` | 2026-08-31T07:39Z | — |
| `dejenerelik_audit_2026-08-26.md` | `e367c9d211952044e435144f768c659f39fa9a5da1ecdf849b52dd5963ef36dd` | 2026-08-26T08:12Z | — |
| `dejenerelik_audit_2_2026-08-26.md` | `c05c4816bcfa5056732bf108bc3263003f0dce5ed2e85fb7e03a86f9c7720cd4` | 2026-08-26T09:20Z | — |
| `errata_iddia_defteri_2026-08-11.md` | `5506cdd0aae0874ad6328df083a75232928172e72e54814df83b9c786cd8ea08` | 2026-08-11T06:35Z | — |
| `errata_prereg_o2_maske_unit_2026-08-20.md` | `9eb8133646f80748ddb7a9b0303a3e7bd6352d0ea933feda8e341072730370d8` | 2026-08-20T11:41Z | — |
| `errata_yarim_2026-08-24.md` | `a3837a87e5cd3e4fbb845499bcd994cbe737f9fd264cfec04ffe64eda1f5be82` | 2026-08-24T06:21Z | — |
| `verdict_task_vector_angle_2026-09-12.md` | `865e319bd51ebb88f947e7db5c8bc09100cb00ef61b02db0efc6a58996e405b3` | 2026-09-12T23:45Z | — |
| `verdict_davranissal_anchor_2026-08-27.md` | `a8105626d418e68a0be7cd045319edc9950c951d87f2ac28f3593b1f27166b62` | 2026-08-27T23:36Z | — |
| `verdict_d2_c3_2026-08-28.md` | `93348d154f6654aa85a7330809ab4fd34746b00921bf4334323c47aa31658eb4` | 2026-08-28T09:23Z | 2026-08-28T12:20Z |
| `verdict_d_dpo_delta_2026-08-27.md` | `a000fe815ab4dbdc2a25b18a1023c49a5a693e4112fefa9b7b3e0b6e1f462b2b` | 2026-08-27T22:55Z | — |
| `verdict_e799_opening_2026-08-25.md` | `32a5424ae9baee4a76c03d44f3d32401d90975203397f9999d76abe6b5941e4e` | 2026-08-25T08:36Z | 2026-09-05T21:07Z |
| `verdict_plainness_2026-08-27.md` | `98c7b003ab1ea318bbcebb78f546e6dd5ecbc428652c691888f7c58ac10e4181` | 2026-08-27T22:26Z | — |
| `verdict_shadow_reading_16_2026-08-26.md` | `6214e0c6deaa0bd71f269399a30bae7aaf397ecd57321a4ec60611a1b9aa24e9` | 2026-08-26T04:14Z | 2026-08-26T13:38Z |
| `verdict_form_count_2026-08-28.md` | `21ef61ee79364140aeada222d3d2c4a39fc2e40af0486417c05ec24f4ae859f4` | 2026-08-28T09:34Z | — |
| `verdict_data_tilt_zero_difference_2026-09-14.md` | `ee8957a329ca30ce45f291f5c1059879f9d117f796d3b169b530bae20453a2db` | 2026-09-14T03:50Z | — |
| `verdict_data_tilt_neutral_seed2_2026-09-14.md` | `43022c2386df4179b2e91a403cffd6d74876169b518bd3da0e7d270c52d55b65` | 2026-09-14T09:02Z | 2026-09-14T09:55Z |
| `verdict_system_prompt2_2026-09-13.md` | `a64f3474656afafaa6fb09b52c264bc0ea27f9c3db4369754f231da35e1b0258` | 2026-09-13T10:25Z | — |
| `verdict_simpo_bare_2026-09-14.md` | `cc780c20d8e0be882f658d7b7f2dc5e0d4ca435f3878ee117dcadf7b69159157` | 2026-09-14T01:58Z | — |
| `verdict_capability_16_2026-09-13.md` | `f20869732098678f3e8c05400e9eb7839ea3d5fd42d35c6356f96f118bf136b3` | 2026-09-13T15:46Z | — |
| `verdict_capability_2026-09-13.md` | `37723a673586c3ef61b3dd84365d0f1250928002ef89287abe665ce717434b18` | 2026-09-13T09:00Z | 2026-09-13T15:46Z |
| `person_dial_minimal_pair_feasibility_2026-09-04.md` | `80969abe7a210bcc0254c106e687534db4f48f6daba6bcc76dd68d2be50596d4` | 2026-09-04T11:05Z | — |
| `komsu_11711_measurement_2026-09-01.md` | `671499db6469cc63a8cc5a1e2b58327b946703e852ea669e00433de727cbdfb0` | 2026-09-01T12:52Z | — |
| `komsu_validation_2026-09-01.md` | `cc2e79cdd8ef7b0eda0face19e056ef79f7e9ec94089c34fc25f3d107f9e29ec` | 2026-09-01T11:50Z | — |
| `komsu_register_2026-09-01.md` | `8a14179e97c85987940cd30e93f0973df2f152a1a129bdf14ff1298bc7890e56` | 2026-09-01T15:29Z | — |
| `window_package_2026-09-02.md` | `595cdc8b5d3b3dc7c8ddf9eb4812aa68b28ef2062fc53785e5fe75067d6aaf2e` | 2026-09-02T06:38Z | 2026-09-02T06:41Z |
| `failed_predictions_count_2026-09-14.md` | `1f7b0d0457b6805b7a9b7d4de3b61945311de8ac1af73f9f212bbd9e0c61ed12` | 2026-09-14T09:55Z | 2026-09-14T11:33Z |
| `filtered_2x2_verdict_2026-09-07.md` | `5ff36fb35b7139eda08e0ddd707636837baab2277d3ec93e3194830174f4619f` | 2026-09-07T09:39Z | — |
| `verdict_family_panel_2026-08-15.md` | `6e9c345a7b75d4d0dea5e3ad4a473d6dac991e147a84ff78cec54d828d39aa6b` | 2026-08-15T12:59Z | — |
| `verdict_impersonal_correction_seventh_family_2026-08-15.md` | `f978132e797931d3c1d1f565f07c1f70cde2fd871f9c4e31c3df6862f98a8fcb` | 2026-08-15T21:08Z | — |
| `verdict_addressee_2026-08-22.md` | `55bac7cb2f777252db75c71c38425ec4eac4741a0f9888548594cfd605791b6c` | 2026-08-22T21:13Z | — |
| `verdict_secim_queue_2026-08-27.md` | `36b280d5fab617cf16140295986c0c421c974aaadc5efb89801fe1806d052a9f` | 2026-08-27T12:06Z | — |
| `verdict_v100_kelime_payda_2026-09-22.md` | `39ffe10866374a83d5b23442eac93724c645175e3f0478c66e309f3467963915` | 2026-09-22T09:56Z | — |
| `verdict_v102_taze_uyum_2026-09-23.md` | `7d7e557b4e26987312a81e5c7f6ac62063852e2b6408a0b95b3505bc30d280c4` | 2026-09-23T07:32Z | — |
| `verdict_v103_diskten_2026-09-23.md` | `97e737a80659cf9cc5a9a34b4c8eb4bb84762226014081acb387ea884905a03c` | 2026-09-23T09:25Z | — |
| `verdict_v104_base_birinci_2026-09-23.md` | `d7cbd90ba45c7d28a281ce2cc6ef95e889faa1d6d421d187f17cd9a9c136283c` | 2026-09-23T14:16Z | — |
| `verdict_v105_template_birinci_suzgecsiz_2026-09-23.md` | `68faf7498fe5d99fa56ed3dc78eed57cf3ecfbab5b4776db5cfb85716ce9a52a` | 2026-09-23T15:21Z | — |
| `verdict_olmo2_7b_template_2026-09-24.md` | `beed44483be14bcbcdb0fc32f26a6a4b95751919b80be0618e03ffb8eb409079` | 2026-09-24T21:34Z | 2026-09-24T22:31Z |
| `verdict_tulu70b_template_2026-09-25.md` | `49646e931ceaa7bf59f47645a48af439957ad4415221181954fba56fb84b5c05` | 2026-09-25T02:30Z | — |
