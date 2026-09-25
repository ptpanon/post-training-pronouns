# Record of failed predictions

Appendix B of the paper lists, one line per registration, the registrations that carry a bar or an outcome name fixed in advance, whose results the paper reports, and that carry at least one failed prediction. The counting rule was committed before the count (`rule_v90_failed_count_2026-09-17.md`). This record gives, for every failed prediction, where its wording and its score are written: file, line and the file's SHA-256. The files are the Turkish originals, released at camera-ready; `SEALS_INDEX.md` carries their bars and outcome names.

Of the 32 registrations counted, 19 carry at least one failed prediction, 46 failed predictions in all. Not counted: two operational predictions (whether runs would finish) and two base-rate statements.

## 1 · The registered bare-prompt direction of the second person (Depersonalization, $M_1$)

Internal name: `PREREG-9`

- **Failed prediction.** The second-person decline would appear only in the Tülu/OLMo lineage (a local pattern).
- **Measured.** The prediction failed on arm A and could not be measured on arm B.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `B9b` | `prereg_depersonalization16_2026-08-31.md`:291 | `handover_overnight_2026-08-31.md`:63 |

## 2 · The task-matched ground: the neutral panel's prompts inside the URIAL frame

Internal name: `PREREG-URIAL`

- **Failed prediction.** On the identical-string comparison, 6–8 families would fall and 2–4 rise with intervals excluding zero, the fallers mostly AI2 and Qwen and every riser Gemma or Llama. The base models' median in the URIAL frame would be 12–18 per thousand.
- **Measured.** 10 families fell and 6 rose, both outside their bands; 9/10 fallers were AI2 or Qwen, but one riser, OLMo-3-7B (AI2), was neither Gemma nor Llama. The base median was 20.13. (scored on the 1,024-token card)

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `URIAL-F2` | `prediction_urial_2026-09-07.md`:72 | `handover_2026-09-09t0822z.md`:58 |
| `URIAL-F3` | `prediction_urial_2026-09-07.md`:73 | `handover_2026-09-09t0822z.md`:58 |
| `URIAL-F5` | `prediction_urial_2026-09-07.md`:75 | `handover_2026-09-09t0822z.md`:58 |

## 3 · The default system line, present or absent inside the same string

Internal name: `M-SISTEM-2`

- **Failed prediction.** Among the interval-disjoint families, the signs of the shift would be indistinguishable from random (sign agreement u below 0.543).
- **Measured.** Sign agreement measured u = 0.5565, above the 0.543 bar.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `F-MS2-2` | `prediction_system_prompt2_2026-09-13.md`:37-38 | `verdict_system_prompt2_2026-09-13.md`:55 |

## 4 · Two ladders outside the AI2 and Zephyr recipes (SmolLM2-1.7B, NeuralHermes-2.5)

Internal name: `V61-MERDIVEN`

- **Failed prediction.** On both the SmolLM2 and Hermes ladders the outcome would be the one named «same step».
- **Measured.** Both ladders read «different step»: the supervised step moved −3.3356 (SmolLM2-1.7B) and −5.4761 (NeuralHermes-2.5) with intervals excluding zero, the preference step +0.0740 and +0.1497 with intervals covering zero.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `F-V61-1a` | `prediction_v61_2026-09-11.md`:70 | `prediction_v61_2026-09-11.md`:8 |

## 5 · Re-scoring person-swapped responses with two public reward models

Internal name: `V61-PUANLAMA`

- **Failed prediction.** Both reward-model scorers would read as penalising.
- **Measured.** Both scorers read as rewarding (Δ = −0.0004 and −0.5971, person-free minus person-marked score).

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `F-V61-2a` | `prediction_v61_2026-09-11.md`:72 | `prediction_v61_2026-09-11.md`:10 |

## 6 · The person dial (four arms of 7,537 preference pairs selected on person marking)

Internal name: `KISI KADRANI`

- **Failed prediction.** The unselected arm's shift against the supervised checkpoint would be −1.0 to −2.5. On the person arms force would separate from the persons, the force-selected arm would not train force at this dose, and the embedding estimator would move less than 1 null-sd in every arm.
- **Measured.** The layer-separation, force-below-dose and embedding-band predictions all failed on the same measurement, so the dial is a register dial rather than a person dial; the unselected arm's actual shift was +0.225.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `Q5` | `prediction_person_dial_2026-09-03.md`:15 | `report_person_dial_2026-09-04.md`:148-149 |
| `Q7` | `prediction_person_dial_2026-09-03.md`:17 | `report_person_dial_2026-09-04.md`:142-144 |
| `Q8` | `prediction_person_dial_2026-09-03.md`:18 | `report_person_dial_2026-09-04.md`:142-144 |
| `Q9` | `prediction_person_dial_2026-09-03.md`:19 | `report_person_dial_2026-09-04.md`:142-144 |

## 7 · A third targeted run on minimal pairs built by exchanging a single determiner

Internal name: `KK-2`

- **Failed prediction.** On the minimal-pair arms the mean dial magnitude would be 3.0–5.5 per thousand, force would move at z 2.0–6.0 on the person arms, and the downward shift would exceed the upward one. The embedding estimator would move less than 1 null-sd in every arm.
- **Measured.** The dial size, the force z, the down-over-up asymmetry and the embedding-estimator band were each scored as not holding. (arms below the degeneracy shelf; magnitudes not read)

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `Q3` | `prediction_person_dial_minimal_pair_2026-09-04.md`:13 | `report_person_dial_minimal_pair_2026-09-05.md`:78 |
| `Q5` | `prediction_person_dial_minimal_pair_2026-09-04.md`:15 | `report_person_dial_minimal_pair_2026-09-05.md`:80 |
| `Q8` | `prediction_person_dial_minimal_pair_2026-09-04.md`:18 | `report_person_dial_minimal_pair_2026-09-05.md`:83 |
| `Q9` | `prediction_person_dial_minimal_pair_2026-09-04.md`:19 | `report_person_dial_minimal_pair_2026-09-05.md`:84 |

## 8 · Pairs without person content

Internal name: `KK-3F`

- **Failed prediction.** The arm trained on pairs without person difference would not withdraw (the first-ranked of 7 outcomes, at 0.34), and it would withdraw less than the neutral arm, with a shift above −1.1224.
- **Measured.** The arm withdrew (the outcome given 0.26), with a shift of −1.2041; the ranking's Brier score was 0.774.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `F-KK3F-1` | `prediction_data_tilt_zero_difference_2026-09-13.md`:28 | `verdict_data_tilt_zero_difference_2026-09-14.md`:60 |
| `F-KK3F-2` | `prediction_data_tilt_zero_difference_2026-09-13.md`:29 | `verdict_data_tilt_zero_difference_2026-09-14.md`:61 |
| `F-KK3F-3` | `prediction_data_tilt_zero_difference_2026-09-13.md`:30 | `verdict_data_tilt_zero_difference_2026-09-14.md`:62 |

## 9 · The reference-free objective: released DPO and SimPO checkpoints from one supervised base

Internal name: `SimPO`

- **Failed prediction.** From the same supervised base, the SimPO step would withdraw less than the DPO step, with non-overlapping intervals.
- **Measured.** The SimPO step was larger than the DPO step; the DPO reference step read «same sign» and no leg fell below the degeneracy shelf, so the prediction was scorable.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `F-SIMPO-2` | `prediction_simpo_2026-09-10.md`:32 | `verdict_simpo_bare_2026-09-14.md`:38 |

## 10 · The scorer fit: a joint fit with first-person density and two surface controls

Internal name: `YARGIC-IMZASI-2`

- **Failed prediction.** In the joint fit the second-person coefficient would stay between 0.10 and 0.19 in absolute value, its margin over the other controls would fall below 1.5 (a general style penalty), and it would be largest on the helpfulness rating.
- **Measured.** The coefficient measured 0.1925, 0.0025 above the upper bound; the margin was 3.13; and the largest coefficient fell on instruction following.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `Q2` | `prediction_judge_signature_2_2026-09-03.md`:9 | `verdict_judge_signature_2_2026-09-03.md`:75 |
| `Q3` | `prediction_judge_signature_2_2026-09-03.md`:10 | `verdict_judge_signature_2_2026-09-03.md`:76 |
| `Q5` | `prediction_judge_signature_2_2026-09-03.md`:12 | `verdict_judge_signature_2_2026-09-03.md`:78 |

## 11 · The mechanical form counters (bare imperative, second-person obligation, hedge, interrogative directive, hortative)

Internal name: `FORM-SAYIM`

- **Failed prediction.** Aligned models would use fewer bare imperatives (≥4/6 families), more hedges (≥5/6) and more interrogative directives (≥4/6) than their bases, with the bare-imperative drop at the SFT→DPO rung on ≥2/3 ladders. The hedge counter would give the largest, most stable difference, sentences per generation would rise (≥4/6), the intact-only filter would change no cell's outcome, and hortatives would rise and clear the bar in ≥4/6.
- **Measured.** Against thresholds of 4, 5, 2 and 4, the first set measured 2/6, 1/6, 0/3 and 1/6; the second set measured 1, 3, 29 and 2 against thresholds of 4, 4, 30 and 4.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `BF-1` | `prediction_form.md`:7 | `verdict_form_count_2026-08-28.md`:227 |
| `BF-2` | `prediction_form.md`:8 | `verdict_form_count_2026-08-28.md`:228 |
| `BF-3` | `prediction_form.md`:9 | `verdict_form_count_2026-08-28.md`:229 |
| `BF-6` | `prediction_form.md`:12 | `verdict_form_count_2026-08-28.md`:232 |
| `PF-2` | `prediction_form_executor.md`:9 | `verdict_form_count_2026-08-28.md`:239 |
| `PF-3` | `prediction_form_executor.md`:10 | `verdict_form_count_2026-08-28.md`:240 |
| `PF-4` | `prediction_form_executor.md`:11 | `verdict_form_count_2026-08-28.md`:241 |
| `PF-6` | `prediction_form_executor.md`:13 | `verdict_form_count_2026-08-28.md`:243 |

## 12 · The reading cascade: identical text scored under a real and a matched pseudo-neutral context

Internal name: `F5-16`

- **Failed prediction.** On the CMV substrate, the share of families whose interpersonal-command channel separates positively would lie between 0.375 and 0.625.
- **Measured.** The CMV share measured 0.333.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `BF5-2` | `prediction_f5_2026-08-25.md`:7 | `verdict_shadow_reading_16_2026-08-26.md`:69 |

## 13 · Injecting the pre-specified weight-space dominance direction

Internal name: `D2V2-N`

- **Failed prediction.** For Tülu-8B and Zephyr-7B, the dominance-direction arm would register movement along that direction («on-axis»).
- **Measured.** Both read «off-axis»: the movement registered outside the dominance direction rather than along it.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `BDN-4` | `prediction_steering_2026-08-23.md`:15 | `verdict_steering_2026-08-23.md`:66 |
| `BDN-5` | `prediction_steering_2026-08-23.md`:16 | `verdict_steering_2026-08-23.md`:67 |

## 14 · The grammatical mood swap: rule-based imperative–request minimal pairs under a short and a long frame

Internal name: `E-YALINLIK`

- **Failed prediction.** A dominant-phrased request would register on the command side (the dominant-request trap cell firing) in at least 3 of 6 families.
- **Measured.** The dominant-request cell fired in 2 of 6 families (0.333).

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `BE-2` | `prediction_e.md`:7-8 | `verdict_plainness_2026-08-27.md`:80 |

## 15 · Undoing the preference delta by interpolation

Internal name: `D2`

- **Failed prediction.** The erasure would sit in the single DPO weight delta, so undoing that delta would work on OLMo-2-32B (a monotone curve approaching the base at α = −0.5) and on Zephyr (dSFT→dDPO).
- **Measured.** On OLMo-2-32B the curve was monotone but did not approach the base, and Zephyr read as scattered.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `BD2-1` | `prediction_d2.md`:9-11 | `verdict_d2_c3_2026-08-28.md`:24-27 |
| `BD2-2` | `prediction_d2.md`:12 | `verdict_d2_c3_2026-08-28.md`:119 |

## 16 · The behavioural anchor: the supervised weights stepped either way along the dominance direction at a fixed KL budget

Internal name: `C3`

- **Failed prediction.** The ruler would show the dominance direction on Zephyr, with both perturbation branches inside the bar, and on Tülu as well.
- **Measured.** Neither showed it: on Zephyr the plus branch cleared the bar (0.0000) and the minus branch did not (0.2000); on Tülu the branches read 0.3500 and 0.2833.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `BC3-1` | `prediction_c3.md`:8-9 | `verdict_d2_c3_2026-08-28.md`:125 |
| `BC3-2` | `prediction_c3.md`:10 | `verdict_d2_c3_2026-08-28.md`:126 |

## 17 · Rung-by-rung reading with the embedding estimator

Internal name: `P7-ÖLCEK2`

- **Failed prediction.** No pair's embedding-estimator interval would lie wholly below 0, the largest shift would fall at an interior rung in every family with at least 3 dense rungs, and Gemma-3-27B's dense pair would show no net erasure. On every OLMo-2 ladder the largest step would be SFT→DPO.
- **Measured.** OLMo2-32B's base→SFT interval lay wholly below zero ([−0.0347, −0.0048]), the Qwen2.5 maximum fell at the end rung (14B), and Gemma-3-27B moved +0.0314 with an interval excluding zero. The largest OLMo-2 step was SFT→DPO at 13B but DPO→Inst at 32B.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `B1` | `prediction_overnight_2026-08-21.md`:11-12 | `verdict_scale2_2026-08-22.md`:81 |
| `B2` | `prediction_overnight_2026-08-21.md`:13-14 | `verdict_scale2_2026-08-22.md`:82 |
| `B3` | `prediction_overnight_2026-08-21.md`:15-17 | `verdict_scale2_2026-08-22.md`:83 |
| `B5` | `prediction_overnight_2026-08-21.md`:20-21 | `verdict_scale2_2026-08-22.md`:85 |

## 18 · The first scorer fit, whose specificity margin set the later bar

Internal name: `YARGIC-IMZASI-1`

- **Failed prediction.** The second-person coefficient would stay under the floor, with |β| between 0.005 and 0.050, and the first-person coefficient would be larger in absolute value. The modal-density placebo would not clear the floor.
- **Measured.** The second-person coefficient measured 0.2116, 4.2 times the predicted ceiling; the first-person coefficient was smaller (0.179 against 0.212); and the placebo prediction did not hold.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `P2` | `prediction_judge_signature_2026-09-03.md`:13 | `verdict_judge_signature_2026-09-03.md`:78 |
| `P3` | `prediction_judge_signature_2026-09-03.md`:14 | `verdict_judge_signature_2026-09-03.md`:79 |
| `P4` | `prediction_judge_signature_2026-09-03.md`:15 | `verdict_judge_signature_2026-09-03.md`:80 |

## 19 · The low-dose targeted preference arms (four DPO arms on form-selected pairs)

Internal name: `TERAZI (KOL-GECESI)`

- **Failed prediction.** The dominance-axis shift would follow each arm's direction, negative for imperative-down and positive for imperative-up, with |Δ| above the MDE.
- **Measured.** Imperative-up came out negative and already below the bar, and no arm exceeded the bar.

| prediction | wording (file:line) | score (file:line) |
|---|---|---|
| `B1` | `prereg_arm_gecesi_2026-08-30.md`:50-51 | `terazi_verdict_2026-08-31.md`:21-22 |

## Scored and failed, not reported in the paper

- the injected-needle control (`R2`), `BR2-1`: wording `prediction_r2.md`:12 · score `verdict_r2_opening_2026-08-28.md`:102
- a six-counter reading of three training ladders (`PREREG-8`), `B8a`: wording `prereg_8_addressee_silme_2026-09-01.md`:67-68 · score `debts.md`:12947-12951

## Files

| file | SHA-256 |
|---|---|
| `debts.md` | `4ab66b3d5eef50a141edf1e3f4530c08b9c64c83ef200b041101180db446f575` |
| `handover_2026-09-09t0822z.md` | `aba426e237e6d0a25dcf36500253f95268b0070ccef274b3a5e17e8c493762c8` |
| `handover_overnight_2026-08-31.md` | `e76bbef7f43090d879775aebe941cd00f4b93092c8cffcd8d3841fe2cb4485ca` |
| `prediction_c3.md` | `ad7f295323de1eda37855d5b853ff67f5f8c3bf795cba7080355b5eeef6c04f7` |
| `prediction_d2.md` | `e73817289e8d1af146f97b609baa9fd12e6bc7fb9e88cfdc855b5efa4d257a30` |
| `prediction_data_tilt_zero_difference_2026-09-13.md` | `3a155ba889fac9ce09bb65ee74b8027dd37535ea34f57b22df5ab2aa9f12f960` |
| `prediction_e.md` | `08f36252aec0c143d6103c72e04695ed9d8820a5d4644cc694a48a2514cd8e8e` |
| `prediction_f5_2026-08-25.md` | `18660a4e5c09ebac0eb10a622ed7969f4684f9a619a9ecdd7e9bc47f7e1275f8` |
| `prediction_form.md` | `f1c0ba2ec7054cce84e2ca79cb9b5a441c8d794aa8dde93ed18b2070e522dc70` |
| `prediction_form_executor.md` | `9e06dc491f6213baa9fa11c07274de6e23e29a20d3fad9972c73956fce8e8b27` |
| `prediction_judge_signature_2026-09-03.md` | `c169b5ab8afe48875091772c0ba350a93eb538e6c466fdcc4549a616c1f6dcf5` |
| `prediction_judge_signature_2_2026-09-03.md` | `324c100a7c86910d01a695bfbae2043f5bde4ca93f27b982ab7983dc6c9cff5e` |
| `prediction_overnight_2026-08-21.md` | `873c994183e65e04e9448af2a1d57b5060138ed49e50df0d218436d471b8923a` |
| `prediction_person_dial_2026-09-03.md` | `1612a89322d4d413496d0abcd2b4744a302d944d734bee0c19c3e222463860d3` |
| `prediction_person_dial_minimal_pair_2026-09-04.md` | `5550542c5dde2be97540bfc0230d9e5ba7dd656317f8f42e6afea0029f13f53c` |
| `prediction_r2.md` | `6b9611914ce6568ae59ff38cc3c5e3f00718df5f33f63d7a4026af54e70c1dd0` |
| `prediction_simpo_2026-09-10.md` | `109778c1707fd3b96deb17d072dfc8d10de652712d0e1acdc9943bcf02cbbc17` |
| `prediction_steering_2026-08-23.md` | `4f0b4225c5b41556e3200706ebc4f8cf767f8fda2674d1303e3d968fe8b25270` |
| `prediction_system_prompt2_2026-09-13.md` | `7c60ba30c22fbef4526d08f7d8d90c9e6beefee5f13058f512317cae0fab30d0` |
| `prediction_urial_2026-09-07.md` | `46d02f04d8e18803b41745cc23c6038dfd7b546b18adbcaf550a408b44ff2268` |
| `prediction_v61_2026-09-11.md` | `68e8e606b517370e26e2f1793b370b42ba86229126d53d1837cea3496436bcbc` |
| `prereg_8_addressee_silme_2026-09-01.md` | `af9c749bb6259553a15873dd4eff7687185e06dbec1327d7283f9726176d9480` |
| `prereg_arm_gecesi_2026-08-30.md` | `833d5bc58237ca61c108fe3f06c9113c4584ab3cc9679954e11b19e8c3eff694` |
| `prereg_depersonalization16_2026-08-31.md` | `72a7c4f4968175a5a5b6f236437e75aa451e65e080ae39593c89c79a91c3e13c` |
| `report_person_dial_2026-09-04.md` | `b617b26de38d9d86e38dfe916054058c8675fc770b1069fdfaf7e40c1e3393f6` |
| `report_person_dial_minimal_pair_2026-09-05.md` | `932d8fe591fb1f14b21eafa90432c50f903448d03eb2dd21b3c0e2ad72acf880` |
| `rule_v90_failed_count_2026-09-17.md` | `f74a9776925c8506cdacf2c5ceec6d3abba81d3b3fcd18dbfb7ccdab950a9844` |
| `terazi_verdict_2026-08-31.md` | `b0546284499166907f4613ba3eb807239c787231af448d5764d9b3bc691ea7a3` |
| `verdict_d2_c3_2026-08-28.md` | `93348d154f6654aa85a7330809ab4fd34746b00921bf4334323c47aa31658eb4` |
| `verdict_data_tilt_zero_difference_2026-09-14.md` | `ee8957a329ca30ce45f291f5c1059879f9d117f796d3b169b530bae20453a2db` |
| `verdict_form_count_2026-08-28.md` | `21ef61ee79364140aeada222d3d2c4a39fc2e40af0486417c05ec24f4ae859f4` |
| `verdict_judge_signature_2026-09-03.md` | `ca1274ff8315062297ef52b422faa4918d7314d275f1887ac785748b374d514e` |
| `verdict_judge_signature_2_2026-09-03.md` | `16e192577e171637d417e696f65bf959338bc660c2c76c0ecf1c359eb34c9f9e` |
| `verdict_plainness_2026-08-27.md` | `98c7b003ab1ea318bbcebb78f546e6dd5ecbc428652c691888f7c58ac10e4181` |
| `verdict_r2_opening_2026-08-28.md` | `739d0ad8c5ccaaaf2e5d526c09230f94d93eca7ecc6d05f1dc9411aa6ac6e559` |
| `verdict_scale2_2026-08-22.md` | `27e5cf8f89357064371e3ddad32103cfe9457654b328ae25990c9b4b135bc4c0` |
| `verdict_shadow_reading_16_2026-08-26.md` | `6214e0c6deaa0bd71f269399a30bae7aaf397ecd57321a4ec60611a1b9aa24e9` |
| `verdict_simpo_bare_2026-09-14.md` | `cc780c20d8e0be882f658d7b7f2dc5e0d4ca435f3878ee117dcadf7b69159157` |
| `verdict_steering_2026-08-23.md` | `1ce06a74ad7b41960c8f83f8f1a43684c99ba9a722b8709fda0b6e147afb1613` |
| `verdict_system_prompt2_2026-09-13.md` | `a64f3474656afafaa6fb09b52c264bc0ea27f9c3db4369754f231da35e1b0258` |
