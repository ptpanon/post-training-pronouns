# Table B · Reviewer XghG, Q1: findings when task, role cues and format are varied separately

*Every count is read from the card named in its row; the cards are in `results/` of this repository.*

Counts are models, aligned minus base per 1,000 tokens. ↓ = interval excludes zero and falls · ↑ = interval excludes zero and rises · ~ = interval covers zero · /n = models read (95% prompt-clustered intervals; instruction-clustered on the ordinary instructions). † = fewer than 16 read, named below the table.

| Condition | Identical string? | Task requested? | Role cue in the prompt? | In its trained format? | "I" | "you" | Cards (in `results/`) |
|---|---|---|---|---|---|---|---|
| Raw continuation, four further seeds pooled | Yes | No | No | base Yes / aligned No | ↓16 ↑0 ~0 /16 | ↓16 ↑0 ~0 /16 | I: `results/v97_t1_dogrudan_birinci_2026-09-19.json`<br>you: `results/four_seed_level_2026-09-16.json` |
| Raw continuation, one seed (registered reading) | Yes | No | No | base Yes / aligned No | ↓16 ↑0 ~0 /16 | ↓16 ↑0 ~0 /16 | I: `results/reviewer_q1_panel_primary_2026-09-15.json`<br>you: `results/PREREG9_A_OLCUM_2026-08-31.json` |
| Chat template, filtered cells, one seed | Yes | No | Yes | base No / aligned Yes | ↓16 ↑0 ~0 /16 (derived) | ↓7 ↑7 ~2 /16 | I: `results/template_filtered_first_person_2026-09-15.json`<br>you: `results/template_filtered_2026-09-07.json` |
| Chat template, all cells, one seed | Yes | No | Yes | base No / aligned Yes | ↓16 ↑0 ~0 /16 | ↓8 ↑6 ~2 /16 | I: `results/v105_template_birinci_suzgecsiz_2026-09-23.json`<br>you: `results/v23_template_count_2026-09-06.json` |
| In-context prompt, each checkpoint in its own format | No | base Yes / aligned Partly | Yes | Yes | ↓16 ↑0 ~0 /16 | ↓11 ↑5 ~0 /16 | I: `results/reviewer_q1_urial_primary_2026-09-15.json`<br>you: `results/URIAL_VERDICT_2048_16_K2048_2026-09-11.json` |
| In-context prompt, identical string | Yes | Yes | Yes | base Yes / aligned No | ↓15 ↑0 ~1 /16 | ↓10 ↑5 ~1 /16 | I: `results/reviewer_q1_urial_primary_2026-09-15.json`<br>you: `results/URIAL_VERDICT_2048_16_K2048_2026-09-11.json` |
| Person-stripped in-context prompt, each checkpoint in its own format (six prefix conditions) | No | base Yes / aligned Partly | base Partly / aligned Yes | Yes | no card | ↓9 ↑5 ~1 /15 † | I: no card<br>you: `results/urial_verdict_impersonal_x1_2026-09-10.json` |
| ELICIT-99, each checkpoint in its own format | No | Yes | Yes | base Partly / aligned Yes | ↓15 ↑0 ~0 /15 (derived) † | ↓12 ↑0 ~3 /15 † | I: `results/P11_2X2_ELICIT_2026-09-03.json`<br>you: `results/P11_2X2_ELICIT_2026-09-03.json` |
| ELICIT-99, identical string | Yes | Yes | Yes | base No / aligned Yes | ↓15 ↑0 ~1 /16 | ↓12 ↑1 ~3 /16 | I: `results/v98_elicit_ozdes_2026-09-21.json`<br>you: `results/v98_elicit_ozdes_2026-09-21.json` |
| Ordinary instructions, each checkpoint in its own format | No | Yes | Yes | Yes | ↓12 ↑2 ~2 /16 | ↓16 ↑0 ~0 /16 | I: `results/v91_konusan_sayimi_2026-09-17.json`<br>you: `results/siradan_instruction_reading_2026-09-17.json` |
| Ordinary instructions, identical string | Yes | Yes | Yes | base Yes / aligned No | ↓9 ↑1 ~6 /16 | ↓11 ↑3 ~2 /16 | I: `results/v96_q4_urial_ozdes_2026-09-19.json`<br>you: `results/v96_q4_urial_ozdes_2026-09-19.json` |
| Ordinary instructions, identical person-stripped string | Yes | Yes | Partly | base Yes / aligned No | ↓10 ↑1 ~5 /16 | ↓6 ↑7 ~3 /16 | I: `results/v96_k3_impersonal_ozdes_2026-09-19.json`<br>you: `results/v96_k3_impersonal_ozdes_2026-09-19.json` |
| Ordinary instructions, base on the person-stripped frame, aligned in its own format | No | Yes | base Partly / aligned Yes | Yes | ↓12 ↑2 ~2 /16 | ↓16 ↑0 ~0 /16 | I: `results/v96_q4_urial_ozdes_2026-09-19.json`<br>you: `results/v96_q4_urial_ozdes_2026-09-19.json` |

† Person-stripped in-context row: Mistral-7B-v0.3 not read (its person-stripped prompt exceeds the 1,024-token window its corpus was generated at). ELICIT-99 own-format row: Qwen2.5-1.5B not scored (base outputs below the degeneracy floor).
"I" is a direct count (capitalised US excluded) except where marked derived (all first- and second-person forms minus the second-person count).

**What no condition combines.**
- The submitted version's Limitations said: "No condition here is at once identical-string, in-distribution for the aligned checkpoint and task-posing. Each meets at most two." Twelve of the thirteen rows above fit that sentence. The revision corrects it.
- ELICIT-99 on the identical string does not: both checkpoints read the aligned model's chat template, the strings match in all 16 models, and the prompts ask for a reply. There "I" falls in 15 of 16, and "you" falls in 12, rises in 1 and does not separate in 3.
- That reading is descriptive, with no bar. 9 of the 12 "you" falls and 11 of the 15 "I" falls clear the paired placebo. On prose lines alone "you" splits (5 fall, 5 rise, 6 do not separate) and "I" falls in 10 and rises in 0.
- What no row combines is an identical string, a requested task and both checkpoints in their own trained format. In the ELICIT-99 row the base reads a template it was not trained on, so attribution to weights alone stays limited there too.
- No row poses a task without a role cue, so these prompts do not vary task and role cue separately. The person-stripped frames remove the second person from the cue and keep the assistant role.

**Basis for the three design columns.**
- *Task:* the prompt requests a response. The debate prompts alone request none; ELICIT-99 and the ordinary instructions do; the in-context frame poses the assistant task.
- *Role cue:* the string names speakers or gives the model a turn: template turn markers and system lines, the `User:` / `Assistant:` scaffold, or the `# Query:` / `# Answer:` frame.
- *Trained format:* aligned means its own chat template. Base means plain text, which the paper extends to the plain-text in-context frame ("each checkpoint is prompted in the format it was trained for").
