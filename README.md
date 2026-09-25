# post-training-pronouns

Code, pre-registrations, measurement cards and generation manifests behind the paper
*Post-Training Withdraws "I" from Language Models, and Pipelines Disagree About "You"* (anonymous submission).

## Layout

- `paper/figures/` holds the generators (`*.py`) that write every generated table, figure and sentence of the paper. Next to each are its outputs (`*.tex`) and a provenance record (`*.meta.json`) listing the cards it read.
- `paper/checks/` holds the checks run when the paper is built: generated-layer freshness, printed table rows, reference targets, count consistency, anonymity and draft phrases.
- `results/` holds the measurement cards (JSON). Every number in the paper is read from one of these files.
- `SEALS_INDEX.md` indexes the registrations (pre-registrations, pre-run notices, predictions and reading rules), each committed before the run or reading it describes: for each, the bar in one English sentence, the outcome names fixed in advance, the recorded score, and every file's SHA-256 digest and commit timestamp. Pre-registrations are released as an index carrying each registration's bar, outcome names, commit timestamp and digest; the original documents follow at camera-ready. The verdict reports are listed there with their digests.
- `FAILED_PREDICTIONS.md` is the record of failed predictions that Appendix B of the paper points to: for every failed prediction, the file and line where its wording and its score are written, with the file's digest.
- `COMPANION_MATERIAL.md` holds the appendix passages cut from the submitted version, unchanged and one section each: the mood-swap instrument, the two dominance-lexicon passages, the instruments-that-mislead paragraph, the three weight-space readings (training stages with the embedding estimator, the gradient direction, undoing the preference delta) and their three figures, the exploratory profile map and reading cascade, the closing permutation check, the assistant-role anchor cell, the reader test, and four shorter passages. Two figure references in the paper point here.
- `preregistration/` holds only an extract of one prediction file (one line of probabilities) that a figure generator reads.
- `scripts/` holds the measurement and generation code that produced the cards.
- `generation_manifests/` holds one manifest per generated output set (one checkpoint on one prompt set): model snapshot, decoding parameters, seed and row counts. The generated texts themselves are not included.

## Installation

```
python3.10 -m venv .venv && . .venv/bin/activate
pip install "numpy==2.2.*" "matplotlib==3.10.*" "pypdf==6.14.*" "PyMuPDF==1.28.*" "scikit-learn==1.7.*"
```

This is enough to re-run the figure generators and the checks. The measurement scripts additionally need
`torch`, `transformers` and `spacy` (with `en_core_web_sm`), the model weights named in each generation manifest,
and GPUs.

## Requirements

Figures and checks need Python 3.10 with numpy 2.2, matplotlib 3.10, pypdf 6.14, PyMuPDF 1.28 and scikit-learn 1.7.
The scripts in `scripts/` also use torch, transformers and spaCy. They need model weights and GPUs, and are released so the measurements can be inspected.

## Reproducing the generated layer of the paper

```
export DNH_ROOT="$(pwd)"
export DNH_DATA="$(pwd)/generation_manifests"
python paper/checks/gate_generated_layer_fresh.py paper/figures
```

The check re-runs every generator from the cards and compares each output with the file in this tree.
On this tree it reproduces 63 generators with no stale output.
5 generators need data that is not released, and they exit with an error:

- `paper/figures/example_pair.py`: prints model continuations; needs the generated texts and the original commit history.
- `paper/figures/f0_example.py`: prints the model continuations of Figure 0; needs the generated texts.
- `paper/figures/example_template.py`: prints model continuations; needs the generated texts and the original commit history.
- `paper/figures/mood_swap_ruler_figure.py`: needs the Warriner et al. (2013) affective norms lexicon, available from its authors.
- `paper/figures/new_figure_set.py`: needs the Warriner et al. (2013) affective norms lexicon, available from its authors.

## Notes

- All paths are resolved from `DNH_ROOT` and `DNH_DATA`.
- File and identifier names were normalised for release, and code comments and docstrings were removed mechanically; each Python file was checked to parse to the same syntax tree as its original with those removed. The SHA-256 prefixes inside `*.meta.json` refer to the files as they were when each output was generated.
- The model checkpoints trained for the paper are not included. They will be released at camera-ready, together with the original registration documents that `SEALS_INDEX.md` indexes by digest.
- Paths to files that are not part of this release were rewritten to `unreleased/<file name>`. Free-text notes and messages in `scripts/` (log lines, reasons returned next to an outcome name, notes written into cards) were emptied mechanically; outcome names, keys, patterns and comparisons were left as they are, and every edited Python file was checked to parse to the same syntax tree with its string contents normalised.

## Vocabulary of the registrations

The paper uses one vocabulary (Appendix Table A7). The registrations, and therefore `SEALS_INDEX.md`,
`FAILED_PREDICTIONS.md` and `COMPANION_MATERIAL.md`, keep the working names they were written with:
`rung` = training stage · `ladder` = checkpoint sequence · `bare` = raw continuation · `templated` = chat template ·
`ground` = prompt set · `task-matched ground` = in-context assistant prompt · `leg` = one checkpoint's outputs ·
`arm` = a training run, a prefix condition or a judge, by context · `tilt` = pronoun gap · `dose` = data size ·
`family` = model (one base and its aligned checkpoint).

## Naming

File names, JSON keys and many function names are Turkish words written without diacritics, kept as they were when the results were produced so that every recorded path and key still resolves. `GLOSSARY.md` gives the English meaning of the recurring parts, with counts measured on this tree.

## License

Code (`scripts/`, `paper/figures/*.py`, `paper/checks/`) is released under the Apache License 2.0 (`LICENSE`).
Measurement cards, generation manifests, generated tables and the documentation files are released under
Creative Commons Attribution 4.0 International (`LICENSE-CC-BY-4.0.md`).
