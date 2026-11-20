# Instructions for Claude (UCU AI 2026, HW3)

Read this file first. It holds everything decided in earlier homeworks, so the user does not have to
explain it again. `WORKFLOW.md` describes the repo layout and the helper code.

## The user

Anna Kryvulia, student of the UCU AI 2026 course (GitHub `anna-kryvulia`, Kaggle `annakryvulia`), works on
a Mac. Background in CV/ML is limited: explain every term fully, in plain words, the first time it appears.

## How to work with the user (rules from earlier homeworks)

- **Do not execute code yourself.** Write the code and give the commands; the user runs them and pastes
  the output. If the next piece of code depends on an earlier output, ask the user to run it first.
- On the Mac use `python3`, not `python`. Give each shell command in its own ```bash block.
- The user commits and pushes. Give the git commands; do not commit on your own.
- Notebooks: every code cell starts with a one-line comment saying what it does. No meta notes in
  notebooks (no "✍️ Your conclusions", no "Runs on ..." hints), no empty cells.
- **Notebooks must run without errors or warnings** (graded). Fix the cause of a warning instead of
  hiding it. `env.setup()` logs every distinct warning; the last cell of each notebook prints
  `env.warnings_summary(P)`, which should be empty.
- Fold bonus tasks into the main tasks instead of separate notebooks.
- Training runs on GPU on Kaggle (30 GPU hours per week) or Colab; the Mac is for EDA, results and the report.
- **No models, libraries or tools of Russian origin.** Check the origin before suggesting anything.
  Known cases: WBF / `ensemble-boxes` (Russian author), Albumentations (mostly Russian founders),
  CatBoost (Yandex), ruBERT and other Russian-made models.
- The report (`report.md`) is written by the user. Help with plans, tables (Markdown pipe tables),
  checking numbers and wording, but keep it in the user's voice.

## HW3 deliverables and grading

HW3 has no GitHub Pages report template (unlike HW2): code and the report go to the **LMS**, followed
by a one-on-one defense. The report is **1–2 pages of text** (tables and figures may go in an
appendix); longer reports lose points. It must not be written by AI tools (code may be). It names the
two selected Kaggle submission files.

| Part | Points | What they expect |
|---|---|---|
| Private leaderboard | 2 | normalized to 0.5–2.0 by final private standing |
| EDA | 1 | class balance by language and domain, text lengths, examples of each class, how these inform choices; what makes fusion different |
| Baseline | 1 | TF-IDF word/char n-grams + classical ML, small parameter search, per-class metrics, confusion matrix |
| Encoder | 1 | fine-tuned multilingual transformer (XLM-R, mDeBERTa), handling of texts longer than max length, comparison with the baseline on the same split |
| Your approach | 2 | at least two further experiments, each vs a fixed reference (long-text handling, other backbones, perplexity features, stylometry, ensembles, class weights / thresholds for macro-F1, ...) |
| Report and defense | 3 | methodology, results incl. failed attempts, final model with run instructions; ability to explain |
| Bonus: xAI | +1 | SHAP, LIME or token attributions, with conclusions |

The defense can lower any part if the work cannot be explained or reproduced. Points are deducted for
long unfiltered printouts, runtime errors and unstructured code. Meaningful experiments count even if
they do not improve the score.

Competition rules: no proprietary LLMs (GPT, Claude, Gemini) to label, generate or extract features;
open-weight models are fine. No external data (the source dataset included): train on `train.csv`
only, pretrained weights allowed. Fit vectorizers / scalers / feature selection on train rows only.
Independent work; cite external code and ideas.

## HW3 facts

- Kaggle competition: `ai-generated-text-detection-hw3-ucu-ai-2026`.
- Files: `train.csv` (272,352 rows: `id, text, lang, domain, label`), `test.csv` (25,000 rows: `id, text,
  lang, domain`), `sample_submission.csv` (`id, label`).
- Labels in train: `ai` 121,940, `fusion` 83,682, `human` 66,730.
- Languages: `en-EN` and `de-DE` (about 81% / 19% in train, 85% / 15% in test).
- Domains (8): student_essays, bundestag, cnn_news, blog_authorship_corpus, house_of_commons,
  euro_court_cases, arxiv_papers, gutenberg. Their shares differ between train and test.
- Source: a processed subset of GhostWriter (Schaaf, Bönisch, Mehler, LREC 2026). Texts are 20–3000
  words (median 463). **A human text and its ai/fusion versions are always on the same side of the
  train/test split**, but there is no group column: rebuild the groups (e.g. by text overlap) and
  split validation by group.
- Metric: macro-F1 over human / ai / fusion (`f1_score(y, pred, average="macro")`), also for validation.
- Leaderboard: public 30% of test (7,501 rows), private 70% (17,499). 5 submissions per day, select 2 final.
- Deadlines (23:59 Kyiv): Kaggle final submission **2026-11-23**; report and code on LMS **2026-11-24**.
- A 20–50k fixed subsample is enough for a transformer; keep it fixed when comparing models.
- Local data lives in `<repo>/data/` (git-ignored), where `env.py` looks for it.
- Validation: `results/split.csv.gz` from `notebooks/01_eda.ipynb` (source groups by TF-IDF linking,
  10 stratified group folds, fold 0 = validation, `weight` = test-like domain weights, `sub40k` = fixed
  transformer subsample). Load with `hwkit.split.load_split`; score with `hwkit.split.scores` and compare
  models on `macro_f1_testlike`.

## Lessons from HW2 (object detection)

- **Validation must look like the test set.** In HW2 the val frames had much larger resolutions than
  test, and plain val mAP was misleading; a "test-like" val subset fixed that. Here compare `lang` and
  `domain` shares of train and test and build validation to match.
- **Avoid leakage in the split.** Texts that share a source (same document, author, prompt) must not be
  split between train and val.
- **Do not tune on the public leaderboard.** It is a small part of the test set (30% in HW2); the grade
  uses the private part. Choose settings on validation, and pick two final submissions on Kaggle.
- One change per experiment against a baseline, so each effect is measured on its own.
- Save predictions on val and test for every run: ensembles and post-processing then need no retraining.
- pandas pitfall: `df.clip` is a DataFrame method, not a column; use `df["clip"]` style for columns
  whose names clash with methods.
- Kaggle: "Copy & Edit" to make your own copy; "Save Version" to keep outputs; attach a saved version's
  output as an input of the next session to continue training.
