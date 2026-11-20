# HW3 workflow

## Layout

```
notebooks/      one notebook per task, numbered 01_, 02_, ... (copy _template.ipynb)
configs/        experiments/<run_id>.yaml, one config per experiment (`base:` inherits another config)
src/hwkit/      helpers shared by all notebooks
  env.py          detects Mac / Colab / Kaggle, sets data + output folders, seed, warning log
  config.py       YAML configs with `base:` inheritance
  tracker.py      runs: stages done, per-epoch history, metrics, experiments.csv (crash-safe)
  download.py     gets the competition data (Kaggle API) on Colab or the Mac
  submission.py   writes and checks the Kaggle submission CSV
  split.py        load_split(): fixed validation split (fold 0) + test-like weights; scores(): macro-F1 etc.
scripts/        status, collect_results, clean_notebook
results/        small files committed to git: experiments.csv, runs/<id>/ metrics & history, split.csv.gz
submissions/    submitted CSVs
images/         figures for the report
data/, outputs/ local only, git-ignored
```

## Where things run

- **Mac:** EDA, results, report. Data goes to `data/` (git-ignored).
- **Kaggle / Colab:** training. Push to GitHub first: the setup cell of each notebook clones the repo
  and installs `requirements.txt`.
  - Kaggle: create the notebook from the competition page (the data is attached), turn on Internet and
    the GPU. Outputs go to `/kaggle/working/outputs`. To keep them, *Save Version*, then attach that
    version's output to the next session: finished runs are restored automatically.
  - Colab: put `KAGGLE_API_TOKEN` in Colab Secrets; the data is downloaded once and cached on Google
    Drive (`MyDrive/ucu-hw3/`), and outputs are saved on Drive.

## Runs

Each run lives in `outputs/runs/<run_id>/` with `config.yaml`, `stages.json`, `history.csv`,
`metrics.json`, `checkpoints/` and `predictions/`. Guard each step:

```python
run = Run("baseline", cfg, P)
if not run.is_done("train"):
    ...                                   # resume from run.ckpt("last.pt") if it exists
    run.mark_done("train")
if not run.is_done("eval_val"):
    run.log_metrics("val", {"score": ..., "accuracy": ...})
    run.mark_done("eval_val")
```

Re-running a notebook then skips finished work. `status(P)` shows what is done.

## Results into the repo

```bash
PYTHONPATH=src python3 scripts/collect_results.py --src "<Kaggle or Drive outputs folder>"
```

Copies metrics, histories and configs (not checkpoints or predictions) into `results/`, and figures into `images/`.

```bash
python3 scripts/clean_notebook.py notebooks/03_something.ipynb
```

Removes Kaggle's progress-bar widgets, which GitHub cannot render.
