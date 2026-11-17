"""Get the competition data onto whatever machine we are on.

* Kaggle notebook: nothing to do, the data is attached read-only under /kaggle/input.
* Colab: the first time, download the zip with the Kaggle API and keep a copy on Google Drive.
  Afterwards unzip it from Drive to the fast local disk (/content/data) at the start of each session.
* Mac: the same download into <repo>/data (git-ignored), or put the files there by hand.

Kaggle credentials, any of:
  * new API token: ~/.kaggle/access_token file, or the KAGGLE_API_TOKEN environment variable
  * legacy key:    ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY environment variables
On Colab, store KAGGLE_API_TOKEN (or the legacy pair) in the "Secrets" panel (key icon).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from .env import COMPETITION, DRIVE_ROOT, Paths

REQUIRED = ["train.csv", "test.csv", "sample_submission.csv"]


def _load_colab_secrets() -> None:
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        if k in os.environ:
            continue
        try:
            os.environ[k] = userdata.get(k)
        except Exception:          # secret not defined or notebook access not granted
            pass


def is_ready(data_root: Path) -> bool:
    return all((data_root / f).exists() for f in REQUIRED)


def ensure_data(paths: Paths, drive_cache: bool = True) -> Path:
    """Returns the data folder; downloads and unzips the competition data if it is not there yet.
    drive_cache=False (Colab): keep the zip on the local disk only, when Google Drive has no room."""
    data = paths.data
    if paths.platform == "kaggle" or is_ready(data):
        print(f"data ready at {data}")
        return data
    data.mkdir(parents=True, exist_ok=True)
    _load_colab_secrets()

    zip_name = f"{COMPETITION}.zip"
    on_drive = paths.platform == "colab" and drive_cache
    cached = DRIVE_ROOT / "data" / zip_name if on_drive else paths.cache / zip_name
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(cached.parent)],
                       check=True)
    local_zip = cached
    if on_drive:                                  # copy once to the local disk, unzip from there
        local_zip = paths.cache / zip_name
        if not local_zip.exists():
            shutil.copy(cached, local_zip)
    print(f"unzipping {local_zip} -> {data}")
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(data)
    if paths.platform == "colab":
        local_zip.unlink()
    return data
