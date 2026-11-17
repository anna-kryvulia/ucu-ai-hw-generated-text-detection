"""What is done and what still needs to run?  PYTHONPATH=src python3 scripts/status.py [--outputs PATH]"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", help="outputs folder (default: this machine's)")
args = ap.parse_args()
if args.outputs:
    os.environ["HWKIT_OUTPUTS"] = args.outputs

from hwkit.env import get_paths  # noqa: E402
from hwkit.tracker import status  # noqa: E402

pd.set_option("display.width", 200)
print(status(get_paths(mount_drive=False)).to_string(index=False))
