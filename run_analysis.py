"""Rebuild every deliverable from a clean checkout, in dependency order.

There is exactly one computation path: the capstone notebook. This script does not re-implement
the analysis; it executes the notebook (which writes the figures and the call list), then
generates and test-runs the Kaggle edition, then builds the deck from the saved figures.

    python run_analysis.py              # everything
    python run_analysis.py --skip-kaggle
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
NB = PROJECT / "notebooks" / "00_bank_churn_final.ipynb"


def run(*args, cwd=PROJECT):
    print("$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def execute(notebook: Path, cwd: Path):
    run(sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=1800", notebook, cwd=cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-kaggle", action="store_true")
    a = ap.parse_args()

    run(sys.executable, "tools/build_final.py")
    execute(NB, NB.parent)                                   # computes everything, writes figures

    if not a.skip_kaggle:
        run(sys.executable, "tools/make_kaggle_notebook.py")
        kaggle_nb = PROJECT / "notebooks" / "kaggle" / f"{PROJECT.name}_kaggle.ipynb"
        # Prove it runs outside the repo: execute in an empty folder, as Kaggle would.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / kaggle_nb.name
            shutil.copy(kaggle_nb, copy)
            execute(copy, Path(tmp))
            shutil.copy(copy, kaggle_nb)

    run(sys.executable, "deck/build_deck.py")


if __name__ == "__main__":
    main()
