"""Generate the Kaggle notebook from the GitHub capstone. Never edit the output by hand.

Kaggle cannot see this repo, so it has no `src/` to import. Rather than rewriting the code into
one flat namespace (which would change how every cell reads), the generator replaces the
capstone's setup cell with one `%%writefile` cell per `src/` module. Each cell recreates that
module verbatim under /kaggle/working/src/, and the capstone's own `from src import ...` then
works unchanged. The code a Kaggle reader sees is exactly the repo's code.

Data: `src/acquisition.find_raw_csv()` already looks under /kaggle/input first, so attaching the
dataset `barelydedicated/bank-customer-churn-modeling` is all the notebook needs. (Without it,
the notebook downloads the same file if internet is enabled.)

Run:  python tools/make_kaggle_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT = Path(__file__).resolve().parent.parent
CAPSTONE = PROJECT / "notebooks" / "00_bank_churn_final.ipynb"
OUT = PROJECT / "notebooks" / "kaggle" / f"{PROJECT.name}_kaggle.ipynb"
REPO_URL = f"https://github.com/Cena-S/{PROJECT.name}"

# Dependency order: config first, then the modules that import it.
MODULES = ["__init__", "config", "acquisition", "features", "eda", "models", "evaluate",
           "explain", "business", "figures"]


def build() -> Path:
    cap = nbf.read(CAPSTONE, as_version=4)
    present = {p.stem for p in (PROJECT / "src").glob("*.py")}
    if present != set(MODULES):
        raise SystemExit(f"src/ modules changed; update MODULES. src has {sorted(present)}")

    cells = [nbf.v4.new_markdown_cell(
        f"> **Kaggle edition, generated from the GitHub capstone.** The repository is canonical: "
        f"[{REPO_URL}]({REPO_URL}). This notebook is produced by `tools/make_kaggle_notebook.py`; "
        f"if anything here disagrees with the repo, the repo wins.\n>\n"
        f"> **Data:** attach the Kaggle dataset `barelydedicated/bank-customer-churn-modeling` (CC0).")]

    for cell in cap.cells:
        if "src-setup" in cell.metadata.get("tags", []):
            cells.append(nbf.v4.new_markdown_cell(
                "### Setup: the project's `src/` modules\n\nKaggle has no access to the repo, so "
                "each cell below writes one module from `src/` verbatim to `/kaggle/working/src/`. "
                "The analysis cells then import them exactly as the GitHub notebook does."))
            cells.append(nbf.v4.new_code_cell("import os\nos.makedirs('src', exist_ok=True)"))
            for m in MODULES:
                body = (PROJECT / "src" / f"{m}.py").read_text(encoding="utf-8")
                cells.append(nbf.v4.new_code_cell(
                    f"%%writefile src/{m}.py\n# --- from src/{m}.py in the repo (canonical) ---\n{body}"))
            src = cell.source.replace(
                "sys.path.insert(0, str(Path.cwd().parent))     # the project root, so `src` imports",
                "sys.path.insert(0, str(Path.cwd()))            # the src/ package written above")
            cells.append(nbf.v4.new_code_cell(src))
        elif cell.cell_type == "code":
            cells.append(nbf.v4.new_code_cell(cell.source))
        else:
            src = cell.source.replace(
                "python run_analysis.py      # downloads the data, executes this notebook, regenerates the Kaggle notebook and the deck",
                f"git clone {REPO_URL} && python run_analysis.py")
            cells.append(nbf.v4.new_markdown_cell(src))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = cap.metadata.get("kernelspec", {})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
