"""Get the raw data, prove it is the file we think it is, and load it with its schema checked.

The download goes through Kaggle's public dataset endpoint, which answers without an API
token for public datasets (it redirects to a signed storage URL). On Kaggle itself the file
is already mounted under /kaggle/input, so nothing is downloaded there.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd

from . import config

DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{config.KAGGLE_DATASET}"

EXPECTED_DTYPES = {
    "CreditScore": "int", "Geography": "object", "Gender": "object", "Age": "int",
    "Tenure": "int", "Balance": "float", "NumOfProducts": "int", "HasCrCard": "int",
    "IsActiveMember": "int", "EstimatedSalary": "float", "Exited": "int",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_raw_csv() -> Path | None:
    """Kaggle's mounted input first, then the local data/raw copy."""
    kaggle_hits = sorted(Path("/kaggle/input").rglob(config.RAW_FILE)) if Path("/kaggle/input").exists() else []
    if kaggle_hits:
        return kaggle_hits[0]
    local = config.RAW_DIR / config.RAW_FILE
    return local if local.exists() else None


def download() -> Path:
    """Fetch the CC0 dataset zip and extract the one CSV into data/raw/."""
    import requests

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(DOWNLOAD_URL, allow_redirects=True, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extract(config.RAW_FILE, config.RAW_DIR)
    return config.RAW_DIR / config.RAW_FILE


def load_raw() -> pd.DataFrame:
    """Load the raw table, downloading it once if needed, and fail loudly on schema drift."""
    path = find_raw_csv() or download()
    digest = sha256(path)
    if config.RAW_SHA256 and digest != config.RAW_SHA256:
        raise ValueError(f"{path} has SHA-256 {digest}, expected the pinned {config.RAW_SHA256}. "
                         "Every published number was produced on the pinned file.")
    df = pd.read_csv(path)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """The checks the rest of the code silently relies on."""
    missing = set(EXPECTED_DTYPES) - set(df.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")
    for col, kind in EXPECTED_DTYPES.items():
        ok = {"int": pd.api.types.is_integer_dtype, "float": pd.api.types.is_numeric_dtype,
              "object": pd.api.types.is_object_dtype}[kind](df[col])
        if not ok:
            raise TypeError(f"{col} should be {kind}, got {df[col].dtype}")
    if df[config.RAW_FEATURES + [config.TARGET]].isna().any().any():
        raise ValueError("unexpected missing values in the raw features")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise ValueError("target must be binary 0/1")
    if df["CustomerId"].duplicated().any():
        raise ValueError("duplicate CustomerId: one customer appears twice")
