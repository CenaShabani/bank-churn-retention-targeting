# `src/acquisition.py`

## What this module is for
It gets the raw CSV, proves it is the exact file every published number came from, and loads it
with its schema checked. It is the first thing the notebook calls; it uses `config` only.

## Public surface
- `load_raw() -> pd.DataFrame`: 10,000 × 14, validated.
- `find_raw_csv() -> Path | None`: `/kaggle/input/**/Churn_Modelling.csv` first, then `data/raw/`.
- `download() -> Path`: fetch the zip from Kaggle's public endpoint and extract the CSV.
- `validate(df) -> None`: raises on schema drift.
- `sha256(path) -> str`.

## Walkthrough
`load_raw` finds or downloads the file, compares its SHA-256 with `config.RAW_SHA256` and refuses
to continue on a mismatch. Then it reads the CSV and calls `validate`, which checks four things the
rest of the code silently assumes: the expected columns with the expected types, no missing values
in features or target, a binary 0/1 target, and one row per `CustomerId`.

## The non-obvious bits
- **No API token needed.** `https://www.kaggle.com/api/v1/datasets/download/<owner>/<slug>`
  answers anonymously for public *datasets* with a redirect to signed storage. (Competition data
  does not, which is why the source's competition file isn't used.)
- **Kaggle first.** On Kaggle the dataset is mounted under `/kaggle/input`, so nothing is
  downloaded; locally the file lives in `data/raw/` (git-ignored).
- **The checksum runs before parsing**, so a silently re-uploaded dataset fails at load time
  instead of producing slightly different numbers.
- The type check for floats uses `is_numeric_dtype`, because pandas may read an all-integer float
  column as int.

## How it can break
- No network and no local file: `download` raises an HTTP or connection error.
- The dataset owner re-uploads a changed file: the SHA check raises, naming both hashes.
- Tests pin `validate` (duplicate customer, missing column, clean data). The download itself is not
  unit-tested; the notebook run exercises it, and the Kaggle edition's test-run in an empty folder
  goes through the download path.
