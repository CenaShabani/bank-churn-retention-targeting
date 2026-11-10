# `src/__init__.py`

## What this module is for
It marks `src/` as a package so the notebook can write `from src import models, business, ...`.
It holds a one-line docstring naming the project and nothing else.

## Public surface
None. Importing `src` runs no code.

## Walkthrough
A single docstring. Each module is imported explicitly by name, never re-exported here, so a
reader of the notebook can see which module every function comes from (`business.policy_table`,
not a bare `policy_table`).

## The non-obvious bits
The Kaggle generator (`tools/make_kaggle_notebook.py`) writes this file too, as the first
`%%writefile` cell, because without it `from src import ...` fails on Kaggle.

## How it can break
If it is deleted, every notebook import fails immediately and loudly, which is the right failure.
