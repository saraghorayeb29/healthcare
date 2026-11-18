# HRV and ECG coursework

Original code from Group 18's third-year project at Imperial College London in 2022.
Sara Ghorayeb contributed to this group project. This is not solely authored work.

The code processes ECG recordings, extracts snippets, detects R peaks, calculates HRV parameters, and explores classification.
The source is republished from [molbioeng/Molecool](https://github.com/molbioeng/Molecool).

## Files

- `legacy/*.ipynb`: all 11 original top-level notebooks, with saved outputs and metadata removed.
- `legacy/misc/*.py`: all 10 original Python source modules.
- `tests/test_legacy.py`: publication checks and synthetic smoke tests for the original noise and filtering modules.
- `requirements-test.txt`: dependencies for those checks only.

## Run the checks

From this project folder, with Python 3.12:

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The tests check notebook structure, Python syntax, and removal of private paths and record identifiers.
They also run original noise generation and filtering functions on synthetic arrays.
They do not execute the historical notebooks or reproduce the project's clinical results.

## Historical notebook use

Open notebooks with `legacy/` as the working directory. Configure your own authorized dataset before running them.
Paths now point to `data/raw/` and `data/processed/Database1/` beneath `legacy/`.
Generic `subject_###` strings replace original record identifiers. They are placeholders, not included patient data.

The clinical database, processed recordings, and historical results are not included.
The test requirements are not a complete notebook environment.
Historical imports also include pandas, matplotlib, pyarrow, tqdm, neurokit2, biosignalsnotebooks, pyECG, peakutils, seaborn, PyWavelets, and padasip.

These notebooks remain coursework, not a maintained clinical pipeline.
Known limitations include missing names in some notebook imports, notebook-only execution, and unfinished helper functions.
The original pNN50 implementation and frequency-domain calculations also need scientific review before reuse.
No clinical validity or historical end-to-end execution is claimed.

## Attribution and license

Read [ATTRIBUTION.md](ATTRIBUTION.md) before reusing the group source.
The root MIT license does not apply to `legacy/`.
