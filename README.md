# Healthcare projects

Code from Sara Ghorayeb's healthcare projects, organized by project.

- [PDAC liquid biopsy](pdac-liquid-biopsy/): biomarker panel analysis on public supplementary data.
- [HRV and ECG](hrv-ecg/): original Imperial College London group coursework with synthetic smoke tests.
- [Counterfactual image generation](counterfactual-image-generation/): paper link only, with no code copied or published.

Counterfactual project code is omitted. Its project entry links to the public arXiv paper only.

## Run the checks

Use Python 3.12. Each command runs from the repository root.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e './pdac-liquid-biopsy[test]'
python -m pip install -r hrv-ecg/requirements-test.txt
(cd pdac-liquid-biopsy && pdac-panel fetch && python -m pytest -q && pdac-panel run)
(cd hrv-ecg && python -m pytest -q)
```

The PDAC command downloads two public spreadsheets and verifies their pinned SHA-256 hashes.
It writes analysis results locally. Neither the spreadsheets nor generated results are committed.
HRV checks use synthetic arrays and compile notebook cells without executing the historical clinical analysis.

No clinical database, saved notebook outputs, manuscripts, account credentials, or employer material is included.
This code is for research and coursework, not clinical diagnosis.

## License

[LICENSE](LICENSE) grants MIT permission for Sara's own code, tests, and repository configuration.
The original group source under `hrv-ecg/legacy/` is explicitly excluded from that grant.
Read [HRV attribution](hrv-ecg/ATTRIBUTION.md) for its source and reuse limits.
