# Liquid biopsy panels for pancreatic cancer

In December 2022 I wrote a research proposal for my MEng at Imperial College London: *Liquid Biopsy Biomarkers for the Early Detection of Pancreatic Ductal Adenocarcinoma (PDAC) using Machine Learning*. It set a plan and it held no results. This repository runs that plan on the public data that the proposal named.

**The claim to test.** A panel that joins two families of blood biomarkers, tumour DNA and proteins, detects PDAC better than a panel from one family.

## Result

Data: 221 people with resectable PDAC and 182 healthy people, from Cohen et al. 2017. One third is held out (74 patients, 61 healthy people).

- All six candidate biomarkers pass the selection step. They give 63 panels and 252 models.
- 7 of 63 panels pass the three criteria of the proposal. Each of the 7 joins a DNA marker with protein markers. No panel from one family passes.
- The selected panel is KRAS + CA19-9 + CEA + HGF + OPN.

| Panel, on the 135 held-out people | AUC [95% interval] | Patients detected | Healthy people called positive |
|---|---|---|---|
| Selected: KRAS + CA19-9 + CEA + HGF + OPN | 0.939 [0.893, 0.975] | 63 of 74 | 3 of 61 |
| Best panel of proteins only: CA19-9 + CEA + HGF + OPN | 0.914 [0.864, 0.957] | 52 of 74 | 3 of 61 |
| CA19-9 alone | 0.911 [0.861, 0.954] | 47 of 74 | 0 of 61 |
| Best panel of DNA markers only: KRAS + cfDNA | 0.752 [0.669, 0.831] | 30 of 74 | 1 of 61 |
| The test of the paper, fixed thresholds | no score | 53 of 74 | 1 of 61 |

The AUC gain of the two-family panel over the protein panel is +0.025, with a 95% interval from -0.006 to +0.061. The direction supports the claim. The size of the gain is not established, because the interval includes zero.

## Run it

```
python -m pip install -e ".[test]"
pdac-panel fetch
python -m pytest -q
pdac-panel run
```

`fetch` downloads Datasets S1 and S2 of the paper and checks their SHA-256 hashes. The data is not in this repository. `run` takes about one minute and writes `results/results.json` and `results/panels.csv`.

## Where the run differs from the proposal

Every setting is a constant at the top of `pdac_panel/analysis.py`. I fixed them before I scored the held-out set, and I scored it once.

| Part | The proposal | The run |
|---|---|---|
| Data | A study in the European Genome-phenome Archive | The two supplementary tables of the paper of that study |
| Candidates | The biomarkers that the dataset holds | KRAS mutation found in plasma (yes or no), plasma DNA concentration, CA19-9, CEA, HGF, OPN |
| Detection limits | Not covered | The lowest reported value of each protein differs between the patient table and the control table. All values are raised to one shared limit, so no model can learn the table |
| Generalized linear model | Named, with no type | Logistic regression |
| Top 10% of panels | No ranking rule | Ranked by 5-fold cross-validation inside the training set |
| Small AUC difference | No number | At most 0.05 between training and held-out, for each of the four models |

## Limits

- The controls are healthy people. People with pancreatitis, cysts or diabetes are the hard cases in screening, and this data has none. The proposal named this limit.
- One split and 135 held-out people. The intervals are wide.
- The threshold aimed at 99% of healthy people called negative. On the held-out set it calls 3 of 61 positive.
- The panel does not beat the test of the paper. It detects more patients and calls more healthy people positive.
- The KRAS call is the call of the paper. It is not recomputed from sequencing reads.
- No second cohort.

## Source of the data

Cohen JD, Javed AA, Thoburn C, et al. Combined circulating tumor DNA and protein biomarker-based liquid biopsy for the earlier detection of pancreatic cancers. PNAS 114(38), 2017. https://doi.org/10.1073/pnas.1704961114
