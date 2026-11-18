"""Downloads the two supplementary datasets of Cohen et al. 2017 and reads them into one table."""
import hashlib
import pathlib
import urllib.request

import numpy as np
import openpyxl
import pandas as pd

RAW = pathlib.Path("data/raw")
URL = ("https://europepmc.org/api/fulltextRepo?pprId=PMC5617273&type=FILE&fileName={name}"
       "&mimeType=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
FILES = {
    "pnas.1704961114.sd01.xlsx": "36753d87c72d1c4003958fa0b3877ad5980bbd63d9a5dfeaf3105ea5b49fa8f9",
    "pnas.1704961114.sd02.xlsx": "86a8af0b8b7cb962918de3456db34ff6a2d4714d914467df115a5fcca5c6cd55",
}
PROTEINS = ["CA19-9", "CEA", "HGF", "OPN"]
FEATURES = ["KRAS", "cfDNA"] + PROTEINS
FAMILY = {"KRAS": "DNA", "cfDNA": "DNA", "CA19-9": "protein", "CEA": "protein", "HGF": "protein", "OPN": "protein"}


def fetch(raw=RAW):
    """Download both files and stop if a hash differs from the one this analysis ran on."""
    raw.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        path = raw / name
        if not path.exists():
            request = urllib.request.Request(URL.format(name=name), headers={"User-Agent": "Mozilla/5.0"})
            path.write_bytes(urllib.request.urlopen(request, timeout=60).read())
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found != sha:
            raise SystemExit(f"{path}: sha256 {found} differs from the expected {sha}")
        print("ok", path)


def _sheet(path, label):
    """Read one sheet. A filled cell is a value that the authors set to a limit of detection."""
    sheet = openpyxl.load_workbook(path).worksheets[0]
    header = {str(cell.value).strip().split(" plasma concentration")[0]: i for i, cell in enumerate(sheet[3]) if cell.value}
    rows = []
    for row in sheet.iter_rows(min_row=4):
        if row[0].value is None or row[1].value is None:
            continue
        record = {
            "id": row[0].value,
            "pdac": label,
            "KRAS": int(row[header["Mutation identified in plasma"]].value != "None detected"),
            "cfDNA": float(row[header["Plasma DNA concentration (ng/mL)"]].value),
            "published_call": row[[i for name, i in header.items() if name.startswith("Combination assay")][0]].value,
        }
        for protein in PROTEINS:
            cell = row[header[protein]]
            record[protein] = float(cell.value)
            record[protein + "_at_limit"] = cell.fill.fill_type is not None
        rows.append(record)
    return rows


def load(raw=RAW):
    """One row per person: 221 with resectable pancreatic cancer, 182 healthy controls."""
    names = list(FILES)
    table = pd.DataFrame(_sheet(raw / names[0], 1) + _sheet(raw / names[1], 0))
    return table


def clip_limits(table):
    """Raise each protein to one shared lower limit.

    The files hold a different lower limit for each assay plate, and patients and controls ran on
    different plates. Without this step a model can tell the two groups apart by the plate.
    """
    table = table.copy()
    floors = {}
    for protein in PROTEINS:
        low = table.loc[table[protein + "_at_limit"] & (table[protein] < table[protein].median()), protein]
        floors[protein] = float(low.max()) if len(low) else float(table[protein].min())
        table[protein] = table[protein].clip(lower=floors[protein])
    return table, floors


def features(table):
    """The model inputs: the mutation flag as 0 or 1, and log10 of each concentration."""
    x = table[FEATURES].copy()
    for name in FEATURES[1:]:
        x[name] = np.log10(x[name])
    return x
