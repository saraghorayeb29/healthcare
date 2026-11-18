"""The plan of the 2022 proposal, in its order: select biomarkers, train every panel, test the panels."""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import FAMILY, FEATURES, clip_limits, features, load

# Every setting below was fixed before the first run on the validation set.
SEED = 20221201
VALIDATION_SHARE = 1 / 3
ALPHA = 0.01
MIN_AUC = 0.9
MAX_GAP = 0.05
TOP_SHARE = 0.10
FOLDS = 5
SPECIFICITY = 0.99
BOOTSTRAPS = 2000
REPORT_MODEL = "GLM"

MODELS = {
    "Ridge": lambda: make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0)),
    "GLM": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000)),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
    "RF": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
}


def score(model, x):
    return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)


def select(x, y):
    """Milestone 1.3: a one-way ANOVA per biomarker on the training set, kept when p < 0.01."""
    f, p = f_classif(x, y)
    return [{"biomarker": name, "family": FAMILY[name], "F": float(a), "p": float(b), "selected": bool(b < ALPHA)}
            for name, a, b in zip(x.columns, f, p)]


def fit_panel(panel, xt, yt, xv, yv):
    """Milestone 2.2 and 3.2: four models for one panel, with scores kept for the later steps."""
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    out = {}
    for name, make in MODELS.items():
        method = "predict_proba" if name in ("GLM", "RF") else "decision_function"
        cv = cross_val_predict(make(), xt[panel], yt, cv=folds, method=method)
        cv = cv[:, 1] if cv.ndim == 2 else cv
        model = make().fit(xt[panel], yt)
        out[name] = {"train": score(model, xt[panel]), "cv": cv, "validation": score(model, xv[panel])}
    return out


def at_specificity(cv_scores, yt, val_scores, yv):
    """Set the threshold on training controls, out of fold, then count calls on the validation set."""
    threshold = np.quantile(cv_scores[yt == 0], SPECIFICITY)
    called = val_scores > threshold
    return {"true_positive": int(called[yv == 1].sum()), "cases": int((yv == 1).sum()),
            "false_positive": int(called[yv == 0].sum()), "controls": int((yv == 0).sum())}


def bootstrap(yv, a, b=None):
    """Percentile interval for an AUC, or for the difference of two AUCs on the same people."""
    rng = np.random.default_rng(SEED)
    cases, controls = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    values = []
    for _ in range(BOOTSTRAPS):
        idx = np.concatenate([rng.choice(cases, len(cases)), rng.choice(controls, len(controls))])
        auc = roc_auc_score(yv[idx], a[idx])
        values.append(auc if b is None else auc - roc_auc_score(yv[idx], b[idx]))
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def run():
    table, floors = clip_limits(load())
    x, y = features(table), table["pdac"].to_numpy()
    xt, xv, yt, yv = train_test_split(x, y, test_size=VALIDATION_SHARE, stratify=y, random_state=SEED)

    anova = select(xt, yt)
    kept = [row["biomarker"] for row in anova if row["selected"]]
    panels = [list(c) for k in range(1, len(kept) + 1) for c in itertools.combinations(kept, k)]

    rows, scores = [], {}
    for panel in panels:
        key = " + ".join(panel)
        scores[key] = fit_panel(panel, xt, yt, xv, yv)
        row = {"panel": key, "size": len(panel), "families": " + ".join(sorted({FAMILY[b] for b in panel}))}
        for name, s in scores[key].items():
            row[f"{name}_train"] = roc_auc_score(yt, s["train"])
            row[f"{name}_cv"] = roc_auc_score(yt, s["cv"])
            row[f"{name}_validation"] = roc_auc_score(yv, s["validation"])
        row["mean_train"] = np.mean([row[f"{m}_train"] for m in MODELS])
        row["mean_cv"] = np.mean([row[f"{m}_cv"] for m in MODELS])
        row["mean_validation"] = np.mean([row[f"{m}_validation"] for m in MODELS])
        row["min_validation"] = min(row[f"{m}_validation"] for m in MODELS)
        row["max_gap"] = max(abs(row[f"{m}_train"] - row[f"{m}_validation"]) for m in MODELS)
        rows.append(row)
    table_out = pd.DataFrame(rows)

    # Milestone 3.1. The rank uses cross-validation on the training set, so the validation set picks nothing.
    table_out["rank_cv"] = table_out["mean_cv"].rank(ascending=False, method="min").astype(int)
    top_n = int(np.ceil(TOP_SHARE * len(table_out)))
    table_out["c1_top"] = table_out["rank_cv"] <= top_n
    table_out["c2_auc"] = table_out["min_validation"] > MIN_AUC
    table_out["c3_gap"] = table_out["max_gap"] <= MAX_GAP
    table_out["passes"] = table_out[["c1_top", "c2_auc", "c3_gap"]].all(axis=1)
    table_out = table_out.sort_values("rank_cv").reset_index(drop=True)
    chosen = table_out.iloc[0]["panel"]

    def best(mask):
        return table_out[mask].iloc[0]["panel"] if mask.any() else None

    compare = {
        "selected panel": chosen,
        "CA19-9 alone": "CA19-9" if "CA19-9" in kept else None,
        "best protein-only panel": best(table_out["families"] == "protein"),
        "best DNA-only panel": best(table_out["families"] == "DNA"),
        "best two-family panel": best(table_out["families"] == "DNA + protein"),
    }
    comparison = []
    for label, key in compare.items():
        if key is None:
            continue
        s = scores[key][REPORT_MODEL]
        entry = {"label": label, "panel": key, "auc": float(roc_auc_score(yv, s["validation"])),
                 "ci": bootstrap(yv, s["validation"]), **at_specificity(s["cv"], yt, s["validation"], yv)}
        comparison.append(entry)

    def difference(a, b):
        sa, sb = scores[a][REPORT_MODEL]["validation"], scores[b][REPORT_MODEL]["validation"]
        return {"a": a, "b": b, "difference": float(roc_auc_score(yv, sa) - roc_auc_score(yv, sb)), "ci": bootstrap(yv, sa, sb)}

    differences = {}
    if compare["best two-family panel"] and compare["best protein-only panel"]:
        differences["two families minus proteins only"] = difference(compare["best two-family panel"], compare["best protein-only panel"])
    if compare["best two-family panel"] and compare["CA19-9 alone"]:
        differences["two families minus CA19-9 alone"] = difference(compare["best two-family panel"], "CA19-9")

    def curve(key, model):
        fpr, tpr, _ = roc_curve(yv, scores[key][model]["validation"])
        return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]

    published = table.loc[xv.index, "published_call"].eq("Positive").to_numpy()
    results = {
        "settings": {"seed": SEED, "validation_share": "1/3", "alpha": ALPHA, "min_auc": MIN_AUC, "max_gap": MAX_GAP,
                     "top_share": TOP_SHARE, "folds": FOLDS, "specificity": SPECIFICITY, "bootstraps": BOOTSTRAPS,
                     "report_model": REPORT_MODEL},
        "cohort": {"cases": int(y.sum()), "controls": int((y == 0).sum()),
                   "train": {"cases": int(yt.sum()), "controls": int((yt == 0).sum())},
                   "validation": {"cases": int(yv.sum()), "controls": int((yv == 0).sum())},
                   "kras_detected": {"cases": int(table.loc[y == 1, "KRAS"].sum()), "controls": int(table.loc[y == 0, "KRAS"].sum())}},
        "limit_floors": floors,
        "anova": anova,
        "panel_count": len(panels), "model_count": len(panels) * len(MODELS), "top_n": top_n,
        "passing": {"c1_top": int(table_out["c1_top"].sum()), "c2_auc": int(table_out["c2_auc"].sum()),
                    "c3_gap": int(table_out["c3_gap"].sum()), "all": int(table_out["passes"].sum())},
        "selected": {"panel": chosen, "passes": bool(table_out.iloc[0]["passes"]),
                     **{k: float(v) for k, v in table_out.iloc[0].items() if k.endswith(("_train", "_cv", "_validation", "_gap"))}},
        "comparison": comparison,
        "differences": differences,
        "published_assay_on_validation": {"true_positive": int(published[yv == 1].sum()), "cases": int((yv == 1).sum()),
                                          "false_positive": int(published[yv == 0].sum()), "controls": int((yv == 0).sum())},
        "roc": {"selected": {m: curve(chosen, m) for m in MODELS},
                "CA19-9": curve("CA19-9", REPORT_MODEL) if "CA19-9" in kept else None},
    }
    return results, table_out
