"""Checks on the data reading and on the rules that keep the validation set out of every choice."""
import pathlib

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from pdac_panel import analysis, data

pytestmark = pytest.mark.skipif(not all((data.RAW / name).exists() for name in data.FILES), reason="run `pdac-panel fetch` first")


@pytest.fixture(scope="module")
def table():
    return data.load()


def test_cohort_matches_the_paper(table):
    assert (table.pdac == 1).sum() == 221 and (table.pdac == 0).sum() == 182
    assert table.id.is_unique


def test_kras_and_published_calls_match_the_paper(table):
    assert table.loc[table.pdac == 1, "KRAS"].sum() == 66
    assert table.loc[table.pdac == 0, "KRAS"].sum() == 1
    assert (table.loc[table.pdac == 1, "published_call"] == "Positive").sum() == 141
    assert (table.loc[table.pdac == 0, "published_call"] == "Positive").sum() == 1


def test_one_lower_limit_for_both_groups(table):
    clipped, floors = data.clip_limits(table)
    for protein in data.PROTEINS:
        assert clipped.loc[clipped.pdac == 1, protein].min() >= floors[protein]
        assert clipped.loc[clipped.pdac == 0, protein].min() == floors[protein]


def test_features_are_finite(table):
    x = data.features(data.clip_limits(table)[0])
    assert list(x.columns) == data.FEATURES and np.isfinite(x.to_numpy()).all()


def test_split_is_stratified_and_disjoint(table):
    y = table.pdac.to_numpy()
    train, validation = train_test_split(np.arange(len(y)), test_size=analysis.VALIDATION_SHARE, stratify=y, random_state=analysis.SEED)
    assert not set(train) & set(validation) and len(train) + len(validation) == 403
    assert abs(y[train].mean() - y[validation].mean()) < 0.01


def test_threshold_uses_training_controls_only():
    yt, yv = np.array([0] * 100 + [1] * 100), np.array([0, 0, 1, 1])
    cv = np.concatenate([np.linspace(0, 1, 100), np.full(100, 5.0)])
    out = analysis.at_specificity(cv, yt, np.array([0.5, 2.0, 0.5, 2.0]), yv)
    assert out == {"true_positive": 1, "cases": 2, "false_positive": 1, "controls": 2}
