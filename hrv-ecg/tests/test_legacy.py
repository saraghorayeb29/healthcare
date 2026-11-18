"""Check the coursework archive without loading clinical data or running historical notebooks."""
import ast
import importlib.util
import json
from pathlib import Path
import re

from IPython.core.inputtransformer2 import TransformerManager
import nbformat
import numpy as np
import pytest

LEGACY = Path(__file__).resolve().parents[1] / "legacy"
NOTEBOOKS = sorted(LEGACY.glob("*.ipynb"))
MODULES = sorted((LEGACY / "misc").glob("*.py"))


def load_module(name):
    """Load an original module by filename without changing its source."""
    spec = importlib.util.spec_from_file_location(name, LEGACY / "misc" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_contains_all_original_source_files():
    assert len(NOTEBOOKS) == 11
    assert len(MODULES) == 10


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_has_no_saved_outputs_or_metadata(path):
    notebook = json.loads(path.read_text())
    nbformat.validate(nbformat.from_dict(notebook))
    assert notebook["metadata"] == {}
    for cell in notebook["cells"]:
        assert cell["metadata"] == {}
        assert "attachments" not in cell
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_code_has_valid_syntax(path):
    notebook = json.loads(path.read_text())
    transformer = TransformerManager()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            compile(transformer.transform_cell(text), f"{path.name}:cell-{index}", "exec")


@pytest.mark.parametrize("path", NOTEBOOKS + MODULES, ids=lambda path: path.name)
def test_source_contains_no_private_paths_or_record_identifiers(path):
    text = path.read_text()
    pattern = r"/Users/|/home/|/Volumes/|[A-Z]:\\|S\d{4}|Jaume|Matthieu|shortcut-targets-by-id"
    assert not re.search(pattern, text)


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_original_python_module_has_valid_syntax(path):
    ast.parse(path.read_text(), filename=path.name)


def test_power_interference_adds_the_documented_frequency():
    noise = load_module("synthetic_noise")
    clean = np.zeros(1000)
    output = noise.powerinterference(clean, fs=1000, amplitude=0.3)
    spectrum = np.abs(np.fft.rfft(output))
    assert int(np.argmax(spectrum)) == 60
    assert np.max(np.abs(output)) == pytest.approx(0.3, abs=0.001)
    np.testing.assert_array_equal(clean, np.zeros(1000))


def test_white_noise_can_be_reproduced_with_a_fixed_seed():
    noise = load_module("synthetic_noise")
    clean = np.zeros(128)
    np.random.seed(2022)
    first = noise.gen_white_noise(clean)
    np.random.seed(2022)
    second = noise.gen_white_noise(clean)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.std(first) > 0


def test_zero_noise_preserves_the_input():
    noise = load_module("synthetic_noise")
    clean = np.linspace(-1, 1, 100)
    np.testing.assert_array_equal(noise.emgnoise(clean, mean=0, std=0), clean)
    np.testing.assert_array_equal(noise.gen_bw_noise(clean, fs=0.05, amplitude=0), clean)


def test_smoothing_matches_a_moving_average():
    filtering = load_module("noise_filtering")
    signal = np.arange(32, dtype=float)
    expected = np.convolve(signal, np.ones(5) / 5, mode="same")
    np.testing.assert_allclose(filtering.smoothing_window(signal, w=5), expected)


def test_lowpass_attenuates_high_frequency_noise():
    filtering = load_module("noise_filtering")
    time = np.arange(4000) / 1000
    low = np.sin(2 * np.pi * 5 * time)
    high = np.sin(2 * np.pi * 100 * time)
    output = filtering.filter_signal(low + high, cutoff=20, sample_rate=1000, order=4)
    center = slice(500, -500)
    low_amplitude = 2 * np.mean(output[center] * low[center])
    high_amplitude = 2 * np.mean(output[center] * high[center])
    assert low_amplitude > 0.9
    assert abs(high_amplitude) < 0.01
