import pytest
import numpy as np
import pandas as pd
from ps4g_io.ps4g import load_ps4g_file, extract_metadata, convert_ps4g

@pytest.fixture
def sample_ps4g_file():
    return "data/sample_test.ps4g"

def test_load_ps4g_file(sample_ps4g_file):
    df = load_ps4g_file(sample_ps4g_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4
    assert 'gameteSet' in df.columns
    assert isinstance(df['gameteSet'].iloc[0], list)

def test_extract_metadata(sample_ps4g_file):
    metadata, gamete_data = extract_metadata(sample_ps4g_file)
    assert isinstance(metadata, dict)
    assert "sample_name" in metadata
    assert isinstance(gamete_data, list)
    assert len(gamete_data) == 3  # 3 gametes defined

# ────────────────────────────────────────────────
#                 Weight Modes
# ────────────────────────────────────────────────

def test_convert_unweighted(sample_ps4g_file):
    result = convert_ps4g(sample_ps4g_file, weight="unweighted", collapse=False)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4  # 4 rows
    assert result.shape[1] > 0   # should have columns

def test_convert_read_weight(sample_ps4g_file):
    result = convert_ps4g(sample_ps4g_file, weight="read", collapse=False)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4
    assert result.shape[1] > 0

def test_convert_global_weight(sample_ps4g_file):
    result = convert_ps4g(sample_ps4g_file, weight="global", collapse=False)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4
    assert result.shape[1] > 0

# ────────────────────────────────────────────────
#                   Collapse Mode
# ────────────────────────────────────────────────

def test_convert_collapsed(sample_ps4g_file):
    result = convert_ps4g(sample_ps4g_file, weight="unweighted", collapse=True)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2  # collapsed by position (2 unique pos values)
    assert result.shape[1] > 0



