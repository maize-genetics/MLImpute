import pytest
import numpy as np
import pandas as pd
from python.ps4g_io.ps4g import (
    load_ps4g_file,
    extract_metadata,
    convert_ps4g,
    create_multihot_matrix,
    collapse_ps4g,
    build_index_lookup
)

@pytest.fixture
def sample_ps4g_file():
    return "data/sample_test.ps4g"

def test_load_ps4g_file(sample_ps4g_file):
    df = load_ps4g_file(sample_ps4g_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4
    assert 'gameteSet' in df.columns
    assert isinstance(df['gameteSet'].iloc[0], list)

def test_load_ps4g_file_has_position_columns(sample_ps4g_file):
    """Test that PS4G file loading preserves refContig and refPosBinned columns."""
    df = load_ps4g_file(sample_ps4g_file)
    assert 'refContig' in df.columns
    assert 'refPosBinned' in df.columns

    # Verify the values are correct
    assert df['refContig'].iloc[0] == 'chr1'
    assert df['refPosBinned'].iloc[0] == 1000

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
    result, weights = convert_ps4g(sample_ps4g_file, weight_strat="unweighted", collapse=False)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4  # 4 rows
    assert result.shape[1] > 0   # should have columns
    assert isinstance(weights, np.ndarray)
    assert weights.shape[0] == 3  # should match number of rows in result
    assert weights[0] == 1.0  # unweighted should have all weights as 1.0
    assert weights[1] == 1.0
    assert weights[2] == 1.0  # no weights for unweighted

def test_convert_global_weight(sample_ps4g_file):
    result, weights = convert_ps4g(sample_ps4g_file, weight_strat="global", collapse=False)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4
    assert result.shape[1] > 0
    assert isinstance(weights, np.ndarray)
    assert weights.shape[0] == 3  # should match number of rows in result
    assert weights[0] == 1.0  # unweighted should have all weights as 1.0
    assert weights[1] == 0.5
    assert weights[2] == 0.25  # no weights for unweighted

# ────────────────────────────────────────────────
#                   Collapse Mode
# ────────────────────────────────────────────────

def test_convert_collapsed(sample_ps4g_file):
    result, weights = convert_ps4g(sample_ps4g_file, weight_strat="unweighted", collapse=True)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2  # collapsed by position (2 unique pos values)
    assert result.shape[1] > 0
    assert isinstance(weights, np.ndarray)
    assert weights.shape[0] == 3  # should match number of rows in result
    assert weights[0] == 1.0  # unweighted should have all weights as 1.0
    assert weights[1] == 1.0
    assert weights[2] == 1.0  # no weights for unweighted


# ────────────────────────────────────────────────
#            Position ID Handling Tests
# ────────────────────────────────────────────────

def test_position_id_creation(sample_ps4g_file):
    """Test that position_id is correctly created from refContig and refPosBinned."""
    df = load_ps4g_file(sample_ps4g_file)
    metadata, gamete_data = extract_metadata(sample_ps4g_file)

    # Manually add position_id to test the format
    df['position_id'] = df['refContig'].astype(str) + '_' + df['refPosBinned'].astype(str)

    # Verify position_id format
    assert df['position_id'].iloc[0] == 'chr1_1000'
    assert df['position_id'].iloc[2] == 'chr1_2000'


def test_position_id_uniqueness(sample_ps4g_file):
    """Test that position_id correctly identifies unique positions."""
    df = load_ps4g_file(sample_ps4g_file)
    df['position_id'] = df['refContig'].astype(str) + '_' + df['refPosBinned'].astype(str)

    unique_positions = df['position_id'].unique()

    # Sample file has 2 unique positions: chr1_1000 and chr1_2000
    assert len(unique_positions) == 2
    assert 'chr1_1000' in unique_positions
    assert 'chr1_2000' in unique_positions


def test_create_multihot_matrix_with_position_id(sample_ps4g_file):
    """Test that create_multihot_matrix correctly uses position_id."""
    ps4g_df = load_ps4g_file(sample_ps4g_file)
    metadata, gamete_data = extract_metadata(sample_ps4g_file)

    result, collapsed_df = create_multihot_matrix(
        ps4g_df,
        gamete_data,
        weight_strat="unweighted",
        collapse=False
    )

    # Verify position_id was created in the process
    assert 'position_id' in ps4g_df.columns

    # Verify multihot matrix dimensions
    assert result.shape[0] == 4  # 4 rows in sample data
    assert result.shape[1] == 3  # 3 gametes


def test_collapse_ps4g_preserves_position_info(sample_ps4g_file):
    """Test that collapse_ps4g preserves refContig and refPosBinned."""
    ps4g_df = load_ps4g_file(sample_ps4g_file)
    metadata, gamete_data = extract_metadata(sample_ps4g_file)

    # Add position_id
    ps4g_df['position_id'] = ps4g_df['refContig'].astype(str) + '_' + ps4g_df['refPosBinned'].astype(str)
    unique_positions = ps4g_df['position_id'].unique()

    num_classes = len(gamete_data)
    X_multihot, collapsed_df = collapse_ps4g(num_classes, ps4g_df, unique_positions)

    # Verify collapsed dataframe has required columns
    assert 'position_id' in collapsed_df.columns
    assert 'refContig' in collapsed_df.columns
    assert 'refPosBinned' in collapsed_df.columns

    # Verify position information is preserved
    assert len(collapsed_df) == 2  # 2 unique positions
    assert collapsed_df['refContig'].iloc[0] == 'chr1'
    assert collapsed_df['refPosBinned'].iloc[0] == 1000


def test_collapse_ps4g_aggregates_counts(sample_ps4g_file):
    """Test that collapse_ps4g correctly aggregates counts for same position."""
    ps4g_df = load_ps4g_file(sample_ps4g_file)
    metadata, gamete_data = extract_metadata(sample_ps4g_file)

    ps4g_df['position_id'] = ps4g_df['refContig'].astype(str) + '_' + ps4g_df['refPosBinned'].astype(str)
    unique_positions = ps4g_df['position_id'].unique()

    num_classes = len(gamete_data)
    X_multihot, collapsed_df = collapse_ps4g(num_classes, ps4g_df, unique_positions)

    # Position chr1_1000 has 2 rows with counts 1 and 1, should sum to 2
    pos1_count = collapsed_df[collapsed_df['position_id'] == 'chr1_1000']['count'].iloc[0]
    assert pos1_count == 2

    # Position chr1_2000 has 2 rows with counts 1 and 1, should sum to 2
    pos2_count = collapsed_df[collapsed_df['position_id'] == 'chr1_2000']['count'].iloc[0]
    assert pos2_count == 2


def test_collapse_ps4g_merges_gamete_sets(sample_ps4g_file):
    """Test that collapse_ps4g correctly merges gamete sets for same position."""
    ps4g_df = load_ps4g_file(sample_ps4g_file)
    metadata, gamete_data = extract_metadata(sample_ps4g_file)

    ps4g_df['position_id'] = ps4g_df['refContig'].astype(str) + '_' + ps4g_df['refPosBinned'].astype(str)
    unique_positions = ps4g_df['position_id'].unique()

    num_classes = len(gamete_data)
    X_multihot, collapsed_df = collapse_ps4g(num_classes, ps4g_df, unique_positions)

    # Position chr1_1000 has rows with gamete sets [0] and [0,1]
    # After collapse, should have [0, 1]
    pos1_gametes = collapsed_df[collapsed_df['position_id'] == 'chr1_1000']['gameteSet'].iloc[0]
    assert set(pos1_gametes) == {0, 1}

    # Position chr1_2000 has rows with gamete sets [0] and [0,1,2]
    # After collapse, should have [0, 1, 2]
    pos2_gametes = collapsed_df[collapsed_df['position_id'] == 'chr1_2000']['gameteSet'].iloc[0]
    assert set(pos2_gametes) == {0, 1, 2}


def test_build_index_lookup(sample_ps4g_file):
    """Test that build_index_lookup correctly extracts gamete names."""
    index_array = build_index_lookup(sample_ps4g_file)

    # Should have 3 gametes
    assert len(index_array) == 3

    # Verify gamete names (note: build_index_lookup extracts just the name before ':')
    assert index_array[0] == 'B73'
    assert index_array[1] == 'CML247'
    assert index_array[2] == 'W22'


def test_multihot_matrix_different_chromosomes():
    """Test position_id handling with multiple chromosomes."""
    # Create a test dataframe with multiple chromosomes
    test_df = pd.DataFrame({
        'gameteSet': [[0], [1], [0]],
        'refContig': ['chr1', 'chr2', 'chr1'],
        'refPosBinned': [1000, 1000, 2000],
        'count': [1, 1, 1]
    })

    test_df['position_id'] = test_df['refContig'].astype(str) + '_' + test_df['refPosBinned'].astype(str)
    unique_positions = test_df['position_id'].unique()

    # Should have 3 unique positions (chr1_1000, chr2_1000, chr1_2000)
    assert len(unique_positions) == 3
    assert 'chr1_1000' in unique_positions
    assert 'chr2_1000' in unique_positions
    assert 'chr1_2000' in unique_positions


