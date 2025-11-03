import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from python.bed_io.bed import output_bed_file, output_collapse_bed, output_predictions


@pytest.fixture
def sample_bed_data():
    """Create sample data for BED file output testing."""
    chroms = np.array(['chr1', 'chr1', 'chr1', 'chr2', 'chr2'])
    positions = np.array([1000, 2000, 3000, 1000, 2000])
    final_predictions = np.array([
        [0, 1],  # B73, CML247
        [0, 1],  # B73, CML247 (same as above)
        [1, 2],  # CML247, W22
        [0, 2],  # B73, W22
        [0, 2],  # B73, W22 (same as above)
    ])
    index_array = ['B73:0', 'CML247:0', 'W22:0']
    return chroms, positions, final_predictions, index_array


@pytest.fixture
def temp_output_file():
    """Create a temporary file for output testing."""
    fd, path = tempfile.mkstemp(suffix='.bed')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_output_bed_file_uncollapsed(sample_bed_data, temp_output_file):
    """Test BED file output without collapsing regions."""
    chroms, positions, final_predictions, index_array = sample_bed_data

    output_bed_file(
        temp_output_file,
        chroms,
        positions,
        final_predictions,
        index_array,
        collapse_bed_regions=False
    )

    # Read the output file
    df = pd.read_csv(temp_output_file, sep='\t')

    # Verify structure
    assert list(df.columns) == ['chrom', 'pos', 'parent1', 'parent2']
    assert len(df) == 5

    # Verify chromosome names are preserved (not indices)
    assert df['chrom'].iloc[0] == 'chr1'
    assert df['chrom'].iloc[3] == 'chr2'

    # Verify positions
    assert df['pos'].iloc[0] == 1000
    assert df['pos'].iloc[1] == 2000

    # Verify parent assignments
    assert df['parent1'].iloc[0] == 'B73:0'
    assert df['parent2'].iloc[0] == 'CML247:0'
    assert df['parent1'].iloc[2] == 'CML247:0'
    assert df['parent2'].iloc[2] == 'W22:0'


def test_output_bed_file_collapsed(sample_bed_data, temp_output_file):
    """Test BED file output with collapsed regions."""
    chroms, positions, final_predictions, index_array = sample_bed_data

    output_bed_file(
        temp_output_file,
        chroms,
        positions,
        final_predictions,
        index_array,
        collapse_bed_regions=True
    )

    # Read the output file
    df = pd.read_csv(temp_output_file, sep='\t')

    # Verify structure for collapsed format
    assert list(df.columns) == ['chrom', 'start', 'end', 'parent1', 'parent2']

    # Should have 3 collapsed regions:
    # 1. chr1:1000-2000 (B73, CML247)
    # 2. chr1:3000-3000 (CML247, W22)
    # 3. chr2:1000-2000 (B73, W22)
    assert len(df) == 3

    # Verify first collapsed region
    assert df['chrom'].iloc[0] == 'chr1'
    assert df['start'].iloc[0] == 1000
    assert df['end'].iloc[0] == 2000
    assert df['parent1'].iloc[0] == 'B73:0'
    assert df['parent2'].iloc[0] == 'CML247:0'

    # Verify second collapsed region
    assert df['chrom'].iloc[1] == 'chr1'
    assert df['start'].iloc[1] == 3000
    assert df['end'].iloc[1] == 3000
    assert df['parent1'].iloc[1] == 'CML247:0'
    assert df['parent2'].iloc[1] == 'W22:0'

    # Verify third collapsed region
    assert df['chrom'].iloc[2] == 'chr2'
    assert df['start'].iloc[2] == 1000
    assert df['end'].iloc[2] == 2000
    assert df['parent1'].iloc[2] == 'B73:0'
    assert df['parent2'].iloc[2] == 'W22:0'


def test_output_collapse_bed_same_chromosome():
    """Test collapsing contiguous regions on the same chromosome."""
    bed_df = pd.DataFrame({
        'chrom': ['chr1', 'chr1', 'chr1', 'chr1'],
        'pos': [1000, 2000, 3000, 4000],
        'parent1': ['B73:0', 'B73:0', 'CML247:0', 'CML247:0'],
        'parent2': ['CML247:0', 'CML247:0', 'W22:0', 'W22:0']
    })

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bed') as f:
        temp_file = f.name

    try:
        output_collapse_bed(bed_df, temp_file)
        result_df = pd.read_csv(temp_file, sep='\t')

        # Should collapse into 2 regions
        assert len(result_df) == 2

        # First region: positions 1000-2000
        assert result_df['chrom'].iloc[0] == 'chr1'
        assert result_df['start'].iloc[0] == 1000
        assert result_df['end'].iloc[0] == 2000
        assert result_df['parent1'].iloc[0] == 'B73:0'
        assert result_df['parent2'].iloc[0] == 'CML247:0'

        # Second region: positions 3000-4000
        assert result_df['chrom'].iloc[1] == 'chr1'
        assert result_df['start'].iloc[1] == 3000
        assert result_df['end'].iloc[1] == 4000
        assert result_df['parent1'].iloc[1] == 'CML247:0'
        assert result_df['parent2'].iloc[1] == 'W22:0'
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_output_collapse_bed_different_chromosomes():
    """Test that regions on different chromosomes are not collapsed together."""
    bed_df = pd.DataFrame({
        'chrom': ['chr1', 'chr1', 'chr2', 'chr2'],
        'pos': [1000, 2000, 1000, 2000],
        'parent1': ['B73:0', 'B73:0', 'B73:0', 'B73:0'],
        'parent2': ['CML247:0', 'CML247:0', 'CML247:0', 'CML247:0']
    })

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bed') as f:
        temp_file = f.name

    try:
        output_collapse_bed(bed_df, temp_file)
        result_df = pd.read_csv(temp_file, sep='\t')

        # Should have 2 regions (one per chromosome)
        assert len(result_df) == 2

        # Verify chromosomes are preserved
        assert result_df['chrom'].iloc[0] == 'chr1'
        assert result_df['chrom'].iloc[1] == 'chr2'

        # Same parents for both regions
        assert result_df['parent1'].iloc[0] == 'B73:0'
        assert result_df['parent1'].iloc[1] == 'B73:0'
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_output_predictions_integration(temp_output_file):
    """Test the full output_predictions workflow with a real PS4G file."""
    ps4g_file = "data/sample_test.ps4g"

    # Create sample predictions for 4 rows from the sample file
    # Note: sample file has 4 rows with positions: 1000, 1000, 2000, 2000
    final_predictions = np.array([
        [0, 1],  # B73, CML247
        [0, 1],  # B73, CML247
        [1, 2],  # CML247, W22
        [1, 2],  # CML247, W22
    ])

    output_predictions(
        ps4g_file,
        temp_output_file,
        final_predictions,
        collapse_bed_regions=False
    )

    # Read the output
    df = pd.read_csv(temp_output_file, sep='\t')

    # Verify basic structure
    assert list(df.columns) == ['chrom', 'pos', 'parent1', 'parent2']
    assert len(df) == 4

    # Verify chromosome names are decoded (not indices)
    assert df['chrom'].iloc[0] == 'chr1'
    assert df['chrom'].iloc[2] == 'chr1'

    # Verify positions are correct (not encoded) - rows 0,1 at 1000; rows 2,3 at 2000
    assert df['pos'].iloc[0] == 1000
    assert df['pos'].iloc[1] == 1000
    assert df['pos'].iloc[2] == 2000
    assert df['pos'].iloc[3] == 2000

    # Verify parent names (note: build_index_lookup returns names without ':0' suffix)
    assert df['parent1'].iloc[0] == 'B73'
    assert df['parent2'].iloc[0] == 'CML247'


def test_output_predictions_collapsed_integration(temp_output_file):
    """Test output_predictions with collapse_bed_regions=True."""
    ps4g_file = "data/sample_test.ps4g"

    # Create predictions where some regions should collapse
    # All 4 rows have same parent assignment, should collapse into 1 region
    final_predictions = np.array([
        [0, 1],  # B73, CML247 at position 1000
        [0, 1],  # B73, CML247 at position 1000
        [0, 1],  # B73, CML247 at position 2000
        [0, 1],  # B73, CML247 at position 2000
    ])

    output_predictions(
        ps4g_file,
        temp_output_file,
        final_predictions,
        collapse_bed_regions=True
    )

    # Read the output
    df = pd.read_csv(temp_output_file, sep='\t')

    # Should be collapsed into 1 region spanning both positions
    assert len(df) == 1
    assert list(df.columns) == ['chrom', 'start', 'end', 'parent1', 'parent2']

    # Verify the collapsed region spans from first to last position
    assert df['chrom'].iloc[0] == 'chr1'
    assert df['start'].iloc[0] == 1000
    assert df['end'].iloc[0] == 2000


def test_chromosome_name_preservation():
    """Test that chromosome names are preserved correctly (not converted to indices)."""
    chroms = np.array(['chr1', 'chr10', 'chr2', 'chrX'])
    positions = np.array([100, 200, 300, 400])
    predictions = np.array([[0, 1], [0, 1], [1, 2], [0, 2]])
    index_array = ['A', 'B', 'C']

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.bed') as f:
        temp_file = f.name

    try:
        output_bed_file(temp_file, chroms, positions, predictions, index_array, collapse_bed_regions=False)
        df = pd.read_csv(temp_file, sep='\t')

        # Verify chromosome names are exactly as provided
        assert df['chrom'].tolist() == ['chr1', 'chr10', 'chr2', 'chrX']
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
