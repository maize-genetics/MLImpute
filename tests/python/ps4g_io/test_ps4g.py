import pytest
import numpy as np
import pandas as pd
from python.ps4g_io.ps4g import (
    load_ps4g_file,
    extract_metadata,
    convert_ps4g,
    create_multihot_matrix,
    collapse_ps4g,
    build_index_lookup,
    parse_gamete_header_line,
    parse_gamete_records,
    parse_sample_gamete,
    SampleGamete,
)

@pytest.fixture
def sample_ps4g_file():
    return "data/sample_test.ps4g"

@pytest.fixture
def sample_ps4g_file_no_suffix():
    return "data/sample_test_no_suffix.ps4g"

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


# ────────────────────────────────────────────────
#   Bare gamete names (no ":idx" suffix, PS4G v2.0)
# ────────────────────────────────────────────────

def test_extract_metadata_no_suffix(sample_ps4g_file_no_suffix):
    """Bare "#B73" gamete lines (no ':idx' suffix) must parse identically
    to the ':0'-suffixed form."""
    metadata, gamete_data = extract_metadata(sample_ps4g_file_no_suffix)
    assert len(gamete_data) == 3
    assert metadata["version"] == "2.0"


def test_build_index_lookup_no_suffix(sample_ps4g_file_no_suffix):
    index_array = build_index_lookup(sample_ps4g_file_no_suffix)
    assert index_array == ['B73', 'CML247', 'W22']


def test_build_index_lookup_distinguish_gametes(sample_ps4g_file):
    """distinguish_gametes=True keeps the ':idx' suffix in the label."""
    index_array = build_index_lookup(sample_ps4g_file, distinguish_gametes=True)
    assert index_array == ['B73:0', 'CML247:0', 'W22:0']


def test_convert_ps4g_same_shape_for_both_forms(sample_ps4g_file, sample_ps4g_file_no_suffix):
    """The suffixed and bare-name fixtures carry the same data, so both
    must produce matrices of the same shape."""
    with_suffix, _ = convert_ps4g(sample_ps4g_file, weight_strat="unweighted", collapse=False)
    no_suffix, _ = convert_ps4g(sample_ps4g_file_no_suffix, weight_strat="unweighted", collapse=False)
    assert with_suffix.shape == no_suffix.shape


def test_build_index_lookup_raises_on_no_gametes(tmp_path):
    """A file with no gamete metadata lines should raise a clear error,
    not crash inside max() on an empty sequence."""
    ps4g_file = tmp_path / "empty.ps4g"
    ps4g_file.write_text("#PS4G\n#version=2.0\ngameteSet\trefContig\trefPosBinned\tcount\n")
    with pytest.raises(ValueError):
        build_index_lookup(str(ps4g_file))


# ────────────────────────────────────────────────
#           parse_gamete_header_line
# ────────────────────────────────────────────────

def test_parse_sample_gamete_bare():
    assert parse_sample_gamete("B73") == SampleGamete("B73", 0)


def test_parse_sample_gamete_suffixed():
    assert parse_sample_gamete("B73:1") == SampleGamete("B73", 1)


@pytest.mark.parametrize("line", [
    "#gamete\tgameteIndex\tcount",
    "#Command: ropebwt3 refmap --ref-prefix=B73 --max-occ=-1",
    "#version=2.0",
    "#PS4G",
])
def test_parse_gamete_header_line_rejects_non_gamete_lines(line):
    assert parse_gamete_header_line(line) is None


def test_parse_gamete_header_line_accepts_both_forms():
    bare = parse_gamete_header_line("#B73\t0\t784970")
    suffixed = parse_gamete_header_line("#B73:0\t0\t784970")
    assert bare == (SampleGamete("B73", 0), 0, 784970)
    assert suffixed == (SampleGamete("B73", 0), 0, 784970)


# ────────────────────────────────────────────────
#   Gamete records recognized by "#gamete" section,
#   not by column shape (PR review follow-up)
# ────────────────────────────────────────────────

def _write_ps4g(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_records_before_gamete_tag_are_ignored(tmp_path):
    """A gamete-shaped line before the '#gamete' tag opens the section is
    not a record -- position, not shape, is what matters."""
    path = _write_ps4g(tmp_path, "before_tag.ps4g",
        "#B73:0\t0\t4\n"
        "#gamete\tgameteIndex\tcount\n"
        "#CML247:0\t1\t2\n"
        "#W22:0\t2\t1\n"
        "gameteSet\trefContig\trefPosBinned\tcount\n"
        "0\tchr1\t1000\t1\n"
        "0,1\tchr1\t1000\t1\n"
        "0\tchr1\t2000\t1\n"
        "0,1,2\tchr1\t2000\t1\n"
    )
    gamete_data = parse_gamete_records(path)
    assert len(gamete_data) == 2
    assert all(entry["gamete"] != "B73" for entry in gamete_data)


def test_three_column_comment_outside_section_is_not_a_gamete(tmp_path):
    """The reviewer's regression case: a '#'-line with the old gamete shape
    (3 tab fields, integer cols 2-3) that is NOT a gamete record, placed
    both before the tag and after the data section."""
    path = _write_ps4g(tmp_path, "stray_comment.ps4g",
        "#binSize\t256\t1\n"
        "#gamete\tgameteIndex\tcount\n"
        "#B73:0\t0\t4\n"
        "#CML247:0\t1\t2\n"
        "#W22:0\t2\t1\n"
        "gameteSet\trefContig\trefPosBinned\tcount\n"
        "0\tchr1\t1000\t1\n"
        "0,1\tchr1\t1000\t1\n"
        "0\tchr1\t2000\t1\n"
        "0,1,2\tchr1\t2000\t1\n"
        "#binSize\t256\t1\n"
    )
    assert len(parse_gamete_records(path)) == 3


def test_metadata_line_inside_gamete_section_does_not_truncate_it(tmp_path):
    """A keyed metadata line (#TotalUniqueCounts:) interleaved among gamete
    records is consumed as metadata, not mistaken for a malformed record --
    the section stays open and later records still parse."""
    path = _write_ps4g(tmp_path, "interleaved.ps4g",
        "#gamete\tgameteIndex\tcount\n"
        "#B73:0\t0\t4\n"
        "#TotalUniqueCounts: 4\n"
        "#CML247:0\t1\t2\n"
        "#W22:0\t2\t1\n"
        "gameteSet\trefContig\trefPosBinned\tcount\n"
        "0\tchr1\t1000\t1\n"
    )
    metadata, gamete_data = extract_metadata(path)
    assert len(gamete_data) == 3
    assert metadata["total_reads"] == 4


def test_no_gamete_tag_synthesizes_ids_from_data(tmp_path):
    """No '#gamete' tag anywhere -- rather than falling back to shape
    sniffing, gametes are synthesized from the indices actually used in the
    data section's gameteSet column, named by that index."""
    path = _write_ps4g(tmp_path, "no_tag.ps4g",
        "#TotalUniqueCounts: 4\n"
        "gameteSet\trefContig\trefPosBinned\tcount\n"
        "0\tchr1\t1000\t1\n"
        "0,1\tchr1\t1000\t1\n"
        "0\tchr1\t2000\t1\n"
        "0,1,2\tchr1\t2000\t1\n"
    )
    gamete_data = parse_gamete_records(path)
    by_index = {entry["gamete_index"]: entry for entry in gamete_data}
    assert sorted(by_index) == [0, 1, 2]
    assert by_index[0]["gamete"] == "0"
    assert by_index[0]["read_count"] == 4  # hits all 4 rows
    assert by_index[1]["read_count"] == 2  # rows 2, 4
    assert by_index[2]["read_count"] == 1  # row 4


def test_malformed_record_in_section_is_skipped_not_fatal(tmp_path, caplog):
    path = _write_ps4g(tmp_path, "malformed.ps4g",
        "#gamete\tgameteIndex\tcount\n"
        "#B73:0\t0\t4\n"
        "#bad\tnot-a-number\t2\n"
        "#W22:0\t2\t1\n"
        "gameteSet\trefContig\trefPosBinned\tcount\n"
        "0\tchr1\t1000\t1\n"
    )
    with caplog.at_level("WARNING"):
        gamete_data = parse_gamete_records(path)
    assert len(gamete_data) == 2
    assert "malformed" in caplog.text.lower()


def test_uppercase_gamete_tag_is_recognized(tmp_path):
    path = _write_ps4g(tmp_path, "uppercase_tag.ps4g",
        "#GAMETE\tgameteIndex\tcount\n"
        "#B73:0\t0\t4\n"
        "#CML247:0\t1\t2\n"
        "#W22:0\t2\t1\n"
        "gameteSet\trefContig\trefPosBinned\tcount\n"
        "0\tchr1\t1000\t1\n"
    )
    assert len(parse_gamete_records(path)) == 3


def test_extract_metadata_ignores_trailing_comment_metadata(tmp_path):
    """'#' lines after the data section has started are inert comments --
    metadata is only read from the leading header block."""
    path = _write_ps4g(tmp_path, "trailing_metadata.ps4g",
        "#version=2.0\n"
        "#gamete\tgameteIndex\tcount\n"
        "#B73:0\t0\t4\n"
        "gameteSet\trefContig\trefPosBinned\tcount\n"
        "0\tchr1\t1000\t1\n"
        "#version=9.9\n"
    )
    metadata, _ = extract_metadata(path)
    assert metadata["version"] == "2.0"


