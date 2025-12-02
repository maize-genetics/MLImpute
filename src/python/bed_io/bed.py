import numpy as np
import pandas as pd

from python.ps4g_io.ps4g import build_index_lookup


def output_bed_file(output_bed, chroms, positions, final_predictions, index_array, collapse_bed_regions=True):
    """
    Output BED file with imputation predictions.

    Args:
        output_bed: Path to output BED file
        chroms: Array of chromosome/contig identifiers
        positions: Array of binned positions
        final_predictions: Array of predicted parent indices (shape: [n, 2])
        index_array: Array mapping gamete indices to gamete names
        collapse_bed_regions: If True, collapse contiguous regions with same parents
    """
    bed_df = pd.DataFrame({
        "chrom": chroms[:len(final_predictions)],
        "pos": positions[:len(final_predictions)],
        "parent1": np.array(index_array)[final_predictions[:, 0]],
        "parent2": np.array(index_array)[final_predictions[:, 1]],
    })
    # Save to BED file
    if not collapse_bed_regions:
        # If not collapsing, we can save directly
        bed_df.to_csv(output_bed, sep="\t", index=False)
    else:
        output_collapse_bed(bed_df, output_bed)


def output_collapse_bed(bed_df, output_bed):
    """
    Collapse contiguous BED regions with the same parent assignments.

    Args:
        bed_df: DataFrame with chrom, pos, parent1, parent2 columns
        output_bed: Path to output BED file
    """
    # Define group boundaries where parent1, parent2, or chrom changes
    group_change = (
            (bed_df["parent1"] != bed_df["parent1"].shift()) |
            (bed_df["parent2"] != bed_df["parent2"].shift()) |
            (bed_df["chrom"] != bed_df["chrom"].shift())
    )
    group_id = group_change.cumsum()
    # Collapse into ranges
    ranges_df = bed_df.groupby(group_id).agg({
        "chrom": "first",
        "pos": ["min", "max"],
        "parent1": "first",
        "parent2": "first"
    }).reset_index(drop=True)
    # Clean up MultiIndex columns
    ranges_df.columns = ["chrom", "start", "end", "parent1", "parent2"]
    # Save to BED file
    ranges_df.to_csv(output_bed, sep="\t", index=False)


def output_predictions(ps4g_file, output_bed, final_predictions, collapse_bed_regions=True):
    """
    Output imputation predictions to BED format file.

    Args:
        ps4g_file: Path to input PS4G file
        output_bed: Path to output BED file
        final_predictions: Array of predicted parent indices (shape: [n, 2])
        collapse_bed_regions: If True, collapse contiguous regions with same parents
    """
    # Read the PS4G file to get chromosome and position information
    ps4g_df = pd.read_csv(ps4g_file, sep="\t", comment="#")
    chroms = ps4g_df['refContig'].values
    positions = ps4g_df['refPosBinned'].values

    # Get gamete index to name mapping
    index_array = build_index_lookup(ps4g_file)

    # Output to BED file
    output_bed_file(output_bed, chroms, positions, final_predictions, index_array, collapse_bed_regions)
