import numpy as np
import pandas as pd

from python.ps4g_io.ps4g import decode_position, build_index_lookup


def output_bed_file(output_bed, chroms, final_predictions, index_array, positions, collapse_bed_regions=True):
    bed_df = pd.DataFrame({
        # TODO: convert chr_idx to chr
        "chrom_idx": chroms[:len(final_predictions)],
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
    # Define group boundaries where parent1, parent2, or chrom changes
    group_change = (
            (bed_df["parent1"] != bed_df["parent1"].shift()) |
            (bed_df["parent2"] != bed_df["parent2"].shift()) |
            (bed_df["chrom_idx"] != bed_df["chrom_idx"].shift())
    )
    group_id = group_change.cumsum()
    # Collapse into ranges
    ranges_df = bed_df.groupby(group_id).agg({
        "chrom_idx": "first",
        "pos": ["min", "max"],
        "parent1": "first",
        "parent2": "first"
    }).reset_index(drop=True)
    # Clean up MultiIndex columns
    ranges_df.columns = ["chrom_idx", "start", "end", "parent1", "parent2"]
    # Save to BED file
    ranges_df.to_csv(output_bed, sep="\t", index=False)


def output_predictions(ps4g_file, output_bed, final_predictions, collapse_bed_regions = True):
    spline_pos = pd.read_csv(ps4g_file, sep="\t", comment="#")['pos']
    decoded = np.vstack(np.vectorize(decode_position)(spline_pos)).T
    chroms, positions = zip(*decoded)
    index_array = build_index_lookup(ps4g_file)
    output_bed_file(output_bed, chroms, final_predictions, index_array, positions, collapse_bed_regions)
