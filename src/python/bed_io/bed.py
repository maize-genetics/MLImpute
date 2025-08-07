import numpy as np
import pandas as pd

from python.ps4g_io.ps4g import decode_position, build_index_lookup


def output_bed_file(output_bed, chroms, final_predictions, index_array, positions):
    bed_df = pd.DataFrame({
        # TODO: convert chr_idx to chr
        "chrom_idx": chroms[:len(final_predictions)],
        "pos": positions[:len(final_predictions)],
        "parent1": np.array(index_array)[final_predictions[:, 0]],
        "parent2": np.array(index_array)[final_predictions[:, 1]],
    })
    # Save to BED file
    bed_df.to_csv(output_bed, sep="\t", index=False)


def output_predictions(ps4g_file, output_bed, final_predictions):
    spline_pos = pd.read_csv(ps4g_file, sep="\t", comment="#")['pos']
    decoded = np.vstack(np.vectorize(decode_position)(spline_pos)).T
    chroms, positions = zip(*decoded)
    index_array = build_index_lookup(ps4g_file)
    output_bed_file(output_bed, chroms, final_predictions, index_array, positions)
