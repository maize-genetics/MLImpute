import pandas as pd
import subprocess
import os
import argparse

def build_fasta_keys(parents, founder):
    fasta_df = pd.DataFrame(
        columns=["fa_chr", "fa_start", "fa_end", "parent_chr", "parent_start", "parent_end", "parent", "seg_length"])

    founder_df = pd.DataFrame(columns=["ref_chr", "ref_start", "ref_end", "parent_chr", "parent_start", "parent_end", "founder"])
    key = []

    for parent in parents: # for each parent, subset founder and aggregate
        parent_df = pd.read_csv(f"{parent}_key.bed", sep="\t", header=None,
                                names=["parent_chr", "parent_start", "parent_end", "ref_chr", "ref_start", "ref_end", "founder"])
        parent_subset = parent_df[parent_df["founder"] == int(founder)]
        founder_df = pd.concat([founder_df, parent_subset], ignore_index=True)
        key.extend([parent]*len(parent_subset))

    founder_df["parent"] = key
    # sort based on ref_chr, ref_start, ref_end
    founder_df = founder_df.sort_values(by=["ref_chr", "ref_start", "ref_end"], ascending=[True, True, True])
    fasta_df["fa_chr"] = founder_df["ref_chr"]
    fasta_df["seg_len"] = (founder_df["parent_end"] - founder_df["parent_start"]).astype(int)
    fasta_df["fa_end"] = fasta_df.groupby("fa_chr")["seg_len"].cumsum()
    fasta_df["fa_start"] = fasta_df["fa_end"] - fasta_df["seg_len"]
    fasta_df = fasta_df.drop(columns=["seg_len"])
    fasta_df["parent_chr"] = founder_df["parent_chr"]
    fasta_df["parent_start"] = founder_df["parent_start"]
    fasta_df["parent_end"] = founder_df["parent_end"]
    fasta_df["parent"] = founder_df["parent"]

    fasta_df.to_csv(f"{founder}_key.bed", sep="\t", index=False, header=False)

def adjust_coords(df, length):
    df.loc[df.index[0], "parent_start"] = 0
    end = df.loc[df.index[0], "parent_end"]

    # iterate from second row onward
    for i in df.index[1:]:
        if df.at[i, "parent_start"] != end:
            df.at[i, "parent_start"] = end
        end = df.at[i, "parent_end"]

    # set last end to full chromosome length
    df.loc[df.index[-1], "parent_end"] = length

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain_files", type=int)
    args = parser.parse_args()

    NAM_founders = ["CML228", "CML322", "CML69", "Ki11", "M162W", "Ms71", "Oh43", "B97", "CML247", "CML333", "HP301", "Ki3",
                    "M37W", "NC350", "Oh7B", "Tzi8", "CML103", "CML277", "CML52", "Il14H", "Ky21", "Mo18W", "NC358", "P39"]

    founder_chroms = {"CML228" : {"chr1" : 311_577_201, "chr2" : 244_763_794, "chr3" : 239_761_349, "chr4" : 254_676_231, "chr5" : 228_823_570,
                                  "chr6" : 175_338_271, "chr7" : 181_540_992, "chr8" : 186_718_620, "chr9" : 167_687_098, "chr10" : 149_925_833},
                      "CML322" : {"chr1" : 304_784_548, "chr2" : 243_324_309, "chr3" : 239_728_129, "chr4" : 257_539_888, "chr5" : 221_631_323,
                                  "chr6" : 175_362_398, "chr7" : 178_688_969, "chr8" : 180_302_813, "chr9" : 164_706_825, "chr10" : 150_084_177},
                      "CML69" : {"chr1" : 305_788_947, "chr2" : 242_308_534, "chr3" : 239_259_890, "chr4" : 255_185_478, "chr5" : 218_448_888,
                                 "chr6" : 173_207_479, "chr7" : 180_887_457, "chr8" : 180_190_510, "chr9" : 162_111_082, "chr10" : 153_170_938},
                      "Ki11" : {"chr1" : 310_658_259, "chr2" : 252_411_040, "chr3" : 240_499_622, "chr4" : 252_694_885, "chr5" : 224_468_191,
                                "chr6" : 178_360_981, "chr7" : 184_781_161, "chr8" : 182_190_944, "chr9" : 165_949_266, "chr10" : 153_305_303},
                      "M162W" : {"chr1" : 306_203_930, "chr2" : 242_470_251, "chr3" : 237_156_436, "chr4" : 252_621_820, "chr5" : 222_291_023,
                                 "chr6" : 186_544_510, "chr7" : 183_763_309, "chr8" : 209_474_767, "chr9" : 166_176_350, "chr10" : 153_192_159},
                      "Ms71" : {"chr1" : 310_202_964, "chr2" : 243_466_302, "chr3" : 240_604_834, "chr4" : 253_912_153, "chr5" : 224_736_366,
                                "chr6" : 179_969_880, "chr7" : 184_671_976, "chr8" : 179_905_066, "chr9" : 163_431_469, "chr10" : 149_277_564},
                      "Oh43" : {"chr1" : 306_440_704, "chr2" : 248_330_814, "chr3" : 240_829_167, "chr4" : 253_228_108, "chr5" : 221_656_378,
                                "chr6" : 179_113_859, "chr7" : 180_662_582, "chr8" : 181_014_384, "chr9" : 167_198_378, "chr10" : 152_214_488},
                      "B97" : {"chr1" : 307_632_032, "chr2" : 252_179_072, "chr3" : 242_347_007, "chr4" : 252_232_367, "chr5" : 222_239_690,
                               "chr6" : 180_431_575, "chr7" : 182_842_045, "chr8" : 183_646_229, "chr9" : 165_161_870, "chr10" : 150_691_437},
                      "CML247" : {"chr1" : 328_910_756, "chr2" : 263_791_054, "chr3" : 240_585_047, "chr4" : 254_335_221, "chr5" : 225_397_941,
                                  "chr6" : 178_806_624, "chr7" : 181_641_468, "chr8" : 185_239_947, "chr9" : 164_3086_29, "chr10" : 153_508_852},
                      "CML333" : {"chr1" : 314_488_983, "chr2" : 249_665_033, "chr3" : 243_215_679, "chr4" : 251_519_228, "chr5" : 225_739_712,
                                  "chr6" : 180_183_266, "chr7" : 184_022_950, "chr8" : 187_782_559, "chr9" : 173_472_473, "chr10" : 151_644_700},
                      "HP301" : {"chr1" : 307_047_731, "chr2" : 250_141_472, "chr3" : 240_039_523, "chr4" : 252_291_229, "chr5" : 221_343_133,
                                 "chr6" : 177_507_800, "chr7" : 181_725_512, "chr8" : 178_051_958, "chr9" : 164_342_879, "chr10" : 151_729_224},
                      "Ki3" : {"chr1" : 312_622_006, "chr2" : 246_815_332, "chr3" : 245_550_371, "chr4" : 252_658_346, "chr5" : 227_524_873,
                               "chr6" : 188_595_388, "chr7" : 179_755_421, "chr8" : 180_35_2562, "chr9" : 164_864_428, "chr10" : 152_190_761},
                      "M37W" : {"chr1" : 306_885_972, "chr2" : 244_314_634, "chr3" : 242_405_911, "chr4" : 251_302_864, "chr5" : 221_795_845,
                                "chr6" : 186_724_520, "chr7" : 183_487_981, "chr8" : 180_780_255, "chr9" : 166_919_533, "chr10" : 155_388_826},
                      "NC350" : {"chr1" : 310_842_591, "chr2" : 249_001_196, "chr3" : 243_851_070, "chr4" : 254_309_806, "chr5" : 223_249_382,
                                 "chr6" : 182_993_951, "chr7" : 180_105_666, "chr8" : 181_843_295, "chr9" : 169_120_828, "chr10" : 151_937_703},
                      "Oh7B" : {"chr1" : 307_239_180, "chr2" : 245_704_635, "chr3" : 237_596_221, "chr4" : 251_422_589, "chr5" : 223_847_857,
                                "chr6" : 180_001_022, "chr7" : 184_573_146, "chr8" : 182_109_197, "chr9" : 205_794_433, "chr10" : 111_214_652},
                      "Tzi8" : {"chr1" : 305_803_203, "chr2" : 245_299_735, "chr3" : 239_545_720, "chr4" : 254_136_052, "chr5" : 221_425_360,
                                "chr6" : 178_662_525, "chr7" : 184_517_021, "chr8" : 182_126_902, "chr9" : 167_326_923, "chr10" : 154_632_701},
                      "CML103" : {"chr1" : 305_897_857, "chr2" : 241_481_138, "chr3" : 238_563_298, "chr4" : 252_598_624, "chr5" : 222_463_916,
                                  "chr6" : 176_230_381, "chr7" : 181_096_865, "chr8" : 179_3524_51, "chr9" : 166_148_394, "chr10" : 149_914_000},
                      "CML277" : {"chr1" : 308_867_798, "chr2" : 247_283_677, "chr3" : 242_596_678, "chr4" : 250_135_365, "chr5" : 220_513_029,
                                  "chr6" : 176_052_463, "chr7" : 181_030_607, "chr8" : 182_825_486, "chr9" : 167_429_005, "chr10" : 150_946_910},
                      "CML52" : {"chr1" : 318_156_096, "chr2" : 259_708_713, "chr3" : 243_974_402, "chr4" : 262_669_390, "chr5" : 231707303,
                                 "chr6" : 182_704_949, "chr7" : 186_094_419, "chr8" : 190_099_871, "chr9" : 169_172_938, "chr10" : 153_584_098},
                      "Il14H" : {"chr1" : 300_139_798, "chr2" : 244_722_702, "chr3" : 235_706_095, "chr4" : 257_978_302, "chr5" : 223_951_407,
                                 "chr6" : 171_793_723, "chr7" : 178_410_452, "chr8" : 178_543_749, "chr9" : 167_710_400, "chr10" : 152_911_007},
                      "Ky21" : {"chr1" : 311_254_152, "chr2" : 243_376_689, "chr3" : 237_194_928, "chr4" : 247_764_743, "chr5" : 219_724_524,
                                "chr6" : 174_022_551, "chr7" : 182_197_498, "chr8" : 178_924_872, "chr9" : 166_334_496, "chr10" : 150_109_315},
                      "Mo18W" : {"chr1" : 307_773_640, "chr2" : 247_669_541, "chr3" : 244_467_206, "chr4" : 252_758_625, "chr5" : 226_800_839,
                                 "chr6" : 180_377_583, "chr7" : 180_330_399, "chr8" : 182_323_521, "chr9" : 164_132_304, "chr10" : 148_387_378},
                      "NC358" : {"chr1" : 306_333_774, "chr2" : 277_154_671, "chr3" : 239_682_009, "chr4" : 253_506_590, "chr5" : 221_841_946,
                                 "chr6" : 184_556_742, "chr7" : 181_107_323, "chr8" : 177_955_777, "chr9" : 165_009_261, "chr10" : 156_023_696},
                      "P39" : {"chr1" : 302_421_781, "chr2" : 244_619_812, "chr3" : 242_478_718, "chr4" : 275_636_967, "chr5" : 222_867_812,
                               "chr6" : 177_971_375, "chr7" : 206_991_990, "chr8" : 176_984_287, "chr9" : 164_153_970, "chr10" : 148_196_188}}

    # add duplicate columns to refkey files
    for founder in NAM_founders:
        refkey = f"{founder}_refkey.bed"
        out = f"{founder}_refkey_temp.bed"
        df = pd.read_csv(refkey, sep="\t", header=None)
        df_new = df[[0, 1, 2, 0, 1, 2, 3]]  # zero-based indexing
        df_new.to_csv(out, sep="\t", header=False, index=False)

    for founder in NAM_founders:
        # File paths
        temp_bed = f"{founder}_refkey_temp.bed"
        chain_file = os.path.join(args.chain_dir, f"{founder}.chain")
        out_bed = f"{founder}_key.bed"

        # Run CrossMap
        subprocess.run([
            "CrossMap",
            "bed",
            chain_file,
            temp_bed,
            out_bed
        ], check=True)

        # Remove temporary file
        os.remove(temp_bed)

    # sort by parent coords and fill in missing chunks
    for founder in NAM_founders:
        key = f"{founder}_key.bed"
        key_df = pd.read_csv(key, sep="\t", header=None, names=["parent_chr", "parent_start", "parent_end", "ref_chr", "ref_start", "ref_end", "founder"])

        # sort once
        sorted_df = key_df.sort_values(
            by=["parent_chr", "parent_start", "parent_end"],
            ascending=[True, True, True]
        )

        # adjust per chromosome, collect parts
        adjusted_parts = []
        for c, length in founder_chroms[founder].items():
            chunk = sorted_df[sorted_df["parent_chr"] == c]
            if chunk.empty:
                continue
            adj = adjust_coords(chunk, int(length))
            # keep the zero-length guard (belt & suspenders)
            adj = adj[adj["parent_start"] != adj["parent_end"]]
            adjusted_parts.append(adj)

        if adjusted_parts:
            out_df = pd.concat(adjusted_parts, ignore_index=True)
        else:
            out_df = sorted_df

        out_df.to_csv(f"{founder}_key.bed", sep="\t", index=False, header=False)

    for i in range(len(NAM_founders)):
        build_fasta_keys(NAM_founders, i)