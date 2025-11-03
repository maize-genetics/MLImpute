from time import perf_counter
import argparse
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm
import math
from multiprocessing import get_context

def count_gaps(seq, bases):
    gaps = 0
    counter = 0
    for base in seq:
        if counter > bases: break
        if base == "-": gaps += 1
        else: counter += 1
    return gaps

def convert_coord(s_info, ref_info, ref_start, ref_end):
    """
    s_info["start"] is the parent alignment start
    ref_info["start"] is the ref alignment start
    ref_start is the desired ref start
    s_info["end"] is the parent alignment end
    ref_info["end"] is the ref alignment end
    ref_end is the desired ref end
    """

    # CASE 1: desired sequence starts before s and ends before s
    if ref_start < ref_info["start"] and ref_end <= ref_info["end"]:
        bases_to_ref_end = ref_end - ref_info["start"]
        gaps_to_ref_end = count_gaps(ref_info["seq"], bases_to_ref_end)  # number of gaps in ref before reaching desired end
        parent_idx_end = bases_to_ref_end + gaps_to_ref_end
        gaps_to_parent_end = count_gaps(s_info["seq"][:parent_idx_end], len(s_info["seq"][:parent_idx_end]))

        if s_info["strand"] == "+":
            parent_start = s_info["start"]
            parent_end = s_info["start"] + parent_idx_end - gaps_to_parent_end
        else:
            parent_start = s_info["chr_length"] - (s_info["start"] + s_info["length"])
            parent_end = parent_start + parent_idx_end - gaps_to_parent_end

    # CASE 2: desired sequence starts before s and ends after s
    elif ref_start < ref_info["start"] and ref_end > ref_info["end"]:
        if s_info["strand"] == "+":
            parent_start = s_info["start"]
            parent_end = s_info["end"]
        else:
            parent_start = s_info["chr_length"] - (s_info["start"] + s_info["length"])
            parent_end = parent_start + s_info["length"]

    # CASE 3: desired sequence starts after s and ends before s (fully contained)
    elif ref_start >= ref_info["start"] and ref_end <= ref_info["end"]:
        bases_to_ref_start = ref_start - ref_info["start"]
        gaps_to_ref_start = count_gaps(ref_info["seq"], bases_to_ref_start)  # number of gaps in ref before reaching desired start
        parent_idx_start = bases_to_ref_start + gaps_to_ref_start
        gaps_to_parent_start = count_gaps(s_info["seq"][:parent_idx_start], len(s_info["seq"][:parent_idx_start]))

        bases_to_ref_end = ref_end - ref_info["start"]
        gaps_to_ref_end = count_gaps(ref_info["seq"], bases_to_ref_end)  # number of gaps in ref before reaching desired end
        parent_idx_end = bases_to_ref_end + gaps_to_ref_end
        gaps_to_parent_end = count_gaps(s_info["seq"][:parent_idx_end], len(s_info["seq"][:parent_idx_end]))

        if s_info["strand"] == "+":
            parent_start = s_info["start"] + parent_idx_start - gaps_to_parent_start
            parent_end = s_info["start"] + parent_idx_end - gaps_to_parent_end
        else:
            parent_start_seq = s_info["chr_length"] - (s_info["start"] + s_info["length"])
            parent_start = parent_start_seq + parent_idx_start - gaps_to_parent_start
            parent_end = parent_start_seq + parent_idx_end - gaps_to_parent_end

    # CASE 4: desired sequence starts after s and ends after s
    elif ref_start >= ref_info["start"] and ref_end > ref_info["end"]:
        bases_to_ref_start = ref_start - ref_info["start"]
        gaps_to_ref_start = count_gaps(ref_info["seq"], bases_to_ref_start)  # number of gaps in ref before reaching desired start
        parent_idx_start = bases_to_ref_start + gaps_to_ref_start
        gaps_to_parent_start = count_gaps(s_info["seq"][:parent_idx_start], len(s_info["seq"][:parent_idx_start]))

        if s_info["strand"] == "+":
            parent_start = s_info["start"] + parent_idx_start - gaps_to_parent_start
            parent_end = s_info["end"]
        else:
            parent_start_seq = s_info["chr_length"] - (s_info["start"] + s_info["length"])
            parent_start = parent_start_seq + parent_idx_start - gaps_to_parent_start
            parent_end = parent_start_seq + s_info["length"]

    else:
        parent_start = None
        parent_end = None

    if parent_start == parent_end:
        parent_start = None
        parent_end = None

    return parent_start, parent_end

def adjust_coords(df: pl.DataFrame, length: int) -> pl.DataFrame:
    n = df.height
    if n == 0:
        return df

    # Add a temporary index column
    df = df.with_row_count("idx")

    # Compute parent_start and parent_end adjustments
    df = df.with_columns([
        # parent_start = 0 for first row, else previous parent_end
        pl.when(pl.col("idx") == 0)
          .then(pl.lit(0))
          .otherwise(pl.col("parent_end").shift(1))
          .alias("parent_start"),

        # parent_end = chromosome length for last row, else original value
        pl.when(pl.col("idx") == (n - 1))
          .then(pl.lit(int(length)))
          .otherwise(pl.col("parent_end"))
          .alias("parent_end"),
    ])

    return df.drop("idx")

def maf_table(maf_file):
    cols = [
        "ref_chr", "ref_start", "ref_end", "ref_length", "ref_strand", "ref_chr_length", "ref_seq",
        "s_chr",   "s_start",   "s_end",   "s_length",   "s_strand",   "s_chr_length",   "s_seq"
    ]

    rows = []
    ref_info = None  # holds the previous 's' line (reference)

    with open(maf_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "a")):
                # skip comments and alignment headers; only care about 's' lines
                continue

            if line.startswith("s"):
                # Split on ANY whitespace to prevent trailing spaces from sneaking in
                fields = [f.strip() for f in line.strip().split("\t")]
                # MAF 's' format: s src start size strand srcSize text
                entry = {
                    "chr":        fields[1],
                    "start":      int(fields[2]),
                    "length":     int(fields[3]),
                    "strand":     fields[4],
                    "chr_length": int(fields[5]),
                    "seq":        fields[6],
                }
                entry["end"] = entry["start"] + entry["length"]  # non-inclusive end

                if ref_info is None:
                    # First 's' in a pair -> reference
                    ref_info = entry
                else:
                    # Second 's' in a pair -> sample; emit a row, then reset
                    s_info = entry
                    rows.append({
                        "ref_chr":        ref_info["chr"],
                        "ref_start":      ref_info["start"],
                        "ref_end":        ref_info["end"],
                        "ref_length":     ref_info["length"],
                        "ref_strand":     ref_info["strand"],
                        "ref_chr_length": ref_info["chr_length"],
                        "ref_seq":        ref_info["seq"],
                        "s_chr":          s_info["chr"],
                        "s_start":        s_info["start"],
                        "s_end":          s_info["end"],
                        "s_length":       s_info["length"],
                        "s_strand":       s_info["strand"],
                        "s_chr_length":   s_info["chr_length"],
                        "s_seq":          s_info["seq"],
                    })
                    ref_info = None  # ready for next pair

    return pl.DataFrame(rows, schema=cols)

def parse_maf_table(maf_table, ref_chr, ref_start, ref_end):
    parent_coordinates = []
    maf_table_subset = maf_table.filter(
        (pl.col("ref_chr") == ref_chr) &
        (pl.col("ref_end") >= ref_start) &
        (pl.col("ref_start") <= ref_end)
    )

    for row in maf_table_subset.iter_rows(named=True):

        s_info = {
            "chr": row["s_chr"],
            "start": row["s_start"],
            "end": row["s_end"],
            "length": row["s_length"],
            "strand": row["s_strand"],
            "chr_length": row["s_chr_length"],
            "seq": row["s_seq"]
        }
        ref_info = {
            "chr": row["ref_chr"],
            "start": row["ref_start"],
            "end": row["ref_end"],
            "length": row["ref_length"],
            "strand": row["ref_strand"],
            "chr_length": row["ref_chr_length"],
            "seq": row["ref_seq"]
        }
        parent_chr = row["s_chr"]
        parent_start, parent_end = convert_coord(s_info, ref_info, ref_start, ref_end)
        if parent_start is not None:
            parent_coordinates.append((parent_chr,parent_start,parent_end))

    return parent_coordinates

def convert_ref_coords_map(parent, parent_chroms):
    maf_file = f"/workdir/smm477/uncrossed_phg/alignment_files/{parent}.maf"
    parent_df = pl.read_csv(
        f"{parent}_refkey.bed",
        separator="\t", has_header=False,
        new_columns=["chr", "start", "end", "founder"]
    )

    t0 = perf_counter()

    maf = maf_table(maf_file)  # build once

    # map: per row -> list of (parent_chr, start, end)
    mapped = parent_df.clone()

    mapped = mapped.with_columns( # TODO: need these to run concurrently (right now they are still row by row)
        pl.struct(["chr", "start", "end"]).map_elements(
            # Convert the list of tuples into a list of dicts
            lambda s: [
                {"parent_chr": c, "parent_start": int(a), "parent_end": int(b)}
                for (c, a, b) in parse_maf_table(maf, s["chr"], s["start"], s["end"])
            ],
            # Explicitly define the expected return type
            return_dtype=pl.List(
                pl.Struct({
                    "parent_chr": pl.Utf8,
                    "parent_start": pl.Int64,
                    "parent_end": pl.Int64,
                })
            ),
        ).alias("parent_coords")
    )

    # normalize empty results to []
    mapped = mapped.filter(pl.col("parent_coords").list.len() > 0)

    # Expand those into three columns
    mapped = mapped.explode("parent_coords").unnest("parent_coords")

    # choose output columns + rename to match your previous function
    key_df = mapped[[
        "chr", "start", "end", "parent_chr", "parent_start", "parent_end", "founder"
    ]].rename({
        "chr": "ref_chr",
        "start": "ref_start",
        "end": "ref_end",
        "parent_chr": "parent_chr",
        "parent_start": "parent_start",
        "parent_end": "parent_end",
    })

    if key_df.height == 0:
        open(f"{parent}_key.bed", "w").close()
        print(f"[convert_ref_coords_map] wrote 0 rows for {parent} in {perf_counter() - t0:.2f}s")
        return

    # single global sort
    sorted_df = key_df.sort(
        by=["parent_chr", "parent_start", "parent_end"],
        descending=[False, False, False]
    )

    # per-chrom adjust
    adjusted_parts: list[pl.DataFrame] = []
    for c, length in parent_chroms.items():
        chunk = sorted_df.filter(pl.col("parent_chr") == c)
        if chunk.height == 0:
            continue
        adj = adjust_coords(chunk, int(length))  # your Polars version
        adj = adj.filter(pl.col("parent_start") != pl.col("parent_end"))
        adjusted_parts.append(adj)

    out_df = pl.concat(adjusted_parts) if adjusted_parts else sorted_df

    out_df.write_csv(f"{parent}_key.bed", separator="\t", include_header=False)

    t1 = perf_counter()
    print(f"[convert_ref_coords_map] wrote {len(out_df):,} rows for {parent} in {t1 - t0:.2f}s")

# --- keep your count_gaps, convert_coord, adjust_coords, maf_table, parse_maf_table as defined ---

def _process_chr_chunk(shard_path: str, chr_name: str,
                       starts: list[int], ends: list[int], founders: list[str]) -> list[dict]:
    chr_name = str(chr_name).strip()
    maf_chr = pl.read_parquet(shard_path)

    out_rows: list[dict] = []
    for ref_start, ref_end, founder in zip(starts, ends, founders):
        coords = parse_maf_table(maf_chr, chr_name, int(ref_start), int(ref_end))
        if not coords:
            continue
        for parent_chr, ps, pe in coords:
            if ps is None or pe is None or pe <= ps:
                continue
            out_rows.append({
                "ref_chr": chr_name,
                "ref_start": int(ref_start),
                "ref_end": int(ref_end),
                "parent_chr": parent_chr,
                "parent_start": int(ps),
                "parent_end": int(pe),
                "founder": founder,
            })
    return out_rows





def convert_ref_coords_map_chunks(parent: str, parent_chroms: dict[str, int]):
    maf_file = f"/workdir/smm477/uncrossed_phg/alignment_files/{parent}.maf"
    parent_df = pl.read_csv(
        f"{parent}_refkey.bed",
        separator="\t",
        has_header=False,
        new_columns=["chr", "start", "end", "founder"]
    ).with_columns(pl.col("chr").cast(pl.Utf8).str.strip_chars())

    # shard once (your existing function)
    shard_dir = f".maf_shards_{parent}"
    shards = shard_maf_by_chr(maf_file, shard_dir)

    # groups only for chroms that have a shard
    chroms = parent_df.get_column("chr").unique(maintain_order=True).to_list()
    groups = {c: parent_df.filter(pl.col("chr") == pl.lit(c))
              for c in chroms if c in shards}

    total_rows = sum(g.height for g in groups.values())
    if total_rows == 0:
        open(f"{parent}_key.bed", "w").close()
        print(f"[convert_ref_coords_map] wrote 0 rows for {parent}")
        return

    # modest parallelism; avoid oversubscription
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    t0 = perf_counter()
    max_workers = min(max(2, (os.cpu_count() or 1) // 2), 8)
    ctx = get_context("spawn")

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        fut2size: dict = {}
        for chr_name, g in groups.items():
            for fut, sz in _submit_chr_batches(ex, shards[chr_name], chr_name, g):
                fut2size[fut] = sz

        with tqdm(total=total_rows, unit="rows", desc="Mapping (batches)") as pbar:
            for fut in as_completed(fut2size):
                rows = fut.result()
                if rows:
                    all_rows.extend(rows)
                pbar.update(fut2size[fut])

    if not all_rows:
        open(f"{parent}_key.bed", "w").close()
        print(f"[convert_ref_coords_map] wrote 0 rows for {parent} in {perf_counter()-t0:.2f}s")
        return

    # Build DF, sort, per-chrom adjust, write (same as you have)
    key_df = pl.DataFrame(all_rows).sort(["parent_chr", "parent_start", "parent_end"])

    adjusted_parts: list[pl.DataFrame] = []
    for c, length in parent_chroms.items():
        chunk = key_df.filter(pl.col("parent_chr") == pl.lit(c))
        if chunk.height == 0:
            continue
        adj = adjust_coords(chunk, int(length))
        adj = adj.filter(pl.col("parent_start") != pl.col("parent_end"))
        adjusted_parts.append(adj)

    out_df = pl.concat(adjusted_parts) if adjusted_parts else key_df
    out_df.write_csv(f"{parent}_key.bed", separator="\t", include_header=False)
    print(f"[convert_ref_coords_map] wrote {out_df.height:,} rows for {parent} in {perf_counter()-t0:.2f}s")


def shard_maf_by_chr(maf_file: str, out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    maf = maf_table(maf_file).with_columns(
        pl.col("ref_chr").cast(pl.Utf8).str.strip_chars()
    )

    parts = maf.partition_by("ref_chr", as_dict=True)
    shards: dict[str, str] = {}

    for key, g in parts.items():
        # Polars gives tuple keys: ('chr1',)
        chr_name = key[0] if isinstance(key, (tuple, list)) else key
        chr_name = str(chr_name).strip()
        if not chr_name:
            continue

        path = os.path.join(out_dir, f"{chr_name}.parquet")
        if not os.path.exists(path):
            g.write_parquet(path)
        shards[chr_name] = path
    return shards



BATCH_SIZE = 20_000  # tune: aim for ~0.5–2s per batch

def _submit_chr_batches(ex, shard_path: str, chr_name: str, g: pl.DataFrame):
    """Yield (future, batch_size) for each batch of rows in chromosome group g."""
    n = g.height
    if n == 0:
        return
    nbatches = math.ceil(n / BATCH_SIZE)
    for i in range(nbatches):
        lo = i * BATCH_SIZE
        hi = min((i + 1) * BATCH_SIZE, n)
        sub = g.slice(lo, hi - lo)
        fut = ex.submit(
            _process_chr_chunk,
            shard_path,
            chr_name,
            sub["start"].to_list(),
            sub["end"].to_list(),
            sub["founder"].to_list(),
        )
        yield fut, (hi - lo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=str)
    args = parser.parse_args()

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

    convert_ref_coords_map_chunks(args.parent, founder_chroms[args.parent])