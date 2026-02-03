from collections import defaultdict
from typing import List, Tuple, Optional

Row = Tuple[str, int, int, List[str]]  # chrom, start, end, extra_cols


def parse_bed(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                # skip header lines like: chrom start end ...
                continue
            extra = parts[3:]
            rows.append((chrom, start, end, extra))
    return rows

def read_fai_lengths(fai_path: str) -> dict:
    lengths = {}
    with open(fai_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                lengths[parts[0]] = int(parts[1])
    return lengths

def make_contiguous(rows: List[Row], chr_lengths: Optional[dict] = None) -> List[Row]:
    by_chr = defaultdict(list)
    for r in rows:
        by_chr[r[0]].append(r)

    out: List[Row] = []
    for chrom, lst in by_chr.items():
        # sort by start, then end
        lst.sort(key=lambda x: (x[1], x[2]))

        if not lst:
            continue

        # force first start = 0
        chrom_rows = [list(x) for x in lst]  # mutable
        chrom_rows[0][1] = 0

        # force adjacency by midpoint between end_i and start_{i+1}
        for i in range(len(chrom_rows) - 1):
            end_i = chrom_rows[i][2]
            start_next = chrom_rows[i + 1][1]
            if end_i != start_next:
                mid = (end_i + start_next) // 2
                chrom_rows[i][2] = mid
                chrom_rows[i + 1][1] = mid

        # force last end to chromosome length if provided
        if chr_lengths and chrom in chr_lengths:
            chrom_rows[-1][2] = chr_lengths[chrom]

        # sanity: ensure non-decreasing coordinates
        for i, (c, s, e, extra) in enumerate(chrom_rows):
            if e < s:
                # if something weird happened, clamp
                chrom_rows[i][2] = s

        out.extend((c, s, e, extra) for (c, s, e, extra) in chrom_rows)

    # optional: global sort for nicer output
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def write_bed(rows: List[Row], path: str) -> None:
    with open(path, "w") as f:
        for chrom, start, end, extra in rows:
            f.write("\t".join([chrom, str(start), str(end), *extra]) + "\n")