
############
# DONT USE #
############



#!/bin/bash
# Usage: ./fix_bed_boundaries.sh <chr_lengths_file> <bed_file>
# chr_lengths_file format: ">chr1 308452471"
# bed_file format: tab-separated with header: chrom start end parent1 parent2

awk '
BEGIN { OFS="\t" }

# First file: load chr lengths
NR==FNR {
    chr = substr($1, 2)     # strip leading ">"
    len[chr] = $2
    next
}

# Second file: BED
/^chrom/ { print; next }   # pass header through unchanged

{
    if ($1 != cur_chr) {
        if (cur_chr != "") flush()
        cur_chr = $1
        delete lines
        nlines = 0
    }
    lines[++nlines] = $0
}

END { if (cur_chr != "") flush() }

function flush(    i, n, flds) {
    # set start of first line to 0
    n = split(lines[1], flds, "\t")
    flds[2] = 0
    lines[1] = rejoin(flds, n)

    # set end of last line to chr length
    n = split(lines[nlines], flds, "\t")
    flds[3] = len[cur_chr]
    lines[nlines] = rejoin(flds, n)

    for (i = 1; i <= nlines; i++)
        print lines[i]
}

function rejoin(arr, n,    i, s) {
    s = arr[1]
    for (i = 2; i <= n; i++) s = s OFS arr[i]
    return s
}
' "$1" "$2"
