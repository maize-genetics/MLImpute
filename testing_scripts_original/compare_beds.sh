#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 bed_file_1 bed_file_2" >&2
    exit 1
fi

bed1="$1"
bed2="$2"

awk '
NR == FNR {
    file1[FNR] = $0
    max1 = FNR
    next
}

{
    line_num = FNR

    if (!(line_num in file1)) {
        print "Line " line_num ": extra in file 2"
        print "file2:", $0
        print ""
    } else if ($0 != file1[line_num]) {
        print "Line " line_num ": different"
        print "file1:", file1[line_num]
        print "file2:", $0
        print ""
    }

    seen[line_num] = 1
}

END {
    for (i = 1; i <= max1; i++) {
        if (!(i in seen)) {
            print "Line " i ": extra in file 1"
            print "file1:", file1[i]
            print ""
        }
    }
}
' "$bed1" "$bed2"