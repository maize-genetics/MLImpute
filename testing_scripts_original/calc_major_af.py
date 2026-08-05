#!/usr/bin/env python3
import sys
import gzip


def open_vcf(path):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.vcf[.gz]>", file=sys.stderr)
        sys.exit(1)

    vcf_path = sys.argv[1]
    n_samples = 24

    total = 0.0
    n_sites = 0

    with open_vcf(vcf_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue

            fields = line.rstrip('\n').split('\t')
            if len(fields) < 10:
                continue

            count_0 = 0
            count_1 = 0

            for allele in fields[9:]:
                if allele == '0':
                    count_0 += 1
                elif allele == '1':
                    count_1 += 1

            major_count = max(count_0, count_1)
            total += major_count / n_samples
            n_sites += 1

    if n_sites == 0:
        print("No variant sites found.", file=sys.stderr)
        sys.exit(1)

    avg = total / n_sites
    print(f"Sites:                        {n_sites}")
    print(f"Average major allele freq:    {avg:.6f}")


if __name__ == '__main__':
    main()
