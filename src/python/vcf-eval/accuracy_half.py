#!/usr/bin/env python3
"""
Allele-aware GT comparison between two VCFs using bcftools query.

Rules implemented:
- Match sites by (CHROM, POS, REF). ALT may differ.
- Compare genotypes by mapping GT allele indices to actual allele strings (REF/ALT),
  then comparing the allele multiset (phase-insensitive by default).
- Treat missing genotypes (./., .|., .) as missing. Missing does not count as match/mismatch.
- Stream-friendly: can optionally write a per-site TSV for *every* site without storing in memory.

Requires:
- bcftools in PATH

Example:
  ./vcf_gt_allele_compare.py --truth truth.vcf.gz --imputed imp.vcf.gz -s SAMPLE \
    --out-sites allele_compare_sites.tsv

Notes:
- If your VCFs are very large, the --out-sites file can be huge.
  Use --only-mismatches or --only-matched-sites to reduce output.
"""

import argparse
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterator

Key = Tuple[str, int, str]  # (CHROM, POS, REF)
MISSING_GTS = {"./.", ".|.", "."}


def write_query_tsv(vcf: str, out_tsv: str, sample: Optional[str], region: Optional[str]) -> None:
    """
    Extract CHROM, POS, REF, ALT, GT from a VCF using bcftools query.
    ALT may include commas; we keep it as one field.
    """
    fmt = r"%CHROM\t%POS\t%REF\t%ALT\t%INFO\t[%GT]\n"
    cmd = ["bcftools", "query", "-f", fmt]
    if sample:
        cmd += ["-s", sample]
    if region:
        cmd += ["-r", region]
    cmd += [vcf]

    with open(out_tsv, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            sys.stderr.write(p.stderr)
            raise SystemExit(p.returncode)


def sort_tsv(in_tsv: str, out_tsv: str) -> None:
    """
    Sort by CHROM (version sort), POS (numeric), REF.
    """
    cmd = ["bash", "-lc", f"LC_ALL=C sort -k1,1V -k2,2n -k3,3 {in_tsv} > {out_tsv}"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(p.returncode)


def parse_gt(gt: str) -> Optional[List[str]]:
    """
    Return list of allele-index strings from GT (e.g., ['0','1']).
    Return None if missing/unknown.
    Supports diploid (0/1, 1|0) and haploid-like (0, 1) encodings.
    """
    gt = gt.strip()
    if not gt or gt in MISSING_GTS:
        return None

    # Determine separator
    sep = "/" if "/" in gt else ("|" if "|" in gt else None)
    if sep is None:
        # haploid-like genotype such as "0" or "1"
        if gt == ".":
            return None
        return [gt]

    parts = gt.split(sep)
    if any(p == "." or p == "" for p in parts):
        return None
    return parts


def gt_to_allele_multiset(
    ref: str,
    alt_field: str,
    gt: str,
    phase_sensitive: bool = False,
) -> Optional[Tuple[str, ...]]:
    """
    Convert GT to a multiset (sorted tuple) of allele strings using REF/ALT strings.
    If phase_sensitive=True, preserve within-genotype order; otherwise sort alleles to ignore phase.
    Returns None for missing/unparseable GT.
    """
    gt_parts = parse_gt(gt)
    if gt_parts is None:
        return None

    alts = [] if alt_field in {".", ""} else alt_field.split(",")

    allele_strings: List[str] = []
    for a in gt_parts:
        try:
            idx = int(a)
        except ValueError:
            return None

        if idx == 0:
            allele_strings.append(ref)
        else:
            j = idx - 1
            if j < 0 or j >= len(alts):
                # GT refers to an ALT index that doesn't exist for this record
                return None
            allele_strings.append(alts[j])

    if phase_sensitive:
        return tuple(allele_strings)
    return tuple(sorted(allele_strings))



def allele_multiset_score(truth_alleles: Tuple[str, ...], imputed_alleles: Tuple[str, ...]) -> float:
    """Return fractional concordance based on multiset allele overlap.

    Score = |intersection multiset| / len(truth_alleles).

    For diploid truth genotypes, this gives:
      - 1.0 for exact match
      - 0.5 if exactly one allele matches (and one doesn't)
      - 0.0 if none match
    """
    if not truth_alleles:
        return 0.0
    t = Counter(truth_alleles)
    i = Counter(imputed_alleles)
    inter = sum((t & i).values())
    return inter / float(len(truth_alleles))

def iter_records(path: str) -> Iterator[Tuple[Key, str, str]]:
    """
    Yield (key=(CHROM,POS,REF), ALT, GT) from a sorted TSV.
    """
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            chrom, pos_s, ref, alt, info, gt = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            try:
                pos = int(pos_s)
            except ValueError:
                continue
            yield (chrom, pos, ref), alt, info, gt


def compare_sorted(
    truth_sorted: str,
    imputed_sorted: str,
    max_report: int,
    phase_sensitive: bool,
    partial_credit: bool,
    out_fh=None,
    only_mismatches: bool = False,
    only_matched_sites: bool = False,
    missing_as_ref: bool = False,
) -> Dict:
    """
    Stream-compare two sorted TSVs.

    If out_fh is provided, write per-site TSV:
      TYPE CHROM POS REF TRUTH_ALT TRUTH_GT IMPUTED_ALT IMPUTED_GT

    Filtering for output file:
    - only_mismatches: write only MISMATCH_ALLELE (and optionally site-missing/extra if you want them too;
      here we include them because they're informative)
    - only_matched_sites: write only sites where key matches; suppress EXTRA/MISSING site lines
    """
    t_iter = iter_records(truth_sorted)
    i_iter = iter_records(imputed_sorted)

    def next_or_none(it):
        try:
            return next(it)
        except StopIteration:
            return None

    def write_site(kind: str, key: Key, t_alt, t_gt, i_alt, i_gt):
        if not out_fh:
            return
        if only_matched_sites and kind in {"MISSING_IN_IMPUTED_SITE", "EXTRA_IN_IMPUTED_SITE"}:
            return
        if only_mismatches and kind not in {"MISMATCH_ALLELE", "MISSING_IN_IMPUTED_SITE", "EXTRA_IN_IMPUTED_SITE"}:
            return
        chrom, pos, ref = key
        out_fh.write(
            f"{kind}\t{chrom}\t{pos}\t{ref}\t"
            f"{t_alt if t_alt is not None else '.'}\t{t_gt if t_gt is not None else '.'}\t"
            f"{i_alt if i_alt is not None else '.'}\t{i_gt if i_gt is not None else '.'}\n"
        )

    counts = {
        "truth_records": 0,
        "imputed_records": 0,
        "site_key_matches": 0,
        "missing_in_imputed_sites": 0,
        "extra_in_imputed_sites": 0,
        "both_missing_gt": 0,
        "truth_missing_gt": 0,
        "imputed_missing_gt": 0,
        "gt_unparseable": 0,
        "compared_sites": 0,
        "gt_allele_matches": 0,
        "gt_allele_mismatches": 0,
        "partial_credit_sum": 0.0,
        "examples": [],
        "af_bin_total": None,
        "af_bin_correct": None,
        "af_bin_accuracy": None,
        "af_bins": None,
        "het_bin_total": None,
        "het_bin_correct": None,
        "het_bin_accuracy": None,
    }

    freq_bins = [i/20 for i in range(21)]  # 0,0.05,...,1
    NBINS = len(freq_bins) - 1

    bin_total = [0] * NBINS
    bin_correct = [0] * NBINS
    het_bin_total = [0] * NBINS
    het_bin_correct = [0] * NBINS

    t = next_or_none(t_iter)
    i = next_or_none(i_iter)

    while t is not None or i is not None:
        if t is not None and (i is None or t[0] < i[0]):
            key, t_alt, t_info, t_gt = t
            counts["truth_records"] += 1

            if missing_as_ref:
                # Treat completely missing site as reference genotype
                truth_parts = parse_gt(t_gt)
                if truth_parts is None:
                    counts["truth_missing_gt"] += 1
                    write_site("TRUTH_MISSING_GT", key, t_alt, t_gt, None, None)
                else:
                    ploidy = len(truth_parts)
                    ref_alleles = tuple([key[2]] * ploidy)

                    t_alleles = gt_to_allele_multiset(
                        key[2], t_alt, t_gt, phase_sensitive
                    )

                    counts["compared_sites"] += 1

                    if partial_credit:
                        counts["partial_credit_sum"] += allele_multiset_score(
                            t_alleles, ref_alleles
                        )

                    if t_alleles == ref_alleles:
                        counts["gt_allele_matches"] += 1
                        write_site("MATCH_ALLELE", key, t_alt, t_gt, None, None)
                    else:
                        counts["gt_allele_mismatches"] += 1
                        write_site("MISMATCH_ALLELE", key, t_alt, t_gt, None, None)

            else:
                counts["missing_in_imputed_sites"] += 1
                write_site("MISSING_IN_IMPUTED_SITE", key, t_alt, t_gt, None, None)

            t = next_or_none(t_iter)
            continue

        if i is not None and (t is None or i[0] < t[0]):
            # imputed has a site not in truth
            key, i_alt, i_info, i_gt = i
            counts["imputed_records"] += 1
            counts["extra_in_imputed_sites"] += 1
            write_site("EXTRA_IN_IMPUTED_SITE", key, None, None, i_alt, i_gt)
            if len(counts["examples"]) < max_report:
                counts["examples"].append(("EXTRA_IN_IMPUTED_SITE", key, None, None, i_alt, i_gt))
            i = next_or_none(i_iter)
            continue

        # Keys match: (CHROM,POS,REF)
        key, t_alt, t_info, t_gt = t
        _,  i_alt, i_info, i_gt = i

        counts["truth_records"] += 1
        counts["imputed_records"] += 1
        counts["site_key_matches"] += 1

        t_alleles = gt_to_allele_multiset(key[2], t_alt, t_gt, phase_sensitive=phase_sensitive)
        i_alleles = gt_to_allele_multiset(key[2], i_alt, i_gt, phase_sensitive=phase_sensitive)

        # Determine missing vs unparseable
        t_missing = parse_gt(t_gt) is None
        i_missing = parse_gt(i_gt) is None

        if t_alleles is None and i_alleles is None:
            if t_missing and i_missing:
                counts["both_missing_gt"] += 1
                write_site("BOTH_MISSING_GT", key, t_alt, t_gt, i_alt, i_gt)
            else:
                counts["gt_unparseable"] += 1
                write_site("GT_UNPARSEABLE", key, t_alt, t_gt, i_alt, i_gt)

        elif t_alleles is None:
            if t_missing:
                counts["truth_missing_gt"] += 1
                write_site("TRUTH_MISSING_GT", key, t_alt, t_gt, i_alt, i_gt)
            else:
                counts["gt_unparseable"] += 1
                write_site("GT_UNPARSEABLE", key, t_alt, t_gt, i_alt, i_gt)

        elif i_alleles is None:
            if i_missing:
                if missing_as_ref:
                    # Treat missing imputed GT as reference genotype
                    truth_parts = parse_gt(t_gt)
                    if truth_parts is None:
                        counts["imputed_missing_gt"] += 1
                        write_site("IMPUTED_MISSING_GT", key, t_alt, t_gt, i_alt, i_gt)
                    else:
                        ploidy = len(truth_parts)

                        # Build reference allele multiset directly
                        ref_alleles = tuple([key[2]] * ploidy)

                        t_alleles = gt_to_allele_multiset(
                            key[2], t_alt, t_gt, phase_sensitive
                        )

                        counts["compared_sites"] += 1

                        if partial_credit:
                            counts["partial_credit_sum"] += allele_multiset_score(
                                t_alleles, ref_alleles
                            )

                        if t_alleles == ref_alleles:
                            counts["gt_allele_matches"] += 1
                            write_site("MATCH_ALLELE", key, t_alt, t_gt, i_alt, i_gt)
                        else:
                            counts["gt_allele_mismatches"] += 1
                            write_site("MISMATCH_ALLELE", key, t_alt, t_gt, i_alt, i_gt)
                else:
                    counts["imputed_missing_gt"] += 1
                    write_site("IMPUTED_MISSING_GT", key, t_alt, t_gt, i_alt, i_gt)
            else:
                counts["gt_unparseable"] += 1
                write_site("GT_UNPARSEABLE", key, t_alt, t_gt, i_alt, i_gt)

        else:
            counts["compared_sites"] += 1

            # --- Minor allele accuracy ---
            ac = None
            an = None

            for field in t_info.split(";"):
                if field.startswith("AC="):
                    try:
                        ac = int(field.split("=")[1])
                    except ValueError:
                        pass
                elif field.startswith("AN="):
                    try:
                        an = int(field.split("=")[1])
                    except ValueError:
                        pass

            # Homozygous truth sites
            if (len(set(t_alleles)) == 1):
                if ac is not None and an is not None and an > 0:
                    alt_freq = ac / an
                    ref_freq = 1 - alt_freq

                    # determine which allele is in the truth genotype
                    if t_alt in t_alleles:      # REF allele present
                        af = alt_freq
                    else:
                        af = ref_freq

                    # determine AF bin
                    bin_idx = min(int(af * 20), 19)
                    if bin_idx >= NBINS:
                        bin_idx = NBINS - 1

                    bin_total[bin_idx] += 1

                    if partial_credit:
                        score = allele_multiset_score(t_alleles, i_alleles)
                        bin_correct[bin_idx] += score
                    else:
                        if t_alleles == i_alleles: bin_correct[bin_idx] += 1

                if partial_credit:
                    counts["partial_credit_sum"] += allele_multiset_score(t_alleles, i_alleles)
                if t_alleles == i_alleles:
                    counts["gt_allele_matches"] += 1
                    write_site("MATCH_ALLELE", key, t_alt, t_gt, i_alt, i_gt)
                else:
                    counts["gt_allele_mismatches"] += 1
                    write_site("MISMATCH_ALLELE", key, t_alt, t_gt, i_alt, i_gt)
                    if len(counts["examples"]) < max_report:
                        counts["examples"].append(("MISMATCH_ALLELE", key, t_alt, t_gt, i_alt, i_gt))

            # Heterozygous truth sites
            else:
                # minor allele frequency
                alt_freq = ac / an
                maf = min(alt_freq, 1 - alt_freq)
                bin_idx = min(int(maf * 20), 19)
                het_bin_total[bin_idx] += 1
                het_bin_correct[bin_idx] += allele_multiset_score(t_alleles, i_alleles)

                if partial_credit:
                    counts["partial_credit_sum"] += allele_multiset_score(t_alleles, i_alleles)
                if t_alleles == i_alleles:
                    counts["gt_allele_matches"] += 1
                    write_site("MATCH_ALLELE", key, t_alt, t_gt, i_alt, i_gt)
                else:
                    counts["gt_allele_mismatches"] += 1
                    write_site("MISMATCH_ALLELE", key, t_alt, t_gt, i_alt, i_gt)
                    if len(counts["examples"]) < max_report:
                        counts["examples"].append(("MISMATCH_ALLELE", key, t_alt, t_gt, i_alt, i_gt))


        bin_accuracy = [
            (bin_correct[i] / bin_total[i]) if bin_total[i] > 0 else None
            for i in range(NBINS)
        ]

        counts["af_bin_total"] = bin_total
        counts["af_bin_correct"] = bin_correct
        counts["af_bin_accuracy"] = bin_accuracy
        counts["af_bins"] = freq_bins

        het_bin_accuracy = [
            (het_bin_correct[i] / het_bin_total[i]) if het_bin_total[i] > 0 else None
            for i in range(NBINS)
        ]

        counts["het_bin_total"] = het_bin_total
        counts["het_bin_correct"] = het_bin_correct
        counts["het_bin_accuracy"] = het_bin_accuracy

        t = next_or_none(t_iter)
        i = next_or_none(i_iter)

    return counts


def main():
    ap = argparse.ArgumentParser(
        description="Allele-aware GT comparison between two VCFs. Match by (CHROM,POS,REF); ALT may differ; ./. treated as missing."
    )
    ap.add_argument("--truth", required=True, help="Truth VCF (.vcf/.vcf.gz)")
    ap.add_argument("--imputed", required=True, help="Imputed VCF (.vcf/.vcf.gz)")
    ap.add_argument("-s", "--sample", default=None, help="Sample name to compare (recommended if multi-sample)")
    ap.add_argument("-r", "--region", default=None, help="Optional region restriction, e.g. chr1:1000000-2000000")
    ap.add_argument("--phase-sensitive", action="store_true",
                    help="If set, treat phased order as meaningful (0|1 != 1|0). Default ignores phase.")
    ap.add_argument("--partial-credit", action="store_true",
                    help="If set, report fractional concordance based on allele overlap (e.g., 0.5 for one allele correct in diploids).")
    ap.add_argument("--max-report", type=int, default=20, help="Max example diffs printed to stdout (default 20)")
    ap.add_argument("--out-sites", default=None, help="Write per-site comparison to this TSV file")
    ap.add_argument("--only-mismatches", action="store_true",
                    help="When --out-sites is set, write only mismatches + missing/extra site lines")
    ap.add_argument("--only-matched-sites", action="store_true",
                    help="When --out-sites is set, suppress EXTRA/MISSING site lines; only write matched keys")
    ap.add_argument("--missing-as-ref", action="store_true",
                   help="Treat missing imputed sites/genotypes as reference calls")
    args = ap.parse_args()

    out_fh = None
    if args.out_sites:
        out_fh = open(args.out_sites, "w")
        out_fh.write("TYPE\tCHROM\tPOS\tREF\tTRUTH_ALT\tTRUTH_GT\tIMPUTED_ALT\tIMPUTED_GT\n")

    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            t_raw = str(td / "truth.raw.tsv")
            i_raw = str(td / "imputed.raw.tsv")
            t_sorted = str(td / "truth.sorted.tsv")
            i_sorted = str(td / "imputed.sorted.tsv")

            write_query_tsv(args.truth, t_raw, args.sample, args.region)
            write_query_tsv(args.imputed, i_raw, args.sample, args.region)

            sort_tsv(t_raw, t_sorted)
            sort_tsv(i_raw, i_sorted)

            res = compare_sorted(
                t_sorted,
                i_sorted,
                args.max_report,
                args.phase_sensitive,
                args.partial_credit,
                out_fh=out_fh,
                only_mismatches=args.only_mismatches,
                only_matched_sites=args.only_matched_sites,
                missing_as_ref=args.missing_as_ref,
            )

        # Summary
        print("VCF allele-aware GT compare (key = CHROM,POS,REF; ALT may differ; ./. missing)")
        for k in [
            "truth_records",
            "imputed_records",
            "site_key_matches",
            "missing_in_imputed_sites",
            "extra_in_imputed_sites",
            "both_missing_gt",
            "truth_missing_gt",
            "imputed_missing_gt",
            "gt_unparseable",
            "compared_sites",
            "gt_allele_matches",
            "gt_allele_mismatches",
        ]:
            print(f"  {k:24s} {res[k]}")

        if res["compared_sites"] > 0:
            concord = res["gt_allele_matches"] / res["compared_sites"]
            print(f"  allele_GT_concordance       {concord:.6f}")
            if args.partial_credit:
                frac = res["partial_credit_sum"] / res["compared_sites"]
                print(f"  partial_allele_concordance  {frac:.6f}")
        else:
            print("  allele_GT_concordance       NA (no comparable sites)")
            if args.partial_credit:
                print("  partial_allele_concordance  NA (no comparable sites)")

        freq_bins = res["af_bins"]
        
        for i in range(len(freq_bins)-1):
            low = freq_bins[i]
            high = freq_bins[i + 1]

            acc = res["af_bin_accuracy"][i]
            n = res["af_bin_total"][i]

            print(low, high, acc, n)



        for i in range(len(freq_bins)-10):
            low = res["af_bins"][i]
            high = res["af_bins"][i + 1] if i + 1 < len(res["af_bins"]) else 1.0

            acc = res["het_bin_accuracy"][i]
            n = res["het_bin_total"][i]

            print("HET", low, high, acc, n)

        print((sum(res["af_bin_total"]) + sum(res["het_bin_total"]) )== res["compared_sites"])

        identical_strict = (
            res["missing_in_imputed_sites"] == 0
            and res["extra_in_imputed_sites"] == 0
            and res["gt_allele_mismatches"] == 0
            and res["truth_missing_gt"] == 0
            and res["imputed_missing_gt"] == 0
            and res["both_missing_gt"] == 0
            and res["gt_unparseable"] == 0
        )

        print(f"  IDENTICAL_STRICT            {str(identical_strict).upper()}")

#        if res["examples"]:
#            print("\nExamples:")
#            print("  TYPE\tCHROM\tPOS\tREF\tTRUTH_ALT\tTRUTH_GT\tIMPUTED_ALT\tIMPUTED_GT")
#            for kind, key, t_alt, t_gt, i_alt, i_gt in res["examples"]:
#                chrom, pos, ref = key
#                print(f"  {kind}\t{chrom}\t{pos}\t{ref}\t{t_alt or '.'}\t{t_gt or '.'}\t{i_alt or '.'}\t{i_gt or '.'}")

        if args.out_sites:
            print(f"\nWrote per-site output to: {args.out_sites}")
            if args.only_mismatches:
                print("  (mode: only mismatches + missing/extra sites)")
            if args.only_matched_sites:
                print("  (mode: only matched sites)")

    finally:
        if out_fh:
            out_fh.close()


if __name__ == "__main__":
    main()


