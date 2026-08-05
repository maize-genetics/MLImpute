## Used to sumarize the results of a whole folder of imputed samples into one averaged summary file


import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Summarize imputation evaluation results by coverage, map setting, and imputer."
    )

    parser.add_argument(
        "-i",
        "--input",
        default="/workdir/irk9/data/phg-cassava/eval_redo/results.tsv",
        help="Input results TSV file."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="/workdir/irk9/data/phg-cassava/eval_redo/sum_results_minimac_impute_beagle.tsv",
        help="Output summary TSV file."
    )

    args = parser.parse_args()

    input_tsv = args.input
    output_tsv = args.output

    df = pd.read_csv(input_tsv, sep="\t")

    metrics = [
        "allele_GT_concordance",
        "partial_allele_concordance",
        "R2_overall",
    ]

    group_cols = [
        "coverage",
        "map",
        "imputer",
    ]

    required_cols = group_cols + metrics
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Convert coverage and metrics to numeric values
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")

    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only the four coverage levels you care about
    coverage_levels = [0.01, 0.1, 1, 2]
    df = df[df["coverage"].isin(coverage_levels)]

    if df.empty:
        raise ValueError(
            "No rows found for coverage levels: "
            + ", ".join(map(str, coverage_levels))
        )

    result = (
        df.groupby(group_cols, as_index=False)[metrics]
        .mean()
        .sort_values(["coverage", "map", "imputer"])
    )

    result.to_csv(output_tsv, sep="\t", index=False)

    print(result.to_string(index=False))
    print(f"\nSaved output to: {output_tsv}")


if __name__ == "__main__":
    main()