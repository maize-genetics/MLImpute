## Used to sumarize the results of a whole folder of imputed samples into one averaged summary file

import argparse
import pandas as pd


def main():
    input_tsv = "/workdir/irk9/data/phg-maize/eval/results.tsv"
    output_tsv = "/workdir/irk9/data/phg-maize/eval/sum_results.tsv"

    df = pd.read_csv(input_tsv, sep="\t")

    metrics = [
        "allele_GT_concordance",
        "partial_allele_concordance",
        "R2_overall",
    ]

    required_cols = ["coverage"] + metrics
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = (
        df.groupby("coverage", as_index=False)[metrics]
        .mean()
        .sort_values("coverage")
    )

    result.to_csv(output_tsv, sep="\t", index=False)

    print(result.to_string(index=False))
    print(f"\nSaved output to: {output_tsv}")


if __name__ == "__main__":
    main()