import pandas as pd

def mb_to_bp(value):
    """
    Convert megabases to base pairs.
    Rounds to nearest integer in case values are decimals.
    """
    return round(float(value) * 1000000)

def main():
    input_csv = "/workdir/irk9/data/maps/maize/12915_2015_187_MOESM2_ESM.csv"
    output_csv = "/workdir/irk9/data/maps/maize/B73_v3.bed"

    df = pd.read_csv(input_csv)

    # Keep and rename columns
    out = pd.DataFrame()
    out["Chr"] = df["Chr"]
    out["start"] = df["Start"].apply(mb_to_bp).astype(int).replace(",", "")
    out["stop"] = df["Stop"].apply(mb_to_bp).astype(int)

    out.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()