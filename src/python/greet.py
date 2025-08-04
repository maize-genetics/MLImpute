import argparse

def main():
    parser = argparse.ArgumentParser(description="Print a greeting with a user-provided word.")
    parser.add_argument("word", help="The word to include in the greeting")
    args = parser.parse_args()

    print(f"Hello, {args.word}! You've been greeted from Python (called from the Rust CMD API)!")

if __name__ == "__main__":
    main()
