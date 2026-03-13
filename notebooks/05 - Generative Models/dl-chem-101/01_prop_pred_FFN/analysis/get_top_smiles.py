import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", default=10, type=int,)
    parser.add_argument("--input-file")
    args = parser.parse_args()
    df = pd.read_csv(args.input_file, sep="\t")
    out_df = df.sort_values(by="preds", axis=0, ascending=False)[:args.top_k]
    for k, v in out_df[["smiles", "preds"]].values:
        print(f"Smiles: {k}\nPred: {v}")
