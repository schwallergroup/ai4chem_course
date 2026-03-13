""" collect_hidden_sweep. """
import yaml
import pandas as pd
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/2022_08_07_hidden_size_sweep_gnn")
    args = parser.parse_args()
    result_dir = Path(args.dir)

    all_outs = []
    all_files = result_dir.rglob("test_results.yaml")
    for res_file in all_files:
        res = yaml.safe_load(open(res_file, "r"))
        hidden = res['args']['hidden_size']
        test_loss = res['test_metrics']['test_loss']
        out = {"hidden": hidden,
               "test_loss": test_loss}
        all_outs.append(out)
    df = pd.DataFrame(all_outs)
    print(df)
