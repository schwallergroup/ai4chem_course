"""
A script which, for each job, plots the following:
1. Training and validation loss
2. Fraction valid
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# set Seaborn themes and color palettes
sns.set_theme("talk", "white")
paired_palette = sns.color_palette("Paired")  # for visualizing data that comes in pairs

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define parameters for the model
parser.add_argument("--jobdir",
                    type=str,
                    default="./output/run_local/",
                    help="Specifies the job directory for the run to visualize.")
args = parser.parse_args()

if __name__ == '__main__':
    # load the results from the training job into a pandas DataFrame
    jobdir = Path(args.jobdir)
    data   = pd.read_csv(jobdir.joinpath("SmilesTrainer_training.csv"))

    # get each of the values to plot
    epochs          = data["epoch"]
    training_loss   = data["training loss"]
    validation_loss = data["validation loss"]
    fraction_valid  = data["fraction valid"]

    # 01 - plot the training and validation losses
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(epochs, training_loss, label="Training loss", color=paired_palette[0] )
    ax.plot(epochs, validation_loss, label="Validation loss", color=paired_palette[1])
    ax.legend()
    ax.set(xlabel="Epoch", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(f"./analysis/{jobdir.name}_loss.png")

    # 02 - plot the fraction valid
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(epochs, fraction_valid, color=paired_palette[2] )
    ax.set(xlabel="Epoch", ylabel="Fraction valid")
    fig.tight_layout()
    fig.savefig(f"./analysis/{jobdir.name}_fraction_valid.png")
