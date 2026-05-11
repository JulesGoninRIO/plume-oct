import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy import stats
import numpy as np
import json
from argparse import ArgumentParser

def density_plot(x, y, output_dir, cmap='mako', bins=100, figsize=(5,4), ylab="SNR [db]", xlab="Total Displacement [mm]"):
    """
    Plot a density plot of the two variables SNR and total displacement.

    Parameters:
    -----------
    x: pd.Series
        The diplacement score
    y: pd.Series
        The SNR score
    output_dir: str
        The output directory for the plot
    cmap: str
        The color map to use
    bins: int
        The number of bins to use
    figsize: tuple
        The size of the figure
    ylab: str
        The label for the y-axis
    xlab: str
        The label for the x-axis
    """
    g = sns.JointGrid(x=x, y=y)

    #center histogram
    sns.histplot(
        x=x, y=y,
        bins=bins, log_scale=(True, False),
        cbar=True, ax=g.ax_joint,norm=clr.LogNorm(),vmin=None, vmax=None, cmap = cmap
    ).set(xlabel=xlab,ylabel=ylab)

    #top and right histograms
    sns.histplot(x=x,ax=g.ax_marg_x,element='step')
    sns.histplot(y=y,ax=g.ax_marg_y,element='step')

    # add correlation coefficient
    pearsonr, p = stats.pearsonr(df["tot_displacement"], df["SNR"])
    g.ax_joint.annotate(f'Pearson r = {pearsonr:.2f}', xy=(0.2, 1))
    plt.tight_layout()
    plt.savefig(output_dir + "/SNR_density_plot.png")

if __name__ == "__main__":
    parser =  ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/config_SNR.json")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config =json.load(config_file)
        header_csv = config["header_csv"]
        output_dir = config["output_dir"]
    df = pd.read_csv(header_csv)
    density_plot(x=df["tot_displacement"],y=df["SNR"],output_dir=output_dir)