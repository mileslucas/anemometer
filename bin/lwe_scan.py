import argparse
import os

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

plt.style.use("ggplot")

parser = argparse.ArgumentParser()
parser.add_argument("filename", nargs="*")
parser.add_argument("-o", "--outdir", default="output", help="output directory")
parser.add_argument("-p", "--plot", action="store_true", help="plot metric")

rootdir = lambda *tokens: os.path.join(os.path.dirname(__file__), *tokens)

def get_masks():
    arrs = np.load(rootdir("data", "mask.npz"))
    return arrs.values()

def compute_metric(filename, masks):
    cube = fits.getdata(filename)
    cube -= np.median(cube, axis=0)
    quads = np.array([np.sum(cube * mask, axis=(1, 2)) for mask in masks])
    mu = np.mean(quads, axis=0)
    sigma = np.std(quads, axis=0)
    deviance = np.sum(quads - mu, axis=0)

    metrics = pd.DataFrame({
        "Q1": quads[0],
        "Q2": quads[1],
        "Q3": quads[2],
        "Q4": quads[3], 
        "mu": mu,
        "sigma": sigma,
        "deviance": deviance
    })
    return metrics

def plot_metrics(metrics, output, filename, show=False):
    
    fig, axes = plt.subplots(4, 4, figsize=(7, 9), sharex=True)
    axes[0].plot(metrics[["Q1", "Q2", "Q3", "Q4"]])
    axes[0].set_ylabel("sums")

    axes[1].plot(metrics["mu"], c="C4")
    axes[1].set_ylabel(r"$\mu$")

    axes[2].plot(metrics["sigma"], c="C5")
    axes[2].set_ylabel(r"$\sigma$")

    axes[3].plot(metrics["deviance"], c="C6")
    axes[3].set_ylabel("deviance")

    axes[3].set_xlabel("frame")

    plt.suptitle(filename)
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output, f"{filename}.png"))
    plt.close(fig)
    

if __name__ == "__main__":
    args = parser.parse_args()
    files = args.filename

    os.makedirs(args.outdir, exist_ok=True)

    masks = get_masks()

    for filename in tqdm.tqdm(files):
        metrics = compute_metric(filename, masks)
        fname = os.path.split(filename)[-1]
        plot_metrics(metrics, args.outdir, fname, args.plot)
        metrics.to_csv(os.path.join(args.outdir, f"{fname}.csv"))
        
