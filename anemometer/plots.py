import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .core import logger, 

# global settings
plt.style.use("ggplot")
plt.rcParams.update({"lines.markersize": 3, "scatter.marker": "."})

def chainplot(offsets, savedir=None, **plot_kwargs):
    """
    Plot offsets as a chain

    offsets : dataframe of offsets
    """
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8), **plot_kwargs)
    ts = range(len(offsets))

    axes[0].scatter(ts, offsets["dx"], c="C0")
    axes[1].scatter(ts, offsets["dy"], c="C1")
    axes[2].scatter(ts, offsets["dr"], c="C2")
    axes[3].scatter(ts, offsets["theta"], c="C3")

    axes[0].set_ylabel("$\Delta x$ [m/s]")
    axes[1].set_ylabel("$\Delta y$ [m/s]")
    axes[2].set_ylabel("$\Delta r$ [m/s]")
    axes[3].set_ylabel(r"$\theta$ [deg]")
    axes[3].set_xlabel("frame")

    quants = offsets[["dx", "dy", "dr", "theta"]].quantile([0.25, 0.5, 0.75], axis=0)
    eps = offsets[["dx_err", "dy_err", "dr_err", "theta_err"]].median(axis=0)

    axes[0].set_title(r"${1:.02f}_{{-{0:.02f}}}^{{+{2:.02f}}}$ ($\epsilon$={3:.02f}) m/s".format(*quants["dx"], eps["dx_err"]))
    axes[1].set_title(r"${1:.02f}_{{-{0:.02f}}}^{{+{2:.02f}}}$ ($\epsilon$={3:.02f}) m/s".format(*quants["dy"], eps["dy_err"]))
    axes[2].set_title(r"${1:.02f}_{{-{0:.02f}}}^{{+{2:.02f}}}$ ($\epsilon$={3:.02f}) m/s".format(*quants["dr"], eps["dr_err"]))
    axes[3].set_title(r"${1:.02f}_{{-{0:.02f}}}^{{+{2:.02f}}}$ ($\epsilon$={3:.02f}) m/s".format(*quants["theta"], eps["theta_err"]))
    plt.tight_layout()

    if savedir is None:
        plt.show(block=True)
    else:
        filename = os.path.join(savedir, "chains.png")
        plt.savefig(filename)
        logger.info(f"saved plot to {filename}")

def marginalplot(offsets, savedir=None, **plot_kwargs):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), **plot_kwargs)

    eps = 2 * offsets[["dx_err", "dy_err", "dr_err", "theta_err"]].median(axis=0)
    theta_bw = eps["theta_err"] if np.isfinite(eps["theta_err"]) else "scott"

    sns.kdeplot(x=offsets["dx"], fill=0.2, color="C0", ax=axes[0, 0], bw_method=eps["dx_err"])
    sns.kdeplot(x=offsets["dy"], fill=0.2, color="C1", ax=axes[0, 1], bw_method=eps["dy_err"])
    sns.kdeplot(x=offsets["dr"], fill=0.2, color="C2", ax=axes[1, 0], bw_method=eps["dr_err"])
    sns.kdeplot(x=offsets["theta"], fill=0.2, color="C3", ax=axes[1, 1], bw_method=theta_bw)

    axes[0, 0].set_xlabel("$\Delta x$ [m/s]")
    axes[0, 1].set_xlabel("$\Delta y$ [m/s]")
    axes[1, 0].set_xlabel("$\Delta r$ [m/s]")
    axes[1, 1].set_xlabel(r"$\theta$ [deg]")

    # plot quantiles
    quants = offsets[["dx", "dy", "dr", "theta"]].quantile([0.25, 0.5, 0.75], axis=0)

    axes[0, 0].axvline(quants["dx"][0], color="C0", ls="--")
    axes[0, 0].axvline(quants["dx"][1], color="C0")
    axes[0, 0].axvline(quants["dx"][2], color="C0", ls="--")
    axes[0, 1].axvline(quants["dy"][0], color="C1", ls="--")
    axes[0, 1].axvline(quants["dy"][1], color="C1")
    axes[0, 1].axvline(quants["dy"][2], color="C1", ls="--")
    axes[1, 0].axvline(quants["dr"][0], color="C2", ls="--")
    axes[1, 0].axvline(quants["dr"][1], color="C2")
    axes[1, 0].axvline(quants["dr"][2], color="C2", ls="--")
    axes[1, 1].axvline(quants["theta"][0], color="C3", ls="--")
    axes[1, 1].axvline(quants["theta"][1], color="C3")
    axes[1, 1].axvline(quants["theta"][2], color="C3", ls="--")

    axes[0, 0].set_title(r"${1:.02f}_{{-{0:.02f}}}^{{+{2:.02f}}}$".format(*quants["dx"]))
    axes[0, 1].set_title(r"${1:.02f}_{{-{0:.02f}}}^{{+{2:.02f}}}$".format(*quants["dy"]))
    axes[1, 0].set_title(r"${1:.02f}_{{-{0:.02f}}}^{{+{2:.02f}}}$".format(*quants["dr"]))
    axes[1, 1].set_title(r"${1:.02f}_{{-{0:.02f}}}^{{+{2:.02f}}}$".format(*quants["theta"]))

    plt.tight_layout()

    if savedir is None:
        plt.show(block=True)
    else:
        filename = os.path.join(savedir, "marginal.png")
        plt.savefig(filename)
        logger.info(f"saved plot to {filename}")

def radialplot(offsets, savedir=None, **plot_kwargs):
    # fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6), **plot_kwargs)

    # g = sns.JointGrid(x=offsets["dx"], y=offsets["dy"])
    eps = offsets[["dx_err", "dy_err"]].max()
    hist = sns.jointplot(
        x=offsets["dx"],
        y=offsets["dy"],
        kind="kde",
        fill=0.1,
        bw_method=eps,
        marginal_kws=dict(fill=0.1),
    )
    hist.ax_joint.set_xlabel("$\Delta x$ [m/s]")
    hist.ax_joint.set_ylabel("$\Delta y$ [m/s]")
    plt.tight_layout()

    if savedir is None:
        plt.show(block=True)
    else:
        filename = os.path.join(savedir, "radial.png")
        plt.savefig(filename)
        logger.info(f"saved plot to {filename}")
