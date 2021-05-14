from argparse import ArgumentParser
import logging
import os
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
import tqdm

plt.style.use("ggplot")
plt.rcParams.update({"lines.markersize": 3, "scatter.marker": "."})

# set up logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# set up arg parser
parser = ArgumentParser("calculate translations between wavefront sensor frames")
parser.add_argument(
    "path",
    type=str,
    help="path to wavefront sensor FITS cube. If loading data, this should be directory containing the calculated offsets (the output directory from previous runs).",
)
parser.add_argument(
    "-N",
    default=400,
    help="Length of window",
    type=int
)
parser.add_argument(
    "-d",
    "--directory",
    help="directory to save outputs, will default to the parent directory of the input file",
)
parser.add_argument(
    "-l",
    "--load",
    help="load offsets from the given path instead of calculating them",
    action="store_true",
)
parser.add_argument(
    "-r",
    "--rate",
    help="rate of acquisition (Hz)",
    default=1000,
    type=int
)
parser.add_argument("-f", "--filter",
action="store_true", help="manually filter first 7 Zernike modes")
parser.add_argument("-s", "--smooth",
action="store_true", help="manually smooth with Gaussian filter")

# constants
PLATESCALE = 8.2 / 45 # m/px
OVERLAP_RATIO = 0.3
rootdir = lambda *tokens: os.path.join(os.path.dirname(__file__), *tokens)
configdir = lambda *tokens: os.path.join("/", "home", "scexao", "AOloop", "AOloop0", *tokens)

# functions
def calculate_offsets(previous, current, mask, dt):

    cross_corr = cross_correlate_masked(current, previous, mask, mask, axes=(0, 1), overlap_ratio=OVERLAP_RATIO)

    maxima = np.stack(np.nonzero(cross_corr == cross_corr.max()), axis=1)
    center = np.mean(maxima, axis=0)
    shift = center - np.array(previous.shape) + 1

    dy, dx = shift * PLATESCALE / dt
    dr = np.hypot(dx, dy)
    theta = np.rad2deg(np.arctan2(dy, dx))
    if theta < 0:
        theta += 360

    # shift_yx = phase_cross_correlation(
    #     current,
    #     previous,
    #     reference_mask=mask,
    #     return_error=False,
    #     overlap_ratio=OVERLAP_RATIO
    # )
    # dy, dx = shift_yx * PLATESCALE / dt
    # dr = np.hypot(dy, dx)
    # theta = np.rad2deg(np.arctan2(dy, dx))
    # if theta < 0:
    #     theta += 360

    return np.real(cross_corr), dx, dy, dr, theta

def cube_offsets(cube, mask, rate, N):
    offsets = np.empty((cube.shape[0] - N, 4))
    corr_shape = cube.shape[1] * 2 - 1
    corrs = np.empty((cube.shape[0] - N, corr_shape, corr_shape))
    vmin, vmax = velocity_lims(rate, N)
    logger.info(f"minimum detectable velocity: {vmin:.02f} m/s")
    logger.info(f"maximum detectable velocity: {vmax:.02f} m/s")
    dt = N / rate
    for i in tqdm.trange(offsets.shape[0], desc="calculating offsets"):
        previous = cube[i] 
        current = cube[i + N]
        out = calculate_offsets(previous, current, mask, dt)
        corrs[i] = out[0]
        offsets[i] = out[1:]

    return corrs, offsets

def velocity_lims(rate, N):
    dt = N / rate
    vmin = 0.5 * PLATESCALE / dt
    vmax = np.minimum(8.2 / dt, 50 * (1 - OVERLAP_RATIO) * PLATESCALE / dt)
    return vmin, vmax

def plot_offsets(dir, offsets, dt):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
    ts = range(len(offsets))
    axes[0].scatter(ts, offsets[:, 0], c="C0")
    axes[1].scatter(ts, offsets[:, 1], c="C1")
    axes[2].scatter(ts, offsets[:, 2], c="C2")
    axes[3].scatter(ts, offsets[:, 3], c="C3")

    axes[0].set_ylabel("$\Delta x$ [m/s]")
    axes[1].set_ylabel("$\Delta y$ [m/s]")
    axes[2].set_ylabel("$\Delta r$ [m/s]")
    axes[3].set_ylabel(r"$\theta$ [deg]")
    axes[3].set_xlabel("iteration")

    xy_err = 0.5 * PLATESCALE / dt
    r_err = np.sqrt(2) * xy_err
    theta_err = np.median(np.rad2deg(xy_err / offsets[:, 2]))
    axes[0].set_title(r"{:.02f} $\pm$ {:.02f} m/s".format(np.median(offsets[:, 0]), xy_err))
    axes[1].set_title(r"{:.02f} $\pm$ {:.02f} m/s".format(np.median(offsets[:, 1]), xy_err))
    axes[2].set_title(r"{:.02f} $\pm$ {:.02f} m/s".format(np.median(offsets[:, 2]), r_err))
    axes[3].set_title(r"{:.02f} $\pm$ {:.02f} $\degree$".format(np.median(offsets[:, 3]), theta_err))
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "chains.png"))

    # # marginal 2d hist
    # g = sns.JointGrid(x=offsets[:, 0], y=offsets[:, 1])
    # hist = sns.jointplot(
    #         x=offsets[:, 0],
    #         y=offsets[:, 1],
    #     kind="kde",
    #     fill=0.2,
    #     bw_method=PLATESCALE / dt,
    #     marginal_kws=dict(fill=0.2),
    # )
    # hist.ax_joint.set_xlabel("Δx [m/s]")
    # hist.ax_joint.set_ylabel("Δy [m/s]")
    # plt.tight_layout()
    # plt.savefig(os.path.join(dir, "2d_hist.png"))

    # # kde plots
    # bw = 2 * xy_err
    # theta_bw = 2 * theta_err if np.isfinite(theta_err) else "scott"
    # fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    # sns.kdeplot(x=offsets[:, 0], fill=0.2, color="C0", ax=axes[0, 0], bw_method=bw)
    # sns.kdeplot(x=offsets[:, 1], fill=0.2, color="C1", ax=axes[0, 1], bw_method=bw)
    # sns.kdeplot(x=offsets[:, 2], fill=0.2, color="C2", ax=axes[1, 0], bw_method=bw * np.sqrt(2))
    # sns.kdeplot(x=offsets[:, 3], fill=0.2, color="C3", ax=axes[1, 1], bw_method=theta_bw)

    # axes[0, 0].set_xlabel("$\Delta x$ [m/s]")
    # axes[0, 1].set_xlabel("$\Delta y$ [m/s]")
    # axes[1, 0].set_xlabel("$\Delta r$ [m/s]")
    # axes[1, 1].set_xlabel(r"$\theta$ [deg]")
    # plt.tight_layout()

    # plt.savefig(os.path.join(dir, "marg_hists.png"))

    logger.info(f"saved plots to '{dir}'")

def save_offsets(dir, df):
    df.to_csv(os.path.join(dir, "offsets.csv"))
    logger.info(f"saved offsets to '{dir}'")


def load_offsets(dir):
    df = pd.from_csv(os.path.join(dir, "offsets.csv"))
    logger.info(f"loaded offsets from '{dir}'")
    return df[["dx", "dy", "dr", "dtheta"]]

def get_mask():
    logger.info("loading DM mask from `aol0_dmmask` and `aol0_dmslaved`")
    m1 = fits.getdata("data/dmmask_2021-03-19_01:33:34.fits")
    m2 = fits.getdata("data/dmslaved_2021-03-19_01:33:34.fits")
    return m1.astype(bool) ^ m2.astype(bool) # xor


def filter_modes(cube):
    # lwe_path = os.path.join(configdir("mkmodestmp"), "fmodes1_05.fits")
    lwe_path = os.path.join(configdir("mkmodestmp"), "fmodes0all.fits")
    lwe_indices = range(7)
    logger.info(f"removing {len(lwe_indices)} modes from {lwe_path}")
    lwe_fullframes = fits.getdata(lwe_path)[lwe_indices]
    lwe_frames = lwe_fullframes.reshape(-1, 50**2).T
    flat_full_frames = cube.reshape(-1, 50**2).T
    lwe_coeffs, _, _, _ = np.linalg.lstsq(lwe_frames, flat_full_frames, rcond=None)
    lwe_recon = np.reshape(lwe_coeffs.T @ lwe_frames.T, cube.shape)
    return cube - lwe_recon

def gaussian_filter(cube, N):
    kernel_size = (1, 3, 3)
    logger.info(f"smoothing cube with Gaussian filter with size {kernel_size}")
    return cube - ndimage.gaussian_filter(cube, kernel_size)

if __name__ == "__main__":
    # parse args
    args = parser.parse_args()
    path = args.path
    if args.directory is None:
        dir = os.path.dirname(os.path.abspath(path))
    else:
        dir = os.path.abspath(args.directory)

    os.makedirs(dir, exist_ok=True)
    logger.debug(f"path: {path}")
    logger.debug(f"output dir: {dir}")

    if args.load:
        offsets = load_offsets(path)
        plot_offsets(path, offsets, args.N / args.rate)
    else:
        # load data and process
        cube = fits.getdata(path)
        logger.info(f"loaded FITS cube of size {cube.shape} from '{path}'")
        if args.filter:
            cube = filter_modes(cube)
        if args.smooth:
            cube = gaussian_filter(cube, args.N)
        logger.info("subtracting median from cube")
        cube -= np.median(cube, axis=0)
        # cube = np.array([cube[i:i + args.N].sum(0) for i in range(cube.shape[0] - args.N)])
        mask = get_mask()
        fits.writeto(os.path.join(dir, "cube.fits"), cube * mask, overwrite=True)
        corrs, offsets = cube_offsets(cube, mask, args.rate, args.N)
        fits.writeto(os.path.join(dir, "corrs.fits"), corrs, overwrite=True)
        save_offsets(dir, pd.DataFrame(offsets, columns=["dx", "dy", "dr", "dtheta"]))
        plot_offsets(dir, offsets, args.N / args.rate)
