from argparse import ArgumentParser
import logging
import os

from astropy.io import fits
import numpy as np
from scipy import ndimage
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
from skimage.registration import phase_cross_correlation
import tqdm

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument(
    "-N", default=400, type=int, help="Number of frames to coadd before viewing"
)
parser.add_argument("-r", "--rate", type=int, default=2000, help="RTC operational rate in Hz")
parser.add_argument("-m", "--mask", action="store_true", help="Apply DM mask to output")
parser.add_argument("-o", "--output", default="coadded_and_subtracted_cube.fits", help="name of output file")
parser.add_argument("-f", "--filter",
action="store_true", help="manually filter 11 LWE modes")

# constants
PLATESCALE = 8.2 / 45 # m/px
OVERLAP_RATIO = 0.05
rootdir = lambda *tokens: os.path.join(os.path.dirname(__file__), *tokens)
configdir = lambda *tokens: os.path.join("/", "home", "scexao", "AOloop", "AOloop0", *tokens)


def velocity_lims(rate, N):
    dt = N / rate
    vmin = 0.5 * PLATESCALE / dt
    vmax = np.minimum(8.2 / dt, 50 * (1 - OVERLAP_RATIO) * PLATESCALE / dt)
    return vmin, vmax

def get_mask():
    dmmask = fits.getdata(rootdir("data", "dmmask_2021-03-19_01:33:34.fits"))
    dmslaved = fits.getdata(rootdir("data", "dmslaved_2021-03-19_01:33:34.fits"))
    return dmmask.astype(bool) ^ dmslaved.astype(bool)

def filter_LWE(cube):
    lwe_path = os.path.join(configdir("mkmodestmp"), "fmodes1_05.fits")
    lwe_frames = fits.getdata(lwe_path).reshape(-1, 50**2).T
    flat_full_frames = cube.reshape(-1, 50**2).T
    lwe_coeffs, _, _, _ = np.linalg.lstsq(lwe_frames, flat_full_frames, rcond=None)
    lwe_recon = np.reshape(lwe_coeffs.T @ lwe_frames.T, cube.shape)
    return cube - lwe_recon

if __name__ == "__main__":
    args = parser.parse_args()
    N = args.N

    vmin, vmax = velocity_lims(args.rate, N)
    logger.info(f"Pre-processing {args.file}")
    logger.info(f"Framerate: {args.rate} Hz\tWindow length: {args.N}\tPlatescale: {PLATESCALE:.2f} m/px")
    logger.info(f"Minimum detectable wind speed: {vmin:.2f} m/s")
    logger.info(f"Maximum detectable wind speed: {vmax:.2f} m/s")

    # open file
    cube = fits.getdata(args.file)

    if args.filter:
        cube = filter_LWE(cube)

    # remove static structure
    resid_cube = cube - np.median(cube, axis=0)

    # coadd cube
    coadded_cube = np.array([
        resid_cube[i:i+N].sum(axis=0) for i in range(0, cube.shape[0], N)
    ])
    # coadded_cube = resid_cube[::N]

    # X = coadded_cube.reshape(coadded_cube.shape[0], -1)
    # U, S, Vt = np.linalg.svd(X)
    # Y = np.linalg.multi_dot([U[:, :5], np.diag(S[:5]), Vt[:5, :]])
    # Y = Y.reshape(coadded_cube.shape)
    # Y = ndimage.gaussian_filter(coadded_cube, (1, 5, 5))
    # Y = np.zeros_like(coadded_cube)
    # gauss_highpass = coadded_cube - lowpass
    # Y = np.median(coadded_cube, axis=0)

    coadded_cube #-= Y

    logger.debug(f"size after coadding: {coadded_cube.shape}")

    mask = get_mask()

    correllelagrams = []
    shifts = []
    for i in tqdm.trange(coadded_cube.shape[0] - 1):
        previous, current = coadded_cube[i], coadded_cube[i + 1]
        cross_corr = cross_correlate_masked(current, previous, mask, mask, axes=(0, 1), overlap_ratio=OVERLAP_RATIO)
        correllelagrams.append(cross_corr)

        maxima = np.stack(np.nonzero(cross_corr == cross_corr.max()), axis=1)
        center = np.mean(maxima, axis=0)
        shift = center - np.array(previous.shape) + 1

        dy, dx = -shift
        dr = np.sqrt(dx**2 + dy**2)
        dt = np.arctan2(dy, dx)
        logger.debug(f"dx: {dx:.01f} px\tdy: {dy:.01f} px\tdr: {dr:.01f} px")
        speedscale = PLATESCALE * args.rate / args.N
        logger.debug(f"{dx * speedscale:.01f} m/s\t{dy * speedscale:.01f} m/s\t{dr * speedscale:.01f} m/s")

    correllelagrams = np.real(np.array(correllelagrams))


    fits.writeto("correllelagrams.fits", correllelagrams, overwrite=True)

    if args.mask:
        output_cube = coadded_cube * mask
    else:
        output_cube = coadded_cube

    fits.writeto(args.output, output_cube, overwrite=True)
    logger.info(f"file saved to {args.output}")
