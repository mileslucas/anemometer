
import os
from glob import glob
import logging

from astropy.io import fits
import numpy as np
from scipy import ndimage
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
import tqdm

## set up logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


## Constants
PLATESCALE = 8.2 / 45 # m/px
OVERLAP_RATIO = 0.3

## Directories
def rootdir(*tokens):
    """Root directory of the repository"""
    os.path.join(os.path.dirname(__file__), "..", *tokens)

def datadir(*tokens):
    rootdir("data", *tokens)

# make sure local directories exist
[os.makedirs(p, exist_ok=True) for p in [datadir(), plotsdir()]]

def configdir(*tokens):
    """Directory with AO loop configuration files"""
    os.path.join("/", "home", "scexao", "AOloop", "AOloop0", *tokens)

## Functions
def get_DM_mask():
    path_mask = max(glob(configdir("dmmask/dmmask*.fits")), key=os.path.getctime)
    path_slaved = max(glob(configdir("dmmask/dmslaved*.fits")), key=os.path.getctime)
    dmmask = fits.getdata(path_mask)
    dmslaved = fits.getdata(path_slaved)
    logger.info(f"loaded DM mask from {path_mask} and {path_slaved}")
    return dmmask.astype(bool) ^ dmslaved.astype(bool) # xor

def velocity_lims(rate, N):
    """Returns the (vmin, vmax) detectable velocities for the given loop rate and window length"""
    dt = N / rate
    vmin = 0.5 * PLATESCALE / dt
    vmax = np.minimum(8.2 / dt, 50 * (1 - OVERLAP_RATIO) * PLATESCALE / dt)
    return vmin, vmax


def calculate_offsets(previous, current, mask, dt):
    """Calculate offsets between two frames"""
    cross_corr = cross_correlate_masked(current, previous, mask, mask, axes=(0, 1), overlap_ratio=OVERLAP_RATIO)

    maxima = np.stack(np.nonzero(cross_corr == cross_corr.max()), axis=1)
    center = np.mean(maxima, axis=0)
    shift = center - np.array(previous.shape) + 1

    dy, dx = shift * PLATESCALE / dt
    dr = np.hypot(dx, dy)
    theta = np.mod(np.rad2deg(np.arctan2(dy, dx)), 360)
    offsets = np.array([dx, dy, dr, theta])
    xy_err = 0.5 * PLATESCALE / dt
    r_err = np.sqrt(2) * xy_err
    theta_err = np.rad2deg(xy_err / r_err)
    errors = np.array([xy_err, xy_err, r_err, theta_err])
    return np.real(cross_corr), offsets, errors


def cube_offsets(cube, mask, rate, N):
    """Calculate offsets between frames for an entire cube"""
    offsets = np.empty((cube.shape[0] - N, 4))
    errors = np.copy(offsets)
    corr_shape = cube.shape[1] * 2 - 1
    corrs = np.empty((cube.shape[0] - N, corr_shape, corr_shape))
    vmin, vmax = velocity_lims(rate, N)
    logger.info(f"minimum detectable velocity: {vmin:.02f} m/s")
    logger.info(f"maximum detectable velocity: {vmax:.02f} m/s")
    dt = N / rate
    for i in tqdm.trange(offsets.shape[0], desc="calculating offsets"):
        previous = cube[i] 
        current = cube[i + N]
        corr, off, err = calculate_offsets(previous, current, mask, dt)
        corrs[i] = corr
        offsets[i] = off
        errors[i] = err

    return corrs, offsets, errors


def filter_zernike(cube, nz=7):
    """Filter out Zernike modes"""
    basis_path = configdir("mkmodestmp", "fmodes0all.fits")
    indices = range(nz)
    basis = fits.getdata(basis_path)[indices]
    basis_flat = basis.reshape(-1, 50**2).T
    cube_flat = cube.reshape(-1, 50**2).T
    coeffs, _, _, _ = np.linalg.lstsq(basis_flat, cube_flat, rcond=None)
    recon = np.reshape(coeffs.T @ basis_flat.T, cube.shape)
    logger.info(f"removed {nz} modes from {basis_path}")
    return cube - recon

def filter_gaussian(cube, size=(1, 3, 3)):
    """High-pass Gaussian filter"""
    loworder = ndimage.gaussian_filter(cube, size)
    logger.info(f"removed Gaussian filter with size {size}")
    return cube - loworder


